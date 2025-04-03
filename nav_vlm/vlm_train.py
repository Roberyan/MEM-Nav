from nav_vlm.vlm_prompts import (
    SYSTEM_PANORAMA_VIEWS,
    HINT_PANORAMA_VIEWS,
    ACTION_PANORAMA_VIEWS
)
from nav_vlm.dataset import NavDemoDataset
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from PIL import Image
import torch
import os
import json
from datasets import Dataset
import wandb

def prepare_hf_message(rgb_path, depth_path, object_goal, demo_action):
    nav_prompt =  f"{HINT_PANORAMA_VIEWS.format(goal_object=object_goal)}\
                    {ACTION_PANORAMA_VIEWS.format(goal_object=object_goal)}"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PANORAMA_VIEWS}]},
        {   
            "role": "user", 
            "content": [
                {"type": "image", "image":rgb_path}, #  Image.open(rgb_path).convert("RGB")
                {"type": "image", "image": depth_path}, # Image.open(depth_path).convert("L")
                {"type": "text", "text": nav_prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": demo_action}]},
    ]
    return messages

def build_train_data(demo_dataset):
    data = []
    for demo_episode in demo_dataset:
        episode_id = demo_episode["episode_id"]
        for i, act in enumerate(demo_episode["demonstration"]):
            action_name, repeat = act
            target = f"{action_name} {repeat}"
            messages = prepare_hf_message(
                rgb_path=demo_episode["rgb_paths"][i],
                depth_path=demo_episode["depth_paths"][i],
                object_goal=demo_episode["object_category"],
                demo_action=target
            )
            data.append(messages)
    return data

def get_demo_dataset(data_root, scene=None):
    return NavDemoDataset(
        root_dir=data_root, 
        scene_id=scene
    )

def find_assistant_content_sublist_indexes(l):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))
if __name__ == "__main__":
    ROOT_DIR = "data/datasets/objectnav/mp3d_70k_demos_for_vlm"
    SCENE = "17DRP5sb8fy"
    nav_demo_dataset = get_demo_dataset(ROOT_DIR, SCENE)
    nav_demo_dataset = build_train_data(nav_demo_dataset)
    eval_dataset = nav_demo_dataset[:1000]
    train_dataset = nav_demo_dataset[1000:]
    # hf_dataset = Dataset.from_list(train_dataset)
    # train_dataset = hf_dataset.select(list(range(1000, len(hf_dataset))))
    # eval_dataset = hf_dataset.select(list(range(0, 1000)))
    
    
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct" 
    # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config,
        use_cache=False
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(model_id, use_fast=True)
    processor.tokenizer.padding_side = "right"
    
    # === LoRA Config ===
    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # === Data Collator ===
    def collate_fn(examples):
        texts = [
            processor.apply_chat_template(example, tokenize=False) 
            for example in examples]
        
        image_inputs, video_inputs = process_vision_info(examples)
        
        batch = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        input_ids_lists = batch['input_ids'].tolist()
        labels_list = []
        for ids_list in input_ids_lists:
            label_ids = [-100] * len(ids_list) # -100 is the ignore index in loss function
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
            labels_list.append(label_ids)
        batch['labels'] = torch.tensor(labels_list, dtype=torch.int64)
        return batch

    # -------------------------------
    # Training Arguments and Trainer Setup
    # -------------------------------
    training_args = SFTConfig(
        output_dir="experiment_results/qwen25vl_lora",
        num_train_epochs=2,
        dataloader_num_workers=4,  # Must be > 0
        dataloader_prefetch_factor=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8, 
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        # Optimizer and scheduler settings
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=100,
        eval_steps=1000,
        save_steps=1000,
        eval_strategy="steps",
        save_strategy="steps",
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        report_to="wandb",  # Enable wandb logging
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        remove_unused_columns=False,
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=512, # Maximum sequence length for input
    )
    
    wandb.init(
        project="VLM-NAV",  # change this
        name="qwen2.5-3b-instruct-trl-sft-nav",  # change this
        config=training_args,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor.tokenizer,
    )

    # -------------------------------
    # Start Training
    # -------------------------------
    trainer.train()
    trainer.save_model()  # Saves LoRA adapter weights
    
    