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
import torch
import wandb
from tqdm import tqdm
import argparse
import random

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

def build_hf_action_data(demo_dataset):
    data = []
    for demo_episode in tqdm(demo_dataset, desc="Demo episodes"):
        for i, act in enumerate(demo_episode["demonstration"]):
            target = "; ".join(f"{act[i]} {act[i+1]}" for i in range(0, len(act), 2))
            messages = prepare_hf_message(
                rgb_path=demo_episode["rgb_paths"][i],
                depth_path=demo_episode["depth_paths"][i],
                object_goal=demo_episode["object_category"],
                demo_action=target
            )
            data.append(messages)
    return data

def get_demo_dataset(
    data_root="data/datasets/objectnav/mp3d_70k_demos_for_vlm", 
    scene=None, 
    demo_merge=0.5, 
    episode_merge=0.8,
    p_merge_all=0.1
):
    return NavDemoDataset(
        root_dir=data_root, 
        scene_id=scene,
        demo_merge_ratio=demo_merge,
        episode_merge_ratio=episode_merge,
        p_merge_all_episode=p_merge_all
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

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Qwen2.5-VL model")
    
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct" )
    
    parser.add_argument('--demo_root_dir', type=str, default="data/datasets/objectnav/mp3d_70k_demos_for_vlm")
    
    parser.add_argument('--output_dir', type=str, default="experiment_results/qwen25vl_lora")
    parser.add_argument('--load_in_4bit', action='store_true', default=False)
    parser.add_argument('--load_in_8bit', action='store_true', default=False)
    parser.add_argument('--lora_target', type=str, default="all-linear")
    
    parser.add_argument('--report_to', type=str, default="wandb")
    
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2")
    
    
    parser.add_argument('--deepspeed_config', type=str, default="nav_vlm/zero2.json")
    # Add a --local_rank argument to accept distributed launcher arguments.
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.local_rank == 1:
        args.report_to = None
    
    SCENE = "17DRP5sb8fy"
    nav_demo_dataset = get_demo_dataset(
        args.demo_root_dir,
        # SCENE,
        demo_merge=0.8, # how much episode to merge
        episode_merge=0.7, # in each demo, how much action to merge
        p_merge_all=0.2 # merge all action in that demo
    )
    nav_demo_dataset.increase_stop_demos()
    nav_demo_dataset.merge_demos(allow_full_merge=True, remove_merged_step=False)
    all_demos = nav_demo_dataset.get_demos()
    random.seed(2025)
    random.shuffle(all_demos)
    
    num_eval = 223
    eval_demos = all_demos[:num_eval]
    train_demos = all_demos[num_eval:]
    
    eval_demos = NavDemoDataset.remove_long_demos(eval_demos, 300)
    train_demos = NavDemoDataset.remove_long_demos(train_demos, 500)
    
    print(f"Train data: {len(train_demos)}, Eval data: {len(eval_demos)}.")
    
    eval_dataset = build_hf_action_data(eval_demos)
    train_dataset =  build_hf_action_data(train_demos)
    
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        # device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        attn_implementation=args.attn_implementation,
        use_cache=False
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_id, use_fast=True)
    processor.tokenizer.padding_side = "right"
    
    # === LoRA Config ===
    lora_target = []
    # only add LoRA target modules in the language model
    for name, module in model.named_modules():
        if "model" not in name or "lm_head" in name:
            continue
        if args.lora_target == "all-linear":
            if isinstance(module, torch.nn.Linear):
                lora_target.append(name)
        else:
            if args.lora_target in name:
                lora_target.append(name)
    
    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        target_modules=lora_target,
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
        output_dir=args.output_dir,
        num_train_epochs=1,
        dataloader_num_workers=4,  # Must be > 0
        dataloader_prefetch_factor=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4, 
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        # Optimizer and scheduler settings
        optim="paged_adamw_32bit",
        learning_rate=1e-5,
        lr_scheduler_type="cosine",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=50,
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
        report_to=args.report_to,  # Enable wandb logging
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        remove_unused_columns=False,
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=512, # Maximum sequence length for input
        deepspeed=args.deepspeed_config
    )
    
    if args.report_to == "wandb":
        wandb.init(
            project="VLM-NAV",  # change this
            name="qwen2.5-3b-instruct-trl-sft-nav-all-demo-mix-add-stop-add",  # change this
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
    
    