import os
import torch
import torch.nn as nn

from habitat_baselines.il.env_based.policy.onav_base import ObjectNavBase
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from torchvision.transforms.functional import to_pil_image
# hint_info_mem="""Here are some suggestions to help you make the best decision:
# (1) Fully utilize your current view and memory clips to make the long term decision.
# (2) Derive the room information from the views and reason about how likely the goal object can be found in this room, explore the room if you are confident, else you should better leave and search for another room.
# (3) Your memory should be of assistance for you to learn the environment's structure relations and make better choice for the navigation."""
MP3D_CATEGORIES = [
    'chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa', 'bed', 'chest_of_drawers','plant', 'sink', 'toilet', 'stool', 'towel', 'tv_monitor', 'shower', 'bathtub', 'counter', 'fireplace', 'gym_equipment', 'seating', 'clothes',
]

MP3D_CATEGORY_TO_ID = {cat: idx for idx, cat in enumerate(MP3D_CATEGORIES, start=0)}
ID_TO_MP3D_CATEGORY = {v: k for k, v in MP3D_CATEGORY_TO_ID.items()}

def generate_mem_prompt(objects_list):
    objects_str = ", ".join(objects_list)       
    return (
        f"Describe the environment in detail, focusing on navigable spaces, obstacles, "
        f"and key objects: {objects_str}. "
        "Pay attention to the spatial relationships and form a topdown map to benefit navigation."
    )

class ObjectNavHFVLM(ObjectNavBase):
    system_message="""You are an intelligent indoor navigation robot. Your task is finding a given goal object as soon as possible. You only know what you see in the current image, and you must reason about where to go next."""
    
    action_info="""Allowed actions (choose exactly one, output only the action name):
    'LOOK_DOWN', 'LOOK_UP', 'MOVE_FORWARD', 'STOP', 'TURN_LEFT', 'TURN_RIGHT'{previous_action_state}Do not STOP until you manage to find and around current goal: {goal_object}. """
    
    hint_info_no_mem="""Here are some suggestions to help you make the best decision:
    (1) You can not move without navigable areas, confirm whether there are visible floor area in the image, avoid getting close to obstacles. If there is no navigable areas in image, you should change your direction first.
    (2) If {goal_object} is within image, take actions to go near it. Otherwise, explore the environment logically to gain useful clues for {goal_object}, do not wander aimlessly.
    (3) Think before action, analyze the appeared room type in the image, and reason about whether {goal_object} is likely to occur in current room. If you are confident, explore within the room, otherwise leave current room and search for another room."""

    @property
    def output_size(self):
        return self.model_config.hidden_size
    
    def build_model(self, observation_space, model_config, num_actions):
        print(f"Using huggingface transformers {model_config.model_class} for navigation, loading...")
        
        self.hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_config.model_class, 
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_config.model_class,
            use_fast=True
        )
        if model_config.lora_checkpoint:
            print(f"Lora adapter found, loading {model_config.lora_checkpoint} to {model_config.model_class}...")
            self.hf_model.load_adapter(model_config.lora_checkpoint)
        
        print(f"Huggingface model loaded")

    def prepare_hf_messages(self, rgb_image, object_goal, former_action):
        previous_action_state = "\n"
        if len(former_action) and former_action[-1][1]:
            previous_action_state=f"\nYour former action {former_action[-1][0]} leads to collision.\n"
        nav_prompt = f"{self.action_info.format(goal_object=object_goal, previous_action_state=previous_action_state)}\n{self.hint_info_no_mem.format(goal_object=object_goal)}\nYour action choice: "
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": rgb_image,
                    },
                    {
                        "type": "text",
                        "text": nav_prompt,
                    },
                ],
            }
        ]
        return messages

    def forward(self, batch, history_embeds, prev_actions, step_id):
        # dict_keys(['rgb', 'semantic', 'depth', 'objectgoal', 'compass', 'gps'])
        device = batch['gps'].device
        
        object_goal = ID_TO_MP3D_CATEGORY[batch['objectgoal'].item()]    
        
        rgb_image = batch['rgb'].squeeze(0).cpu().permute(2,0,1)
        rgb_image = to_pil_image(rgb_image)
        rgb_image.save("/home/marmot/Boyang/MEM-Nav/tmp/latest_view.png")
        
        llm_messages = self.prepare_hf_messages(rgb_image, object_goal, prev_actions)
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            llm_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=rgb_image,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = self.hf_model.generate(
            **inputs, 
            max_new_tokens=128
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        actions = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return actions, history_embeds