import os
import torch
import torch.nn as nn
import random
from habitat_baselines.il.env_based.policy.onav_base import ObjectNavBase
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from habitat_baselines.il.env_based.policy.llm_prompt import(
    ACTION_PROMPT,
    SYSTEM_LLM_SINGLE_VIEW,
    GLOBAL_HINT_SINGLE_VIEW,
    INSTANT_HINT_SINGLE_VIEW,
    SYSTEM_LLM_SURROUNDING_VIEWS,
    GLOBAL_HINT_SURROUNDING_VIEWS,
    INSTANT_HINT_SURROUNDING_VIEWS
)
from nav_vlm.extract_demo_data import create_rgb_panorama, create_depth_panorama
import numpy as np
from qwen_vl_utils import process_vision_info
from nav_vlm.vlm_prompts import (
    SYSTEM_PANORAMA_VIEWS,
    HINT_PANORAMA_VIEWS,
    ACTION_PANORAMA_VIEWS
)

MP3D_CATEGORIES = [
    'chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa', 'bed', 'chest_of_drawers','plant', 'sink', 'toilet', 'stool', 'towel', 'tv_monitor', 'shower', 'bathtub', 'counter', 'fireplace', 'gym_equipment', 'seating', 'clothes',
]

MP3D_CATEGORY_TO_ID = {cat: idx for idx, cat in enumerate(MP3D_CATEGORIES, start=0)}
ID_TO_MP3D_CATEGORY = {v: k for k, v in MP3D_CATEGORY_TO_ID.items()}

class ObjectNavHFVLM(ObjectNavBase):
    action_space = {
        'STOP': 0, 'MOVE_FORWARD': 1, 'TURN_LEFT': 2, 'TURN_RIGHT': 3, 'LOOK_UP': 4, 'LOOK_DOWN': 5
    }
    
    system_message="""You are an intelligent indoor navigation robot. Your task is finding a given goal object as soon as possible."""
    
    # 'STOP': 0, 'MOVE_FORWARD': 1, 'TURN_LEFT': 2, 'TURN_RIGHT': 3, 'LOOK_UP': 4, 'LOOK_DOWN': 5
    action_info="""
    Allowed actions (choose exactly one, output only the action name):
    'STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN'
    {previous_action_state}
    DO NOT STOP until you find current goal: {goal_object}.
    Your action choice:
    """
    
    global_hint_info="""
    Navigate to current goal: {goal_object}.
    think before you act"""

    instant_hint_info="""
    Hint:
        • NEVER choose STOP unless {goal_object} is visible AND within reach.
        • TURN_LEFT / TURN_RIGHT = rotate horizontally to reveal new directions and find alternative paths.
        • LOOK_UP/ LOOK_DOWN = tilt camera vertically to reveal upper or lower parts of current view.
        • If {goal_object} is visible:
            - Center it if needed.
            - Plan a clear path toward it (avoiding obstacles) and approach."""

    @property
    def output_size(self):
        return self.model_config.hidden_size
    
    def build_model(self, observation_space, model_config, num_actions):
        print(f"Using huggingface transformers {model_config.model_class} for navigation, loading...")
        self.nav_type = model_config.nav_type
        self.action_type = model_config.action_type
        self.init_prompt()
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
        self.hf_model.eval()
        print(f"Huggingface model loaded")

    def init_prompt(self):
        if self.nav_type == "single_view":
            self.action_info = ACTION_PROMPT
            self.system_message = SYSTEM_LLM_SINGLE_VIEW
            self.global_hint_info = GLOBAL_HINT_SINGLE_VIEW
            self.instant_hint_info = INSTANT_HINT_SINGLE_VIEW
        elif self.nav_type == "surrounding_views":
            self.action_info = ACTION_PROMPT
            self.system_message = SYSTEM_LLM_SURROUNDING_VIEWS
            self.global_hint_info = GLOBAL_HINT_SURROUNDING_VIEWS
            self.instant_hint_info = INSTANT_HINT_SURROUNDING_VIEWS
    
    def prepare_hf_messages(self, rgb_input, object_goal, former_action):
        previous_action_state = "\n"
        if len(former_action):
            previous_action_state=f"\nYour last step action: {former_action[-1][0]}"
            if former_action[-1][1]:
                previous_action_state += ", leads to collision!\n"
            else:
                previous_action_state+="\n"
        
        nav_prompt =  f"{self.global_hint_info.format(goal_object=object_goal)}\
                        {self.instant_hint_info.format(goal_object=object_goal)}\
                        {self.action_info.format(goal_object=object_goal, previous_action_state=previous_action_state)}"
        
        message_content = []
        if self.nav_type == "single_view":
            message_content.append({"type": "image", "image": rgb_input})
        elif self.nav_type == "surrounding_views":
            for img in rgb_input:
                message_content.append({"type": "image", "image": img})
        
        message_content.append({"type": "text", "text": nav_prompt})
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_message}],
            },
            {
                "role": "user",
                "content": message_content
            }
        ]
        return messages

    def forward(self, batch, history_embeds, prev_actions):
        # dict_keys(['rgb', 'semantic', 'depth', 'objectgoal', 'compass', 'gps'])
        device = batch['gps'].device
        
        object_goal = ID_TO_MP3D_CATEGORY[batch['objectgoal'].item()]    
        
        if self.nav_type == "single_view":
            image_inputs = batch['rgb'].squeeze(0).cpu().permute(2,0,1)
            image_inputs = to_pil_image(image_inputs)
            llm_messages = self.prepare_hf_messages(image_inputs, object_goal, prev_actions)
            # rgb_image.save("/home/marmot/Boyang/MEM-Nav/tmp/latest_view.png")
            # Preparation for inference
            text = self.processor.apply_chat_template(
                llm_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(llm_messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            # Inference: Generation of the output
            generated_ids = self.hf_model.generate(
                **inputs, 
                max_new_tokens=256
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            actions = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for i in range(len(actions)):
                if actions[i] not in self.action_space.keys():
                    print(f"Generation not following instructions, '{actions[i]}' is not a valid action, replace with random one")
                    actions[i] = random.choice(list(self.action_space.keys()))
            return actions, history_embeds
        elif self.nav_type == "surrounding_views":
            rgb_panorama = create_rgb_panorama(batch['rgb_right'].squeeze(0).cpu(), batch['rgb_front'].squeeze(0).cpu(), batch['rgb_left'].squeeze(0).cpu())
            rgb_panorama = Image.fromarray(rgb_panorama)
            # rgb_panorama.save("tmp/rgb_views.png")
            
            depth_panorama = create_depth_panorama(batch['depth_right'].squeeze(0).cpu(), batch['depth_front'].squeeze(0).cpu(), batch['depth_left'].squeeze(0).cpu())
            depth_panorama = Image.fromarray((depth_panorama*255).astype(np.uint8))
            # depth_panorama.save("tmp/depth_views.png")
            
            former_acts = ""
            warnings = ""
            
            if len(prev_actions):
                acts = []
                for i, (act, if_collision) in enumerate(prev_actions):
                    if if_collision and i < len(prev_actions)-1:
                        continue
                    acts.append(act)
                if len(acts) > 10:
                    acts = acts[-10:]
                former_acts += "\nLast 10 action sequence: " + "; ".join(acts)

                # Check for meaningless turning
                if len(acts) >= 4:
                    last4 = acts[-4:]
                    # Check if all 4 are turning actions
                    is_all_turning = all(a.startswith("TURN") for a in last4)
                    if is_all_turning:
                        net_rotation = sum(
                            int(a.split()[1]) if a.startswith("TURN_LEFT") else -int(a.split()[1])
                            for a in last4 if a.startswith("TURN")
                        )
                        if net_rotation == 0:
                            warnings += (
                                f"\nWARNING: recent turning sequence \"{'; '.join(last4)}\" results in no rotation. "
                                "Avoid repeating canceling turns!\n"
                            )
                
                # check for all turning even though seemingly meaningful
                if len(acts) >= 6 and all(a.startswith("TURN") for a in acts[-6:]): 
                    warnings +=  (
                        "\nWARNING: Your recent actions indicate repeated turning at the same location. "
                        "You should take actions to change your location.\n"
                    )
                
                # Collision
                if prev_actions[-1][1]:
                    last_action = prev_actions[-1][0]
                    warnings += (
                        f"\nWARNING: last action \"{last_action}\" leads to collision. "
                        "DO NOT choose this action! Try turning instead.\n"
                    )
            
            nav_prompt = (
                f"{HINT_PANORAMA_VIEWS.format(goal_object=object_goal)}\n"
                f"{former_acts}"
                f"{warnings}"
                f"{ACTION_PANORAMA_VIEWS.format(goal_object=object_goal)}"
            )
            
            llm_messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PANORAMA_VIEWS}]},
                {   
                    "role": "user", 
                    "content": [
                        {"type": "image", "image":rgb_panorama}, #  Image.open(rgb_path).convert("RGB")
                        {"type": "image", "image": depth_panorama}, # Image.open(depth_path).convert("L")
                        {"type": "text", "text": nav_prompt},
                    ],
                }
            ]
            
        
            # Preparation for inference
            text = self.processor.apply_chat_template(
                llm_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(llm_messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            # Inference: Generation of the output
            if len(warnings):
                generated_ids = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    top_k=10,
                    top_p=0.95,
                    temperature=0.7,
                    num_return_sequences=5
                )
                # Since num_return_sequences > 1, repeat input_ids accordingly
                input_ids_expanded = inputs.input_ids.repeat_interleave(5, dim=0)
                # Trim prompt from output
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids_expanded, generated_ids)
                ]
                actions = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                if "collision" in warnings:
                    no_forward_acts = [act for act in actions if "TURN" in act]
                    if len(no_forward_acts):
                        actions = random.sample(no_forward_acts, 1)
                    else:
                        random_turn=f"{random.choice(['TURN_LEFT', 'TURN_RIGHT'])} 1"
                        actions = random.sample(actions, 1)
                        actions.insert(0, random_turn)
                elif "turning" in warnings:
                    no_turn_acts = [act for act in actions if "TURN" not in act]
                    if len(no_turn_acts):
                        actions = random.sample(no_turn_acts, 1)
                    else:
                        actions = random.sample(actions, 1)
                        actions.append("MOVE_FORWARD 1")
                print(former_acts, warnings, actions, "\n")
            else:
                generated_ids = self.hf_model.generate(
                    **inputs, 
                    max_new_tokens=256
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                actions = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            
                print(former_acts, "\n", actions, "\n")
            
            return actions, history_embeds