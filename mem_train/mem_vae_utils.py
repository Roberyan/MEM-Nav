import torch
from lavis.models import load_model_and_preprocess
import torchvision.transforms as T
to_pil = T.ToPILImage()

# Define a custom warmup-decay scheduler.# Define a warmup-decay learning rate scheduler.
class WarmupDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, warmup_min_lr, warmup_max_lr, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_min_lr = warmup_min_lr
        self.warmup_max_lr = warmup_max_lr
        super(WarmupDecayLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            # Linear warmup: increase lr from warmup_min_lr to warmup_max_lr.
            lr = self.warmup_min_lr + (self.warmup_max_lr - self.warmup_min_lr) * (step / self.warmup_steps)
        else:
            # Linear decay from warmup_max_lr to 0 over the remaining steps.
            decay_steps = self.total_steps - self.warmup_steps
            current_decay = step - self.warmup_steps
            lr = self.warmup_max_lr * (1 - current_decay / decay_steps)
            lr = max(lr, 0)
        return [lr for _ in self.base_lrs]

def load_blip2_model_lavis(
    model_name="blip2_feature_extractor", 
    type="pretrain_vitL"
):
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=model_name, 
        model_type=type, 
        is_eval=True
    )

    return model, vis_processors, txt_processors

def load_instructblip_model_lavis(
    model_name="blip2_t5_instruct",
    type="flant5xl"
):
    assert model_name in ["blip2_t5_instruct"], f"{model_name} hasn't been modified in Larvis code, can not be used in this project"
    
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=model_name, 
        model_type=type, 
        is_eval=True
    )
    return model, vis_processors, txt_processors

def generate_mem_prompt(objects_list):
    objects_str = ", ".join(objects_list)       
    return PROMPT_MEM_GEN.format(objects_str=objects_str)

# return batch wise embeddings
def prepare_blip2_embeddings(model, vis_processor, txt_processor, rgb_views, prompt_input, device=None):
    # Determine device from model if not provided.
    if device is None:
        device = model.device
    
    rgb_views_batch = rgb_views.unsqueeze(0) if rgb_views.dim() == 4 else rgb_views # single sample, expand a batch dimension
    
    B, num_views, C, H, W = rgb_views_batch.size()
    
    # Flatten the batch and view dimensions: (B * 4, 3, H, W)
    rgb_views_flat = rgb_views_batch.reshape(B * num_views, C, H, W)
    
    rgb_inputs_batch = torch.stack([vis_processor["eval"](to_pil(rgb_views_flat[i])) for i in range(B*num_views)]).to(device)
    blip_embeds_flat = model.get_qformer_features({"image": rgb_inputs_batch, "prompt": prompt_input})
    blip_embeds_batch = blip_embeds_flat.reshape(B, num_views, blip_embeds_flat.size(1), blip_embeds_flat.size(2))
    return blip_embeds_batch


PROMPT_MEM_GEN = """From the first‑person indoor RGB view, describe the visible environment to support navigation. Analyze only what's directly visible and identify three categories: continuous navigable floor regions (free space), static obstacles (walls, furniture), and all visible objects from {objects_str} — for each object include its category, bounding box, approximate distance (m), and egocentric direction (left/center/right). Leverage any additional visible cues to infer room type and capture spatial relationships between elements (e.g., ‘chair is 1.2 m ahead, slightly right of table’)."""
