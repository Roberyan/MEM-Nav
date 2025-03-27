import torch
import torch.nn as nn
import numpy as np
from lavis.models import load_model_and_preprocess
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from gym import spaces
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


PROMPT_MEM_GEN = """Describe the visible environment in details. Analyze only what's directly visible and identify three categories: continuous navigable floor regions (free space), static obstacles (walls, furniture), and all visible objects from {objects_str} — for each object include its category, bounding box, approximate distance (m), and egocentric direction (left/center/right)."""


class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=128,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint, map_location=torch.device('cpu'))

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                # Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations, no_fc_layer=False):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            obs_depth = observations["depth"]
            if len(obs_depth.size()) == 5:
                observations["depth"] = obs_depth.contiguous().view(
                    -1, obs_depth.size(2), obs_depth.size(3), obs_depth.size(4)
                )
            x = self.visual_encoder(observations)
            x = x.flatten(start_dim=1)

        x = x.detach()
        if no_fc_layer:
            return x

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)

class DepthEncoder(object):
    def __init__(self, 
                device, 
                observation_space=spaces.Dict({"depth": spaces.Box(low=0.0, high=1.0, shape=(224, 224, 1), dtype=np.float32)}), 
                ckpt_file="data/ddppo-models/gibson-2plus-resnet50.pth", 
                backbone="resnet50", 
                batch_size=64) -> None:
        depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=256, # useless
            checkpoint=ckpt_file,
            backbone=backbone,
            trainable=False,
        )
        depth_encoder.eval()
        self.model = depth_encoder.to(device)
        self.device = device
        self.batch_size = batch_size
    
    def extract_fts(self, depth_images):
        '''
        depth_images: Tensor (batch, height, width, 1)
        '''
        fts = []
        for i in range(0, len(depth_images), self.batch_size):
            ft = self.model.visual_encoder(
                {'depth': depth_images[i: i+self.batch_size]}
            ) # (batch, 128, 4, 4)
            fts.append(ft.flatten(start_dim=1).data.cpu().numpy())
        fts = np.concatenate(fts, 0)
        return fts
