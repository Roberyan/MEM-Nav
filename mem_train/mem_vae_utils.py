import random
from lavis.models import load_model_and_preprocess

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

def generate_mem_prompt(objects_list):
    objects_str = ", ".join(objects_list)       
    return (
        f"Describe the environment in detail, focusing on navigable spaces, obstacles, "
        f"and key objects: {objects_str}. "
        "Pay attention to the spatial relationships and form a topdown map to benefit navigation."
    )
