import random
from lavis.models import load_model_and_preprocess

# shuffle views order while keeping the corresponding across [rgb_views, one-hot_views, embedding views]
def shuffle_views(view_info_groups):
    indices = list(range(len(view_info_groups[0])))
    random.shuffle(indices)

    for idx in range(len(view_info_groups)):
        view_info_groups[idx] = [view_info_groups[idx][i] for i in indices]

    return view_info_groups

def load_blip2_model_lavis(
    model_name="blip2_feature_extractor", 
    type="pretrain_vitL"
):
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=model_name, 
        model_type=type, 
        is_eval=True)

    return model, vis_processors, txt_processors