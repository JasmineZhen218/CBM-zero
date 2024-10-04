import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_utils import get_data


def get_clip_text_features(device, clip_model_name, concept_bank):
    clip_model, clip_preprocess = clip.load(
            clip_model_name.replace("_", "/"), device=device
        )
    # encode text
    with torch.no_grad():
        text_features = clip_model.encode_text(clip.tokenize(concept_bank).to(device))
    return text_features
    
def get_clip_image_features(device, clip_model_name, data_name):
    # load clip
    clip_model, clip_preprocess = clip.load(
            clip_model_name.replace("_", "/"), device=device
        )
    # Load the dataset
    dataset_train = get_data(
            "{}_train".format(data_name),
            clip_preprocess,
        )
    dataset_val = get_data(
            "{}_val".format(data_name),
            clip_preprocess,
        )
    # encode image
    image_features_train = []
    print("Encoding training images with CLIP")
    with torch.no_grad():
        for x, _ in tqdm(DataLoader(dataset_train, batch_size=256, shuffle=False)):
            x = x.to(device)
            features = clip_model.encode_image(x)
            image_features_train.append(features.detach())
    image_features_train = torch.cat(image_features_train)
    image_features_val = []
    print("Encoding validation images with CLIP")
    with torch.no_grad():
        for x, _ in tqdm(DataLoader(dataset_val, batch_size=256, shuffle=False)):
            x = x.to(device)
            features = clip_model.encode_image(x)
            image_features_val.append(features.detach())
    image_features_val = torch.cat(image_features_val)
    return image_features_train, image_features_val


def calculate_clip_similarity(x, y):
    if x.dtype != torch.float32:
        x = x.float()
    if y.dtype != torch.float32:
        y = y.float()
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    return torch.matmul(x, y.T).float()
