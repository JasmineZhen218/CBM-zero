import os
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_utils import get_data, ActivationDataset
from clip_utils import get_clip_image_features
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d1, d2, K):
        super(MLP, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(d1, d2),
            nn.ReLU(),
        )
        self.output = nn.Linear(d2, K)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def load_clip_mlp_model(model_name):
    hidden_dim = int(model_name.split("-h")[1].split('_')[0])
    if model_name.split('_')[-1] == 'cifar10':
        K =10
    elif '_cifar100' in model_name:
        K = 100
    elif '_cub' in model_name:
        K = 200
    elif '_imagenet' in model_name:
        K = 1000
    elif '_mnist' in model_name:
        K = 10
    elif 'food101' in model_name:
        K = 101
    elif 'awa2' in model_name:
        K = 50
    else:
        raise ValueError("Dataset not supported")
    model = MLP(d1 = 768, d2=hidden_dim, K=K)
    return model


def load_clip_lp_model(model_name):
    if model_name.split('_')[-1] == 'cifar10':
        K =10
    elif '_cifar100' in model_name:
        K = 100
    elif '_cub' in model_name:
        K = 200
    elif '_imagenet' in model_name:
        K = 1000
    elif '_mnist' in model_name:
        K = 10
    elif 'food101' in model_name:
        K = 101
    elif 'awa2' in model_name:
        K = 50
    else:
        raise ValueError("Dataset not supported")
    model = nn.Linear(768, K)
    return model

def get_clip_mlp_fcn(black_box_model_name, device):
    assert black_box_model_name.startswith("clip_mlp_")
    bb_model = load_clip_mlp_model(black_box_model_name)
    if os.path.exists("saved_bb_models/{}.pt".format(black_box_model_name)):
        state_dict = torch.load(
                    "saved_bb_models/{}.pt".format(black_box_model_name),
                    map_location=device,
                    weights_only=True,
                )
        bb_model.load_state_dict(state_dict)
        bb_model.to(device)
        proj_activation2class = bb_model.output.weight
        proj_activation2class_bias = bb_model.output.bias
    else:
        raise ValueError("black-box model does not exist: {}".format(black_box_model_name))
    return proj_activation2class, proj_activation2class_bias


def get_clip_lp_fcn(black_box_model_name, device):
    assert black_box_model_name.startswith("clip_lp_")
    bb_model = load_clip_lp_model(black_box_model_name)
    if os.path.exists("saved_bb_models/{}.pt".format(black_box_model_name)):
        state_dict = torch.load(
                    "saved_bb_models/{}.pt".format(black_box_model_name),
                    map_location=device,
                    weights_only=True,
                )
        bb_model.load_state_dict(state_dict)
        bb_model.to(device)
        proj_activation2class = bb_model.weight
        proj_activation2class_bias = bb_model.bias
    else:
        raise ValueError("black-box model does not exist: {}".format(black_box_model_name))
    return proj_activation2class, proj_activation2class_bias


def get_clip_mlp_features(device, black_box_model_name, clip_image_embeddings_train_path, clip_image_embeddings_val_path):
    assert black_box_model_name.startswith("clip_mlp_")
    clip_model_name = black_box_model_name.split('-h')[0].split('clip_mlp_')[1]
    data_name = black_box_model_name.split('_')[-1]
    bb_model = load_clip_mlp_model(black_box_model_name)
    if os.path.exists("saved_bb_models/{}.pt".format(black_box_model_name)):
        state_dict = torch.load(
                    "saved_bb_models/{}.pt".format(black_box_model_name),
                    map_location=device,
                    weights_only=True,
                )
        bb_model.load_state_dict(state_dict)
        bb_model.to(device)
    else:
        raise ValueError("black-box model does not exist: {}".format(black_box_model_name))

    if os.path.exists(clip_image_embeddings_train_path) and os.path.exists(clip_image_embeddings_val_path):
        print("Load saved clip image embeddings from: {} and {}".format(clip_image_embeddings_train_path, clip_image_embeddings_val_path))
        clip_image_features_train = torch.load(clip_image_embeddings_train_path, weights_only=True, map_location=device)
        clip_image_features_val = torch.load(clip_image_embeddings_val_path, weights_only=True, map_location=device)
    else:
        print("Extract clip image features with CLIP model: {}".format(clip_model_name))
        clip_image_features_train, clip_image_features_val = get_clip_image_features(device, clip_model_name, data_name)
        torch.save(clip_image_features_train, clip_image_embeddings_train_path)
        torch.save(clip_image_features_val, clip_image_embeddings_val_path)
    data_train = ActivationDataset(clip_image_features_train)
    data_val = ActivationDataset(clip_image_features_val)
    bb_features_train = []
    print("Extracting hidden space features of the black-box model (training data)")
    with torch.no_grad():
        for x in tqdm(DataLoader(data_train, batch_size=256, shuffle=False)):
            x = x.to(device).float()
            features = bb_model.features(x)
            bb_features_train.append(features.view(features.size(0), -1))
    bb_features_train = torch.cat(bb_features_train)
    bb_features_val = []
    print("Extracting hidden space features of the black-box model (validation data)")
    with torch.no_grad():
        for x in tqdm(DataLoader(data_val, batch_size=256, shuffle=False)):
            x = x.to(device).float()
            features = bb_model.features(x)
            bb_features_val.append(features.view(features.size(0), -1))
    bb_features_val = torch.cat(bb_features_val)
    return bb_features_train, bb_features_val