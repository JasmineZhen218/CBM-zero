import os
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_utils import get_data, ActivationDataset
import torch.nn as nn

def calculate_clip_similarity(x, y):
    if x.dtype != torch.float32:
        x = x.float()
    if y.dtype != torch.float32:
        y = y.float()
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    return torch.matmul(x, y.T).float()

def loss_fn(outs, clip_similarities):
    return torch.norm(outs - clip_similarities, p=2)**2 / (outs.shape[0])
def ensure_full_rank(proj_layer, device):
    W = proj_layer.weight
    M, d = W.shape[0], W.shape[1]
    assert M >= d
    rank = torch.linalg.matrix_rank(W)
    if rank == d:
        return
    else:
        print(f"Rank of weight matrix: {rank}")
        U, S, V = torch.svd(W)
        S = S[:rank]
        for perm in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            S_ = torch.cat([S, (torch.zeros(d - rank) + perm).to(device)])
            W_ = torch.matmul(U, torch.matmul(torch.diag(S_), V.T))
            rank_ = torch.linalg.matrix_rank(W_)
            if rank_ == d:
                break
        S = S_
        W = W_
        rank = torch.linalg.matrix_rank(W)
        assert rank == d
        proj_layer.weight = torch.nn.Parameter(W)
        return
    
def get_clip_text_features(device, clip_model_name, concept_bank, clip_text_embeddings_path):
    if os.path.exists(clip_text_embeddings_path):
        text_features = torch.load(clip_text_embeddings_path, weights_only=True, map_location=device)
    else:
        # load clip
        clip_model, clip_preprocess = clip.load(
            clip_model_name.replace("_", "/"), device=device
        )
        # encode text
        with torch.no_grad():
            text_features = clip_model.encode_text(clip.tokenize(concept_bank).to(device))
        torch.save(text_features, clip_text_embeddings_path)
    return text_features
    
def get_clip_image_features(device, clip_model_name, data_name, clip_image_embeddings_train_path, clip_image_embeddings_val_path):
    if os.path.exists(clip_image_embeddings_train_path) and os.path.exists(clip_image_embeddings_val_path):
        image_features_train = torch.load(clip_image_embeddings_train_path, map_location=device, weights_only=True)
        image_features_val = torch.load(clip_image_embeddings_val_path, map_location=device, weights_only=True)
    else:
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
            for x, y in tqdm(DataLoader(dataset_train, batch_size=256, shuffle=False)):
                x = x.to(device)
                features = clip_model.encode_image(x)
                image_features_train.append(features.detach())
        image_features_train = torch.cat(image_features_train)
        image_features_val = []
        print("Encoding validation images with CLIP")
        with torch.no_grad():
            for x, y in tqdm(DataLoader(dataset_val, batch_size=256, shuffle=False)):
                x = x.to(device)
                features = clip_model.encode_image(x)
                image_features_val.append(features.detach())
        image_features_val = torch.cat(image_features_val)
        torch.save(image_features_train, clip_image_embeddings_train_path)
        torch.save(image_features_val, clip_image_embeddings_val_path)
    return image_features_train, image_features_val

def get_clip_scores(device, clip_model_name, data_name, power, concept_bank, 
                    clip_text_embeddings_path, clip_image_embeddings_train_path, clip_image_embeddings_val_path, 
                    clip_similarities_train_path, clip_similarities_val_path, save_path_mean, save_path_std):
    if os.path.exists(clip_similarities_train_path) and os.path.exists(clip_similarities_val_path):
        clip_similarities_train = torch.load(clip_similarities_train_path, map_location=device, weights_only=True)
        clip_similarities_val = torch.load(clip_similarities_val_path, map_location=device, weights_only=True)
    else:
        text_features = get_clip_text_features(device, clip_model_name, concept_bank, clip_text_embeddings_path)
        image_features_train, image_features_val = get_clip_image_features(device, clip_model_name, data_name, 
                                                                           clip_image_embeddings_train_path, clip_image_embeddings_val_path)
        clip_similarities_train = calculate_clip_similarity(image_features_train, text_features)
        clip_similarities_val = calculate_clip_similarity(image_features_val, text_features)
        # exponential transformation
        clip_similarities_train = clip_similarities_train**power
        clip_similarities_val = clip_similarities_val**power
    if os.path.exists(save_path_mean) and os.path.exists(save_path_std):
        clip_similarities_train_mean = torch.load(save_path_mean, map_location=device, weights_only=True)
        clip_similarities_train_std = torch.load(save_path_std, map_location=device, weights_only=True)
    else:
        clip_similarities_train_mean = clip_similarities_train.mean()
        clip_similarities_train_std = clip_similarities_train.std()
        torch.save(clip_similarities_train_mean, save_path_mean)
        torch.save(clip_similarities_train_std, save_path_std)
    return clip_similarities_train, clip_similarities_val, clip_similarities_train_mean, clip_similarities_train_std

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

def get_bb_fcn(black_box_model_name, device):
    if black_box_model_name.startswith("clip_mlp"):
        bb_model = load_clip_mlp_model(black_box_model_name)
        state_dict = torch.load(
            "saved_black_box_models/{}.pt".format(black_box_model_name), weights_only=True,
        )
        bb_model.load_state_dict(state_dict)
        proj_activation2class = bb_model.output.weight
        proj_activation2class_bias = bb_model.output.bias
    else:
        raise ValueError("Unknown black-box model: {}".format(black_box_model_name))
    return proj_activation2class.to(device), proj_activation2class_bias.to(device)

def get_bb_features(device, black_box_model_name, data_name, 
                    bb_features_train_path, bb_features_val_path, 
                    clip_image_embeddings_train_path, clip_image_embeddings_val_path):
    if os.path.exists(bb_features_train_path) and os.path.exists(bb_features_val_path):
        bb_features_train = torch.load(bb_features_train_path, map_location=device, weights_only=True)
        bb_features_val = torch.load(bb_features_val_path, map_location=device, weights_only=True)
    else:
        if black_box_model_name.startswith("clip_lp_"):
            bb_features_train, bb_features_val = get_clip_image_features(device, black_box_model_name, data_name, 
                                                                         clip_image_embeddings_train_path, clip_image_embeddings_val_path)
        elif black_box_model_name.startswith("clip_mlp_"):
            clip_model_name = black_box_model_name.split("_")[2]
            bb_model = load_clip_mlp_model(black_box_model_name)
            state_dict = torch.load(
                "saved_black_box_models/{}.pt".format(black_box_model_name),
                map_location=device,
                weights_only=True,
            )
            bb_model.load_state_dict(state_dict)
            bb_model.to(device)
            clip_image_features_train, clip_image_features_val = get_clip_image_features(
                device, clip_model_name, data_name, clip_image_embeddings_train_path, clip_image_embeddings_val_path
            )
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
            # save
            torch.save(bb_features_train, bb_features_train_path)
            torch.save(bb_features_val, bb_features_val_path)
        else:
            raise ValueError("Unknown black-box model: {}".format(black_box_model_name))
    return bb_features_train, bb_features_val
    
        
