import os
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_utils import get_data, ActivationDataset
import torch.nn as nn


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
    


# def get_clip_scores(device, clip_model_name, data_name, power, concept_bank, 
#                     clip_text_embeddings_path, clip_image_embeddings_train_path, clip_image_embeddings_val_path, 
#                     clip_similarities_train_path, clip_similarities_val_path, save_path_mean, save_path_std):
#     text_features = get_clip_text_features(device, clip_model_name, concept_bank, clip_text_embeddings_path)
#         image_features_train, image_features_val = get_clip_image_features(device, clip_model_name, data_name, 
#                                                                            clip_image_embeddings_train_path, clip_image_embeddings_val_path)
#         clip_similarities_train = calculate_clip_similarity(image_features_train, text_features)
#         clip_similarities_val = calculate_clip_similarity(image_features_val, text_features)
#         # exponential transformation
#         clip_similarities_train = clip_similarities_train**power
#         clip_similarities_val = clip_similarities_val**power
#     if os.path.exists(save_path_mean) and os.path.exists(save_path_std):
#         clip_similarities_train_mean = torch.load(save_path_mean, map_location=device, weights_only=True)
#         clip_similarities_train_std = torch.load(save_path_std, map_location=device, weights_only=True)
#     else:
#         clip_similarities_train_mean = clip_similarities_train.mean()
#         clip_similarities_train_std = clip_similarities_train.std()
#         torch.save(clip_similarities_train_mean, save_path_mean)
#         torch.save(clip_similarities_train_std, save_path_std)
#     return clip_similarities_train, clip_similarities_val, clip_similarities_train_mean, clip_similarities_train_std




    
        
