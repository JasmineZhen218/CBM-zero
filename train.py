import os
import torch
import argparse
import numpy as np
import pandas as pd
from utils import set_seed, remove_duplicates
from data_utils import load_labels, get_data
from clip_utils import get_clip_image_features, get_clip_text_features, calculate_clip_similarity
from clip_MLP import get_clip_mlp_features, get_clip_mlp_fcn
from cbm_utils import ensure_full_rank, loss_fn
from evaluation_utils import calculate_x_factuality

parser = argparse.ArgumentParser(description="Settings for creating CBM")
parser.add_argument("--data_name", type=str, default="cifar10", help="data name")
parser.add_argument(
    "--concept_set_source", type=str, default="cifar10_conceptnet", help="concept set source"
)
parser.add_argument(
    "--black_box_model_name",
    type=str,
    default="clip_mlp_ViT-L_14-h64_cifar10",
    help="black-box classification model name",
) 
parser.add_argument(
    "--power", type=int, default=5, help="exponential transformation for the clip scores"
)
parser.add_argument(
    "--lambd",
    type=float,
    default=2.0,
    help="regularization parameter for the projection",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.99,
    help="L1 norm ratio",
)
parser.add_argument(
    "--n_iter", type=int, default=1000, help="number of iterations for optimization"
)
parser.add_argument(
    "--clip_model_name", type=str, default="ViT-L_14", help="clip model name"
)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--device", type=str, default="cuda:5", help="device to use")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument(
    "--log", action="store_true", help="use concept set list"
)
args = parser.parse_args()
set_seed(args.seed)


if args.log:
    import wandb
    wandb.init(
            project="transparent_cbm",
            name="Data[{}]_ClassiModel[{}]_ClipModel[{}]_Sim[{}]_ConceptSource[{}]".format(
                args.data_name,
                args.classi_model_name,
                args.clip_model_name,
                args.concept_set_source,
            ),
            tags=["T-CBM"],
        )
    wandb.config.update(args)

def train_cbm_zero(args):
    # get save path
    os.makedirs("saved_bb_features", exist_ok=True)
    os.makedirs("saved_clip_features", exist_ok=True)
    os.makedirs("saved_projections", exist_ok=True)
    os.makedirs("saved_cx", exist_ok=True)
    projection_path = f"saved_projections/Data[{args.data_name}]_ClassiModel[{args.black_box_model_name}]_ClipModel[{args.clip_model_name}]_Concept[{args.concept_set_source}]_Lambda[{args.lambd}]_Power[{args.power}].pt"
    bb_features_train_path = f"saved_bb_features/Data[{args.data_name}]_Model[{args.black_box_model_name}]_train.pt"
    bb_features_val_path = f"saved_bb_features/Data[{args.data_name}]_Model[{args.black_box_model_name}]_val.pt"
    bb_last_fcn_w_path = f"saved_bb_last_FCN/{args.black_box_model_name}_w.pt"
    bb_last_fcn_b_path = f"saved_bb_last_FCN/{args.black_box_model_name}_b.pt"
    clip_text_embeddings_path = f"saved_clip_features/Data[{args.data_name}]_text_Model[{args.clip_model_name}]_Concept[{args.concept_set_source}].pt"
    clip_image_embeddings_train_path = f"saved_clip_features/Data[{args.data_name}]_image_train_Model[{args.clip_model_name}].pt"
    clip_image_embeddings_val_path = f"saved_clip_features/Data[{args.data_name}]_image_val_Model[{args.clip_model_name}].pt"
    cx_train_path = f"saved_cx/Data[{args.data_name}]_Model[{args.clip_model_name}]_Concept[{args.concept_set_source}]_Power[{args.power}]_train.pt"
    cx_val_path = f"saved_cx/Data[{args.data_name}]_Model[{args.clip_model_name}]_Concept[{args.concept_set_source}]_Power[{args.power}]_val.pt"

    print("Explaining the classification model: {} on dataset: {} with concept bank: {}".format(args.black_box_model_name, args.data_name, args.concept_set_source))
    # load class names
    with open("asset/class_names/{}.txt".format(args.data_name)) as f:
        classes = f.read().split("\n")
    # load concept bank
    with open("asset/concept_bank/{}.txt".format(args.concept_set_source)) as f:
        concept_bank = f.read().split("\n")
    concept_bank = [c for c in concept_bank if c not in ["", " "]]
    concept_bank = [i for i in concept_bank if i not in classes]
    concept_bank = remove_duplicates(concept_bank)

    if os.path.exists(cx_train_path) and os.path.exists(cx_val_path):
        print("Load saved clip similarities from: {} and {}".format(cx_train_path, cx_val_path))
        cx_train = torch.load(cx_train_path, map_location=args.device, weights_only=True)
        cx_val = torch.load(cx_val_path, map_location=args.device, weights_only=True)
    else:
        if args.data_name == 'cub':
            annotations_local = pd.read_csv("asset/ground_truth/{}_annotations_local.csv".format(args.data_name))
            data_train = get_data(
                "{}_train".format(args.data_name),
                None,
            )
            data_val = get_data(
                "{}_val".format(args.data_name),
                None,
            )
            annotations_local_pivot = annotations_local.pivot(
                index="image_name", columns="attribute", values="is_present"
            )
            image_ids_train = [
                "/".join(image_id.split("/")[-2:]) for image_id, _ in data_train.imgs
            ]
            image_ids_val = [
                "/".join(image_id.split("/")[-2:]) for image_id, _ in data_val.imgs
            ]
            # reorder the columns to match the order of the concepts
            annotations_local_pivot = annotations_local_pivot[concept_bank]
            cx_train = torch.tensor(
                annotations_local_pivot.loc[image_ids_train].values
            ).float().to(args.device)
            cx_val = torch.tensor(
                annotations_local_pivot.loc[image_ids_val].values
            ).float().to(args.device)
        else:
            # text features encoded by clip text encoder
            if os.path.exists(clip_text_embeddings_path):
                print("Load saved clip text embeddings from: {}".format(clip_text_embeddings_path))
                text_features = torch.load(clip_text_embeddings_path, weights_only=True, map_location=args.device)
            else:
                print("Extract clip text features with CLIP model: {}".format(args.clip_model_name))
                text_features = get_clip_text_features(args.device, args.clip_model_name, concept_bank)
                torch.save(text_features, clip_text_embeddings_path)
            # image features encoded by clip image encoder
            if os.path.exists(clip_image_embeddings_train_path) and os.path.exists(clip_image_embeddings_val_path):
                print("Load saved clip image embeddings from: {} and {}".format(clip_image_embeddings_train_path, clip_image_embeddings_val_path))
                clip_image_features_train = torch.load(clip_image_embeddings_train_path, weights_only=True, map_location=args.device)
                clip_image_features_val = torch.load(clip_image_embeddings_val_path, weights_only=True, map_location=args.device)
            else:
                print("Extract clip image features with CLIP model: {}".format(args.clip_model_name))
                clip_image_features_train, clip_image_features_val = get_clip_image_features(args.device, args.clip_model_name, args.data_name)
                torch.save(clip_image_features_train, clip_image_embeddings_train_path)
                torch.save(clip_image_features_val, clip_image_embeddings_val_path)
            # cosine similarity between image and text features
            print("Calculate cosine similarity between image and text features")
            clip_similarities_train = calculate_clip_similarity(clip_image_features_train, text_features)
            clip_similarities_val = calculate_clip_similarity(clip_image_features_val, text_features)
            # exponential transformation
            print("Exponential transformation with power: {}".format(args.power))
            cx_train = clip_similarities_train**args.power
            cx_val = clip_similarities_val**args.power
            # normalize
            print("Normalize clip similarities")
            cx_train_mean = cx_train.mean()
            cx_train_std = cx_train.std()
            cx_train = (cx_train - cx_train_mean) / cx_train_std
            cx_val = (cx_val - cx_train_mean) / cx_train_std
            torch.save(cx_train, cx_train_path)
            torch.save(cx_val, cx_val_path)

    # load black-box's hidden space embeddings and last FCN layer
    if os.path.exists(bb_features_train_path) and os.path.exists(bb_features_val_path):
        print("Load saved hidden space features from: {} and {}".format(bb_features_train_path, bb_features_val_path))
        bb_features_train = torch.load(bb_features_train_path, map_location=args.device, weights_only=True)
        bb_features_val = torch.load(bb_features_val_path, map_location=args.device, weights_only=True)
    else:
        print("Extract hidden space features of the black-box model: {}".format(args.black_box_model_name))
        if args.black_box_model_name.startswith("clip_lp_"):
            bb_features_train, bb_features_val = clip_image_features_train, clip_image_features_val
        elif args.black_box_model_name.startswith("clip_mlp_"):
            bb_features_train, bb_features_val = get_clip_mlp_features(args.device, args.black_box_model_name, clip_image_embeddings_train_path, clip_image_embeddings_val_path)
            torch.save(bb_features_train, bb_features_train_path)
            torch.save(bb_features_val, bb_features_val_path)
        else:
            print("Unknown black-box model: {}, please save hidden space features (prior to last FCN) on your own".format(args.black_box_model_name))
            raise ValueError("Unknown black-box model: {}".format(args.black_box_model_name))

    print("Load last FCN layer of the classification model")
    if os.path.exists(bb_last_fcn_w_path) and os.path.exists(bb_last_fcn_b_path):
        proj_activation2class = torch.load(bb_last_fcn_w_path,  map_location=args.device, weights_only=True)
        proj_activation2class_bias = torch.load(bb_last_fcn_b_path,  map_location=args.device, weights_only=True)
    else:
        proj_activation2class, proj_activation2class_bias = get_clip_mlp_fcn(args.black_box_model_name, args.device)
        torch.save(proj_activation2class, bb_last_fcn_w_path)
        torch.save(proj_activation2class_bias, bb_last_fcn_b_path)
    # check the accuracy of the black-box model
    logits_val = (
        bb_features_val @ proj_activation2class.T + proj_activation2class_bias)
    predictions_val = torch.argmax(logits_val, dim=1)
    _, labels_val = load_labels(args.data_name)
    labels_val = torch.tensor(labels_val).to(args.device)
    acc_val = torch.sum(predictions_val == labels_val).item() / len(labels_val)
    print("Accuracy of the black-box model: {:.4f}".format(acc_val))

    # check dimensions
    print("Check dimensions")
    d = bb_features_train.size(1)
    M = len(concept_bank)
    N_train = bb_features_train.size(0)
    N_val = bb_features_val.size(0)
    K = len(classes)
    print("Number of concepts: {}".format(M))
    print("Number of training samples: {}".format(N_train))
    print("Number of validation samples: {}".format(N_val))
    print("Dimension of black-box's hidden space features: {}".format(d))
    assert M >= d

    # init projection layer
    proj_layer = torch.nn.Linear(in_features=d, out_features=M, bias=True)
    proj_layer.to(args.device)
    optimizer = torch.optim.Adam(proj_layer.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
    ensure_full_rank(proj_layer, args.device)
    optimizer.zero_grad()
    val_previous_loss = float("inf")
    best_loss = float("inf")
    patience = 10
    wait = 0
    # tensor([4635, 1651, 5274,  ..., 5887, 4084, 2488])
    for i in range(args.n_iter):
        # sample a mini-batch
        random_indices = torch.randperm(N_train)[:10000]
        cx_train_ = cx_train[random_indices, :].to(args.device)
        bb_features_train_ = bb_features_train[random_indices, :].to(args.device)
        outs_train_ = proj_layer(bb_features_train_)  # N x M
        mse_loss_train = loss_fn(outs_train_, cx_train_)
        proj_concept2activation = torch.linalg.pinv(proj_layer.weight)  # (d, M)
        assert proj_concept2activation.shape[0] == d
        regul_loss = (args.alpha * torch.norm(proj_concept2activation, p=1)
                    + (1 - args.alpha) * torch.norm(proj_concept2activation, p=2) ** 2) / d
        loss_train = mse_loss_train + args.lambd * regul_loss
        loss_train.backward()
        optimizer.step()
        ensure_full_rank(proj_layer, args.device)
        optimizer.zero_grad()
        if (i + 1) % 10 == 0:
            with torch.no_grad():
                bb_features_val = bb_features_val.to(args.device)
                outs_val = proj_layer(bb_features_val)
                mse_loss_val = loss_fn(outs_val, cx_val)
                loss_val = (
                        mse_loss_val + args.lambd * regul_loss
                    )
                print(
                        "iter = {}\n\
                                    \tTrain|lr={:.4f}|MSE loss = {:.4f} | Regularization loss = {:.1f} | loss = {:.4f}\n\
                                    \tVal  |lr={:.4f}|MSE loss = {:.4f} | Regularization loss = {:.1f} | loss = {:.4f}".format(
                            i + 1,
                            optimizer.param_groups[0]["lr"],
                            mse_loss_train.item(),
                            regul_loss.item(),
                            loss_train.item(),
                            optimizer.param_groups[0]["lr"],
                            mse_loss_val.item(),
                            regul_loss.item(),
                            loss_val.item(),
                        )
                    )
                if args.log:
                    wandb.log(
                                {
                                    "Step": i,
                                    "loss/train": loss_train.item(),
                                    "loss/val": loss_val.item(),
                                    "mse_loss/train": mse_loss_train.item(),
                                    "mse_los/val": mse_loss_val.item(),
                                    "regul_loss": regul_loss.item(),
                                }
                            )
                if loss_val < best_loss:
                    print("Save model")
                    best_loss = loss_val
                    torch.save(proj_layer.weight.detach().cpu(), projection_path)
                    torch.save(
                                proj_layer.bias.detach().cpu(),
                                projection_path.replace(".pt", "_bias.pt"),
                            )
                # check lr
                lr_now = optimizer.param_groups[0]["lr"]
                if lr_now <= 1e-3:
                    print("Convergence reached with lr")
                    break
                else:
                    wait = 0
                val_previous_loss = loss_val
                scheduler.step(loss_val)

    print("Mapping from hidden space to concept space is saved at: {}".format(projection_path))
    # construct the CBM
    print("Derive the projection from concepts to classes")
    proj_activation2concept = torch.load(projection_path, map_location=args.device, weights_only=True)
    proj_activation2concept_bias = torch.load(
                projection_path.replace(".pt", "_bias.pt"), map_location=args.device, weights_only=True
            )
    assert torch.linalg.matrix_rank(proj_activation2concept) == d
    proj_concept2class = torch.matmul(
        proj_activation2class, torch.linalg.pinv(proj_activation2concept)
    )  # (k, d) @ (d, M) -> (k, M)
    # new bias: b-AW^+h
    proj_concept2class_bias = proj_activation2class_bias - proj_activation2class@torch.linalg.pinv(proj_activation2concept)@proj_activation2concept_bias
    # check the accuracy of CBM
    concept_bottleneck = (
                torch.matmul(bb_features_val, proj_activation2concept.T)
                + proj_activation2concept_bias
            )
    logits_val = (
                proj_concept2class @ concept_bottleneck.T + proj_concept2class_bias.reshape(-1, 1)
            ).T

    predictions_val = torch.argmax(logits_val, dim=1)
    _, labels_val = load_labels(args.data_name)
    labels_val = torch.tensor(labels_val).to(args.device)
    acc_val = torch.sum(predictions_val == labels_val).item() / N_val
    print("Accuracy of reconstructed CBM: {:.4f}".format(acc_val))
    if args.log:
        wandb.log({"acc/val": acc_val})

    feature_importance = proj_concept2class.detach().cpu().numpy() - proj_concept2class.detach().cpu().numpy().mean(
            axis=0
        )
    ground_truth = pd.read_csv("asset/ground_truth/{}.csv".format(args.concept_set_source))
    mean, std = calculate_x_factuality(
        args.data_name, classes, np.array(concept_bank), ground_truth, feature_importance, top_k=10
    )

if __name__ == "__main__":
    args = parser.parse_args()
    train_cbm_zero(args)
