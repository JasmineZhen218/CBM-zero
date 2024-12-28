import torch
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
# customized functions
from Utils.utils import set_seed, remove_duplicates
from Utils.data_utils import load_labels
from Utils.evaluation_utils import calculate_x_factuality


parser = argparse.ArgumentParser(description="Settings for creating CBM")
parser.add_argument("--data_name", type=str, default="cifar10", help="data name")
parser.add_argument(
    "--concept_set_source",
    type=str,
    default="cifar10_conceptnet",
    help="concept set source",
)
parser.add_argument(
    "--black_box_model_name",
    type=str,
    default="clip_mlp_ViT-L_14-h64_cifar10",
    help="black-box classification model name",
)
parser.add_argument(
    "--power",
    type=int,
    default=5,
    help="exponential transformation for the clip scores",
)
parser.add_argument(
    "--pc_threshold",
    type=float,
    default=0.8,
    help="pearson correlation threshold for filtering concepts",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.5,
    help="L1 norm ratio",
)
parser.add_argument(
    "--n_iter", type=int, default=5000, help="number of iterations for optimization"
)
parser.add_argument(
    "--clip_model_name", type=str, default="ViT-L_14", help="clip model name"
)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--device", type=str, default="cuda:2", help="device to use")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--log", action="store_true", help="use concept set list")
args = parser.parse_args()
set_seed(args.seed)


def test_cbm_zero(args):
    # get save path
    projection_path = f"checkpoints/Data[{args.data_name}]_ClassiModel[{args.black_box_model_name}]_ClipModel[{args.clip_model_name}]_Concept[{args.concept_set_source}]_Power[{args.power}].pt"
    bb_features_val_path = f"saved_bb_features/Data[{args.data_name}]_Model[{args.black_box_model_name}]_val.pt"
    bb_last_fcn_w_path = f"saved_bb_last_FCN/{args.black_box_model_name}_w.pt"
    bb_last_fcn_b_path = f"saved_bb_last_FCN/{args.black_box_model_name}_b.pt"
    cx_val_path = f"saved_cx/Data[{args.data_name}]_Model[{args.clip_model_name}]_Concept[{args.concept_set_source}]_Power[{args.power}]_val.pt"
    # load class names
    print("Load class names")
    with open("asset/class_names/{}.txt".format(args.data_name)) as f:
        classes = f.read().split("\n")
    # load concept bank
    print("Load concept bank")
    with open("asset/concept_bank/{}.txt".format(args.concept_set_source)) as f:
        concept_bank = f.read().split("\n")
    concept_bank = [c for c in concept_bank if c not in ["", " "]]
    concept_bank = [i for i in concept_bank if i not in classes]
    concept_bank = remove_duplicates(concept_bank)
    # load cx
    print("Load clip similarities")
    cx_val = torch.load(cx_val_path, map_location=args.device, weights_only=True)
    # load black-box's hidden space embeddings and last FCN layer
    print("Load black-box model's hidden features")
    bb_features_val = torch.load(
            bb_features_val_path, map_location=args.device, weights_only=True
        )
    proj_activation2class = torch.load(
            bb_last_fcn_w_path, map_location=args.device, weights_only=True
        )
    proj_activation2class_bias = torch.load(
            bb_last_fcn_b_path, map_location=args.device, weights_only=True
        )
    # check the accuracy of the black-box model
    print("Check the accuracy of the black-box model")
    logits_val = (
        bb_features_val.to(args.device) @ proj_activation2class.T
        + proj_activation2class_bias
    )
    predictions_val = torch.argmax(logits_val, dim=1)
    _, labels_val = load_labels(args.data_name)
    labels_val = torch.tensor(labels_val).to(args.device)
    acc_val = torch.sum(predictions_val == labels_val).item() / len(labels_val)
    print("Accuracy of the black-box model: {:.4f}".format(acc_val))

    # filter concepts by pearson correlation
    ground_truth = pd.read_csv(
        "asset/ground_truth/{}.csv".format(args.concept_set_source)
    )
    c_indices = torch.load(
        f"saved_projections/Data[{args.data_name}]_ClassiModel[{args.black_box_model_name}]_ClipModel[{args.clip_model_name}]_Concept[{args.concept_set_source}]_Power[{args.power}]_c_indices.pt",
    )
    concept_bank = [concept_bank[i] for i in c_indices]
    ground_truth = ground_truth.loc[ground_truth["concept"].isin(concept_bank)]
    cx_val = cx_val[:, c_indices]

    # load saved model
    best_model = torch.load(projection_path, args.device)
    proj_activation2concept = best_model["weight"].detach()
    proj_activation2concept_bias = best_model["bias"].detach()
    proj_concept2class = torch.matmul(
        proj_activation2class, torch.linalg.pinv(proj_activation2concept)
    )  # (k, d) @ (d, M) -> (k, M)
    proj_concept2class_bias = (
        proj_activation2class_bias
        - proj_activation2class
        @ torch.linalg.pinv(proj_activation2concept)
        @ proj_activation2concept_bias
    ) #new bias: b-AW^+h

    # check the accuracy of CBM
    concept_bottleneck = (
        torch.matmul(bb_features_val, proj_activation2concept.T)
        + proj_activation2concept_bias
    )
    logits_val = (
        proj_concept2class @ concept_bottleneck.T
        + proj_concept2class_bias.reshape(-1, 1)
    ).T
    predictions_val = torch.argmax(logits_val, dim=1)
    acc_val = torch.sum(predictions_val == labels_val).item() / len(labels_val)
    print("Accuracy of CBM: {:.4f}".format(acc_val))

    # alignment
    cx_val_hat = (
        torch.matmul(bb_features_val, proj_activation2concept.T)
        + proj_activation2concept_bias
    )
    pearson_corr = []
    for c in range(cx_val_hat.shape[1]):
        corr, _ = pearsonr(cx_val[:, c].cpu(), cx_val_hat[:, c].cpu())
        pearson_corr.append(corr)
    pearson_corr = torch.tensor(pearson_corr)
    print("Concept alignment: {:.4f}".format(pearson_corr.mean().item()))
    # sparsity
    threshold = 0.01
    sparsity = torch.sum(torch.abs(proj_concept2class) < threshold).item() / (proj_concept2class.shape[0] * proj_concept2class.shape[1])
    print("Concept sparsity (ratio of zero weights): {:.4f}".format(sparsity))

    # x-factuality@10
    feature_importance = (
        proj_concept2class.detach().cpu().numpy()
        - proj_concept2class.detach().cpu().numpy().mean(axis=0)
    )
    mean, std = calculate_x_factuality(
        args.data_name,
        classes,
        np.array(concept_bank),
        ground_truth,
        feature_importance,
        top_k=10,
    )
    print(
        f"Gloabl explanation quality: X-factuality@10 = {mean:.3f} ± {std:.3f}"
    )

if __name__ == "__main__":
    args = parser.parse_args()
    test_cbm_zero(args)
