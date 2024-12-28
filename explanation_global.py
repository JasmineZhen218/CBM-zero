import random
import torch
import argparse
import numpy as np
import pandas as pd
# customized functions
from Utils.utils import set_seed, remove_duplicates
from Utils.evaluation_utils import calculate_x_factuality
from Utils.visualization_utils import draw_global_concept_importance, draw_global_concept_importance_contrast


parser = argparse.ArgumentParser(description="Settings for creating CBM")
parser.add_argument("--data_name", type=str, default="cifar10", help="data name")
parser.add_argument("--class_name", type=str, default=None, help="class name to visualize")
parser.add_argument("--class_name_contrast", type=str, default=None, help="contrast class name to visualize")
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


def global_explanations(args):
    # get save path
    projection_path = f"checkpoints/Data[{args.data_name}]_ClassiModel[{args.black_box_model_name}]_ClipModel[{args.clip_model_name}]_Concept[{args.concept_set_source}]_Power[{args.power}].pt"
    bb_last_fcn_w_path = f"saved_bb_last_FCN/{args.black_box_model_name}_w.pt"
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
    # load black-box's hidden space embeddings and last FCN layer
    print("Load black-box model's hidden features")
    proj_activation2class = torch.load(
        bb_last_fcn_w_path, map_location=args.device, weights_only=True
    )
    # filter concepts by pearson correlation
    ground_truth = pd.read_csv(
        "asset/ground_truth/{}.csv".format(args.concept_set_source)
    )
    c_indices = torch.load(
        f"saved_projections/Data[{args.data_name}]_ClassiModel[{args.black_box_model_name}]_ClipModel[{args.clip_model_name}]_Concept[{args.concept_set_source}]_Power[{args.power}]_c_indices.pt",
    )
    concept_bank = [concept_bank[i] for i in c_indices]
    ground_truth = ground_truth.loc[ground_truth["concept"].isin(concept_bank)]
    # load saved model
    best_model = torch.load(projection_path, args.device)
    proj_activation2concept = best_model["weight"].detach()
    proj_concept2class = torch.matmul(
        proj_activation2class, torch.linalg.pinv(proj_activation2concept)
    )  # (k, d) @ (d, M) -> (k, M)

    # x-factuality@10
    # quantitatively evaluate the global explanation quality
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
    print(f"Gloabl explanation quality: X-factuality@10 = {mean:.3f} ± {std:.3f}")

    # qualitative evaluation
    if args.class_name is None:
        args.class_name = random.choice(classes)
        print(f"Class name is randomly selected: {args.class_name}")
    # show the global concept importance for a specific class
    draw_global_concept_importance(
        args.data_name, classes, args.class_name, concept_bank, feature_importance, top_k=10
    )
    # show the contrast of global concept importance between two classes
    if args.class_name_contrast is None:
        args.class_name_contrast = random.choice(classes)
        print(f"Contrast class name is randomly selected: {args.class_name_contrast}")
    draw_global_concept_importance_contrast(
        args.data_name,
        classes,
        args.class_name,
        args.class_name_contrast,
        concept_bank,
        feature_importance,
    )

if __name__ == "__main__":
    args = parser.parse_args()
    global_explanations(args)
