import torch
import argparse
import numpy as np
import pandas as pd
import random
# customized functions
from Utils.utils import set_seed, remove_duplicates
from Utils.data_utils import load_labels, get_data
from Utils.visualization_utils import draw_local_explanations

parser = argparse.ArgumentParser(description="Settings for creating CBM")
parser.add_argument("--data_name", type=str, default="cifar10", help="data name")
parser.add_argument("--class_name", type=str, default=None, help="class name to visualize")
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


def local_explanations(args):
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
    logits_val = (
        bb_features_val.to(args.device) @ proj_activation2class.T
        + proj_activation2class_bias
    )
    predictions_val = torch.argmax(logits_val, dim=1)
    _, labels_val = load_labels(args.data_name)
    labels_val = torch.tensor(labels_val).to(args.device)

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
    )  # new bias: b-AW^+h

    outs_val = (
        torch.matmul(bb_features_val.cpu(), proj_activation2concept.cpu().T)
        + proj_activation2concept_bias.cpu()
    )
    data_val = get_data(
        "{}_val".format(args.data_name),
        None,
    )
    if args.class_name is None:
        args.class_name = random.choice(classes)
        print(f"Class name is randomly selected: {args.class_name}")
    indices = np.where(
        (predictions_val.cpu() == labels_val.cpu()).numpy()
        & (labels_val.cpu() == classes.index(args.class_name)).numpy()
    )[0]

    image_id = random.choice(indices)
    image = data_val[image_id][0]
    label = labels_val[image_id]
    predicted_label = predictions_val[image_id]
    class_name = classes[label]
    predicted_class = classes[predicted_label]

    logit_n = (
        (logits_val[image_id][predicted_label] - logits_val[image_id].mean())
        .detach()
        .cpu()
    )
    proj_concept2class_n = (proj_concept2class[predicted_label] - proj_concept2class.mean(dim=0)).detach().cpu()
    concept_activation = outs_val[image_id].cpu()
    concept_contributions_n = proj_concept2class_n * concept_activation
    

    draw_local_explanations(
        args.data_name,
        concept_bank,
        image,
        image_id,
        class_name,
        predicted_class,
        logit_n,
        concept_contributions_n,
        proj_concept2class_n,
        concept_activation,
        max_display=10,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    local_explanations(args)
