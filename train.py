import os
import torch
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
# customized functions
from Utils.utils import set_seed, remove_duplicates
from Utils.data_utils import load_labels, get_data, process_cub_annotations
from Utils.clip_utils import (
    get_clip_image_features,
    get_clip_text_features,
    calculate_clip_similarity,
)
from Utils.bb_utils import get_clip_mlp_features, get_clip_mlp_fcn, get_clip_lp_fcn
from Utils.cbm_utils import ensure_full_rank, loss_fn
from Utils.evaluation_utils import calculate_x_factuality
from Utils.visualization_utils import present_lambda_search


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
    default=0,
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
        tags=["CBM-zero"],
    )
    wandb.config.update(args)


def load_cx(
    cx_train_path,
    cx_val_path,
    clip_text_embeddings_path,
    clip_image_embeddings_train_path,
    clip_image_embeddings_val_path,
    concept_bank,
    args,
):
    """
    Load or compute the concept features (exponential transformed, normalized CLIP similarities OR ground truth)
    Args:
        cx_train_path (str): Path to save/load the training cosine similarities.
        cx_val_path (str): Path to save/load the validation cosine similarities.
        clip_text_embeddings_path (str): Path to save/load the CLIP text embeddings.
        clip_image_embeddings_train_path (str): Path to save/load the CLIP image embeddings for training data.
        clip_image_embeddings_val_path (str): Path to save/load the CLIP image embeddings for validation data.
        concept_bank (list): List of concepts to be used for generating text embeddings.
        args (argparse.Namespace): Arguments containing various settings and configurations.
    Returns:
        tuple: A tuple containing:
            - cx_train (torch.Tensor): Concept features (c(x)) for the training data. shape: (N_train, M)
            - cx_val (torch.Tensor): Concept features (c(x)) for the validation data. shape: (N_val, M)
    """
    
    if os.path.exists(cx_train_path) and os.path.exists(cx_val_path):
        print(
            "Load saved clip similarities from: {} and {}".format(
                cx_train_path, cx_val_path
            )
        )
        cx_train = torch.load(cx_train_path, map_location="cpu", weights_only=True)
        cx_val = torch.load(cx_val_path, map_location="cpu", weights_only=True)
    else:
        if args.data_name == "cub":
            annotations_local_pivot = process_cub_annotations(concept_bank)
            data_train = get_data(
                "{}_train".format("cub"),
                None,
            )
            data_train = get_data(
                "{}_train".format("cub"),
                None,
            )
            data_val = get_data(
                "{}_val".format("cub"),
                None,
            )
            image_ids_train = [
                "/".join(image_id.split("/")[-2:]) for image_id, _ in data_train.imgs
            ]
            image_ids_val = [
                "/".join(image_id.split("/")[-2:]) for image_id, _ in data_val.imgs
            ]
            cx_train = (
                torch.tensor(annotations_local_pivot.loc[image_ids_train].values)
                .float()
                .to(args.device)
            )
            cx_val = (
                torch.tensor(annotations_local_pivot.loc[image_ids_val].values)
                .float()
                .to(args.device)
            )
        else:
            # text features encoded by clip text encoder
            if os.path.exists(clip_text_embeddings_path):
                print(
                    "Load saved clip text embeddings from: {}".format(
                        clip_text_embeddings_path
                    )
                )
                text_features = torch.load(
                    clip_text_embeddings_path,
                    weights_only=True,
                    map_location="cpu",
                )
            else:
                print(
                    "Extract clip text features with CLIP model: {}".format(
                        args.clip_model_name
                    )
                )
                text_features = get_clip_text_features(
                    "cpu", args.clip_model_name, concept_bank
                )
                torch.save(text_features, clip_text_embeddings_path)
            # image features encoded by clip image encoder
            if os.path.exists(clip_image_embeddings_train_path) and os.path.exists(
                clip_image_embeddings_val_path
            ):
                print(
                    "Load saved clip image embeddings from: {} and {}".format(
                        clip_image_embeddings_train_path, clip_image_embeddings_val_path
                    )
                )
                clip_image_features_train = torch.load(
                    clip_image_embeddings_train_path,
                    weights_only=True,
                    map_location="cpu",
                )
                clip_image_features_val = torch.load(
                    clip_image_embeddings_val_path,
                    weights_only=True,
                    map_location="cpu",
                )
            else:
                print(
                    "Extract clip image features with CLIP model: {}".format(
                        args.clip_model_name
                    )
                )
                clip_image_features_train, clip_image_features_val = (
                    get_clip_image_features("cpu", args.clip_model_name, args.data_name)
                )
                torch.save(clip_image_features_train, clip_image_embeddings_train_path)
                torch.save(clip_image_features_val, clip_image_embeddings_val_path)
            # cosine similarity between image and text features
            print("Calculate cosine similarity between image and text features")
            clip_similarities_train = calculate_clip_similarity(
                clip_image_features_train, text_features
            )
            clip_similarities_val = calculate_clip_similarity(
                clip_image_features_val, text_features
            )
            # exponential transformation
            print("Exponential transformation with power: {}".format(args.power))
            cx_train = clip_similarities_train**args.power
            cx_val = clip_similarities_val**args.power
            # normalize
            print("Normalize clip similarities")
            cx_train_mean = cx_train.mean(dim=0)
            cx_train_std = cx_train.std(dim=0)
            cx_train = (cx_train - cx_train_mean) / cx_train_std
            cx_val = (cx_val - cx_train_mean) / cx_train_std
            torch.save(cx_train, cx_train_path)
            torch.save(cx_val, cx_val_path)
    return cx_train, cx_val


def load_bb(
    bb_features_train_path,
    bb_features_val_path,
    clip_image_embeddings_train_path,
    clip_image_embeddings_val_path,
    bb_last_fcn_w_path,
    bb_last_fcn_b_path,
    args,
):
    """
    Load black-box model features and the last fully connected layer weights.
    This function loads the training and validation features for a black-box model
    from the specified paths. If the features do not exist, it attempts to generate
    them based on the model name specified in the arguments. Additionally, it loads
    the weights and biases of the last fully connected layer of the classification model.
    Args:
        bb_features_train_path (str): Path to the training features of the black-box model.
        bb_features_val_path (str): Path to the validation features of the black-box model.
        clip_image_embeddings_train_path (str): Path to the training image embeddings for CLIP model.
        clip_image_embeddings_val_path (str): Path to the validation image embeddings for CLIP model.
        bb_last_fcn_w_path (str): Path to the weights of the last fully connected layer.
        bb_last_fcn_b_path (str): Path to the biases of the last fully connected layer.
        args (Namespace): Arguments containing model configuration and device information.
    Returns:
        tuple: A tuple containing:
            - bb_features_train (Tensor): Training features of the black-box model. shape: (N_train, d)
            - bb_features_val (Tensor): Validation features of the black-box model. shape: (N_val, d)
            - proj_activation2class (Tensor): Weights of the last fully connected layer. shape: shape: (K, d)
            - proj_activation2class_bias (Tensor): Biases of the last fully connected layer. shape: (K,)
    Raises:
        ValueError: If the black-box model name is unknown.
    """
    if os.path.exists(bb_features_train_path) and os.path.exists(bb_features_val_path):
        bb_features_train = torch.load(
            bb_features_train_path, map_location=args.device, weights_only=True
        )
        bb_features_val = torch.load(
            bb_features_val_path, map_location=args.device, weights_only=True
        )
    else:
        if args.black_box_model_name.startswith("clip_lp_"):
            bb_features_train = torch.load(
                clip_image_embeddings_train_path,
                weights_only=True,
                map_location=args.device,
            ).float()

            bb_features_val = torch.load(
                clip_image_embeddings_val_path,
                weights_only=True,
                map_location=args.device,
            ).float()
        elif args.black_box_model_name.startswith("clip_mlp_"):
            bb_features_train, bb_features_val = get_clip_mlp_features(
                args.device,
                args.black_box_model_name,
                clip_image_embeddings_train_path,
                clip_image_embeddings_val_path,
            )
            torch.save(bb_features_train, bb_features_train_path)
            torch.save(bb_features_val, bb_features_val_path)
        else:
            print(
                "Unknown black-box model: {}, please save hidden space features (prior to last FCN) on your own".format(
                    args.black_box_model_name
                )
            )
            raise ValueError(
                "Unknown black-box model: {}".format(args.black_box_model_name)
            )

    print("Load last FCN layer of the classification model")
    if os.path.exists(bb_last_fcn_w_path) and os.path.exists(bb_last_fcn_b_path):
        proj_activation2class = torch.load(
            bb_last_fcn_w_path, map_location=args.device, weights_only=True
        )
        proj_activation2class_bias = torch.load(
            bb_last_fcn_b_path, map_location=args.device, weights_only=True
        )
    else:
        if args.black_box_model_name.startswith("clip_mlp_"):
            proj_activation2class, proj_activation2class_bias = get_clip_mlp_fcn(
                args.black_box_model_name, args.device
            )

        elif args.black_box_model_name.startswith("clip_lp_"):
            proj_activation2class, proj_activation2class_bias = get_clip_lp_fcn(
                args.black_box_model_name, args.device
            )
        else:

            print(
                "Unknown black-box model: {}, please save last FCN layer on your own".format(
                    args.black_box_model_name
                )
            )
            raise ValueError(
                "Unknown black-box model: {}".format(args.black_box_model_name)
            )
        torch.save(proj_activation2class, bb_last_fcn_w_path)
        torch.save(proj_activation2class_bias, bb_last_fcn_b_path)
    return (
        bb_features_train,
        bb_features_val,
        proj_activation2class,
        proj_activation2class_bias,
    )


def calculate_closed_form_solution(
    cx_train, cx_val, bb_features_train, bb_features_val
):
    """
    Calculate the closed-form solution for linear regression and evaluate its performance.
    Args:
        cx_train (torch.Tensor): Training target values.
        cx_val (torch.Tensor): Validation target values.
        bb_features_train (torch.Tensor): Training features from the black-box model.
        bb_features_val (torch.Tensor): Validation features from the black-box model.
    Returns:
        tuple: A tuple containing:
            - W_hat (torch.Tensor): The calculated weight matrix. 
            - pearson_corr (torch.Tensor): Pearson correlation coefficients between 
              the predicted and actual validation target values for each target dimension.
    """    
    print("Calculate closed-form solution")
    F = bb_features_train.cpu()
    F = torch.cat([F, torch.ones((bb_features_train.size(0), 1), device="cpu")], dim=1)
    C = cx_train.cpu()
    W_hat = C.T @ F @ torch.linalg.pinv(F.T @ F)
    # inference
    cx_val_hat = (
        torch.cat(
            [
                bb_features_val.cpu(),
                torch.ones((bb_features_val.size(0), 1), device="cpu"),
            ],
            dim=1,
        )
        @ W_hat.T
    )
    pearson_corr = []
    for c in range(cx_train.size(1)):
        corr, _ = pearsonr(cx_val[:, c].cpu(), cx_val_hat[:, c].cpu())
        pearson_corr.append(corr)
    pearson_corr = torch.tensor(pearson_corr)
    return W_hat, pearson_corr


def filter_concepts_by_pearson_correlation(
    c_indices, pearson_corr, concept_bank, ground_truth, cx_train, cx_val, args
):
    """
    Filters concepts based on Pearson correlation and returns the filtered results.
    Args:
        c_indices (torch.tensor): Indices of the concepts to be filtered. shape: (M,)
        pearson_corr (torch.tensor): Pearson correlation values. shape: (M,)
        concept_bank (list): List of all available concepts. shape: (M,)
        ground_truth (pandas.DataFrame): DataFrame containing ground truth concepts.
        cx_train (torch.tensor): Training data with concept features.  shape: (N_train, M)
        cx_val (numpy.ndarray): Validation data with concept features. shape: (N_val, M)
        args (argparse.Namespace): Arguments containing the Pearson correlation threshold.
    Returns:
        tuple: A tuple containing:
            - concept_bank (list): Filtered list of concepts. shape: (M',)
            - ground_truth (pandas.DataFrame): Filtered ground truth DataFrame. 
            - cx_train (numpy.ndarray): Filtered training data. shape: (N_train, M')
            - cx_val (numpy.ndarray): Filtered validation data. shape: (N_val, M')
    """    
    # filter
    concept_bank = [concept_bank[i] for i in c_indices]
    ground_truth = ground_truth.loc[ground_truth["concept"].isin(concept_bank)]
    pearson_corr = pearson_corr[c_indices]
    # best alignment
    best_pearson_corr = pearson_corr.mean().item()
    print(
        "Filter unaligned concepts by pearson correlation with threshold: {:.4f}".format(
            args.pc_threshold
        )
    )
    print(
        "{} concepts are selected with mean pearson correlation: {:.4f} (max: {:.4f} - min: {:.4f})".format(
            len(concept_bank), best_pearson_corr, max(pearson_corr), min(pearson_corr)
        )
    )
    cx_train = cx_train[:, c_indices]
    cx_val = cx_val[:, c_indices]
    return concept_bank, ground_truth, cx_train, cx_val


def train(
    args,
    lambd,
    cx_train,
    cx_val,
    bb_features_train,
    bb_features_val,
    proj_activation2class,
    concept_bank,
    classes,
):
    """
    Train a invertible projection layer to map black-box features to concept space.
    Args:
        args (Namespace): A namespace containing training parameters such as device, learning rate, 
                          number of iterations, and regularization parameters.
        lambd (float): Regularization parameter.
        cx_train (Tensor): Training concept labels. shape: (N_train, M)
        cx_val (Tensor): Validation concept labels. shape: (N_val, M)
        bb_features_train (Tensor): Training black-box features. shape: (N_train, d) 
        bb_features_val (Tensor): Validation black-box features. shape: (N_val, d)
        proj_activation2class (Tensor): Projection matrix from black-box model's hidden space to class space, $A$. shape: (K, d)
        concept_bank (list): List of concepts. shape: (M,)
        classes (list): List of classes names. shape: (K,)
    Returns:
        tuple: A tuple containing:
            - pearson_corr (Tensor): Pearson correlation coefficients for each concept. shape: (M,)
            - sparsity (float): Sparsity measure of the projection from concept to class space.
            - best_model (dict): State dictionary of the best model. 
    """    
    print("Regularization parameter: {:.4f}".format(lambd))
    d = bb_features_train.size(1)
    M = len(concept_bank)
    N_train = bb_features_train.size(0)
    K = len(classes)
    assert M >= d
    proj_layer = torch.nn.Linear(in_features=d, out_features=M, bias=True)
    proj_layer.to(args.device)
    optimizer = torch.optim.Adam(proj_layer.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
    ensure_full_rank(proj_layer, args.device)
    optimizer.zero_grad()
    best_loss = float("inf")
    for i in tqdm(range(args.n_iter)):
        random_indices = torch.randperm(N_train)[:10000]
        cx_train_ = cx_train[random_indices, :].to(args.device)
        bb_features_train_ = bb_features_train[random_indices, :].to(args.device)
        outs_train_ = proj_layer(bb_features_train_)  # N x M
        mse_loss_train = loss_fn(outs_train_, cx_train_)
        proj_concept2activation = torch.linalg.pinv(proj_layer.weight)  # (d, M)
        proj_concept2class = torch.matmul(
            proj_activation2class, proj_concept2activation
        )  # (k, d) @ (d, M) -> (k, M)
        assert proj_concept2activation.shape[0] == d
        regul_loss = (
            args.alpha * torch.norm(proj_concept2class, p=1)
            + (1 - args.alpha) * torch.norm(proj_concept2class, p=2) ** 2
        ) / K
        loss_train = mse_loss_train + lambd * regul_loss
        loss_train.backward()
        optimizer.step()
        ensure_full_rank(proj_layer, args.device)
        optimizer.zero_grad()
        if (i + 1) % 10 == 0:
            with torch.no_grad():
                bb_features_val = bb_features_val.to(args.device)
                outs_val = proj_layer(bb_features_val)
                mse_loss_val = loss_fn(outs_val, cx_val.to(args.device))
                loss_val = mse_loss_val + lambd * regul_loss
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_model = proj_layer.state_dict()
                # check lr
                lr_now = optimizer.param_groups[0]["lr"]
                if lr_now <= 1e-3:
                    print("Convergence reached with lr")
                    break
                scheduler.step(loss_val)

    proj_activation2concept = best_model["weight"].detach()
    proj_activation2concept_bias = best_model["bias"].detach()
    proj_concept2class = torch.matmul(
        proj_activation2class, torch.linalg.pinv(proj_activation2concept)
    )  # (k, d) @ (d, M) -> (k, M)
    cx_val_hat = (
        torch.matmul(bb_features_val, proj_activation2concept.T)
        + proj_activation2concept_bias
    )
    # alignment
    pearson_corr = []
    for c in range(M):
        corr, _ = pearsonr(cx_val[:, c].cpu(), cx_val_hat[:, c].cpu())
        pearson_corr.append(corr)
    pearson_corr = torch.tensor(pearson_corr)
    # mean_corr = pearson_corr.mean().item()
    # sparsity
    threshold = 0.01
    sparsity = torch.sum(torch.abs(proj_concept2class) < threshold).item() / (K * M)
    return pearson_corr, sparsity, best_model


def train_cbm_zero(args):
    """
    Train the CBM-zero model with the given arguments.
    Args:
        args (Namespace): The arguments required for training, including:
            - data_name (str): The name of the dataset.
            - black_box_model_name (str): The name of the black-box model.
            - clip_model_name (str): The name of the CLIP model.
            - concept_set_source (str): The source of the concept set.
            - power (int): The power parameter.
            - pc_threshold (float): The Pearson correlation threshold.
            - device (str): The device to use for computation (e.g., 'cpu' or 'cuda').
    Returns:
        None
    This function performs the following steps:
        1. Creates necessary ditectories paths for saving features and checkpoints.
        2. Loads class names and concept bank.
        3. Loads concept features (c(x)). 
        4. Loads black-box model's hidden features and last FCN layer.
        5. Checks the accuracy of the black-box model.
        6. Calculates the closed-form solution to measure maximum concept alignment.
        7. Filters concepts by Pearson correlation. 
        8. Searches for the best lambda value.
        9. Trains the model with the best lambda.
        10. Saves the best model.
        11. Checks the accuracy of the CBM.
        12. Calculates and prints the global explanation quality (X-factuality@10).
    """

    # get save path
    os.makedirs("saved_bb_features", exist_ok=True)
    os.makedirs("saved_clip_features", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("saved_cx", exist_ok=True)
    projection_path = f"checkpoints/Data[{args.data_name}]_ClassiModel[{args.black_box_model_name}]_ClipModel[{args.clip_model_name}]_Concept[{args.concept_set_source}]_Power[{args.power}].pt"
    bb_features_train_path = f"saved_bb_features/Data[{args.data_name}]_Model[{args.black_box_model_name}]_train.pt"
    bb_features_val_path = f"saved_bb_features/Data[{args.data_name}]_Model[{args.black_box_model_name}]_val.pt"
    bb_last_fcn_w_path = f"saved_bb_last_FCN/{args.black_box_model_name}_w.pt"
    bb_last_fcn_b_path = f"saved_bb_last_FCN/{args.black_box_model_name}_b.pt"
    clip_text_embeddings_path = f"saved_clip_features/Data[{args.data_name}]_text_Model[{args.clip_model_name}]_Concept[{args.concept_set_source}].pt"
    clip_image_embeddings_train_path = f"saved_clip_features/Data[{args.data_name}]_image_train_Model[{args.clip_model_name}].pt"
    clip_image_embeddings_val_path = f"saved_clip_features/Data[{args.data_name}]_image_val_Model[{args.clip_model_name}].pt"
    cx_train_path = f"saved_cx/Data[{args.data_name}]_Model[{args.clip_model_name}]_Concept[{args.concept_set_source}]_Power[{args.power}]_train.pt"
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
    cx_train, cx_val = load_cx(
        cx_train_path,
        cx_val_path,
        clip_text_embeddings_path,
        clip_image_embeddings_train_path,
        clip_image_embeddings_val_path,
        concept_bank,
        args,
    )
    # load black-box's hidden space embeddings and last FCN layer
    print("Load black-box model's hidden features")
    (
        bb_features_train,
        bb_features_val,
        proj_activation2class,
        proj_activation2class_bias,
    ) = load_bb(
        bb_features_train_path,
        bb_features_val_path,
        clip_image_embeddings_train_path,
        clip_image_embeddings_val_path,
        bb_last_fcn_w_path,
        bb_last_fcn_b_path,
        args,
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

    # calculate closed-form solution
    print("Closed-form solution: Find the best alignment each concept can achieve")
    _, pearson_corr = calculate_closed_form_solution(
        cx_train, cx_val, bb_features_train, bb_features_val
    )
    # filter concepts by pearson correlation
    ground_truth = pd.read_csv(
        "asset/ground_truth/{}.csv".format(args.concept_set_source)
    )
    c_indices = torch.argwhere(pearson_corr > args.pc_threshold).squeeze()
    concept_bank, ground_truth, cx_train, cx_val = (
        filter_concepts_by_pearson_correlation(
            c_indices, pearson_corr, concept_bank, ground_truth, cx_train, cx_val, args
        )
    )
    torch.save(
        c_indices,
        f"saved_projections/Data[{args.data_name}]_ClassiModel[{args.black_box_model_name}]_ClipModel[{args.clip_model_name}]_Concept[{args.concept_set_source}]_Power[{args.power}]_c_indices.pt",
    )

    print("---------Search for the best lambda---------")
    Lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [
        i for i in range(1, 11, 1)
    ]
    Mean_corrs = []
    Sparsities = []
    for lambd in Lambdas:
        (
            pearson_corr,
            sparsity,
            _,
        ) = train(
            args,
            lambd,
            cx_train,
            cx_val,
            bb_features_train,
            bb_features_val,
            proj_activation2class,
            concept_bank,
            classes,
        )
        Mean_corrs.append(pearson_corr.mean().item())
        Sparsities.append(sparsity)

    best_lambda = present_lambda_search(args, Lambdas, Mean_corrs, Sparsities)
    # train the model with the best lambda
    print(f"\nTrain with Best lambda: {best_lambda}")
    (
        pearson_corr,
        sparsity,
        best_model,
    ) = train(
        args,
        best_lambda,
        cx_train,
        cx_val,
        bb_features_train,
        bb_features_val,
        proj_activation2class,
        concept_bank,
        classes,
    )
    print("Concept alignment: {:.4f}".format(pearson_corr.mean().item()))
    print("Concept sparsity (ratio of zero weights): {:.4f}".format(sparsity))
    # save best model
    torch.save(best_model, projection_path)
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
    train_cbm_zero(args)
