import torch
import argparse
import numpy as np
import pandas as pd
from path import get_save_path
from utils import set_seed, remove_duplicates
from data_utils import load_labels
from model_utils import get_bb_features, get_bb_fcn, get_clip_scores, ensure_full_rank, loss_fn
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
    default=1,
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

# args.data_name = "cifar100"
# args.concept_set_source = "cifar100_conceptnet"
# args.black_box_model_name = "clip_mlp_ViT-L_14-h256_cifar100"
# init wandb
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

# get save path
(projection_path, 
bb_features_train_path, bb_features_val_path, 
clip_text_embeddings_path, 
clip_image_embeddings_train_path, clip_image_embeddings_val_path, 
clip_similarities_train_path, clip_similarities_val_path, 
clip_similarities_train_mean_path, clip_similarities_train_std_path)= get_save_path(args)

print("Explaining the classification model: {} \non dataset: {} \nwith concept bank: {}".format(args.black_box_model_name, args.data_name, args.concept_set_source))
# load class names
with open("asset/class_names/{}.txt".format(args.data_name)) as f:
    classes = f.read().split("\n")
# load concept bank
with open("asset/concept_bank/{}.txt".format(args.concept_set_source)) as f:
    concept_bank = f.read().split("\n")
concept_bank = [c for c in concept_bank if c not in ["", " "]]
concept_bank = [i for i in concept_bank if i not in classes]
concept_bank = remove_duplicates(concept_bank)
# load clip scores
(clip_similarities_train, clip_similarities_val, 
 clip_similarities_train_mean, clip_similarities_train_std) = get_clip_scores(args.device, 
                                                                args.clip_model_name, args.data_name, args.power, concept_bank, 
                                                                clip_text_embeddings_path, clip_image_embeddings_train_path, clip_image_embeddings_val_path, 
                                                                clip_similarities_train_path, clip_similarities_val_path, 
                                                                clip_similarities_train_mean_path, clip_similarities_train_std_path)
# normalize clip scores
clip_similarities_train = (clip_similarities_train - clip_similarities_train_mean) / clip_similarities_train_std
clip_similarities_val = (clip_similarities_val - clip_similarities_train_mean) / clip_similarities_train_std

# load black-box's hidden space embeddings and last FCN layer
print("Load hidden space features of the black-box model")
bb_features_train, bb_features_val = get_bb_features(args.device, args.black_box_model_name, args.data_name, 
                    bb_features_train_path, bb_features_val_path, 
                    clip_image_embeddings_train_path, clip_image_embeddings_val_path)
print("Load last FCN layer of the classification model")
proj_activation2class, proj_activation2class_bias = get_bb_fcn(args.black_box_model_name, args.device)
# check the accuracy of the black-box model
logits_val = (
    bb_features_val @ proj_activation2class.T + proj_activation2class_bias)
predictions_val = torch.argmax(logits_val, dim=1)
_, labels_val = load_labels(args.data_name)
labels_val = torch.tensor(labels_val).to(args.device)
acc_val = torch.sum(predictions_val == labels_val).item() / len(labels_val)
print("Accuracy on validation set: {:.4f}".format(acc_val))

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
for i in range(args.n_iter):
    # sample a mini-batch
    random_indices = torch.randperm(N_train)[:10000]
    clip_similarities_train_ = clip_similarities_train[random_indices, :].to(args.device)
    bb_features_train_ = bb_features_train[random_indices, :].to(args.device)
    outs_train_ = proj_layer(bb_features_train_)  # N x M
    mse_loss_train = loss_fn(outs_train_, clip_similarities_train_)
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
            mse_loss_val = loss_fn(outs_val, clip_similarities_val)
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
print("Accuracy on validation set: {:.4f}".format(acc_val))
if args.log:
    wandb.log({"acc/val": acc_val})

feature_importance = proj_concept2class.detach().cpu().numpy() - proj_concept2class.detach().cpu().numpy().mean(
        axis=0
    )
ground_truth = pd.read_csv("asset/ground_truth/{}.csv".format(args.concept_set_source))
mean, std = calculate_x_factuality(
    args.data_name, classes, np.array(concept_bank), ground_truth, feature_importance, top_k=10
)

