import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def present_lambda_search(args, Lambdas, Mean_corrs, Sparsities): 
    """
    Visualizes the lambda search process and determines the best lambda value.

    This function plots the relationship between regularization strength (Lambdas),
    concept sparsity (Sparsities), and concept alignment (Mean_corrs). It highlights
    the "reasonable zone" where both sparsity and alignment are within 90% of their
    maximum values. The function saves the plot and returns the best lambda value.

    Parameters:
        args (Namespace): A namespace object containing the following attributes:
            - clip_model_name (str): The name of the CLIP model.
            - power (int): The power parameter.
            - data_name (str): The name of the dataset.
        Lambdas (list of float): A list of lambda values (regularization strengths).
        Mean_corrs (list of float): A list of mean correlation values corresponding to the lambdas.
        Sparsities (list of float): A list of sparsity values corresponding to the lambdas.

    Returns:
        float: The best lambda value determined from the reasonable zone.
    """
    max_corr = max(Mean_corrs)
    max_sparsity = max(Sparsities)
    indices = [i for i in range(len(Mean_corrs)) if Mean_corrs[i] >= 0.9*max_corr and Sparsities[i] >= 0.9*max_sparsity]
    # save the model
    f, ax = plt.subplots()
    ax.plot(Lambdas, Sparsities, marker='o', label="Concept Sparsity")
    ax.plot(Lambdas, Mean_corrs, marker='o', label="Concept Alignement")
    ax.set_xlabel("Regularization strength")
    ax.legend()
    ax.grid()
    # draw resonable zone
    if len(indices) > 0:
        ax.axvline(x=Lambdas[indices[0]], color='r', linestyle='--')
        ax.axvline(x=Lambdas[indices[-1]], color='r', linestyle='--')
        best_lambda = Lambdas[indices[0]]
            # add text "reasonable zone"
        ax.text(Lambdas[indices[0]]+1, 0.5, "Reasonable zone")
            # shade the zone
        ax.fill_between(Lambdas[indices[0]:indices[-1]+1], 0, 1, color='red', alpha=0.1)
    else:
        best_lambda = 2
    os.makedirs("results/lambda", exist_ok=True)
    f.savefig(f"results/lambda/{args.clip_model_name}_{args.power}_{args.data_name}.png")
    return best_lambda


def draw_global_concept_importance(
    data_name, classes, class_name, concept_bank, feature_importance, top_k = 10
):
    """
    Draws a bar plot representing the global concept importance for a specific class.
    Parameters:
        data_name (str): The name of the dataset.
        classes (list): A list of class names.
        class_name (str): The name of the class for which the concept importance is to be drawn.
        concept_bank (list): A list of all possible concepts.
        feature_importance (numpy.arrays): feature importances for the designated class. shape: (num_concepts,)
        top_k (int, optional): The number of top concepts to display. Default is 10.
    Returns:
        None: The function saves the plot as a PNG file in the specified directory.
    """
    i_class = classes.index(class_name)
    f, ax = plt.subplots(figsize=(5, 3))
    mean = feature_importance[i_class]
    idx = np.argsort(mean)[-top_k :][::-1]
    x = np.arange(top_k)
    y = mean[idx]
    xlabel = np.array(concept_bank)[idx]
    ax.bar(x, y, color=sns.color_palette("Set2")[2], alpha=0.8)
    ax.set_title(
        f"Global concept importance for <{class_name}>", fontsize=12, fontweight="bold"
    )
    ax.set(xticks=x, xticklabels=xlabel)
    ax.set_xticklabels(xlabel, rotation=90)
    ax.set(ylabel="Concept Weight")
    os.makedirs(f"explanations/global/{data_name}", exist_ok=True)
    f.savefig(
        f"explanations/global/{data_name}/{class_name}.png",
        bbox_inches="tight",
    )


def draw_global_concept_importance_contrast(
    data_name,
    classes,
    class_name_i,
    class_name_j,
    concept_bank,
    feature_importance,
):
    """
    Draws a bar plot to visualize the global concept importance contrast between two classes.
    Parameters:
        data_name (str): The name of the dataset.
        classes (list): A list of class names.
        class_name_i (str): The name of the first class.
        class_name_j (str): The name of the second class.
        concept_bank (list): A list of concept names.
        feature_importance (numpy.array): feature importance values for all the classes. shape: (num_classes, num_concepts)
    Returns:
        None : The function saves the plot as a PNG file in the specified directory.
    """
    i_class = classes.index(class_name_i)
    j_class = classes.index(class_name_j)
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    f.subplots_adjust(hspace=0)
    diff = feature_importance[i_class] - feature_importance[j_class]
    idx = np.argsort(diff)[::-1]
    y = [diff[i] for i in idx[:5]] + [0, 0, 0] + [diff[i] for i in idx[-5:]]
    x = np.arange(len(y))
    xlabel = np.concatenate(
        [np.array(concept_bank)[idx[:5]], [" "] * 3, np.array(concept_bank)[idx[-5:]]]
    )
    ax.bar(x, y, color=sns.color_palette("Set2")[0], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabel, rotation=90)
    ax.set_title(
        "Global importance for distinguishing <{}> from <{}>".format(
            class_name_i, class_name_j
        ),
        fontsize=10,
        fontweight="bold",
    )
    ax.text(6, 0, ".......", fontsize=12, ha="center")
    os.makedirs(f"explanations/global/{data_name}", exist_ok=True)
    f.savefig(
        f"explanations/global/{data_name}/{class_name_i}_{class_name_j}.png",
        bbox_inches="tight",
    )


def draw_local_explanations(
        data_name, 
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
    ):

    """
    Visualizes local explanations for a given image by plotting the image, 
    concept contributions, and concept activations with weights.
    Parameters:
        data_name (str): Name of the dataset.
        concept_bank (list): List of concept names.
        image (ndarray): The image to be visualized.
        image_id (str): Identifier for the image.
        class_name (str): The true class name of the image.
        predicted_class (str): The predicted class name of the image.
        logit_n (float): The logit value for the predicted class.
        concept_contributions_n (ndarray): Array of concept contributions to the logit. shape: (num_concepts,)
        proj_concept2class_n (Tensor): Tensor of projected concept weights to the class. shape: (num_concepts,)
        concept_activation (ndarray): Array of concept activations. shape: (num_concepts,)
        max_display (int, optional): Maximum number of concepts to display. Default is 10.
    Returns:
        None: The function saves the plot as a PNG file in the specified directory.
    """
    concept_contributions_ratio = concept_contributions_n / logit_n * 100
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), tight_layout=True, gridspec_kw={'width_ratios': [2, 1, 1]})
    # ax1: image
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title(
        f"Label = {class_name} | Prediction = {predicted_class}",
        fontsize=12,
        fontweight="bold"
    )
    sorted_inds = np.argsort(concept_contributions_ratio)[-max_display:]
    # ax2: concept contribution
    ax2.barh(
        np.arange(len(sorted_inds)),
        concept_contributions_ratio[sorted_inds],
        color=sns.color_palette("Set2")[2],
        label="Concept Contribution",
    )
    ax2.legend()
    ax2.set_yticks(np.arange(len(sorted_inds)))
    concept_bank = np.array(concept_bank)
    yticklabels = [
        (
            '"NOT ' + concept_bank[i] + '"'
            if concept_activation[i] < 0
            else '"' + concept_bank[i] + '"'
        )
        for i in sorted_inds
    ]
    ax2.set_yticklabels(
        yticklabels,
        fontsize=12,
    )
    ax2.set_xlabel("Contribution (%)")
    ax2.set_title("Local explanation", fontsize=12, fontweight="bold")
    # ax3: concept activation and weights
    ax3.barh(
        np.arange(len(sorted_inds)),
        concept_activation[sorted_inds],
        color=sns.color_palette("Set2")[0],
        label="Concept Activation",
    )
    ax3.set_yticks(np.arange(len(sorted_inds)))
    ax3.set_title("Concept Activation and Weights", fontsize=12, fontweight="bold")
    ax3.set_yticklabels(
        ['"' + i + '"' for i in concept_bank[sorted_inds].tolist()],
        fontsize=12,
    )
    ax3.set_xlabel("Concept Activation")
    ax3_ = ax3.twiny()
    ax3_.barh(
        np.arange(len(sorted_inds)),
        proj_concept2class_n.detach().numpy()[sorted_inds],
        color=sns.color_palette("Set2")[1],
        label="Concept Weight",
    )
    ax3_.set_xlabel("Concept Weight")
    for bar in ax3_.patches:
        bar.set_height(0.4)

    ax3_.legend(loc="upper right", bbox_to_anchor=(-0.1, 1.05))
    ax3.legend(loc="upper right", bbox_to_anchor=(-0.05, 0))
    
    os.makedirs(f"explanations/local/{data_name}", exist_ok=True)
    f.savefig(
        f"explanations/local/{data_name}/{class_name}_{image_id}.png",
        bbox_inches="tight",
    )
