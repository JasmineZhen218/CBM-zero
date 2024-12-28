import matplotlib.pyplot as plt
import os

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