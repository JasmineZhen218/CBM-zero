import numpy as np
import pandas as pd
import torch
def calculate_x_factuality(
   data_name, classes, concepts_bank, ground_truth, feature_importance, top_k=10
):
    """
    Calculate the factuality of top-weighted positive concepts for given classes.
    Parameters:
        data_name (str): The name of the dataset.
        classes (list): A list of class names.
        concepts_bank (list): A list of concepts.
        ground_truth (pd.DataFrame): A DataFrame containing ground truth data with columns 'class', 'concept', and 'weight'.
        feature_importance (numpy.array): normalized global concept weights: (num_classes, num_concepts)
        top_k (int, optional): The number of top concepts to consider. Default is 10.
    Returns:
        tuple: A tuple containing the mean and standard deviation of the top-weighted positive concept ratios.
    """
    # calculate the ratio of top-weighted reasonable concepts
    Ratios = []
    for i, cls in enumerate(classes):
        if cls not in ground_truth["class"].values:
            # skip classes not in ground truth
            continue
        if len(ground_truth.loc[ground_truth["class"] == cls, 'concept'].unique()) < 10:
            # skip classes with less than 10 concepts
            continue
        weights = feature_importance[i]
        top_concepts_idx = np.argsort(weights)[::-1][:top_k]
        top_concepts = np.array(concepts_bank)[top_concepts_idx]
        mask = (ground_truth["class"] == cls) * (ground_truth["concept"].isin(top_concepts))
        df_m = ground_truth[mask]
        if data_name == "cub":
            Ratios.append(
                    len(df_m.loc[df_m["weight"] >= 50, "concept"].unique()) / top_k
                )
        else:
            Ratios.append(len(df_m.loc[df_m["weight"] > 0, "concept"].unique()) / top_k)
    mean = np.mean(Ratios)
    std = np.std(Ratios)
    return mean, std
