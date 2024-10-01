import numpy as np
import pandas as pd
import torch
def calculate_x_factuality(
    data_name, classes, concepts_bank, ground_truth, feature_importance, top_k=10
):
    # top-weighted positive concept ratio
    Ratios = []
    for i, cls in enumerate(classes):
            if cls not in ground_truth["class"].values:
                continue
            if len(ground_truth.loc[ground_truth["class"] == cls, 'concept'].unique()) < 5:
                continue
            weights = feature_importance[i]
            top_concepts_idx = np.argsort(weights)[::-1][:top_k]
            top_concepts = np.array(concepts_bank)[top_concepts_idx]
            mask = (ground_truth["class"] == cls) * (ground_truth["concept"].isin(top_concepts))
            df_m = ground_truth[mask]
            ratios = []
            if data_name == "cub":
                ratios.append(
                    len(df_m.loc[df_m["weight"] >= 50, "concept"].unique()) / top_k
                )
                Ratios.append(ratios)
            else:
                ratios.append(len(df_m.loc[df_m["weight"] > 0, "concept"].unique()) / top_k)
                Ratios.append(ratios)
    Ratios = np.array(Ratios)
    mean = np.mean(Ratios, axis=0)
    std = np.std(Ratios, axis=0)
    print(f"top_k = {top_k} |")
    print("\t Top weighted positive concept: {:.3f} +/- {:.3f}".format(mean[0], std[0]))
    return mean[0], std[0]
