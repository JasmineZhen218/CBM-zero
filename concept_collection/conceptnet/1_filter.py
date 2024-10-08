
from tqdm import tqdm
import random
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Filter concepts")
parser.add_argument("--data_name", type=str, default="cifar10", help="data name")
parser.add_argument("--KEEP_PER_CLASS", type=int, default=10, help="keep # concepts per class")
parser.add_argument("--CLASS_SIM_CUTOFF", type=float, default=0.8, help="class similarity cutoff")
parser.add_argument("--CONCEPT_SIM_CUTOFF", type=float, default=0.8, help="concept similarity cutoff")


def filter_too_similar_to_cls(
    concepts, classes, sim_cutoff, device="cuda", print_prob=0
):
    # first check simple text matches
    concepts = list(concepts)
    concepts = sorted(concepts)
    mpnet_model = SentenceTransformer("all-mpnet-base-v2")
    class_features_m = mpnet_model.encode(classes)
    concept_features_m = mpnet_model.encode(concepts)
    dot_prods_m = class_features_m @ concept_features_m.T
    to_delete = []
    for i in range(len(classes)):
        for j in range(len(concepts)):
            if '\' ' in concepts[j]:
                if j not in to_delete:
                    to_delete.append(j)
                    print(
                        "Class:{} - Concept:{} - Deleting {}".format(
                            classes[i], concepts[j], concepts[j]
                        )
                    )
                    continue
            prod = dot_prods_m[i, j]
            if prod >= sim_cutoff:
                if j not in to_delete:
                    to_delete.append(j)
                    print(
                        "Class:{} - Concept:{}, sim:{:.3f} - Deleting {}".format(
                            classes[i], concepts[j], dot_prods_m[i, j], concepts[j]
                        )
                    )

    to_delete = sorted(to_delete)[::-1]
    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts


def filter_too_similar(concepts, sim_cutoff, device="cuda", print_prob=0):
    mpnet_model = SentenceTransformer("all-mpnet-base-v2")
    concept_features = mpnet_model.encode(concepts)
    dot_prods_m = concept_features @ concept_features.T
    to_delete = []
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            prod = dot_prods_m[i, j]
            if prod >= sim_cutoff and i != j:
                if i not in to_delete and j not in to_delete:
                    to_print = random.random() < print_prob
                    # Deletes the concept with lower average similarity to other concepts - idea is to keep more general concepts
                    if np.sum(dot_prods_m[i]) < np.sum(dot_prods_m[j]):
                        to_delete.append(i)
                        print(
                            "{} - {} , sim:{:.4f} - Deleting {}".format(
                                concepts[i],
                                concepts[j],
                                dot_prods_m[i, j],
                                concepts[i],
                            )
                        )
                    else:
                        to_delete.append(j)
                        print(
                            "{} - {} , sim:{:.4f} - Deleting {}".format(
                                concepts[i],
                                concepts[j],
                                dot_prods_m[i, j],
                                concepts[j],
                            )
                        )

    to_delete = sorted(to_delete)[::-1]
    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts
def check_relations(
    concept_dict,
    classes,
    concepts_list,
    relations=[
        "ObstructedBy",
        "Antonym",
        'DistinctFrom',
        "ExternalURL"

    ],
):
    df = pd.DataFrame({})
    index = 0
    for cls in tqdm(classes):
        print(cls)
        obj = concept_dict[cls]
        for dicti in obj["edges"]:
            rel = dicti["rel"]["label"]
            weight = dicti["weight"]
            if (
                rel not in relations
                and dicti["start"]["language"] == "en"
                and dicti["end"]["language"] == "en"
                and weight >= 1
            ):
                if cls in dicti["start"]["label"] and cls not in dicti["end"]["label"]:
                    concept = (
                        dicti["end"]["label"]
                        .lower()
                        .replace("a ", "")
                        .replace("an ", "")
                        .replace("the ", "")
                        .replace("The ", "")
                        .replace("A ", "")
                    )
                    if concept in concepts_list:
                        print(
                            "\t'{}' {} '{}'| weight = {:.2f}".format(
                                cls, rel, concept, weight
                            )
                        )
                        df.loc[index, "class"] = cls
                        df.loc[index, "concept"] = concept
                        df.loc[index, "relation"] = rel
                        df.loc[index, "relation_direction"] = "class -> concept"
                        df.loc[index, "weight"] = weight
                        index += 1
                elif (
                    cls in dicti["end"]["label"] and cls not in dicti["start"]["label"]
                ):
                    concept = (
                        dicti["start"]["label"]
                        .lower()
                        .replace("a ", "")
                        .replace("an ", "")
                        .replace("the ", "")
                        .replace("The ", "")
                        .replace("A ", "")
                    )
                    if concept in concepts_list:
                        print(
                            "\t'{}' {} '{}| weight = {:.2f}'".format(
                                concept, rel, cls, weight
                            )
                        )
                        df.loc[index, "class"] = cls
                        df.loc[index, "concept"] = concept
                        df.loc[index, "relation"] = rel
                        df.loc[index, "relation_direction"] = "concept -> class"
                        df.loc[index, "weight"] = weight
                        index += 1
                else:
                    continue
    return df

def filter_concepts(args):
    data_name = args.data_name
    KEEP_PER_CLASS = args.KEEP_PER_CLASS
    CLASS_SIM_CUTOFF = args.CLASS_SIM_CUTOFF
    CONCEPT_SIM_CUTOFF = args.CONCEPT_SIM_CUTOFF
    print("Filtering concepts for {}".format(data_name))
    df = pd.read_csv("concept_collection/results/{}_conceptnet_raw.csv".format(data_name))
    classes = df["class"].unique()
    concepts = df["concept"].unique()
    print("intial concepts: {}".format(len(concepts)))
    concepts = [concept for concept in concepts if len(concept) >= 3 and len(concept) <= 10]
    mask = df["concept"].isin(concepts)
    df = df[mask]

    concepts = []
    for cls in classes:
        df_ = df[df["class"] == cls]
        concepts_ = (
            df_.groupby("concept")
            .sum()
            .sort_values("weight", ascending=False)
            .index[:KEEP_PER_CLASS]
            .values
        )
        concepts.extend(concepts_)
    concepts = list(set(concepts))
    df = df[df["concept"].isin(concepts)]
    concepts = filter_too_similar_to_cls(concepts, classes, sim_cutoff=CLASS_SIM_CUTOFF)
    concepts = filter_too_similar(concepts, sim_cutoff=CONCEPT_SIM_CUTOFF, device="cuda")
    mask = df["concept"].isin(concepts)
    df = df[mask]
    print("final concepts: {}".format(len(concepts)))

    df.to_csv("concept_collection/results/{}_conceptnet_ground_truth.csv".format(data_name,KEEP_PER_CLASS ), index=False)
    concept_dict = {}
    for cls in classes:
        concept_dict[cls] = list(set(df[df["class"] == cls]["concept"].values.tolist()))

    with open("concept_collection/results/{}_conceptnet.txt".format(data_name, KEEP_PER_CLASS), "w") as f:
        for concept in concepts:
            f.write(concept + "\n")


if __name__ == "__main__":
    args = parser.parse_args()
    filter_concepts(args)
