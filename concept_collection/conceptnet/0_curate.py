from tqdm import tqdm
import requests
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(description="Querying ConceptNet for classes")
parser.add_argument("--data_name", type=str, default="cifar10", help="data name")

def get_init_conceptnet(
    concept_dict,
    classes,
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
        obj = concept_dict[cls]
        for dicti in obj["edges"]:
            rel = dicti["rel"]["label"]
            weight = dicti["weight"]
            if (
                rel not in relations
                and dicti["start"]["language"] == "en"
                and dicti["end"]["language"] == "en"
                # and weight >= 1
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
                    # print(
                    #     "\t'{}' {} '{}'| weight = {:.2f}".format(
                    #         cls, rel, concept, weight
                    #     )
                    # )
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
                    # print(
                    #     "\t'{}' {} '{}| weight = {:.2f}'".format(
                    #         concept, rel, cls, weight
                    #     )
                    # )
                    df.loc[index, "class"] = cls
                    df.loc[index, "concept"] = concept
                    df.loc[index, "relation"] = rel
                    df.loc[index, "relation_direction"] = "concept -> class"
                    df.loc[index, "weight"] = weight
                    index += 1
                else:
                    continue
    return df


def curate_concepts(args):
    data_name = args.data_name
    with open("asset/class_names/{}.txt".format(data_name)) as f:
        classes = f.read().split("\n")
    f.close()
    print("Querying ConceptNet for classe names of {}".format(data_name))
    LIMIT = 20000
    concept_sets = {}
    for cls in tqdm(classes):
        response = requests.get(
            "http://api.conceptnet.io/query?node=/c/en/{}&limit={}".format(cls, LIMIT)
        )
        obj = response.json()
        concept_sets[cls] = obj

    with open("concept_collection/results/{}_conceptnet_raw.json".format(data_name), "w") as f:
        json.dump(concept_sets, f)
    f.close()

    df = get_init_conceptnet(concept_sets, classes)
    print("{} unique concepts found for {} classes".format(len(df["concept"].unique()), len(classes)))
    df.to_csv("concept_collection/results/{}_conceptnet_raw.csv".format(data_name), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    curate_concepts(args)