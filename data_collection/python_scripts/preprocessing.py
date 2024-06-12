import json
import numpy as np
import pandas as pd
import os
import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from argparse import ArgumentParser, Namespace
from taxons import TAXON_DICT


def preprocess(args):
    """
    Preprocesses the downloaded observations (images + metadata) by filtering invalid observations and splitting the data
    into train, validation, and test data. The filtered file including the data splits is saved at
    `args.data_path/preprocessed_labels.csv`.

    Parameters
    ----------
    args : Namespace
        Parsed arguments.
    """
    # Setup.
    classes = TAXON_DICT.keys()

    # Load metadata including labels of the images.
    file = os.path.join(args.data_path, "labels.csv")
    df = pd.read_csv(file)

    # Path to preprocessed labels.
    preprocessed_file = os.path.join(args.data_path, "preprocessed_labels.csv")
    if not os.path.isfile(preprocessed_file):
        # Counts the number of images per taxon.
        taxon_counts = {}

        # Iterate over each observation (image + metadata) to perform a filtering.
        for row_idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            img_file = os.path.join(
                args.data_path,
                row["taxon_name"],
                f"{row['observation_id']}.jpeg",
            )
            not_existing = not os.path.exists(img_file)
            not_an_img = row["photo_large_url"].endswith("gif")
            too_many = taxon_counts.get(row["taxon_name"], 0) >= args.max_images_per_taxon
            not_in_classes = row["taxon_name"] not in classes
            invalid_license = row["photo_license_code"] not in args.license or row["license_code"] not in args.license
            if not_existing or not_an_img or too_many or not_in_classes or invalid_license:
                df.drop(row_idx, inplace=True)
            else:
                taxon_counts[row["taxon_name"]] = taxon_counts.get(row["taxon_name"], 0) + 1


        # Split preprocessed data into train, validation, and test sets.
        split = np.array(["train"] * len(df))
        indices = np.arange(len(df), dtype=int)
        labels = df["taxon_name"].values
        n_classes = len(np.unique(labels))
        training_indices, test_indices = train_test_split(
            indices,
            test_size=(args.no_of_test_images_per_taxon + args.no_of_validation_images_per_taxon) * n_classes,
            shuffle=True,
            random_state=42,
            stratify=labels,
        )
        validation_indices, test_indices = train_test_split(
            test_indices,
            test_size=args.no_of_test_images_per_taxon * n_classes,
            shuffle=True,
            random_state=42,
            stratify=labels[test_indices],
        )
        split[validation_indices] = "valid"
        split[test_indices] = "test"
        df["split"] = split

        # Store preprocessed data including splits.
        df.to_csv(preprocessed_file, index=False)

        # Provide information about preprocessed data.
        print("Number of images per taxon:")
        print(taxon_counts)
        print("Number of training images per taxon:")
        print(np.unique(labels[df.split.values == "train"], return_counts=True))
        print("Number of validation images per taxon:")
        print(np.unique(labels[df.split.values == "valid"], return_counts=True))
        print("Number of test images per taxon:")
        print(np.unique(labels[df.split.values == "test"], return_counts=True))
    else:
        print(f"{preprocessed_file} already exists. Delete, rename or move it to create a new one.")

    # Path to be annotated file.
    json_file = os.path.join(args.data_path, "annotation-tasks.json")
    csv_file = os.path.join(args.data_path, "annotation-tasks.csv")
    df = pd.read_csv(preprocessed_file)
    df = df.query("split == 'train'")
    indices = np.arange(len(df), dtype=int)
    batches = np.array([-1 for _ in range(len(df))], dtype=int)
    rskf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for i, (_, indices) in enumerate(rskf.split(indices, df["taxon_id"].values)):
        batches[indices] = i
    df = df[["observation_id", "photo_large_url", "photo_attribution", "taxon_name"]]
    df["batch"] = batches
    df = df.sample(frac=1, random_state=42)
    for i in range(args.no_of_batches):
        for j in classes:
            is_batch = df["batch"].values == i
            is_class = df["taxon_name"].values == j
            filtered_rows = df.loc[is_batch & is_class]
            df = pd.concat([df, filtered_rows.iloc[:1]])
    df = df.sample(frac=1, random_state=0)
    data = []
    for row_idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        row_dict = {
            "id": row_idx,
            "data": {
                "observation_id": row["observation_id"],
                "batch": int(row["batch"]),
                "photo_large_url": row["photo_large_url"].replace("/large.", "/medium."),
                "photo_attribution": row["photo_attribution"],
            },
        }
        data.append(row_dict)
    if not os.path.isfile(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        print(f"{csv_file} already exists. Delete, rename or move it to create a new one.")
    if not os.path.isfile(json_file):
        with open(json_file, "w") as f:
            json.dump(data, f)
    else:
        print(f"{json_file} already exists. Delete, rename or move it to create a new one.")


if __name__ == "__main__":
    # Define arguments for preprocessing.
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./",
        help="Defines the absolute path to the directory, where the downloaded dataset is stored.",
    )
    parser.add_argument(
        "--max_images_per_taxon",
        type=int,
        default=1050,
        help="Defines the maximum number of images per taxon, i.e., animal class.",
    )
    parser.add_argument(
        "--license",
        type=list,
        default=["CC0", "CC-BY", "CC-BY-NC"],
        help="Defines the list of requested license codes.",
    )
    parser.add_argument(
        "--no_of_test_images_per_taxon",
        type=int,
        default=300,
        help="Defines the number of test images per taxon.",
    )
    parser.add_argument(
        "--no_of_validation_images_per_taxon",
        type=int,
        default=50,
        help="Defines the number of validation images per taxon.",
    )
    parser.add_argument(
        "--no_of_batches",
        type=int,
        default=10,
        help="Defines the number of batches of annotation tasks.",
    )
    parser.add_argument(
        "--add_list",
        type=int,
        default=0,
        help="Add list to annotate ranks of labels.",
    )

    # Preprocess data.
    preprocess(parser.parse_args())
