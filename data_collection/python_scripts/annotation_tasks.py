import json
import numpy as np
import pandas as pd
import os

from argparse import ArgumentParser, Namespace


def annotation_tasks(args):
    """
    Create batches of annotation tasks.

    Parameters
    ----------
    args : Namespace
        Parsed arguments.
    """

    # Path to be annotated file.
    csv_file = os.path.join(args.data_path, "annotation-tasks.csv")
    new_file = "annotation_tasks"
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
        is_batch = np.array([False] * len(df))
        for b in args.batches:
            new_file += f"-{b}"
            is_batch |= df["batch"].values == b
        batch_rows = df.loc[is_batch]
        new_csv_file = os.path.join(args.data_path, f"{new_file}.csv")
        if not os.path.isfile(new_csv_file):
            batch_rows.to_csv(new_csv_file, index=False)
        else:
            print(f"{new_csv_file} already exists. Delete, rename or move it to create a new one.")
        data = []
        new_json_file = os.path.join(args.data_path, f"{new_file}.json")
        for row_idx, row in batch_rows.iterrows():
            if args.use_local_files:
                photo_large_url = f"/data/local-files/?d=images/{row['observation_id']}.jpeg"
            else:
                photo_large_url = row["photo_large_url"].replace("/large.", "/medium.")
            row_dict = {
                "id": row_idx,
                "data": {
                    "observation_id": row["observation_id"],
                    "batch": int(row["batch"]),
                    "photo_large_url": photo_large_url,
                    "photo_attribution": row["photo_attribution"],
                },
            }
            data.append(row_dict)
        if not os.path.isfile(new_json_file):
            with open(new_json_file, "w") as f:
                json.dump(data, f)
        else:
            print(f"{new_json_file} already exists. Delete, rename or move it to create a new one.")
    else:
        print(f"{csv_file} does not exist. Create it via running `preprocessing.py`.")


if __name__ == "__main__":
    # Define arguments for scraping.
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./",
        help="Defines the absolute path to the directory, where the scraped dataset is stored.",
    )
    parser.add_argument(
        "--batches",
        type=json.loads,
        default="[0,1]",
        help="Defines the batches to be included as annotation tasks."
    )
    parser.add_argument(
        "--use_local_files",
        type=int,
        default=1,
        help="Integer flag whether local files are to be loaded as part of the annotation tasks.",
    )

    # Preprocess data.
    annotation_tasks(parser.parse_args())
