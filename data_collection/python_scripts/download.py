import os
import pandas as pd
import PIL.Image
import tqdm
import time

from argparse import ArgumentParser, Namespace
from pathlib import Path
from pyinaturalist import (
    Observation,
    get_observations,
)
from rich import print
from taxons import TAXON_DICT


def download(args):
    """
    Download images and their metadata according to given arguments. As a result, for each class one folder of images
    is created and `labels.csv`, which contains the metadata (including labels) for each image.

    Parameters
    ----------
    args: Namespace
        Parsed arguments.
    """
    data_path = Path(args.data_path)
    os.makedirs(data_path, exist_ok=True)

    t = time.time()
    for taxon_name, taxon_id in TAXON_DICT.items():
        print(f"Animal: {taxon_name}")
        taxon_dir = data_path / taxon_name
        os.makedirs(taxon_dir, exist_ok=True)
        for page in range(1, args.n_pages + 1):
            try:
                labels = pd.read_csv(data_path / "labels.csv").to_dict(orient="list")
            except FileNotFoundError:
                labels = {}
            print(f"\tPage: {page}")
            try:
                time.sleep(args.request_time_interval)
                response = get_observations(
                    taxon_id=taxon_id,
                    has=["photo"],
                    license=args.license,
                    photo_license=args.license,
                    quality_grade=args.quality_grade,
                    per_page=args.per_page,
                    page=page,
                    term_id=args.term_id,
                    term_value_id=args.term_value_id,
                    order_by=args.order_by,
                    order=args.order,
                )
            except Exception as e:
                print("Response failure. Continue with next page.")
                print(e)
                continue
            observations = Observation.from_json_list(response)
            for obs in tqdm.tqdm(observations):
                if obs.id in labels.get("observation_id", []):
                    print("Observation already loaded. Continue with next observation.")
                    continue
                try:
                    time.sleep(args.request_time_interval)
                    raw = obs.photos[0].open(size="large")
                    img = PIL.Image.open(raw).convert("RGB")
                    img.save(taxon_dir / f"{obs.id}.jpeg", "jpeg")
                    labels.setdefault("observation_id", []).append(obs.id)
                    labels.setdefault("taxon_id", []).append(taxon_id)
                    labels.setdefault("taxon_name", []).append(taxon_name)
                    labels.setdefault("exact_taxon_id", []).append(obs.taxon.id)
                    labels.setdefault("exact_taxon_name", []).append(obs.taxon.full_name)
                    labels.setdefault("place_guess", []).append(obs.place_guess)
                    labels.setdefault("location_x", []).append(
                        obs.location[0] if isinstance(obs.location, tuple) else "None"
                    )
                    labels.setdefault("location_y", []).append(
                        obs.location[1] if isinstance(obs.location, tuple) else "None"
                    )
                    labels.setdefault("captive", []).append(obs.captive)
                    labels.setdefault("user_id", []).append(obs.user.id)
                    labels.setdefault("license_code", []).append(obs.license_code)
                    labels.setdefault("uri", []).append(obs.uri)
                    labels.setdefault("photo_large_url", []).append(obs.photos[0].large_url)
                    labels.setdefault("photo_id", []).append(obs.photos[0].id)
                    labels.setdefault("photo_attribution", []).append(obs.photos[0].attribution)
                    labels.setdefault("photo_license_code", []).append(obs.photos[0].license_code)
                    labels.setdefault("species_guess", []).append(obs.species_guess)
                    labels.setdefault("quality_grade", []).append(obs.quality_grade)
                    labels.setdefault("identifications_count", []).append(obs.identifications_count)
                    labels.setdefault("identifications_most_agree", []).append(obs.identifications_most_agree)
                    labels.setdefault("identifications_most_disagree", []).append(obs.identifications_most_disagree)
                    labels.setdefault("identifications_some_agree", []).append(obs.identifications_some_agree)
                    labels.setdefault("num_identification_agreements", []).append(obs.num_identification_agreements)
                    labels.setdefault("num_identification_disagreements", []).append(
                        obs.num_identification_disagreements
                    )
                except Exception as e:
                    print("\nImage failure. Continue with next image.")
                    print(e)
                    continue
            labels_df = pd.DataFrame(labels)
            labels_df.to_csv(data_path / "labels.csv", index=False)
    print(f"Finished after: {(time.time()-t)/60:.2f} min")


if __name__ == "__main__":
    # Define arguments for scraping.
    parser = ArgumentParser()
    parser.add_argument(
        "--request_time_interval",
        type=int,
        default=10,
        help="Defines the time intervals between the requests of data.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./",
        help="Defines the absolute path to the directory, where the images and metadata are saved.",
    )
    parser.add_argument(
        "--n_pages",
        type=int,
        default=10,
        help="Defines the number of pages to be searched for images.",
    )
    parser.add_argument(
        "--per_page",
        type=int,
        default=200,
        help="Defines the number of images per page to be downloaded, where 200 is the maximum.",
    )
    parser.add_argument(
        "--license",
        type=list,
        default=["CC0", "CC-BY", "CC-BY-NC"],
        help="Defines the list of requested license codes.",
    )
    parser.add_argument(
        "--quality_grade",
        type=str,
        default="research",
        help="Defines the quality grade of the annotations assigned to the respective image.",
    )
    parser.add_argument(
        "--term_id",
        type=int,
        default=None,
        help="Define the type of annotation for defining the `term_value_id`.",
    )
    parser.add_argument(
        "--term_value_id",
        type=int,
        default=None,
        help="Defines constrains regarding the values of the selected annotation type (cf. `term_id`).",
    )
    parser.add_argument(
        "--order_by",
        type=str,
        default="created_at",
        help="Defines the variable for ordering the images and metadata.",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="desc",
        help="Defines whether the order is descending ('desc') or ascending (`asc`).",
    )

    # Scrape images and their metadata.
    download(parser.parse_args())
