import os
import re
import shutil

import pystow
import zipfile
from pathlib import Path

pattern = "R([0-9]*)|X([0-9]*)|Y([0-9]*)|Z([0-9]*)"
add_space_re = r"^(\W+)|(\W+)$"


def split_and_modify_atoms(SMILES):
    splitted_SMILES = list(SMILES)
    modified_SMILES = re.sub(r"\s+(?=[a-z])", "", " ".join(map(str, splitted_SMILES)))
    return modified_SMILES


def replacer(match):
    part = match.group(0)
    text = (
        part.replace("1", "!")
        .replace("2", "$")
        .replace("3", "^")
        .replace("4", "<")
        .replace("5", ">")
        .replace("6", "?")
        .replace("7", "£")
        .replace("8", "¢")
        .replace("9", "€")
        .replace("0", "§")
    )
    return text


def add_space(match_):
    if match_.group(1) is not None:
        return "{} ".format(match_.group(1))
    else:
        return " {}".format(match_.group(2))


def encoder(SMILES):
    replaced_SMILES = re.sub(pattern, replacer, SMILES)
    splitted_SMILES = split_and_modify_atoms(replaced_SMILES)
    modified_SMILES = " ".join(
        [re.sub(add_space_re, add_space, word) for word in splitted_SMILES.split()]
    )
    return modified_SMILES


def decoder(predictions):
    modified = (
        predictions.replace("!", "1")
        .replace("$", "2")
        .replace("^", "3")
        .replace("<", "4")
        .replace(">", "5")
        .replace("?", "6")
        .replace("£", "7")
        .replace("¢", "8")
        .replace("€", "9")
        .replace("§", "0")
    )
    return modified


# Downloads the model and unzips the file downloaded, if the model is not present on the working directory.
def download_trained_weights(model_url: str, model_path: str, verbose=1):
    """This function downloads the trained models and tokenizers to a default
    location. After downloading the zipped file the function unzips the file
    automatically. If the model exists on the default location this function
    will not work.

    Args:
        model_url (str): trained model url for downloading.
        model_path (str): model default path to download.

    Returns:
        path (str): downloaded model.
    """
    # Download trained models
    if verbose > 0:
        print("Downloading trained model to " + str(model_path))
    model_path = pystow.ensure("DECIMER-V2", url=model_url)
    if verbose > 0:
        print(model_path)
        print("... done downloading trained model!")

    with zipfile.ZipFile(model_path.as_posix(), "r") as zip_ref:
        zip_ref.extractall(model_path.parent.as_posix())

    # Delete zipfile after downloading
    if Path(model_path).exists():
        Path(model_path).unlink()


def ensure_models(default_path: str, model_urls: dict) -> dict:
    """Function to ensure models are present locally.

    Convenient function to ensure model downloads before usage.
    Models are re-downloaded if the URL changes (e.g., new Zenodo record).

    Args:
        default_path (str): Default path for model data
        model_urls (dict): Dictionary containing model names as keys and their corresponding URLs as values

    Returns:
        dict: A dictionary containing model names as keys and their local paths as values
    """
    model_paths = {}

    for model_name, model_url in model_urls.items():
        model_path = os.path.join(default_path, f"{model_name}_model")
        saved_model_file = os.path.join(model_path, "saved_model.pb")
        version_file = os.path.join(model_path, ".model_url")

        # Check if model needs to be downloaded:
        # 1. saved_model.pb doesn't exist, OR
        # 2. The URL has changed (model was updated)
        needs_download = not os.path.exists(saved_model_file)

        if not needs_download and os.path.exists(version_file):
            with open(version_file, "r") as f:
                cached_url = f.read().strip()
            if cached_url != model_url:
                needs_download = True
                print(f"Model {model_name} has been updated, re-downloading...")

        if needs_download:
            # Clean up incomplete/corrupted model directory if it exists
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            download_trained_weights(model_url, default_path)

            # Store the URL used for this download
            os.makedirs(model_path, exist_ok=True)
            with open(version_file, "w") as f:
                f.write(model_url)

        model_paths[model_name] = model_path

    return model_paths
