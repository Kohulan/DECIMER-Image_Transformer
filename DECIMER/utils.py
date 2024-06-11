import os
import re
import shutil

import DECIMER.config as config

pattern = "R([0-9]*)|X([0-9]*)|Y([0-9]*)|Z([0-9]*)"
add_space_re = "^(\W+)|(\W+)$"


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


def ensure_models(default_path: str, model_urls: dict) -> dict:
    """Function to ensure models are present locally.

    Convenient function to ensure model downloads before usage

    Args:
        default_path (str): Default path for model data
        model_urls (dict): Dictionary containing model names as keys and their corresponding URLs as values

    Returns:
        dict: A dictionary containing model names as keys and their local paths as values
    """
    model_paths = {}
    # Store st_size of each model
    model_sizes = {
        "DECIMER": 28080309,
        "DECIMER_HandDrawn": 28080328,
    }
    for model_name, model_url in model_urls.items():
        model_path = os.path.join(default_path, f"{model_name}_model")
        if os.path.exists(model_path) and os.stat(
            os.path.join(model_path, "saved_model.pb")
        ).st_size != model_sizes.get(model_name):
            print(f"Working with model {model_name}")
            shutil.rmtree(model_path)
            config.download_trained_weights(model_url, default_path)
        elif not os.path.exists(model_path):
            config.download_trained_weights(model_url, default_path)

        # Store the model path
        model_paths[model_name] = model_path
    return model_paths
