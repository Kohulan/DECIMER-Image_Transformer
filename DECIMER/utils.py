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


def ensure_model(
    default_path: str,
    model_url: str = "https://zenodo.org/record/8300489/files/models.zip",
):
    """Function to ensure model is present locally

    Convenient function to ensure model download before usage

    Args:
        default path (str): default path for DECIMER data
        model_url (str): trained model url for downloading
    """

    model_path = os.path.join(default_path.as_posix(), "DECIMER_model")
    print(model_path)

    if (
        os.path.exists(model_path)
        and os.stat(os.path.join(model_path, "saved_model.pb")).st_size != 28080309
    ):
        shutil.rmtree(model_path)
        config.download_trained_weights(model_url, default_path)
    elif not os.path.exists(model_path):
        config.download_trained_weights(model_url, default_path)
