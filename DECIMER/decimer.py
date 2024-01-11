import logging
import os
import pickle
import sys
from typing import List
from typing import Tuple

import pystow
import tensorflow as tf

import DECIMER.config as config
import DECIMER.utils as utils

# Silence tensorflow model loading warnings.
logging.getLogger("absl").setLevel("ERROR")

# Silence tensorflow errors - not recommended if your model is not working properly.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set the absolute path
HERE = os.path.dirname(os.path.abspath(__file__))

# Set model to run on default GPU and allow memory to grow as much as needed.
# This allows us to run multiple instances of inference in the same GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set path
default_path = pystow.join("DECIMER-V2")

# download models to a default location
utils.ensure_model(default_path=default_path)

# Load important pickle files which consists the tokenizers and the maxlength setting
tokenizer = pickle.load(
    open(
        os.path.join(
            default_path.as_posix(), "DECIMER_model", "assets", "tokenizer_SMILES.pkl"
        ),
        "rb",
    )
)


def main():
    """
    This function take the path of the image as user input
    and returns the predicted SMILES as output in CLI.

    Agrs:
        str: image_path

    Returns:
        str: predicted SMILES

    """
    if len(sys.argv) != 2:
        print("Usage: {} $image_path".format(sys.argv[0]))
    else:
        SMILES = predict_SMILES(sys.argv[1])
        print(SMILES)


def detokenize_output(predicted_array: int) -> str:
    """
    This function takes the predited tokens from the DECIMER model
    and returns the decoded SMILES string.

    Args:
        predicted_array (int): Predicted tokens from DECIMER

    Returns:
        (str): SMILES representation of the molecule
    """
    outputs = [tokenizer.index_word[i] for i in predicted_array[0].numpy()]
    prediction = (
        "".join([str(elem) for elem in outputs])
        .replace("<start>", "")
        .replace("<end>", "")
    )
    return prediction


def detokenize_output_add_confidence(
    predicted_array: tf.Tensor,
    confidence_array: tf.Tensor,
) -> List[Tuple[str, float]]:
    """
    This function takes the predicted array of tokens as well as the confidence values
    returned by the Transformer Decoder and returns a list of tuples
    that contain each token of the predicted SMILES string and the confidence
    value.

    Args:
        predicted_array (tf.Tensor): Transformer Decoder output array (predicted tokens)

    Returns:
        str: SMILES string
    """
    prediction_with_confidence = [
        (
            tokenizer.index_word[predicted_array[0].numpy()[i]],
            confidence_array[i].numpy(),
        )
        for i in range(len(confidence_array))
    ]
    # remove start and end tokens
    prediction_with_confidence_ = prediction_with_confidence[1:-1]

    decoded_prediction_with_confidence = list(
        [(utils.decoder(tok), conf) for tok, conf in prediction_with_confidence_]
    )
    decoded_prediction_with_confidence.append(prediction_with_confidence_[-1])
    return decoded_prediction_with_confidence


# Load DECIMER model_packed
DECIMER_V2 = tf.saved_model.load(default_path.as_posix() + "/DECIMER_model/")


def predict_SMILES(image_path: str) -> str:
    """
    This function takes an image path (str) and returns the SMILES
    representation of the depicted molecule (str).

    Args:
        image_path (str): Path of chemical structure depiction image

    Returns:
        (str): SMILES representation of the molecule in the input image
    """
    chemical_structure = config.decode_image(image_path)
    predicted_tokens, _ = DECIMER_V2(chemical_structure)
    predicted_SMILES = utils.decoder(detokenize_output(predicted_tokens))
    return predicted_SMILES


def predict_SMILES_with_confidence(image_path: str) -> List[Tuple[str, float]]:
    """
    This function takes an image path (str) and returns a list of tuples
    that contain each token of the predicted SMILES string and the confidence
    level from the last layer of the Transformer decoder.

    Args:
        image_path (str): Path of chemical structure depiction image

    Returns:
        (List[Tuple[str, float]]): Tuples that contain the tokens and the confidence
            values of the predicted SMILES
    """
    decodedImage = config.decode_image(image_path)
    predicted_tokens, confidence_values = DECIMER_V2(tf.constant(decodedImage))
    predicted_SMILES_with_confidence = detokenize_output_add_confidence(
        predicted_tokens, confidence_values
    )
    return predicted_SMILES_with_confidence


if __name__ == "__main__":
    main()
