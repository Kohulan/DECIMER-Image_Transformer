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

model_urls = {
    "DECIMER": "https://zenodo.org/record/8300489/files/models.zip",
    "DECIMER_HandDrawn": "https://zenodo.org/records/10781330/files/DECIMER_HandDrawn_model.zip",
}


def get_models(model_urls: dict):
    """Download and load models from the provided URLs.

    This function downloads models from the provided URLs to a default location,
    then loads tokenizers and TensorFlow saved models.

    Args:
        model_urls (dict): A dictionary containing model names as keys and their corresponding URLs as values.

    Returns:
        tuple: A tuple containing loaded tokenizer and TensorFlow saved models.
            - tokenizer (object): Tokenizer for DECIMER model.
            - DECIMER_V2 (tf.saved_model): TensorFlow saved model for DECIMER.
            - DECIMER_Hand_drawn (tf.saved_model): TensorFlow saved model for DECIMER HandDrawn.
    """
    # Download models to a default location
    model_paths = utils.ensure_models(default_path=default_path, model_urls=model_urls)

    # Load tokenizers
    tokenizer_path = os.path.join(
        model_paths["DECIMER"], "assets", "tokenizer_SMILES.pkl"
    )
    tokenizer = pickle.load(open(tokenizer_path, "rb"))

    # Load DECIMER models
    DECIMER_V2 = tf.saved_model.load(model_paths["DECIMER"])
    DECIMER_Hand_drawn = tf.saved_model.load(model_paths["DECIMER_HandDrawn"])

    return tokenizer, DECIMER_V2, DECIMER_Hand_drawn


tokenizer, DECIMER_V2, DECIMER_Hand_drawn = get_models(model_urls)


def detokenize_output(predicted_array: int) -> str:
    """This function takes the predicted tokens from the DECIMER model and
    returns the decoded SMILES string.

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
    """This function takes the predicted array of tokens as well as the
    confidence values returned by the Transformer Decoder and returns a list of
    tuples that contain each token of the predicted SMILES string and the
    confidence value.

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

    return decoded_prediction_with_confidence


def predict_SMILES(
    image_path: str, confidence: bool = False, hand_drawn: bool = False
) -> str:
    """Predicts SMILES representation of a molecule depicted in the given image.

    Args:
        image_path (str): Path of chemical structure depiction image
        confidence (bool): Flag to indicate whether to return confidence values along with SMILES prediction
        hand_drawn (bool): Flag to indicate whether the molecule in the image is hand-drawn

    Returns:
        str: SMILES representation of the molecule in the input image, optionally with confidence values
    """
    chemical_structure = config.decode_image(image_path)

    model = DECIMER_Hand_drawn if hand_drawn else DECIMER_V2
    predicted_tokens, confidence_values = model(tf.constant(chemical_structure))

    predicted_SMILES = utils.decoder(detokenize_output(predicted_tokens))

    if confidence:
        predicted_SMILES_with_confidence = detokenize_output_add_confidence(
            predicted_tokens, confidence_values
        )
        return predicted_SMILES, predicted_SMILES_with_confidence

    return predicted_SMILES


def main():
    """This function take the path of the image as user input and returns the
    predicted SMILES as output in CLI.

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


if __name__ == "__main__":
    main()
