import logging
import os
import pickle
import sys
import threading
from typing import List
from typing import Tuple

import numpy as np
import pystow
import tensorflow as tf

import DECIMER.pre_process as pre_process
import DECIMER.utils as utils

# Silence tensorflow model loading warnings.
logging.getLogger("absl").setLevel("ERROR")

# Silence tensorflow errors - not recommended if your model is not working properly.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ============================================================================
# Keras 2 to Keras 3 Compatibility Layer
# ============================================================================
class KerasCompatibilityUnpickler(pickle.Unpickler):
    """Custom unpickler to handle Keras 2.x tokenizers in Keras 3 environment.

    This handles the module path changes between Keras 2 and Keras 3:
    - keras.preprocessing.text -> tensorflow.keras.preprocessing.text
    """

    def find_class(self, module, name):
        # Redirect old Keras 2.x imports to TensorFlow Keras
        if module.startswith("keras."):
            module = module.replace("keras.", "tensorflow.keras.")
        return super().find_class(module, name)


def load_tokenizer(tokenizer_path: str) -> object:
    """Load tokenizer with Keras 2/3 compatibility.

    Args:
        tokenizer_path (str): Path to the pickled tokenizer file

    Returns:
        tokenizer: Loaded tokenizer object
    """
    try:
        # Try standard pickle load first (for Keras 3 tokenizers)
        with open(tokenizer_path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        if "keras.preprocessing" in str(e):
            # Use compatibility unpickler for Keras 2.x tokenizers
            with open(tokenizer_path, "rb") as f:
                return KerasCompatibilityUnpickler(f).load()
        else:
            raise


# Set the absolute path
HERE = os.path.dirname(os.path.abspath(__file__))

# Set model to run on default GPU and allow memory to grow as much as needed.
# This allows us to run multiple instances of inference in the same GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# The models are NOT downloaded or loaded when this module is imported. They are
# downloaded (if missing) and loaded the first time they are needed, either by the
# first predict_SMILES() call or by an explicit load_models(), and cached afterwards.
# They live in $PYSTOW_HOME/DECIMER-V2/ (PYSTOW_HOME defaults to ~/.data), read when
# the models are first needed. A directory that already holds them is used as it is,
# without any network access.
model_urls = {
    "DECIMER": "https://zenodo.org/record/8300489/files/models.zip",
    "DECIMER_HandDrawn": "https://zenodo.org/records/10781330/files/DECIMER_HandDrawn_model.zip",
}


def _get_default_path():
    """Directory that holds (or will hold) the downloaded models.

    Looked up when the models are first needed, so that PYSTOW_HOME can still be set
    after DECIMER was imported. Nothing is created here.
    """
    return pystow.join("DECIMER-V2", ensure_exists=False)


def get_models(model_urls: dict):
    """Download and load models from the provided URLs.

    This function downloads models from the provided URLs to a default location,
    then loads tokenizers and TensorFlow saved models. It loads everything at once
    and does not touch the cached models of predict_SMILES(); most callers want
    load_models() instead.

    Args:
        model_urls (dict): A dictionary containing model names as keys and their corresponding URLs as values.

    Returns:
        tuple: A tuple containing loaded tokenizer and TensorFlow saved models.
            - tokenizer (object): Tokenizer for DECIMER model.
            - DECIMER_V2 (tf.saved_model): TensorFlow saved model for DECIMER.
            - DECIMER_Hand_drawn (tf.saved_model): TensorFlow saved model for DECIMER HandDrawn.
    """
    # Download models to a default location
    model_paths = utils.ensure_models(
        default_path=_get_default_path(), model_urls=model_urls
    )

    # Load tokenizers
    tokenizer_path = os.path.join(
        model_paths["DECIMER"], "assets", "tokenizer_SMILES.pkl"
    )
    tokenizer = load_tokenizer(tokenizer_path)

    # Load DECIMER models
    DECIMER_V2 = tf.saved_model.load(model_paths["DECIMER"])
    DECIMER_Hand_drawn = tf.saved_model.load(model_paths["DECIMER_HandDrawn"])

    return tokenizer, DECIMER_V2, DECIMER_Hand_drawn


def _load_tokenizer():
    """Ensure the printed-structure model files are present and load its tokenizer.

    The tokenizer of the printed-structure model decodes the output of both models.
    """
    model_paths = utils.ensure_models(
        default_path=_get_default_path(),
        model_urls={"DECIMER": model_urls["DECIMER"]},
    )
    return load_tokenizer(
        os.path.join(model_paths["DECIMER"], "assets", "tokenizer_SMILES.pkl")
    )


def _load_saved_model(model_name: str):
    """Ensure the files of one model are present, then load its SavedModel."""
    model_paths = utils.ensure_models(
        default_path=_get_default_path(),
        model_urls={model_name: model_urls[model_name]},
    )
    return tf.saved_model.load(model_paths[model_name])


# Module-level names that are filled on first use. Once loaded, an object is stored
# in this module's globals under the same name that decimer 2.8.0 and older bound at
# import, so `from DECIMER.decimer import tokenizer, DECIMER_V2` keeps working.
_lazy_loaders = {
    "tokenizer": _load_tokenizer,
    "DECIMER_V2": lambda: _load_saved_model("DECIMER"),
    "DECIMER_Hand_drawn": lambda: _load_saved_model("DECIMER_HandDrawn"),
}
_load_lock = threading.RLock()


def _lazy(name: str):
    """Return tokenizer, DECIMER_V2 or DECIMER_Hand_drawn, loading it on first use.

    Downloads the files first if they are missing. The lock makes concurrent first
    calls download and load once. A failed download or load is not cached, so the
    next call tries again.
    """
    try:
        return globals()[name]
    except KeyError:
        pass
    with _load_lock:
        if name not in globals():
            globals()[name] = _lazy_loaders[name]()
        return globals()[name]


def __getattr__(name: str):
    """Load tokenizer, DECIMER_V2 and DECIMER_Hand_drawn on first attribute access."""
    if name == "default_path":
        return _get_default_path()
    if name in _lazy_loaders:
        return _lazy(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_models(printed: bool = True, hand_drawn: bool = False) -> None:
    """Download (if missing) and load the models now, instead of at the first prediction.

    Call this once at start-up if the first predict_SMILES() call should not pay for
    the download and the loading, for example in a server or when building an image.
    Models that are already loaded are not loaded again. The tokenizer, which both
    models need, is always loaded.

    Args:
        printed (bool): Load the model for printed chemical structures (the default
            model of predict_SMILES).
        hand_drawn (bool): Also load the hand-drawn model. load_models(hand_drawn=True)
            loads everything that decimer 2.8.0 and older loaded on import.
    """
    with _load_lock:
        _lazy("tokenizer")
        if printed:
            _lazy("DECIMER_V2")
        if hand_drawn:
            _lazy("DECIMER_Hand_drawn")


def detokenize_output(predicted_array: int) -> str:
    """This function takes the predicted tokens from the DECIMER model and
    returns the decoded SMILES string.

    Args:
        predicted_array (int): Predicted tokens from DECIMER

    Returns:
        (str): SMILES representation of the molecule
    """
    index_word = _lazy("tokenizer").index_word
    outputs = [index_word[i] for i in predicted_array[0].numpy()]
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
    index_word = _lazy("tokenizer").index_word
    prediction_with_confidence = [
        (
            index_word[predicted_array[0].numpy()[i]],
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
    image_input: [str, np.ndarray], confidence: bool = False, hand_drawn: bool = False
) -> str:
    """Predicts SMILES representation of a molecule depicted in the given image.

    Args:
        image_input (str or np.ndarray): Path of chemical structure depiction image or a numpy array representing the image.
        confidence (bool): Flag to indicate whether to return confidence values along with SMILES prediction.
        hand_drawn (bool): Flag to indicate whether the molecule in the image is hand-drawn.

    Returns:
        str: SMILES representation of the molecule in the input image, optionally with confidence values.
    """
    chemical_structure = pre_process.decode_image(image_input)

    # The first call downloads the models that are missing and loads the ones it needs
    # (the tokenizer, and the printed or the hand-drawn model); later calls reuse them.
    _lazy("tokenizer")
    model = _lazy("DECIMER_Hand_drawn" if hand_drawn else "DECIMER_V2")
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
