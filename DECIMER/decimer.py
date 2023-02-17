import os
import sys
import logging
import pickle
import pystow
import tensorflow as tf
import DECIMER.config as config
import DECIMER.utils as utils

# Silence tensorflow model loading warnings.
logging.getLogger("absl").setLevel("ERROR")

# Silence tensorflow errors. optional not recommened if your model is not working properly.
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

# model download location
model_url = "https://zenodo.org/record/7624994/files/models.zip"
model_path = str(default_path) + "/DECIMER_model/"

# download models to a default location
if not os.path.exists(model_path):
    config.download_trained_weights(model_url, default_path)


# Load important pickle files which consists the tokenizers and the maxlength setting

tokenizer = pickle.load(
    open(
        default_path.as_posix() + "/DECIMER_model/assets/tokenizer_SMILES.pkl",
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
    predicted_tokens = DECIMER_V2(chemical_structure)
    predicted_SMILES = utils.decoder(detokenize_output(predicted_tokens))

    return predicted_SMILES


if __name__ == "__main__":
    main()
