# -*- coding: UTF-8 -*-
# © Kohulan Rajan - 2020

"""Decimer CLI."""

import os
import pathlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import sys
import pickle
import pystow
from selfies import decoder
from .Network import helper

HERE = pathlib.Path(__file__).resolve().parent.joinpath("")

__all__ = [
    "main",
    "load_trained_model",
    "evaluate",
    "predict_SMILES",
    "predict_batches",
]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main():
    global image_features_extracter, transformer, max_length, SELFIES_tokenizer
    get_QUOTE = helper.get_quote()
    if len(sys.argv) < 3 or sys.argv[1] == "--help" or sys.argv[1] == "--h":
        print(
            "\nDefault Usage:\ndecimer --image Image.png\n",
            "e.g. decimer --image Image.pngSample_Images/caffeine.png\n",
            "Note: The default model is set to predict Canonical SMILES\n",
            "\nAvailable Models:\n",
            "- Canonical : Model trained on images depicted using canonical SMILES\n",
            "- Isomeric : Model trained on images depicted using isomeric SMILES, which includes stereochemical information + ions\n",
            "- Augmented: Model trained on images depicted using isomeric SMILES with augmentations",
            "\n\nUsage for single image:\ndecimer --model Canonical --image Image.png\n",
            "\nUsage for folder containing multiple images:\ndecimer --model Canonical --dir path/to/folder\n\n\n",
            get_QUOTE,
        )
        sys.exit()

    # Argument to run DECIMER for a given Image
    elif len(sys.argv) == 3 and sys.argv[1] == "--image":
        model_id = "Canonical"

        (
            image_features_extracter,
            transformer,
            max_length,
            SELFIES_tokenizer,
        ) = load_trained_model(model_id)

        get_SMILES = predict_SMILES(sys.argv[2])
        print("Predicted SMILES for " + sys.argv[2] + " :" + get_SMILES)
        print("\n\n" + get_QUOTE)

    elif len(sys.argv) == 5 and sys.argv[3] == "--image":
        model_id = sys.argv[2]

        (
            image_features_extracter,
            transformer,
            max_length,
            SELFIES_tokenizer,
        ) = load_trained_model(model_id)

        get_SMILES = predict_SMILES(sys.argv[4])
        print("Predicted SMILES for " + sys.argv[4] + " :" + get_SMILES)
        print("\n\n" + get_QUOTE)

    elif len(sys.argv) == 5 and sys.argv[3] == "--dir":
        model_id = sys.argv[2]

        (
            image_features_extracter,
            transformer,
            max_length,
            SELFIES_tokenizer,
        ) = load_trained_model(model_id)

        file_name = predict_batches(sys.argv[4])

        print(
            "Predicted SMILES for images in the folder "
            + sys.argv[4]
            + "is in the "
            + file_name
            + " file\n"
        )
        print("\n\n" + get_QUOTE)
    # Call help, if the user arguments did not satisfy the rules.
    else:
        # print(len(sys.argv))
        print("\nSee help using python DECIMER_V1.py --help")
        print("\n\n" + get_QUOTE)


def load_trained_model(model_id: str):
    """Load a pre-trained model included with :mod:`decimer`.

    :param model_id: The name of the model. Should be one of: "Augmented", "Canonical", or "Isomeric"
    :returns: A quadruple of an image feature extractor that comes from
        :func:`decimer.Network.helper.load_image_features_extract_model`, a transformer from
        :func:`decimer.Network.helper.load_transformer`, the max length, and a tokenizer.
    """
    # load important assets
    SELFIES_tokenizer, max_length = helper.load_assets(model_id)
    if model_id == "Canonical":
        vocabulary = "max_length"
        get_type = max_length
    else:
        vocabulary = "SELFIES_tokenizer"
        get_type = SELFIES_tokenizer
    transformer, target_size = helper.load_transformer(vocabulary, get_type)
    image_features_extracter = helper.load_image_features_extract_model(target_size)

    # restoring the latest checkpoint in checkpoint_dir
    model_default_path = pystow.join("decimer", "Trained_Models")
    checkpoint_path = str(model_default_path) + "/" + model_id + "/"
    # print(checkpoint_path)
    model_url = "https://storage.googleapis.com/iupac_models_trained/DECIMER_transformer_models/DECIMER_trained_models_v1.0.zip"
    if not os.path.exists(checkpoint_path):
        helper.download_trained_weights(model_url, checkpoint_path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00051)

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))

    return image_features_extracter, transformer, max_length, SELFIES_tokenizer


# Evaluator
def evaluate(image):
    temp_input = tf.expand_dims(helper.load_image(image)[0], 0)
    img_tensor_val = image_features_extracter(temp_input)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3])
    )

    output = tf.expand_dims([SELFIES_tokenizer.word_index["<start>"]], 0)
    result = []
    end_token = SELFIES_tokenizer.word_index["<end>"]

    for i in range(max_length):
        dec_mask = helper.create_masks_decoder(output)

        predictions, _ = transformer(img_tensor_val, output, False, dec_mask)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == end_token:
            return result

        result.append(SELFIES_tokenizer.index_word[int(predicted_id)])
        output = tf.concat([output, predicted_id], axis=-1)

    return result


# Predictor helper function
def predict_SMILES(image_path):
    predicted_SELFIES = evaluate(image_path)

    predicted_SMILES = decoder(
        "".join(predicted_SELFIES).replace("<start>", "").replace("<end>", ""),
        constraints="hypervalent",
    )

    return predicted_SMILES


# Batch predicted helper function
def predict_batches(dir_path):
    dirlist = os.listdir(dir_path)
    file_out = open("../../Predicted_SMILES.txt", "w")
    for file in dirlist:
        predicted_SMILES = predict_SMILES(dir_path + "/" + file)
        file_out.write(file + "\t" + predicted_SMILES + "\n")
    file_out.close()

    return "Predicted_SMILES.txt"


if __name__ == "__main__":
    main()
