# -*- coding: UTF-8 -*-
# © Kohulan Rajan - 2020
# helper functions for the predictions
import pickle
import pathlib
from .Transformer import I2S_Model_Transformer
import efficientnet.tfkeras as efn
import tensorflow as tf
import subprocess
import urllib.request
import pystow
from ..assets import HERE
from typing import Tuple, Union
import random

# load tokenizer and max length
def load_assets(model_id):
    tokenizer = pickle.load(
        open(HERE.joinpath(model_id, "SELFIES_tokenizer.pkl"), "rb")
    )
    max_length = pickle.load(
        open(HERE.joinpath(model_id, "SELFIES_max_length.pkl"), "rb")
    )

    return tokenizer, max_length


# load parameters for the model
def load_transformer(
    vocabulary: str,
    get_type,
) -> Tuple[I2S_Model_Transformer.Transformer, Tuple[int, int, int]]:
    """Load a transformer with given vocabulary and tokenizer.

    :param vocabulary:
    :param get_type:
    :return:
    """
    if vocabulary == "SELFIES_tokenizer":
        target_vocab_size = len(get_type.word_index) + 1
    else:
        target_vocab_size = get_type + 1
    num_layer = 4
    d_model = 512
    dff = 2048
    num_heads = 8
    row_size = 10
    col_size = 10
    dropout_rate = 0.1
    target_size = (299, 299, 3)

    transformer = I2S_Model_Transformer.Transformer(
        num_layer,
        d_model,
        num_heads,
        dff,
        row_size,
        col_size,
        target_vocab_size,
        max_pos_encoding=target_vocab_size,
        rate=dropout_rate,
    )

    return transformer, target_size


# Resize image and decode
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = efn.preprocess_input(img)
    return img, image_path


def load_image_features_extract_model(target_size):
    # Using EfficientnetB3 and using the pretrained Imagenet weights
    image_model = efn.EfficientNetB3(
        weights="noisy-student", input_shape=target_size, include_top=False
    )

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    return image_features_extract_model


# Downloads the model and unzips the file downloaded, if the model is not present on the working directory.
def download_trained_weights(model_url, model_path, verbose=1):
    # Download trained models
    if verbose > 0:
        print("Downloading trained model to " + str(model_path))
        model_path = pystow.ensure("decimer", url=model_url)
        print(model_path)
    if verbose > 0:
        print("... done downloading trained model!")
        subprocess.run(
            [
                "unzip",
                model_path.as_posix(),
                "-d",
                model_path.parent.as_posix(),
            ]
        )


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


# Generate a random quote for whimsical fun!


def random_line(file_name):
    line = next(file_name)
    for num, aline in enumerate(file_name, 2):
        if random.randrange(num):
            continue
        line = aline
    return line


def get_quote():
    quote = random_line(open((HERE.joinpath("Quotes_database")), "r"))
    return quote
