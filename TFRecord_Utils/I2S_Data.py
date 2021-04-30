# © Kohulan Rajan - 2020
import tensorflow as tf
import re
import numpy as np
import os
import sys
import time
import json
from glob import glob
from PIL import Image
import pickle

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Image preprocessing using InceptionV3 with Keras API


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def data_loader(PATH, total_data):

    # read the Captions file
    with open(PATH, "r") as txt_file:
        smiles = txt_file.read()

    # storing the captions and the image name in vectors
    all_smiles = []
    all_img_name = []

    for line in smiles.split("\n"):
        # Split the ID and SMILES to seperate tokens
        tokens = line.split(",")
        # Add start and end annotations to SMILES string
        image_id = str(tokens[0]) + ".jpg"
        caption = "<start> " + str(tokens[1].rstrip()) + " <end>"

        # PATH of training images
        full_image_path = PATH + image_id

        all_img_name.append(full_image_path)
        all_smiles.append(caption)

    train_captions, img_name_vector = (all_smiles, all_img_name)

    # selecting the required amount of captions from the data set (optional)
    num_examples = total_data
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]

    # print ("Img ",img_name_vector[1]," caption ",train_captions[1])
    print(
        "Selected Data ", len(train_captions), "All data ", len(all_smiles), flush=True
    )

    # Using InceptionV3 and using the pretrained Imagenet weights
    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # Defining the maximum number of tokens to generate
    def calc_max_length(tensor):
        return max(len(t) for t in tensor)

    # choosing the top 5000 words from the vocabulary
    top_k = 500
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=top_k, oov_token="<unk>", filters='!"$&:;?^`{}~ ', lower=False
    )
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    tokenizer.word_index["<pad>"] = 0
    tokenizer.index_word[0] = "<pad>"

    # padding each vector to the max_length of the captions, if the max_length parameter is not provided, pad_sequences calculates that automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
        train_seqs, padding="post"
    )

    # calculating the max_length used to store the attention weights
    max_length = calc_max_length(train_seqs)

    # Create training and validation sets using 90-10 split
    img_name_train, cap_train = img_name_vector, cap_vector

    return cap_train, tokenizer, max_length, image_features_extract_model
