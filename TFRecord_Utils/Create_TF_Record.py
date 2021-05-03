# -*- coding: UTF-8 -*-
# © Kohulan Rajan - 2020
import tensorflow as tf
import os
import I2S_Data
import pickle
import efficientnet.tfkeras as efn

# Initial Setup for GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main():
    imgs_dir_path = "Train_Images/"
    captions_path = "Captions.txt"
    num_shards = int(102400 / 1024)  # corresponds to total train files

    get_train_tfrecord(imgs_dir_path, captions_path, num_shards)


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = efn.preprocess_input(img)
    return img, image_path


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_train_tfrecord(imgs_dir_path, captions_path, num_shards):

    with open(captions_path, "r") as txt_file:
        smiles = txt_file.read()

    # storing the captions and the image name in vectors
    all_smiles = []
    all_img_name = []

    for line in smiles.split("\n"):
        # Split the ID and SMILES to seperate tokens
        tokens = line.split(",")
        # Add start and end annotations to SMILES string
        image_id = str(tokens[0]) + ".jpg"
        try:
            caption = "<start> " + str(tokens[1].rstrip()) + " <end>"
        except IndexError as e:
            print(e, flush=True)

        # PATH of training images
        full_image_path = imgs_dir_path + image_id

        all_img_name.append(full_image_path)
        all_smiles.append(caption)

    train_captions, img_name_vector = (all_smiles, all_img_name)
    num_examples = 102400
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]

    # Get Tokenizer from a captions file
    (
        cap_train,
        tokenizer,
        max_length,
        image_features_extract_model,
    ) = I2S_Data.data_loader(captions_path, num_examples)

    with open("tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)

    with open("max_length.pkl", "wb") as file:
        pickle.dump(max_length, file)

    for i in range(num_shards):
        subsets_num = int(len(train_captions) / num_shards) + 1
        sub_split_img_id = img_name_vector[i * subsets_num : (i + 1) * subsets_num]
        sub_split_captions = train_captions[i * subsets_num : (i + 1) * subsets_num]
        sub_split_cap_train = cap_train[i * subsets_num : (i + 1) * subsets_num]
        # print(sub_split_captions)
        # print(sub_split_cap_train)
        # print(len(sub_split_img_id))

        tfrecord_name = "tfrecords/" + "train-%02d.tfrecord" % i
        writer = tf.io.TFRecordWriter(tfrecord_name)
        counter = 0
        for j in range(len(sub_split_img_id)):

            img = tf.expand_dims(load_image(sub_split_img_id[j])[0], 0)

            batch_features = image_features_extract_model(img)
            batch_features = tf.reshape(
                batch_features, (batch_features.shape[0], -1, batch_features.shape[3])
            )
            # print(batch_features.shape)
            caption_ = sub_split_cap_train[j]
            # image_id_ = sub_split_img_id[counter]
            counter = counter + 1
            feature = {
                #'image_id': _bytes_feature(image_id_.encode('utf8')),
                "image_raw": _bytes_feature(batch_features.numpy().tostring()),
                "caption": _bytes_feature(caption_.tostring()),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            serialized = example.SerializeToString()
            writer.write(serialized)
        print("%s write to tfrecord success!" % tfrecord_name)


if __name__ == "__main__":
    main()
