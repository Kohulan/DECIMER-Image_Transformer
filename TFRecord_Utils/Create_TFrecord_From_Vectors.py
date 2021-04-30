# -*- coding: UTF-8 -*-
# © Kohulan Rajan - 2020
import tensorflow as tf
import os
import numpy as np
import pickle
import efficientnet.tfkeras as efn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+")

args = parser.parse_args()
print(args)
# Initial Setup for GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

for file_in in args.file:

    def main():
        train_captions = pickle.load(
            open("Capts_Path_" + file_in + "_" + file_in + ".pkl", "rb")
        )
        img_name_vector = pickle.load(
            open("Images_Path_" + file_in + "_" + file_in + ".pkl", "rb")
        )

        print("Total number of selected SMILES Strings: ", len(train_captions), "\n")
        num_shards = int(len(train_captions) / 128)  # corresponds to total train files

        file_index = num_shards * int(file_in)

        get_train_tfrecord(num_shards, train_captions, img_name_vector, file_index)

    target_size = (299, 299, 3)

    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = efn.preprocess_input(img)
        return img, image_path

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = (
                value.numpy()
            )  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def get_train_tfrecord(num_shards, train_captions, img_name_vector, file_index):

        print("Total number of TFrecords: ", num_shards, flush=True)

        # Using InceptionV3 and using the pretrained Imagenet weights
        image_model = efn.EfficientNetB3(
            weights="noisy-student",  # Choose between imagenet and 'noisy-student'
            #         weights='imagenet',
            input_shape=target_size,
            include_top=False,
        )
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output
        image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

        for i in range(num_shards):
            subsets_num = int(len(train_captions) / num_shards)
            sub_split_img_id = img_name_vector[i * subsets_num : (i + 1) * subsets_num]
            sub_split_cap_train = train_captions[
                i * subsets_num : (i + 1) * subsets_num
            ]

            tfrecord_name = "1Mio_TFR/" + "train-%02d.tfrecord" % file_index
            writer = tf.io.TFRecordWriter(tfrecord_name)
            counter = 0
            for j in range(len(sub_split_img_id)):

                img = tf.expand_dims(load_image(sub_split_img_id[j])[0], 0)

                batch_features = image_features_extract_model(img)
                batch_features = tf.reshape(
                    batch_features,
                    (batch_features.shape[0], -1, batch_features.shape[3]),
                )
                # print(batch_features.shape)

                # print(decoded_image.shape)
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
            print("%s write to tfrecord success!" % tfrecord_name, flush=True)
            file_index = file_index + 1

    if __name__ == "__main__":
        main()
