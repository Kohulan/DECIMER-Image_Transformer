# -*- coding: UTF-8 -*-
# © Kohulan Rajan - 2020
import argparse
import os
import pickle

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+")

args = parser.parse_args()
print(args)
# Initial Setup for GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = (
                value.numpy()
            )  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def get_train_tfrecord(num_shards, train_captions, img_name_vector, file_index):
        print("Total number of TFrecords: ", num_shards, flush=True)

        for i in range(num_shards):
            subsets_num = int(len(train_captions) / num_shards)
            sub_split_img_id = img_name_vector[i * subsets_num : (i + 1) * subsets_num]
            sub_split_cap_train = train_captions[
                i * subsets_num : (i + 1) * subsets_num
            ]

            tfrecord_name = "check/" + "train-%02d.tfrecord" % file_index
            writer = tf.io.TFRecordWriter(tfrecord_name)
            counter = 0
            for j in range(len(sub_split_img_id)):
                # print(decoded_image.shape)
                caption_ = sub_split_cap_train[j]
                # image_id_ = sub_split_img_id[counter]
                counter = counter + 1
                feature = {
                    # 'image_id': _bytes_feature(image_id_.encode('utf8')),
                    "image_raw": _bytes_feature(tf.io.read_file(sub_split_img_id[j])),
                    "caption": _bytes_feature(caption_.tostring()),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                serialized = example.SerializeToString()
                writer.write(serialized)
            print("%s write to tfrecord success!" % tfrecord_name, flush=True)
            file_index = file_index + 1

    if __name__ == "__main__":
        main()
