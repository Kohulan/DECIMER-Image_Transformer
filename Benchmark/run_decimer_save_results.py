import os
import sys

from DECIMER import predict_SMILES


def main():
    """This script runs Decimer on every image in a given directory (first
    argument) and saves the results in a text file with a given ID (second
    argument)."""
    im_path = sys.argv[1]
    save_ID = sys.argv[2]

    # Don't start from beginning if a benchmark run aborted for some reason
    with open("{}.txt".format(save_ID), "a+") as output:
        lines = output.readlines()
        already_processed = list([line.split("\t")[0] for line in lines])

    for im in os.listdir(im_path):
        if im not in already_processed:
            with open("{}.txt".format(save_ID), "a") as output:
                smiles = predict_SMILES(os.path.join(im_path, im))
                output.write("{}\t{}\n".format(im, smiles))
                already_processed.append(im)
                print(im)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main()
    else:
        print("Usage: {} image_dir benchmark_ID".format(sys.argv[0]))
