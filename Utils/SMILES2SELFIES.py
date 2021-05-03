"""
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by : Kohulan.R on 2020/04/04
"""
import sys
import numpy as np
from selfies import encoder
import argparse

"""Translates a SMILES into a SELFIES.
	
	The SMILES file should passed as an argument to the code and the the code uses the selfies.encoder to encode the SMILES into SELFIES.
	If a SMILES string cannot be encoded the ID of the SMILES string will get printed.
	Args: SMILES filepath.


"""
parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+")

args = parser.parse_args()
print(args)
for file_in in args.file:
    file_out = open(str(file_in) + "_SELFIES.txt", "w")

    with open(file_in, "r") as fp:
        for i, line in enumerate(fp):
            id_ = line.strip().split("\t")[0]
            smiles = line.strip().split("\t")[1]

            try:
                encoded = encoder(smiles)
                file_out.write(id_ + "," + encoded + "\n")
            except Exception as e:
                print(id_)

file_out.close()
