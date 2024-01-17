import os
import sys

from rdkit import Chem
from rdkit import DataStructs


def compare_molecules_inchi_match(
    input_file_path: str, reference_directory: str
) -> None:
    """This function checks if the molecules in the DECIMER results to a set of
    reference mol-files using Standard InChI.

    Args:
            input_file (str): Path of file that contains image names and SMILES as created by run_decimer_save_results.py
            reference_directory (str): Path of directory with mol/sd files as they come with the benchmark datasets
    """
    """"""
    # We assume that the input file has the structure $img_name\t$smiles\n per line
    tanimoto_list = []
    perfect_match_count = 0
    with open(input_file_path, "r") as input_file:
        lines = input_file.readlines()
        # assert len(os.listdir(reference_directory)) == len(lines)
        for line in lines:
            ID, smiles = line.split("\t")
            smiles = smiles[:-1]  # Delete \n at the end
            # Generate mol objects
            input_mol = Chem.MolFromSmiles(smiles)
            if input_mol:
                for format in [".mol", ".sdf", ".MOL"]:
                    if os.path.exists(
                        os.path.join(reference_directory, ID[:-4] + format)
                    ):
                        reference_mol_suppl = Chem.SDMolSupplier(
                            os.path.join(reference_directory, ID[:-4] + format)
                        )

                # Generate Inchi
                input_inchi = Chem.inchi.MolToInchi(
                    input_mol
                )  # rdmolfiles.MolToSmiles(input_mol, True)
                for reference_mol in reference_mol_suppl:
                    if reference_mol:
                        reference_inchi = Chem.inchi.MolToInchi(
                            reference_mol
                        )  # .rdmolfiles.MolToSmiles(reference_mol, True)#

                        # Count perfect InChI string match
                        if input_inchi == reference_inchi:
                            perfect_match_count += 1

                        # Generate fingerprints and Tanimoto similarity
                        reference_fp = Chem.RDKFingerprint(reference_mol)
                        input_fp = Chem.RDKFingerprint(input_mol)
                        tanimoto = DataStructs.FingerprintSimilarity(
                            reference_fp, input_fp
                        )
                        tanimoto_list.append(tanimoto)
                        break

    print(
        "{} out of {} ({}%)".format(
            perfect_match_count, len(lines), perfect_match_count / len(lines) * 100
        )
    )
    print("Average Tanimoto: {}".format(sum(tanimoto_list) / len(tanimoto_list)))
    tanimoto_one_count = len(
        [tanimoto for tanimoto in tanimoto_list if tanimoto == 1.0]
    )
    print(
        "Tanimoto 1.0 count: {} out of {} ({}%)".format(
            tanimoto_one_count, len(lines), tanimoto_one_count / len(lines)
        )
    )


def main():
    if len(sys.argv) != 3:
        print(
            '"Usage of this function: {} decimer_benchmark_results_file directory-with-reference-molfiles'.format(
                sys.argv[0]
            )
        )
    if len(sys.argv) == 3:
        compare_molecules_inchi_match(sys.argv[1], sys.argv[2])
    sys.exit(1)


if __name__ == "__main__":
    main()
