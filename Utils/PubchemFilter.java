/*
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Copyright (c) 2019, Kohulan Rajan
 */

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.config.IsotopeFactory;
import org.openscience.cdk.config.Isotopes;
import org.openscience.cdk.graph.ConnectivityChecker;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IAtomContainerSet;
import org.openscience.cdk.interfaces.IElement;
import org.openscience.cdk.interfaces.IMolecularFormula;
import org.openscience.cdk.qsar.descriptors.molecular.WeightDescriptor;
import org.openscience.cdk.qsar.result.IDescriptorResult;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.smiles.SmiFlavor;
import org.openscience.cdk.smiles.SmilesGenerator;
import org.openscience.cdk.tools.manipulator.MolecularFormulaManipulator;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

public class PubchemFilter {
	public static void main(String[] args) throws Exception {
		for (String s : args) {
			// Directories according to your data files
			File fileout = new File(s.substring(0, s.length() - 4) + "_Selected.txt");

			if (!fileout.exists()) {
				fileout.createNewFile();
			}
			BufferedReader input_file = new BufferedReader(new FileReader(s));
			BufferedWriter bufferWriter = new BufferedWriter(new FileWriter(fileout.getAbsoluteFile(), true));
			SmilesGenerator sg = new SmilesGenerator(SmiFlavor.Absolute);
			String line = input_file.readLine();

			while (line != null) {
				String[] splitted_line = line.split("\t");
				try {
					SmilesParser smi = new SmilesParser(DefaultChemObjectBuilder.getInstance());
					IAtomContainer mol_new = smi.parseSmiles(splitted_line[1]);
					//IAtomContainer mol_new = AtomContainerManipulator.removeHydrogens(molecule);
					String smi_ = sg.create(mol_new);

					IDescriptorResult weight = new WeightDescriptor().calculate(mol_new).getValue();
					Object fragments = ConnectivityChecker.partitionIntoMolecules(mol_new);
					int fragmentCount = ((IAtomContainerSet) fragments).getAtomContainerCount();

					// System.err.println(splitted_line[0] + " "+ mol_new.getBondCount() +" " +
					// Double.valueOf(weight.toString()) + " "+ fragmentCount+
					// SelectedElements(mol_new) + " "+ HIsotope(mol_new) +"\n");

					// Molecule Filtering rules
					if (mol_new.getBondCount() > 3 && mol_new.getBondCount() <= 40
							&& Double.valueOf(weight.toString()) <= 1500.0 && fragmentCount <= 1
							&& SelectedElements(mol_new) != true && HIsotope(mol_new) != true) {
						bufferWriter.write(splitted_line[0] + "," + smi_ + "\n");
					}

				} catch (Exception e) {
					System.err.println(splitted_line[0] + " Error");
					//e.printStackTrace();
				}
				line = input_file.readLine();

			}
			input_file.close();
			bufferWriter.close();

			System.out.println("All process successfully completed!!");
		}
	}

	// Function to check selected Elements
	public static boolean SelectedElements(IAtomContainer molecule) {
		IMolecularFormula mols = MolecularFormulaManipulator.getMolecularFormula(molecule);
		List<IElement> elements = MolecularFormulaManipulator.elements(mols);
		List<String> String_elements = new ArrayList<String>();
		for (IElement i : elements) {
			String_elements.add(i.getSymbol());
		}

		HashMap<String, String> Selected_Elements = new HashMap<>();
		Selected_Elements.put("C", "Atom Symbol");
		Selected_Elements.put("H", "Atom Symbol");
		Selected_Elements.put("N", "Atom Symbol");
		Selected_Elements.put("O", "Atom Symbol");
		Selected_Elements.put("P", "Atom Symbol");
		Selected_Elements.put("S", "Atom Symbol");
		Selected_Elements.put("F", "Atom Symbol");
		Selected_Elements.put("Cl", "Atom Symbol");
		Selected_Elements.put("Br", "Atom Symbol");
		Selected_Elements.put("I", "Atom Symbol");
		Selected_Elements.put("Se", "Atom Symbol");
		Selected_Elements.put("B", "Atom Symbol");

		try {
			for (String element : String_elements) {
				if (Selected_Elements.containsKey(element)) {
					@SuppressWarnings("unused")
					int x = 0;
				} else {
					return true;

				}
			}

		} catch (Exception e) {
			System.err.println(" Error");
			e.printStackTrace();
		}
		return false;

	}

	// Function to check Hydrogen Isotopes
	public static boolean HIsotope(IAtomContainer molecule) throws Exception {
		// configure Isotopes
		IsotopeFactory iso_H = Isotopes.getInstance();

		// Print file name with Hydrogen Isotopes
		try {
			iso_H.configureAtoms(molecule);
			for (IAtom atom : molecule.atoms()) {
				if (atom.getSymbol() == "H") {
					if (atom.getMassNumber() > 1) {
						return true;

					}
				}
			}

		} catch (Exception e) {
			
			System.err.println(" Error ");

			e.printStackTrace();
		}
		return false;
	}

}
