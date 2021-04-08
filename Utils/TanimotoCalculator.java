/*
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Copyright (c) 2019, Kohulan Rajan
 */

import java.awt.GraphicsEnvironment;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.BitSet;

import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.fingerprint.Fingerprinter;
import org.openscience.cdk.fingerprint.PubchemFingerprinter;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.similarity.Tanimoto;
import org.openscience.cdk.smiles.SmiFlavor;
import org.openscience.cdk.smiles.SmilesGenerator;
import org.openscience.cdk.smiles.SmilesParser;

public class TanimotoCalculator {

	String temp = "";
	String moleculeTitle = null;
	int moleculeCount = 0;
	boolean verbose = false;

	public static void main(String[] args) throws Exception {
		Fingerprinter fingerprinter = new Fingerprinter();
		PubchemFingerprinter pubchemfingerprinter = new PubchemFingerprinter(null);
		System.setProperty("java.awt.headless", "true");
		long startTime = System.nanoTime();
		GraphicsEnvironment.getLocalGraphicsEnvironment();
		System.out.println("Headless mode: " + GraphicsEnvironment.isHeadless());

		File fileout = new File(args[0].substring(0, args[0].length()) + "_TanimotoCDK");

		if (!fileout.exists()) {
			fileout.createNewFile();
		}
		BufferedReader input_file = new BufferedReader(
				new FileReader(args[0]));
		BufferedWriter bufferWriter = new BufferedWriter(new FileWriter(fileout.getAbsoluteFile(), true));

		String line = input_file.readLine();
		while (line != null) {
			String[] splitted_line = line.split("\t");
			try {
				SmilesParser smi = new SmilesParser(DefaultChemObjectBuilder.getInstance());
				IAtomContainer ori_molecule = smi.parseSmiles(splitted_line[0]);
				IAtomContainer pred_molecule = smi.parseSmiles(splitted_line[2]);
				BitSet fingerprint1 = pubchemfingerprinter.getBitFingerprint(ori_molecule).asBitSet();
				BitSet fingerprint2 = pubchemfingerprinter.getBitFingerprint(pred_molecule).asBitSet();
				double tanimoto_coefficient = Tanimoto.calculate(fingerprint1, fingerprint2);
				SmilesGenerator sg      = new SmilesGenerator(SmiFlavor.Absolute);
				String          smi_ori  = sg.create(ori_molecule);
				String          smi_pred  = sg.create(pred_molecule);
				bufferWriter.write(smi_ori + "\tOriginalSmiles\t" + smi_pred
						+ "\tPredictedSmiles\tTanimoto Similarity: " + tanimoto_coefficient + "\n");
			} catch (Exception e) {
				System.out.println(e.getLocalizedMessage());
				bufferWriter.write("String rejected\n");
			}
			line = input_file.readLine();
		}
		input_file.close();
		bufferWriter.close();

		long endTime = System.nanoTime() - startTime;
		double seconds = (double) endTime / 1000000000.0;
		DecimalFormat d = new DecimalFormat(".###");
		System.out.println("Calculations completed successfully!! Time: " + d.format(seconds) + " seconds");
	}
}
