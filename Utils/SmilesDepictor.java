/*
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Copyright (c) 2019, Kohulan Rajan
 */

import java.text.DecimalFormat;

import javax.imageio.ImageIO;
import javax.vecmath.Point2d;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;

import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.depict.DepictionGenerator;
import org.openscience.cdk.geometry.GeometryTools;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.layout.StructureDiagramGenerator;
import org.openscience.cdk.renderer.color.UniColor;
import org.openscience.cdk.renderer.generators.standard.StandardGenerator;
import org.openscience.cdk.smiles.SmilesParser;

public class SmilesDepictor {
	String temp = "";
	String moleculeTitle = null;
	int moleculeCount = 0;
	boolean verbose = false;

	public static void main(String[] args) throws Exception {
		long startTime = System.nanoTime();
		for (String s : args) {
		IAtomContainer molecule = null;

		// Directories according to your data files
		String file_name = s.substring(0, s.length()- 4);
		File fileout = new File(file_name + "_Generated.txt");
		File image_directory = new File(file_name);

		if (!fileout.exists()) {
			fileout.createNewFile();
		}
    	if (!image_directory.exists()){
        image_directory.mkdir();
   		}

		BufferedReader input_file = new BufferedReader(new FileReader(s));
		BufferedWriter bufferWriter = new BufferedWriter(new FileWriter(fileout.getAbsoluteFile(), true));

		String line = input_file.readLine();
		while (line != null) {
			String[] splitted_line = line.split(",");
			try {
			SmilesParser smi = new SmilesParser(DefaultChemObjectBuilder.getInstance());
			molecule = smi.parseSmiles(splitted_line[1]);

			StructureDiagramGenerator sdg = new StructureDiagramGenerator();
			sdg.setMolecule(molecule);
			sdg.generateCoordinates(molecule);
			molecule = sdg.getMolecule();
			Point2d point = GeometryTools.get2DCenter(molecule);
			DecimalFormat df2 = new DecimalFormat("#");

			double i = Math.random();
			i = i * 360.0 + 0.0;
			GeometryTools.rotate(molecule, point, (i * Math.PI / 180.0));
			DepictionGenerator dptgen = new DepictionGenerator().withSize(299, 299)
			.withAtomValues().withParam(StandardGenerator.StrokeRatio.class, 1.0)
			.withAnnotationColor(Color.BLACK).withParam(StandardGenerator.AtomColor.class, new UniColor(Color.BLACK))
			.withBackgroundColor(Color.WHITE);
			BufferedImage image = dptgen.depict(molecule).toImg();
			String String_i = df2.format(i);
			
			ImageIO.write(image, "png",
					new File(file_name+"/" + splitted_line[0] + ".png"));

			bufferWriter.write(file_name+"/"+splitted_line[0] + "," + splitted_line[1] + "\n");
		}
			 catch(Exception e) { 
			 	System.out.println("toString(): "  + e.toString());
			 }
			

			line = input_file.readLine();
		}
		input_file.close();
		bufferWriter.close();

		long endTime = System.nanoTime() - startTime;
		double seconds = (double) endTime / 1000000000.0;
		DecimalFormat d = new DecimalFormat(".###");
		System.out.println(
				"Image arrays generated..\nAll images rendered successfully!! Time: " + d.format(seconds) + " seconds");
	}}
}
