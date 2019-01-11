package io;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Output {
	private BufferedWriter writer;

	public Output(File file) {
		try {
			writer = new BufferedWriter(new FileWriter(file));
		} catch (IOException e) {
			System.out.println("Coudn't initialize file writer");
		}
	}

	public void save(ArrayList<double[]> toSave) {
		double[][] saving = new double[toSave.size()][];
		for (int i = 0; i < saving.length; i++)
			saving[i] = toSave.get(i);
		save(saving);
	}
	
	public void save(double[][] toSave) {
		try {
			for (int i = 0; i < toSave.length; i++) {
				StringBuilder line = new StringBuilder();
				for (int j = 0; j < toSave[i].length; j++) {
					line.append(toSave[i][j]);
					if (j < toSave[i].length - 1)
						line.append(",");
				}
				writer.write(line.toString());
				writer.newLine();
			}
			writer.close();
		} catch (IOException es) {
			System.out.println("Couldn't save");
		}
	}

}
