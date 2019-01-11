package io;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Output {
	private BufferedWriter writer;

	public Output(File file) {
		try {
			writer = new BufferedWriter(new FileWriter(file));
		} catch (IOException e) {
			System.out.println("Coudn't initialize file writer");
		}
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
				System.out.println("adding: " + line.toString());
				writer.write(line.toString());
				writer.newLine();
			}
			writer.close();
		} catch (IOException es) {
			System.out.println("Couldn't save");
		}
	}

}
