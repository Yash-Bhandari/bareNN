package io;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Input {
	BufferedReader reader;
	boolean nextLine = true;
	String s;

	public Input(File file) {
		try {
			this.reader = new BufferedReader(new FileReader(file));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public double[] readLine() {
		ArrayList<Double> temp = new ArrayList<Double>();
		double[] output;
		if (nextLine)
			try {
				s = reader.readLine();
			} catch (IOException e) {
				System.out.println("couldn't read");
			}
		int i = 0;
		while (i < s.length()) {
			int start = i;
			while (++i < s.length() && s.charAt(i) != ',');
			temp.add(Double.valueOf(s.substring(start, i++)));
		}
		output = new double[temp.size()];
		for (int j = 0; j < temp.size(); j++)
			output[j] = temp.get(j).doubleValue();

		return output;
		// output = new double[s.substring(0, )]
	}

}
