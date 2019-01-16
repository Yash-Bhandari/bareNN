package io;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Input {
    BufferedReader reader;

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
        String s = "";
        try {
            s = reader.readLine();
        } catch (IOException e) {
            System.out.println("couldn't read");
        }
        if (s == null)
            return null;
        int i = 0;
        while (i < s.length()) {
            int start = i;
            while (++i < s.length() && s.charAt(i) != ',' && s.charAt(i) != '|')
                ;
            temp.add(Double.valueOf(s.substring(start, i++)));
        }
        output = new double[temp.size()];
        for (int j = 0; j < temp.size(); j++)
            output[j] = temp.get(j).doubleValue();

        return output;
        // output = new double[s.substring(0, )]
    }

    public int[] readLineInt() {
        ArrayList<Integer> temp = new ArrayList<Integer>();
        int[] output;
        String s = "";
        try {
            s = reader.readLine();
        } catch (IOException e) {
            System.out.println("couldn't read");
        }
        if (s == null)
            return null;
        int i = 0;
        while (i < s.length()) {
            int start = i;
            while (++i < s.length() && s.charAt(i) != ',' && s.charAt(i) != '|')
                ;
            temp.add(Integer.valueOf(s.substring(start, i++)));
        }
        output = new int[temp.size()];
        for (int j = 0; j < temp.size(); j++)
            output[j] = temp.get(j);

        return output;
    }
}
