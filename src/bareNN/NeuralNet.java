package bareNN;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class NeuralNet {
    private static final double midPodouble = 0.5; // Midpodouble of logistic equation
    private static final double slope = 5; // Steepness of logistic equation
    private static final double limit = 1; // Limit of logistic equation

    private ArrayList<double[]> blackBox;
    private double[][] trainingData;
    
    public NeuralNet() {
        blackBox = new ArrayList<double[]>();
        addLayer(4);
        addLayer(4);
        trainingData = readData();
    }

    private void addLayer(int size) {
        if (!blackBox.isEmpty()) // Adds array representing the connections between previous layer and new layer
            blackBox.add(new double[blackBox.get(blackBox.size() - 1).length * size]);
        blackBox.add(new double[size]);
    }
    
    private double[][] readData() {
        double[][] output;
        try {
        FileReader fReader = new FileReader(new File("training.txt"));
        BufferedReader reader = new BufferedReader(new FileReader(new File("training.txt")));
        String s = reader.readLine();
        
        int numLines = Integer.parseInt(s.substring(0, s.indexOf((' '))));
        //output = new double[s.substring(0, )]
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static double logistic(double input) {
        double power = -slope * (input - midPodouble);
        double output = limit / (1 + Math.exp(power));
        assert output >= 0 && output <= 1;
        return output;
    }
    
    
    private static double sqError(double[] outputs, double[] answer) {
        double sqError = 0;
        for (int i = 0; i < outputs.length; i++) {
            sqError += (outputs[i] - answer[i]) * (outputs[i] - answer[i]);
        }
        return sqError;
    }
    
    public static void main(String[] args) {
        new NeuralNet();
    }
}
