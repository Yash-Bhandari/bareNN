package bareNN;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import io.Input;
import io.Output;

public class NeuralNet {
	private static final double midPodouble = 0.5; // Midpoint of logistic equation
	private static final double slope = 5; // Steepness of logistic equation
	private static final double limit = 1; // Limit of logistic equation

	// Even arrays are layers.
	// Odd arrays are connections.
	// Last element in connection arrays is that connection's bias.
	private ArrayList<double[]> blackBox;

	// Each array is one training example.
	// Indices 0, 1, ... , inputSize-1 are input nodes.
	// Indices inputSize, inputSize+1, ... , inputSize+outputSize-1 are correct
	// output nodes.
	private double[][] trainingData;
	private int inputSize; // size of input
	private int outputSize;

	public NeuralNet() {
		blackBox = new ArrayList<double[]>();
		addLayer(4);
		addLayer(4);
		// trainingData = readData();
		getTrainingData(new Input(new File("src/io/training.txt")));
		new Output(new File("src/io/output.txt")).save(trainingData);
	}

	private void addLayer(int size) {
		if (!blackBox.isEmpty()) // Adds array representing the connections between previous layer and new layer.
			blackBox.add(new double[blackBox.get(blackBox.size() - 1).length * size]);
		blackBox.add(new double[size]);
	}

	private void getTrainingData(Input in) {
		double[] metaData = in.readLine();
		int numExamples = (int)metaData[0]; // Number of training examples
		inputSize = (int) metaData[1];
		outputSize = (int) metaData[2];
		trainingData = new double[numExamples][];
		
		for (int i = 0; i < numExamples; i++) {
			trainingData[i] = in.readLine();
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
