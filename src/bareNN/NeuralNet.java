package bareNN;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import io.Input;
import io.Output;

public class NeuralNet {

	private BlackBox blackBox; // Matrix representation of neural net.
	// Each array is one training example.
	// Indices 0, 1, ... , inputSize-1 are input nodes.
	// Indices inputSize, inputSize+1, ... , inputSize+outputSize-1 are correct
	// output nodes.
	private double[][] trainingData;
	private int inputSize; // size of input layer
	private int outputSize; // size of output layer

	public NeuralNet() {
		blackBox = new BlackBox();
		getTrainingData(new Input(new File("src/io/training.txt")));
		blackBox.addLayer(inputSize);
		blackBox.addLayer(outputSize);
		inputSize = blackBox.inputSize();
		outputSize = blackBox.outputSize();
		blackBox.save("src/io/output.txt");
		
	}


	
	private void evalNode(int layer, int node) {
		
	}

	private void getTrainingData(Input in) {
		double[] metaData = in.readLine();
		int numExamples = (int) metaData[0]; // Number of training examples
		inputSize = (int) metaData[1];
		outputSize = (int) metaData[2];
		trainingData = new double[numExamples][];

		for (int i = 0; i < numExamples; i++) {
			trainingData[i] = in.readLine();
		}
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
