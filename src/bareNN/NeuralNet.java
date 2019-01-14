package bareNN;

import java.io.BufferedReader;
import java.io.DataOutput;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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
	private double[][] trainingAnswers;
	private double delta = 0.1; // Step size in gradient descent
	private int inputSize; // size of input layer
	private int outputSize; // size of output layer
	private final String savePath;
	private String trainingPath = "saves/digit/Data/mnist_train.csv";

	public NeuralNet(String savePath, int[] layers) {
		this.savePath = savePath + "/savedNet";
		// int[] params = { 4, 5 };
		blackBox = new BlackBox(layers, true);
		getTrainingData(trainingPath);
		// System.out.println(cost());
		// blackBox.clearLayers();
		// blackBox.save(savePath);
		// backPropagation(500);
	}

	public NeuralNet(String saveLocation, int numLayers) {
		this.savePath = saveLocation;
		blackBox = new BlackBox(saveLocation, numLayers);
		blackBox.clearLayers();
		inputSize = blackBox.inputSize();
		outputSize = blackBox.outputSize();
		getTrainingData(trainingPath);
	}

	private void getTrainingData(String path) {
		Input in = new Input(new File(path));
		double[] metaData = in.readLine();
		int numExamples = (int) metaData[0]; // Number of training examples
		inputSize = (int) metaData[1];
		outputSize = (int) metaData[2];
		trainingData = new double[numExamples][];
		trainingAnswers = new double[numExamples][];

		for (int i = 0; i < numExamples; i++) {
			double[] line = in.readLine();
			trainingAnswers[i] = new double[outputSize];
			trainingAnswers[i][(int) line[0]] = 1;
			trainingData[i] = new double[inputSize];
			for (int j = 1; j < inputSize; j++) {
				trainingData[i][j] = line[j] / 255;
			}
		}
	}

	public double[] apply(double[] input) {
		assert input.length == inputSize;
		return blackBox.eval(input);
	}

	public double cost() {
		assert trainingAnswers.length == trainingData.length;
		double cost = 0;
		for (int i = 0; i < trainingData.length; i++) {
			double singleCost = sqError(blackBox.eval(trainingData[i]), trainingAnswers[i]) / trainingData.length;

			cost += singleCost;
		}
		return cost;
	}

	public void backPropagation(int iterations, double stepSize) {
		for (int j = 0; j < blackBox.numLayers() - 1; j++) {
			for (int i = 0; i < iterations; i++) {
				double[] normalizedDescent = normalize(descent(j), stepSize);
				blackBox.adjust(j, normalizedDescent);
				double cost = cost();
				System.out.println("Iteration " + i + " has a cost of " + cost);
			}
		}
	}

	public void save() {
		blackBox.save(savePath);
	}

	public void save(String altSavePath) {
		blackBox.save(altSavePath);
	}

	// Returns a copy of the given vector rescaled to the given magnitude
	private double[] normalize(double[] vector, double magnitude) {
		double originalMag = 0;
		double[] output = new double[vector.length];
		for (int i = 0; i < vector.length; i++)
			originalMag += vector[i] * vector[i];
		originalMag = Math.sqrt(originalMag);
		for (int i = 0; i < vector.length; i++)
			output[i] = vector[i] / originalMag * magnitude;
		return output;
	}

	private double[] descent(int layer) {
		double initialCost = cost();
		// sign indicates direction to move variable, magnitude is how much it lowers
		// cost.
		double[] descent = new double[blackBox.getWeights(layer).length + blackBox.getBiases(layer).length];
		
		for (int j = 0; j < trainingData.length; j++) {
			blackBox.eval(trainingData[j]);
			double startError = sqError(blackBox.outputLayer(), trainingAnswers[j]);
			for (int i = 0; i < blackBox.numWeights(layer); i++) {
				
			}
		}

		for (int i = 0; i < blackBox.numWeights(layer); i++) {
			if (i % 1000 == 0)
				System.out.println(
						"finished weight " + i + " out of " + blackBox.numWeights(layer) + " in layer " + layer);
			blackBox.addToWeight(layer, i, delta);
			double positiveChange = cost() - initialCost; // Testing increasing the weight
			blackBox.addToWeight(layer, i, -2 * delta);
			double negativeChange = cost() - initialCost; // Testing decreasing the weight
			blackBox.addToWeight(layer, i, delta); // Returns to normal
			if (positiveChange < 0 || negativeChange < 0) {
				if (positiveChange < negativeChange)
					descent[i] = Math.abs(positiveChange);
				else
					descent[i] = negativeChange;
			} else
				descent[i] = 0;
		}

		for (int i = 0; i < blackBox.getBiases(layer).length; i++) {
			blackBox.addToBias(layer, i, delta);
			double positiveChange = cost() - initialCost; // Testing increasing the bias
			blackBox.addToBias(layer, i, -2 * delta);
			double negativeChange = cost() - initialCost; // Testing decreasing the bias
			blackBox.addToBias(layer, i, delta); // Returns to normal

			if (positiveChange < 0 || negativeChange < 0) {
				if (positiveChange < negativeChange)
					descent[blackBox.getWeights(layer).length + i] = Math.abs(positiveChange);
				else
					descent[blackBox.getWeights(layer).length + i] = negativeChange;
			} else
				descent[i] = 0;
		}
		return descent;
	}

	public String getSaveLocation() {
		return savePath;
	}

	private static double sqError(double[] outputs, double[] answer) {
		double sqError = 0;
		for (int i = 0; i < outputs.length; i++) {
			assert outputs[i] != Double.NaN;
			sqError += (outputs[i] - answer[i]) * (outputs[i] - answer[i]);
		}
		assert sqError != Double.NaN;
		return sqError;
	}

	/*
	 * public static void main(String[] args) { int[] layers = { 4, 5, 5}; NeuralNet
	 * net = new NeuralNet("saves/3layer/", layers); //NeuralNet net = new
	 * NeuralNet("saves/3layer/savedNet.txt", 2); net.backPropagation(200);
	 * //net.save(); System.out.println(net.cost());
	 * System.out.println(net.getSaveLocation()); double[] test = { 1, 0, 0, 1 };
	 * System.out.println(Arrays.toString(net.apply(test))); }
	 */
}
