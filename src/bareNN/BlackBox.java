package bareNN;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;

import io.Output;

public class BlackBox {
	private static final double midPodouble = 0.5; // Midpoint of logistic equation
	private static final double slope = 5; // Steepness of logistic equation
	private static final double limit = 1; // Limit of logistic equation

	// Even arrays are layers.
	// Odd arrays are connections.
	// Last element in connection arrays is that connection's bias.
	private ArrayList<double[]> matrix;

	private double[] inputLayer;
	private double[] outputLayer;
	private int inputSize; // size of input layer
	private int outputSize; // size of output layer

	public BlackBox() {
		matrix = new ArrayList<double[]>();
		inputLayer = matrix.get(0);
		outputLayer = matrix.get(matrix.size() - 1);
		populateConnections();
	}

	public void addLayer(int size) {
		if (!matrix.isEmpty()) // Adds array representing the connections between previous layer and new layer.
			matrix.add(new double[matrix.get(matrix.size() - 1).length * size]);
		matrix.add(new double[size]);
	}

	public void eval(double[] input) {
		assert input.length == inputSize;

	}

	public void evalLayer(int layer) {

	}

	public void evalNode(int layer, int node) {
		assert layer < numLayers() - 1;
		for (int i = 0; i < layerSize(layer + 1); i++) {
			addToNode(layer + 1, i, connectionValue(layer, node, i));
		}
	}

	private void addToNode(int layer, int node, double delta) {
		assert layer < numLayers();
		getLayer(layer)[node] += delta;
	}

	// Saves the contents of the matrix to the given file;
	public void save(String path) {
		new Output(new File(path)).save(matrix);
	}

	// Populates connections in matrix with random values near 0
	private void populateConnections() {
		Random r = new Random();
		for (int i = 1; i < matrix.size(); i += 2) {
			for (int j = 0; j < matrix.get(i).length; j++)
				matrix.get(i)[j] = r.nextDouble() / 10;
		}
	}

	private static double logistic(double input) {
		double power = -slope * (input - midPodouble);
		double output = limit / (1 + Math.exp(power));
		assert output >= 0 && output <= 1;
		return output;
	}

	// Returns the value of the connection between node1 in layer and node2 in
	// layer+1
	private double connectionValue(int layer, int node1, int node2) {
		return getConnection(layer)[node1 * layerSize(layer + 1) + node2];
	}

	private double nodeValue(int layer, int node) {
		return matrix.get(layer)[node];
	}

	private int layerSize(int layer) {
		return getLayer(layer).length;
	}

	/**
	 * Returns the ith layer. The input layer is layer 0.
	 * 
	 * @param i The index of the layer
	 * @return The ith layer
	 */
	public double[] getLayer(int i) {
		return matrix.get(i * 2);
	}

	public double[] getConnection(int startLayer) {
		return matrix.get(startLayer * 2 + 1);
	}

	public int inputSize() {
		return inputSize;
	}

	public int outputSize() {
		return outputSize;
	}

	public int numLayers() {
		return (matrix.size() - 1) / 2;
	}

	public double[] inputLayer() {
		return inputLayer;
	}

	public double[] outputLayer() {
		return outputLayer;
	}

}
