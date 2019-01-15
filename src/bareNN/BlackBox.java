package bareNN;

import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

import javax.swing.text.LayeredHighlighter;

import io.Input;
import io.Output;

public class BlackBox {
	private static final double midPodouble = 0.5; // Midpoint of logistic equation
	private static final double slope = 5; // Steepness of logistic equation
	private static final double limit = 1; // Limit of logistic equation
	private String path;

	private ArrayList<double[]> layers;
	private ArrayList<double[]> weights;
	private ArrayList<double[]> biases;
	private ArrayList<double[]> cacheLayer;
	private ArrayList<double[]> trainingLayers;

	private int inputSize; // size of input layer
	private int outputSize; // size of output layer
	private int[] iterating = new int[3];

	public BlackBox(int[] addLayers, boolean populate) {
		layers = new ArrayList<double[]>();
		weights = new ArrayList<double[]>();
		biases = new ArrayList<double[]>();
		for (int i = 0; i < addLayers.length; i++)
			addLayer(addLayers[i]);
		cacheLayer = new ArrayList<double[]>(layers.size());
		trainingLayers = new ArrayList<double[]>(layers.size());
		for (int i = 0; i < layers.size(); i++) {
			cacheLayer.add(null);
			trainingLayers.add(null);
		}
		build(populate);
	}

	public BlackBox(String path, int numLayers) {
		this.path = path;
		Input in = new Input(new File(path + ".txt"));
		layers = new ArrayList<double[]>();
		weights = new ArrayList<double[]>();
		biases = new ArrayList<double[]>();

		layers.add(in.readLine());
		for (int i = 1; i < numLayers; i++) {
			weights.add(in.readLine());
			biases.add(in.readLine());
			layers.add(in.readLine());
		}
		cacheLayer = new ArrayList<double[]>(layers.size());
		trainingLayers = new ArrayList<double[]>(layers.size());
		for (int i = 0; i < layers.size(); i++) {
			cacheLayer.add(null);
			trainingLayers.add(null);
		}
		build(false);
	}

	// To be called after layers have been added
	public void build(boolean populate) {
		if (populate)
			populate();
		inputSize = getLayer(0).length;
		outputSize = getLayer(numLayers() - 1).length;
	}

	// WARNING: DO NOT SCROLL PAST THIS POINT

	private void addLayer(int size) {
		if (numLayers() > 0) { // Adds array representing the weights between previous layer and new layer.
			weights.add(new double[size * layerSize(numLayers() - 1)]);
			biases.add(new double[size]);
		}
		layers.add(new double[size]);
	}

	private void setLayer(int layer, double[] newLayer) {
		layers.set(layer, newLayer);
	}

	public void clearLayers() {
		for (int i = 0; i < numLayers(); i++)
			setLayer(i, new double[getLayer(i).length]);
	}

	public void clearTrainingLayers() {
		for (int i = 0; i < numLayers(); i++)
			trainingLayers.set(i, new double[getLayer(i).length]);
	}

	/**
	 * Returns the output of the blackbox after changing the weightIndexth weight
	 * connecting from startLayer to startLayer+1 by delta. Uses the cached
	 * computation to save on time.
	 * 
	 * @param startLayer  The index of the starting layer
	 * @param weightIndex The index of the weight/bias to change
	 * @param delta       The amount to change the weight/bias by
	 * @return The output of the blackbox after changing the specified weight
	 */
	public double[] trainingEval(int startLayer, int weightIndex, double delta) {
		clearTrainingLayers();
		trainingLayers.set(startLayer + 1, cacheLayer.get(startLayer + 1));
		int node = weightIndex % layerSize(startLayer + 1);
		double change = weightIndex < numWeights(startLayer)
				? delta * trainingLayers.get(startLayer)[weightIndex / layerSize(startLayer)]
				: delta;
		trainingLayers.get(startLayer + 1)[node] = cacheLayer.get(startLayer + 1)[node] + change;
		if (startLayer == numLayers() - 2) {
			trainingLayers.set(startLayer + 1, softMax(trainingLayers.get(startLayer + 1)));
		} else
			sigmoid(trainingLayers.get(startLayer + 1));

		for (int i = startLayer + 1; i < numLayers() - 1; i++) {
			trainEvalLayer(i);
		}
		return trainingOutput();
	}

	public double[] eval(double[] input) {
		assert input.length == inputSize;
		clearLayers();
		setLayer(0, input);
		for (int i = 0; i < numLayers() - 1; i++)
			evalLayer(i);
		// for (double d : outputLayer())
		// assert !Double.isNaN(d);
		return outputLayer();
	}

	public void trainEvalLayer(int layer) {
		double[] weights = getWeights(layer);
		for (int i = 0; i < numWeights(layer); i++) {
			trainingLayers.get(layer + 1)[i % layerSize(layer + 1)] += weights[i]
					* trainingLayers.get(layer)[i / layerSize(layer)];
		}
		if (layer == numLayers() - 2)
			trainingLayers.set(layer + 1, softMax((trainingLayers.get(layer + 1))));
		else
			sigmoid(trainingLayers.get(layer + 1));
	}

	public void evalLayer(int layer) {
		for (int i = 0; i < getLayer(layer).length; i++)
			evalNode(layer, i);
		addBiases(layer);

		try {
			cacheLayer.set(layer, Arrays.copyOf(getLayer(layer), layerSize(layer)));
		} catch (IndexOutOfBoundsException e) {
			cacheLayer.add(layer, Arrays.copyOf(getLayer(layer), layerSize(layer)));
		}
		if (layer == numLayers() - 2)
			setLayer(layer + 1, softMax(getLayer(layer + 1)));
		else
			for (int i = 0; i < getLayer(layer + 1).length; i++) {
				setNode(layer + 1, i, sigmoid(nodeValue(layer + 1, i))); // Final transformation of answer
			}
	}

	public void evalNode(int layer, int node) {
		assert layer < numLayers() - 1;
		for (int i = 0; i < layerSize(layer + 1); i++) {
			addToNode(layer + 1, i, getWeight(layer, node, i) * nodeValue(layer, node));
		}
	}

	// Adds biases from the given layer to layer+1
	private void addBiases(int layer) {
		double[] weights = getBiases(layer);
		for (int i = 0; i < weights.length; i++) {
			addToNode(layer + 1, i, weights[i]);
		}
	}

	public void adjust(int layer, double[] adjustments) {
		assert adjustments.length == weights.get(layer).length + biases.get(layer).length;
		for (int i = 0; i < numWeights(layer); i++)
			addToWeight(layer, i, adjustments[i]);
		for (int i = 0; i < numBiases(layer); i++)
			addToBias(layer, i, adjustments[numWeights(layer) + i]);
	}

	public void addToBias(int layer, int endNode, double delta) {
		biases.get(layer)[endNode] += delta;
	}

	private void addToNode(int layer, int node, double delta) {
		assert layer < numLayers();
		getLayer(layer)[node] += delta;
	}

	private void addToNodeTraining(int layer, int node, double delta) {
		assert layer < numLayers();
		trainingLayers.get(layer);
	}

	private void setNode(int layer, int node, double value) {
		getLayer(layer)[node] = value;
	}

	public void addToWeight(int startLayer, int node1, int node2, double delta) {
		setWeight(startLayer, node1, node2, getWeight(startLayer, node1, node2) + delta);
	}

	public void addToWeightTraining(int startLayer, int weightIndex, double delta) {
		addToNodeTraining(startLayer + 1, weightIndex % layerSize(startLayer),
				delta * cacheLayer.get(startLayer)[weightIndex / layerSize(startLayer)]);
		// setWeight(startLayer, node1, node2, getWeight(startLayer, node1, node2) +
		// delta);
	}

	public void addToWeight(int startLayer, int weightIndex, double delta) {
		weights.get(startLayer)[weightIndex] += delta;
	}

	private void setBias(int startLayer, int endNode, double value) {
		biases.get(startLayer)[endNode] = value;
	}

	// Saves the contents of the matrix to the given file;
	public void save(String path) {
		ArrayList<double[]> out = new ArrayList<double[]>();
		out.add(getLayer(0));
		for (int i = 1; i < numLayers(); i++) {
			out.add(getWeights(i - 1));
			out.add(getBiases(i - 1));
			out.add(getLayer(i));
		}
		new Output(new File(path + ".txt")).save(out);
		System.out.println("saved to " + path + ".txt");
	}

	// Populates weights and biases in matrix with random values near 0
	private void populate() {
		Random r = new Random();
		for (int i = 0; i < numLayers() - 1; i++) {
			for (int k = 0; k < layerSize(i + 1); k++) {
				for (int j = 0; j < layerSize(i); j++)
					setWeight(i, j, k, r.nextDouble() / 10);
				setBias(i, k, r.nextDouble() / 10);
			}
		}
	}

	private static double logistic(double input) {
		double power = -slope * (input - midPodouble);
		double output = limit / (1 + Math.exp(power));
		assert output >= 0 && output <= 1;
		return output;
	}

	private static double sigmoid(double input) {
		return 1 / (1 + Math.exp(input));
	}

	private static void sigmoid(double[] input) {
		for (int i = 0; i < input.length; i++)
			input[i] = sigmoid(input[i]);
	}

	private static double[] softMax(double[] input) {
		double[] out = new double[input.length];
		double divisor = 0;
		for (double d : input) {
			double temp = Math.exp(d);
			// if (!Double.isNaN((temp)))
			divisor += Math.exp(d);
		}
		for (int i = 0; i < input.length; i++) {
			out[i] = Math.exp(input[i]) / divisor;
			assert !Double.isNaN(out[i]);
		}
		return out;
	}

	private void setWeight(int startLayer, int node1, int node2, double value) {
		weights.get(startLayer)[node1 * layerSize(startLayer + 1) + node2] = value;
	}

	public int numWeights(int startLayer) {
		return weights.get(startLayer).length;
	}

	public int numBiases(int startLayer) {
		return biases.get(startLayer).length;
	}

	// Returns a copy of the biases
	public double[] getBiases(int startLayer) {
		return biases.get(startLayer);
	}

	// Returns the value of the weight between node1 in layer and node2 in
	// layer+1
	public double getWeight(int layer, int node1, int node2) {
		return getWeights(layer)[node1 * layerSize(layer + 1) + node2];
	}

	private double nodeValue(int layer, int node) {
		return getLayer(layer)[node];
	}

	private double nodeValueTraining(int layer, int node) {
		return cacheLayer.get(layer)[node];
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
		return layers.get(i);
	}

	public double[] getWeights(int startLayer) {
		return weights.get(startLayer);
	}

	public int inputSize() {
		return inputSize;
	}

	public int outputSize() {
		return outputSize;
	}

	public int numLayers() {
		return layers.size();
	}

	public double[] inputLayer() {
		return getLayer(1);
	}

	public double[] outputLayer() {
		return getLayer(numLayers() - 1);
	}

	public double[] trainingOutput() {
		return trainingLayers.get(numLayers() - 1);
	}

	/*
	 * @Override public Iterator<int[]> iterator() {
	 * 
	 * }
	 * 
	 * private class weightIterator implements Iterator {
	 * 
	 * @Override public boolean hasNext() { // TODO Auto-generated method stub
	 * return false; }
	 * 
	 * @Override public Object next() { // TODO Auto-generated method stub return
	 * null; }
	 * 
	 * }
	 */

}
