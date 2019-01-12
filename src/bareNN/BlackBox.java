package bareNN;

import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

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

    private int inputSize; // size of input layer
    private int outputSize; // size of output layer
    private int[] iterating = new int[3];

    public BlackBox(int[] addLayers, boolean populate) {
        layers = new ArrayList<double[]>();
        weights = new ArrayList<double[]>();
        biases = new ArrayList<double[]>();
        for (int i = 0; i < addLayers.length; i++)
            addLayer(addLayers[i]);
        build(true);
    }

    public BlackBox(String path, int numLayers) {
        this.path = path;
        Input in = new Input(new File(path));
        layers = new ArrayList<double[]>();
        weights = new ArrayList<double[]>();
        biases = new ArrayList<double[]>();
        double[] line;
        layers.add(in.readLine());
        for (int i = 1; i < numLayers; i++) {
            weights.add(in.readLine());
            biases.add(in.readLine());
            layers.add(in.readLine());
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

    private void addLayer(int size) {
        if (numLayers() > 0) // Adds array representing the weights between previous layer and new layer.
            weights.add(new double[size * layerSize(numLayers() - 1)]);
        layers.add(new double[size]);
        biases.add(new double[size]);
    }

    private void setLayer(int layer, double[] newLayer) {
        layers.set(layer, newLayer);
    }

    public void clearLayers() {
        for (int i = 0; i < numLayers(); i++)
            setLayer(i, new double[getLayer(i).length]);
    }

    public double[] eval(double[] input) {
        assert input.length == inputSize;
        Arrays.copyOf(input, inputSize);
        clearLayers();
        setLayer(0, input);
        for (int i = 0; i < numLayers() - 1; i++)
            evalLayer(i);
        return outputLayer();
    }

    public void evalLayer(int layer) {
        for (int i = 0; i < getLayer(layer).length; i++)
            evalNode(layer, i);
        addBiases(layer);
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

    private void addToNode(int layer, int node, double delta) {
        assert layer < numLayers();
        getLayer(layer)[node] += delta;
    }

    private void setNode(int layer, int node, double value) {
        getLayer(layer)[node] = value;
    }

    void addWeight(int startLayer, int node1, int node2, double delta) {
        setWeight(startLayer, node1, node2, getWeight(startLayer, node1, node2) + delta);
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

    private static double[] softMax(double[] input) {
        double[] out = new double[input.length];
        double divisor = 0;
        for (double d : input)
            divisor += Math.exp(d);
        for (int i = 0; i < input.length; i++) {
            out[i] = Math.exp(input[i]) / divisor;
        }
        return out;
    }

    private void setWeight(int startLayer, int node1, int node2, double value) {
        weights.get(startLayer)[node1 * layerSize(startLayer + 1) + node2] = value;
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

    private int layerSize(int layer) {
        return getLayer(layer).length;
    }

    /**
     * Returns the ith layer. The input layer is layer 0.
     * 
     * @param i
     *            The index of the layer
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

}
