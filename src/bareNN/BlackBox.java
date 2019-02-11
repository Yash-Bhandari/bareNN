package bareNN;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import io.Input;
import io.Output;

public class BlackBox {

    private ArrayList<double[]> layers;
    private ArrayList<double[]> weights;
    private ArrayList<double[]> biases;

    public BlackBox(int[] addLayers, boolean populate) {
        layers = new ArrayList<double[]>();
        weights = new ArrayList<double[]>();
        biases = new ArrayList<double[]>();
        for (int i = 0; i < addLayers.length; i++)
            addLayer(addLayers[i]);
        build(populate);
    }

    public BlackBox(String path, int numLayers) {
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
        build(false);
    }

    // To be called after layers have been added
    public void build(boolean populate) {
        if (populate)
            populate();
    }

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

    public double[] eval(double[] input) {
        assert input.length == inputSize();
        clearLayers();
        setLayer(0, input);
        for (int i = 0; i < numLayers() - 1; i++)
            evalLayer(i);
        // for (double d : outputLayer())
        // assert !Double.isNaN(d);
        double sum = 0;
        for (double d : outputLayer())
            sum += d;
        return outputLayer();
    }

    public void evalLayer(int layer) {
        for (int i = 0; i < getLayer(layer).length; i++)
            evalNode(layer, i);
        addBiases(layer);
        for (double d : getLayer(layer + 1))
            assert !Double.isNaN(d);

        for (int i = 0; i < getLayer(layer + 1).length; i++)
            setNode(layer + 1, i, sigmoid(nodeValue(layer + 1, i)));

        /*
         * if (layer == numLayers() - 2) setLayer(layer + 1, softMax(getLayer(layer + 1))); // Final transformation of
         * answer
         */
    }

    public void evalNode(int layer, int node) {
        assert layer < numLayers() - 1;
        for (int i = 0; i < layerSize(layer + 1); i++) {
            addToNode(layer + 1, i, getWeight(layer, node, i) * nodeValue(layer, node));
        }
    }

    // Adds biases from the given layer to layer+1
    private void addBiases(int layer) {
        for (int i = 0; i < numBiases(layer); i++) {
            addToNode(layer + 1, i, getBiases(layer)[i]);
            assert !Double.isNaN(nodeValue(layer + 1, i));
        }
    }

    public void adjust(int layer, double[] adjustments) {
        assert adjustments.length == weights.get(layer).length + biases.get(layer).length;
        for (double d : adjustments)
            assert !Double.isNaN(d);
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

    private void setNode(int layer, int node, double value) {
        getLayer(layer)[node] = value;
    }

    public void addToWeight(int startLayer, int node1, int node2, double delta) {
        setWeight(startLayer, node1, node2, getWeight(startLayer, node1, node2) + delta);
    }

    // Adds to weight/bias at the given index
    public void addToWeight(int startLayer, int weightIndex, double delta) {
        if (weightIndex < numWeights(startLayer))
            weights.get(startLayer)[weightIndex] += delta;
        else
            biases.get(startLayer)[weightIndex % numWeights(startLayer)] += delta;
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
                    setWeight(i, j, k, r.nextDouble() / 20);
                setBias(i, k, r.nextDouble() / 20);
            }
        }
    }

    public double[][] getGradient(double[] answers) {
        return dCostWrtWeights(answers, dCostWrtNodes(answers));
    }

    // Returns the partial derivatives of the cost w.r.t. the pre-activation values of each node in the network.
    // The i,jth index corresponds to the derivative of the cost w.r.t. the jth node in the ith layer
    private double[][] dCostWrtNodes(double[] answers) {
        double[][] dCostWrtNodes = new double[numLayers()][];
        dCostWrtNodes[numLayers()-1] = new double[outputSize()];

        for (int i = 0; i < dCostWrtNodes[numLayers()-1].length; i++)
            dCostWrtNodes[numLayers() - 1][i] = 2 * (answers[i] - outputLayer()[i]) * sigmoidPrime(outputLayer()[i]);

        for (int layer = numLayers()-2; layer >= 1; layer--)
            dCostWrtNodes[layer] = dCostWrtLayer(layer, dCostWrtNodes[layer+1]);
        
        return dCostWrtNodes;
    }

    // Returns the partial derivatives of the cost w.r.t. the pre-activation values of each node in the startLayer
    private double[] dCostWrtLayer(int startLayer, double[] dCostWrtEndLayer) {
        double[] output = new double[layerSize(startLayer)];
        for (int startNode = 0; startNode < output.length; startNode++) {
            double startNodeValue = nodeValue(startLayer, startNode);
            for (int endNode = 0; endNode < layerSize(startLayer + 1); endNode++)
                output[startNode] += sigmoidPrime(startNodeValue) * getWeight(startLayer, startNode, endNode);
        }
        return output;
    }

    // Returns matrix where the i,jth index corresponds to the partial derivative of the cost w.r.t. the jth weight
    // connecting from the ith layer to the i+1th layer.
    private double[][] dCostWrtWeights(double[] answers, double[][] dCostWrtNodes) {
        double[][] output = new double[numLayers() - 1][];
        for (int i = 0; i < output.length; i++) {
            output[i] = new double[numWeights(i) + numBiases(i)];
            for (int j = 0; j < output[i].length; j++) {
                if (j < numWeights(i))
                    output[i][j] = nodeValue(i, startNode(i, j)) * dCostWrtNodes[i+1][endNode(i, j)];
                else 
                    output[i][j] = dCostWrtNodes[i+1][endNode(i, j)];
            }
        }
        return output;
    }

    // Returns the partial derivatives of the cost with respect to all of the weights connecting from startLayer to
    // startLayer+1 for the input that was last evaluated
    public double[] weightDerivatives(double[] answers, int startLayer) {
        double[][] dSoftMax = softMaxPrime();

        double[] derivatives = new double[numWeights(startLayer) + numBiases(startLayer)];
        for (int i = 0; i < derivatives.length; i++) {
            derivatives[i] = dCostWrtWeight(answers, startLayer, i);
        }
        return derivatives;

    }

    // Derivative of the total cost with respect to a certain weight
    public double dCostWrtWeight(double[] answers, int weightLayer, int weightIndex) {
        assert answers.length == outputSize();
        double derivative = 0;
        for (int i = 0; i < outputSize(); i++) {
            derivative += dOutputCostWrtWeight(i, answers[i], weightLayer, weightIndex);
        }
        return derivative;
    }

    // Derivative of the cost of a single output node with respect to a certain weight
    private double dOutputCostWrtWeight(int outputNode, double answer, int weightLayer, int weightIndex) {
        double dCostWrtNode = 2 * (answer - outputLayer()[outputNode]);
        double dNodeWrtWeight = dNodeWrtWeight(numLayers() - 1, outputNode, weightLayer, weightIndex, true);
        return dCostWrtNode * dNodeWrtWeight;
    }

    // Derivative of the activation or raw value of a node with respect to a certain weight
    private double dNodeWrtWeight(int nodeLayer, int nodeIndex, int weightLayer, int weightIndex, boolean activated) {
        double nodeValue = nodeValue(nodeLayer, nodeIndex);
        if (nodeLayer == weightLayer + 1) {
            if (endNode(nodeLayer - 1, weightIndex) != nodeIndex)
                return 0;
            if (weightIndex >= numWeights(weightLayer)) // Bias
                return activated ? sigmoidPrime(nodeValue) : nodeValue;
            if (activated)
                return sigmoidPrime(nodeValue) * nodeValue(nodeLayer - 1, startNode(nodeLayer - 1, weightIndex)); // Weight
            else
                return nodeValue(nodeLayer - 1, startNode(nodeLayer - 1, weightIndex));
        } else {
            double derivative = activated ? sigmoidPrime(nodeValue(nodeLayer, nodeIndex)) : 1;
            assert 2 == 1;
            return Double.NaN; // Placeholder
        }
    }

    // Derivative of the activation value of one node with respect to the activation value of another
    private double dNode1WrtNode2(int node1, int layer1, int node2, int layer2) {
        if (layer1 == layer2 + 1)
            return sigmoidPrime(nodeValue(layer1, node1)) * getWeight(layer2, node2, node1);
        else
            return 999999999; // Placeholder
    }

    private double dCostWrtNode(int node, int layer, double[] priorDerivatives) {
        return 4;
    }

    // Returns an 2d array where the i,j index is the partial derivative of the activation value
    // of the ith output node with respect to the raw value of the jth output node
    private double[][] softMaxPrime() {
        double[][] derivatives = new double[outputSize()][outputSize()];
        for (int i = 0; i < derivatives.length; i++) {
            for (int j = 0; j < derivatives.length; j++) {
                double delta = (i == j) ? 1 : 0;
                derivatives[i][j] = outputLayer()[i] * (delta - outputLayer()[j]);
            }
        }
        return derivatives;
    }

    private double sigmoid(double input) {
        return 1 / (1 + Math.exp(-1 * input));
    }

    /**
     * Returns the derivative of f(x), where f is the sigmoid function
     * 
     * @param input
     *            f(x), not x
     * @return f'(x)
     */
    private double sigmoidPrime(double input) {
        return input - input * input;
    }

    private double[] softMax(double[] input) {
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
    public double getWeight(int layer, int startNode, int endNode) {
        return getWeights(layer)[startNode * layerSize(layer + 1) + endNode];
    }

    private double nodeValue(int layer, int node) {
        return getLayer(layer)[node];
    }

    private int layerSize(int layer) {
        return getLayer(layer).length;
    }

    // Returns the starting node corresponding to a given weight
    private int startNode(int startLayer, int weightIndex) {
        return weightIndex / layerSize(startLayer + 1);
    }

    // Returns the ending node correspoinding to a given weight
    private int endNode(int startLayer, int weightIndex) {
        return weightIndex % layerSize(startLayer + 1);
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
        return inputLayer().length;
    }

    public int outputSize() {
        return outputLayer().length;
    }

    public int numLayers() {
        return layers.size();
    }

    public double[] inputLayer() {
        return getLayer(0);
    }

    public double[] outputLayer() {
        return getLayer(numLayers() - 1);
    }

    /*
     * @Override public Iterator<int[]> iterator() {
     * 
     * }
     * 
     * private class weightIterator implements Iterator {
     * 
     * @Override public boolean hasNext() { // TODO Auto-generated method stub return false; }
     * 
     * @Override public Object next() { // TODO Auto-generated method stub return null; }
     * 
     * }
     */

}
