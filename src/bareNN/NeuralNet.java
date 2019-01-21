package bareNN;

import java.io.BufferedReader;
import java.io.DataOutput;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import javax.sql.rowset.Joinable;

import io.Input;
import io.Output;

public class NeuralNet {

    private BlackBox blackBox;
    private double[][] trainingData;
    private double[][] trainingAnswers;
    private double[][] testData;
    private double[][] testAnswers;
    private double[] descent; // Gradient descent vector
    private double delta = 0.1; // Step size in gradient descent
    private int inputSize; // size of input layer
    private int outputSize; // size of output layer
    private int offset = 0;
    private int numExamples = 1000; // Number of examples from the training data that will be used to train
    private int numThreads = 3;
    private final String savePath;
    private String trainingPath = "saves/digit/Data/mnist_train.csv";
    private String testPath = "saves/digit/Data/mnist_test.csv";

    public NeuralNet(String savePath, int[] layers) {
        this.savePath = savePath + "/savedNet";
        // int[] params = { 4, 5 };
        blackBox = new BlackBox(layers, true);
        inputSize = blackBox.inputSize();
        outputSize = blackBox.outputSize();
        getTrainingData();
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
        getTrainingData();
    }

    private void getTrainingData() {
        Input in = new Input(new File(trainingPath));
        Input testIn = new Input(new File(testPath));

        // int numExamples = 600;// (int) metaData[0]; // Number of training examples
        in.readLine();
        trainingData = new double[numExamples][];
        trainingAnswers = new double[numExamples][];
        testData = new double[(int) testIn.readLine()[0]][];
        System.out.println(testData.length);
        testAnswers = new double[testData.length][];
        for (int i = 0; i < offset; i++)
            in.readLine();
        for (int i = 0; i < numExamples; i++) {
            double[] line = in.readLine();
            trainingAnswers[i] = new double[outputSize];
            trainingAnswers[i][(int) line[0]] = 1;
            trainingData[i] = new double[inputSize];
            for (int j = 1; j < inputSize; j++) {
                trainingData[i][j] = line[j] / 255;
            }
        }
        for (int i = 0; i < testData.length; i++) {
            double[] line = testIn.readLine();
            testAnswers[i] = new double[outputSize];
            testAnswers[i][(int) line[0]] = 1;
            testData[i] = new double[inputSize];
            for (int j = 1; j < inputSize; j++) {
                testData[i][j] = line[j] / 255;
            }
        }
    }

    public double[] eval(double[] input, boolean smooth) {
        assert input.length == inputSize;
        if (!smooth)
            return blackBox.eval(input);
        else
            return (smoothOutput(blackBox.eval(input)));
    }

    /**
     * Takes an array of inputs and returns an array of the indices of the highest output node for each node.
     * 
     * @param inputs
     *            A two dimensional double array, with the ith array being the ith input. The first element of each
     *            input array is assumed to be the answer for the input.
     * @return An integer array whose ith element corresponds to the highest confidence prediction for the ith input.
     */
    public int[] classify(double[][] inputs) {
        int[] classifications = new int[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            double[] output = eval(Arrays.copyOfRange(inputs[i], 1, inputSize + 1), false);
            classifications[i] = indexOfMax(output);
        }
        return classifications;
    }

    public int classify(double[] input) {
        double[][] newInput = { input };
        return classify(newInput)[0];
    }

    public double cost() {
        return cost(blackBox);
    }

    private double cost(BlackBox box) {
        assert trainingAnswers.length == trainingData.length;
        double cost = 0;
        for (int i = 0; i < trainingData.length; i++) {
            cost += sqError(box.eval(trainingData[i]), trainingAnswers[i]) / trainingData.length;
        }
        return cost;
    }

    public double testCost() {
        double cost = 0;
        for (int i = 0; i < testData.length; i++) {
            cost += sqError(blackBox.eval(testData[i]), testAnswers[i]) / testData.length;
        }
        return cost;
    }

    public void backPropagation(int iterations, double[] learningRate, boolean saveBetweenLayers) {
        for (int i = 0; i < iterations; i++) {
            double initialCost = cost();
            for (int layer = blackBox.numLayers() - 2; layer >= 0; layer--) {
                blackBox.adjust(layer, normalize(gradient(layer), learningRate[layer]));
                double cost = cost();
                System.out.println("Iteration " + i + " has a cost of " + cost);
                if (cost < initialCost)
                    blackBox.save(savePath);
            }
        }
    }

    // returns array with every element negative
    private double[] negative(double[] input) {
        for (int i = 0; i < input.length; i++)
            input[i] = -1 * input[i];
        return input;
    }

    public void save() {
        blackBox.save(savePath);
    }

    public void save(String altSavePath) {
        blackBox.save(altSavePath);
    }

    // Sets the highest value to 1, all others to 0
    private double[] smoothOutput(double[] output) {
        int indexOfMax = indexOfMax(output);
        for (int i = 0; i < output.length; i++)
            output[i] = i == indexOfMax ? 1 : 0;
        return output;
    }

    // Returns a copy of the given vector rescaled to the given magnitude
    private double[] normalize(double[] vector, double magnitude) {
        double originalMag = 0;
        double[] output = new double[vector.length];
        for (int i = 0; i < vector.length; i++)
            originalMag += vector[i] * vector[i];
        originalMag = Math.sqrt(originalMag);
        if (originalMag == 0) {
            System.out.println("NO CHANGE");
            return vector;
        }
        for (int i = 0; i < vector.length; i++) {
            output[i] = vector[i] / originalMag * magnitude;
        }
        return output;
    }

    private int indexOfMax(double[] input) {
        int highest = 0;
        for (int i = 0; i < input.length; i++)
            if (input[i] > input[highest])
                highest = i;
        return highest;
    }

    private int numNonZero(double[] input) {
        int numNonZero = 0;
        for (double d : input)
            if (d != 0)
                numNonZero++;
        return numNonZero;
    }

    private double[] gradient(int layer) {
        double[] gradient = new double[blackBox.numWeights(layer) + blackBox.numBiases(layer)];
        for (int i = 0; i < numExamples; i++) {
            blackBox.eval(trainingData[i]);
            double[] exampleGradient = blackBox.weightDerivatives(trainingAnswers[i], 0);
            for (int j = 0; j < exampleGradient.length; j++) {
                if (exampleGradient[j] != 0)
                    gradient[j] += exampleGradient[j] / numExamples;
            }
        }
        return gradient;
    }

    private double[] gradientDescent(int layer) {
        double initialCost = cost();
        // sign indicates direction to move variable, magnitude is how much it lowers
        // cost.

        descent = new double[blackBox.getWeights(layer).length + blackBox.getBiases(layer).length];
        System.out.println("Starting descent on layer " + layer + " with " + descent.length + " weights");
        Thread[] threads = new Thread[numThreads];
        for (int i = 0; i < numThreads; i++) {
            threads[i] = new Thread(new descentThread(layer, i * descent.length / numThreads,
                    (i + 1) * descent.length / numThreads, initialCost, i));
        }

        for (Thread t : threads)
            t.start();

        for (Thread t : threads)
            try {
                t.join();
            } catch (InterruptedException e1) {
                e1.printStackTrace();
            }

        /*
         * Thread t1 = new Thread(new descentThread(layer, 0, descent.length / 4, initialCost, 1)); Thread t2 = new
         * Thread(new descentThread(layer, descent.length / 4, descent.length / 2, initialCost, 2)); Thread t3 = new
         * Thread(new descentThread(layer, descent.length / 2, 3 * descent.length / 4, initialCost, 3)); Thread t4 = new
         * Thread(new descentThread(layer, 3 * descent.length / 4, descent.length, initialCost, 4)); t1.start();
         * t2.start(); t3.start(); t4.start();
         * 
         * try { t1.join(); t2.join(); t3.join(); t4.join(); } catch (InterruptedException e) { e.printStackTrace(); }
         */
        System.out.println("There were " + numNonZero(descent) + " non-zero changes");
        return descent;

    }

    public String getSaveLocation() {
        return savePath;
    }

    public static double sqError(double[] outputs, double[] answer) {
        double sqError = 0;
        for (int i = 0; i < outputs.length; i++) {
            assert outputs[i] != Double.NaN;
            sqError += (outputs[i] - answer[i]) * (outputs[i] - answer[i]);
        }
        assert sqError != Double.NaN;
        return sqError;
    }

    private class descentThread implements Runnable {
        BlackBox tempBox;
        int layer;
        int index;
        int startIndex;
        int endIndex;
        int threadNumber;
        double initialCost;

        // startIndex inclusive, endIndex exclusive
        public descentThread(int layer, int startIndex, int endIndex, double initialCost, int threadNumber) {
            tempBox = new BlackBox(savePath, blackBox.numLayers());
            this.layer = layer;
            this.index = startIndex;
            this.startIndex = startIndex;
            this.endIndex = endIndex;
            this.threadNumber = threadNumber;
            this.initialCost = initialCost;
        }

        public void run() {
            for (int i = 0; i < trainingData.length; i++) {
                tempBox.eval(trainingData[i]);
                for (int j = 0; j < descent.length; j++) {
                    // tempBox.derivative(layer, j);
                }
            }
        }

    }

    /*
     * public static void main(String[] args) { int[] layers = { 4, 5, 5}; NeuralNet net = new
     * NeuralNet("saves/3layer/", layers); //NeuralNet net = new NeuralNet("saves/3layer/savedNet.txt", 2);
     * net.backPropagation(200); //net.save(); System.out.println(net.cost());
     * System.out.println(net.getSaveLocation()); double[] test = { 1, 0, 0, 1 };
     * System.out.println(Arrays.toString(net.apply(test))); }
     */
}
