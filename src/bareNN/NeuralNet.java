package bareNN;

import java.io.BufferedReader;
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
    private int inputSize; // size of input layer
    private int outputSize; // size of output layer
    private String savePath = "src/io/savedNet";

    public NeuralNet() {
        //blackBox = new BlackBox("src/io/savedNet.txt", 2);
        int[] params = { 4, 4 };
        blackBox = new BlackBox(params, false);
        getTrainingData(new Input(new File("src/io/training.txt")));
        System.out.println(cost());
        // blackBox.clearLayers();
        blackBox.save(savePath);
        backPropagation(30);
        blackBox.save(savePath+"1");
    }

    private void getTrainingData(Input in) {
        double[] metaData = in.readLine();
        int numExamples = (int) metaData[0]; // Number of training examples
        inputSize = (int) metaData[1];
        outputSize = (int) metaData[2];
        trainingData = new double[numExamples][];
        trainingAnswers = new double[numExamples][];

        for (int i = 0; i < numExamples; i++) {
            double[] line = in.readLine();
            trainingData[i] = Arrays.copyOfRange(line, 0, inputSize);
            trainingAnswers[i] = Arrays.copyOfRange(line, inputSize, inputSize + outputSize);
        }
    }

    private double cost() {
        assert trainingAnswers.length == trainingData.length;
        double cost = 0;
        for (int i = 0; i < trainingData.length; i++) {
            double singleCost = sqError(blackBox.eval(trainingData[i]), trainingAnswers[i]) / trainingData.length;
            cost += singleCost;
        }
        return cost;
    }

    private void backPropagation(int iterations) {
        for (int i = 0; i < iterations; i++) {
            double[] normalizedDescent = normalize(descent(), 0.1);
            blackBox.adjust(0, normalizedDescent);
            System.out.println("Iteration " + i + " has a cost of " + cost());
            blackBox.save("src/io/descent/iteration" + i + ".txt");
        }
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

    private double[] descent() {
        double initialCost = cost();
        double delta = 0.1;
        int layer = 0;
        // sign indicates direction to move variable, magnitude is how much it lowers
        // cost.
        double[] descent = new double[blackBox.getWeights(layer).length + blackBox.getBiases(layer).length];

        for (int i = layer; i < blackBox.getWeights(layer).length; i++) {
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
            }
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
            }
        }
        return descent;
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
