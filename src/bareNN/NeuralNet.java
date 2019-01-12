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
        blackBox = new BlackBox("src/io/savedNet.txt", 2);
        //int[] params = {4, 4};
        //blackBox = new BlackBox(params);
        getTrainingData(new Input(new File("src/io/training.txt")));
        //System.out.println(cost());
        blackBox.clearLayers();
        blackBox.save(savePath);
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

    private void backPropagation() {
        double initialCost = cost();
        double delta = 0.1;
        double[] weightDerivatives = new double[blackBox.getWeights(0).length];
        double[] biasDerivatives = new double[blackBox.getBiases(0).length];
        

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
