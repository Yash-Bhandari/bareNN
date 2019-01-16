package app;

import java.io.File;
import java.util.Arrays;

import bareNN.NeuralNet;
import io.Input;

public class App {

    static NeuralNet net;

    public static void main(String[] args) {
        // int[] layers = { 784, 20, 10 };
        // NeuralNet net = new NeuralNet("saves/digit", layers);
        net = new NeuralNet("saves/digit/savedNet", 3);
        System.out.println(net.cost());
        net.save();
        //trainNet();
        test(10000);
        
    }

    private static void trainNet() {
        double[] stepSizes = { 0.1, 0.1 };
        net.backPropagation(2, stepSizes);
        // net.apply(new double[784]);
        net.save();
        System.out.println(net.cost());
    }

    private static void test(int numTest) {
        Input in = new Input(new File("saves/digit/Data/mnist_test.csv"));
        in.readLine();
        double[][] test = new double[numTest][];
        int[] answers = new int[numTest];
        for (int i = 0; i < numTest; i++) {
            test[i] = in.readLine();
            answers[i] = (int) test[i][0];
        }
        int[] predictions = net.classify(test);
        for (int i = 0; i < predictions.length; i++)
            System.out.println("The computer predicted " + predictions[i] + " while the real answer was " + answers[i]);
        int numCorrect = 0;
        for (int i = 0; i < predictions.length; i++)
            if (predictions[i] == answers[i])
                numCorrect++;
        System.out.println("The computer predicted " + numCorrect + " out of " + numTest + " correctly");
    }

}
