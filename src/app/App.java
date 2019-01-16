package app;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import bareNN.NeuralNet;
import io.ImageReader;
import io.Input;

public class App {

    static NeuralNet net;

    public static void main(String[] args) {
        // int[] layers = { 784, 20, 10 };
        // NeuralNet net = new NeuralNet("saves/digit", layers);
        net = new NeuralNet("saves/digit/savedNet", 3);
        System.out.println(net.cost());
        net.save();
        trainNet();
        //test(10000);
        /*
         * Input in = new Input(new File("saves/digit/Data/mnist_test.csv"));
         * in.readLine(); try { for (int i = 0; i < 30; i++) ImageReader.writeImage(28,
         * 28, Arrays.copyOfRange(in.readLineInt(), 1, 785), "saves/digit/images/test" +
         * i + ".png"); } catch (IOException e) { // TODO Auto-generated catch block
         * e.printStackTrace(); }
         */
    }

    private static void trainNet() {
        double[] stepSizes = { 1, 0.1 };
        for (int i = 0; i < 3; i++) {
            double initial = net.cost();
            net.backPropagation(1, stepSizes);
            if (net.cost() > initial)
                break;
            else
                net.save();
            test(10000);
        }
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
        // for (int i = 0; i < predictions.length; i++)
        // System.out.println("The computer predicted " + predictions[i] + " while the
        // real answer was " + answers[i]);
        int numCorrect = 0;
        for (int i = 0; i < predictions.length; i++)
            if (predictions[i] == answers[i])
                numCorrect++;
        System.out.println("The computer predicted " + numCorrect + " out of " + numTest + " correctly");
    }

}
