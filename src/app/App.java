package app;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import org.ejml.simple.SimpleMatrix;

import bareNN.NeuralNet;
import io.ImageReader;
import io.Input;

public class App {

	static NeuralNet net;

	public static void main(String[] args) throws Exception {
		// int[] layers = { 784, 30, 10 };
		// net = new NeuralNet("saves/digit", layers);
		net = new NeuralNet("saves/digit/savedNet", 3);
		System.out.println(net.cost());
		//net.save();
		//testMatrix();
		// double[] image = ImageReader.readImage("saves/digit/images/seven.png");
		// System.out.println(Arrays.toString(image));
		// System.out.println("The picture is of a " + net.classify(image));

		trainNet();
		// test(0, true);
		/*
		 * Input in = new Input(new File("saves/digit/Data/mnist_test.csv"));
		 * in.readLine(); try { for (int i = 0; i < 30; i++) ImageReader.writeImage(28,
		 * 28, Arrays.copyOfRange(in.readLineInt(), 1, 785), "saves/digit/images/test" +
		 * i + ".png"); } catch (IOException e) { // TODO Auto-generated catch block
		 * e.printStackTrace(); }
		 */
	}

	private static void trainNet() {
		int oldCorrect = test(10000, false);
		double[] stepSizes = { 0.3, 0.2 };
		for (int i = 0; i < 3; i++) {
			double initial = net.cost();
			net.backPropagation(1, stepSizes, true);
			double newCost = net.cost();
			int correct = test(10000, false);
			if (Double.isNaN(newCost) || newCost > initial || correct < oldCorrect)
				break;
			else
				net.save();
		}
		// net.apply(new double[784]);
		// net.save();
		System.out.println(net.cost());
	}

	private static int test(int numTest, boolean print) {
		Input in = new Input(new File("saves/digit/Data/mnist_test.csv"));
		in.readLine();
		double[][] test = new double[numTest][];
		int[] answers = new int[numTest];
		for (int i = 0; i < numTest; i++) {
			test[i] = in.readLine();
			answers[i] = (int) test[i][0];
		}
		int[] predictions = net.classify(test);
		if (print)
			for (int i = 0; i < predictions.length; i++)
				System.out.println("The computer predicted that the " + i + "th digit was " + predictions[i]
						+ " while the real answer was " + answers[i]);
		int numCorrect = 0;
		for (int i = 0; i < predictions.length; i++)
			if (predictions[i] == answers[i])
				numCorrect++;
		System.out.println("The computer predicted " + numCorrect + " out of " + numTest + " correctly");
		return numCorrect;
	}

	private static void testMatrix() {
		double[] vector = { 1, 3, 5, 1 };
		double data[][] = new double[4][4];
		for (int i = 0; i < 4; i++)
			data[i][i] = 1;
		SimpleMatrix s = new SimpleMatrix(data);
		SimpleMatrix v = new SimpleMatrix(4, 1, false, vector);
		System.out.println(s);
	}

}
