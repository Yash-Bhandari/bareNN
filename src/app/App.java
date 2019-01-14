package app;

import bareNN.NeuralNet;

public class App {
    public static void main(String[] args) {
        //int[] layers = { 784, 20, 10 };
        //NeuralNet net = new NeuralNet("saves/digit", layers);
        NeuralNet net = new NeuralNet("saves/digit/savedNet", 3);
        System.out.println(net.cost());
        net.save();
        net.backPropagation(2, 2);
        //net.apply(new double[784]);
        net.save();
        System.out.println("starting");
        System.out.println(net.cost());
    }
}
