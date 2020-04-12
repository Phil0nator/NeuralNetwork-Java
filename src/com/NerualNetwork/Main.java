package com.NerualNetwork;

public class Main {

    public static void printArray(double[] d){
        System.out.print("[");
        System.out.print(d[0]);
        for(int i = 1 ; i < d.length;i++){
            System.out.print(", "+d[i]);
        }
        System.out.println("]");
    }

    public static void main(String[] args) {
	// write your code here

        int[] config = {5,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2};
        NeuralNetwork ntest = new NeuralNetwork(10,1,config);
        double[] testData = {0.0,1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9};
        double[] testCorrect = {5.5};

        NeuralNetworkTrainData testtrain = new NeuralNetworkTrainData(100);
        for( int i = 0 ; i < 100; i ++){
            testtrain.push(testData,testCorrect);
        }
        for( int i = 0 ; i < 10000; i ++){
            ntest.train(testtrain);
        }

        printArray(ntest.predict(testData));
        ntest.saveTo("MyNetwork.nns");
    }
}

