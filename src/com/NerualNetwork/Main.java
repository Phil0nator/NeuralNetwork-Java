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

        int[] config = {2};
        NeuralNetwork test = new NeuralNetwork(3,1,config);
        double[] t1 = {0,0,1};
        double[] t2 = {1,1,1};
        double[] t3 = {1,0,1};
        double[] t4 = {0,1,1};
        double[] c1 = {0};
        double[] c2 = {1};
        double[] c3 = {1};
        double[] c4 = {0};

        double[] testData = {1,1,1};
        NeuralNetworkTrainData data = new NeuralNetworkTrainData(4);
        data.push(t1,c1);
        data.push(t2,c2);
        data.push(t3,c3);
        data.push(t4,c4);
        for(int i = 0 ; i < 100000;i++){
            //System.out.println("TRAIN: ");
            test.train(data);
        }
        printArray(test.predict(testData));

    }

}

