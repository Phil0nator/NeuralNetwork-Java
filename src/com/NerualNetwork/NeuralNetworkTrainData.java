package com.NerualNetwork;

class TooMuchDataException extends RuntimeException{

    TooMuchDataException(String msg){
        super(msg);
    }

}


public class NeuralNetworkTrainData {

    public double[][] input;
    public double[][] correct;
    private int inpCounter = 0;
    private int correctCounter = 0;

    NeuralNetworkTrainData(int length){

        input = new double[length][];
        correct = new double[length][];

    }

    void push(double[] inp, double[] result){
        input[inpCounter]=inp;
        correct[correctCounter]=result;
        inpCounter++;
        correctCounter++;

        if(inpCounter > input.length||correctCounter>correct.length){
            throw new TooMuchDataException("You've tried to add more data than you've allocated");
        }

    }

}
