package com.NerualNetwork;

import java.util.Random;

import static java.lang.Math.exp;


public class Node {
    static private final Random r = new Random();
    static private final double E = Math.E;
    static private final double MAX_WEIGHT_DIFF = 1.0;
    static private final double WEIGHT_SIG_FIG = 10000;



    private double value = 0.0;
    private double preTransferValue = 0.0;
    private double[] weights;
    private double bias = 0.0;
    private double error = -.1;
    private Node[] parents;
    private NeuralNetwork parent;

    Node(NeuralNetwork p){
        parent = p;
        parents = new Node[0];
        weights = new double[0];
        //biases = new double[0];

    }

    double[] getInitialArrayFor(int len){
        double[] out = new double[len];
        for(int i = 0 ; i < len;i++){
            out[i] = ((double)r.nextInt((int)MAX_WEIGHT_DIFF*(int)WEIGHT_SIG_FIG)/WEIGHT_SIG_FIG)-(MAX_WEIGHT_DIFF/2.0);
        }
        return out;
    }

    void print(){
        System.out.println("Node   -  "+this+"  - :");
        System.out.println("Weights: "+weights.toString());
        //System.out.println("Biases: "+biases.toString());

    }

    double sig(double t){
        return (1/( 1 + Math.pow(E,(-1*t))));
    }

    double sigDiv(double t){
        return t * (1.0-t);
    }

    Node(Node[] p, NeuralNetwork par){
        parent = par;
        parents = p;
        weights = getInitialArrayFor(p.length);
        //biases = getInitialArrayFor(p.length);
    }

    void setValue(double val){
        value = sig(val);
    }

    double getValue(){return value;}

    double getWeightedValue(double weight){

        return value*weight;

    }

    boolean findValue(){

        value = 0.0;
        for(int i = 0 ; i < parents.length;i++){
            try {

                value += parents[i].getWeightedValue(weights[i]);
            }catch(Exception e){
                e.printStackTrace();
                return false;
            }
        }
        value+=bias;
        preTransferValue = value;
        value = sig(value);
        return true;
    }

    void setErrorAsOutput(double correct){

        error = (correct - value) * sigDiv(value);

    }

    void setErrorAsMiddle(){

        error = 0.0;
        for(int i = 0 ;i < parents.length;i++){

            error+= (parents[i].error * weights[i]) * sigDiv(value);

        }


    }

    void applyError(){

        for(int i = 0 ;i < weights.length;i++) {

            weights[i] += parent.LEARNING_RATE * parents[i].error * parents[i].value;

        }

    }

}
