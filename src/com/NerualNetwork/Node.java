package com.NerualNetwork;

import java.util.Random;

import static java.lang.Math.exp;

/**
 * Node
 * Contains the value and weight of a node
 */
public class Node {
    /**
     * A java.util.Random object to determine initial weight values
     */
    static private final Random r = new Random();
    /**
     * Constant value for mathematical e
     * @value
     */
    static private final double E = Math.E;
    /**
     * Constant value to define the maximum weight distance from 0
     * EX: 1.0   ->     -1.0 < x < 1.0
     */
    static private final double MAX_WEIGHT_DIFF = 1.0;
    /**
     * Constant to determine the number of decimals to be randomized in the initial weight values
     */
    static private final double WEIGHT_SIG_FIG = 10000;


    /**
     * The value contained by the node. (Commonly shown as x or a in equations)
     */
    private double value = 0.0;
    /**
     * The value before it is run through sigmoid
     * @deprecated
     */
    private double preTransferValue = 0.0;
    /**
     * Double array containing the weights between this node, and every one of its parent nodes.
     * Commonly shown as W in equations
     */
    private double[] weights;
    /**
     * The bias of this node
     * Commonly shown as b in equations
     */
    private double bias = 0.0;
    /**
     * The error of this node
     * Commonly shown as E in equations
     */
    private double error = -.1;
    /**
     * An array of the parent nodes for this node
     */
    private Node[] parents;
    /**
     * A reference to the NeuralNetwork object to which this node belongs
     */
    private NeuralNetwork parent;

    /**
     * Constructor 1:
     * @param p The parent network
     *
     * This constructor is only used for nodes in the input layer.
     */
    Node(NeuralNetwork p){
        parent = p;
        parents = new Node[0];
        weights = new double[0];
        //biases = new double[0];

    }

    /**
     * getInitialArrayFor()
     * @param len the length of the array to be returned
     * @return a new double array of length len, containing random values determined by MAX_WEIGHT_DIFF, and WEIGHT_SIG_FIG
     */
    double[] getInitialArrayFor(int len){
        double[] out = new double[len];
        for(int i = 0 ; i < len;i++){
            out[i] = ((double)r.nextInt((int)MAX_WEIGHT_DIFF*(int)WEIGHT_SIG_FIG)/WEIGHT_SIG_FIG)-(MAX_WEIGHT_DIFF/2.0);
        }
        return out;
    }

    /**
     * Prints out references held by node.
     * Used for debugging only
     */
    void print(){
        System.out.println("Node   -  "+this+"  - :");
        System.out.println("Weights: "+weights.toString());
        //System.out.println("Biases: "+biases.toString());

    }

    /**
     * Sigmoid
     * @param t input
     * @return sigmoid(t)
     */
    double sig(double t){
        return (1/( 1 + Math.pow(E,(-1*t))));
    }

    /**
     * Sigmoid's Derivative
     * @param t input
     * @return ddxσ( t )
     */
    double sigDiv(double t){
        return t * (1.0-t);
    }

    /**
     * Constructor 2:
     * @param p Array of parent nodes
     * @param par Parent network
     *
     *
     * Used for middle and output layers
     */
    Node(Node[] p, NeuralNetwork par){
        parent = par;
        parents = p;
        weights = getInitialArrayFor(p.length);
        //biases = getInitialArrayFor(p.length);
    }

    /**
     * Mutator for the value
     * @param val new value
     */
    void setValue(double val){
        //value = sig(val);
        value = val;
    }

    /**
     * Accessor for the value
     * @return the value of the node
     */
    double getValue(){return value;}

    /**
     * Accessor for the value, with correct weight applied
     * @param weight the weight between the accessing node, and this node
     * @return the weighted value of this node
     */
    double getWeightedValue(double weight){

        return value*weight;

    }

    /**
     * Goes through parent nodes and determines a new value based on their weights and their values.
     * Commonly shown as ∑ w[i]a[i]
     * @return returns true if successful.
     */
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

    /**
     * Used only in the output layer to set error based on correct values
     * @param correct the intended result of a given activation
     */
    void setErrorAsOutput(double correct){

        error = (correct - value) * sigDiv(value);

    }

    /**
     * Used only for middle layers to set error based on other nodes
     */
    void setErrorAsMiddle(){

        error = 0.0;
        for(int i = 0 ;i < parents.length;i++){

            error+= (parents[i].error * weights[i]) * sigDiv(value);

        }


    }

    /**
     * Applies error to weights
     */
    void applyError(){

        for(int i = 0 ;i < weights.length;i++) {

            weights[i] += parent.LEARNING_RATE * parents[i].error * parents[i].value;

        }

    }

    /**
     * Accessor for weights
     * @return the node's weights
     */
    double[] getWeights(){
        return weights;
    }
}
