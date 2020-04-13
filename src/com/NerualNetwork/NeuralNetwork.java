package com.NerualNetwork;

import java.io.File;
import java.io.FileWriter;

/**
 * Malformatted Data Exception:
 *  Is thrown when the data being fed to the network does not match
 *  the size of the layer to which it is being fed.
 */
class MalformattedDataException extends RuntimeException{

    public MalformattedDataException(String message){
        super(message);
    }

}

/**
 * Unfilled Trian Data Exception:
 *  Is thrown when the training data given to the network has NullPointers in it.
 */
class UnfilledTrainDataException extends RuntimeException{
    UnfilledTrainDataException(String msg){
        super(msg);
    }

}


/**
 * Neural Network class:
 *  Holds the layers, and takes all high level actions
 *
 */
class NeuralNetwork {
    /**
     * LEARNING_RATE:
     *  Affects the rate at which weights are adjusted
     *  (Must be between 0.0 and 1.0)
     */
    public double LEARNING_RATE = 1;

    /**
     * The Input layer is where all the incoming data is accepted.
     * The nodes of the input layer have no weights, no error, and no parents
     */
    private Node[] inputLayer;
    /**
     * The output layer is where the data is outputed in its final form
     * Its parents are the last middle layer
     */
    private Node[] outputLayer;
    /**
     * The middle layers contain all extra layers between the input and output layers.
     * WARNING: Can be jagged depending on the user's input
     */
    private Node[][] middleLayers;

    /**
     * Constructor:
     * @param inputSize the size of the input layer in nodes
     * @param outputSize the size of the output layer in nodes
     * @param middleLayerConfig an array of integers containing the lengths of the middle layers.
     *                          The length of the array determines the number of middle layers,
     *                          Each integer represents the length of one middle layer.
     *                          The order goes from input-side to output.
     */
    NeuralNetwork(int inputSize, int outputSize, int[] middleLayerConfig){

        inputLayer = new Node[inputSize];
        outputLayer = new Node[outputSize];
        for(int i = 0 ; i < inputSize;i++){
            inputLayer[i] = new Node(this);
        }
        middleLayers = new Node[middleLayerConfig.length][];
        for(int i = 0;i<middleLayerConfig.length;i++){
            middleLayers[i] = new Node[middleLayerConfig[i]];
            if(i == 0){
                for(int j = 0 ; j < middleLayers[i].length;j++){
                    middleLayers[i][j] = new Node(inputLayer,this);
                }
            }else{
                for(int j = 0; j < middleLayers[i].length;j++){
                    middleLayers[i][j] = new Node(middleLayers[i-1],this);
                }
            }
        }
        for(int i = 0 ; i < outputLayer.length;i++) {
            outputLayer[i] = new Node(middleLayers[middleLayers.length - 1], this);
            outputLayer[i].isHidden = false;
        }

    }

    /**
     * feedData(formattedData):
     * @param formattedData an array of doubles to be given to the input layer
     * @exception  MalformattedDataException when the data in the NeuralNetworkTrainData is malformatted.
     */
    void feedData(double[] formattedData){

        if(formattedData.length!=inputLayer.length){
            throw new MalformattedDataException("The length of the input data and that of the input layer do not match");
        }
        for(int i = 0 ; i < formattedData.length;i++){
            inputLayer[i].setValue(formattedData[i]);
        }

    }


    /**
     * getUnformattedOutput()
     * @return the double type values of the output layer.
     *          Will return whatever exists in the output layer at the time of calling
     *          Will have the same length as the output layer
     */
    double[] getUnformattedOutput(){
        double[] outpt = new double[outputLayer.length];
        int i = 0;
        for(Node n: outputLayer){
            outpt[i] = (n.getValue());
            i++;
        }

        return outpt;
    }

    /**
     * runLoadedData()
     *  Will feed any data that exists in the input layer through the network
     */
    void runLoadedData(){

        for(int i = 0 ; i < middleLayers.length;i++){
            for(Node n:middleLayers[i]) {
                n.findValue();
            }
        }

        for(Node n : outputLayer){
            n.findValue();
        }

    }

    /**
     * feedCorrect()
     * @param formattedData an array of doubles with the same length as the output layer.
     *                      Will be fed into the output layer in order to determine errors.
     * @exception  MalformattedDataException when the data in the NeuralNetworkTrainData is malformatted.
     */
    void feedCorrect(double[] formattedData){

        if(formattedData.length < outputLayer.length){
            throw new MalformattedDataException("The length of the correct data and that of the output layer do not match");
        }
        int i = 0;
        for(Node n : outputLayer){
            n.setErrorAsOutput(formattedData[i]);

            i++;
        }


    }

    /**
     * findErrors()
     * Goes through all layers and calculates errors.
     * Will not apply any errors
     *
     */
    void findErrors(){

        for(int i = middleLayers.length-1 ; i > -1;i--){

            for(Node n : middleLayers[i]){

                n.setErrorAsMiddle();

            }
        }

    }

    /**
     * ApplyErrors()
     * Goes through all layers and applies their errors.
     */
    void applyErrors(){

        for(int i = middleLayers.length-1 ; i > -1;i--){

            for(Node n : middleLayers[i]){

                n.applyError();

            }
        }
    }

    /**
     * predict()
     * @param formattedData double array with the same length as the input layer.
     *                      The data for the NeuralNetwork to predict based off of.
     * @return double array with the same length as the output layer.
     *          Represents the prediction of the network for a given input.
     */
    double[] predict(double[] formattedData){

        feedData(formattedData);
        runLoadedData();
        return getUnformattedOutput();

    }

    /**
     * train()
     * @param data A NeuralNetworkTrainData object to train with.
     *             Will use all data given within the object.
     * @exception  UnfilledTrainDataException when the NeuralNetworkTrainData object has not been filled.
     * @exception  MalformattedDataException when the data in the NeuralNetworkTrainData is malformatted.
     *
     * @see NeuralNetworkTrainData
     * @see MalformattedDataException
     * @see UnfilledTrainDataException
     */
    void train(NeuralNetworkTrainData data){
        double[] feeder;
        double[] correct;
        for(int i = 0 ; i < data.input.length; i++){

            try{

                feeder = data.input[i];
                correct = data.correct[i];

            }catch (Exception e){
                throw new UnfilledTrainDataException("You have not filled the data you allocated in your training data");

            }
            feedData(feeder);
            runLoadedData();
            feedCorrect(correct);
            findErrors();
            applyErrors();
        }
    }

    /**
     * Create new save file at the given path
     * @param path filepath in which to save
     * @see NeuralNetwork#getConfigForSave() for format
     * @see NeuralNetwork#getNodeDataForSave() for format
     */
    void saveTo(String path){
        try {
            File output = new File(path);
            FileWriter fw = new FileWriter(output);
            String toWrite = getConfigForSave()+getNodeDataForSave();
            fw.write(toWrite);
            fw.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    /**
     * Get the configuration metadata
     * @return a string version of the parameter middleLayerConfig for the the constructor
     */
    String getConfigForSave(){

        String out = "CONFIG:"+inputLayer.length + "," + outputLayer.length;
        for(Node[] ns : middleLayers){
            out += ","+ns.length;
        }
        return out;

    }

    /**
     * Get the node data for the network as a string
     * @return formatted node data
     */
    String getNodeDataForSave(){
        String out = "\n";
        for(Node[] ns : middleLayers){

            for(Node n : ns){
                for(double d : n.getWeights()){
                    out+=","+d;
                }
                out+=";";
            }
        }
        for(Node n : outputLayer){
            for(double d : n.getWeights()){
                out+=","+d;
            }
            out+=";";
        }
        return out;
    }

    /**
     * Load the neural network from a file
     * @param f File object with correctly formatted data in it
     * @exception MalformattedDataException is thrown when the data in the file is incorrectly formatted
     * @see NeuralNetwork#saveTo(String path) for formatting
     */
    NeuralNetwork(File f){



    }

}
