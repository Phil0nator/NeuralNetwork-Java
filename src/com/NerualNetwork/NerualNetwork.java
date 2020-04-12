package com.NerualNetwork;


class MalformattedDataException extends RuntimeException{

    public MalformattedDataException(String message){
        super(message);
    }

}
class UnfilledTrainDataException extends RuntimeException{
    UnfilledTrainDataException(String msg){
        super(msg);
    }

}




class NeuralNetwork {
    public double LEARNING_RATE = 1;

    private Node[] inputLayer;
    private Node[] outputLayer;
    private Node[][] middleLayers;



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
        }

    }

    void feedData(double[] formattedData){

        if(formattedData.length!=inputLayer.length){
            throw new MalformattedDataException("The length of the input data and that of the input layer do not match");
        }
        for(int i = 0 ; i < formattedData.length;i++){
            inputLayer[i].setValue(formattedData[i]);
        }

    }

    double[] getUnformattedOutput(){
        double[] outpt = new double[outputLayer.length];
        int i = 0;
        for(Node n: outputLayer){
            outpt[i] = n.getValue();
            i++;
        }

        return outpt;
    }

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

    void feedCorrect(double[] formattedData){

        if(formattedData.length != outputLayer.length){
            throw new MalformattedDataException("The length of the input data and that of the input layer do not match");

        }
        int i = 0;
        for(Node n : outputLayer){
            n.setErrorAsOutput(formattedData[i]);
            i++;
        }


    }

    void findErrors(){

        for(int i = middleLayers.length-1 ; i > -1;i--){

            for(Node n : middleLayers[i]){

                n.setErrorAsMiddle();

            }
        }

    }

    void applyErrors(){

        for(int i = middleLayers.length-1 ; i > -1;i--){

            for(Node n : middleLayers[i]){

                n.applyError();

            }
        }
    }

    double[] predict(double[] formattedData){

        feedCorrect(formattedData);
        runLoadedData();
        return getUnformattedOutput();

    }

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

}
