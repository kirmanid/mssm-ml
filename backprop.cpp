#include <iostream>
#include <vector>
#include </Eigen/Core>

using namespace std;
using namespace Eigen;

class mlp {
        struct layer {
            function<double(double)> activationWrapper;
            MatrixXd weights;
            VectorXd biases;
        };
        MatrixXd forwardPropLayer (MatrixXd,int);
        vector<layer> layers;
        double computeDerivative(int,int,int);
        double reluPrime (double);
    public:
        double learningRate;
        vector<MatrixXd> zList, aList; // z is weighted outputs before activation, a is after activation, vector is aList and zList per layer
        vector<int> nodesPerLayer;
        void makeRandomizedLayers (vector<function<double(double)>>);
        MatrixXd predict (MatrixXd features);
        void computeGradientStep ();
        void applyGradientStep ();
        vector<MatrixXd> gradientStep;
        double target;
} nn;

MatrixXd mlp::forwardPropLayer (MatrixXd inputs, int layerIndex){
    MatrixXd weightedInput;
    layer layerSpecs = layers[layerIndex];
    weightedInput = inputs * layerSpecs.weights;
    weightedInput.rowwise() += layerSpecs.biases.transpose(); //adds biases to each sample in dataset

    zList[layerIndex+1] = weightedInput; // zList[n] has one row per sample, one column per node of the *next* layer. (same for aList)
    MatrixXd activations = weightedInput.unaryExpr(layerSpecs.activationWrapper);
    aList[layerIndex+1] = activations; 

    return activations;
}

void mlp::makeRandomizedLayers (vector<function<double(double)>> activationFunctions){ 
    layer layer;
    for (int i = 0; i<activationFunctions.size(); i++){
        layer.activationWrapper = activationFunctions[i];
        MatrixXd weights (nodesPerLayer[i],nodesPerLayer[i+1]);
        layer.weights = weights.setRandom();
        VectorXd biases (nodesPerLayer[i+1]);
        layer.biases = biases.setRandom();
        layers.push_back(layer);
    }
}

MatrixXd mlp::predict (MatrixXd features){
    MatrixXd input = features;
    aList[0] = features;
    for (int layerIndex = 0; layerIndex<layers.size(); layerIndex++){
        input = forwardPropLayer (input, layerIndex);
    }
    return input; // "input" refers to the values passed to the next layer, and eventually held by the output layer
}

double mlp::reluPrime(double x){
    if (x>0){
        return 1;
    } else {
        return .01; 
    }
}

double mlp::computeDerivative (int weightLayer, int weightJ, int weightK) { // derivative of cost function w/ respect to given weight w. W connects kth neuron on prev. layer, and jth neuron on current layer.
    double 
        a, // a, or a_k^(l-1), is the activation of the neuron that the weight is connected to on the previous layer.
           // Along with c and b, a is one of the factors of dC/dw_(j,k)^l, which is what this function computes. 

        b, // b, or d/dx (activtation(z_j)), is the derivative of the activation function applied to the weighted outputs of the jth neuron on the current layer. 
           //Note that weighted input != activation, because weighted inputs are passed to the activation function.

        c; // c is the derivative of the cost functuion with respect to the activations of the (jth) neuron of the previous layer that the weight is connected to.

    if (weightK == -1){ //if bias instead of weight
        a = 1; 
    } else {
        a = aList[weightLayer](weightK);
    }
    b = reluPrime(zList[weightLayer](weightJ));
    if (weightLayer == 2){ //if the weight is connected to output layer
        c = 2*(aList[weightLayer](weightJ) - target); // Asumes that the cost function is mean squared error. Equivalent to 2(a_j^l - y_j)
    } else {
        c = 0;
        for (int j = 0; j < nodesPerLayer[weightLayer+1]; j++){
            c += computeDerivative(weightLayer + 1, j, weightK);
        }
    }
    return a*b*c;
}

void mlp::computeGradientStep(){ // layer is vector index (i), for Matrices, j by k
    gradientStep.resize(nodesPerLayer.size()-2);
    for (int i = 1; i < nodesPerLayer.size()-1; i++){
        MatrixXd weightGradients (nodesPerLayer[i],nodesPerLayer[i+i]);

        for (int j = 0; j < nodesPerLayer[i]; j++){

            for (int k = 0; k < nodesPerLayer[i]; k++){

                weightGradients(j,k) = computeDerivative(i,j,k) * learningRate;
            }
        }
        gradientStep[i] = weightGradients;
    }
}

double relu (double x){ // actually leaky relu, not pure relu
    if (x>0){
        return x;
    } else {
        return x*0.01;
    }
}

int main(){
    srand(42);
    nn.aList.resize(5); nn.zList.resize(5);
    nn.learningRate = 0.01

    MatrixXd features (1,2); // One row per instance of data, one column per feature
    features << /*0, 0,
                1, 0,
                0, 1,*/
                1, 1; //XOR dataset
    /* 
    MatrixXd targets (4,1);
    targets << 0, 1, 1, 0;  // a.k.a the y values for ML model
    nn.targets = targets;*/

    nn.target = 0; //trying with one sample first
    

    nn.nodesPerLayer = {int(features.cols()), 5, 5 ,1}; // this includes input, hidden, and output nodes. For N features, there are N input nodes.
    vector<function<double(double)>> activationFunctions = {relu, relu, relu}; // length of this vector should be one less than that of nodesPerLayer

    nn.makeRandomizedLayers(activationFunctions);
    cout << nn.predict (features) << "\n\n";
    
    nn.computeGradientStep();
    nn.applyGradientStep();
    }