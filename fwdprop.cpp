#include <iostream>
#include <vector>
#include </home/dk/Downloads/eigen-git-mirror/Eigen/Core> /// yeah, I know

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
    public:
        void makeRandomizedLayers (vector<int>, vector<function<double(double)>>);
        MatrixXd predict (MatrixXd features);
} nn;

MatrixXd mlp::forwardPropLayer (MatrixXd inputs, int layerIndex){
    MatrixXd weightedInput;
    layer layerSpecs = layers[layerIndex];
    weightedInput = inputs * layerSpecs.weights;
    weightedInput.rowwise() += layerSpecs.biases.transpose(); //adds biases to each sample in dataset
    MatrixXd activations = weightedInput.unaryExpr(layerSpecs.activationWrapper);

    return activations;
}

void mlp::makeRandomizedLayers (vector<int> nodesPerLayer, vector<function<double(double)>> activationFunctions){ 
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
    for (int layerIndex = 0; layerIndex<layers.size(); layerIndex++){
        input = forwardPropLayer (input, layerIndex);
    }
    return input; // "input" refers to the values passed to the next layer, and eventually held by the output layer
}

double relu (double x){
    if (x<0){
        return 0;
    } else {
        return x;
    }
}

double nothing(double x){
    return x;
}

double leakyRelu(double x){
    if (x<0){
        return x*-0.01;
    } else {
        return x;
    }
}

/// add tanh, sigmoid, softmax
/// make member functions, make actvationFunctions a vector of pointers to those methods

int main(){
    srand(42);
    MatrixXd features(5,10); // One row per instance of data, one column per feature
    features.setRandom();

    vector<int> nodesPerLayer = {int(features.cols()), 100, 100 ,1}; // this includes input, hidden, and output nodes. For N feaatures, there are N input nodes.
    vector<function<double(double)>> activationFunctions = {relu, relu, relu}; // length of this vector should be one less than that of nodesPerLayer

    nn.makeRandomizedLayers(nodesPerLayer, activationFunctions);
    cout << nn.predict (features) << endl;
    }