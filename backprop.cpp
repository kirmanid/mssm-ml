#include <iostream>
#include <vector>
#include </home/dk/Downloads/eigen-git-mirror/Eigen/Core>

using namespace std;
using namespace Eigen;

class mlp {
        struct layerConnection {
            function<double(double)> activationWrapper;
            MatrixXd weights, weightDeltas;
            VectorXd biases, biasDeltas;
        };
        struct layerNodes {
            MatrixXd aList, zList;
            // for the above, #rows = #samples , #cols = #neurons in given layer
            // aList element ---> (activation function) ---> corresponding zList element
        };
        double computeC(int,int,int);
        MatrixXd forwardPropLayer (MatrixXd,int);
        vector<layerConnection> connections;
    public:
        double learningRate;
        MatrixXd targets;
        MatrixXd features;
        vector<int> nodesPerLayer;
        double computeDerivative(int, int, int, int);
        vector<layerNodes> nodes;
        double reluPrime(double);
        void incrementDeltas(int);
        void assignNewWeights();
        void train (int);
        void makeRandomizedConnections (vector<function<double(double)>>);
        MatrixXd predict (MatrixXd features);
        double cost (MatrixXd, MatrixXd);
} nn;

MatrixXd mlp::forwardPropLayer (MatrixXd inputs, int connectionIndex){
    MatrixXd weightedInput, activations;
    layerConnection connection = connections[connectionIndex];
    weightedInput = inputs * connection.weights;
    weightedInput.rowwise() += connection.biases.transpose(); //adds biases to each sample in dataset
    activations = weightedInput.unaryExpr(connection.activationWrapper);

    nodes[connectionIndex+1].zList = weightedInput;
    nodes[connectionIndex+1].aList = activations;
    return activations;
}

void mlp::makeRandomizedConnections (vector<function<double(double)>> activationFunctions){ 
    layerConnection connection;
    for (int i = 0; i<activationFunctions.size(); i++){
        MatrixXd weights (nodesPerLayer[i],nodesPerLayer[i+1]);
        MatrixXd weightDeltas (nodesPerLayer[i],nodesPerLayer[i+1]);

        VectorXd biases (nodesPerLayer[i+1]);
        VectorXd biasDeltas (nodesPerLayer[i+1]);

        connection.activationWrapper = activationFunctions[i];
        connection.weights = weights.setRandom();
        connection.biases = biases.setRandom();
        connection.weightDeltas = weightDeltas;
        connection.biasDeltas = biasDeltas;
        connections.push_back(connection);
    }
}

MatrixXd mlp::predict (MatrixXd features){
    MatrixXd input = features;
    nodes[0].aList = features; // Although the input layer has no activations, we use the 0th aList to store its values
    for (int connectionIndex = 0; connectionIndex < connections.size(); connectionIndex++){
        input = forwardPropLayer (input, connectionIndex);
    }
    return input; // "input" refers to the values passed to the next layer, and eventually held by the output layer
}

// derivative of cost function w/ respect to given weight w. W connects kth neuron on 'prev' layer, and jth neuron on current layer. Sample refers to the index of the sample in the dataset/
double mlp::computeDerivative (int connectionIndex, int weightJ, int weightK, int sample) { 
    double 
        a, // a, or a_k^(l-1), is the activation of the neuron that the weight is connected to on the previous layer.
           // Along with c and b, a is one of the factors of dC/dw_(j,k)^l, which is what this function computes. 

        b, // b, or d/dx (activation(z_j^l)), is the derivative of the activation function applied to the weighted outputs of the jth neuron on the current layer. 
           //Note that weighted input != activation, because weighted inputs are passed to the activation function.

        c; // c is the derivative of the cost function with respect to the activations of the (jth) neuron of the current layer that the weight is connected to.

    int currentLayerIndex = connectionIndex + 1; // For a weight/bias that 'connects' two adjcacent layers, these refer to the indices of the current and previous layer
    int prevLayerIndex = connectionIndex;

    if (weightK == -1){ //if bias instead of weight
        a = 0.5; /// I don't know why this makes it work, but it does
    } else {
        a = nodes[prevLayerIndex].aList(sample, weightK);
    }
    b = reluPrime(nodes[currentLayerIndex].zList(sample, weightJ));
    c = computeC(connectionIndex, weightJ, sample);

    //'verificiation' step using slope of secant line, small epsilon
    double epsilon, cost_1, cost_2, secantSlope;
    epsilon = 10^-6;
    cost_1 = cost(targets, predict(features));
    if (weightK != -1){
        connections[connectionIndex].weights(weightK, weightJ) += epsilon;
        cost_2 = cost(targets, predict(features));
        connections[connectionIndex].weights(weightK, weightJ) -= epsilon;
    } else {
        connections[connectionIndex].biases[weightJ] += epsilon;
        cost_2 = cost(targets, predict(features));
        connections[connectionIndex].biases[weightJ] -= epsilon;
    }
    predict (features); // necessary, as predict() changes some state, but using a false weight/bias, as it has been artificially increased by epsilon
    secantSlope = (cost_2 - cost_1)/ epsilon;
    cout <<"Secant slope, Derivative Slope: "<< secantSlope << ", " << a*b*c << endl;

    return a*b*c;
}

double mlp::computeC (int connectionIndex, int weightJ, int sample){
    int currentLayerIndex = connectionIndex + 1;
    int prevLayerIndex = connectionIndex;

    double
        c,

        b_next, // d/dx (activation(z_j^l+1)), equivalent to b as defined in computeDerivative, but on the next layer. 
        //If the weight/bias is not connected to the output layer, c = \sum_{j=0}^(size of next layer, that weight is not connected to) of (b_next)*(c_next)*(e)

        c_next, // The output of this function, but applied to the next layer (dC/da_j^(l+1), where a is from aList), and j is cycled through

        e; // w_(j,k)^(l+1). From my understanding, this refers to a weight one connectionIndex ahead, where old weightJ --> new weightK, and j is cycled through

    if (currentLayerIndex == nodesPerLayer.size() - 1){ //if the weight is connected to output layer
        c = 2*(nodes[currentLayerIndex].aList(sample, weightJ) - targets(sample, weightJ)); // Asumes that the cost function is mean squared error. Equivalent to 2(a_j^l - y_j)
    } else {
        c = 0;
        //Remember, weightJ now serves as the new k-index, as we have moved forward a layer, and the new j is being iterated through
        for (int j = 0;  j < nodesPerLayer[currentLayerIndex + 1] ; j++){
            c_next = computeC(connectionIndex + 1, j, sample);
            b_next = reluPrime(nodes[currentLayerIndex + 1].zList(sample, j));
            e = connections[connectionIndex+1].weights(weightJ,j);
            c += c_next * b_next * e; 
        }
    }
    return c;
}


// mean squared error
double mlp::cost(MatrixXd a, MatrixXd b){ 
    MatrixXd diff = a - b;
    diff = diff.array().pow(2);
    return diff.sum() / diff.size();

} 

// nested for loop gradient descent (abomination)
void mlp::incrementDeltas(int sample){
    for (int connectionIndex = 0; connectionIndex < nodesPerLayer.size() - 1; connectionIndex++){
        for (int k = 0; k < connections[connectionIndex].weights.rows(); k++){ //-1 indicates that it starts at the bias
            for (int j = 0; j < connections[connectionIndex].weights.cols(); j++){ 
                connections[connectionIndex].weightDeltas(k,j) -= learningRate * computeDerivative(connectionIndex,j,k,sample); // gradient descent on weights
                connections[connectionIndex].biasDeltas[j] -= learningRate * computeDerivative(connectionIndex,j,-1,sample); // gradient descent on biases
            }
        }
    }
}

// weight assignment
void mlp::assignNewWeights(){
    for (int connectionIndex = 0; connectionIndex < nodesPerLayer.size() - 1; connectionIndex++){
        connections[connectionIndex].weights +=  connections[connectionIndex].weightDeltas;
        connections[connectionIndex].biases += connections[connectionIndex].biasDeltas;
        // set biasDeltas and weightDeltas back to zero
        connections[connectionIndex].weightDeltas *= 0;
        connections[connectionIndex].biasDeltas *= 0;
    }
}

void mlp::train(int epochs){
    for (int l = 0; l < epochs; l++){
        for (int m = 0; m < features.rows(); m++){
            predict(features);
            incrementDeltas(m);
        }
        assignNewWeights();
        cout << "Epoch " << l << endl;
        cout << "Cost: " << cost(targets, predict(features)) << endl;
    }
}

double relu (double x){
    if (x<0){
        return x*0.01;
    } else {
        return x;
    }
}

double mlp::reluPrime (double x){
    if (x<0){
        return 0.01;
    } else {
        return 1;
    }
}

int main(){
    srand(41);
    nn.learningRate = 0.05;

    MatrixXd features(4,2); // One row per dataset sample, one column per feature
    features << 0, 0,
            1, 0,
            0, 1,
            1, 1; //XOR dataset
    nn.features = features;

    MatrixXd targets (4,1); // One row per dataset sample, one column per target length
    targets << 0,
        1,
        1,
        0; // XOR dataset targets
    nn.targets = targets;

    nn.nodesPerLayer = {int(features.cols()), 10, 10, int(targets.cols())}; // Corresponds to layer index. Includes input, hidden, and output nodes. For N feaatures, there are N input nodes.
    nn.nodes.resize(nn.nodesPerLayer.size());
    vector<function<double(double)>> activationFunctions = {relu, relu, relu}; // Corresponds to connection index. Length of this vector should be one less than that of nodesPerLayer.
    nn.makeRandomizedConnections(activationFunctions);

    cout << nn.predict (features) << "\n\n";
    nn.train(251);
    cout << "\n XOR-trained outputs:" << endl;
    cout << nn.predict (features) << "\n\n";
    }

/* TO DO LIST:
-

NONESSENTIAL:

-move stuff to private

- sort out how activation function passing is going to work. current ideas:
    - pointers to functions are passsed
    - object containing both activation function (double in, double out), and its first derivative(double in, double out)
        -or, even better, have them both be MatrixXd in, MatrixXd out (for better parallelism)

- put eigen in include path of compiler, then fix line 3

- add comments when needed, general cleanup

*/