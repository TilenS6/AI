#include "ai.h"

AI::AI(int inputN, vector<int> hiddenLayersNs, int outputN) {
    int hiddenLayersCount = hiddenLayersNs.size();
    Neuron initN;
    initN.activation = 0.0;
    initN.bias = 0.0;
    initN.weights.clear();

    layers.resize(2 + hiddenLayersCount);

    layers.front().neurons.resize(inputN, initN);
    for (int i = 0; i < hiddenLayersCount; ++i) {
        layers.at(i + 1).neurons.resize(hiddenLayersNs.at(i), initN);
    }
    layers.back().neurons.resize(outputN, initN);

    // weights (input + hidden)
    for (int i = 0; i < layers.size() - 1; ++i) {
        int sz = layers.at(i).neurons.size();
        int nextSz = layers.at(i + 1).neurons.size();
        for (int j = 0; j < sz; ++j) {
            layers.at(i).neurons.at(j).weights.resize(nextSz, 0.0);
        }
    }
}

// -------------------------
inline double AI::normFunct(double a) {
    return (a > 0) * a - (a - 1) * (a > 1);
}

template <typename... T>
void AI::setInputs(T... args) {
    if (sizeof...(args) == layers.at(0).neurons.size())
        handle_setInputs(0, args...);
    else
        cout << "Arguments missmatch! void AI::setInputs(T... args)\n";
}

template <typename... T>
void AI::handle_setInputs(int count, double a, T... rest) {
    layers.at(0).neurons.at(count).activation = a;
    handle_setInputs(count + 1, rest...);
}
void AI::handle_setInputs(int count) {
}

void AI::process() {
    if (layers.size() == 0 || layers.front().neurons.size() == 0 || layers.back().neurons.size() == 0) {
        cout << "AI network not configured properly! void AI::process()";
        return;
    }
    outputs.clear();
    outputs.reserve(layers.back().neurons.size());
    for (int i = 1; i < layers.size(); ++i) { // this layer
        Layer &thisLayer = layers.at(i);
        for (int k = 0; k < thisLayer.neurons.size(); ++k) { // this neuron
            double sum = 0;
            int iprev = i - 1;
            vector<Neuron> &prevNeurons = layers.at(iprev).neurons;
            for (int j = 0; j < prevNeurons.size(); ++j) { // prev. neurons
                Neuron &n = prevNeurons.at(j);
                sum += n.activation * n.weights.at(k);
            } // END prev. neurons
            sum += thisLayer.neurons.at(k).bias;
            thisLayer.neurons.at(k).activation = normFunct(sum);
        } // END this neuron
    } // END this layer

    for (int i = 0; i < layers.back().neurons.size(); ++i) {
        outputs.push_back(layers.back().neurons.at(i).activation);
    }
}

void AI::randomizeWeights(double from, double to, unsigned int seed) {
    if (from > to) swap(from, to);
    srand(seed);
    double diff = to - from;

    for (int i = 0; i < layers.size() - 1; ++i) {                                 // this layer
        for (int j = 0; j < layers.at(i).neurons.size(); ++j) {                   // this neuron
            for (int k = 0; k < layers.at(i).neurons.at(j).weights.size(); ++k) { // one of his weights
                double val = (rand() % (int)(diff * 100)) / (double)100.0 + from;
                layers.at(i).neurons.at(j).weights.at(k) = val;
            }
        }
    }
}
void AI::randomizeBiases(double from, double to, unsigned int seed) {
    if (from > to) swap(from, to);
    srand(seed);
    double diff = to - from;

    for (int i = 1; i < layers.size(); ++i) {                   // this layer
        for (int j = 0; j < layers.at(i).neurons.size(); ++j) { // this neuron
            layers.at(i).neurons.at(j).bias = ((rand() % (int)(diff * 100)) / (double)100.0) - to;
        }
    }
}