#include "ai.h"

AI::AI(int inputN, vector<int> hiddenLayersNs, int outputN) {
    cout << inputN << "-";
    for (int i = 0; i < hiddenLayersNs.size(); ++i)
        cout << hiddenLayersNs[i] << "-";
    cout << outputN << "\n";
    
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

void AI::setInputs(vector<double> in) {
    if (in.size() != layers.at(0).neurons.size()) {
        cout << "Arguments missmatch! void AI::setInputs(T... args)\n";
        return;
    }
    for (int i = 0; i < in.size(); ++i) {
        layers[0].neurons.at(i).activation = in[i];
    }
}

void AI::process() {
    if (layers.size() == 0 || layers.front().neurons.size() == 0 || layers.back().neurons.size() == 0) {
        cout << "AI network not configured properly! void AI::process()";
        return;
    }
    output.clear();
    output.reserve(layers.back().neurons.size());
    for (int i = 1; i < layers.size(); ++i) { // this layer
        Layer &thisLayer = layers.at(i);
        for (int k = 0; k < thisLayer.neurons.size(); ++k) { // this neuron
            double sum = 0;
            vector<Neuron> &prevNeurons = layers.at(i - 1).neurons;
            for (int j = 0; j < prevNeurons.size(); ++j) { // prev. neurons
                Neuron &n = prevNeurons.at(j);
                sum += n.activation * n.weights.at(k);
            } // END prev. neurons
            sum += thisLayer.neurons.at(k).bias;
            thisLayer.neurons.at(k).activation = normFunct(sum);
        } // END this neuron
    } // END this layer

    for (int i = 0; i < layers.back().neurons.size(); ++i) {
        output.push_back(layers.back().neurons.at(i).activation);
    }
}

void AI::print(bool middle_layers = false) {
    cout << "IN:";
    for (int i = 0; i < layers.front().neurons.size(); ++i) {
        cout << "\t" << layers.front().neurons[i].activation;
    }
    cout << "\n";
    if (middle_layers) {
        for (int i = 1; i < layers.size() - 1; ++i) { // select 1 layer
            for (int j = 0; j < layers[i].neurons.size(); ++j) { // select 1 neuron
                cout << "\t" << layers[i].neurons[i].activation;
            }
            cout << "\n";
        }
    }
    cout << "OUT:";
    for (int i = 0; i < layers.back().neurons.size(); ++i) {
        cout << "\t" << layers.back().neurons[i].activation;
    }
    cout << "\n";
}
void AI::printWB() {
    for (int i = 0; i < layers.size(); ++i) { // select 1 layer
        cout << "L" << i << "\n";
        for (int j = 0; j < layers[i].neurons.size(); ++j) { // select 1 neuron
            cout << i << ":\t";
            if (i == 0) cout << "/";
            else cout << layers[i].neurons[j].bias;
            cout << "\tW:\t";
            if (i == layers.size() - 1) {
                cout << "/";
            } else {
                for (int k = 0; k < layers[i].neurons[j].weights.size(); ++k) {
                    cout << layers[i].neurons[j].weights[k] << "\t";
                }
            }
            cout << "\n";
        }
        cout << "-------------------------\n";
    }
}

void AI::randomizeWeights(double from, double to, unsigned int seed) {
    if (from > to)
        swap(from, to);
    srand(seed);
    double diff = to - from;

    for (int i = 0; i < layers.size() - 1; ++i) { // this layer
        for (int j = 0; j < layers.at(i).neurons.size(); ++j) { // this neuron
            for (int k = 0; k < layers.at(i).neurons.at(j).weights.size(); ++k) { // one of his weights
                double val = (rand() % (int)(diff * 100)) / (double)100.0 + from;
                layers.at(i).neurons.at(j).weights.at(k) = val;
            }
        }
    }
}
void AI::randomizeBiases(double from, double to, unsigned int seed) {
    if (from > to)
        swap(from, to);
    srand(seed);
    double diff = to - from;

    for (int i = 1; i < layers.size(); ++i) { // this layer
        for (int j = 0; j < layers.at(i).neurons.size(); ++j) { // this neuron
            layers.at(i).neurons.at(j).bias = ((rand() % (int)(diff * 100)) / (double)100.0) - to;
        }
    }
}

#define writeToFile(val) f.write((char *)&val, sizeof(val));
#define readFromFile(val) f.read((char *)&val, sizeof(val));

/*
u16: unq, ver
u16: layerC (
    u16: neuronsC (
        double: bias
        u16: weightsC (
            double: weight
        )
    )
)
*/
void AI::saveToFile(string fileName) {
    ofstream f;
    f.open(fileName, ios::out | ios::binary | ios::trunc);
    // writing

    uint16_t unq = FILE_UNIQUE_NMB;
    uint16_t ver = FILE_VER;
    writeToFile(unq);
    writeToFile(ver);

    uint16_t layerC = layers.size();
    writeToFile(layerC);
    for (int i = 0; i < layerC; ++i) {
        uint16_t neuronsC = layers[i].neurons.size();
        writeToFile(neuronsC);
        for (int j = 0; j < neuronsC; ++j) {
            double bias = layers[i].neurons[j].bias;
            writeToFile(bias);
            uint16_t weightsC = layers[i].neurons[j].weights.size();
            writeToFile(weightsC);
            for (int k = 0; k < weightsC; ++k) {
                double weight = layers[i].neurons[j].weights[k];
                writeToFile(weight);
            }
        }
    }
    f.close();
}
/// @return **False** on success and **true** on fail
bool AI::loadFromFile(string fileName) {
    ifstream f;
    f.open(fileName, ios::in | ios::binary);
    uint16_t unq, ver;
    readFromFile(unq);
    readFromFile(ver);
    if (unq != FILE_UNIQUE_NMB || ver != FILE_VER) return true;

    layers.clear();

    uint16_t layerC;
    readFromFile(layerC);
    for (int i = 0; i < layerC; ++i) {
        Layer ly;
        layers.push_back(ly);
        uint16_t neuronsC;
        readFromFile(neuronsC);
        for (int j = 0; j < neuronsC; ++j) {
            Neuron nr;
            layers.back().neurons.push_back(nr);
            double bias;
            readFromFile(bias);
            layers.back().neurons.back().bias = bias;
            uint16_t weightsC;
            readFromFile(weightsC);
            for (int k = 0; k < weightsC; ++k) {
                double weight;
                readFromFile(weight);
                layers.back().neurons.back().weights.push_back(weight);
            }
        }
    }

    f.close();
    return false;
}