#pragma once
// pre-includes, all .h files needed for this library
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

// for file OPs
#define FILE_VER 1
#define FILE_UNIQUE_NMB 60091 // 2B

class AITrainer;

struct Neuron {
    double activation;
    double bias;
    vector<double> weights; // on next layer
};

struct Layer {
    vector<Neuron> neurons;
};

class AI {
    vector<Layer> layers;

    inline double normFunct(double);

public:
    vector<double> output;

    // --------
    AI(int, vector<int>, int);

    void setInputs(vector<double>);
    void process();
    void print(bool);
    void printWB(); // weights + biases

    // randomization
    void randomizeWeights(double = -1.0, double = 1.0, unsigned int = 1234);
    void randomizeBiases(double = -1.0, double = 1.0, unsigned int = 1234);

    // file OPs
    void saveToFile(string);
    bool loadFromFile(string);

    // TODO? graphical data visualization
    friend class AITrainer;
};

// post-includes, all .cpp files
#include "ai.cpp"

// dependencies on this file
#include "aiTrainer.h"

// za vse skup:
/*
SIGMOID: o(x) = 1/(1+e^-x)
na 0 je 0.5
bolj + ==> bolj 1
bolj - ==> bolj 0

b1 = o(w1*a1 + w2*a2 + ... + wn*an + BIAS)

* sigmoid je oldschool + slow learner
ReLU = {x<0: =0; x>0: =x (pa najbrz cap na 1)}
*/

// performance
/*
#include <xmmintrin.h>
_mm_prefetch(reinterpret_cast<char*>(&a), _MM_HINT_T0); // preload to L0 cache



template <class T>
inline T mult(T a, T b) {
    return reinterpret_cast<T>((a * b) >> (sizeof(T)*8));
}



*/