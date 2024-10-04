#pragma once
// pre-includes, all .h files needed for this library
#include <iostream>
#include <vector>
using namespace std;

struct Neuron {
    double activation;
    double bias;
    vector<double> weights;
};

struct Layer {
    vector<Neuron> neurons;
};

class AI {
    vector<Layer> layers;

    template <typename... T>
    void handle_setInputs(int, double, T...);
    void handle_setInputs(int);

    inline double normFunct(double);

public:
    vector<double> outputs;

    // functions below

    AI(int, vector<int>, int);

    template <typename... T>
    void setInputs(T...);
    void process();

    void randomizeWeights(double = -1.0, double = 1.0, unsigned int = 1234);
    void randomizeBiases(double = -1.0, double = 1.0, unsigned int = 1234);

    // TODO save & load to file (weights & biases)
    // TODO visualization
};

// post-includes, all .cpp files
#include "ai.cpp"

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
_mm_prefetch(reinterpret_cast<char*>(&a), _MM_HINT_T0);



template <class T>
inline T mult(T a, T b) {
    return reinterpret_cast<T>((a * b) >> (sizeof(T)*8));
}



*/