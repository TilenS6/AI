#pragma once
// pre-includes, all .h files needed for this library
#include <iostream>
#include <vector>
#include "ai.h"
using namespace std;

// input & output binding
// evaluating
// generation & population control

class AI;

class AITrainer {
    vector<AI> ais;
    vector<double*> bindIn, bindOut;
    AI *aiTemplate;

    double maxGenDif;
    int genSize;

    bool aiInited, genInited;

public:
    AITrainer();

    // setting-up
    bool initAI(int, int, int);
    bool initGenerations(int, double);
    bool bindInputs(vector<double *>);
    bool bindOutputs(vector<double *>);

    bool step();
    bool nextGeneration(vector<double>); // give scores
};

// post-includes, all .cpp files
#include "aiTrainer.cpp"