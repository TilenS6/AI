#include "aiTrainer.h"
/*
vector<AI> ais;
bool readyForRun;
*/

AITrainer::AITrainer() {
    ais.clear();
    bindIn.clear();
    bindOut.clear();

    aiInited = false;
    genInited = false;
    aiTemplate = nullptr;

    maxGenDif = 0;
    genSize = 0;
}

// setting-up
/// @return true on error, false on success
bool AITrainer::initAI(int inputs, int hiddenLayersC, int outputs) {
    if (inputs <= 0 || outputs <= 0 || hiddenLayersC < 0) {
        aiInited = false;
        cout << "AITrainer::initAI(...): ERROR, invalid parameters!\n";
        return true;
    }

    vector<int> hiddenLV;
    int dif = outputs - inputs;
    for (int i = 0; i < hiddenLayersC; ++i) {
        double mult = (i + 1) / (double)(hiddenLayersC + 1);
        hiddenLV.push_back(inputs + dif * mult);
    }
    aiTemplate = new AI(inputs, hiddenLV, outputs);
    aiInited = true;
    return false;
}
/// @return true on error, false on success
bool AITrainer::initGenerations(int genSize, double maxGenDif) {
    if (genSize <= 0 || maxGenDif <= 0 || maxGenDif > 1) {
        genInited = false;
        cout << "AITrainer::initGenerations(...): ERROR, invalid parameters!\n";
        return true;
    }

    this->genSize = genSize;
    this->maxGenDif = maxGenDif;
    genInited = true;
    return false;
}
/// @return true on error, false on success
bool AITrainer::bindInputs(vector<double *> in) {
    if (aiTemplate == nullptr) {
        cout << "AITrainer::bindInputs(...): AI not yet inited with AITrainer::initAI(...)!\n";
        bindIn.clear();
        return true;
    }
    if (in.size() != aiTemplate->layers.front().neurons.size()) {
        cout << "AITrainer::bindInputs(...): ERROR, invalid parameter!\n";
        bindIn.clear();
        return true;
    }

    bindIn = in;
}
/// @return true on error, false on success
bool AITrainer::bindOutputs(vector<double *> out) {
    if (aiTemplate == nullptr) {
        cout << "AITrainer::bindOutputs(...): AI not yet inited with AITrainer::initAI(...)!\n";
        bindOut.clear();
        return true;
    }
    if (out.size() != aiTemplate->layers.back().neurons.size()) {
        cout << "AITrainer::bindOutputs(...): ERROR, invalid parameter!\n";
        bindOut.clear();
        return true;
    }

    bindOut = out;
}

/// @return true on error, false on success
bool AITrainer::step() {
    if (!(aiInited && genInited)) {
        cout << "AITrainer::step(): ERROR, AI not inited or GENERATIONS not inited!\n";
        return true;
    }

    for (int i = 0; i < ais.size(); ++i) { // each AI
        AI *thisAI = &ais[i];
        for (int j = 0; j < bindIn.size(); ++j) { // copy each neuron activation level
            thisAI->layers.front().neurons.at(j).activation = *bindIn[j];
        }

        thisAI->process();

        for (int j = 0; j < bindOut.size(); ++j) { // copy each neuron activation level
            *bindOut[j] = thisAI->output.at(j);
        }
    }


    // TODOO inputi & outputi morjo bit PER AI, razlicni so
}
/// @return true on error, false on success
bool AITrainer::nextGeneration(vector<double>) { // give scores
    if (!(aiInited && genInited)) {
        cout << "AITrainer::nextGeneration(...): ERROR, AI not inited or GENERATIONS not inited!\n";
        return true;
    }

    // TODOO
}