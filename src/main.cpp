#include <iostream>
#include "ai/ai.h"
using namespace std;

int main(int argc, char *argv[]) {
/*  
    AI ai(2, { 2 }, 2); // _ input neurons, _ neurons in one hidden layer, _ outputs
    ai.randomizeWeights(-1, 1, time(0));
    ai.randomizeBiases(-1, 1, time(0));

    ai.setInputs({ .7, -1.0 });
    ai.process();
    cout << "RESULTS:\n";
    ai.print(true);
*/

    AITrainer tr;
    tr.initAI(10, 3, 2);
    return 0;
}