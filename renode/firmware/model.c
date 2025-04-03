#include "model.h"

/*
// Mark weights and bias as used and place in .data
__attribute__((used)) __attribute__((section(".data")))
int weights[6] = {500, 300, 200, 400, 100, 600};

__attribute__((used)) __attribute__((section(".data")))
int bias = -100000;  // = -100.0 * 1000
*/
__attribute__((used))
int mock_model_predict(volatile int input[6]) {
    /*
    int score = bias;
    for (int i = 0; i < 6; i++) {
        score += weights[i] * input[i];
    }
        */
       int local_weights[6] = {500, 300, 200, 400, 100, 600};
       int local_bias = -100000;
   
       int score = local_bias;
       for (int i = 0; i < 6; i++) {
           score += local_weights[i] * input[i];
       }

    return score;
}