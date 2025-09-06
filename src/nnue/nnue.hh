#ifndef nnue_hh
#define nnue_hh

#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>

#include "../position.hh"

// SIMD optimizations
#if __ARM_NEON && __aarch64__
    #define USE_SIMD
    #include "arm_neon.h"
#endif

#define INPUT_SIZE 768  // 64 squares * 6 pieces * 2 colors
#define HIDDEN_SIZE 512 // arbitrary

// commonly used values for nnue
#define SCALE 400
#define QA 255 // hidden layer quantization
#define QB 64 // output quantization


namespace NNUE {
    extern int16_t hidden_weights[INPUT_SIZE][HIDDEN_SIZE];
    extern int16_t hidden_biases[HIDDEN_SIZE];
    extern int16_t hidden_layer[HIDDEN_SIZE];
    extern int16_t output_weights[HIDDEN_SIZE];
    extern int16_t output_bias;

    void init(std::string weights_file);
    int evaluate(Position& position);
    int evaluate_incremental();
};

#endif