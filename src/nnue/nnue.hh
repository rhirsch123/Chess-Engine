#ifndef nnue_hh
#define nnue_hh

#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>


// SIMD optimizations
#if __ARM_NEON && __aarch64__
    #define USE_NEON
    #include "arm_neon.h"
#endif

#define INPUT_SIZE 768  // 64 squares * 6 pieces * 2 colors
#define HIDDEN_SIZE 1024 // arbitrary
#define OUTPUT_BUCKETS 8 // different output weights depending on piece count

// commonly used values for nnue
#define SCALE 400
#define QA 255 // hidden layer quantization
#define QB 64 // output quantization

#ifdef USE_NEON
const int16x8_t zero = vdupq_n_s16(0);
const int16x8_t max  = vdupq_n_s16(QA);
#endif

namespace NNUE {
    extern int16_t hidden_weights[INPUT_SIZE][HIDDEN_SIZE];
    extern int16_t hidden_biases[HIDDEN_SIZE];
    extern int16_t output_weights_stm[OUTPUT_BUCKETS][HIDDEN_SIZE];
    extern int16_t output_weights_opp[OUTPUT_BUCKETS][HIDDEN_SIZE];
    extern int16_t output_bias[OUTPUT_BUCKETS];
    extern int16_t accumulators[2][HIDDEN_SIZE];

    void init(std::string weights_file);

    int black_index(int square, int piece);
    int get_output_bucket(uint64_t occupancy);

    int evaluate(int board[8][8], int turn, int output_bucket);
    int evaluate_incremental(int turn, int output_bucket);
};

#endif