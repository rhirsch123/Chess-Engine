#ifndef nnue_hh
#define nnue_hh

#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#include "../types.hh"
#include "../position.hh"
#include "simd.hh"

#define INPUT_SIZE 768  // 64 squares * 6 pieces * 2 colors
#define L1_SIZE 1792
#define L2_SIZE 16
#define L3_SIZE 32

// commonly used values for nnue
#define SCALE 400
#define QA 255 // layer 1 quantization
#define QB 64 // layer 2 quantization

class Position;

namespace NNUE {
    const std::string nnue_file = "nnue.bin";

    void init();

    // chess is horizontally invariant, so the network is trained with each
    // perspective's king always on the left to reduce the state space
    static inline bool is_mirrored(int king_square) {
        return (king_square % 8) > 3;
    }

    template<Color perspective> static inline int make_index(int square, int piece, bool mirror) {
        if constexpr (perspective == WHITE) {
            return (square ^ (mirror * 0b111)) * 12 + piece - 1;
        } else {
            // flip square vertically, flip piece color
            return (square ^ 0b111000 ^ (mirror * 0b111)) * 12 + (piece <= 6 ? piece + 5 : piece - 7);
        }
    }

    struct Accumulator {
        alignas(64) int16_t acc[2][L1_SIZE];
        DirtyPieces dps;
        bool clean;
    };

    void reset_accumulators(Position& position, Accumulator& accumulator);
    void update_accumulators(Accumulator* accumulator);
    void activate_accumulators(Accumulator& accumulator, int turn);

    int evaluate(Position& position);
    int evaluate_incremental(Accumulator& accumulator, int turn);
};

#endif