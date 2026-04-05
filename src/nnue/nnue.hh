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
#define L1_SIZE 1536
#define L2_SIZE 16
#define L3_SIZE 32

// commonly used values for nnue
#define SCALE 400
#define QA 255 // layer 1 quantization
#define QB 64 // layer 2 quantization

class Position;

namespace NNUE {
    const std::string nnue_file = "nnue.bin";

    enum DirtyType {
        QUIET,
        CAPTURE,
        CASTLE,
        PROMOTION,
        EN_PASSANT,
        CAP_PROMO,
        NONE
    };

    struct DirtyPieces {
        int white_add0;
        int black_add0;
        int white_sub0;
        int black_sub0;
        int white_add1;
        int black_add1;
        int white_sub1;
        int black_sub1;

        DirtyType type;
    };

    struct AccInfo {
        DirtyPieces dps;
        bool clean;
    };

    
    void init();

    // chess is horizontally invariant, so the network is trained with each
    // perspective's king always on the left to reduce the state space
    static inline bool is_mirrored(int king_square) {
        return (king_square % 8) > 3;
    }

    static inline int white_index(int square, int piece, bool mirror) {
        return (square ^ (mirror * 0b111)) * 12 + piece - 1;
    }
    static inline int black_index(int square, int piece, bool mirror) {
        // flip square vertically, flip piece color
        return (square ^ 0b111000 ^ (mirror * 0b111)) * 12 + (piece <= 6 ? piece + 5 : piece - 7);
    }

    void reset_accumulators(Position& position);
    void set_dirty(int ply, DirtyPieces& dps);
    void update_accumulators(int ply);
    void clean_accumulators(int ply);

    void set_active(int ply);

    int evaluate(Position& position);
    int evaluate_incremental(int ply, int turn);
};

#endif