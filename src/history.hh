#ifndef history_hh
#define history_hh

#include "types.hh"
#include "move.hh"

static constexpr int MAX_HISTORY = 16384;
static constexpr int MAX_CH_PLY = 2;
static constexpr int MAX_CORRHIST = 1024;
static constexpr int CORRHIST_SIZE = 16384;

// [from_square][to_square]
using QuietHistory = int[64][64];

// [ply][to_square][piece]
using ContinuationHistory = int[MAX_DEPTH + MAX_CH_PLY][64][12];

// [to_square][piece_type][capture_type]
using CaptureHistory = int[64][6][5];

// moves that caused cutoff at the same depth level
using KillerHistory = Move[MAX_DEPTH];

// [color][hash index]
using PawnCorrectionHistory = int[2][CORRHIST_SIZE];

// [color][hash index]
using NonPawnCorrectionHistory = int[2][CORRHIST_SIZE];


inline int history_bonus(int depth) {
    return 16 * depth * depth + 100 * depth;
}
inline int history_malus(int depth) {
    return -16 * depth * depth - 100 * depth;
}

// gravity formula
inline void update_history_entry(int& hist, int update, int max = MAX_HISTORY) {
    hist += update - hist * std::abs(update) / max;
}


inline void update_quiet_history(QuietHistory& hist, int from, int to, int depth, bool good) {
    int update = good ? history_bonus(depth) : history_malus(depth);
    update_history_entry(hist[from][to], update);
}

inline void update_continuation_history(ContinuationHistory& hist, int ply, int to, int piece, int depth, bool good) {
    int update = good ? history_bonus(depth) : history_malus(depth);
    
    // avoid negative indexing when ply is too low
    ply += MAX_CH_PLY;
    update_history_entry(hist[ply - 1][to][piece], update);
    update_history_entry(hist[ply - 2][to][piece], update);
}

inline void update_capture_history(CaptureHistory& hist, int to, int piece_type, int capture, int depth, bool good) {
    int update = good ? history_bonus(depth) : history_malus(depth);
    update_history_entry(hist[to][piece_type][capture], update);
}

inline void update_pawn_corrhist(PawnCorrectionHistory& hist, int color, uint64_t key, int depth, int diff) {
    int update = diff * depth / 8;
    update = std::min(update, 128);
    update = std::max(update, -128);
    update_history_entry(hist[color][key % CORRHIST_SIZE], update, MAX_CORRHIST);
}


inline void update_nonpawn_corrhist(NonPawnCorrectionHistory& hist, int color, uint64_t white_key, uint64_t black_key,
                                    int depth, int diff) {
    int update = diff * depth / 8;
    update = std::min(update, 128);
    update = std::max(update, -128);
    update_history_entry(hist[color][white_key % CORRHIST_SIZE], update, MAX_CORRHIST);
    update_history_entry(hist[color][black_key % CORRHIST_SIZE], update, MAX_CORRHIST);
}

#endif