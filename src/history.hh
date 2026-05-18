#ifndef history_hh
#define history_hh

#include "types.hh"
#include "move.hh"

static constexpr int MAX_HISTORY = 16384;
static constexpr int MAX_CH_PLY = 2;

// [from_square][to_square]
using QuietHistory = int[64][64];

// [ply][to_square][piece]
using ContinuationHistory = int[MAX_DEPTH + MAX_CH_PLY][64][12];

// [to_square][piece_type][capture_type]
using CaptureHistory = int[64][6][5];

// moves that caused cutoff at the same depth level
using KillerHistory = Move[MAX_DEPTH][2];


inline int history_bonus(int depth) {
    return 16 * depth * depth + 100 * depth;
}
inline int history_malus(int depth) {
    return -16 * depth * depth - 100 * depth;
}

// gravity formula
inline void update_history_entry(int& hist, int update) {
    hist += update - hist * std::abs(update) / MAX_HISTORY;
}

inline void update_quiet_history(QuietHistory& quiet_history, int from, int to, int depth, bool good) {
    int update = good ? history_bonus(depth) : history_malus(depth);
    update_history_entry(quiet_history[from][to], update);
}

inline void update_continuation_history(ContinuationHistory& cont_history, int ply, int to, int piece, int depth, bool good) {
    int update = good ? history_bonus(depth) : history_malus(depth);
    
    // avoid negative indexing when ply is too low
    ply += MAX_CH_PLY;
    update_history_entry(cont_history[ply - 1][to][piece], update);
    update_history_entry(cont_history[ply - 2][to][piece], update);
}

inline void update_capture_history(CaptureHistory& capture_history, int to, int piece_type, int capture, int depth, bool good) {
    int update = good ? history_bonus(depth) : history_malus(depth);
    update_history_entry(capture_history[to][piece_type][capture], update);
}

inline void update_killers(KillerHistory& killers, Move move, int ply) {
    if (killers[ply][0] != move) {
        killers[ply][1] = killers[ply][0];
        killers[ply][0] = move;
    }
}

#endif