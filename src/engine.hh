#ifndef engine_hh
#define engine_hh

#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <utility>
#include <chrono>

#include <iostream>
#include <math.h>
#include <arm_neon.h>

#include "transposition_table.hh"
#include "move.hh"
#include "position.hh"
#include "move_generator.hh"

#define INF 1000000000


class Engine {
public:
    int MAX_DEPTH = 64;

    int score = 0;
    Move best_move;

    bool timed_game;
    
    // in milliseconds
    int time_per_move;

    double minutes;
    // in seconds
    double increment;
    
    // in milliseconds
    int time_left;

    TranspositionTable transposition_table;

    // move order heuristic: holds {from_square, to_square} of past good moves
    int MAX_HISTORY = 16384;
    int quiet_history[64][64] = {{}};
    void update_quiet_history(int from, int to, int depth, bool good) {
        int update = 16 * depth * depth + 128 * (depth - 1);
        if (!good) {
            update = -update;
        }

        quiet_history[from][to] += update - quiet_history[from][to] * std::abs(update) / MAX_HISTORY;
    }

    // {to_square, piece_type, capture_type}
    int capture_history[64][6][5] = {{{}}};
    void update_capture_history(int to, int piece, int capture, int depth, bool good) {
        int update = depth > 13 ? 32 : 16 * depth * depth + 128 * std::max(depth - 1, 0);
        if (!good) {
            update = -update;
        }

        capture_history[to][piece][capture] += update - capture_history[to][piece][capture] * std::abs(update) / MAX_HISTORY;
    }

    // move order heuristic: moves that caused cutoff at the same depth level
    Move killers[64][2] = {{}};
    void update_killers(Move move, int depth) {
        if (killers[depth][0] != move) {
            killers[depth][1] = killers[depth][0];
            killers[depth][0] = move;
        }
    }

    // play random opening from database for the first few moves
    bool use_book;
    
    Engine(int time_per_move, bool use_book);
    Engine(double minutes, double increment, bool use_book);

    bool mate_score(int val);

    int quiescense(Position& position, int alpha, int beta, int current_depth);
    int negamax(Position& position, int remaining_depth, int current_depth, int alpha, int beta,
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time);
    int aspiration_window(Position& position, int remaining_depth, int estimate,
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time);
    Move get_move(Position& position, bool verbose = false);

    void reset();

    // info for analysis/debugging
    int negamax_nodes = 0;
    int quiescense_nodes = 0;
    int count = 0;

    int NULL_PRUNE_DEPTH = 3;
    int RFP_DEPTH = 8;
    int RFP_SCALE = 70;

    int LMP_DEPTH = 5;
    float LMP_IMPROVING_BASE = 3.5;
    float LMP_IMPROVING_SCALE = 0.6;
    float LMP_BASE = 2.5;
    float LMP_SCALE = 0.5;

    int HISTORY_DIVISOR = 7000;

    int FUTILITY_PRUNE_DEPTH = 6;
    int FUTILITY_PRUNE_BASE = 100;
    int FUTILITY_PRUNE_SCALE = 100;

    int FP_CAP_SCALE = 250;
    int FP_CAP_BASE = 210;
    int FP_CAP_HIST = 7;

    int SEE_PRUNE_SCALE = -92;

    int ASPIRATION_DELTA = 21;


    // HCE weights are part of engine class for fine tuning
    int ISOLATED_PAWN_WEIGHT = 11;
    int DOUBLED_PAWN_WEIGHT = 10;
    int BACKWARDS_PAWN_WEIGHT = 13;
    int PAWN_STORM_WEIGHT = 5;
    int PAWN_CHAIN_WEIGHT = 7;
    int PASSED_PAWN_WEIGHT = 13;
    int KNIGHT_OUTPOST_WEIGHT = 7;
    int BISHOP_PAIR_WEIGHT = 13;
    int BISHOP_PAWN_COMPLEX_WEIGHT = 6;
    int ROOK_BEHIND_KING_WEIGHT = 25;
    int QUEEN_OUT_EARLY_WEIGHT = 31;
    int OPEN_KING_WEIGHT = 4;
    int KING_OPEN_COLUMN_WEIGHT = 9;
    int KING_POSITION_WEIGHT = 25;
    int PIECE_DIFFERENCE_WEIGHT = 115;
    int TEMPO_WEIGHT = 11;
};

#endif