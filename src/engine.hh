#ifndef engine_hh
#define engine_hh

#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <utility>
#include <chrono>

#include "transposition_table.hh"
#include "move.hh"
#include "position.hh"
#include "move_generator.hh"

#define INF 1000000000
#define MAX_DEPTH 32


class Engine {
public:
    // work into code with if statements to compare different versions in test.cc
    bool test_condition = false;

    Move best_move;
    bool move_found;

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
    int quiet_history[64][64] = {{}};
    void update_quiet_history(int from, int to, int depth, bool good) {
        int update = depth > 13 ? 32 : 16 * depth * depth + 128 * std::max(depth - 1, 0);
        if (!good) {
            update = -update;
        }

        quiet_history[from][to] += update - quiet_history[from][to] * std::abs(update) / 16384;
    }

    // {to_square, piece_type, capture_type}
    int capture_history[64][6][5] = {{{}}};
    void update_capture_history(int to, int piece, int capture, int depth, bool good) {
        int update = depth > 13 ? 32 : 16 * depth * depth + 128 * std::max(depth - 1, 0);
        if (!good) {
            update = -update;
        }

        capture_history[to][piece][capture] += update - capture_history[to][piece][capture] * std::abs(update) / 16384;
    }

    // move order heuristic: moves that caused cutoff at the same depth level
    Move killers[MAX_DEPTH][2] = {{}};
    void update_killers(Move move, int depth) {
        if (killers[depth][0] != move) {
            killers[depth][1] = killers[depth][0];
            killers[depth][0] = move;
        }
    }

    // play random opening from database for the first few moves so it is not deterministic
    bool use_book;
    
    Engine(int time_per_move, bool use_book);
    Engine(double minutes, double increment, bool use_book);

    int quiescense(Position& position, int alpha, int beta, int current_depth);

    int negamax(Position& position, int remaining_depth, int current_depth, int alpha, int beta,
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time);

    int aspiration_window(Position& position, int remaining_depth, int estimate, std::chrono::time_point<std::chrono::high_resolution_clock> start_time);
    
    Move get_move(Position& position, bool verbose = false);

    // info for analysis/debugging
    int total_nodes = 0;
    int evaluated_positions = 0;

    float NULL_PRUNE_DEPTH = 2;
    float FUTILITY_PRUNE_DEPTH = 5;
    float ASPIRATION_DELTA = 30;

    // evaluation weights are part of engine class for fine tuning
    float ISOLATED_PAWN_WEIGHT = 11;
    float DOUBLED_PAWN_WEIGHT = 10;
    float BACKWARDS_PAWN_WEIGHT = 13;
    float PAWN_STORM_WEIGHT = 5;
    float PAWN_CHAIN_WEIGHT = 7;
    float PASSED_PAWN_WEIGHT = 13;
    float KNIGHT_OUTPOST_WEIGHT = 7;
    float BISHOP_PAIR_WEIGHT = 13;
    float BISHOP_PAWN_COMPLEX_WEIGHT = 6;
    float ROOK_BEHIND_KING_WEIGHT = 25;
    float QUEEN_OUT_EARLY_WEIGHT = 31;
    float OPEN_KING_WEIGHT = 4;
    float KING_OPEN_COLUMN_WEIGHT = 9;
    float KING_POSITION_WEIGHT = 25;
    float PIECE_DIFFERENCE_WEIGHT = 115;
    float TEMPO_WEIGHT = 11;
};

#endif