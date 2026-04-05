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
#include "movepick.hh"
#include "types.hh"

class Engine {
public:
    Move best_move;

    struct SearchLimits {
        time_point start_time;
        int search_time = 0;
        int nodes = 0;
        std::atomic<bool>* stop_search = nullptr;
    } limits;

    static constexpr int Hash = 16;
    TranspositionTable transposition_table;

    struct SearchStack {
        int static_eval;
        Move move;
    };
    SearchStack stack[MAX_DEPTH];

    void make_move(Position& position, Move move, int ply);

    // move order heuristic: holds {from_square, to_square} of past good moves
    #define MAX_HISTORY 16384
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
        int update = 16 * depth * depth + 128 * (depth - 1);
        if (!good) {
            update = -update;
        }

        capture_history[to][piece][capture] += update - capture_history[to][piece][capture] * std::abs(update) / MAX_HISTORY;
    }

    // move order heuristic: moves that caused cutoff at the same depth level
    Move killers[MAX_DEPTH][2] = {{}};
    void update_killers(Move move, int depth) {
        if (killers[depth][0] != move) {
            killers[depth][1] = killers[depth][0];
            killers[depth][0] = move;
        }
    }
    
    Engine();

    int quiescense(Position& position, int alpha, int beta, int current_depth);
    int negamax(Position& position, int remaining_depth, int current_depth, int alpha, int beta, Move exclude_move = Move());
    int aspiration_window(Position& position, int remaining_depth, int estimate);
    Move get_move(Position& position, SearchInfo info = SearchInfo());

    void reset();

    // info for analysis/debugging
    int total_nodes = 0;
    int count = 0;

    // tunable search parameters
    int RFP_SCALE = 70;
    float LMP_IMPROVING_BASE = 3.0;
    float LMP_IMPROVING_SCALE = 0.75;
    float LMP_BASE = 2.5;
    float LMP_SCALE = 0.5;
    int HISTORY_DIVISOR = 7000;
    int FUTILITY_PRUNE_BASE = 100;
    int FUTILITY_PRUNE_SCALE = 100;
    int FP_CAP_SCALE = 250;
    int FP_CAP_BASE = 210;
    int SEE_PRUNE_SCALE = -92;
    int ASPIRATION_DELTA = 20;
};

#endif