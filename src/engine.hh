#ifndef engine_hh
#define engine_hh

#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <utility>
#include <chrono>
#include <math.h>

#include "nnue/nnue.hh"
#include "transposition_table.hh"
#include "move.hh"
#include "position.hh"
#include "history.hh"
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

    NNUE::Accumulator accumulators[MAX_DEPTH];
    int acc_index;

    // move order heuristics
    QuietHistory quiet_history;
    ContinuationHistory cont_history;
    CaptureHistory capture_history;
    KillerHistory killers;
    PawnCorrectionHistory pawn_corrhist;
    NonPawnCorrectionHistory nonpawn_corrhist;

    // late move pruning cutoff by depth and improving
    int lmp_table[MAX_DEPTH][2];
    void init_lmp_table() {
        for (int i = 0; i < MAX_DEPTH; i++) {
            lmp_table[i][0] = LMP_BASE + LMP_SCALE * i * i;
            lmp_table[i][1] = LMP_IMPROVING_BASE + LMP_IMPROVING_SCALE * i * i;
        }
    }

    // late move reduction by depth and moves played
    int lmr_table[MAX_DEPTH][MAX_MOVES];
    void init_lmr_table() {
        for (int i = 0; i < MAX_DEPTH; i++) {
            for (int j = 0; j < MAX_MOVES; j++) {
                if (i == 0 || j == 0) {
                    lmr_table[i][j] = 0;
                } else {
                    lmr_table[i][j] = LMR_BASE + LMR_SCALE * log(i) * log(j);
                }
            }
        }
    }
    
    Engine();
    void init();

    void clean_accumulators(int ply);
    int evaluation(Position& position);
    int get_corrhist_adjustment(Position& position);

    void make_move(Position& position, Move move, int ply);
    void unmake_move(Position& position);

    int quiescense(Position& position, int alpha, int beta, int current_depth);
    int negamax(Position& position, int remaining_depth, int current_depth, int alpha, int beta, Move exclude_move = Move());
    int aspiration_window(Position& position, int remaining_depth, int estimate);
    Move get_move(Position& position, SearchInfo info = SearchInfo());

    // info for analysis/debugging
    int total_nodes = 0;
    int count = 0;

    // tunable search parameters
    int RFP_SCALE = 70;
    int NMP_SCALE = 20;
    int NMP_OFFSET = 125;
    float LMP_BASE = 2.5;
    float LMP_SCALE = 0.5;
    float LMP_IMPROVING_BASE = 3;
    float LMP_IMPROVING_SCALE = 0.75;
    int LMR_BASE = 700;
    int LMR_SCALE = 390;
    int LMR_PV = 1024;
    int LMR_TTCAP = 1024;
    int LMR_IMPROVING = 512;
    int LMR_CAPTURE = 1024;
    int LMR_HISTORY = 2400;
    int LMR_GIVES_CHECK = 1024;
    int FP_BASE = 250;
    int FP_SCALE = 130;
    int FP_CAP_SCALE = 250;
    int FP_CAP_BASE = 210;
    int SEE_PRUNE_SCALE = 100;
    int ASPIRATION_DELTA = 20;
    int PAWN_CORRHIST_WEIGHT = 20;
    int NONPAWN_CORRHIST_WEIGHT = 15;
};

#endif