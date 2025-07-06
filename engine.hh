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

#define INF 1000000000
#define MAX_DEPTH 32


class Engine {
    // time per move in milliseconds
    int time_limit;

    // there are two main replacement policies for TTs, so I have one of each
    TranspositionTable replace_tt;
    TranspositionTable depth_tt;

    // holds {from_square, to_square} of past good moves for move order heuristic
    int history[64][64] = {{}};

    // play random opening from database for the first few moves so it is not deterministic
    bool use_book;
    
public:
    Engine(int time_limit = 1000, bool use_book = true);

    int value(Position& position, int current_depth, int max_depth, int alpha, int beta,
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time);
    Move get_move(Position& position, bool verbose = false);

    std::string playing;

    // info for analysis/debugging
    int total_nodes = 0;
    int evaluated_positions = 0;

    // evaluation weights are part of engine class for fine tuning
    float MATERIAL_WEIGHT = 1;

    float ISOLATED_PAWN_WEIGHT = 11;
    float DOUBLED_PAWN_WEIGHT = 10;
    float BACKWARDS_PAWN_WEIGHT = 13;
    float PAWN_STORM_WEIGHT = 5;
    float PAWN_CHAIN_WEIGHT = 7;
    float PASSED_PAWN_WEIGHT = 13;
    float KNIGHT_MOBILITY_WEIGHT = 1;
    float KNIGHT_OUTPOST_WEIGHT = 7;
    float BISHOP_PAIR_WEIGHT = 14;
    float BISHOP_MOBILITY_WEIGHT = 1;
    float BISHOP_PAWN_COMPLEX_WEIGHT = 6;
    float ROOK_MOBILITY_WEIGHT = 1;
    float ROOK_BEHIND_KING_WEIGHT = 27;
    float QUEEN_OUT_EARLY_WEIGHT = 31;
    float QUEEN_MOBILITY_WEIGHT = 1;
    float KING_TROPISM_WEIGHT = 1;
    float KING_ATTACKERS_WEIGHT = 18;
    float OPEN_KING_WEIGHT = 3;
    float KING_OPEN_COLUMN_WEIGHT = 9;
    float KING_POSITION_WEIGHT = 28;
    float CENTER_CONTROL_WEIGHT = 11;
    float PIECE_DIFFERENCE_WEIGHT = 115;


    // bonuses indexed by number of possible moves not controlled by opponent's pawns
    int knight_mobility_bonus[9] = {
        -15, -7, -4, -2, 1, 3, 5, 8, 13
    };

    int bishop_mobility_bonus[14] = {
        -15, -14, -10, -8, -6, -4, -2, -1, 1, 4, 7, 13, 20, 35
    };

    int rook_mobility_bonus[15] = {
        -20, -17, -16, -14, -13, -11, -8, -4, -2, 1, 6, 7, 10, 19, 25
    };

    int queen_mobility_bonus[28] = {
        -45, -35, -30, -15, -8, 0, 5, 8, 10, 13, 15, 18, 20, 23, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25
    };


    // king tropism bonuses indexed by piece type
    int near_opp_king_bonus[6] = {
        0, 3, -1, 5, 6, 0
    };
};

#endif