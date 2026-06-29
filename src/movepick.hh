#ifndef movepick_hh
#define movepick_hh

#include <algorithm>

#include "move.hh"
#include "position.hh"
#include "movegen.hh"
#include "engine.hh"

class Engine;

enum MoveStage : int {
    STAGE_HASH_MOVE = 0,
    STAGE_GEN_TACTICS,
    STAGE_GOOD_TACTICS,
    STAGE_KILLER,
    STAGE_GEN_QUIETS,
    STAGE_QUIETS,
    STAGE_BAD_TACTICS
};

struct ScoredMove {
    int score;
    Move move;

    ScoredMove() = default;
    ScoredMove(Move move, int score) : move(move), score(score) {}
};

struct ScoredMoveList {
    // initializing this is slow
    union {
        ScoredMove list[MAX_MOVES];
    };
    int size;

    ScoredMoveList() : size(0) {}

    inline void add(ScoredMove move) {
        list[size++] = move;
    }

    inline ScoredMove* begin() {
        return list;
    }

    inline ScoredMove* end() {
        return list + size;
    }
};

// generate tactics first, then quiet moves
class MovePicker {
    Position& position;
    Engine& engine;
    ScoredMoveList scored_moves;
    MoveList bad_tactics;
    int current_depth;
    Move hash_move;
    Move killer;
    int index = 0;
public:
    bool only_tactics;
    int stage = STAGE_HASH_MOVE;

    MovePicker(Position& position, Engine& engine, int current_depth, Move hash_move = Move(), bool only_tactics = false);
    template<MoveGenType T> void get_scored_moves();
    Move next_move();
};

#endif