#ifndef movepick_hh
#define movepick_hh

#include <algorithm>

#include "move.hh"
#include "position.hh"
#include "movegen.hh"
#include "engine.hh"

class Engine;

enum MoveStage {
    HASH_MOVE,
    GOOD_TACTICS,
    QUIETS,
    BAD_TACTICS
};

struct ScoredMove {
    int score;
    Move move;

    ScoredMove() = default;
    ScoredMove(Move move, int score) : move(move), score(score) {}
};

struct ScoredMoveList {
    ScoredMove list[MAX_MOVES];
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
public:
    bool only_tactics;
    MoveStage stage = HASH_MOVE;
    int index = 0;

    MovePicker(Position& position, Engine& engine, int current_depth, Move hash_move = Move(), bool only_tactics = false);
    template<MoveGenType T> void get_sorted_moves();
    Move next_move();
};

#endif