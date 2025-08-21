#ifndef move_generator_hh
#define move_generator_hh

#include <vector>
#include <algorithm>

#include "position.hh"
#include "engine.hh"

class Engine;

enum MoveStage {
    GOOD_TACTICS,
    QUIETS,
    BAD_TACTICS
};

// generate tactics first, then quiet moves - less sorting
class MoveGenerator {
    Position& position;
    Engine& engine;
    int current_depth;
    std::vector<Move> tactics;
    std::vector<Move> quiets;
    std::vector<Move> bad_tactics;
public:
    MoveStage stage = GOOD_TACTICS;
    int index = 0;

    MoveGenerator(Position& position, Engine& engine, int current_depth);
    
    Move next_move();
};

#endif