#ifndef movegen_hh
#define movegen_hh

#include "bitboards.hh"
#include "position.hh"
#include "move.hh"
#include "nnue/simd.hh"

class Position;

struct MoveList {
    Move moves[MAX_MOVES];
    int size;

    MoveList() : size(0) {};

    inline void add(Move move) {
        moves[size++] = move;
    }

    inline Move* begin() {
        return moves;
    }

    inline Move* end() {
        return moves + size;
    }
};

void get_pseudo_legal_moves(Position& position, MoveList* move_list, MoveGenType move_type = ALL);
bool is_legal(Position& position, Move move);
void get_legal_moves(Position& position, MoveList* move_list);
bool is_psuedo_legal(Position& position, Move move);
bool no_legal_moves(Position& position);

uint64_t bulk_perft(Position& position, int depth);

#endif