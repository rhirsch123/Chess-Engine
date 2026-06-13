#ifndef position_hh
#define position_hh

#include <vector>
#include <string>
#include <stack>
#include <cmath>

#include "types.hh"
#include "bitboards.hh"
#include "movegen.hh"
#include "move.hh"
#include "zobrist.hh"
#include "nnue/nnue.hh"


struct PositionState {
    // indexed top to bottom, left to right, from white's perspective (a8 is 0)
    uint8_t board[64];

    // bitboards indexed by piece type, color
    uint64_t piece_maps[7][2];

    uint64_t hash_value;
    uint64_t pawn_hash;
    uint64_t nonpawn_hash[2];

    uint64_t checkers;
    uint64_t pinned[2];

    // index in position history of the last irreversable move
    int last_threefold_reset = 0;

    int castle_rights;
    int en_passant_col;
    int fifty_move_count;
    int white_material;
    int black_material;
};


class Position {
public:
    std::vector<PositionState> stack;
    PositionState* state;
    int half_moves;
    int turn;

    void set_keys();
    void set_fen(std::string fen);
    Position(std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    static inline int get_color(int piece) {
        return piece > 6;
    }

    static inline int get_piece_type(int piece) {
        return piece - 6 * get_color(piece) - 1;
    }

    int piece_on(int square) {
        return state->board[square];
    }

    uint64_t piece_bb(int piece_type, int color) {
        return state->piece_maps[piece_type][color];
    }

    uint64_t occupancy() {
        return piece_bb(ALL_PIECES, WHITE) | piece_bb(ALL_PIECES, BLACK);
    }

    uint64_t pos_key() {
        return state->hash_value;
    }

    uint64_t pawn_key() {
        return state->pawn_hash;
    }

    uint64_t nonpawn_key(int color) {
        return state->nonpawn_hash[color];
    }

    uint64_t checkers() {
        return state->checkers;
    }

    uint64_t pinned() {
        return state->pinned[turn];
    }

    int castle_rights() {
        return state->castle_rights;
    }

    int ep_col() {
        return state->en_passant_col;
    }

    int fifty_move_count() {
        return half_moves - state->last_threefold_reset;
    }

    int king_square(int color) {
        return lsb(piece_bb(KING, color));
    }


    uint64_t get_attackers(int square, int color, uint64_t blockers);
    bool is_attacked(int square);
    uint64_t get_pinned(int color);

    void put_piece(int piece, int square);
    void remove_piece(int piece, int square);
    DirtyPieces make_move(Move move);
    void pop();

    bool SEE(Move move, int threshold = 0);

    TerminalState get_draw();
    TerminalState get_terminal_state();
};

#endif