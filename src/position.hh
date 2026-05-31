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


struct MoveInfo {
    uint64_t prev_hash;
    uint64_t prev_pawn_hash;
    uint64_t prev_checkers;
    Move move;
    int prev_en_passant;
    int prev_fifty_move;
    int captured_piece;
    int prev_threefold_reset;
    int prev_castle;
};


class Position {
public:
    // indexed top to bottom, left to right, from white's perspective (a8 is 0)
    uint8_t board[64];

    // bitboards indexed by piece type, color
    uint64_t piece_maps[7][2];
    
    // for detecting threefold repetition - indexed by ply
    std::vector<uint64_t> position_history;
    // index in position history of the last irreversable move
    int last_threefold_reset = 0;

    // holds history needed to unmake last move
    std::vector<MoveInfo> move_stack;

    uint64_t hash_value;
    uint64_t pawn_hash;

    uint64_t checkers;

    int castle_rights;
    int en_passant_col;
    int fifty_move_count;
    int white_material;
    int black_material;
    int half_moves;
    int turn;

    void set_fen(std::string fen);
    Position(std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    static inline int get_color(int piece) {
        return piece > 6;
    }

    static inline int get_piece_type(int piece) {
        return piece - 6 * get_color(piece) - 1;
    }

    uint64_t pieces();

    bool is_attacked(int square);
    uint64_t get_attackers(int square, int color, uint64_t blockers = 0ULL);

    void make_move(Move move);
    void unmake_move(MoveInfo& move_info);
    void pop();

    bool SEE(Move move, int threshold = 0);

    TerminalState get_draw();
    TerminalState get_terminal_state();
};

#endif