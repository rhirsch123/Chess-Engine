#ifndef position_hh
#define position_hh

#include <vector>
#include <string>
#include <stack>
#include <cmath>

#include "move.hh"
#include "zobrist.hh"

// gcc built-in functions
#define num_set_bits(bitboard) __builtin_popcountll(bitboard)
#define least_set_bit(bitboard) __builtin_ctzll(bitboard)

enum piece {
    WHITE_PAWN = 1,
    WHITE_KNIGHT = 2,
    WHITE_BISHOP = 3,
    WHITE_ROOK = 4,
    WHITE_QUEEN = 5,
    WHITE_KING = 6,
    BLACK_PAWN = 7,
    BLACK_KNIGHT = 8,
    BLACK_BISHOP = 9,
    BLACK_ROOK = 10,
    BLACK_QUEEN = 11,
    BLACK_KING = 12
};

enum piece_type {
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5
};

enum color {
    WHITE = 0,
    BLACK = 1
};

struct pair {
    int first;
    int second;
};

struct move_info {
    uint64_t prev_hash;
    Move move;
    int prev_en_passant;
    int prev_fifty_move;
    int captured_piece;
    int prev_threefold_reset;
    int prev_repetitions;
    bool prev_castle[4];
};

struct directions {
    std::vector<pair> queen;
    std::vector<pair> rook;
    std::vector<pair> bishop;
    std::vector<pair> knight;
};


class Position {
public:
    static uint64_t rook_masks[64];
    // precompute rook moves given square and board occupancy.
    // edges of the board can be excluded in occupancy because the rook must stop there.
    // this gives 12 possible blockers per rook position. 2 ^ 12 = 4096.
    static uint64_t rook_moves[64][4096];

    // similar for bishops
    static uint64_t bishop_masks[64];
    static uint64_t bishop_moves[64][512];

    static int values[6];

    static struct directions directions;
    // for hashing
    static Zobrist zobrist;

    static uint64_t knight_moves[64];
    static uint64_t white_pawn_attacks[64];
    static uint64_t black_pawn_attacks[64];
    static uint64_t king_moves[64];

    // for detecting threefold repetition - indexed by half moves
    uint64_t position_history[512];
    // index in position history of the last irreversable move
    int last_threefold_reset = 0;
    
    int board[8][8];

    // holds history needed to unmake last move
    std::stack<struct move_info> move_stack;

    // bitboards indexed by piece type, color
    uint64_t piece_maps[6][2];

    // redundant but convenient to have
    uint64_t white_pieces;
    uint64_t black_pieces;

    uint64_t hash_value;

    // {white row, col, black row, col}
    int king_positions[4];

    // {white queenside, kingside, black queenside, kingside}
    bool can_castle[4];

    int en_passant_col;
    int fifty_move_count;
    int white_material;
    int black_material;
    int half_moves;
    int turn;
    int repetitions;

    void init_piece_moves();

    Position();
    Position(int board[8][8], color turn);
    
    static void print_bitboard(uint64_t bitboard);
    static int get_color(int piece);
    static int get_piece_type(int piece);

    static int char_to_piece(char c);

    uint64_t get_rook_moves(int square, uint64_t blockers);
    uint64_t get_bishop_moves(int square, uint64_t blockers);
    uint64_t get_castle_moves();
    uint64_t get_pawn_pushes(int row, int col);
    uint64_t get_en_passant(int row, int col);

    bool in_check(int king_row, int king_col);
    bool in_check();
    uint64_t get_piece_moves(int row, int col);
    bool is_promotion(int start_row, int start_col, int end_row, int end_col);

    std::vector<Move> get_pseudo_legal_moves();
    bool is_legal(const Move& move);
    std::vector<Move> get_legal_moves();

    void make_move(const Move& move);
    void unmake_move(struct move_info move_info);
    void pop();

    std::string get_draw();
    bool no_legal_moves();

    // < 0: unknown, 0: no legal moves, > 0: at least one legal move
    std::string get_terminal_state(int legals_exist = -1);
};

#endif