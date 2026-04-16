#ifndef types_h
#define types_h

#include <chrono>
#include <atomic>

#define INF 1000000000

// should fit in int16_t for transposition table
#define MATE_SCORE 32000

// max search depth
#define MAX_DEPTH 64

#define MAX_MOVES 256

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

struct SearchInfo {
    // in milliseconds
    int time_left = 0;
    int increment = 0;
    int movetime = 0;

    int depth = 0;
    int nodes = 0;

    bool uci = false;

    // game options
    bool timed_game = false;
    bool fixed_time = false;
    bool fixed_depth = false;
    bool fixed_nodes = false;

    // play random opening from database for the first few moves
    bool use_book = false;

    bool verbose = false;
    std::atomic<bool>* stop_search = nullptr;
};

static int piece_values[6] = { 100, 313, 339, 546, 944, 0 };

enum Piece : uint8_t {
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

enum PieceType {
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
    ALL_PIECES = 6
};

enum Color {
    WHITE = 0,
    BLACK = 1
};

enum MoveGenType {
    TACTIC,
    QUIET,
    ALL
};

enum CastleRights {
    WHITE_KINGSIDE = 1,
    WHITE_QUEENSIDE = 1 << 1,
    BLACK_KINGSIDE = 1 << 2,
    BLACK_QUEENSIDE = 1 << 3
};

enum TerminalState {
    NONE = 0,
    WHITE_MATE = 1,
    BLACK_MATE = 2,
    STALEMATE = 3,
    FIFTY_MOVE_RULE = 4,
    THREEFOLD_REPETITION = 5,
    INSUFFICIENT_MATERIAL = 6
};

enum TTBound : uint8_t {
    EXACT,
    LOWER_BOUND,
    UPPER_BOUND
};

#define TT_DEPTH_UNSEARCHED -1

struct TTEntry {
    uint16_t key = 0;
    int16_t value;
    int16_t static_eval;
    uint16_t best_move;
    TTBound type;
    int8_t depth;
};

#endif