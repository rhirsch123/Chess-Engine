#include <chrono>

#include "move.hh"
#include "position.hh"
#include "engine.hh"


Engine engine(1, false);

int leafs = 0;
int captures = 0;
int en_passant = 0;
int checks = 0;
int checkmates = 0;
void perft(Position& position, int max_depth) {
    if (position.half_moves == max_depth) {
        leafs++;

        if (position.in_check()) {
            checks++;
            if (!position.get_terminal_state().empty() && position.get_draw().empty()) checkmates++;
        }

        return;
    }

    MoveGenerator movegen(position, engine, position.half_moves);

    while (true) {
        Move move = movegen.next_move();
        if (!move) break;
        
        int cap = position.board[move.end_row()][move.end_col()];
        if (cap && position.half_moves + 1 == max_depth) captures++;
        if (!cap && Position::get_piece_type(position.board[move.start_row()][move.start_col()]) == PAWN &&
            move.start_col() != move.end_col() && position.half_moves + 1 == max_depth) {
            
            en_passant++;
            captures++;
        }

        position.make_move(move);
        perft(position, max_depth);
        position.pop();
    }
}

int main() {

    /* int init_board[8][8] = {
        {BLACK_ROOK, 0, 0, 0, BLACK_KING, 0, 0, BLACK_ROOK},
        {BLACK_PAWN, 0, BLACK_PAWN, BLACK_PAWN, BLACK_QUEEN, BLACK_PAWN, BLACK_BISHOP, 0},
        {BLACK_BISHOP, BLACK_KNIGHT, 0, 0, BLACK_PAWN, BLACK_KNIGHT, BLACK_PAWN, 0},
        {0, 0, 0, WHITE_PAWN, WHITE_KNIGHT, 0, 0, 0},
        {0, BLACK_PAWN, 0, 0, WHITE_PAWN, 0, 0, 0},
        {0, 0, WHITE_KNIGHT, 0, 0, WHITE_QUEEN, 0, BLACK_PAWN},
        {WHITE_PAWN, WHITE_PAWN, WHITE_PAWN, WHITE_BISHOP, WHITE_BISHOP, WHITE_PAWN, WHITE_PAWN, WHITE_PAWN},
        {WHITE_ROOK, 0, 0, 0, WHITE_KING, 0, 0, WHITE_ROOK}
    };
    Position position(init_board, WHITE, nnue); */

    std::string nnue = "nnue/nnue.bin";
    Position position(nnue);

    auto start_time = std::chrono::high_resolution_clock::now();
    perft(position, 6);
    auto end_time = std::chrono::high_resolution_clock::now();
    int time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    printf("time: %f\n", (float) time_taken / 1000);
    printf("nodes: %d\n", leafs);
    printf("captures: %d\n", captures);
    printf("en passant: %d\n", en_passant);
    printf("checks: %d\n", checks);
    printf("checkmates: %d\n", checkmates);
}