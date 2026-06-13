#include <chrono>
#include <string>

#include "move.hh"
#include "position.hh"
#include "engine.hh"
#include "movegen.hh"
#include "movepick.hh"


Engine engine;

uint64_t leafs = 0;
uint64_t captures = 0;
uint64_t en_passant = 0;
uint64_t castles = 0;
uint64_t checks = 0;
uint64_t double_checks = 0;
uint64_t checkmates = 0;
void perft(Position& position, int max_depth) {
    if (position.half_moves == max_depth) {
        leafs++;

        if (position.checkers()) {
            checks++;
            if (popcount(position.checkers()) > 1) double_checks++;
            if (position.get_terminal_state() && !position.get_draw()) checkmates++;
        }

        return;
    }

    bool next_leaf = position.half_moves + 1 == max_depth;

    MovePicker movegen(position, engine, position.half_moves);
    while (true) {
        Move move = movegen.next_move();
        if (!move) break;

        int cap = position.piece_on(move.to());
        if (cap && next_leaf) {
            captures++;
        }

        int piece_type = Position::get_piece_type(position.piece_on(move.from()));
        if (!cap && piece_type == PAWN && move.from_col() != move.to_col() && next_leaf) {
            en_passant++;
            captures++;
        } else if (piece_type == KING && std::abs(move.from_col() - move.to_col()) > 1 && next_leaf) {
            castles++;
        }

        position.make_move(move);
        perft(position, max_depth);
        position.pop();
    }
}

int main(int argc, char* argv[]) {
    Position position;

    int depth = 6;
    bool bulk = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-b") {
            bulk = true;
        } else {
            depth = std::stoi(argv[i]);
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    if (bulk) {
        leafs = bulk_perft(position, depth);
    } else {
        perft(position, depth);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    int time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    printf("time: %f\n", (float) time_taken / 1000);
    printf("nodes: %llu\n", leafs);
    printf("nps: %f M\n", (double) leafs / (time_taken * 1000));
    printf("captures: %llu\n", captures);
    printf("en passant: %llu\n", en_passant);
    printf("castles: %llu\n", castles);
    printf("checks: %llu\n", checks);
    printf("double checks: %llu\n", double_checks);
    printf("checkmates: %llu\n", checkmates);
}