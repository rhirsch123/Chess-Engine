#include "move.hh"
#include "position.hh"
#include "engine.hh"

#define LEARN_RATE 0.01

// Joona Kiiski's practical tuning method used for Stockfish
// evaluation/search parameters are adjusted one by one
int main() {
    float delta = 25;
    float weight = 10;
    for (int i = 0; i < 10000; i++) {
        printf("i: %d\n", i);
        Position position;

        Engine tune_up_engine((float) 0.002, 0.005);
        Engine tune_down_engine((float) 0.002, 0.005);

        tune_up_engine.ASPIRATION_DELTA = weight + delta;
        tune_down_engine.ASPIRATION_DELTA = weight - delta;

        std::string terminal;
        while (1) {
            terminal = position.get_terminal_state();
            if (!terminal.empty()) {
                printf("game end: %s\n", terminal.c_str());
                printf("%d moves\n", position.half_moves / 2);
                break;
            }

            // alternate which plays as white each game
            Move move;
            if (i % 2 == 0) {
                if (position.turn == WHITE) {
                    move = tune_up_engine.get_move(position);
                } else {
                    move = tune_down_engine.get_move(position);
                }
            } else {
                if (position.turn == BLACK) {
                    move = tune_up_engine.get_move(position);
                } else {
                    move = tune_down_engine.get_move(position);
                }
            }

            position.make_move(move);
        }

        if ((terminal == "white win" && i % 2 == 0) ||
            (terminal == "black win" && i % 2 == 1)) {
            weight += delta * LEARN_RATE;
        } else if (position.get_draw().empty()) {
            weight -= delta * LEARN_RATE;
        }

        printf("new weight = %.3f;\n", weight);
    }
}