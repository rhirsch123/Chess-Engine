#include "move.hh"
#include "position.hh"
#include "engine.hh"

#define LEARN_RATE 0.01

int main() {
    std::string nnue = "nnue/nnue.bin";

    float delta = 20;
    float weight = 100;
    for (int i = 0; i < 10000; i++) {
        printf("i: %d\n", i);
        Position position(nnue);

        Engine tune_up_engine(0.01, 0.01, true);
        Engine tune_down_engine(0.01, 0.01, true);

        tune_up_engine.RFP_SCALE = weight + delta;
        tune_down_engine.RFP_SCALE = weight - delta;

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