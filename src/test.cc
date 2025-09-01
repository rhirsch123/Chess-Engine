#include <string>

#include "move.hh"
#include "position.hh"
#include "engine.hh"


int main(int argc, char * argv[]) {
    int old_count = 0;
    int new_count = 0;
    int draw_count = 0;

    int num_games = argc == 1 ? 100 : atoi(argv[1]);

    std::string nnue = "nnue/nnue.bin";

    for (int i = 0; i < num_games; i++) {
        printf("i: %d\n", i);

        Position position(nnue);

        Engine old_engine(0.02, 0.02, true);
        Engine new_engine(0.02, 0.02, true);

        new_engine.test_condition = true;

        std::string terminal;
        while (1) {
            terminal = position.get_terminal_state();
            if (!terminal.empty()) {
                break;
            }

            // alternate which plays as white each game
            Move move;
            if (i % 2 == 0) {
                if (position.turn == WHITE) {
                    move = old_engine.get_move(position);
                } else {
                    move = new_engine.get_move(position);
                }
            } else {
                if (position.turn == BLACK) {
                    move = old_engine.get_move(position);
                } else {
                    move = new_engine.get_move(position);
                }
            }

            position.make_move(move);
        }

        if (!position.get_draw().empty()) {
            draw_count++;
        } else if ((terminal == "white win" && i % 2 == 0) ||
            (terminal == "black win" && i % 2 == 1)) {
            old_count++;
        } else {
            new_count++;
        }

        printf("old: %d\n", old_count);
        printf("new: %d\n", new_count);
        printf("draw: %d\n\n", draw_count);
    }
}