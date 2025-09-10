#include <string>
#include <sstream>
#include <iostream>
#include <vector>

#include "position.hh"
#include "engine.hh"
#include "move.hh"

// run to interact with tools like cutechess-cli, banksiagui, etc.

int main(int argc, char * argv[]) {
	std::string dir = std::string(argv[0]);
	int i = dir.find_last_of("/\\");
	if (i == std::string::npos) {
		dir = ".";
	} else {
		dir = dir.substr(0, i);
	}

	std::string nnue = dir + "/nnue/nnue.bin";

    Engine engine(0, false);
    std::unique_ptr<Position> position;

    std::string line;
	while (std::getline(std::cin, line)) {
		if (line == "uci") {
			std::cout << "id name Shmembot\n";
			std::cout << "id author Ryan Hirsch\n";
			std::cout << "uciok\n";
		} 
		else if (line == "isready") {
			std::cout << "readyok\n";
		} 
		else if (line == "ucinewgame") {
            engine.reset();
		} 
		else if (line.rfind("position", 0) == 0) {
			std::istringstream iss(line);
			std::string token;
			iss >> token;

			std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

			iss >> token;
			if (token == "startpos") {
				iss >> token;
			} else if (token == "fen") {
				fen.clear();
				while (iss >> token && token != "moves")
					fen += token + " ";
			}

			std::vector<std::string> moves;
			if (token == "moves") {
				while (iss >> token)
					moves.push_back(token);
			}

            position = std::make_unique<Position>(fen, nnue);
            for (auto move_str : moves) {
                position->make_move(Move(move_str));
            }
		} 
		else if (line.rfind("go", 0) == 0) {
			int depth = 0, movetime = 0, wtime = 0, btime = 0, winc = 0, binc = 0;

			std::istringstream iss(line);
			std::string token;
			iss >> token;
			while (iss >> token) {
				if (token == "depth") iss >> depth;
				else if (token == "movetime") iss >> movetime;
				else if (token == "wtime") iss >> wtime;
				else if (token == "btime") iss >> btime;
				else if (token == "winc") iss >> winc;
				else if (token == "binc") iss >> binc;
			}

            if (depth) {
                engine.timed_game = false;
				engine.time_per_move = INF;
                engine.MAX_DEPTH = depth;
            } else if (movetime) {
                engine.timed_game = false;
                engine.time_per_move = movetime;
            } else {
                engine.timed_game = true;
                if (position->turn == WHITE) {
                    engine.time_left = wtime;
                    engine.increment = (double) winc / 1000;
                } else {
                    engine.time_left = btime;
                    engine.increment = (double) binc / 1000;
                }
            }

            Move best_move = engine.get_move(*position);
			std::cout << "bestmove " << best_move.to_uci() << "\n";
		} 
		else if (line == "quit") {
			break;
		}
	}

    return 0;
}