#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>

#include "position.hh"
#include "engine.hh"
#include "move.hh"
#include "movegen.hh"

// executable to interact with tools like fastchess, GUIs, etc.

int main(int argc, char * argv[]) {
    Engine engine;
    Position position;

    std::thread search_thread;
    std::atomic<bool> stop_search(false);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "uci") {
            printf("id name Shmembot\n");
            printf("id author Ryan Hirsch\n");

            printf("option name Hash type spin default %d min 1 max 1048576\n", engine.Hash);

            printf("uciok\n");
        } else if (line.rfind("setoption", 0) == 0) {
            std::istringstream iss(line);
            std::string token, name, value;
            iss >> token;

            while (iss >> token) {
                if (token == "name") {
                    iss >> name;
                } else if (token == "value") {
                    iss >> value;
                    break;
                }
            }

            if (name == "Hash") {
                int hash = std::stoi(value);
                hash = std::max(hash, 1);
                hash = std::min(hash, 1048576);
                engine.transposition_table.resize(hash);
            }
        } else if (line == "isready") {
            printf("readyok\n");
        } else if (line == "ucinewgame") {
            engine.reset();
        } else if (line.rfind("position", 0) == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token;

            std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

            iss >> token;
            if (token == "startpos") {
                iss >> token;
            } else if (token == "fen") {
                fen.clear();
                while (iss >> token && token != "moves") {
                    fen += token + " ";
                }
            }

            position.set_fen(fen);

            if (token == "moves") {
                while (iss >> token) {
                    position.make_move(Move(token));
                }
            }
        } else if (line.rfind("perft", 0) == 0) {
            std::istringstream iss(line);
            std::string arg;
            iss >> arg;
            iss >> arg;
            int depth = std::stoi(arg);
            printf("%llu\n", bulk_perft(position, depth));
        } else if (line.rfind("go", 0) == 0) {
            if (search_thread.joinable()) {
                stop_search.store(true);
                search_thread.join();
            }
            stop_search.store(false);

            int depth = 0, nodes = 0, movetime = 0, wtime = 0, btime = 0, winc = 0, binc = 0;

            std::istringstream iss(line);
            std::string token;
            iss >> token;
            while (iss >> token) {
                if (token == "infinite") depth = MAX_DEPTH;
                else if (token == "depth") iss >> depth;
                else if (token == "nodes") iss >> nodes;
                else if (token == "movetime") iss >> movetime;
                else if (token == "wtime") iss >> wtime;
                else if (token == "btime") iss >> btime;
                else if (token == "winc") iss >> winc;
                else if (token == "binc") iss >> binc;
            }

            SearchInfo info;
            if (depth) {
                info.fixed_depth = true;
                info.depth = depth;
            } else if (nodes) {
                info.fixed_nodes = true;
                info.nodes = nodes;
            } else if (movetime) {
                info.fixed_time = true;
                info.movetime = std::max(movetime - 30, 10);
            } else {
                info.timed_game = true;
                if (position.turn == WHITE) {
                    info.time_left = wtime;
                    info.increment = winc;
                } else {
                    info.time_left = btime;
                    info.increment = binc;
                }
            }
            info.uci = true;
            info.stop_search = &stop_search;

            search_thread = std::thread([&engine, &position, &info]() {
                Move best_move = engine.get_move(position, info);
                printf("bestmove %s\n", best_move.to_uci().c_str());
                fflush(stdout);
            });
            
        } else if (line == "stop") {
            stop_search.store(true);
            if (search_thread.joinable()) {
                search_thread.join();
            }
        } else if (line == "quit") {
            stop_search.store(true);
            if (search_thread.joinable()) {
                search_thread.join();
            }
            break;
        }
    }

    return 0;
}