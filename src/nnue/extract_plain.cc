#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cstdint>
#include <limits>

// extract binary training data from the .plain version of a binpack file

std::vector<uint64_t> encode_board(std::string& fen, int to_square, char * turn, bool * is_capture) {    
    // bitboards: white, black, pawns, knights, bishops, rooks, queens, kings
    std::vector<uint64_t> encoded(8, 0);
    
    std::istringstream fen_stream(fen);
    std::string fen_board, fen_castling, fen_ep, fen_turn;
    int full_moves, half_moves;

    fen_stream >> fen_board >> fen_turn >> fen_castling >> fen_ep >> half_moves >> full_moves;
    *turn = fen_turn[0];

    *is_capture = false;

    int index = 0;
    for (char c : fen_board) {
        if (c == '/') {
            continue;
        } else if (isdigit(c)) {
            index += c - '0';
        } else {
            if (c == 'P') {
                encoded[0] |= 1ULL << index;
                encoded[2] |= 1ULL << index;
            } else if (c == 'N') {
                encoded[0] |= 1ULL << index;
                encoded[3] |= 1ULL << index;
            } else if (c == 'B') {
                encoded[0] |= 1ULL << index;
                encoded[4] |= 1ULL << index;
            } else if (c == 'R') {
                encoded[0] |= 1ULL << index;
                encoded[5] |= 1ULL << index;
            } else if (c == 'Q') {
                encoded[0] |= 1ULL << index;
                encoded[6] |= 1ULL << index;
            } else if (c == 'K') {
                encoded[0] |= 1ULL << index;
                encoded[7] |= 1ULL << index;
            } else if (c == 'p') {
                encoded[1] |= 1ULL << index;
                encoded[2] |= 1ULL << index;
            } else if (c == 'n') {
                encoded[1] |= 1ULL << index;
                encoded[3] |= 1ULL << index;
            } else if (c == 'b') {
                encoded[1] |= 1ULL << index;
                encoded[4] |= 1ULL << index;
            } else if (c == 'r') {
                encoded[1] |= 1ULL << index;
                encoded[5] |= 1ULL << index;
            } else if (c == 'q') {
                encoded[1] |= 1ULL << index;
                encoded[6] |= 1ULL << index;
            } else if (c == 'k') {
                encoded[1] |= 1ULL << index;
                encoded[7] |= 1ULL << index;
            }

            if (index == to_square) {
                *is_capture = true;
            }
            index++;
        }
    }

    return encoded;
}

struct label {
    int16_t eval;
    uint16_t game_ply;
    int8_t result;
    int8_t turn;
};


int main() {
    std::string positions_out = "positions.bin";
    std::string labels_out = "labels.bin";

    // clear
    std::ofstream(positions_out, std::ios::trunc);
    std::ofstream(labels_out, std::ios::trunc);

    std::string input_file = "binpack_data.plain";

    std::vector<std::vector<uint64_t>> positions;
    std::vector<label> labels;

    int num_positions = 0;
    const int batch_size = 100000;

    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        perror("error opening input file");
        return 1;
    }

    std::string line;
    std::string fen, move;
    int16_t score = 0;
    int16_t ply = 0;
    int8_t result = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    int count = 0;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;

        if (line.rfind("fen ", 0) == 0) {
            fen = line.substr(4);
            if (fen == "8/8/8/8/8/8/8/8 w - - 0 1") {
                break;
            }
        } else if (line.rfind("move ", 0) == 0) {
            move = line.substr(5);
        } else if (line.rfind("score ", 0) == 0) {
            score = std::stoi(line.substr(6));
        } else if (line.rfind("ply ", 0) == 0) {
            ply = std::stoi(line.substr(4));
        } else if (line.rfind("result ", 0) == 0) {
            result = std::stoi(line.substr(7));
        } else if (line == "e") {
            count++;
            if (ply > 5 && score != 0 && score < 2000 && score > -2000 && count % 2 == 0) {
                int to_col = move[2] - 'a';
                int to_row = 8 - (move[3] - '0');
                int to_square = to_row * 8 + to_col;
                bool is_capture;
                char turn;
                int16_t half_moves;

                auto encoded = encode_board(fen, to_square, &turn, &is_capture);

                if (is_capture) {
                    continue;
                }

                positions.push_back(encoded);
                num_positions++;

                label l = {
                    score,
                    ply,
                    result,
                    static_cast<int8_t>(turn == 'w' ? 0 : 1)
                };
                labels.push_back(l);

                if (num_positions % 10000 == 0)
                    std::cout << "position: " << num_positions << "\n";

                if (num_positions % batch_size == 0) {
                    // shuffle
                    std::vector<size_t> perm(positions.size());
                    std::iota(perm.begin(), perm.end(), 0);
                    std::shuffle(perm.begin(), perm.end(), gen);

                    std::ofstream pos_file(positions_out, std::ios::binary | std::ios::app);
                    for (size_t i : perm)
                        pos_file.write(reinterpret_cast<char*>(positions[i].data()), 8 * sizeof(uint64_t));

                    std::ofstream label_file(labels_out, std::ios::binary | std::ios::app);
                    for (size_t i : perm) {
                        label_file.write(reinterpret_cast<char*>(&labels[i]), sizeof(label));
                    }

                    positions.clear();
                    labels.clear();
                }
            }
        }
    }

    if (!positions.empty()) {
        std::vector<size_t> perm(positions.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), gen);

        std::ofstream pos_file(positions_out, std::ios::binary | std::ios::app);
        for (size_t i : perm)
            pos_file.write(reinterpret_cast<char*>(positions[i].data()), 8 * sizeof(uint64_t));

        std::ofstream label_file(labels_out, std::ios::binary | std::ios::app);
        for (size_t i : perm) {
            label_file.write(reinterpret_cast<char*>(&labels[i]), sizeof(label));
        }
    }

    std::cout << "positions: " << num_positions << "\n";
    return 0;
}