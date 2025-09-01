#ifndef move_hh
#define move_hh

#include <sstream>

// important for move data to be compact for transposition table.
// 4 bits for promotion (enum piece_type), 6 bits for from_square, 6 bits for to_square
class Move {
public:
    int exchange;
    uint16_t move;

    Move() : move(0), exchange(0) {}
    
    Move(uint8_t from_square, uint8_t to_square, uint8_t promote_to = 0, int exchange = 0) : exchange(exchange) {
        move = (promote_to << 12) | (from_square << 6) | to_square;
    }

    Move(uint16_t move_data) : move(move_data), exchange(0) {}

    Move(std::string uci_move) {
        int from_col = uci_move[0] - 'a';
        int from_row = 8 - (uci_move[1] - '0');
        int to_col = uci_move[2] - 'a';
        int to_row = 8 - (uci_move[3] - '0');
        int promotion = 0;
        if (uci_move.length() == 5) {
            char p = uci_move[4];
            if (p == 'n') promotion = 1;
            else if (p == 'b') promotion = 2;
            else if (p == 'r') promotion = 3;
            else if (p == 'q') promotion = 4;
        }

        int from_square = from_row * 8 + from_col;
        int to_square = to_row * 8 + to_col;
        move = (promotion << 12) | (from_square << 6) | to_square;
    }

    int from() const {
        return (move >> 6) & 0x3F;
    }

    int to() const {
        return move & 0x3F;
    }

    int promote_to() const {
        return (move >> 12) & 0xF;
    }

    int start_row() const {
        return from() / 8;
    }

    int start_col() const {
        return from() % 8;
    }

    int end_row() const {
        return to() / 8;
    }

    int end_col() const {
        return to() % 8;
    }

    std::string toString() const {
        std::stringstream ss;
        ss  << start_row() << start_col() << end_row() << end_col();
        int promotion = promote_to();
        if (promotion == 1) {
            ss << 'N';
        } else if (promotion == 2) {
            ss << 'B';
        } else if (promotion == 3) {
            ss << 'R';
        } else if (promotion == 4) {
            ss << 'Q';
        } else {
            ss << 'X';
        }
        return ss.str();
    }

    std::string to_uci() const {
        if (!move) {
            return "0000";
        }

        int start_rank = 8 - start_row();
        char start_file = 'a' + start_col();
        int end_rank = 8 - end_row();
        char end_file = 'a' + end_col();

        std::string uci_move = start_file + std::to_string(start_rank) +
            end_file + std::to_string(end_rank);

        if (promote_to()) {
            int promotion = promote_to();
            if (promotion == 1) uci_move += "n";
            else if (promotion == 2) uci_move += "b";
            else if (promotion == 3) uci_move += "r";
            else if (promotion == 4) uci_move += "q";
        }

        return uci_move;
    }

    bool operator==(const Move& other) const {
        return move == other.move;
    }

    bool operator!=(const Move& other) const {
        return move != other.move;
    }

    explicit operator bool() const {
        return move;
    }
};

#endif
