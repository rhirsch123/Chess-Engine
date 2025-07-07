#ifndef move_hh
#define move_hh

#include <sstream>

// important for move data to be compact for transposition table.
// 4 bits for promotion (enum piece_type), 6 bits for from_square, 6 bits for to_square
class Move {
public:
    int priority;
    uint16_t move;

    Move() : move(0), priority(0) {}
    
    Move(uint8_t from_square, uint8_t to_square, uint8_t promote_to = 0, int priority = 0) : priority(priority) {
        move = (promote_to << 12) | (from_square << 6) | to_square;
    }

    Move(uint16_t move_data) : move(move_data), priority(0) {}

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

    bool operator==(const Move& other) const {
        return move == other.move;
    }
};

#endif
