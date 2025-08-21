#ifndef evaluation_hh
#define evaluation_hh

#include "position.hh"
#include "move.hh"
#include "engine.hh"

#define LIGHT_SQUARES 0xAA55AA55AA55AA55ULL
#define DARK_SQUARES 0x55AA55AA55AA55AAULL

namespace Evaluation {
    // precomputed bitboards of squares around king considered for king attackers
    // every square adjacent to king, plus 3 more squares facing opponent's side
    const uint64_t king_rings[64] = {
        0x30302, 0x70705, 0xe0e0a, 0x1c1c14, 
        0x383828, 0x707050, 0xe0e0a0, 0xc0c040, 
        0x3030203, 0x7070507, 0xe0e0a0e, 0x1c1c141c, 
        0x38382838, 0x70705070, 0xe0e0a0e0, 0xc0c040c0, 
        0x303020300, 0x707050700, 0xe0e0a0e00, 0x1c1c141c00, 
        0x3838283800, 0x7070507000, 0xe0e0a0e000, 0xc0c040c000, 
        0x30302030000, 0x70705070000, 0xe0e0a0e0000, 0x1c1c141c0000, 
        0x383828380000, 0x707050700000, 0xe0e0a0e00000, 0xc0c040c00000, 
        0x30203030000, 0x70507070000, 0xe0a0e0e0000, 0x1c141c1c0000, 
        0x382838380000, 0x705070700000, 0xe0a0e0e00000, 0xc040c0c00000, 
        0x3020303000000, 0x7050707000000, 0xe0a0e0e000000, 0x1c141c1c000000, 
        0x38283838000000, 0x70507070000000, 0xe0a0e0e0000000, 0xc040c0c0000000, 
        0x302030300000000, 0x705070700000000, 0xe0a0e0e00000000, 0x1c141c1c00000000, 
        0x3828383800000000, 0x7050707000000000, 0xe0a0e0e000000000, 0xc040c0c000000000, 
        0x203030000000000, 0x507070000000000, 0xa0e0e0000000000, 0x141c1c0000000000, 
        0x2838380000000000, 0x5070700000000000, 0xa0e0e00000000000, 0x40c0c00000000000
    };

    // bonuses indexed by number of possible moves not controlled by opponent's pawns
    const int knight_mobility_bonus[9] = {
        -15, -7, -4, -2, 1, 3, 5, 8, 13
    };

    const int bishop_mobility_bonus[14] = {
        -15, -14, -10, -8, -6, -4, -2, -1, 1, 4, 7, 13, 20, 35
    };

    const int rook_mobility_bonus[15] = {
        -20, -17, -16, -14, -13, -11, -8, -4, -2, 1, 6, 7, 10, 19, 25
    };

    const int queen_mobility_bonus[28] = {
        -45, -35, -30, -15, -8, 0, 5, 8, 10, 13, 15, 18, 20, 23,
        25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25
    };


    // king tropism bonuses indexed by piece type
    const int near_opp_king_bonus[6] = {
        0, 3, -1, 5, 6, 0
    };


    // indexed by piece type
    const int king_attack_weights[5] = { 0, 2, 2, 3, 5 };
    const int check_weights[5] = { 0, 1, 1, 2, 3 };

    // indexed by sum of attack weights
    const int king_danger[50] = {
        0, 0, 1, 2, 3, 5, 7, 9, 12, 15,
        18, 22, 26, 30, 35, 40, 45, 50, 56, 62,
        68, 75, 82, 87, 92, 97, 105, 113, 122, 131,
        140, 150, 170, 180, 190, 200, 213, 225, 237, 250,
        260, 272, 283, 295, 307, 320, 330, 342, 355, 365
    };

    bool safe_move(Position& position, Move& move);
    int evaluation(Position& position, Engine& engine);
}

#endif