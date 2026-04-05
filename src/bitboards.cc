#include "bitboards.hh"

// for debugging
void print_bitboard(uint64_t bitboard) {
    for (int i = 0; i < 64; i++) {
        if (i % 8 == 0) {
            printf("\n");
        }
        printf("%c", (bitboard & (1ULL << i)) ? '1' : '0');
    }
    printf("\n\n");
}

// given a mask, generate all possible subsets of occupancy
std::vector<uint64_t> get_all_subsets(uint64_t mask) {
    std::vector<int> bit_positions;
    for (int i = 0; i < 64; i++) {
        if (mask & (1ULL << i)) {
            bit_positions.push_back(i);
        }
    }

    std::vector<uint64_t> variations;
    for (int subset = 0; subset < (1 << bit_positions.size()); subset++) {
        uint64_t occupancy = 0ULL;
        for (int i = 0; i < bit_positions.size(); i++) {
            if (subset & (1 << i)) {
                occupancy |= (1ULL << bit_positions[i]);
            }
        }
        variations.push_back(occupancy);
    }
    return variations;
}

// get rook attack mask excluding edges and not considering blocking
uint64_t get_rook_mask(int square) {
    uint64_t mask = 0ULL;
    int row = square / 8;
    int col = square % 8;

    // horizontal
    for (int i = col - 1; i >= 1; i--) {
        mask |= square_bitboard(row * 8 + i);
    }
    for (int i = col + 1; i <= 6; i++) {
        mask |= square_bitboard(row * 8 + i);
    }

    // vertical
    for (int i = row - 1; i >= 1; i--) {
        mask |= square_bitboard(i * 8 + col);
    }
    for (int i = row + 1; i <= 6; i++) {
        mask |= square_bitboard(i * 8 + col);
    }

    return mask;
}

// generate bitboard of attacked squares given position of rook and board occupancy
uint64_t get_rook_attacks(int square, uint64_t occupancy) {
    uint64_t attacks = 0ULL;
    int row = square / 8;
    int col = square % 8;

    // left
    for (int i = col - 1; i >= 0; i--) {
        uint64_t bb = square_bitboard(row * 8 + i);
        attacks |= bb;
        if (occupancy & bb) {
            break;
        }
    }
    // right
    for (int i = col + 1; i < 8; i++) {
        uint64_t bb = square_bitboard(row * 8 + i);
        attacks |= bb;
        if (occupancy & bb) {
            break;
        }
    }
    // up
    for (int i = row - 1; i >= 0; i--) {
        uint64_t bb = square_bitboard(i * 8 + col);
        attacks |= bb;
        if (occupancy & bb) {
            break;
        }
    }
    // down
    for (int i = row + 1; i < 8; i++) {
        uint64_t bb = square_bitboard(i * 8 + col);
        attacks |= bb;
        if (occupancy & bb) {
            break;
        }
    }

    return attacks;
}


// same for bishops:

uint64_t get_bishop_mask(int square) {
    uint64_t mask = 0ULL;
    int row = square / 8;
    int col = square % 8;

    // top right
    for (int r = row + 1, c = col + 1; r < 7 && c < 7; r++, c++) {
        mask |= square_bitboard(r * 8 + c);
    }
    // top left
    for (int r = row + 1, c = col - 1; r < 7 && c > 0; r++, c--) {
        mask |= square_bitboard(r * 8 + c);
    }
    // bottom right
    for (int r = row - 1, c = col + 1; r > 0 && c < 7; r--, c++) {
        mask |= square_bitboard(r * 8 + c);
    }
    // bottom left
    for (int r = row - 1, c = col - 1; r > 0 && c > 0; r--, c--) {
        mask |= square_bitboard(r * 8 + c);
    }

    return mask;
}

uint64_t get_bishop_attacks(int square, uint64_t occupancy) {
    uint64_t attacks = 0ULL;
    int row = square / 8;
    int col = square % 8;

    // top right
    for (int r = row + 1, c = col + 1; r < 8 && c < 8; r++, c++) {
        uint64_t bb = square_bitboard(r * 8 + c);
        attacks |= bb;
        if (occupancy & bb) {
            break;
        }
    }
    // top left
    for (int r = row + 1, c = col - 1; r < 8 && c >= 0; r++, c--) {
        uint64_t bb = square_bitboard(r * 8 + c);
        attacks |= bb;
        if (occupancy & bb) {
            break;
        }
    }
    // bottom right
    for (int r = row - 1, c = col + 1; r >= 0 && c < 8; r--, c++) {
        uint64_t bb = square_bitboard(r * 8 + c);
        attacks |= bb;
        if (occupancy & bb) {
            break;
        }
    }
    // bottom left
    for (int r = row - 1, c = col - 1; r >= 0 && c >= 0; r--, c--) {
        uint64_t bb = square_bitboard(r * 8 + c);
        attacks |= bb;
        if (occupancy & bb) {
            break;
        }
    }

    return attacks;
}


static uint64_t random_U64() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, ~0ULL);
    return dist(gen);
}

// generate perfect hash function for occupancies to rook/bishop attack index through brute force
// values are hard-coded into main code
static uint64_t find_magic_number(int square, bool rook) {
    int relevant_bits; // max relevant bits that can block attacks
    uint64_t mask;
    if (rook) {
        mask = get_rook_mask(square);
        relevant_bits = 12;
    } else {
        // bishop
        mask = get_bishop_mask(square);
        relevant_bits = 9;
    }

    std::vector<uint64_t> occupancies = get_all_subsets(mask);
    std::unordered_map<uint64_t, uint64_t> attack_table;

    while (true) {
        uint64_t magic = random_U64() & random_U64() & random_U64(); // sparse
            
        bool found = true;
        attack_table.clear();

        for (uint64_t o : occupancies) {
            uint64_t index = (o * magic) >> (64 - relevant_bits);
            uint64_t attacks = rook ? get_rook_attacks(square, o) : get_bishop_attacks(square, o);

            if (attack_table.count(index) && attack_table[index] != attacks) {
                found = false;
                break;
            } else {
                attack_table[index] = attacks;
            }
        }

        if (found) {
            return magic;
        }
    }
}

/*
Ex:
int main() {
    printf("rook:\n");
    for (int square = 0; square < 64; square++) {
        if (square % 4 == 0) printf("\n");
        printf("0x%llx, ", find_magic_number(square, true));
    }

    printf("\n\nbishop:\n");
    for (int square = 0; square < 64; square++) {
        if (square % 4 == 0) printf("\n");
        printf("0x%llx, ", find_magic_number(square, false));
    }

    return 0;
}
*/


// row and column mask by square, excluding edges
uint64_t rook_masks[64];
// precompute rook moves given square and board occupancy.
// edges of the board can be excluded in occupancy because the rook must stop there.
// this gives a max of 12 possible blockers per rook position. 2 ^ 12 = 4096.
uint64_t rook_moves[64][4096];

// similar for bishops
uint64_t bishop_masks[64];
uint64_t bishop_moves[64][512];

// precomputed moves for each square
uint64_t pawn_attacks[64][2];
uint64_t knight_moves[64];
uint64_t king_moves[64];

// for square1 in check by square2, holds the squares that could block or capture the checker
uint64_t check_blocks[64][64];

void init_bitboards() {
    std::vector<std::tuple<int ,int>> knight_dirs = {
        {2,1}, {2,-1}, {-2,1}, {-2,-1}, {1,2}, {1,-2}, {-1,2}, {-1,-2}
    };
    std::vector<std::tuple<int ,int>> king_dirs = {
        {1,0}, {-1,0}, {0,1}, {0,-1}, {1,1}, {-1,-1}, {1,-1}, {-1,1}
    };

    for (int square = 0; square < 64; square++) {
        int row = square / 8;
        int col = square % 8;

        // knight moves
        uint64_t moves = 0ULL;
        for (auto d : knight_dirs) {
            int new_row = row + std::get<0>(d);
            int new_col = col + std::get<1>(d);
            if (new_row >= 0 && new_row < 8 && new_col >= 0 && new_col < 8) {
                moves |= square_bitboard(new_row * 8 + new_col);
            }
        }
        knight_moves[square] = moves;
        
        // pawn attacks
        uint64_t white_attacks = 0ULL;
        uint64_t black_attacks = 0ULL;
        if (col > 0) {
            if (row > 0) {
                white_attacks |= square_bitboard((row - 1) * 8 + (col - 1));
            }
            if (row < 7) {
                black_attacks |= square_bitboard((row + 1) * 8 + (col - 1));
            }
        }
        if (col < 7) {
            if (row > 0) {
                white_attacks |= square_bitboard((row - 1) * 8 + (col + 1));
            }
            if (row < 7) {
                black_attacks |= square_bitboard((row + 1) * 8 + (col + 1));
            }
        }
        pawn_attacks[square][WHITE] = white_attacks;
        pawn_attacks[square][BLACK] = black_attacks;

        // king moves
        moves = 0ULL;
        for (auto d : king_dirs) {
            int new_row = row + std::get<0>(d);
            int new_col = col + std::get<1>(d);
            if (new_row >= 0 && new_row < 8 && new_col >= 0 && new_col < 8) {
                moves |= square_bitboard(new_row * 8 + new_col);
            }
        }
        king_moves[square] = moves;

        // rook moves
        rook_masks[square] = get_rook_mask(square);
        std::vector<uint64_t> occupancies = get_all_subsets(rook_masks[square]);
        for (uint64_t occ : occupancies) {
            uint64_t index = (occ * rook_magic_numbers[square]) >> (64 - 12);
            rook_moves[square][index] = get_rook_attacks(square, occ);
        }

        // bishop moves
        bishop_masks[square] = get_bishop_mask(square);
        occupancies = get_all_subsets(bishop_masks[square]);
        for (uint64_t occ : occupancies) {
            uint64_t index = (occ * bishop_magic_numbers[square]) >> (64 - 9);
            bishop_moves[square][index] = get_bishop_attacks(square, occ);
        }
    }

    // check blocks
    for (int s1 = 0; s1 < 64; s1++) {
        for (int s2 = 0; s2 < 64; s2++) {
            if (get_bishop_moves(s1, 0ULL) & square_bitboard(s2)) {
                check_blocks[s1][s2] = get_bishop_moves(s1, square_bitboard(s2)) &
                                       get_bishop_moves(s2, square_bitboard(s1));
            } else if (get_rook_moves(s1, 0ULL) & square_bitboard(s2)) {
                check_blocks[s1][s2] = get_rook_moves(s1, square_bitboard(s2)) &
                                       get_rook_moves(s2, square_bitboard(s1));
            } else {
                check_blocks[s1][s2] = 0ULL;
            }

            // capture the checker
            check_blocks[s1][s2] |= square_bitboard(s2);
        }
    }
}


uint64_t get_check_blocks(int s1, int s2) {
    return check_blocks[s1][s2];
}


uint64_t get_pawn_attacks(int square, int color) {
    return pawn_attacks[square][color];
}

uint64_t get_knight_moves(int square) {
    return knight_moves[square];
}

uint64_t get_king_moves(int square) {
    return king_moves[square];
}

uint64_t get_bishop_moves(int square, uint64_t blockers) {
    blockers &= bishop_masks[square];
    uint64_t index = (blockers * bishop_magic_numbers[square]) >> (64 - 9);
    return bishop_moves[square][index];
}

uint64_t get_rook_moves(int square, uint64_t blockers) {
    blockers &= rook_masks[square];
    uint64_t index = (blockers * rook_magic_numbers[square]) >> (64 - 12);
    return rook_moves[square][index];
}

uint64_t get_queen_moves(int square, uint64_t blockers) {
    return get_bishop_moves(square, blockers) | get_rook_moves(square, blockers);
}