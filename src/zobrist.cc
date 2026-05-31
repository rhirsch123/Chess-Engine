#include "zobrist.hh"

namespace Zobrist {
    uint64_t piece_table[12][64];
    uint64_t castle_table[4];
    uint64_t en_passant_table[8];
    uint64_t turn_key;
    uint64_t pawn_table[64];

    uint64_t seed = 314159265;
    static uint64_t random_u64() {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        return seed * 2685821657736338717LL;
    }

    void init() {
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 64; j++) {
                piece_table[i][j] = random_u64();
            }
        }
    
        for (int i = 0; i < 4; i++) {
            castle_table[i] = random_u64();
        }
        
        for (int i = 0; i < 8; i++) {
            en_passant_table[i] = random_u64();
        }
        
        turn_key = random_u64();

        for (int i = 0; i < 64; i++) {
            pawn_table[i] = random_u64();
        }
    }

    uint64_t get_hash(uint8_t board[64], int castle_rights, int ep_col, int turn) {
        uint64_t hash = 0ULL;

        for (int square = 0; square < 64; square++) {
            int piece = board[square];
            if (piece) {
                hash ^= piece_table[piece - 1][square];
            }
        }

        for (int i = 0; i < 4; i++) {
            if (castle_rights & (1 << i)) {
                hash ^= castle_table[i];
            }
        }

        if (ep_col >= 0 && ep_col < 8) {
            hash ^= en_passant_table[ep_col];
        }

        if (turn == BLACK) {
            hash ^= turn_key;
        }

        return hash;
    }

    uint64_t get_pawn_hash(uint64_t pawns) {
        uint64_t hash = 0ULL;
        while (pawns) {
            hash ^= pawn_table[pop_lsb(pawns)];
        }
        return hash;
    }
}