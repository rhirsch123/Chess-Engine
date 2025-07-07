#ifndef zobrist_hh
#define zobrist_hh

#include <cstdint>
#include <random>

class Zobrist {
public:
	uint64_t piece_table[12][64];
	uint64_t castle_table[4];
	uint64_t en_passant_table[8];
	uint64_t turn_key;

	Zobrist();

	uint64_t hash_position(const int board[8][8], int turn,
		const bool can_castle[4], int en_passant_col);
};

#endif