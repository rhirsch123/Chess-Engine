#include "zobrist.hh"

Zobrist::Zobrist() {
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<uint64_t> dis;

	for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 64; j++) {
			piece_table[i][j] = dis(gen);
		}
	}
   
	for (int i = 0; i < 4; i++) {
		castle_table[i] = dis(gen);
	}
	
	for (int i = 0; i < 8; i++) {
		en_passant_table[i] = dis(gen);
	}
	
	turn_key = dis(gen);
}

uint64_t Zobrist::hash_position(const int board[8][8], int turn, const bool can_castle[4],
								int en_passant_col) {
	uint64_t hash = 0ULL;

	for (int row = 0; row < 8; row++) {
		for (int col = 0; col < 8; col++) {
			char piece = board[row][col];
			if (piece) {
				int square = row * 8 + col;
				hash ^= piece_table[piece - 1][square];
			}
		}
	}

	for (int i = 0; i < 4; i++) {
		if (can_castle[i]) {
			hash ^= castle_table[i];
		}
	}

	if (en_passant_col >= 0 && en_passant_col < 8) {
		hash ^= en_passant_table[en_passant_col];
	}

	if (turn == 1) { // black
		hash ^= turn_key;
	}

	return hash;
}