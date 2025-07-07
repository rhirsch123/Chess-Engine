#ifndef evaluation_hh
#define evaluation_hh

#include "position.hh"
#include "move.hh"
#include "engine.hh"

#define LIGHT_SQUARES 0xAA55AA55AA55AA55ULL
#define DARK_SQUARES 0x55AA55AA55AA55AAULL

namespace Evaluation {
	bool safe_move(Position& position, Move move, int piece = 0);
	int evaluation(Position& position, Engine& engine);
}

#endif