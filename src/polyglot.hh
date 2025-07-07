#ifndef polyglot_hh
#define polyglot_hh

#include <string>
#include <fstream>

#include "move.hh"
#include "position.hh"

namespace Polyglot {
    Move get_book_move(const Position& position, std::string file);
}

#endif