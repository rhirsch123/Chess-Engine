#ifndef zobrist_hh
#define zobrist_hh

#include <cstdint>

#include "bitboards.hh"
#include "types.hh"

namespace Zobrist {
    extern uint64_t piece_table[12][64];
    extern uint64_t castle_table[4];
    extern uint64_t en_passant_table[8];
    extern uint64_t turn_key;

    void init();
};

#endif