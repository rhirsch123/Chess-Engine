#ifndef transposition_table_hh
#define transposition_table_hh

#include <cstdint>
#include <vector>
#include <cstring>

#include "move.hh"
#include "types.hh"

class TranspositionTable {
    TTEntry* table = nullptr;
    size_t size = 0;

public:

    TranspositionTable(size_t megabytes) {
        resize(megabytes);
    }

    ~TranspositionTable() {
        if (table) {
            free(table);
        }
    }

    uint64_t get_index(uint64_t key) {
        #if defined(__SIZEOF_INT128__)
            return static_cast<uint64_t>(((static_cast<__uint128_t>(key) * static_cast<__uint128_t>(size)) >> 64));
        #else
            return key % size;
        #endif
    }

    void prefetch(uint64_t key) {
        __builtin_prefetch(&table[get_index(key)]);
    }

    void insert(uint64_t key, int16_t value, int16_t static_eval, uint16_t best_move, TTBound type, int8_t depth) {
        const uint64_t index = get_index(key);
        TTEntry& current = table[index];
        uint16_t key16 = uint16_t(key);

        if (best_move || key16 != current.key) {
            current.best_move = best_move;
        }

        if (   key16 != current.key
            || depth >= current.depth
            || (type == EXACT_BOUND && current.type != EXACT_BOUND)) {

            current.key = key16;
            current.value = value;
            current.static_eval = static_eval;
            current.type = type;
            current.depth = depth;
        }
    }

    bool get(uint64_t key, TTEntry& entry) {
        const uint64_t index = get_index(key);
        const TTEntry& candidate = table[index];

        if (candidate.key && candidate.key == uint16_t(key)) {
            entry = candidate;
            return true;
        }
        return false;
    }

    void clear() {
        std::memset(table, 0, size * sizeof(TTEntry));
    }

    void resize(int megabytes) {
        if (table) {
            free(table);
        }

        size_t bytes = megabytes * 1024 * 1024;
        size = bytes / sizeof(TTEntry);
        table = (TTEntry *) malloc(bytes);
        clear();
    }
};

#endif