#ifndef transposition_table_hh
#define transposition_table_hh

#include <cstdint>
#include <vector>

#include "move.hh"

enum bound_type : uint8_t {
    EXACT,
    LOWER_BOUND,
    UPPER_BOUND
};
struct tt_entry {
    uint64_t key = 0ULL;
    int value;
    uint16_t best_move;
    bound_type type;
    uint8_t depth;
};

class TranspositionTable {
    std::vector<tt_entry> table;
    size_t size;
    bool always_replace;

public:
    TranspositionTable(size_t megabytes, bool always_replace) : always_replace(always_replace) {
        size_t bytes = megabytes * 1024 * 1024;
        size = bytes / sizeof(tt_entry);
        table.resize(size);
    }

    void insert(const tt_entry& entry) {
        size_t index = entry.key % size;
		tt_entry& current = table[index];

        if (always_replace) {
            if (current.key != entry.key || entry.depth >= current.depth || (entry.type == EXACT && current.type != EXACT)) {
                current = entry;
            }
        } else {
		    if (!current.key || entry.depth >= current.depth) {
		    	current = entry;
		    }
        }
    }

    bool get(uint64_t key, tt_entry& entry) {
		size_t index = key % size;
		const tt_entry& candidate = table[index];

		if (candidate.key && candidate.key == key) {
			entry = candidate;
			return true;
		}
		return false;
	}
};

#endif