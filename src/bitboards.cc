#include <vector>
#include <cstdint>
#include <random>
#include <unordered_map>

// get rook attack mask (excluding edges) not considering blocking
uint64_t get_rook_mask(int square) {
    uint64_t mask = 0ULL;
    int row = square / 8;
    int col = square % 8;

    // horizontal
    for (int i = col - 1; i >= 1; i--) {
        mask |= (1ULL << (row * 8 + i));
    }
    for (int i = col + 1; i <= 6; i++) {
        mask |= (1ULL << (row * 8 + i));
    }

    // vertical
    for (int i = row - 1; i >= 1; i--) {
        mask |= (1ULL << (i * 8 + col));
    }
    for (int i = row + 1; i <= 6; i++) {
        mask |= (1ULL << (i * 8 + col));
    }

    return mask;
}

// given a rook mask, generate all possible subsets of occupancy along those ranks/files
std::vector<uint64_t> get_all_occupancies(uint64_t mask) {
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

// generate bitboard of attacked squares given position of rook and board occupancy
uint64_t get_rook_attacks(int square, uint64_t occupancy) {
    uint64_t attacks = 0ULL;
    int row = square / 8;
    int col = square % 8;

    // left
    for (int i = col - 1; i >= 0; i--) {
        attacks |= (1ULL << (row * 8 + i));
        if (occupancy & (1ULL << (row * 8 + i))) {
            break;
        }
    }
    // right
    for (int i = col + 1; i < 8; i++) {
        attacks |= (1ULL << (row * 8 + i));
        if (occupancy & (1ULL << (row * 8 + i))) {
            break;
        }
    }
    // up
    for (int i = row - 1; i >= 0; i--) {
        attacks |= (1ULL << (i * 8 + col));
        if (occupancy & (1ULL << (i * 8 + col))) {
            break;
        }
    }
    // down
    for (int i = row + 1; i < 8; i++) {
        attacks |= (1ULL << (i * 8 + col));
        if (occupancy & (1ULL << (i * 8 + col))) {
            break;
        }
    }

    return attacks;
}


uint64_t random_U64() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, ~0ULL);
    return dist(gen);
}

// generate perfect hash function for occupancies to rook attack index through brute force
uint64_t find_rook_magic_number(int square) {
    int relevant_bits = 12; // max relevant bits that can block rook attacks
    std::vector<uint64_t> occupancies = get_all_occupancies(get_rook_mask(square));
    std::unordered_map<uint64_t, uint64_t> attack_table;

    while (true) {
        uint64_t magic = random_U64() & random_U64() & random_U64(); // sparse

        if (__builtin_popcountll((magic * 0x7EDD5E59A4E28E5B) & 0xFF00000000000000) < 6) {
            continue; // heuristic
        }
            
        bool found = true;
        attack_table.clear();

        for (uint64_t o : occupancies) {
            uint64_t index = (o * magic) >> (64 - relevant_bits);
            uint64_t attacks = get_rook_attacks(square, o);

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
int main() {
    // find magic numbers to hard code into main code

    for (int square = 0; square < 64; square++) {
        printf("0x%llx, \n", find_rook_magic_number(square));
    }

    return 0;
}
*/

uint64_t rook_magic_numbers[64] = {
    0x418000122086c000, 0x80200084104000, 0x40041088a00040, 0x10080104000201,
    0x3001022040800111, 0x4140010010820040, 0x1040040e40090080, 0x480148000402100,
    0x3004800180144000, 0x8028901924024000, 0x4818400428040811, 0x921000820800210,
    0x520801008000112, 0x1000220104008010, 0x1022101002506002, 0x9300108100020,
    0x4004808000904020, 0x102881400aa0611, 0x42002408020, 0x80041c004408800,
    0x21200808000340, 0x801481000840, 0xa40200400403108, 0x20000228403,
    0x4050a210020800, 0x2444020c0300, 0x140060082000, 0x10200c42001200,
    0x5020812100842804, 0x48010100082c0, 0x11402081040, 0x200084041000280,
    0x2000800410200024, 0x4840240c3400, 0x818401000400804, 0x30080804110480,
    0x13408001204020, 0x4a08040100100, 0xd0008a002000c0, 0xc028010040800822,
    0x30104000808006, 0x8008420400408182, 0x82201400840040, 0x2100620040248100,
    0x120404a800240488, 0x84883095000c2806, 0x4004098020001, 0x21240084c010,
    0x61158004400008, 0x34002020900008c0, 0xa06004400402, 0x61001000a20110,
    0x5008100c0100, 0x20202008c200080, 0xc21000a00011044, 0x800100002000d0,
    0x11004080001021, 0x404110888029a2, 0x102804022069026, 0x8000200214100009,
    0x8864200052012, 0x1003008028044209, 0x815013a903030042, 0x5045000480204211,
};


// same for bishops:
uint64_t get_bishop_mask(int square) {
    uint64_t mask = 0ULL;
    int row = square / 8;
    int col = square % 8;

    // top right
    for (int r = row + 1, c = col + 1; r < 7 && c < 7; r++, c++) {
        mask |= (1ULL << (r * 8 + c));
    }
    // top left
    for (int r = row + 1, c = col - 1; r < 7 && c > 0; r++, c--) {
        mask |= (1ULL << (r * 8 + c));
    }
    // bottom right
    for (int r = row - 1, c = col + 1; r > 0 && c < 7; r--, c++) {
        mask |= (1ULL << (r * 8 + c));
    }
    // bottom left
    for (int r = row - 1, c = col - 1; r > 0 && c > 0; r--, c--) {
        mask |= (1ULL << (r * 8 + c));
    }

    return mask;
}


uint64_t get_bishop_attacks(int square, uint64_t occupancy) {
    uint64_t attacks = 0ULL;
    int row = square / 8;
    int col = square % 8;

    // top right
    for (int r = row + 1, c = col + 1; r < 8 && c < 8; r++, c++) {
        attacks |= (1ULL << (r * 8 + c));
        if (occupancy & (1ULL << (r * 8 + c))) {
            break;
        }
    }
    // top left
    for (int r = row + 1, c = col - 1; r < 8 && c >= 0; r++, c--) {
        attacks |= (1ULL << (r * 8 + c));
        if (occupancy & (1ULL << (r * 8 + c))) {
            break;
        }
    }
    // bottom right
    for (int r = row - 1, c = col + 1; r >= 0 && c < 8; r--, c++) {
        attacks |= (1ULL << (r * 8 + c));
        if (occupancy & (1ULL << (r * 8 + c))) {
            break;
        }
    }
    // bottom left
    for (int r = row - 1, c = col - 1; r >= 0 && c >= 0; r--, c--) {
        attacks |= (1ULL << (r * 8 + c));
        if (occupancy & (1ULL << (r * 8 + c))) {
            break;
        }
    }

    return attacks;
}


uint64_t find_bishop_magic_number(int square) {
    int relevant_bits = 9;
    std::vector<uint64_t> occupancies = get_all_occupancies(get_bishop_mask(square));
    std::unordered_map<uint64_t, uint64_t> attack_table;

    while (true) {
        uint64_t magic = random_U64() & random_U64() & random_U64(); // sparse

        if (__builtin_popcountll((magic * 0x7EDD5E59A4E28E5B) & 0xFF00000000000000) < 6) {
            continue; // heuristic
        }
            
        bool found = true;
        attack_table.clear();

        for (uint64_t o : occupancies) {
            uint64_t index = (o * magic) >> (64 - relevant_bits);
            uint64_t attacks = get_bishop_attacks(square, o);

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
int main() {
    // find magic numbers to hard code into main code

    for (int square = 0; square < 64; square++) {
        printf("0x%llx, \n", find_bishop_magic_number(square));
    }

    return 0;
}
*/

uint64_t bishop_magic_numbers[64] = {
    0x10404e0140a812c, 0x300842401042080, 0x800a048080048112, 0x41004a0121002000,
    0x901408010000, 0x4000802103004060, 0x24404a01400040, 0xa201a00b810500,
    0x2202408061020206, 0x90220404200c5c1, 0x504c212100804, 0x80000a8080100004,
    0x4203140000005, 0xc000844c20020000, 0x108001410002000, 0x820814010800,
    0x4042200040a0, 0x20004684110c0c24, 0x800c14808210002, 0x1000020482000,
    0x488100424027480, 0x8010220704004200, 0xc2040020844c1000, 0x882004042086,
    0x28040101408083, 0x1c08008006d00448, 0x3000094291001421, 0x540040050410020,
    0x8001010084504000, 0x8256028000082000, 0x2216201120448400, 0x202082409200400,
    0x81801805003b800, 0x484800200020, 0x404200d000021100, 0x1108120084180080,
    0x64008400080410, 0x4010e10214040, 0x4021004002090, 0x2088202408000600,
    0x102001420008120, 0x411004812022000, 0x4202008400c0, 0x17004480220804,
    0x20200600898040, 0x4080404401000120, 0x88225082120022, 0x1180008a1e80a58,
    0x4182046a82014011, 0x808148050c201100, 0x4020481040111014, 0x4000002102421000,
    0x214000006d00d017, 0x1a80a02028900800, 0x40120008a800402, 0x10a0b0010810800,
    0x800400206500, 0x30411041200a6110, 0x2002000020100804, 0x802100020420028,
    0x880408060540, 0x400804002004104, 0x102311a208008490, 0x244208400184010
};