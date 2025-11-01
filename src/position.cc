#include "position.hh"

int Position::values[6] = { 100, 300, 300, 500, 900, 2000 };

struct directions Position::directions = {
    { {1,0}, {-1,0}, {0,1}, {0,-1}, {1,1}, {-1,-1}, {1,-1}, {-1,1} },
    { {1,0}, {-1,0}, {0,1}, {0,-1} },
    { {1,1}, {-1,-1}, {1,-1}, {-1,1} },
    { {2,1}, {2,-1}, {-2,1}, {-2,-1}, {1,2}, {1,-2}, {-1,2}, {-1,-2} },
};

Zobrist Position::zobrist;

uint64_t Position::knight_moves[64];
uint64_t Position::pawn_attacks[64][2];
uint64_t Position::king_moves[64];

uint64_t Position::rook_masks[64];
uint64_t Position::rook_moves[64][4096];

uint64_t Position::bishop_masks[64];
uint64_t Position::bishop_moves[64][512];

extern std::vector<uint64_t> get_all_occupancies(uint64_t mask);

extern uint64_t get_rook_mask(int square);
extern uint64_t get_rook_attacks(int square, uint64_t occupancy);
extern uint64_t rook_magic_numbers[64];

extern uint64_t get_bishop_mask(int square);
extern uint64_t get_bishop_attacks(int square, uint64_t occupancy);
extern uint64_t bishop_magic_numbers[64];


void Position::init_piece_moves() {
    for (int square = 0; square < 64; square++) {
        int row = square / 8;
        int col = square % 8;

        // knight moves
        uint64_t moves = 0ULL;
        for (auto d : directions.knight) {
            int new_row = row + d.first;
            int new_col = col + d.second;
            if (new_row >= 0 && new_row < 8 && new_col >= 0 && new_col < 8) {
                moves |= (1ULL << (new_row * 8 + new_col));
            }
        }
        knight_moves[square] = moves;
        
        // pawn attacks
        uint64_t white_attacks = 0ULL;
        uint64_t black_attacks = 0ULL;
        if (col > 0) {
            if (row > 0) {
                white_attacks |= (1ULL << ((row - 1) * 8 + (col - 1)));
            }
            if (row < 7) {
                black_attacks |= (1ULL << ((row + 1) * 8 + (col - 1)));
            }
        }
        if (col < 7) {
            if (row > 0) {
                white_attacks |= (1ULL << ((row - 1) * 8 + (col + 1)));
            }
            if (row < 7) {
                black_attacks |= (1ULL << ((row + 1) * 8 + (col + 1)));
            }
        }
        pawn_attacks[square][WHITE] = white_attacks;
        pawn_attacks[square][BLACK] = black_attacks;

        // king moves
        moves = 0ULL;
        for (auto d : directions.queen) {
            int new_row = row + d.first;
            int new_col = col + d.second;
            if (new_row >= 0 && new_row < 8 && new_col >= 0 && new_col < 8) {
                moves |= (1ULL << (new_row * 8 + new_col));
            }
        }
        king_moves[square] = moves;

        // rook moves
        rook_masks[square] = get_rook_mask(square);
        std::vector<uint64_t> occupancies = get_all_occupancies(rook_masks[square]);
        for (uint64_t occ : occupancies) {
            uint64_t index = (occ * rook_magic_numbers[square]) >> (64 - 12);
            rook_moves[square][index] = get_rook_attacks(square, occ);
        }

        // bishop moves
        bishop_masks[square] = get_bishop_mask(square);
        occupancies = get_all_occupancies(bishop_masks[square]);
        for (uint64_t occ : occupancies) {
            uint64_t index = (occ * bishop_magic_numbers[square]) >> (64 - 9);
            bishop_moves[square][index] = get_bishop_attacks(square, occ);
        }
    }
}

// initialize starting position
Position::Position(std::string nnue)
: turn(WHITE), en_passant_col(-1), fifty_move_count(0), half_moves(0), repetitions(1) {
    int start_material = values[QUEEN] + values[ROOK] * 2 + values[BISHOP] * 2 +
        values[KNIGHT] * 2 + values[PAWN] * 8;
    
    white_material = start_material;
    black_material = start_material;

    int init_board[8][8] = {
        {BLACK_ROOK, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN, BLACK_KING, BLACK_BISHOP, BLACK_KNIGHT, BLACK_ROOK},
        {BLACK_PAWN, BLACK_PAWN, BLACK_PAWN, BLACK_PAWN, BLACK_PAWN, BLACK_PAWN, BLACK_PAWN, BLACK_PAWN},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {WHITE_PAWN, WHITE_PAWN, WHITE_PAWN, WHITE_PAWN, WHITE_PAWN, WHITE_PAWN, WHITE_PAWN, WHITE_PAWN},
        {WHITE_ROOK, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN, WHITE_KING, WHITE_BISHOP, WHITE_KNIGHT, WHITE_ROOK}
    };

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            board[i][j] = init_board[i][j];
        }
    }

    king_positions[0] = 7;
    king_positions[1] = 4;
    king_positions[2] = 0;
    king_positions[3] = 4;

    for (int i = 0; i < 4; i++) {
        can_castle[i] = true;
    }

    hash_value = zobrist.hash_position(board, turn, can_castle, en_passant_col);
    position_history[0] = hash_value;

    init_piece_moves();

    piece_maps[KNIGHT][WHITE] = 0x4200000000000000;
    piece_maps[KNIGHT][BLACK] = 0x42;

    piece_maps[PAWN][WHITE] = 0x00FF000000000000;
    piece_maps[PAWN][BLACK] = 0xFF00;

    piece_maps[BISHOP][WHITE] = 0x2400000000000000;
    piece_maps[BISHOP][BLACK] = 0x24;

    piece_maps[ROOK][WHITE] = 0x8100000000000000;
    piece_maps[ROOK][BLACK] = 0x81;

    piece_maps[QUEEN][WHITE] = 0x800000000000000;
    piece_maps[QUEEN][BLACK] = 0x8;

    piece_maps[KING][WHITE] = 1ULL << (king_positions[0] * 8 + king_positions[1]);
    piece_maps[KING][BLACK] = 1ULL << (king_positions[2] * 8 + king_positions[3]);

    white_pieces = piece_maps[KING][WHITE] | piece_maps[QUEEN][WHITE] | piece_maps[ROOK][WHITE] |
        piece_maps[BISHOP][WHITE] | piece_maps[KNIGHT][WHITE] | piece_maps[PAWN][WHITE];
    black_pieces = piece_maps[KING][BLACK] | piece_maps[QUEEN][BLACK] | piece_maps[ROOK][BLACK] |
        piece_maps[BISHOP][BLACK] | piece_maps[KNIGHT][BLACK] | piece_maps[PAWN][BLACK];
    
    NNUE::init(nnue);
    eval[0] = NNUE::evaluate(board, turn, NNUE::get_output_bucket(white_pieces | black_pieces));
}

// set up custom position
Position::Position(int init_board[8][8], int turn, std::string nnue)
: turn(turn), en_passant_col(-1), fifty_move_count(0), half_moves(0), repetitions(1) {
    white_material = 0;
    black_material = 0;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            board[i][j] = init_board[i][j];
        }
    }

    for (int i = 0; i < 4; i++) {
        can_castle[i] = true;
    }

    for (int i = PAWN; i <= KING; i++) {
        piece_maps[i][WHITE] = 0ULL;
        piece_maps[i][BLACK] = 0ULL;
    }

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            int piece = board[row][col];
            if (!piece) {
                continue;
            }
            int piece_type = get_piece_type(piece);
            int color = get_color(piece);
            
            if (piece == WHITE_KING) {
                king_positions[0] = row;
                king_positions[1] = col;
                if (row != 7 || col != 4) {
                    can_castle[0] = false;
                    can_castle[1] = false;
                }
            } else if (piece == BLACK_KING) {
                king_positions[2] = row;
                king_positions[3] = col;
                if (row != 0 || col != 4) {
                    can_castle[2] = false;
                    can_castle[3] = false;
                }
            } else if (color == WHITE) {
                white_material += values[piece_type];
            } else {
                black_material += values[piece_type];
            }

            piece_maps[piece_type][color] |= 1ULL << (row * 8 + col);
        }
    }

    if (num_set_bits(piece_maps[KING][WHITE]) != 1 ||
        num_set_bits(piece_maps[KING][BLACK]) != 1) {
        perror("invalid position");
    }

    if (board[0][0] != BLACK_ROOK) {
        can_castle[2] = false;
    }
    if (board[0][7] != BLACK_ROOK) {
        can_castle[3] = false;
    }
    if (board[7][0] != WHITE_ROOK) {
        can_castle[0] = false;
    }
    if (board[7][7] != WHITE_ROOK) {
        can_castle[1] = false;
    }

    hash_value = zobrist.hash_position(board, turn, can_castle, en_passant_col);
    position_history[0] = hash_value;

    init_piece_moves();

    white_pieces = piece_maps[KING][WHITE] | piece_maps[QUEEN][WHITE] | piece_maps[ROOK][WHITE] |
        piece_maps[BISHOP][WHITE] | piece_maps[KNIGHT][WHITE] | piece_maps[PAWN][WHITE];
    black_pieces = piece_maps[KING][BLACK] | piece_maps[QUEEN][BLACK] | piece_maps[ROOK][BLACK] |
        piece_maps[BISHOP][BLACK] | piece_maps[KNIGHT][BLACK] | piece_maps[PAWN][BLACK];
    
    NNUE::init(nnue);
    eval[0] = NNUE::evaluate(board, turn, NNUE::get_output_bucket(white_pieces | black_pieces));
}


Position::Position(std::string fen, std::string nnue)
: fifty_move_count(0), repetitions(1) {
    std::istringstream fen_stream(fen);
    std::string fen_board, fen_turn, fen_castling, fen_ep;
    int full_moves;

    fen_stream >> fen_board >> fen_turn >> fen_castling >> fen_ep >> half_moves >> full_moves;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            board[i][j] = 0;
        }
    }
    for (int i = 0; i < 4; i++) {
        can_castle[i] = false;
    }
    for (int i = PAWN; i <= KING; i++) {
        piece_maps[i][WHITE] = 0ULL;
        piece_maps[i][BLACK] = 0ULL;
    }

    int index = 0;
    for (char c : fen_board) {
        if (c == '/') {
            continue;
        } else if (isdigit(c)) {
            index += c - '0';
        } else {
            int piece;
            if (c == 'P') {
                piece = WHITE_PAWN;
            } else if (c == 'N') {
                piece = WHITE_KNIGHT;
            } else if (c == 'B') {
                piece = WHITE_BISHOP;
            } else if (c == 'R') {
                piece = WHITE_ROOK;
            } else if (c == 'Q') {
                piece = WHITE_QUEEN;
            } else if (c == 'K') {
                piece = WHITE_KING;
            } else if (c == 'p') {
                piece = BLACK_PAWN;
            } else if (c == 'n') {
                piece = BLACK_KNIGHT;
            } else if (c == 'b') {
                piece = BLACK_BISHOP;
            } else if (c == 'r') {
                piece = BLACK_ROOK;
            } else if (c == 'q') {
                piece = BLACK_QUEEN;
            } else if (c == 'k') {
                piece = BLACK_KING;
            }
            board[index / 8][index % 8] = piece;
            index++;
        }
    }

    turn = (fen_turn == "w") ? WHITE : BLACK;

    for (char c : fen_castling) {
        if (c == 'K') can_castle[1] = true;
        else if (c == 'Q') can_castle[0] = true;
        else if (c == 'k') can_castle[3] = true;
        else if (c == 'q') can_castle[2] = true;
    }

    if (fen_ep != "-" && fen_ep != "–") {
        en_passant_col = fen_ep[0] - 'a';
    } else {
        en_passant_col = -1;
    }

    white_material = 0;
    black_material = 0;

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            int piece = board[row][col];
            if (!piece) {
                continue;
            }
            int piece_type = get_piece_type(piece);
            int color = get_color(piece);
            
            if (piece == WHITE_KING) {
                king_positions[0] = row;
                king_positions[1] = col;
            } else if (piece == BLACK_KING) {
                king_positions[2] = row;
                king_positions[3] = col;
            } else if (color == WHITE) {
                white_material += values[piece_type];
            } else {
                black_material += values[piece_type];
            }

            piece_maps[piece_type][color] |= 1ULL << (row * 8 + col);
        }
    }

    hash_value = zobrist.hash_position(board, turn, can_castle, en_passant_col);
    position_history[0] = hash_value;

    init_piece_moves();

    white_pieces = piece_maps[KING][WHITE] | piece_maps[QUEEN][WHITE] | piece_maps[ROOK][WHITE] |
        piece_maps[BISHOP][WHITE] | piece_maps[KNIGHT][WHITE] | piece_maps[PAWN][WHITE];
    black_pieces = piece_maps[KING][BLACK] | piece_maps[QUEEN][BLACK] | piece_maps[ROOK][BLACK] |
        piece_maps[BISHOP][BLACK] | piece_maps[KNIGHT][BLACK] | piece_maps[PAWN][BLACK];
    
    NNUE::init(nnue);
    eval[0] = NNUE::evaluate(board, turn, NNUE::get_output_bucket(white_pieces | black_pieces));
}


// for debugging
void Position::print_bitboard(uint64_t bitboard) {
    for (int i = 0; i < 64; i++) {
        if (i % 8 == 0) {
            printf("\n");
        }
        printf("%c", (bitboard & (1ULL << i)) ? '1' : '0');
    }
    printf("\n\n");
}

int Position::get_color(int piece) {
    return piece > 6;
}

int Position::get_piece_type(int piece) {
    return piece <= 6 ? piece - 1 : piece - 7;
}


int Position::char_to_piece(char c) {
    if (c == 'Q') {
        return QUEEN;
    } else if (c == 'R') {
        return ROOK;
    } else if (c == 'N') {
        return KNIGHT;
    } else if (c == 'B') {
       return BISHOP; 
    } else if (c == 'P') {
        return PAWN;
    } else if (c == 'K') {
        return KING;
    }
    return -1;
}


uint64_t Position::get_rook_moves(int square, uint64_t blockers) {
    blockers &= rook_masks[square];

    uint64_t index = (blockers * rook_magic_numbers[square]) >> (64 - 12);
    return rook_moves[square][index];
}

uint64_t Position::get_bishop_moves(int square, uint64_t blockers) {
    blockers &= bishop_masks[square];

    uint64_t index = (blockers * bishop_magic_numbers[square]) >> (64 - 9);
    return bishop_moves[square][index];
}


bool Position::in_check(int king_row, int king_col) {
    int king_square = king_row * 8 + king_col;
    uint64_t king_mask = 1ULL << king_square;

    uint64_t all_pieces = king_mask | piece_maps[KING][!turn] | piece_maps[QUEEN][WHITE] | piece_maps[QUEEN][BLACK] |
                          piece_maps[ROOK][WHITE] | piece_maps[ROOK][BLACK] | piece_maps[BISHOP][WHITE] |
                          piece_maps[BISHOP][BLACK] | piece_maps[KNIGHT][WHITE] | piece_maps[KNIGHT][BLACK] |
                          piece_maps[PAWN][WHITE] | piece_maps[PAWN][BLACK];

    if ((piece_maps[ROOK][!turn] | piece_maps[QUEEN][!turn]) & get_rook_moves(king_square, all_pieces)) {
        return true;
    }

    if ((piece_maps[BISHOP][!turn] | piece_maps[QUEEN][!turn]) & get_bishop_moves(king_square, all_pieces)) {
        return true;
    }
    
    if (knight_moves[king_square] & piece_maps[KNIGHT][!turn]) {
        return true;
    }

    uint64_t pawn_mask = pawn_attacks[king_square][turn];
    if (pawn_mask & piece_maps[PAWN][!turn]) {
        return true;
    }
    
    return (piece_maps[KING][!turn] & king_moves[king_square]) != 0;
}

bool Position::in_check() {
    if (turn == WHITE) {
        return in_check(king_positions[0], king_positions[1]);
    }
    return in_check(king_positions[2], king_positions[3]);
}


// get piece moves helper function
uint64_t Position::get_castle_moves() {
    uint64_t moves = 0ULL;
    if (!in_check()) {
        if (turn == WHITE) {
            // kingside
            if (can_castle[1] && !board[7][5] && !board[7][6]) {
                moves |= (1ULL << (7 * 8 + 6));
            }
            // queenside
            if (can_castle[0] && !board[7][3] && !board[7][2] && !board[7][1]) {
                moves |= (1ULL << (7 * 8 + 2));
            }
        } else {
            if (can_castle[3] && !board[0][5] && !board[0][6]) {
                moves |= (1ULL << 6);
            }
            if (can_castle[2] && !board[0][3] && !board[0][2] && !board[0][1]) {
                moves |= (1ULL << 2);
            }
        }
    }
    return moves;
}

// get piece moves helper function
uint64_t Position::get_pawn_pushes(int row, int col) {
    uint64_t pushes = 0ULL;
    int direction = (turn == WHITE) ? -1 : 1;
    // one square forward
    if (!board[row + direction][col]) {
        pushes |= (1ULL << ((row + direction) * 8 + col));
    }
    // two squares forward
    if (((direction == -1 && row == 6) || (direction == 1 && row == 1)) &&
        !board[row + direction][col] && !board[row + 2 * direction][col]) {
        pushes |= (1ULL << ((row + 2 * direction) * 8 + col));
    }
    return pushes;
}

// get piece moves helper function
uint64_t Position::get_en_passant(int row, int col) {
    if (en_passant_col < 0) {
        return 0ULL;
    }

    int direction = (turn == WHITE) ? -1 : 1;

    for (int dc = -1; dc <= 1; dc += 2) {
        int newCol = col + dc;
        if (newCol >= 0 && newCol < 8 && row + direction >= 0 && row + direction < 8) {
            if (en_passant_col == newCol && ((row == 3 && direction == -1) ||
                (row == 4 && direction == 1))) {
                
                return (1ULL << ((row + direction) * 8 + newCol));
            }
        }
    }
    return 0ULL;
}

// get pseudo legal moves helper function - returns bit map instead of move objects
uint64_t Position::get_piece_moves(int row, int col) {
    int square = row * 8 + col;
    int piece = board[row][col];

    if (!piece || turn != get_color(piece)) {
        return 0ULL;
    }

    uint64_t friendly_pieces = turn == WHITE ? white_pieces : black_pieces;
    uint64_t all_pieces = white_pieces | black_pieces;

    int piece_type = get_piece_type(piece);
    if (piece_type == PAWN) {
        uint64_t captures = pawn_attacks[square][turn] & (all_pieces & (~friendly_pieces));
        return captures | get_pawn_pushes(row, col) | get_en_passant(row, col);
    }
    if (piece_type == QUEEN) {
        return (get_rook_moves(square, all_pieces) | get_bishop_moves(square, all_pieces)) & (~friendly_pieces);
    }
    if (piece_type == ROOK) {
        return get_rook_moves(square, all_pieces) & (~friendly_pieces);
    }
    if (piece_type == BISHOP) {
        return get_bishop_moves(square, all_pieces) & (~friendly_pieces);
    }
    if (piece_type == KNIGHT) {
        return knight_moves[square] & (~friendly_pieces);
    }
    return (king_moves[square] | get_castle_moves()) & (~friendly_pieces);
}


bool Position::is_promotion(int start_row, int start_col, int end_row, int end_col) {
    return (end_row == 0 && board[start_row][start_col] == WHITE_PAWN) ||
           (end_row == 7 && board[start_row][start_col] == BLACK_PAWN);
}


// moves that would be/are legal not considering if they leave the king in check
std::vector<Move> Position::get_pseudo_legal_moves(MoveType move_type) {
    std::vector<Move> moves;
    moves.reserve(128);

    uint64_t pieces = turn == WHITE ? white_pieces : black_pieces;
    while (pieces) {
        int from_square = least_set_bit(pieces);
        pieces &= pieces - 1;

        int row = from_square / 8;
        int col = from_square % 8;

        int piece = board[row][col];

        uint64_t move_map = get_piece_moves(row, col);
        if (move_type == TACTIC || move_type == QUIET) {
            uint64_t tactics = turn == WHITE ? black_pieces : white_pieces;
            if (piece == WHITE_PAWN) {
                // 8th rank
                tactics |= 0xff | get_en_passant(row, col);
            } else if (piece == BLACK_PAWN) {
                // 1st rank
                tactics |= 0xff00000000000000 | get_en_passant(row, col);
            }
            
            if (move_type == TACTIC) {
                move_map &= tactics;
            } else {
                move_map &= ~tactics;
            }
        }
        
        while (move_map) {
            // finds least significant set bit position
            int to_square = least_set_bit(move_map);
            // clear that bit
            move_map &= move_map - 1;

            int to_row = to_square / 8;
            int to_col = to_square % 8;

            if (is_promotion(row, col, to_row, to_col)) {
                for (auto p : {QUEEN, ROOK, BISHOP, KNIGHT}) {
                    moves.emplace_back(from_square, to_square, p, values[p] - values[PAWN] + 1);
                }
            } else {
                int capture = board[to_row][to_col];
                // mvv-lva
                int exchange = capture ? values[get_piece_type(capture)] - values[get_piece_type(piece)] + 1 : 0;

                moves.emplace_back(from_square, to_square, 0, exchange);
            }
        }
    }
    return moves;
}

// move passed in should be pseudo legal
bool Position::is_legal(const Move& move) {
    int king_row;
    int king_col;
    if (turn == WHITE) {
        king_row = king_positions[0];
        king_col = king_positions[1];
    } else {
        king_row = king_positions[2];
        king_col = king_positions[3];
    }

    int start_row = move.start_row();
    int start_col = move.start_col();
    int end_row = move.end_row();
    int end_col = move.end_col();

    // check castle
    if (king_row == start_row && king_col == start_col && std::abs(start_col - end_col) > 1) {
        std::vector<int> k_cols = (start_col > end_col) ? std::vector<int>{start_col - 1, start_col - 2} // king moves left
                                                        : std::vector<int>{start_col + 1, start_col + 2};
        // king cannot castle through check
        for (auto k : k_cols) {
            if (in_check(start_row, k)) {
                return false;
            }
        }
        return true;
    }

    int piece = board[start_row][start_col];
    int piece_type = get_piece_type(piece);

    // check en passant
    bool en_passant = false;
    if (get_piece_type(piece) == PAWN && start_col != end_col && !board[end_row][end_col]) {
        board[start_row][end_col] = 0;
        en_passant = true;

        int capture_square = start_row * 8 + end_col;
        if (piece == WHITE_PAWN) {
            piece_maps[PAWN][BLACK] &= ~(1ULL << capture_square);
        } else if (piece == BLACK_PAWN) {
            piece_maps[PAWN][WHITE] &= ~(1ULL << capture_square);
        }
    }

    int capture = board[end_row][end_col];

    board[end_row][end_col] = piece;
    board[start_row][start_col] = 0;
    
    int k_row;
    int k_col;
    if (king_row == start_row && king_col == start_col) {
        k_row = end_row;
        k_col = end_col;
    } else {
        k_row = king_row;
        k_col = king_col;
    }

    int start_square = move.from();
    int end_square = move.to();
    // piece moved
    if (piece_type != KING) {
        piece_maps[piece_type][turn] &= ~(1ULL << start_square);
        piece_maps[piece_type][turn] |= (1ULL << end_square);
    }

    // piece captured
    if (capture) {
        piece_maps[get_piece_type(capture)][!turn] &= ~(1ULL << end_square);
    }

    bool legal = !in_check(k_row, k_col);
    
    // undo board changes
    if (en_passant) {
        board[start_row][end_col] = turn == WHITE ? BLACK_PAWN : WHITE_PAWN;

        int capture_square = start_row * 8 + end_col;
        if (piece == WHITE_PAWN) {
            piece_maps[PAWN][BLACK] |= (1ULL << capture_square);
        } else {
            piece_maps[PAWN][WHITE] |= (1ULL << capture_square);
        }
    }
    board[end_row][end_col] = capture;
    board[start_row][start_col] = piece;

    if (piece_type != KING) {
        piece_maps[piece_type][turn] |= (1ULL << start_square);
        piece_maps[piece_type][turn] &= ~(1ULL << end_square);
    }

    if (capture) {
        piece_maps[get_piece_type(capture)][!turn] |= (1ULL << end_square);
    }

    return legal;
}


std::vector<Move> Position::get_legal_moves() {
    std::vector<Move> legal_moves;
    legal_moves.reserve(64);

    for (auto move : get_pseudo_legal_moves()) {
        if (is_legal(move)) {
            legal_moves.push_back(move);
        }
    }
    
    return legal_moves;
}


void Position::make_move(const Move& move) {
    bool null_descendant = (hash_value == 0ULL);

    // copy accumulator
    std::memcpy(acc_white_stack[half_moves], NNUE::accumulators[WHITE], HIDDEN_SIZE * sizeof(int16_t));
    std::memcpy(acc_black_stack[half_moves], NNUE::accumulators[BLACK], HIDDEN_SIZE * sizeof(int16_t));

    int start_square = move.from();
    int end_square = move.to();

    int start_row = start_square / 8;
    int start_col = start_square % 8;
    int end_row = end_square / 8;
    int end_col = end_square % 8;

    int piece = board[start_row][start_col];
    int piece_type = get_piece_type(piece);
    int capture = board[end_row][end_col];

    bool prev_castle[4];
    for (int i = 0; i < 4; i++) {
        prev_castle[i] = can_castle[i];
    }
    struct move_info m = {
        hash_value,
        move,
        en_passant_col,
        fifty_move_count,
        capture,
        last_threefold_reset,
        repetitions,
        {prev_castle[0], prev_castle[1], prev_castle[2], prev_castle[3]}
    };
    move_stack[half_moves] = m;

    board[start_row][start_col] = 0;
    hash_value ^= zobrist.piece_table[piece - 1][start_square];

    // nnue
    int num_dirty = 1;
    dirty_piece dps[3];
    dps[0].piece = piece;
    dps[0].from = start_square;
    dps[0].to = end_square;


    // update piece maps
    if (piece_type != KING) {
        piece_maps[piece_type][turn] &= ~(1ULL << start_square);
        // unless promotion:
        piece_maps[piece_type][turn] |= (1ULL << end_square);
    }

    if (capture) {
        hash_value ^= zobrist.piece_table[capture - 1][end_square];
        int capture_type = get_piece_type(capture);
        piece_maps[capture_type][!turn] &= ~(1ULL << end_square);

        if (turn == WHITE) {
            black_material -= values[capture_type];
        } else {
            white_material -= values[capture_type];
        }

        // update castle
        if (capture == WHITE_ROOK) {
            if (end_row == 7 && end_col == 0 && can_castle[0]) {
                can_castle[0] = false;
                hash_value ^= zobrist.castle_table[0];
            } else if (end_row == 7 && end_col == 7 && can_castle[1]) {
                can_castle[1] = false;
                hash_value ^= zobrist.castle_table[1];
            }
        } else if (capture == BLACK_ROOK) {
            if (end_row == 0 && end_col == 0 && can_castle[2]) {
                can_castle[2] = false;
                hash_value ^= zobrist.castle_table[2];
            } else if (end_row == 0 && end_col == 7 && can_castle[3]) {
                can_castle[3] = false;
                hash_value ^= zobrist.castle_table[3];
            }
        }

        num_dirty++;
        dps[1].piece = capture;
        dps[1].from = end_square;
        dps[1].to = -1;
    }

    int promotion = move.promote_to();
    if (promotion) {
        int new_piece = promotion + 1 + (6 * turn);
        board[end_row][end_col] = new_piece;

        hash_value ^= zobrist.piece_table[new_piece - 1][end_square];
        piece_maps[promotion][turn] |= (1ULL << end_square);

        // clear pawn on last rank
        piece_maps[piece_type][turn] &= ~(1ULL << end_square);

        if (turn == WHITE) {
            white_material += values[promotion] - values[PAWN];
        } else {
            black_material += values[promotion] - values[PAWN];
        }

        num_dirty++;
        dps[0].to = -1; // clear pawn
        dps[num_dirty - 1].piece = new_piece;
        dps[num_dirty - 1].from = -1;
        dps[num_dirty - 1].to = end_square;
    } else if (piece_type == KING && std::abs(start_col - end_col) > 1) {
        // check castle
        board[end_row][end_col] = piece;
        hash_value ^= zobrist.piece_table[piece - 1][end_square];

        num_dirty++;

        int direction = start_col - end_col;
        if (direction > 0) {
            int rook = board[end_row][0];
            board[end_row][end_col + 1] = rook;
            board[end_row][0] = 0;

            int rook_start = start_row * 8;
            int rook_end = end_row * 8 + end_col + 1;
            hash_value ^= zobrist.piece_table[rook - 1][rook_start];
            hash_value ^= zobrist.piece_table[rook - 1][rook_end];

            piece_maps[ROOK][turn] &= ~(1ULL << rook_start);
            piece_maps[ROOK][turn] |= (1ULL << rook_end);

            dps[num_dirty - 1].piece = rook;
            dps[num_dirty - 1].from = rook_start;
            dps[num_dirty - 1].to = rook_end;
        } else {
            int rook = board[end_row][7];
            board[end_row][end_col - 1] = rook;
            board[end_row][7] = 0;

            int rook_start = start_row * 8 + 7;
            int rook_end = end_row * 8 + end_col - 1;
            hash_value ^= zobrist.piece_table[rook - 1][rook_start];
            hash_value ^= zobrist.piece_table[rook - 1][rook_end];

            piece_maps[ROOK][turn] &= ~(1ULL << rook_start);
            piece_maps[ROOK][turn] |= (1ULL << rook_end);

            dps[num_dirty - 1].piece = rook;
            dps[num_dirty - 1].from = rook_start;
            dps[num_dirty - 1].to = rook_end;
        }
    } else if (piece_type == PAWN && start_col != end_col &&
               !board[end_row][end_col]) {
        // en passant
        board[start_row][end_col] = 0;
        board[end_row][end_col] = piece;

        int pawn_idx = piece == WHITE_PAWN ? BLACK_PAWN - 1 : WHITE_PAWN - 1;
        int square = start_row * 8 + end_col;
        hash_value ^= zobrist.piece_table[pawn_idx][square];

        hash_value ^= zobrist.piece_table[piece - 1][end_square];

        if (turn == WHITE) {
            // clear en passant capture
            piece_maps[PAWN][BLACK] &= ~(1ULL << square);
            black_material -= values[PAWN];
        } else {
            piece_maps[PAWN][WHITE] &= ~(1ULL << square);
            white_material -= values[PAWN];
        }

        num_dirty++;
        dps[num_dirty - 1].piece = piece == WHITE_PAWN ? BLACK_PAWN : WHITE_PAWN;
        dps[num_dirty - 1].from = square;
        dps[num_dirty - 1].to = -1;
    } else {
        board[end_row][end_col] = piece;
        hash_value ^= zobrist.piece_table[piece - 1][end_square];
    }

    // update castle rights
    if (piece == WHITE_KING) {
        if (can_castle[0]) {
            hash_value ^= zobrist.castle_table[0];
        }
        if (can_castle[1]) {
            hash_value ^= zobrist.castle_table[1];
        }

        can_castle[0] = false;
        can_castle[1] = false;
        king_positions[0] = end_row;
        king_positions[1] = end_col;
    } else if (piece == BLACK_KING) {
        if (can_castle[2]) {
            hash_value ^= zobrist.castle_table[2];
        }
        if (can_castle[3]) {
            hash_value ^= zobrist.castle_table[3];
        }

        can_castle[2] = false;
        can_castle[3] = false;
        king_positions[2] = end_row;
        king_positions[3] = end_col;
    } else if (piece == WHITE_ROOK) {
        if (start_row == 7 && start_col == 0 && can_castle[0]) {
            hash_value ^= zobrist.castle_table[0];
            can_castle[0] = false;
        } else if (start_row == 7 && start_col == 7 && can_castle[1]) {
            hash_value ^= zobrist.castle_table[1];
            can_castle[1] = false;
        }
    } else if (piece == BLACK_ROOK) {
        if (start_row == 0 && start_col == 0 && can_castle[2]) {
            hash_value ^= zobrist.castle_table[2];
            can_castle[2] = false;
        } else if (start_row == 0 && start_col == 7 && can_castle[3]) {
            hash_value ^= zobrist.castle_table[3];
            can_castle[3] = false;
        }
    }

    // update kings
    piece_maps[KING][WHITE] = 1ULL << (king_positions[0] * 8 + king_positions[1]);
    piece_maps[KING][BLACK] = 1ULL << (king_positions[2] * 8 + king_positions[3]);

    // update white/black maps
    white_pieces = piece_maps[KING][WHITE] | piece_maps[QUEEN][WHITE] | piece_maps[ROOK][WHITE] |
        piece_maps[BISHOP][WHITE] | piece_maps[KNIGHT][WHITE] | piece_maps[PAWN][WHITE];
    black_pieces = piece_maps[KING][BLACK] | piece_maps[QUEEN][BLACK] | piece_maps[ROOK][BLACK] |
        piece_maps[BISHOP][BLACK] | piece_maps[KNIGHT][BLACK] | piece_maps[PAWN][BLACK];

    // update en passant col
    if (en_passant_col != -1) {
        hash_value ^= zobrist.en_passant_table[en_passant_col];
    }
    bool en_passant_possible = false;
    if (piece_type == PAWN && std::abs(start_row - end_row) > 1) {
        if (turn == BLACK) {
            if ((start_col - 1 >= 0 && board[3][start_col - 1] == WHITE_PAWN) ||
                (start_col + 1 < 8 && board[3][start_col + 1] == WHITE_PAWN)) {
                en_passant_possible = true;
            }
        } else {
            if ((start_col - 1 >= 0 && board[4][start_col - 1] == BLACK_PAWN) ||
                (start_col + 1 < 8 && board[4][start_col + 1] == BLACK_PAWN)) {
                en_passant_possible = true;
            }
        }
    }
    if (en_passant_possible) {
        en_passant_col = start_col;
        hash_value ^= zobrist.en_passant_table[en_passant_col];
    } else {
        en_passant_col = -1;
    }

    half_moves++;
   
    if (capture || piece_type == PAWN) {
        last_threefold_reset = half_moves;
    }

    turn = !turn;
    hash_value ^= zobrist.turn_key;

    // threefold repetition
    if (!null_descendant) {
        position_history[half_moves] = hash_value;
        if (repetitions < 3) {
            repetitions = 1;
            for (int i = half_moves - 2; i >= last_threefold_reset; i -= 2) {
                if (position_history[i] == hash_value) {
                    repetitions++;
                }
                if (repetitions == 3) {
                    break;
                }
            }
        }
    } else {
        hash_value = 0ULL;
    }

    // efficiently update accumulator - only have to worry about the few indices that changed this move
    for (int i = 0; i < num_dirty; i++) {
        dirty_piece dp = dps[i];
        if (dp.from >= 0) {
            // subtract from accumulators
            int white_index = dp.from * 12 + dp.piece - 1;
            int black_index = NNUE::black_index(dp.from, dp.piece);
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                NNUE::accumulators[WHITE][j] -= NNUE::hidden_weights[white_index][j];
                NNUE::accumulators[BLACK][j] -= NNUE::hidden_weights[black_index][j];
            }
        }
        if (dp.to >= 0) {
            // add to accumulators
            int white_index = dp.to * 12 + dp.piece - 1;
            int black_index = NNUE::black_index(dp.to, dp.piece);
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                NNUE::accumulators[WHITE][j] += NNUE::hidden_weights[white_index][j];
                NNUE::accumulators[BLACK][j] += NNUE::hidden_weights[black_index][j];
            }
        }
    }

    eval[half_moves] = NNUE::evaluate_incremental(turn, NNUE::get_output_bucket(white_pieces | black_pieces));
}


void Position::unmake_move(struct move_info move_info) {
    Move move = move_info.move;
    int capture = move_info.captured_piece;

    turn = !turn;
    half_moves--;

    int start_square = move.from();
    int end_square = move.to();

    int start_row = start_square / 8;
    int start_col = start_square % 8;
    int end_row = end_square / 8;
    int end_col = end_square % 8;

    int piece = board[end_row][end_col];
    int piece_type = get_piece_type(piece);
    board[start_row][start_col] = piece;
    board[end_row][end_col] = capture;

    if (piece_type != KING) {
        piece_maps[piece_type][turn] |= (1ULL << start_square); // unless promotion
        piece_maps[piece_type][turn] &= ~(1ULL << end_square);
    }

    if (capture) {
        int capture_type = get_piece_type(capture);
        piece_maps[capture_type][!turn] |= (1ULL << end_square);

        if (turn == WHITE) {
            black_material += values[capture_type];
        } else {
            white_material += values[capture_type];
        }
    }

    // castle
    if (piece_type == KING && std::abs(start_col - end_col) > 1) {
        int direction = start_col - end_col;
        if (direction > 0) { // queenside
            board[end_row][0] = board[end_row][end_col + 1];
            board[end_row][end_col + 1] = 0;

            int rook = board[end_row][0];
            int rook_start = start_row * 8;
            int rook_end = end_row * 8 + end_col + 1;

            piece_maps[ROOK][turn] |= (1ULL << rook_start);
            piece_maps[ROOK][turn] &= ~(1ULL << rook_end);
        } else { // kingside
            board[end_row][7] = board[end_row][end_col - 1];
            board[end_row][end_col - 1] = 0;

            int rook = board[end_row][7];
            int rook_start = start_row * 8 + 7;
            int rook_end = end_row * 8 + end_col - 1;

            piece_maps[ROOK][turn] |= (1ULL << rook_start);
            piece_maps[ROOK][turn] &= ~(1ULL << rook_end);
        }
    }

    // en passant
    if (piece_type == PAWN && start_col != end_col && !capture) {
        board[end_row][end_col] = 0;
        board[start_row][end_col] = (turn == WHITE) ? BLACK_PAWN : WHITE_PAWN;

        int square = start_row * 8 + end_col;
        piece_maps[PAWN][!turn] |= (1ULL << square);

        if (turn == WHITE) {
            black_material += values[PAWN];
        } else {
            white_material += values[PAWN];
        }
    }

    int promotion = move.promote_to();
    if (promotion) {
        board[start_row][start_col] = (turn == WHITE) ? WHITE_PAWN : BLACK_PAWN;

        piece_maps[PAWN][turn] |= (1ULL << start_square);

        if (turn == WHITE) {
            white_material -= values[promotion] - values[PAWN];
        } else {
            black_material -= values[promotion] - values[PAWN];
        }

        piece_maps[promotion][turn] &= ~(1ULL << start_square);
        piece_maps[promotion][turn] &= ~(1ULL << end_square);
    }

    for (int i = 0; i < 4; i++) {
        can_castle[i] = move_info.prev_castle[i];
    }

    // update kings
    if (piece == WHITE_KING) {
        king_positions[0] = start_row;
        king_positions[1] = start_col;
    } else if (piece == BLACK_KING) {
        king_positions[2] = start_row;
        king_positions[3] = start_col;
    }

    piece_maps[KING][WHITE] = 1ULL << (king_positions[0] * 8 + king_positions[1]);
    piece_maps[KING][BLACK] = 1ULL << (king_positions[2] * 8 + king_positions[3]);

    // update white/black maps
    white_pieces = piece_maps[KING][WHITE] | piece_maps[QUEEN][WHITE] | piece_maps[ROOK][WHITE] |
        piece_maps[BISHOP][WHITE] | piece_maps[KNIGHT][WHITE] | piece_maps[PAWN][WHITE];
    black_pieces = piece_maps[KING][BLACK] | piece_maps[QUEEN][BLACK] | piece_maps[ROOK][BLACK] |
        piece_maps[BISHOP][BLACK] | piece_maps[KNIGHT][BLACK] | piece_maps[PAWN][BLACK];

    repetitions = move_info.prev_repetitions;
    last_threefold_reset = move_info.prev_threefold_reset;
    en_passant_col = move_info.prev_en_passant;
    fifty_move_count = move_info.prev_fifty_move;
    hash_value = move_info.prev_hash;

    // revert accumulator
    std::memcpy(NNUE::accumulators[WHITE], acc_white_stack[half_moves], HIDDEN_SIZE * sizeof(int16_t));
    std::memcpy(NNUE::accumulators[BLACK], acc_black_stack[half_moves], HIDDEN_SIZE * sizeof(int16_t));
}

void Position::pop() {
    if (half_moves > 0) {
        unmake_move(move_stack[half_moves - 1]);
    }
}


std::string Position::get_draw() {
    if (half_moves - last_threefold_reset >= 100) {
        return "50 move rule";
    }

    if (repetitions >= 3) {
        return "threefold repetition";
    }

    // insufficient material
    if (white_material + black_material > values[BISHOP] * 2) {
        return "";
    }

    int num_pieces = num_set_bits(white_pieces | black_pieces) - 2;
    
    if (num_pieces == 0) {
        return "insufficient material";
    }
    
    if (num_pieces == 1 && (piece_maps[KNIGHT][WHITE] || piece_maps[KNIGHT][BLACK] ||
        piece_maps[BISHOP][WHITE] || piece_maps[BISHOP][BLACK])) {

        return "insufficient material";
    }

    if (num_pieces == 2) {
        // one bishop each is a draw if same color
        uint64_t LIGHT_SQUARES = 0xAA55AA55AA55AA55ULL;
        uint64_t DARK_SQUARES = 0x55AA55AA55AA55AAULL;

        if ((((LIGHT_SQUARES & piece_maps[BISHOP][WHITE]) && (LIGHT_SQUARES & piece_maps[BISHOP][BLACK])) ||
             ((DARK_SQUARES & piece_maps[BISHOP][WHITE]) && (DARK_SQUARES & piece_maps[BISHOP][BLACK])))) {
            return "insufficient material";
        }
    }

    return "";
}


bool Position::no_legal_moves() {
    uint64_t pieces = turn == WHITE ? white_pieces : black_pieces;
    while (pieces) {
        int from_square = least_set_bit(pieces);
        pieces &= pieces - 1;

        int row = from_square / 8;
        int col = from_square % 8;

        uint64_t move_map = get_piece_moves(row, col);
        while (move_map) {
            int to_square = least_set_bit(move_map);
            move_map &= move_map - 1;

            if (is_legal(Move(row * 8 + col, to_square))) {
                return false;
            }
        }
    }

    return true;
}

std::string Position::get_terminal_state(int legals) {
    if (legals <= 0 && (!legals || no_legal_moves())) {
        if (in_check()) {
            std::string winner = (turn == WHITE) ? "black" : "white";
            return winner + " win";
        } else {
            return "stalemate";
        }
    }

    return get_draw();
}