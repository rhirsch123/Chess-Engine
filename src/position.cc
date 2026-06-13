#include "position.hh"

void Position::set_keys() {
    state->hash_value = 0ULL;
    state->pawn_hash = 0ULL;
    state->nonpawn_hash[WHITE] = 0ULL;
    state->nonpawn_hash[BLACK] = 0ULL;

    uint64_t pcs = occupancy();
    while (pcs) {
        int square = pop_lsb(pcs);
        int piece = piece_on(square);

        state->hash_value ^= Zobrist::piece_table[piece - 1][square];

        if (get_piece_type(piece) == PAWN) {
            state->pawn_hash ^= Zobrist::piece_table[piece - 1][square];
        } else {
            state->nonpawn_hash[get_color(piece)] ^= Zobrist::piece_table[piece - 1][square];
        }
    }

    for (int i = 0; i < 4; i++) {
        if (castle_rights() & (1 << i)) {
            state->hash_value ^= Zobrist::castle_table[i];
        }
    }

    if (ep_col() >= 0 && ep_col() < 8) {
        state->hash_value ^= Zobrist::en_passant_table[ep_col()];
    }

    if (turn == BLACK) {
        state->hash_value ^= Zobrist::turn_key;
    }
}

void Position::set_fen(std::string fen) {
    stack.resize(512);
    half_moves = 0;
    state = &stack[half_moves];

    std::istringstream fen_stream(fen);
    std::string fen_board, fen_turn, fen_castling, fen_ep;
    int full_moves;
    fen_stream >> fen_board >> fen_turn >> fen_castling >> fen_ep >> state->fifty_move_count >> full_moves;

    std::memset(state->board, 0, sizeof(state->board));
    state->castle_rights = 0;
    for (int i = PAWN; i <= ALL_PIECES; i++) {
        state->piece_maps[i][WHITE] = 0ULL;
        state->piece_maps[i][BLACK] = 0ULL;
    }

    state->white_material = 0;
    state->black_material = 0;

    int square = 0;
    for (char c : fen_board) {
        if (c == '/') {
            continue;
        } else if (isdigit(c)) {
            square += c - '0';
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

            state->board[square] = piece;

            int piece_type = get_piece_type(piece);
            int color = get_color(piece);

            if (color == WHITE) {
                state->white_material += piece_values[piece_type];
            } else {
                state->black_material += piece_values[piece_type];
            }

            state->piece_maps[piece_type][color] |= square_bb(square);
            state->piece_maps[ALL_PIECES][color] |= square_bb(square);

            square++;
        }
    }

    turn = (fen_turn == "w") ? WHITE : BLACK;

    for (char c : fen_castling) {
        if (c == 'K') state->castle_rights |= WHITE_KINGSIDE;
        else if (c == 'Q') state->castle_rights |= WHITE_QUEENSIDE;
        else if (c == 'k') state->castle_rights |= BLACK_KINGSIDE;
        else if (c == 'q') state->castle_rights |= BLACK_QUEENSIDE;
    }

    if (fen_ep != "-" && fen_ep != "–") {
        state->en_passant_col = fen_ep[0] - 'a';
    } else {
        state->en_passant_col = -1;
    }

    set_keys();
    state->checkers = get_attackers(king_square(turn), !turn, occupancy());
    state->pinned[WHITE] = get_pinned(WHITE);
    state->pinned[BLACK] = get_pinned(BLACK);
}

Position::Position(std::string fen) {
    Zobrist::init();
    init_bitboards();    
    set_fen(fen);
}


uint64_t Position::get_attackers(int square, int color, uint64_t blockers) {
    uint64_t attackers = piece_bb(PAWN, color) & get_pawn_attacks(square, !color);
    attackers |= piece_bb(KNIGHT, color) & get_knight_moves(square);
    attackers |= (piece_bb(BISHOP, color) | piece_bb(QUEEN, color)) & get_bishop_moves(square, blockers);
    attackers |= (piece_bb(ROOK, color) | piece_bb(QUEEN, color)) & get_rook_moves(square, blockers);
    attackers |= piece_bb(KING, color) & get_king_moves(square);

    return attackers;
}

bool Position::is_attacked(int square) {
    return bool(get_attackers(square, !turn, occupancy()));
}

uint64_t Position::get_pinned(int color) {
    int king_sq = king_square(color);
    uint64_t pinned = 0ULL;
    uint64_t pinners = (get_rook_moves(king_sq, 0ULL) & (piece_bb(ROOK, !color) | piece_bb(QUEEN, !color))) |
                       (get_bishop_moves(king_sq, 0ULL) & (piece_bb(BISHOP, !color) | piece_bb(QUEEN, !color)));

    uint64_t occ = occupancy() & ~pinners;
    while (pinners) {
        int square = pop_lsb(pinners);
        uint64_t between = get_check_blocks(king_sq, square) & occ;
        if (popcount(between) == 1 && (between & piece_bb(ALL_PIECES, color))) {
            pinned |= between;
        }
    }

    return pinned;
}


void Position::put_piece(int piece, int square) {
    int piece_type = get_piece_type(piece);
    int color = get_color(piece);

    state->board[square] = piece;
    state->piece_maps[piece_type][color] |= square_bb(square);
    state->piece_maps[ALL_PIECES][color] |= square_bb(square);

    state->hash_value ^= Zobrist::piece_table[piece - 1][square];
    if (piece_type == PAWN) {
        state->pawn_hash ^= Zobrist::piece_table[piece - 1][square];
    } else {
        state->nonpawn_hash[color] ^= Zobrist::piece_table[piece - 1][square];
    }
}

void Position::remove_piece(int piece, int square) {
    int piece_type = get_piece_type(piece);
    int color = get_color(piece);

    state->board[square] = 0;
    state->piece_maps[piece_type][color] &= ~square_bb(square);
    state->piece_maps[ALL_PIECES][color] &= ~square_bb(square);
    
    state->hash_value ^= Zobrist::piece_table[piece - 1][square];
    if (piece_type == PAWN) {
        state->pawn_hash ^= Zobrist::piece_table[piece - 1][square];
    } else {
        state->nonpawn_hash[color] ^= Zobrist::piece_table[piece - 1][square];
    }
}

DirtyPieces Position::make_move(Move move) {
    half_moves++;
    if (half_moves >= stack.size()) {
        stack.resize(stack.size() * 2);
    }
    std::memcpy(&stack[half_moves], state, sizeof(PositionState));
    state = &stack[half_moves];

    int start_square = move.from();
    int end_square = move.to();

    int start_row = start_square / 8;
    int start_col = start_square % 8;
    int end_row = end_square / 8;
    int end_col = end_square % 8;

    int piece = piece_on(start_square);
    int piece_type = get_piece_type(piece);
    int capture = piece_on(end_square);

    // nnue
    DirtyPieces dps;
    DirtyType type = DIRTY_QUIET;

    int white_king = king_square(WHITE);
    int black_king = king_square(BLACK);
    bool white_mirror = NNUE::is_mirrored(white_king);
    bool black_mirror = NNUE::is_mirrored(black_king);

    dps.white_sub0 = NNUE::make_index<WHITE>(start_square, piece, white_mirror);
    dps.black_sub0 = NNUE::make_index<BLACK>(start_square, piece, black_mirror);
    dps.white_add0 = NNUE::make_index<WHITE>(end_square, piece, white_mirror);
    dps.black_add0 = NNUE::make_index<BLACK>(end_square, piece, black_mirror);

    if (capture) {
        remove_piece(capture, end_square);

        int capture_type = get_piece_type(capture);
        if (turn == WHITE) {
            state->black_material -= piece_values[capture_type];
        } else {
            state->white_material -= piece_values[capture_type];
        }

        // update castle
        if (capture == WHITE_ROOK) {
            if (end_square == 63 && (castle_rights() & WHITE_KINGSIDE)) {
                state->castle_rights &= ~WHITE_KINGSIDE;
                state->hash_value ^= Zobrist::castle_table[lsb(WHITE_KINGSIDE)];
            } else if (end_square == 56 && (castle_rights() & WHITE_QUEENSIDE)) {
                state->castle_rights &= ~WHITE_QUEENSIDE;
                state->hash_value ^= Zobrist::castle_table[lsb(WHITE_QUEENSIDE)];
            } 
        } else if (capture == BLACK_ROOK) {
            if (end_square == 7 && (castle_rights() & BLACK_KINGSIDE)) {
                state->castle_rights &= ~BLACK_KINGSIDE;
                state->hash_value ^= Zobrist::castle_table[lsb(BLACK_KINGSIDE)];
            } else if (end_square == 0 && (castle_rights() & BLACK_QUEENSIDE)) {
                state->castle_rights &= ~BLACK_QUEENSIDE;
                state->hash_value ^= Zobrist::castle_table[lsb(BLACK_QUEENSIDE)];
            }
        }

        type = DIRTY_CAPTURE;
        dps.white_sub1 = NNUE::make_index<WHITE>(end_square, capture, white_mirror);
        dps.black_sub1 = NNUE::make_index<BLACK>(end_square, capture, black_mirror);
    }

    remove_piece(piece, start_square);
    int promotion = move.promote_to();
    int promo_piece = promotion + 1 + (6 * turn);
    put_piece(promotion ? promo_piece : piece, end_square);

    if (promotion) {
        if (turn == WHITE) {
            state->white_material += piece_values[promotion] - piece_values[PAWN];
        } else {
            state->black_material += piece_values[promotion] - piece_values[PAWN];
        }

        if (capture) {
            type = DIRTY_CAP_PROMO;
        } else {
            type = DIRTY_PROMOTION;
        }
        dps.white_add0 = NNUE::make_index<WHITE>(end_square, promo_piece, white_mirror);
        dps.black_add0 = NNUE::make_index<BLACK>(end_square, promo_piece, black_mirror);

    } else if (piece_type == KING && std::abs(start_col - end_col) > 1) {
        // castle
        int direction = end_col - start_col;
        int rook_start;
        int rook_end;
        if (direction < 0) { // king moves left
            rook_start = start_row * 8;
            rook_end = end_row * 8 + end_col + 1;
        } else {
            rook_start = start_row * 8 + 7;
            rook_end = end_row * 8 + end_col - 1;
        }

        int rook = piece_on(rook_start);

        remove_piece(rook, rook_start);
        put_piece(rook, rook_end);

        type = DIRTY_CASTLE;
        dps.white_sub1 = NNUE::make_index<WHITE>(rook_start, rook, white_mirror);
        dps.black_sub1 = NNUE::make_index<BLACK>(rook_start, rook, black_mirror);
        dps.white_add1 = NNUE::make_index<WHITE>(rook_end, rook, white_mirror);
        dps.black_add1 = NNUE::make_index<BLACK>(rook_end, rook, black_mirror);

    } else if (piece_type == PAWN && start_col != end_col && !capture) {
        // en passant
        int captured_pawn = piece == WHITE_PAWN ? BLACK_PAWN : WHITE_PAWN;
        int capture_square = start_row * 8 + end_col;
        remove_piece(captured_pawn, capture_square);

        if (turn == WHITE) {
            state->black_material -= piece_values[PAWN];
        } else {
            state->white_material -= piece_values[PAWN];
        }

        type = DIRTY_EP;
        dps.white_sub1 = NNUE::make_index<WHITE>(capture_square, captured_pawn, white_mirror);
        dps.black_sub1 = NNUE::make_index<BLACK>(capture_square, captured_pawn, black_mirror);
    }

    // update castle rights
    if (piece == WHITE_KING) {
        if (castle_rights() & WHITE_KINGSIDE) {
            state->hash_value ^= Zobrist::castle_table[lsb(WHITE_KINGSIDE)];
        }
        if (castle_rights() & WHITE_QUEENSIDE) {
            state->hash_value ^= Zobrist::castle_table[lsb(WHITE_QUEENSIDE)];
        }
        state->castle_rights &= ~WHITE_KINGSIDE;
        state->castle_rights &= ~WHITE_QUEENSIDE;
    } else if (piece == BLACK_KING) {
        if (castle_rights() & BLACK_KINGSIDE) {
            state->hash_value ^= Zobrist::castle_table[lsb(BLACK_KINGSIDE)];
        }
        if (castle_rights() & BLACK_QUEENSIDE) {
            state->hash_value ^= Zobrist::castle_table[lsb(BLACK_QUEENSIDE)];
        }
        state->castle_rights &= ~BLACK_KINGSIDE;
        state->castle_rights &= ~BLACK_QUEENSIDE;
    } else if (piece == WHITE_ROOK) {
        if (start_square == 63 && (castle_rights() & WHITE_KINGSIDE)) {
            state->hash_value ^= Zobrist::castle_table[lsb(WHITE_KINGSIDE)];
            state->castle_rights &= ~WHITE_KINGSIDE;
        } else if (start_square == 56 && (castle_rights() & WHITE_QUEENSIDE)) {
            state->hash_value ^= Zobrist::castle_table[lsb(WHITE_QUEENSIDE)];
            state->castle_rights &= ~WHITE_QUEENSIDE;
        }
    } else if (piece == BLACK_ROOK) {
        if (start_square == 7 && (castle_rights() & BLACK_KINGSIDE)) {
            state->hash_value ^= Zobrist::castle_table[lsb(BLACK_KINGSIDE)];
            state->castle_rights &= ~BLACK_KINGSIDE;
        } else if (start_square == 0 && (castle_rights() & BLACK_QUEENSIDE)) {
            state->hash_value ^= Zobrist::castle_table[lsb(BLACK_QUEENSIDE)];
            state->castle_rights &= ~BLACK_QUEENSIDE;
        }
    }

    // update en passant col
    if (ep_col() != -1) {
        state->hash_value ^= Zobrist::en_passant_table[ep_col()];
    }
    bool en_passant_possible = false;
    if (piece_type == PAWN && std::abs(start_row - end_row) > 1) {
        if (turn == BLACK) {
            if ((start_col - 1 >= 0 && piece_on(3 * 8 + start_col - 1) == WHITE_PAWN) ||
                (start_col + 1 < 8 && piece_on(3 * 8 + start_col + 1) == WHITE_PAWN)) {
                en_passant_possible = true;
            }
        } else {
            if ((start_col - 1 >= 0 && piece_on(4 * 8 + start_col - 1) == BLACK_PAWN) ||
                (start_col + 1 < 8 && piece_on(4 * 8 + start_col + 1) == BLACK_PAWN)) {
                en_passant_possible = true;
            }
        }
    }
    if (en_passant_possible) {
        state->en_passant_col = start_col;
        state->hash_value ^= Zobrist::en_passant_table[ep_col()];
    } else {
        state->en_passant_col = -1;
    }
   
    if (capture || piece_type == PAWN) {
        state->last_threefold_reset = half_moves;
    }

    turn = !turn;
    state->hash_value ^= Zobrist::turn_key;

    state->checkers = get_attackers(king_square(turn), !turn, occupancy());
    state->pinned[turn] = get_pinned(turn);

    dps.type = type;
    return dps;
}


void Position::pop() {
    state = &stack[--half_moves];
    turn = !turn;
}


// static exchange evaluation
bool Position::SEE(Move move, int threshold) {
    int piece = move.promote_to() ? move.promote_to() : piece_on(move.from());
    int move_turn = get_color(piece);
    int current_turn = !move_turn;
    int capture = piece_on(move.to());
    int exchange = capture ? piece_values[get_piece_type(capture)] : 0;
    int last_attacker = get_piece_type(piece);

    if (exchange >= piece_values[last_attacker] + threshold) {
        return true;
    }

    // sliding piece blockers - start out as every piece
    uint64_t blockers = occupancy();

    int start_square = move.from();
    int end_square = move.to();
    
    // update blockers with move
    blockers &= ~square_bb(start_square);
    blockers |= square_bb(end_square);

    uint64_t attackers = get_attackers(end_square, current_turn, blockers) & blockers;

    while (attackers) {
        int next_attacker;
        int next_attacker_square;
        for (int weakest_piece = PAWN; weakest_piece <= KING; weakest_piece++) {
            if (weakest_piece == KING) {
                if (!(get_attackers(end_square, !current_turn, blockers) & blockers)) {
                    if (current_turn == move_turn) {
                        exchange += piece_values[last_attacker];
                    } else {
                        exchange -= piece_values[last_attacker];
                    }
                }
                return exchange >= threshold;
            }

            uint64_t lowest_attackers = piece_bb(weakest_piece, current_turn) & attackers;
            if (lowest_attackers) {
                // choose arbitrary lowest attacker if multiple
                next_attacker_square = lsb(lowest_attackers);
                next_attacker = weakest_piece;
                break;
            }
        }

        if (current_turn == move_turn) {
            exchange += piece_values[last_attacker];
            if (exchange - piece_values[next_attacker] >= threshold) {
                return true;
            }
        } else {
            exchange -= piece_values[last_attacker];
            if (exchange + piece_values[next_attacker] < threshold) {
                return false;
            }
        }

        last_attacker = next_attacker;

        // remove used piece from sliding piece blockers and possible next attackers
        blockers &= ~square_bb(next_attacker_square);

        current_turn = !current_turn;
        attackers = get_attackers(end_square, current_turn, blockers) & blockers;
    }

    return exchange >= threshold;
}


// check if the position is drawn, ignoring stalemate
TerminalState Position::get_draw() {
    if (fifty_move_count() >= 100) {
        return FIFTY_MOVE_RULE;
    }

    // insufficient material
    int num_pieces = popcount(occupancy()) - 2;
    if (num_pieces == 0) {
        return INSUFFICIENT_MATERIAL;
    }
    if (num_pieces == 1 && (piece_bb(KNIGHT, WHITE) || piece_bb(KNIGHT, BLACK)
        || piece_bb(BISHOP, WHITE) || piece_bb(BISHOP, BLACK))) {

        return INSUFFICIENT_MATERIAL;
    }
    if (num_pieces == 2) {
        // one bishop each is a draw if same color
        constexpr uint64_t LIGHT_SQUARES = 0xAA55AA55AA55AA55ULL;
        constexpr uint64_t DARK_SQUARES = 0x55AA55AA55AA55AAULL;

        if ((((LIGHT_SQUARES & piece_bb(BISHOP, WHITE)) && (LIGHT_SQUARES & piece_bb(BISHOP, BLACK)))||
            ((DARK_SQUARES & piece_bb(BISHOP, WHITE)) && (DARK_SQUARES & piece_bb(BISHOP, BLACK))))) {
            return INSUFFICIENT_MATERIAL;
        }
    }

    int repetitions = 1;
    for (int i = half_moves - 4; i >= state->last_threefold_reset; i -= 2) {
        if (stack[i].hash_value == pos_key()) {
            repetitions++;
        }
        if (repetitions == 3) {
            return THREEFOLD_REPETITION;
        }
    }

    return NOT_TERMINAL;
}

TerminalState Position::get_terminal_state() {
    if (no_legal_moves(*this)) {
        if (checkers()) {
            if (turn == WHITE) {
                return BLACK_MATE;
            }
            return WHITE_MATE;
        } else {
            return STALEMATE;
        }
    }

    return get_draw();
}