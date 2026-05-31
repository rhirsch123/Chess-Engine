#include "position.hh"

void Position::set_fen(std::string fen) {
    std::istringstream fen_stream(fen);
    std::string fen_board, fen_turn, fen_castling, fen_ep;
    int full_moves;
    fen_stream >> fen_board >> fen_turn >> fen_castling >> fen_ep >> fifty_move_count >> full_moves;

    half_moves = 0;

    for (int i = 0; i < 64; i++) {
        board[i] = 0;
    }
    castle_rights = 0;
    for (int i = PAWN; i <= ALL_PIECES; i++) {
        piece_maps[i][WHITE] = 0ULL;
        piece_maps[i][BLACK] = 0ULL;
    }

    white_material = 0;
    black_material = 0;

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

            board[square] = piece;

            int piece_type = get_piece_type(piece);
            int color = get_color(piece);

            if (color == WHITE) {
                white_material += piece_values[piece_type];
            } else {
                black_material += piece_values[piece_type];
            }

            piece_maps[piece_type][color] |= square_bitboard(square);
            piece_maps[ALL_PIECES][color] |= square_bitboard(square);

            square++;
        }
    }

    turn = (fen_turn == "w") ? WHITE : BLACK;

    for (char c : fen_castling) {
        if (c == 'K') castle_rights |= WHITE_KINGSIDE;
        else if (c == 'Q') castle_rights |= WHITE_QUEENSIDE;
        else if (c == 'k') castle_rights |= BLACK_KINGSIDE;
        else if (c == 'q') castle_rights |= BLACK_QUEENSIDE;
    }

    if (fen_ep != "-" && fen_ep != "–") {
        en_passant_col = fen_ep[0] - 'a';
    } else {
        en_passant_col = -1;
    }

    position_history.resize(1024);
    move_stack.resize(1024);

    hash_value = Zobrist::get_hash(board, castle_rights, en_passant_col, turn);
    pawn_hash = Zobrist::get_pawn_hash(piece_maps[PAWN][WHITE] | piece_maps[PAWN][BLACK]);
    position_history[0] = hash_value;
    checkers = get_attackers(lsb(piece_maps[KING][turn]), !turn);
    NNUE::reset_accumulators(*this);
}

Position::Position(std::string fen) {
    Zobrist::init();
    init_bitboards();    
    NNUE::init();

    set_fen(fen);
}


uint64_t Position::pieces() {
    return piece_maps[ALL_PIECES][WHITE] | piece_maps[ALL_PIECES][BLACK];
}

bool Position::is_attacked(int square) {
    uint64_t all_pieces = pieces();

    if ((piece_maps[ROOK][!turn] | piece_maps[QUEEN][!turn]) & get_rook_moves(square, all_pieces)) {
        return true;
    }
    if ((piece_maps[BISHOP][!turn] | piece_maps[QUEEN][!turn]) & get_bishop_moves(square, all_pieces)) {
        return true;
    }
    if (get_knight_moves(square) & piece_maps[KNIGHT][!turn]) {
        return true;
    }
    if (get_pawn_attacks(square, turn) & piece_maps[PAWN][!turn]) {
        return true;
    }
    return (piece_maps[KING][!turn] & get_king_moves(square)) != 0;
}


uint64_t Position::get_attackers(int square, int color, uint64_t blockers) {
    // default to full occupancy
    if (!blockers) {
        blockers = pieces();
    }

    uint64_t attackers = piece_maps[PAWN][color] & get_pawn_attacks(square, !color);
    attackers |= piece_maps[KNIGHT][color] & get_knight_moves(square);
    attackers |= (piece_maps[BISHOP][color] | piece_maps[QUEEN][color]) & get_bishop_moves(square, blockers);
    attackers |= (piece_maps[ROOK][color] | piece_maps[QUEEN][color]) & get_rook_moves(square, blockers);
    attackers |= piece_maps[KING][color] & get_king_moves(square);

    return attackers;
}


void Position::make_move(Move move) {
    if (half_moves >= position_history.size()) {
        move_stack.resize(position_history.size() * 2);
        position_history.resize(position_history.size() * 2);
    }

    int start_square = move.from();
    int end_square = move.to();

    int start_row = start_square / 8;
    int start_col = start_square % 8;
    int end_row = end_square / 8;
    int end_col = end_square % 8;

    int piece = board[start_square];
    int piece_type = get_piece_type(piece);
    int capture = board[end_square];

    MoveInfo move_info = {
        hash_value,
        pawn_hash,
        checkers,
        move,
        en_passant_col,
        fifty_move_count,
        capture,
        last_threefold_reset,
        castle_rights
    };
    move_stack[half_moves] = move_info;

    // nnue
    NNUE::DirtyPieces dps;
    NNUE::DirtyType type = NNUE::QUIET;

    int white_king = lsb(piece_maps[KING][WHITE]);
    int black_king = lsb(piece_maps[KING][BLACK]);
    bool white_mirror = NNUE::is_mirrored(white_king);
    bool black_mirror = NNUE::is_mirrored(black_king);

    dps.white_sub0 = NNUE::make_index<WHITE>(start_square, piece, white_mirror);
    dps.black_sub0 = NNUE::make_index<BLACK>(start_square, piece, black_mirror);
    dps.white_add0 = NNUE::make_index<WHITE>(end_square, piece, white_mirror);
    dps.black_add0 = NNUE::make_index<BLACK>(end_square, piece, black_mirror);

    board[start_square] = 0;
    hash_value ^= Zobrist::piece_table[piece - 1][start_square];

    if (piece_type == PAWN) {
        pawn_hash ^= Zobrist::pawn_table[start_square];
    }

    // move piece
    piece_maps[piece_type][turn] &= ~square_bitboard(start_square);
    // unless promotion:
    piece_maps[piece_type][turn] |= square_bitboard(end_square);

    if (capture) {
        hash_value ^= Zobrist::piece_table[capture - 1][end_square];
        int capture_type = get_piece_type(capture);
        piece_maps[capture_type][!turn] &= ~square_bitboard(end_square);

        if (capture_type == PAWN) {
            pawn_hash ^= Zobrist::pawn_table[end_square];
        }

        if (turn == WHITE) {
            black_material -= piece_values[capture_type];
        } else {
            white_material -= piece_values[capture_type];
        }

        // update castle
        if (capture == WHITE_ROOK) {
            if (end_square == 63 && (castle_rights & WHITE_KINGSIDE)) {
                castle_rights &= ~WHITE_KINGSIDE;
                hash_value ^= Zobrist::castle_table[lsb(WHITE_KINGSIDE)];
            } else if (end_square == 56 && (castle_rights & WHITE_QUEENSIDE)) {
                castle_rights &= ~WHITE_QUEENSIDE;
                hash_value ^= Zobrist::castle_table[lsb(WHITE_QUEENSIDE)];
            } 
        } else if (capture == BLACK_ROOK) {
            if (end_square == 7 && (castle_rights & BLACK_KINGSIDE)) {
                castle_rights &= ~BLACK_KINGSIDE;
                hash_value ^= Zobrist::castle_table[lsb(BLACK_KINGSIDE)];
            } else if (end_square == 0 && (castle_rights & BLACK_QUEENSIDE)) {
                castle_rights &= ~BLACK_QUEENSIDE;
                hash_value ^= Zobrist::castle_table[lsb(BLACK_QUEENSIDE)];
            }
        }

        type = NNUE::CAPTURE;
        dps.white_sub1 = NNUE::make_index<WHITE>(end_square, capture, white_mirror);
        dps.black_sub1 = NNUE::make_index<BLACK>(end_square, capture, black_mirror);
    }

    int promotion = move.promote_to();
    if (promotion) {
        int new_piece = promotion + 1 + (6 * turn);
        board[end_square] = new_piece;

        hash_value ^= Zobrist::piece_table[new_piece - 1][end_square];
        piece_maps[promotion][turn] |= square_bitboard(end_square);

        // clear pawn on last rank
        piece_maps[piece_type][turn] &= ~square_bitboard(end_square);

        if (turn == WHITE) {
            white_material += piece_values[promotion] - piece_values[PAWN];
        } else {
            black_material += piece_values[promotion] - piece_values[PAWN];
        }

        if (capture) {
            type = NNUE::CAP_PROMO;
        } else {
            type = NNUE::PROMOTION;
        }
        dps.white_add0 = NNUE::make_index<WHITE>(end_square, new_piece, white_mirror);
        dps.black_add0 = NNUE::make_index<BLACK>(end_square, new_piece, black_mirror);

    } else if (piece_type == KING && std::abs(start_col - end_col) > 1) {
        // castle
        board[end_square] = piece;
        hash_value ^= Zobrist::piece_table[piece - 1][end_square];

        type = NNUE::CASTLE;

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

        int rook = board[rook_start];
        board[rook_end] = rook;
        board[rook_start] = 0;

        hash_value ^= Zobrist::piece_table[rook - 1][rook_start];
        hash_value ^= Zobrist::piece_table[rook - 1][rook_end];

        piece_maps[ROOK][turn] &= ~square_bitboard(rook_start);
        piece_maps[ROOK][turn] |= square_bitboard(rook_end);

        dps.white_sub1 = NNUE::make_index<WHITE>(rook_start, rook, white_mirror);
        dps.black_sub1 = NNUE::make_index<BLACK>(rook_start, rook, black_mirror);
        dps.white_add1 = NNUE::make_index<WHITE>(rook_end, rook, white_mirror);
        dps.black_add1 = NNUE::make_index<BLACK>(rook_end, rook, black_mirror);

    } else if (piece_type == PAWN && start_col != end_col && !board[end_square]) {
        // en passant
        int capture_square = start_row * 8 + end_col;
        board[capture_square] = 0;
        board[end_square] = piece;

        int captured_pawn = piece == WHITE_PAWN ? BLACK_PAWN : WHITE_PAWN;
        hash_value ^= Zobrist::piece_table[captured_pawn - 1][capture_square];
        hash_value ^= Zobrist::piece_table[piece - 1][end_square];
        pawn_hash ^= Zobrist::pawn_table[capture_square];
        pawn_hash ^= Zobrist::pawn_table[end_square];

        // clear capture
        if (turn == WHITE) {
            piece_maps[PAWN][BLACK] &= ~square_bitboard(capture_square);
            black_material -= piece_values[PAWN];
        } else {
            piece_maps[PAWN][WHITE] &= ~square_bitboard(capture_square);
            white_material -= piece_values[PAWN];
        }

        dps.white_sub1 = NNUE::make_index<WHITE>(capture_square, captured_pawn, white_mirror);
        dps.black_sub1 = NNUE::make_index<BLACK>(capture_square, captured_pawn, black_mirror);

        type = NNUE::EN_PASSANT;

    } else {
        board[end_square] = piece;
        hash_value ^= Zobrist::piece_table[piece - 1][end_square];

        if (piece_type == PAWN) {
            pawn_hash ^= Zobrist::pawn_table[end_square];
        }
    }

    // update castle rights
    if (piece == WHITE_KING) {
        if (castle_rights & WHITE_KINGSIDE) {
            hash_value ^= Zobrist::castle_table[lsb(WHITE_KINGSIDE)];
        }
        if (castle_rights & WHITE_QUEENSIDE) {
            hash_value ^= Zobrist::castle_table[lsb(WHITE_QUEENSIDE)];
        }
        castle_rights &= ~WHITE_KINGSIDE;
        castle_rights &= ~WHITE_QUEENSIDE;
    } else if (piece == BLACK_KING) {
        if (castle_rights & BLACK_KINGSIDE) {
            hash_value ^= Zobrist::castle_table[lsb(BLACK_KINGSIDE)];
        }
        if (castle_rights & BLACK_QUEENSIDE) {
            hash_value ^= Zobrist::castle_table[lsb(BLACK_QUEENSIDE)];
        }
        castle_rights &= ~BLACK_KINGSIDE;
        castle_rights &= ~BLACK_QUEENSIDE;
    } else if (piece == WHITE_ROOK) {
        if (start_square == 63 && (castle_rights & WHITE_KINGSIDE)) {
            hash_value ^= Zobrist::castle_table[lsb(WHITE_KINGSIDE)];
            castle_rights &= ~WHITE_KINGSIDE;
        } else if (start_square == 56 && (castle_rights & WHITE_QUEENSIDE)) {
            hash_value ^= Zobrist::castle_table[lsb(WHITE_QUEENSIDE)];
            castle_rights &= ~WHITE_QUEENSIDE;
        }
    } else if (piece == BLACK_ROOK) {
        if (start_square == 7 && (castle_rights & BLACK_KINGSIDE)) {
            hash_value ^= Zobrist::castle_table[lsb(BLACK_KINGSIDE)];
            castle_rights &= ~BLACK_KINGSIDE;
        } else if (start_square == 0 && (castle_rights & BLACK_QUEENSIDE)) {
            hash_value ^= Zobrist::castle_table[lsb(BLACK_QUEENSIDE)];
            castle_rights &= ~BLACK_QUEENSIDE;
        }
    }

    for (int i = WHITE; i <= BLACK; i++) {
        piece_maps[ALL_PIECES][i] = 0ULL;
        for (int j = PAWN; j <= KING; j++) {
            piece_maps[ALL_PIECES][i] |= piece_maps[j][i];
        }
    }

    // update en passant col
    if (en_passant_col != -1) {
        hash_value ^= Zobrist::en_passant_table[en_passant_col];
    }
    bool en_passant_possible = false;
    if (piece_type == PAWN && std::abs(start_row - end_row) > 1) {
        if (turn == BLACK) {
            if ((start_col - 1 >= 0 && board[3 * 8 + start_col - 1] == WHITE_PAWN) ||
                (start_col + 1 < 8 && board[3 * 8 + start_col + 1] == WHITE_PAWN)) {
                en_passant_possible = true;
            }
        } else {
            if ((start_col - 1 >= 0 && board[4 * 8 + start_col - 1] == BLACK_PAWN) ||
                (start_col + 1 < 8 && board[4 * 8 + start_col + 1] == BLACK_PAWN)) {
                en_passant_possible = true;
            }
        }
    }
    if (en_passant_possible) {
        en_passant_col = start_col;
        hash_value ^= Zobrist::en_passant_table[en_passant_col];
    } else {
        en_passant_col = -1;
    }

    half_moves++;
   
    if (capture || piece_type == PAWN) {
        last_threefold_reset = half_moves;
    }

    turn = !turn;
    hash_value ^= Zobrist::turn_key;

    // for threefold repetition detection
    position_history[half_moves] = hash_value;

    checkers = get_attackers(lsb(piece_maps[KING][turn]), !turn);

    dps.type = type;
    NNUE::set_dirty(half_moves, dps);

    // if the king switches halves, the accumulator cannot be efficiently updated
    if (piece_type == KING && NNUE::is_mirrored(start_square) != NNUE::is_mirrored(end_square)) {
        NNUE::reset_accumulators(*this);
    }
}


void Position::unmake_move(MoveInfo& move_info) {
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

    int piece = board[end_square];
    int piece_type = get_piece_type(piece);
    board[start_square] = piece;
    board[end_square] = capture;

    piece_maps[piece_type][turn] |= square_bitboard(start_square); // unless promotion
    piece_maps[piece_type][turn] &= ~square_bitboard(end_square);

    if (capture) {
        int capture_type = get_piece_type(capture);
        piece_maps[capture_type][!turn] |= square_bitboard(end_square);

        if (turn == WHITE) {
            black_material += piece_values[capture_type];
        } else {
            white_material += piece_values[capture_type];
        }
    }

    // castle
    if (piece_type == KING && std::abs(start_col - end_col) > 1) {
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

        int rook = board[rook_end];
        board[rook_start] = rook;
        board[rook_end] = 0;

        piece_maps[ROOK][turn] |= square_bitboard(rook_start);
        piece_maps[ROOK][turn] &= ~square_bitboard(rook_end);
    }

    // en passant
    if (piece_type == PAWN && start_col != end_col && !capture) {
        int capture_square = start_row * 8 + end_col;

        board[end_square] = 0;
        board[capture_square] = (turn == WHITE) ? BLACK_PAWN : WHITE_PAWN;

        piece_maps[PAWN][!turn] |= square_bitboard(capture_square);

        if (turn == WHITE) {
            black_material += piece_values[PAWN];
        } else {
            white_material += piece_values[PAWN];
        }
    }

    int promotion = move.promote_to();
    if (promotion) {
        board[start_square] = (turn == WHITE) ? WHITE_PAWN : BLACK_PAWN;

        piece_maps[PAWN][turn] |= square_bitboard(start_square);

        if (turn == WHITE) {
            white_material -= piece_values[promotion] - piece_values[PAWN];
        } else {
            black_material -= piece_values[promotion] - piece_values[PAWN];
        }

        piece_maps[promotion][turn] &= ~square_bitboard(start_square);
        piece_maps[promotion][turn] &= ~square_bitboard(end_square);
    }

    for (int i = WHITE; i <= BLACK; i++) {
        piece_maps[ALL_PIECES][i] = 0ULL;
        for (int j = PAWN; j <= KING; j++) {
            piece_maps[ALL_PIECES][i] |= piece_maps[j][i];
        }
    }

    castle_rights = move_info.prev_castle;
    last_threefold_reset = move_info.prev_threefold_reset;
    en_passant_col = move_info.prev_en_passant;
    fifty_move_count = move_info.prev_fifty_move;
    hash_value = move_info.prev_hash;
    pawn_hash = move_info.prev_pawn_hash;
    checkers = move_info.prev_checkers;
}

void Position::pop() {
    if (half_moves > 0) {
        unmake_move(move_stack[half_moves - 1]);
    }
}


// static exchange evaluation
bool Position::SEE(Move move, int threshold) {
    int piece = move.promote_to() ? move.promote_to() : board[move.from()];
    int move_turn = get_color(piece);
    int current_turn = !move_turn;
    int capture = board[move.to()];
    int exchange = capture ? piece_values[get_piece_type(capture)] : 0;
    int last_attacker = get_piece_type(piece);

    if (exchange >= piece_values[last_attacker] + threshold) {
        return true;
    }

    // sliding piece blockers - start out as every piece
    uint64_t blockers = pieces();

    int start_square = move.from();
    int end_square = move.to();
    
    // update blockers with move
    blockers &= ~square_bitboard(start_square);
    blockers |= square_bitboard(end_square);

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

            uint64_t lowest_attackers = piece_maps[weakest_piece][current_turn] & attackers;
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
        blockers &= ~square_bitboard(next_attacker_square);

        current_turn = !current_turn;
        attackers = get_attackers(end_square, current_turn, blockers) & blockers;
    }

    return exchange >= threshold;
}


// check if the position is drawn, ignoring stalemate
TerminalState Position::get_draw() {
    if (half_moves - last_threefold_reset >= 100) {
        return FIFTY_MOVE_RULE;
    }

    // insufficient material
    int num_pieces = popcount(pieces()) - 2;
    if (num_pieces == 0) {
        return INSUFFICIENT_MATERIAL;
    }
    if (num_pieces == 1 && (piece_maps[KNIGHT][WHITE] || piece_maps[KNIGHT][BLACK] ||
        piece_maps[BISHOP][WHITE] || piece_maps[BISHOP][BLACK])) {

        return INSUFFICIENT_MATERIAL;
    }
    if (num_pieces == 2) {
        // one bishop each is a draw if same color
        constexpr uint64_t LIGHT_SQUARES = 0xAA55AA55AA55AA55ULL;
        constexpr uint64_t DARK_SQUARES = 0x55AA55AA55AA55AAULL;

        if ((((LIGHT_SQUARES & piece_maps[BISHOP][WHITE]) && (LIGHT_SQUARES & piece_maps[BISHOP][BLACK])) ||
            ((DARK_SQUARES & piece_maps[BISHOP][WHITE]) && (DARK_SQUARES & piece_maps[BISHOP][BLACK])))) {
            return INSUFFICIENT_MATERIAL;
        }
    }

    int repetitions = 1;
    for (int i = half_moves - 4; i >= last_threefold_reset; i -= 2) {
        if (position_history[i] == hash_value) {
            repetitions++;
        }
        if (repetitions == 3) {
            return THREEFOLD_REPETITION;
        }
    }

    return NONE;
}

TerminalState Position::get_terminal_state() {
    if (no_legal_moves(*this)) {
        if (checkers) {
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