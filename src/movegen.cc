#include "movegen.hh"


inline void write_piece_moves(uint64_t moves_bb, int start_square, MoveList* move_list) {
#if USE_AVX512_VBMI2
    const __m512i from_squares = _mm512_set1_epi16(uint16_t(start_square << 6));

    const __m512i to_squares =
      _mm512_cvtepi8_epi16(
        _mm512_castsi512_si256(
          _mm512_maskz_compress_epi8(moves_bb, vec_squares)));

    // construct and store move objects in bulk
    const __m512i moves = _mm512_or_si512(from_squares, to_squares);

    _mm512_storeu_si512(move_list->end(), moves);
    move_list->size += popcount(moves_bb);
#else
    while (moves_bb) {
        int to_square = pop_lsb(moves_bb);
        move_list->add(Move(start_square, to_square));
    }
#endif
}

inline void write_pawn_pushes(uint64_t pawns, int offset, MoveList* move_list) {
#if USE_AVX512_VBMI2
    const __m128i from_squares =
        _mm_cvtepi8_epi16(
          _mm512_castsi512_si128(
            _mm512_maskz_compress_epi8(pawns, vec_squares)));
    
    const __m128i to_squares = _mm_adds_epi16(from_squares, _mm_set1_epi16(offset));
    const __m128i moves = _mm_or_si128(_mm_slli_epi16(from_squares, 6), to_squares);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(move_list->end()), moves);
    move_list->size += popcount(pawns);
#else
    while (pawns) {
        int from_square = pop_lsb(pawns);
        move_list->add(Move(from_square, from_square + offset));
    }
#endif
}


inline uint64_t get_castle_moves(Position& position) {
    uint64_t moves = 0ULL;
    if (!position.checkers() && position.castle_rights()) {
        if (position.turn == WHITE) {
            if ((position.castle_rights() & WHITE_KINGSIDE) &&
                !position.piece_on(61) && !position.piece_on(62)) {
                moves |= square_bb(62);
            }
            if ((position.castle_rights() & WHITE_QUEENSIDE) &&
                !position.piece_on(59) && !position.piece_on(58) && !position.piece_on(57)) {
                moves |= square_bb(58);
            }
        } else {
            if ((position.castle_rights() & BLACK_KINGSIDE) &&
                !position.piece_on(5) && !position.piece_on(6)) {
                moves |= square_bb(6);
            }
            if ((position.castle_rights() & BLACK_QUEENSIDE) &&
                !position.piece_on(3) && !position.piece_on(2) && !position.piece_on(1)) {
                moves |= square_bb(2);
            }
        }
    }
    return moves;
}

// moves that would be/are legal not considering if they leave the king in check
void get_pseudo_legal_moves(Position& position, MoveList* move_list, MoveGenType move_type) {
    uint64_t friendly_pieces = position.piece_bb(ALL_PIECES, position.turn);
    uint64_t opp_pieces = position.piece_bb(ALL_PIECES, !position.turn);
    uint64_t all_pieces = friendly_pieces | opp_pieces;
    uint64_t pinned = position.pinned();
    int king_square = position.king_square(position.turn);
    int dir = position.turn == WHITE ? -1 : 1;

    uint64_t targets = ~friendly_pieces;
    if (move_type == GEN_QUIET) {
        targets &= ~opp_pieces;
    } else if (move_type == GEN_TACTIC) {
        targets &= opp_pieces;
    }

    // if double check, moves must be king evasions
    if (popcount(position.checkers()) > 1) {
        uint64_t king_moves = get_king_moves(king_square) & targets;
        write_piece_moves(king_moves, king_square, move_list);
        return;
    }

    // if in check, moves must be king evasions or block/capture the checker
    uint64_t check_blocks =
      position.checkers() ? get_check_blocks(king_square, lsb(position.checkers())) : ~0ULL;

    // quiet pawn moves
    if (move_type != GEN_TACTIC) {
        // pushes
        uint64_t single_push_pawns;
        uint64_t double_push_pawns;
        if (position.turn == WHITE) {
            single_push_pawns = position.piece_bb(PAWN, WHITE) & ~RANK_7 & (~all_pieces << 8);
            double_push_pawns = single_push_pawns & RANK_2 & (~all_pieces << 16);

            single_push_pawns &= check_blocks << 8;
            double_push_pawns &= check_blocks << 16;
        } else {
            single_push_pawns = position.piece_bb(PAWN, BLACK) & ~RANK_2 & (~all_pieces >> 8);
            double_push_pawns = single_push_pawns & RANK_7 & (~all_pieces >> 16);

            single_push_pawns &= check_blocks >> 8;
            double_push_pawns &= check_blocks >> 16;
        }

        write_pawn_pushes(single_push_pawns, dir * 8, move_list);
        write_pawn_pushes(double_push_pawns, dir * 16, move_list);
    }

    // tactical pawn moves
    if (move_type != GEN_QUIET) {
        // en passant
        int ep_col = position.ep_col();
        if (ep_col >= 0) {
            int pawn = WHITE_PAWN + 6 * position.turn;
            int row = 3 + position.turn; // 4 if black
            int from_square = row * 8 + ep_col - 1;
            if (ep_col > 0 && position.piece_on(from_square) == pawn) {
                move_list->add(Move(from_square, (row + dir) * 8 + ep_col));
            }
            from_square += 2;
            if (ep_col < 7 && position.piece_on(from_square) == pawn) {
                move_list->add(Move(from_square, (row + dir) * 8 + ep_col));
            }
        }

        // push promotions
        uint64_t promo_pawns;
        if (position.turn == WHITE) {
            promo_pawns = position.piece_bb(PAWN, WHITE) & RANK_7 & ((~all_pieces & check_blocks) << 8);
        } else {
            promo_pawns = position.piece_bb(PAWN, BLACK) & RANK_2 & ((~all_pieces & check_blocks) >> 8);
        }

        while (promo_pawns) {
            int from_square = pop_lsb(promo_pawns);
            int to_square = from_square + dir * 8;
            for (auto promo : {QUEEN, ROOK, BISHOP, KNIGHT}) {
                move_list->add(Move(from_square, to_square, promo));
            }
        }

        uint64_t attacks;
        uint64_t promo_rank;
        int right_diag;
        int left_diag;
        if (position.turn == WHITE) {
            // shift up
            attacks = position.piece_bb(PAWN, WHITE) >> 8;
            promo_rank = RANK_8;
            right_diag = 7;
            left_diag = 9;
        } else {
            // shift down
            attacks = position.piece_bb(PAWN, BLACK) << 8;
            promo_rank = RANK_1;
            right_diag = -9;
            left_diag = -7;
        }

        // shift left and right
        uint64_t right_captures = ((attacks & 0x7f7f7f7f7f7f7f7f) << 1) & opp_pieces & check_blocks;
        uint64_t left_captures = ((attacks & 0xfefefefefefefefe) >> 1) & opp_pieces & check_blocks;

        uint64_t right_promo_captures = right_captures & promo_rank;
        right_captures &= ~promo_rank;
        uint64_t left_promo_captures = left_captures & promo_rank;
        left_captures &= ~promo_rank;
        
        while (right_captures) {
            int to_square = pop_lsb(right_captures);
            move_list->add(Move(to_square + right_diag, to_square));
        }
        while (left_captures) {
            int to_square = pop_lsb(left_captures);
            move_list->add(Move(to_square + left_diag, to_square));
        }

        while (right_promo_captures) {
            int to_square = pop_lsb(right_promo_captures);
            int from_square = to_square + right_diag;
            for (auto promo : {QUEEN, ROOK, BISHOP, KNIGHT}) {
                move_list->add(Move(from_square, to_square, promo));
            }
        }
        while (left_promo_captures) {
            int to_square = pop_lsb(left_promo_captures);
            int from_square = to_square + left_diag;
            for (auto promo : {QUEEN, ROOK, BISHOP, KNIGHT}) {
                move_list->add(Move(from_square, to_square, promo));
            }
        }
    }

    uint64_t piece_targets = targets & check_blocks;

    // normal piece moves
    uint64_t knights = position.piece_bb(KNIGHT, position.turn) & ~pinned;
    while (knights) {
        int from_square = pop_lsb(knights);
        uint64_t moves = get_knight_moves(from_square) & piece_targets;
        write_piece_moves(moves, from_square, move_list);
    }

    uint64_t bishops = position.piece_bb(BISHOP, position.turn) | position.piece_bb(QUEEN, position.turn);
    while (bishops) {
        int from_square = pop_lsb(bishops);
        uint64_t moves = get_bishop_moves(from_square, all_pieces) & piece_targets;
        if (pinned & square_bb(from_square)) {
            moves &= get_line_bb(from_square, king_square);
        }
        write_piece_moves(moves, from_square, move_list);
    }

    uint64_t rooks = position.piece_bb(ROOK, position.turn) | position.piece_bb(QUEEN, position.turn);
    while (rooks) {
        int from_square = pop_lsb(rooks);
        uint64_t moves = get_rook_moves(from_square, all_pieces) & piece_targets;
        if (pinned & square_bb(from_square)) {
            moves &= get_line_bb(from_square, king_square);
        }
        write_piece_moves(moves, from_square, move_list);
    }

    uint64_t king_moves = get_king_moves(king_square) & targets;
    if (move_type != GEN_TACTIC) {
        king_moves |= get_castle_moves(position);
    }
    write_piece_moves(king_moves, king_square, move_list);
}


// move passed in should be pseudo legal
bool is_legal(Position& position, Move move) {
    int king_square = position.king_square(position.turn);

    int from_square = move.from();
    int to_square = move.to();
    
    int from_row = from_square / 8;
    int from_col = from_square % 8;
    int to_row = to_square / 8;
    int to_col = to_square % 8;

    if (king_square == from_square) {
        if (std::abs(to_col - from_col) > 1) {
            // king cannot castle through check
            int dir = from_col < to_col ? 1 : -1;
            for (int i = 1; i <= 2; i++) {
                int new_col = from_col + i * dir;
                if (position.is_attacked(from_row * 8 + new_col)) {
                    return false;
                }
            }
            return true;
        } else {
            return !position.get_attackers(
              to_square, !position.turn, position.occupancy() & ~square_bb(king_square));
        }
    }

    // all generated major piece moves are legal
    int piece_type = Position::get_piece_type(position.piece_on(from_square));
    if (piece_type != PAWN) {
        return true;
    }

    uint64_t from_bb = square_bb(from_square);
    uint64_t to_bb = square_bb(to_square);

    // en passant
    if (from_col != to_col && !position.piece_on(to_square)) {
        int capture_square = from_row * 8 + to_col;
        uint64_t occ = position.occupancy() ^ from_bb ^ to_bb ^ square_bb(capture_square);

        if ((position.piece_bb(BISHOP, !position.turn) | position.piece_bb(QUEEN, !position.turn)) &
             get_bishop_moves(king_square, occ)) {
            return false;
        }
        return !((position.piece_bb(ROOK, !position.turn) | position.piece_bb(QUEEN, !position.turn)) &
                  get_rook_moves(king_square, occ));
    }

    return !(position.pinned() & from_bb) ||
            (to_bb & get_line_bb(from_square, king_square));
}

void get_legal_moves(Position& position, MoveList* move_list) {
    MoveList pseudo_legals;
    get_pseudo_legal_moves(position, &pseudo_legals);

    for (Move move : pseudo_legals) {
        if (is_legal(position, move)) {
            move_list->add(move);
        }
    }
}


// returns bit map of pseudo legal moves from given square
uint64_t get_piece_moves(Position& position, int square) {
    int turn = position.turn;
    int piece = position.piece_on(square);

    if (!piece || turn != Position::get_color(piece)) {
        return 0ULL;
    }

    uint64_t friendly_pieces = position.piece_bb(ALL_PIECES, turn);
    uint64_t opponent_pieces = position.piece_bb(ALL_PIECES, !turn);
    uint64_t all_pieces = friendly_pieces | opponent_pieces;

    uint64_t pinned = position.pinned();
    int king_square = position.king_square(position.turn);
    uint64_t check_blocks =
      popcount(position.checkers()) > 1 ?  0ULL :
      !position.checkers()              ? ~0ULL :
      get_check_blocks(king_square, lsb(position.checkers()));
    
    uint64_t targets = ~friendly_pieces & check_blocks;
    uint64_t from_bb = square_bb(square);

    int piece_type = Position::get_piece_type(piece);
    if (piece_type == PAWN) {
        uint64_t captures = get_pawn_attacks(square, turn) & opponent_pieces & check_blocks;

        int direction = (turn == WHITE) ? -1 : 1;
        int row = square / 8;
        int col = square % 8;

        uint64_t pushes = 0ULL;
        int one_forward = (row + direction) * 8 + col;
        int two_forward = (row + 2 * direction) * 8 + col;
        // one square forward
        if (!position.piece_on(one_forward)) {
            pushes |= square_bb(one_forward) & targets;
        }
        // two squares forward
        if (((direction == -1 && row == 6) || (direction == 1 && row == 1)) &&
            !position.piece_on(one_forward) && !position.piece_on(two_forward)) {
            pushes |= square_bb(two_forward) & targets;
        }

        uint64_t en_passant = 0ULL;
        if (position.ep_col() >= 0) {
            int ep_row = turn == WHITE ? 2 : 5;
            int ep_square = ep_row * 8 + position.ep_col();
            en_passant = get_pawn_attacks(square, turn) & square_bb(ep_square);
        }

        return captures | pushes | en_passant;
    }
    if (piece_type == QUEEN) {
        uint64_t moves = get_queen_moves(square, all_pieces) & targets;
        if (pinned & from_bb) {
            moves &= get_line_bb(square, king_square);
        }
        return moves;
    }
    if (piece_type == ROOK) {
        uint64_t moves = get_rook_moves(square, all_pieces) & targets;
        if (pinned & from_bb) {
            moves &= get_line_bb(square, king_square);
        }
        return moves;
    }
    if (piece_type == BISHOP) {
        uint64_t moves = get_bishop_moves(square, all_pieces) & targets;
        if (pinned & from_bb) {
            moves &= get_line_bb(square, king_square);
        }
        return moves;
    }
    if (piece_type == KNIGHT) {
        if (from_bb & pinned) {
            return 0ULL;
        }
        return get_knight_moves(square) & targets;
    }

    return (get_king_moves(square) | get_castle_moves(position)) & (~friendly_pieces);
}


bool is_pseudo_legal(Position& position, Move move) {
    // edge case: verify that the promotion flag is valid
    bool pawn_move = square_bb(move.from()) & position.piece_bb(PAWN, position.turn);
    bool pawn_valid = !pawn_move || !(square_bb(move.to()) & (RANK_1 | RANK_8)) || move.promote_to();
    bool promote_valid = pawn_move || !move.promote_to();
    return pawn_valid && promote_valid && (get_piece_moves(position, move.from()) & square_bb(move.to()));
}


bool no_legal_moves(Position& position) {
    uint64_t pieces = position.piece_bb(ALL_PIECES, position.turn);
    while (pieces) {
        int from_square = pop_lsb(pieces);

        uint64_t move_map = get_piece_moves(position, from_square);
        while (move_map) {
            int to_square = pop_lsb(move_map);

            if (is_legal(position, Move(from_square, to_square))) {
                return false;
            }
        }
    }

    return true;
}


uint64_t bulk_perft(Position& position, int depth) {
    uint64_t nodes = 0;
    
    MoveList moves;
    get_legal_moves(position, &moves);

    if (position.half_moves == depth - 1) {
        return nodes + moves.size;
    }

    for (Move move : moves) {
        position.make_move(move);
        nodes += bulk_perft(position, depth);
        position.pop();
    }

    return nodes;
}