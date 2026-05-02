#include "movegen.hh"


inline void write_piece_moves(uint64_t moves_bb, int start_square, MoveList* move_list) {
#if USE_AVX512_VBMI2
    const __m512i from_squares = _mm512_set1_epi16((uint16_t) (start_square << 6));

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


inline uint64_t get_castle_moves(Position& position) {
    uint64_t moves = 0ULL;
    if (!position.checkers && position.castle_rights) {
        if (position.turn == WHITE) {
            if ((position.castle_rights & WHITE_KINGSIDE) && !position.board[61] && !position.board[62]) {
                moves |= square_bitboard(62);
            }
            if ((position.castle_rights & WHITE_QUEENSIDE) && !position.board[59] && !position.board[58] && !position.board[57]) {
                moves |= square_bitboard(58);
            }
        } else {
            if ((position.castle_rights & BLACK_KINGSIDE) && !position.board[5] && !position.board[6]) {
                moves |= square_bitboard(6);
            }
            if ((position.castle_rights & BLACK_QUEENSIDE) && !position.board[3] && !position.board[2] && !position.board[1]) {
                moves |= square_bitboard(2);
            }
        }
    }
    return moves;
}

// moves that would be/are legal not considering if they leave the king in check
void get_pseudo_legal_moves(Position& position, MoveList* move_list, MoveGenType move_type) {
    uint64_t friendly_pieces = position.piece_maps[ALL_PIECES][position.turn];
    uint64_t opp_pieces = position.piece_maps[ALL_PIECES][!position.turn];
    uint64_t all_pieces = friendly_pieces | opp_pieces;
    int dir = position.turn == WHITE ? -1 : 1;

    uint64_t targets = ~friendly_pieces;
    if (move_type == QUIET) {
        targets &= ~opp_pieces;
    } else if (move_type == TACTIC) {
        targets &= opp_pieces;
    }

    // if double check, moves must be king evasions
    if (popcount(position.checkers) > 1) {
        int king_square = lsb(position.piece_maps[KING][position.turn]);
        uint64_t king_moves = get_king_moves(king_square) & targets;
        write_piece_moves(king_moves, king_square, move_list);
        return;
    }

    // pawn moves
    if (move_type != TACTIC) {
        // pushes
        uint64_t single_push_pawns;
        uint64_t double_push_pawns;
        if (position.turn == WHITE) {
            single_push_pawns = position.piece_maps[PAWN][WHITE] & ~RANK_7 & (~all_pieces << 8);
            double_push_pawns = single_push_pawns & RANK_2 & (~all_pieces << 16);
        } else {
            single_push_pawns = position.piece_maps[PAWN][BLACK] & ~RANK_2 & (~all_pieces >> 8);
            double_push_pawns = single_push_pawns & RANK_7 & (~all_pieces >> 16);
        }

        while (double_push_pawns) {
            int from_square = pop_lsb(double_push_pawns);
            // all double pushes can be single pushes
            single_push_pawns &= ~square_bitboard(from_square);
            move_list->add(Move(from_square, from_square + dir * 8));
            move_list->add(Move(from_square, from_square + dir * 16));
        }

        while (single_push_pawns) {
            int from_square = pop_lsb(single_push_pawns);
            move_list->add(Move(from_square, from_square + dir * 8));
        }
    }

    if (move_type != QUIET) {
        // en passant
        int ep_col = position.en_passant_col;
        if (ep_col >= 0) {
            int pawn = WHITE_PAWN + 6 * position.turn;
            int row = 3 + position.turn; // 4 if black
            int from_square = row * 8 + ep_col - 1;
            if (ep_col > 0 && position.board[from_square] == pawn) {
                move_list->add(Move(from_square, (row + dir) * 8 + ep_col));
            }
            from_square += 2;
            if (ep_col < 7 && position.board[from_square] == pawn) {
                move_list->add(Move(from_square, (row + dir) * 8 + ep_col));
            }
        }

        // push promotions
        uint64_t promo_pawns;
        if (position.turn == WHITE) {
            promo_pawns = position.piece_maps[PAWN][WHITE] & RANK_7 & (~all_pieces << 8);
        } else {
            promo_pawns = position.piece_maps[PAWN][BLACK] & RANK_2 & (~all_pieces >> 8);
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
            attacks = position.piece_maps[PAWN][WHITE] >> 8;
            promo_rank = RANK_8;
            right_diag = 7;
            left_diag = 9;
        } else {
            // shift down
            attacks = position.piece_maps[PAWN][BLACK] << 8;
            promo_rank = RANK_1;
            right_diag = -9;
            left_diag = -7;
        }

        // shift left and right
        uint64_t right_captures = ((attacks & 0x7f7f7f7f7f7f7f7f) << 1) & opp_pieces;
        uint64_t left_captures = ((attacks & 0xfefefefefefefefe) >> 1) & opp_pieces;

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

    // if in check, moves must be king evasions or block/capture the checker
    int king_square = lsb(position.piece_maps[KING][position.turn]);
    uint64_t piece_targets = targets;
    if (position.checkers) {
        piece_targets &= get_check_blocks(king_square, lsb(position.checkers));
    }

    // normal piece moves
    uint64_t knights = position.piece_maps[KNIGHT][position.turn];
    while (knights) {
        int from_square = pop_lsb(knights);
        uint64_t moves = get_knight_moves(from_square) & piece_targets;
        write_piece_moves(moves, from_square, move_list);
    }

    uint64_t bishops = position.piece_maps[BISHOP][position.turn] | position.piece_maps[QUEEN][position.turn];
    while (bishops) {
        int from_square = pop_lsb(bishops);
        uint64_t moves = get_bishop_moves(from_square, all_pieces) & piece_targets;
        write_piece_moves(moves, from_square, move_list);
    }

    uint64_t rooks = position.piece_maps[ROOK][position.turn] | position.piece_maps[QUEEN][position.turn];
    while (rooks) {
        int from_square = pop_lsb(rooks);
        uint64_t moves = get_rook_moves(from_square, all_pieces) & piece_targets;
        write_piece_moves(moves, from_square, move_list);
    }

    uint64_t king_moves = get_king_moves(king_square) & targets;
    if (move_type != TACTIC) {
        king_moves |= get_castle_moves(position);
    }
    write_piece_moves(king_moves, king_square, move_list);
}


// move passed in should be pseudo legal
bool is_legal(Position& position, Move move) {
    int king_square = lsb(position.piece_maps[KING][position.turn]);

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
            // update blockers for the case where the king moves backwards in a line of check
            position.piece_maps[ALL_PIECES][position.turn] &= ~square_bitboard(king_square);
            bool legal = !position.is_attacked(to_square);
            position.piece_maps[ALL_PIECES][position.turn] |= square_bitboard(king_square);
            return legal;
        }
    }

    int piece = position.board[from_square];
    int piece_type = Position::get_piece_type(piece);
    int capture = position.board[to_square];

    bool en_passant = false;
    if (capture) {
        position.piece_maps[Position::get_piece_type(capture)][!position.turn] &= ~square_bitboard(to_square);
        position.piece_maps[ALL_PIECES][!position.turn] &= ~square_bitboard(to_square);
    } else if (piece_type == PAWN && from_col != to_col && !position.board[to_square]) {
        en_passant = true;
        int capture_square = from_row * 8 + to_col;
        position.piece_maps[PAWN][!position.turn] &= ~square_bitboard(capture_square);
        position.piece_maps[ALL_PIECES][!position.turn] &= ~square_bitboard(capture_square);
    }

    // piece moved
    position.piece_maps[piece_type][position.turn] &= ~square_bitboard(from_square);
    position.piece_maps[piece_type][position.turn] |= square_bitboard(to_square);
    position.piece_maps[ALL_PIECES][position.turn] &= ~square_bitboard(from_square);
    position.piece_maps[ALL_PIECES][position.turn] |= square_bitboard(to_square);

    bool legal = !position.is_attacked(king_square);
    
    // undo board changes
    if (capture) {
        position.piece_maps[Position::get_piece_type(capture)][!position.turn] |= square_bitboard(to_square);
        position.piece_maps[ALL_PIECES][!position.turn] |= square_bitboard(to_square);
    } else if (en_passant) {
        int capture_square = from_row * 8 + to_col;
        position.piece_maps[PAWN][!position.turn] |= square_bitboard(capture_square);
        position.piece_maps[ALL_PIECES][!position.turn] |= square_bitboard(capture_square);
    }

    position.piece_maps[piece_type][position.turn] |= square_bitboard(from_square);
    position.piece_maps[piece_type][position.turn] &= ~square_bitboard(to_square);
    position.piece_maps[ALL_PIECES][position.turn] |= square_bitboard(from_square);
    position.piece_maps[ALL_PIECES][position.turn] &= ~square_bitboard(to_square);

    return legal;
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


// returns bit map of psuedo legal moves from given square
uint64_t get_piece_moves(Position& position, int square) {
    int turn = position.turn;
    int piece = position.board[square];

    if (!piece || turn != Position::get_color(piece)) {
        return 0ULL;
    }

    uint64_t friendly_pieces = position.piece_maps[ALL_PIECES][turn];
    uint64_t opponent_pieces = position.piece_maps[ALL_PIECES][!turn];
    uint64_t all_pieces = friendly_pieces | opponent_pieces;

    int piece_type = Position::get_piece_type(piece);
    if (piece_type == PAWN) {
        uint64_t captures = get_pawn_attacks(square, turn) & opponent_pieces;

        int direction = (turn == WHITE) ? -1 : 1;
        int row = square / 8;
        int col = square % 8;

        uint64_t pushes = 0ULL;
        int one_forward = (row + direction) * 8 + col;
        int two_forward = (row + 2 * direction) * 8 + col;
        // one square forward
        if (!position.board[one_forward]) {
            pushes |= square_bitboard(one_forward);
        }
        // two squares forward
        if (((direction == -1 && row == 6) || (direction == 1 && row == 1)) &&
            !position.board[one_forward] && !position.board[two_forward]) {
            pushes |= square_bitboard(two_forward);
        }

        uint64_t en_passant = 0ULL;
        if (position.en_passant_col >= 0) {
            int ep_row = turn == WHITE ? 2 : 5;
            int ep_square = ep_row * 8 + position.en_passant_col;
            en_passant = get_pawn_attacks(square, turn) & square_bitboard(ep_square);
        }

        return captures | pushes | en_passant;
    }
    if (piece_type == QUEEN) {
        return get_queen_moves(square, all_pieces) & (~friendly_pieces);
    }
    if (piece_type == ROOK) {
        return get_rook_moves(square, all_pieces) & (~friendly_pieces);
    }
    if (piece_type == BISHOP) {
        return get_bishop_moves(square, all_pieces) & (~friendly_pieces);
    }
    if (piece_type == KNIGHT) {
        return get_knight_moves(square) & (~friendly_pieces);
    }

    return (get_king_moves(square) | get_castle_moves(position)) & (~friendly_pieces);
}


bool is_psuedo_legal(Position& position, Move move) {
    // edge case: verify that the promotion flag is valid
    bool pawn_move = square_bitboard(move.from()) & position.piece_maps[PAWN][position.turn];
    bool pawn_valid = !pawn_move || !(square_bitboard(move.to()) & (RANK_1 | RANK_8)) || move.promote_to();
    bool promote_valid = pawn_move || !move.promote_to();
    return pawn_valid && promote_valid && (get_piece_moves(position, move.from()) & square_bitboard(move.to()));
}


bool no_legal_moves(Position& position) {
    uint64_t pieces = position.piece_maps[ALL_PIECES][position.turn];
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