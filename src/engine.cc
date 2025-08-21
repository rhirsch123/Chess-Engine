#include "engine.hh"
#include "evaluation.hh"
#include "polyglot.hh"

Engine::Engine(int time_per_move, bool use_book)
    : timed_game(false), transposition_table(128),
      time_per_move(time_per_move), use_book(use_book) {
}

Engine::Engine(double minutes, double increment, bool use_book)
    : timed_game(true), transposition_table(128),
      minutes(minutes), increment(increment), use_book(use_book) {
    time_left = minutes * 60 * 1000;
}


// shorter search at the end of negamax to only evaluate quiet positions
int Engine::quiescense(Position& position, int alpha, int beta, int current_depth) {
    std::string terminal_state = position.get_terminal_state();
    if (!terminal_state.empty()) {
        if (terminal_state == "white win") {
            return (INF - current_depth) * (!position.turn - position.turn);
        } else if (terminal_state == "black win") {
            return (INF - current_depth) * (position.turn - !position.turn);
        } else {
            return 0;
        }
    }

    int static_eval = Evaluation::evaluation(position, *this);
    alpha = std::max(alpha, static_eval);

    if (alpha >= beta) {
        return alpha;
    }

    auto pseudo_legal_tactics = position.get_pseudo_legal_moves(TACTIC);
    std::sort(pseudo_legal_tactics.begin(), pseudo_legal_tactics.end(), [&](const Move& a, const Move& b) {
        // mvv-lva
        return a.exchange > b.exchange;
    });

    for (Move move : pseudo_legal_tactics) {
        if (!position.is_legal(move)) {
            continue;
        }

        // delta pruning: if even capturing the piece for free could not raise alpha, skip the move
        int delta_margin = 200;
        int capture = position.board[move.end_row()][move.end_col()];
        int move_value = capture ? Position::values[capture] : 0;
        if (move.promote_to()) {
            move_value += Position::values[move.promote_to()] - Position::values[PAWN];
        }
        if (static_eval + move_value + delta_margin < alpha) {
            continue;
        }

        // SEE pruning
        if (!Evaluation::safe_move(position, move)) {
            continue;
        }

        position.make_move(move);
        int val = -quiescense(position, -beta, -alpha, current_depth + 1);
        position.pop();

        if (val >= beta) {
            return alpha;
        }

        alpha = std::max(alpha, val);
    }

    return alpha;
}


// negamax search with alpha beta pruning
int Engine::negamax(Position& position, int remaining_depth, int current_depth, int alpha, int beta,
                    std::chrono::time_point<std::chrono::high_resolution_clock> start_time) {
    if (move_found && total_nodes % 16 == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >= time_per_move) {
            return INF + 1;
        }
    }

    total_nodes++;

    bool root_node = current_depth == 0;
    bool pv_node = beta > alpha + 1;

    if (!position.get_draw().empty()) {
        return 0;
    }

    bool in_check = position.in_check();
    if (remaining_depth <= 0 && !in_check) {
        return quiescense(position, alpha, beta, current_depth);
    }
    remaining_depth = std::max(0, remaining_depth);

    int old_alpha = alpha;

    Move hash_move;
    tt_entry entry;
    bool transposition_found = transposition_table.get(position.hash_value, entry);
    if (transposition_found) {
        if (entry.depth >= remaining_depth && position.repetitions == 1 &&
            position.half_moves - position.last_threefold_reset < 90 &&
            num_set_bits(position.white_pieces | position.black_pieces) > 3) {

            int val = entry.value;

            if (entry.type == EXACT) {
                if (root_node) {
                    best_move = Move(entry.best_move);
                }
                return val;
            } else if (entry.type == UPPER_BOUND) {
                if (val <= alpha) {
                    if (root_node) {
                        best_move = Move(entry.best_move);
                    }
                    return val;
                }
                beta = std::min(beta, val);
            } else if (entry.type == LOWER_BOUND) {
                if (val >= beta) {
                    if (root_node) {
                        best_move = Move(entry.best_move);
                    }
                    return val;
                }
                alpha = std::max(alpha, val);
            }
        }

        hash_move = Move(entry.best_move);
    }

    // null move pruning
    // first try doing nothing and passing the turn. zugzwang is unlikely in middle games
    int major_material = (position.turn == WHITE ? position.white_material : position.black_material) - 
        num_set_bits(position.piece_maps[PAWN][position.turn]) * Position::values[PAWN];
    bool do_null_prune = !pv_node && !in_check && remaining_depth >= NULL_PRUNE_DEPTH &&
        major_material > Position::values[ROOK] && position.hash_value &&
        (!transposition_found || entry.type == LOWER_BOUND || entry.value >= beta);
    
    int static_eval = INF + 1;
    if (do_null_prune) {
        static_eval = Evaluation::evaluation(position, *this);
    }
    if (do_null_prune && static_eval >= beta) {
        uint64_t hash = position.hash_value;
        int en_passant_col = position.en_passant_col;
        position.turn = !position.turn;
        position.hash_value = 0ULL;
        position.en_passant_col = -1;

        int R = 3 + remaining_depth / 4;
        int val = -negamax(position, remaining_depth - 1 - R, current_depth + 1, -beta, -beta + 1, start_time);

        position.turn = !position.turn;
        position.hash_value = hash;
        position.en_passant_col = en_passant_col;
        
        if (val >= beta) {
            if (std::abs(val) >= INF - MAX_DEPTH) {
                return beta;
            }
            return val;
        }
    }

    int local_max = -INF - 1;

    // can check hash move before generating other moves
    if (transposition_found) {
        position.make_move(hash_move);
        int val = -negamax(position, remaining_depth - 1, current_depth + 1, -beta, -alpha, start_time);
        position.pop();

        // time cutoff
        if (std::abs(val) > INF) {
            return val;
        }

        if (val > local_max) {
            local_max = val;
        }

        int from_square = hash_move.from();
        int to_square = hash_move.to();

        if (local_max >= beta) {
            if (!hash_move.exchange) {
                update_quiet_history(from_square, to_square, remaining_depth, true);
                update_killers(hash_move, current_depth);
            } else {
                int piece_type = Position::get_piece_type(position.board[hash_move.start_row()][hash_move.start_col()]);
                int capture_type = Position::get_piece_type(position.board[hash_move.end_row()][hash_move.end_col()]);
                update_capture_history(to_square, piece_type, capture_type, remaining_depth, true);
            }

            if (root_node) {
                best_move = hash_move;
            }

            return local_max;
        }

        alpha = std::max(alpha, local_max);
    }

    MoveGenerator movegen(position, *this, current_depth);

    Move local_best_move = hash_move;
    int num_moves = transposition_found;

    // futility pruning: if static eval is too far below alpha, non-tactical moves are
    // unlikely to improve position enough to matter, so skip them
    bool futility_prune = !pv_node && remaining_depth <= FUTILITY_PRUNE_DEPTH && !in_check;

    while (true) {
        Move move = movegen.next_move();
        if (!move) {
            break;
        }

        if (futility_prune && movegen.stage == QUIETS && local_max >= -INF) {
            if (static_eval > INF) {
                static_eval = Evaluation::evaluation(position, *this);
            }

            if (static_eval + 100 + 150 * remaining_depth <= alpha) {
                if (remaining_depth <= 3) {
                    break;
                }
                movegen.stage = BAD_TACTICS;
                continue;
            }
        }

        if (move == hash_move || !position.is_legal(move)) {
            continue;
        }
        num_moves++;

        // late move reduction: assumes decent move ordering, reduce search depth of late ordered moves
        int R = 0;
        bool reduce_depth = !root_node && num_moves >= 3 && remaining_depth >= 3;

        if (reduce_depth) {
            if (num_moves <= 6 || remaining_depth <= 5 || in_check || pv_node ||
                move == killers[current_depth][0] || move == killers[current_depth][1]) {
                // reduce less
                R = 1;
            } else {
                R = 2;
            }

            R += (transposition_found && position.board[hash_move.end_row()][hash_move.end_col()]);

            if (!move.exchange) {
                R -= quiet_history[move.from()][move.to()] / 8192;
            }

            reduce_depth = R > 0;
            R = std::max(0, R);
        }

        int val;
        position.make_move(move);

        if (reduce_depth && position.in_check()) {
            // reduce less if move gives check
            R--;
            reduce_depth = R > 0;
        }

        if (local_max < -INF) { // first move
            val = -negamax(position, remaining_depth - 1, current_depth + 1, -beta, -alpha, start_time);
        } else {
            val = -negamax(position, remaining_depth - 1 - R, current_depth + 1, -alpha - 1, -alpha, start_time);
            if ((reduce_depth && val >= beta) || (pv_node && val > alpha)) {
                // research at full depth and full window
                val = -negamax(position, remaining_depth - 1, current_depth + 1, -beta, -alpha, start_time);
            }
        }

        position.pop();

        // time cutoff
        if (std::abs(val) > INF) {
            return val;
        }

        if (val > local_max) {
            local_max = val;
            local_best_move = move;
        }

        int from_square = move.from();
        int to_square = move.to();

        if (local_max >= beta) {
            if (!move.exchange) {
                update_quiet_history(from_square, to_square, remaining_depth, true);
                update_killers(move, current_depth);
            } else {
                int piece_type = Position::get_piece_type(position.board[move.start_row()][move.start_col()]);
                int capture_type = Position::get_piece_type(position.board[move.end_row()][move.end_col()]);
                update_capture_history(to_square, piece_type, capture_type, remaining_depth, true);
            }

            break;
        }

        // decrease history if move did not beat beta
        if (!move.exchange) {
            update_quiet_history(from_square, to_square, remaining_depth, false);
        } else {
            int piece_type = Position::get_piece_type(position.board[move.start_row()][move.start_col()]);
            int capture_type = Position::get_piece_type(position.board[move.end_row()][move.end_col()]);
            update_capture_history(to_square, piece_type, capture_type, remaining_depth, false);
        }

        alpha = std::max(alpha, local_max);
    }

    if (local_max < -INF) { // no legal moves
        return in_check ? -INF + current_depth : 0;
    }

    bound_type tt_bound;
    if (local_max <= old_alpha) {
        tt_bound = UPPER_BOUND;
    } else if (local_max >= beta) {
        tt_bound = LOWER_BOUND;
    } else {
        tt_bound = EXACT;
    }

    if (std::abs(local_max) < INF - MAX_DEPTH && local_max && position.hash_value) { // not terminal or from null move
        transposition_table.insert({ position.hash_value, local_max, local_best_move.move, tt_bound, (uint8_t) remaining_depth });
    }

    if (root_node) {
        best_move = local_best_move;
    }

    return local_max;
}


// iteratively widens search window until value is between alpha and beta
int Engine::aspiration_window(Position& position, int remaining_depth, int estimate, std::chrono::time_point<std::chrono::high_resolution_clock> start_time) {
    int delta = ASPIRATION_DELTA;
    int max_retries = 4;

    int alpha = estimate - delta;
    int beta = estimate + delta;

    for (int i = 0; i < max_retries; i++) {
        int val = negamax(position, remaining_depth, 0, alpha, beta, start_time);

        if (std::abs(val) > INF) {
            return val;
        }

        if (val <= alpha) {
            beta = (alpha + beta) / 2;
            alpha = std::max(-INF - 1, alpha - delta);
        } else if (val >= beta) {
            beta = std::min(INF + 1, beta + delta);
        } else {
            return val;
        }

        delta += delta / 2;
    }
    return negamax(position, remaining_depth, 0, -INF - 1, INF + 1, start_time);
}


Move Engine::get_move(Position& position, bool verbose) {
    auto start_time = std::chrono::high_resolution_clock::now();

    total_nodes = 0;
    evaluated_positions = 0;

    if (use_book && position.half_moves < 10) {
        // Flavio Martin's opening book
        Move book_move = Polyglot::get_book_move(position, "titans.bin");
        if (book_move) {
            if (timed_game) {
                time_left += increment * 1000;
            }
            return book_move;
        } else {
            use_book = false;
        }
    }

    if (timed_game) {
        auto root_legals = position.get_legal_moves();
        if (root_legals.size() == 1) {
            return root_legals[0];
        }

        time_per_move = time_left / 20 + 500 * increment;
    }

    move_found = false;

    int eval;
    int depth;

    for (depth = 1; depth <= MAX_DEPTH; depth++) {
        int val;
        if (depth <= 4) {
            val = negamax(position, depth, 0, -INF - 1, INF + 1, start_time);
        } else {
            val = aspiration_window(position, depth, eval, start_time);
        }

        // time cutoff
        if (std::abs(val) > INF) {
            break;
        }
        
        move_found = true;
        eval = val;
    }

    auto now = std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    if (verbose) {
        printf("time: %f\n", time_taken / 1000);
        printf("depth: %d\n", depth - 1);
        printf("evaluation: %d\n", position.turn == WHITE ? eval : -eval);
        printf("total nodes: %d\n", total_nodes);
        printf("evaluated positions: %d\n\n", evaluated_positions);
    }

    if (timed_game) {
        time_left -= time_taken;
        if (time_left <= 0) {
            printf("lost on time\n");
        }
        time_left += increment * 1000;
    }

    return best_move;
}
