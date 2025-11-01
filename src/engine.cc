#include "engine.hh"
#include "evaluation.hh"
#include "polyglot.hh"

Engine::Engine(int time_per_move, bool use_book)
    : timed_game(false), transposition_table(64),
      time_per_move(time_per_move), use_book(use_book) {
}

Engine::Engine(double minutes, double increment, bool use_book)
    : timed_game(true), transposition_table(64),
      minutes(minutes), increment(increment), use_book(use_book) {
    time_left = minutes * 60 * 1000;
}


static int evaluation(Position& position) {
    return position.eval[position.half_moves];
}

bool Engine::mate_score(int val) {
    return std::abs(val) >= INF - MAX_DEPTH;
}

// shorter search at the end of negamax to only evaluate quiet positions
int Engine::quiescense(Position& position, int alpha, int beta, int current_depth) {
    if (!position.get_draw().empty()) {
        return 0;
    }

    std::string terminal_state = position.get_terminal_state();
    if (!terminal_state.empty()) {
        if (terminal_state != "stalemate") {
            return -INF + current_depth;
        }
        return 0;
    }

    int static_eval = evaluation(position);
    alpha = std::max(alpha, static_eval);

    if (alpha >= beta) {
        return static_eval;
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

        // SEE pruning
        if (!position.in_check() && !Evaluation::SEE(position, move, alpha - static_eval - 50)) {
            continue;
        }

        quiescense_nodes++;
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
    if (negamax_nodes % 16 == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >= time_per_move) {
            return INF + 1;
        }
    }

    bool root_node = current_depth == 0;
    bool pv_node = beta > alpha + 1;
    int old_alpha = alpha;

    if (!root_node) {
        if (!position.get_draw().empty()) {
            return 0;
        }

        // prune if impossible to return a faster mate
        alpha = std::max(alpha, -INF + current_depth);
        beta = std::min(beta, INF - current_depth - 1);
        if (alpha >= beta) {
            return alpha;
        }
    }

    bool in_check = position.in_check();
    if (current_depth >= MAX_DEPTH || (remaining_depth <= 0 && !in_check)) {
        return quiescense(position, alpha, beta, current_depth);
    }
    remaining_depth = std::max(0, remaining_depth);

    Move hash_move;
    tt_entry entry;
    bool transposition_found = transposition_table.get(position.hash_value, entry);
    if (transposition_found) {
        if (!pv_node && entry.depth >= remaining_depth && position.repetitions == 1 &&
            position.half_moves - position.last_threefold_reset < 90 &&
            num_set_bits(position.white_pieces | position.black_pieces) > 3) {

            int val = entry.value;

            if (val >= INF - MAX_DEPTH) {
                val -= current_depth;
            } else if (val <= -INF + MAX_DEPTH) {
                val += current_depth;
            }

            if (entry.type == EXACT) {
                return val;
            } else if (entry.type == UPPER_BOUND) {
                if (val <= alpha) {
                    return val;
                }
                beta = std::min(beta, val);
            } else if (entry.type == LOWER_BOUND) {
                if (val >= beta) {
                    return val;
                }
                alpha = std::max(alpha, val);
            }
        }

        hash_move = Move(entry.best_move);
    }

    int static_eval = evaluation(position);

    // heuristic to make certain pruning more/less aggressive
    bool improving = true;
    if (position.half_moves > 1) {
        improving = position.eval[position.half_moves - 2] < static_eval;
    }

    // reverse futility pruning
    if (!in_check && !pv_node && remaining_depth <= RFP_DEPTH &&
        beta + RFP_SCALE * remaining_depth <= static_eval) {
        return static_eval;
    }

    // null move pruning
    // first try doing nothing and passing the turn. zugzwang is unlikely in middle games
    int major_material = (position.turn == WHITE ? position.white_material : position.black_material) - 
        num_set_bits(position.piece_maps[PAWN][position.turn]) * Position::values[PAWN];

    bool do_null_prune = !pv_node && !in_check && remaining_depth >= NULL_PRUNE_DEPTH &&
        static_eval >= beta && major_material > Position::values[ROOK] && position.hash_value &&
        (!transposition_found || entry.type == LOWER_BOUND || entry.value >= beta);
    
    if (do_null_prune) {
        // make null move
        uint64_t hash = position.hash_value;
        int en_passant_col = position.en_passant_col;
        position.turn = !position.turn;
        position.hash_value = 0ULL;
        position.en_passant_col = -1;
        std::memcpy(position.acc_white_stack[position.half_moves], NNUE::accumulators[WHITE], HIDDEN_SIZE * sizeof(int16_t));
        std::memcpy(position.acc_black_stack[position.half_moves], NNUE::accumulators[BLACK], HIDDEN_SIZE * sizeof(int16_t));
        position.half_moves++;
        int output_bucket = NNUE::get_output_bucket(position.white_pieces | position.black_pieces);
        position.eval[position.half_moves] = NNUE::evaluate_incremental(position.turn, output_bucket);
        
        int R = 4 + (float) remaining_depth / 4;
        int val = -negamax(position, remaining_depth - 1 - R, current_depth + 1, -beta, -beta + 1, start_time);

        // unmake
        position.half_moves--;
        position.turn = !position.turn;
        position.hash_value = hash;
        position.en_passant_col = en_passant_col;
        
        if (val >= beta) {
            if (mate_score(val)) {
                return beta;
            }
            return val;
        }
    }

    int local_max = -INF - 1;

    // can check hash move before generating other moves
    if (transposition_found) {
        int extension = 0;

        negamax_nodes++;
        position.make_move(hash_move);

        extension += position.in_check();
        int val = -negamax(position, remaining_depth - 1 + extension, current_depth + 1, -beta, -alpha, start_time);
        
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
            int capture = position.board[hash_move.end_row()][hash_move.end_col()];
            if (!capture) {
                update_quiet_history(from_square, to_square, remaining_depth, true);
                update_killers(hash_move, current_depth);
            } else {
                int piece_type = Position::get_piece_type(position.board[hash_move.start_row()][hash_move.start_col()]);
                int capture_type = Position::get_piece_type(capture);
                update_capture_history(to_square, piece_type, capture_type, remaining_depth, true);
            }

            if (root_node) {
                best_move = hash_move;
            }

            if (position.hash_value) {
                transposition_table.insert({ position.hash_value, local_max, hash_move.move, LOWER_BOUND, (uint8_t) remaining_depth });
            }

            return local_max;
        }

        alpha = std::max(alpha, local_max);
    }

    MoveGenerator movegen(position, *this, current_depth);

    Move local_best_move = hash_move;
    int num_moves = transposition_found;

    while (true) {
        Move move = movegen.next_move();
        if (!move) {
            break;
        }

        // already tried hash move
        if (move == hash_move) {
            continue;
        }
        num_moves++;

        // late move pruning
        if (!root_node && remaining_depth <= LMP_DEPTH && movegen.stage == QUIETS) {
            int move_bound;
            if (improving) {
                move_bound = LMP_IMPROVING_BASE + LMP_IMPROVING_SCALE * remaining_depth * remaining_depth;
            } else {
                move_bound = LMP_BASE + LMP_SCALE * remaining_depth * remaining_depth;
            }

            if (num_moves >= move_bound) {
                movegen.stage = BAD_TACTICS;
                continue;
            }
        }

        // futility pruning: if static eval is too far below alpha, quiet moves and bad captures are
        // unlikely to improve position enough to matter
        if (!in_check && remaining_depth <= FUTILITY_PRUNE_DEPTH && movegen.stage == QUIETS && local_max >= -INF &&
            static_eval + FUTILITY_PRUNE_BASE + FUTILITY_PRUNE_SCALE * remaining_depth <= alpha) {
            movegen.stage = BAD_TACTICS;
            continue;
        }

        if (!in_check && remaining_depth <= 4 && movegen.stage == BAD_TACTICS && local_max >= -INF) {
            int piece_type = Position::get_piece_type(position.board[move.start_row()][move.start_col()]);
            int capture = position.board[move.end_row()][move.end_col()];
            int capture_type = capture ? Position::get_piece_type(capture) : PAWN;
            int hist = capture_history[move.to()][piece_type][capture_type];
            if (static_eval + FP_CAP_BASE + FP_CAP_SCALE * remaining_depth + Position::values[capture_type] + hist / FP_CAP_HIST <= alpha) {
                continue;
            }
        }
        
        // static exchange evaluation pruning
        if (remaining_depth <= 6 && movegen.stage == QUIETS) {
            int threshold = SEE_PRUNE_SCALE * remaining_depth;
            if (!Evaluation::SEE(position, move, threshold)) {
                continue;
            }
        }

        // late move reduction: assumes decent move ordering, reduce search depth of late ordered moves
        int R = 0; // can be negative for extensions
        if (!pv_node && num_moves >= 3 && remaining_depth >= 3) {
            if (num_moves <= 6 || remaining_depth <= 5 || in_check || move.exchange > 0) {
                // reduce less
                R = 1;
            } else if ((num_moves >= 10 && remaining_depth >= 8) || num_moves >= 15) {
                R = 3;
            } else {
                R = 2;
            }

            R += (!move.exchange && transposition_found && position.board[hash_move.end_row()][hash_move.end_col()]);
        }

        if (!move.exchange) {
            R -= quiet_history[move.from()][move.to()] / HISTORY_DIVISOR;

            if (num_moves < 3 || remaining_depth < 2) {
                R = std::min(0, R);
            }
        }

        if (position.half_moves > 1) {
            // consecutive capture
            Move last_move = position.move_stack[position.half_moves - 1].move;
            R -= (position.board[last_move.end_row()][last_move.end_col()] != 0) && move.exchange;
        }

        int val;
        negamax_nodes++;
        position.make_move(move);

        if (position.in_check() && (movegen.stage == GOOD_TACTICS)) {
            // extend if good move gives check
            R--;
        }

        bool reduce_depth = R > 0;
        R = std::max(R, -1);

        if (local_max < -INF) { // first move
            val = -negamax(position, remaining_depth - 1, current_depth + 1, -beta, -alpha, start_time);
        } else {
            val = -negamax(position, remaining_depth - 1 - R, current_depth + 1, -alpha - 1, -alpha, start_time);

            if (val > alpha && (reduce_depth || pv_node)) {
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

    if (position.hash_value) { // not from null move
        transposition_table.insert({ position.hash_value, local_max, local_best_move.move, tt_bound, (uint8_t) remaining_depth });
    }

    if (root_node) {
        best_move = local_best_move;
    }

    return local_max;
}


// iteratively widens search window until value is between alpha and beta
int Engine::aspiration_window(Position& position, int remaining_depth, int estimate,
                              std::chrono::time_point<std::chrono::high_resolution_clock> start_time) {
    int delta = ASPIRATION_DELTA - remaining_depth / 5;
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

    count = 0;
    negamax_nodes = 0;
    quiescense_nodes = 0;

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
        // save time
        auto root_legals = position.get_legal_moves();
        if (root_legals.size() == 1) {
            return root_legals[0];
        }

        time_per_move = time_left / 10;
    }

    bool move_found = false;

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

        if (timed_game) {
            auto now = std::chrono::high_resolution_clock::now();
            int time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            if (time_taken > time_left / 20 + 500 * increment) {
                break;
            }
        }
        
        move_found = true;
        eval = val;
    }

    if (!move_found) {
        best_move = position.get_legal_moves()[0];
    }

    auto now = std::chrono::high_resolution_clock::now();
    int time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    if (verbose) {
        printf("time: %f\n", (float) time_taken / 1000);
        printf("depth: %d\n", depth - 1);
        printf("evaluation: %d\n", position.turn == WHITE ? eval : -eval);
        printf("negamax nodes: %d\n", negamax_nodes);
        printf("quiescense nodes: %d\n", quiescense_nodes);
        printf("total nps: %f M\n\n", float (negamax_nodes + quiescense_nodes) / (time_taken * 1000));
    }

    if (timed_game) {
        time_left -= time_taken;
        if (time_left <= 0) {
            printf("lost on time\n");
        }
        time_left += increment * 1000;
    }

    score = eval;
 
    return best_move;
}


void Engine::reset() {
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            quiet_history[i][j] = 0;
        }
    }
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < 5; k++) {
                capture_history[i][j][k] = 0;
            }
        }
    }
    for (int i = 0; i < MAX_DEPTH; i++) {
        killers[i][0] = Move();
        killers[i][1] = Move();
    }
    transposition_table.clear();
}