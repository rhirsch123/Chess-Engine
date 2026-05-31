#include "engine.hh"
#include "polyglot.hh"

Engine::Engine() : transposition_table(Hash) {
    init();
}

static inline int evaluation(Position& position) {
    int val = NNUE::evaluate_incremental(position.half_moves, position.turn);
    val = std::max(val, -MATE_SCORE);
    val = std::min(val, MATE_SCORE);
    return val;
}

static inline bool is_mate_score(int val) {
    return std::abs(val) >= MATE_SCORE - MAX_DEPTH;
}

static inline int get_milli_duration(time_point start) {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
}


int Engine::get_corrhist_adjustment(Position& position) {
    int adj = PAWN_CORRHIST_WEIGHT * pawn_corrhist[position.turn][position.pawn_hash % CORRHIST_SIZE];
    return adj / 256;
}

void Engine::make_move(Position& position, Move move, int ply) {    
    total_nodes++;
    stack[ply].move = move;
    position.make_move(move);
}

// shorter search at the end of negamax to only evaluate quiet positions
int Engine::quiescense(Position& position, int alpha, int beta, int current_depth) {
    bool pv_node = beta > alpha + 1;

    if (position.get_draw()) {
        return 0;
    }

    TerminalState terminal_state = position.get_terminal_state();
    if (terminal_state) {
        if (terminal_state != STALEMATE) {
            return -MATE_SCORE + current_depth;
        }
        return 0;
    }

    Move hash_move;
    TTEntry tt;
    bool transposition_found = transposition_table.get(position.hash_value, tt);

    if (transposition_found && tt.best_move) {
        hash_move = Move(tt.best_move);
        if (!is_psuedo_legal(position, hash_move) || !is_legal(position, hash_move)) {
            transposition_found = false;
            hash_move = Move();
        }
    }

    if (transposition_found && !pv_node && tt.depth != TT_DEPTH_UNSEARCHED
        && position.half_moves - position.last_threefold_reset < 90) {

        if (tt.value >= MATE_SCORE - MAX_DEPTH) {
            tt.value -= current_depth;
        } else if (tt.value <= -MATE_SCORE + MAX_DEPTH) {
            tt.value += current_depth;
        }

        if (tt.type == EXACT || (tt.type == UPPER_BOUND && tt.value <= alpha) || 
           (tt.type == LOWER_BOUND && tt.value >= beta)) {
            return tt.value;
        }
    }

    int raw_eval = transposition_found ? tt.static_eval : evaluation(position);
    int static_eval = raw_eval + get_corrhist_adjustment(position);

    if (!transposition_found) {
        transposition_table.insert(
            position.hash_value,
            0,
            (int16_t) raw_eval,
            0,
            EXACT,
            TT_DEPTH_UNSEARCHED
        );
    }

    alpha = std::max(alpha, static_eval);
    if (alpha >= beta || current_depth >= MAX_DEPTH) {
        return static_eval;
    }

    if (hash_move) {
        // if hash move is tactic
        if (position.board[hash_move.to()] || hash_move.promote_to()) {
            make_move(position, hash_move, current_depth);
            int val = -quiescense(position, -beta, -alpha, current_depth + 1);
            position.pop();

            alpha = std::max(alpha, val);
            if (alpha >= beta) {
                return val;
            }
        } else {
            return static_eval;
        }
    }

    bool in_check = position.checkers;

    MovePicker move_picker(position, *this, current_depth, Move(), true);
    while (true) {
        Move move = move_picker.next_move();
        if (!move) {
            break;
        }

        if (move == hash_move) {
            continue;
        }

        // SEE pruning
        if (!in_check && move_picker.stage == BAD_TACTICS) {
            continue;
        }

        make_move(position, move, current_depth);
        int val = -quiescense(position, -beta, -alpha, current_depth + 1);
        position.pop();

        alpha = std::max(alpha, val);
        if (alpha >= beta) {
            return val;
        }
    }

    return alpha;
}


// negamax search with alpha beta pruning
int Engine::negamax(Position& position, int remaining_depth, int current_depth, int alpha, int beta, Move exclude_move) {
    if (total_nodes % 16 == 0) {
        if ((limits.search_time && get_milli_duration(limits.start_time) >= limits.search_time)
            || (limits.nodes && total_nodes >= limits.nodes)
            || (limits.stop_search && limits.stop_search->load())) {
            
            return INF;
        }
    }

    bool root_node = current_depth == 0;
    bool pv_node = beta > alpha + 1;
    int old_alpha = alpha;

    if (!root_node) {
        if (position.get_draw()) {
            return 0;
        }

        // prune if impossible to return a faster mate
        alpha = std::max(alpha, -MATE_SCORE + current_depth);
        beta = std::min(beta, MATE_SCORE - current_depth - 1);
        if (alpha >= beta) {
            return alpha;
        }
    }

    bool in_check = position.checkers;
    if (current_depth >= MAX_DEPTH || (remaining_depth <= 0 && !in_check)) {
        return quiescense(position, alpha, beta, current_depth);
    }
    remaining_depth = std::max(0, remaining_depth);

    // check for transposition
    Move hash_move;
    TTEntry tt;
    bool transposition_found = !exclude_move && transposition_table.get(position.hash_value, tt);

    if (transposition_found && tt.best_move) {
        hash_move = Move(tt.best_move);
        // careful of hash collisions
        if (!is_psuedo_legal(position, hash_move)) {
            transposition_found = false;
            hash_move = Move();
        }
    }

    if (transposition_found && !pv_node && tt.depth >= remaining_depth
        && position.half_moves - position.last_threefold_reset < 90) {

        if (tt.value >= MATE_SCORE - MAX_DEPTH) {
            tt.value -= current_depth;
        } else if (tt.value <= -MATE_SCORE + MAX_DEPTH) {
            tt.value += current_depth;
        }

        if (tt.type == EXACT || (tt.type == UPPER_BOUND && tt.value <= alpha) || 
           (tt.type == LOWER_BOUND && tt.value >= beta)) {
            return tt.value;
        }
    }

    bool tt_capture = hash_move && position.board[hash_move.to()];

    int raw_eval = transposition_found ? tt.static_eval : evaluation(position);
    int static_eval = raw_eval;
    if (!exclude_move) {
        static_eval += get_corrhist_adjustment(position);
    }

    // cache static eval
    if (!transposition_found) {
        transposition_table.insert(
            position.hash_value,
            0,
            (int16_t) raw_eval,
            0,
            EXACT,
            TT_DEPTH_UNSEARCHED
        );
    }

    stack[current_depth].static_eval = static_eval;
    // heuristic to make certain pruning more/less aggressive
    bool improving = true;
    if (current_depth > 1) {
        improving = stack[current_depth - 2].static_eval < static_eval;
    }

    // reverse futility pruning
    if (!in_check && !pv_node && !exclude_move && remaining_depth <= 8 &&
        beta + RFP_SCALE * (remaining_depth - improving) <= static_eval) {
        return static_eval;
    }

    // null move pruning
    // first try doing nothing and passing the turn. Assumes we are not in zugzwang
    uint64_t non_pawns = position.piece_maps[ALL_PIECES][position.turn] & 
                        ~position.piece_maps[PAWN][position.turn] &
                        ~position.piece_maps[KING][position.turn];
    bool do_null_prune = !pv_node && !in_check && !exclude_move && stack[current_depth - 1].move && remaining_depth >= 3
      && static_eval >= beta && non_pawns && (!transposition_found || tt.type == LOWER_BOUND || tt.value >= beta);
    
    if (do_null_prune) {
        // make null move
        total_nodes++;
        stack[current_depth].move = Move();

        uint64_t hash = position.hash_value;
        int ep_col = position.en_passant_col;
        if (ep_col != -1) {
            position.hash_value ^= Zobrist::en_passant_table[ep_col];
        }
        position.en_passant_col = -1;
        position.turn = !position.turn;
        position.hash_value ^= Zobrist::turn_key;
        position.half_moves++;
        
        NNUE::DirtyPieces dps;
        dps.type = NNUE::NONE;
        NNUE::set_dirty(position.half_moves, dps);

        int R = 4 + remaining_depth / 4;
        int val = -negamax(position, remaining_depth - 1 - R, current_depth + 1, -beta, -beta + 1);

        // unmake
        position.half_moves--;
        position.turn = !position.turn;
        position.hash_value = hash;
        position.en_passant_col = ep_col;
        
        if (val >= beta) {
            if (is_mate_score(val)) {
                return beta;
            }
            return val;
        }
    }

    int local_max = -INF;
    Move local_best_move;
    int move_num = 0;
    int moves_played = 0;

    // move loop 
    MovePicker move_picker(position, *this, current_depth, hash_move);
    while (true) {
        Move move = move_picker.next_move();
        if (!move) {
            break;
        }

        if (move == exclude_move) {
            continue;
        }

        int piece = position.board[move.from()];
        int capture = position.board[move.to()];
        int piece_type = Position::get_piece_type(piece);
        int capture_type = Position::get_piece_type(capture);

        move_num++;
        bool move_found = local_max > -INF;

        if (move_found && !root_node && !in_check) {
            // late move pruning
            if (remaining_depth <= 5 && move_picker.stage <= QUIETS
                && move_num >= lmp_table[remaining_depth][improving]) {
                if (move_picker.stage == QUIETS) {
                    move_picker.index = 0;
                    move_picker.stage = BAD_TACTICS;
                    continue;
                } else {
                    move_picker.only_tactics = true;
                }
            }

            // futility pruning
            // if static eval is well below alpha, quiet moves and bad tactics are unlikely to improve
            // the position enough to matter
            if (remaining_depth <= 6 && move_picker.stage <= QUIETS
                && static_eval + FP_BASE + FP_SCALE * remaining_depth <= alpha) {
                if (move_picker.stage == QUIETS) {
                    move_picker.index = 0;
                    move_picker.stage = BAD_TACTICS;
                    continue;
                } else {
                    move_picker.only_tactics = true;
                }
            }

            if (remaining_depth <= 4 && capture && move_picker.stage == BAD_TACTICS) {
                int hist = capture_history[move.to()][piece_type][capture_type];
                if (static_eval + FP_CAP_BASE + FP_CAP_SCALE * remaining_depth +
                    piece_values[capture_type] + hist / 8 <= alpha) {
                    continue;
                }
            }

            // static exchange evaluation pruning
            if (remaining_depth <= 6 && move_picker.stage == QUIETS) {
                int threshold = SEE_PRUNE_SCALE * remaining_depth;
                if (!position.SEE(move, -threshold)) {
                    continue;
                }
            }
        }

        // singular extension: hash move seems to be significantly better than all others
        int extension = 0;
        if (move == hash_move && !root_node && remaining_depth >= 6 && (tt.type == LOWER_BOUND || tt.type == EXACT)
            && tt.depth >= remaining_depth - 3 && !exclude_move) {
            int singular_depth = remaining_depth / 2;
            int singular_beta = tt.value - remaining_depth;
            int val = negamax(position, singular_depth, current_depth + 1, singular_beta - 1, singular_beta, hash_move);

            if (val < singular_beta) {
                extension = 1;
            }
        }

        // late move reduction: assumes decent move ordering, reduce search depth of late ordered moves
        int R = 0; // can be negative for extensions

        if (!pv_node && moves_played >= 2 && remaining_depth >= 3) {
            R = lmr_table[remaining_depth][moves_played];
            R += tt_capture * LMR_TTCAP;
            
            if (capture) {
                R -= LMR_CAPTURE;
            } else {
                R -= LMR_HISTORY * quiet_history[move.from()][move.to()] / MAX_HISTORY;
            }
        }

        make_move(position, move, current_depth);

        transposition_table.prefetch(position.hash_value);

        if (position.checkers && (move_picker.stage == GOOD_TACTICS)) {
            R -= LMR_GIVES_CHECK;
            extension++;
        }

        R /= 1024;
        R = std::max(R, -1);
        bool reduce_depth = R > 0;

        int val;
        if (!move_found) { // first move
            val = -negamax(position, remaining_depth - 1 + extension, current_depth + 1, -beta, -alpha);
        } else {
            val = -negamax(position, remaining_depth - 1 - R, current_depth + 1, -alpha - 1, -alpha);

            if (val > alpha && (reduce_depth || pv_node)) {
                // research at full depth and full window
                val = -negamax(position, remaining_depth - 1, current_depth + 1, -beta, -alpha);
            }
        }

        position.pop();

        moves_played++;

        // stopped search
        if (std::abs(val) > MATE_SCORE) {
            return val;
        }

        if (val > local_max) {
            local_max = val;
            local_best_move = move;
        }

        int from_square = move.from();
        int to_square = move.to();

        if (local_max >= beta) {
            if (!capture) {
                update_quiet_history(quiet_history, from_square, to_square, remaining_depth, true);
                update_continuation_history(cont_history, current_depth, to_square, piece - 1, remaining_depth, true);
                update_killers(killers, move, current_depth);
            } else {
                update_capture_history(capture_history, to_square, piece_type, capture_type, remaining_depth, true);
            }

            break;
        }

        // decrease history if move did not beat beta
        if (!capture) {
            update_quiet_history(quiet_history, from_square, to_square, remaining_depth, false);
            update_continuation_history(cont_history, current_depth, to_square, piece - 1, remaining_depth, false);
        } else {
            update_capture_history(capture_history, to_square, piece_type, capture_type, remaining_depth, false);
        }

        alpha = std::max(alpha, local_max);
    }

    if (local_max <= -INF) { // no legal moves
        return in_check ? -MATE_SCORE + current_depth : 0;
    }

    TTBound tt_bound;
    if (local_max <= old_alpha) {
        tt_bound = UPPER_BOUND;
    } else if (local_max >= beta) {
        tt_bound = LOWER_BOUND;
    } else {
        tt_bound = EXACT;
    }

    if (!exclude_move) { // not from singular search
        transposition_table.insert(
            position.hash_value,
            (int16_t) local_max,
            (int16_t) raw_eval,
            local_best_move.move,
            tt_bound,
            (int8_t) remaining_depth
        );
    }

    // update correction history
    if (!in_check && !position.board[local_best_move.to()]
        && !(tt_bound == LOWER_BOUND && local_max <= static_eval)
        && !(tt_bound == UPPER_BOUND && local_max >= static_eval)) {
        update_pawn_corrhist(pawn_corrhist, position.turn, position.pawn_hash, remaining_depth, local_max - static_eval);
    }

    if (root_node) {
        best_move = local_best_move;
    }

    return local_max;
}


// iteratively widens search window until value is between alpha and beta
int Engine::aspiration_window(Position& position, int remaining_depth, int estimate) {
    int delta = ASPIRATION_DELTA;

    int alpha = estimate - delta;
    int beta = estimate + delta;

    while (true) {
        int val = negamax(position, remaining_depth, 0, alpha, beta);

        if (std::abs(val) > MATE_SCORE) {
            return val;
        }

        if (val <= alpha) {
            beta = (alpha + beta) / 2;
            alpha = std::max(-INF, alpha - delta);
        } else if (val >= beta) {
            beta = std::min(INF, beta + delta);
        } else {
            return val;
        }

        delta += delta / 2;
    }
}


Move Engine::get_move(Position& position, SearchInfo info) {
    limits = SearchLimits();
    limits.start_time = std::chrono::high_resolution_clock::now();

    // default: 1 second per move
    if (!info.timed_game && !info.fixed_time && !info.fixed_depth && !info.fixed_nodes) {
        info.fixed_time = true;
        info.movetime = 1000;
        info.use_book = true;
        info.verbose = true;
    }

    limits.stop_search = info.stop_search;

    total_nodes = 0;
    count = 0;

    if (info.use_book && position.half_moves < 10) {
        // Flavio Martin's opening book
        Move book_move = Polyglot::get_book_move(position, "titans.bin");
        if (book_move) {
            return book_move;
        }
    }

    int optimal_time = info.time_left / 20 + info.increment / 2;
    if (info.timed_game) {
        // save time
        MoveList root_legals;
        get_legal_moves(position, &root_legals);
        if (root_legals.size == 1) {
            if (info.uci) {
                printf("info depth 0 nodes 0 score cp %d nps 0 time 0\n", evaluation(position));
            }
            return root_legals.moves[0];
        }

        // max time to spend on move
        limits.search_time = info.time_left / 10 + info.increment / 2;
    } else if (info.fixed_time) {
        limits.search_time = info.movetime;
    } else {
        limits.search_time = INF;
    }

    if (info.fixed_nodes) {
        limits.nodes = info.nodes;
    }

    bool move_found = false;

    int eval = 0;
    int depth;

    Move prev_best_move;
    int best_move_stability = 0;

    int search_depth = info.fixed_depth ? info.depth : MAX_DEPTH;
    for (depth = 1; depth <= search_depth; depth++) {
        int val = aspiration_window(position, depth, eval);

        // stopped search
        if (std::abs(val) > MATE_SCORE) {
            break;
        }

        move_found = true;
        eval = val;

        // uci output
        int time_taken = std::max(1, get_milli_duration(limits.start_time));
        if (info.uci) {
            std::string score;
            if (is_mate_score(eval)) {
                int mate_in = eval > 0 ? ((MATE_SCORE - eval + 1) / 2) : ((-MATE_SCORE - eval) / 2);
                score = "mate " + std::to_string(mate_in);
            } else {
                score = "cp " + std::to_string(eval);
            }
            printf("info depth %d nodes %d score %s nps %d time %d pv %s\n",
                depth, total_nodes, score.c_str(), 1000 * (total_nodes / time_taken),
                time_taken, best_move.to_uci().c_str());
            fflush(stdout);
        }

        // time management
        if (best_move == prev_best_move) {
            best_move_stability++;
        } else {
            best_move_stability = 0;
        }
        prev_best_move = best_move;

        // try to spend more time in complex positions
        constexpr float best_move_scale[5] = {1.5, 1.2, 1.0, 0.85, 0.7};
        float time_scale = best_move_scale[std::min(best_move_stability, 4)];

        // save time by not interrupting a search when possible
        if (info.timed_game && time_taken > optimal_time * time_scale) {
            break;
        }
    }

    if (!move_found) {
        MoveList root_legals;
        get_legal_moves(position, &root_legals);
        best_move = root_legals.moves[0];
    }

    int time_taken = std::max(1, get_milli_duration(limits.start_time));
    if (info.verbose) {
        printf("time: %f\n", (float) time_taken / 1000);
        printf("depth: %d\n", depth - 1);

        if (is_mate_score(eval)) {
            if (std::abs(MATE_SCORE - eval) <= 1) printf("evaluation: checkmate\n");
            else if (eval > 0) printf("evaluation: M%d\n", (MATE_SCORE - eval) / 2);
            else printf("evaluation: -M%d\n", std::abs(-MATE_SCORE - eval) / 2);
        }
        else printf("evaluation: %d\n", eval);

        printf("nodes: %d\n", total_nodes);
        printf("nps: %f M\n\n", (float) total_nodes / (time_taken * 1000));
        printf("count: %d\n\n", count);
    }

    return best_move;
}


void Engine::init() {
    std::memset(quiet_history, 0, sizeof(quiet_history));
    std::memset(capture_history, 0, sizeof(capture_history));
    std::memset(cont_history, 0, sizeof(cont_history));
    std::memset(killers, 0, sizeof(killers));
    std::memset(pawn_corrhist, 0, sizeof(pawn_corrhist));

    init_lmp_table();
    init_lmr_table();

    transposition_table.clear();
}