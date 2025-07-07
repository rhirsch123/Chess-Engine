#include "engine.hh"
#include "evaluation.hh"
#include "polyglot.hh"

Engine::Engine(int time_limit, bool use_book)
	: replace_tt(64, true), depth_tt(64, false),
	  time_limit(time_limit), use_book(use_book) {
}


// get_move helper function. minimax algorithm with alpha beta pruning
int Engine::value(Position& position, int current_depth, int max_depth, int alpha, int beta,
				  std::chrono::time_point<std::chrono::high_resolution_clock> start_time) {
	if (time_limit > 0) {
		auto now = std::chrono::high_resolution_clock::now();
		if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >= time_limit) {
			return INF + 1;
		}
	}

	total_nodes++;

	if (!position.get_draw().empty()) {
		return 0;
	}

	if (current_depth >= max_depth || current_depth >= MAX_DEPTH) {
		std::string terminal_state = position.get_terminal_state();
		if (!terminal_state.empty()) {
			if ((terminal_state == "white win" && playing == "white") ||
				(terminal_state == "black win" && playing == "black")) {
				return INF - current_depth;
			}
			else if ((terminal_state == "white win" && playing == "black") ||
						(terminal_state == "black win" && playing == "white")) {
				return -INF + current_depth;
			} else {
				return 0;
			}
		}

		evaluated_positions++;
		return Evaluation::evaluation(position, *this);
	}

	uint8_t remaining_depth = max_depth - current_depth;
	Move hash_move;
	tt_entry entry;
	bool transposition_found = depth_tt.get(position.hash_value, entry);
	if (!transposition_found) {
		transposition_found = replace_tt.get(position.hash_value, entry);
	}
	if (transposition_found) {
		if (entry.depth >= remaining_depth && position.repetitions == 1 &&
			position.half_moves - position.last_threefold_reset < 90) {

			int val = playing == "white" ? entry.value : -entry.value;
			if (position.white_material + position.black_material < (Position::values[ROOK] * 2)) {
				val = val > 0 ? val - current_depth : val + current_depth;
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

	bool max_node = (position.turn == WHITE && playing == "white") || (position.turn == BLACK && playing == "black");

	int max_val = std::numeric_limits<int>::min();
	int min_val = std::numeric_limits<int>::max();

	// can check hash move before generating other moves
	if (transposition_found) {
		if (max_node) {
			bool tactic = hash_move.promote_to() || position.board[hash_move.end_row()][hash_move.end_col()];
			bool safe = Evaluation::safe_move(position, hash_move);
			if (tactic && current_depth + 1 == max_depth && max_depth < 10 && safe) {
				max_depth++;
			}

			position.make_move(hash_move);
			int val = value(position, current_depth + 1, max_depth, alpha, beta, start_time);
			position.pop();

			if (current_depth + 1 == max_depth && !safe) {
				int material_lost;
				if (hash_move.promote_to()) {
					material_lost = hash_move.promote_to();
				} else {
					material_lost = Position::get_piece_type(position.board[hash_move.start_row()][hash_move.start_col()]);
				}
				val -= MATERIAL_WEIGHT * Position::values[material_lost];
			}

			int from_square = hash_move.from();
			int to_square = hash_move.to();

			if (val > max_val) {
				max_val = val;
				history[from_square][to_square] += remaining_depth;
			}

			if (max_val >= beta) {
				history[from_square][to_square] += (remaining_depth * remaining_depth);

				return max_val;
			}

			alpha = std::max(alpha, max_val);
		} else {
			bool tactic = hash_move.promote_to() || position.board[hash_move.end_row()][hash_move.end_col()];
			bool safe = Evaluation::safe_move(position, hash_move);
			if (tactic && current_depth + 1 == max_depth && max_depth < 10 && safe) {
				max_depth++;
			}

			position.make_move(hash_move);
			int val = value(position, current_depth + 1, max_depth, alpha, beta, start_time);
			position.pop();

			if (current_depth + 1 == max_depth && !safe) {
				int material_lost;
				if (hash_move.promote_to()) {
					material_lost = hash_move.promote_to();
				} else {
					material_lost = Position::get_piece_type(position.board[hash_move.start_row()][hash_move.start_col()]);
				}
				val += (MATERIAL_WEIGHT * Position::values[material_lost]) / 2;
			}

			int from_square = hash_move.from();
			int to_square = hash_move.to();

			if (val < min_val) {
				min_val = val;
				history[from_square][to_square] += remaining_depth;
			}

			if (min_val <= alpha) {
				history[from_square][to_square] += (remaining_depth * remaining_depth);

				return min_val;
			}

			beta = std::min(beta, min_val);
		}
	}

	bool in_check = position.in_check();

	// heuristic: null move pruning
	// first try doing nothing and passing the turn. zugzwang is unlikely in middle games.
	int major_material = (position.turn == WHITE ? position.white_material : position.black_material) - 
		num_set_bits(position.piece_maps[PAWN][position.turn]) * Position::values[PAWN];
	if (!in_check && remaining_depth >= 2 && major_material > (Position::values[ROOK] * 2) && position.hash_value) {
		uint64_t hash = position.hash_value;
		int en_passant_col = position.en_passant_col;
		position.turn = !position.turn;
		position.hash_value = 0ULL;
		position.en_passant_col = -1;
		int next_depth = remaining_depth > 3 ? max_depth - 2 : max_depth - 1;
		int val;
		if (max_node) {
			val = value(position, current_depth + 1, next_depth, alpha, alpha + 1, start_time);
		} else {
			val = value(position, current_depth + 1, next_depth, beta - 1, beta, start_time);
		}
		position.turn = !position.turn;
		position.hash_value = hash;
		position.en_passant_col = en_passant_col;
		
		if ((max_node && val >= beta) || (!max_node && val <= alpha)) {
			return val;
		}
	}

	auto pseudo_legal_moves = position.get_pseudo_legal_moves();

	// sort on heuristics
	std::sort(pseudo_legal_moves.begin(), pseudo_legal_moves.end(), [&](const Move& a, const Move& b) {
		int a_history = history[a.from()][a.to()];
		int b_history = history[b.from()][b.to()];

		// mvv-lva
		if (a.priority > 0 || b.priority > 0) {
			return a.priority > b.priority;
		}

		return a_history > b_history;
	});

	bool no_legal_moves = !transposition_found;
	Move best_move;
	if (transposition_found) {
		best_move = hash_move;
	}
	int last_improvement = 0;
	int move_num = 0;

	if (max_node) {
		for (const auto& move : pseudo_legal_moves) {
			if (move == hash_move || !position.is_legal(move)) {
				continue;
			}
			no_legal_moves = false;
			move_num++;

			bool tactic = move.promote_to() || position.board[move.end_row()][move.end_col()];

			// late move reduction: reduce search depth of late ordered moves
			// possible to miss shallower tactics if too aggressive
			bool reduce_depth = move_num >= 3 && remaining_depth >= 3;
			int next_depth = reduce_depth ? max_depth - 2 : max_depth;
			if (reduce_depth && (move_num <= 6 || remaining_depth <= 5 || last_improvement < 2 || in_check || tactic)) {
				// reduce less
				next_depth++;
			}

			position.make_move(move);
			if (position.in_check()) {
				reduce_depth = false;
				next_depth = max_depth;
			}
			int val = value(position, current_depth + 1, next_depth, alpha, beta, start_time);
			position.pop();

			if (reduce_depth && val >= beta) {
				next_depth = max_depth;
				position.make_move(move);
				val = value(position, current_depth + 1, next_depth, alpha, beta, start_time);
				position.pop();
			}

			// time cutoff
			if (val > INF) {
				return val;
			}

			// don't want it to think it's winning because it can't see opponent capture back
			if (current_depth + 1 == next_depth && !Evaluation::safe_move(position, move)) {
				int material_lost;
				if (move.promote_to()) {
					material_lost = move.promote_to();
				} else {
					material_lost = Position::get_piece_type(position.board[move.start_row()][move.start_col()]);
				}
				val -= MATERIAL_WEIGHT * Position::values[material_lost];
			}

			int from_square = move.from();
			int to_square = move.to();

			if (val > max_val) {
				max_val = val;
				best_move = move;
				history[from_square][to_square] += remaining_depth;

				last_improvement = 0;
			} else {
				last_improvement++;
			}

			if (max_val >= beta) {
				if (std::abs(max_val) < INF - MAX_DEPTH && position.hash_value && current_depth > 1 && max_val) { // not terminal or from null move
					int white_val = playing == "white" ? max_val : -max_val;
					depth_tt.insert({ position.hash_value, white_val, best_move.move, LOWER_BOUND, remaining_depth });
					replace_tt.insert({ position.hash_value, white_val, best_move.move, LOWER_BOUND, remaining_depth });
				}

				history[from_square][to_square] += (remaining_depth * remaining_depth);

				return max_val;
			}

			alpha = std::max(alpha, max_val);
		}

		if (no_legal_moves) {
			std::string terminal_state = position.get_terminal_state(0);
			if ((terminal_state == "white win" && playing == "white") ||
				(terminal_state == "black win" && playing == "black")) {
				return INF - current_depth;
			}
			else if ((terminal_state == "white win" && playing == "black") ||
						(terminal_state == "black win" && playing == "white")) {
				return -INF + current_depth;
			} else {
				return 0;
			}
		}
		
		if (std::abs(max_val) < INF - MAX_DEPTH && position.hash_value && current_depth > 1 && max_val) {
			int white_val = playing == "white" ? max_val : -max_val;
			depth_tt.insert({ position.hash_value, white_val, best_move.move, EXACT, remaining_depth });
			replace_tt.insert({ position.hash_value, white_val, best_move.move, EXACT, remaining_depth });
		}

		return max_val;
	} else {
		for (const auto& move : pseudo_legal_moves) {
			if (move == hash_move || !position.is_legal(move)) {
				continue;
			}
			no_legal_moves = false;
			move_num++;

			bool tactic = move.promote_to() || position.board[move.end_row()][move.end_col()];

			bool reduce_depth = move_num >= 3 && remaining_depth >= 3;
			int next_depth = reduce_depth ? max_depth - 2 : max_depth;
			if (reduce_depth && (move_num <= 6 || remaining_depth <= 5 || last_improvement < 2 || in_check || tactic)) {
				next_depth++;
			}

			position.make_move(move);
			if (position.in_check()) {
				reduce_depth = false;
				next_depth = max_depth;
			}
			int val = value(position, current_depth + 1, next_depth, alpha, beta, start_time);
			position.pop();

			if (reduce_depth && val <= alpha) {
				next_depth = max_depth;
				position.make_move(move);
				val = value(position, current_depth + 1, next_depth, alpha, beta, start_time);
				position.pop();
			}

			// time cutoff
			if (val > INF) {
				return val;
			}

			if (current_depth + 1 == next_depth && !Evaluation::safe_move(position, move)) {
				int material_lost;
				if (move.promote_to()) {
					material_lost = move.promote_to();
				} else {
					material_lost = Position::get_piece_type(position.board[move.start_row()][move.start_col()]);
				}
				val += (MATERIAL_WEIGHT * Position::values[material_lost]) / 2;
			}

			int from_square = move.from();
			int to_square = move.to();

			if (val < min_val) {
				min_val = val;
				best_move = move;
				history[from_square][to_square] += remaining_depth;

				last_improvement = 0;
			} else {
				last_improvement++;
			}

			if (min_val <= alpha) {
				if (std::abs(min_val) < INF - MAX_DEPTH && position.hash_value && current_depth > 1 && min_val) {
					int white_val = playing == "white" ? min_val : -min_val;
					depth_tt.insert({ position.hash_value, white_val, best_move.move, UPPER_BOUND, remaining_depth });
					replace_tt.insert({ position.hash_value, white_val, best_move.move, UPPER_BOUND, remaining_depth });
				}

				history[from_square][to_square] += (remaining_depth * remaining_depth);

				return min_val;
			}

			beta = std::min(beta, min_val);
		}

		if (no_legal_moves) {
			std::string terminal_state = position.get_terminal_state(0);
			if ((terminal_state == "white win" && playing == "white") ||
				(terminal_state == "black win" && playing == "black")) {
				return INF - current_depth;
			}
			else if ((terminal_state == "white win" && playing == "black") ||
						(terminal_state == "black win" && playing == "white")) {
				return -INF + current_depth;
			} else {
				return 0;
			}
		}

		if (std::abs(min_val) < INF - MAX_DEPTH && position.hash_value && current_depth > 1 && min_val) {
			int white_val = playing == "white" ? min_val : -min_val;
			depth_tt.insert({ position.hash_value, white_val, best_move.move, EXACT, remaining_depth });
			replace_tt.insert({ position.hash_value, white_val, best_move.move, EXACT, remaining_depth });
		}

		return min_val;
	}
}

Move Engine::get_move(Position& position, bool verbose) {
	auto start_time = std::chrono::high_resolution_clock::now();
	total_nodes = 0;
	evaluated_positions = 0;

	if (use_book && position.half_moves < 10) {
		// Flavio Martin's opening book
		Move book_move = Polyglot::get_book_move(position, "titans.bin");
		if (book_move.move) {
			return book_move;
		} else {
			use_book = false;
		}
	}

	playing = position.turn == WHITE ? "white" : "black";

	// moves for the history heuristic become outdated, so have it iteratively decay
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			history[i][j] *= 0.8;
		}
	}

	Move best_move;
	Move prev_depth_best;
	int evaluation;

	for (int i = 1; i <= MAX_DEPTH; i++) {
		int best_value = std::numeric_limits<int>::min();
		int alpha = std::numeric_limits<int>::min();
		int beta = std::numeric_limits<int>::max();
		std::vector<std::pair<int, Move>> vals;

		auto pseudo_legal_moves = position.get_pseudo_legal_moves();

		std::sort(pseudo_legal_moves.begin(), pseudo_legal_moves.end(), [&](const Move& a, const Move& b) {
			if (a == prev_depth_best) return true;
			if (b == prev_depth_best) return false;

			int a_history = history[a.from()][a.to()];
			int b_history = history[b.from()][b.to()];

			if (a.priority > 0 || b.priority > 0) {
				return a.priority > b.priority;
			}
			
			return a_history > b_history;
		});

		for (const auto& move : pseudo_legal_moves) {
			if (!position.is_legal(move)) {
				continue;
			}

			position.make_move(move);
			int val = value(position, 0, i, alpha, beta, start_time);
			position.pop();
			vals.push_back(std::make_pair(val, move));
			alpha = std::max(alpha, val);

			if (val <= INF && val > best_value) {
				best_value = val;
				evaluation = val;
				// this is ok because the best move from the previous depth is always searched first
				best_move = move;
			}

			// time cutoff
			if (val > INF) {
				if (verbose) {
					auto now = std::chrono::high_resolution_clock::now();
					double time = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
					printf("time: %f\n", time / 1000);
					printf("depth: %d\n", i - 1);
					printf("evaluation: %d\n", playing == "white" ? evaluation : -evaluation);
					printf("total nodes: %d\n", total_nodes);
					printf("evaluated positions: %d\n\n", evaluated_positions);
				}

				return best_move;
			}
		}

		if (vals.empty()) {
			perror("no legal moves");
		}

		auto best = std::max_element(vals.begin(), vals.end(),
			[](const std::pair<int, Move>& a, const std::pair<int, Move>& b) {
				return a.first < b.first;
			});
		
		best_move = best->second;
		best_value = best->first;

		prev_depth_best = best_move;
	}

	if (verbose) {
		auto now = std::chrono::high_resolution_clock::now();
		double time = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
		printf("time: %f\n", time / 1000);
		printf("depth: %d\n", MAX_DEPTH);
		printf("evaluation: %d\n", playing == "white" ? evaluation : -evaluation);
		printf("total nodes: %d\n", total_nodes);
		printf("evaluated positions: %d\n\n", evaluated_positions);
	}

	return best_move;
}