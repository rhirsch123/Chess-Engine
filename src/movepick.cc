#include "movepick.hh"

MovePicker::MovePicker(Position& position, Engine& engine, int current_depth, Move hash_move, bool only_tactics)
: position(position), engine(engine), current_depth(current_depth), hash_move(hash_move), only_tactics(only_tactics) {
    killer = engine.killers[current_depth];
    if (killer == hash_move) {
        killer = Move();
    }
}

template<MoveGenType Type>
void MovePicker::get_scored_moves() {
    MoveList buffer;
    get_pseudo_legal_moves(position, &buffer, Type);

    scored_moves.size = 0;

    // score moves
    for (Move move : buffer) {
        int score = 0;
        if constexpr (Type == GEN_TACTIC) {
            score += (1 << 30) * (move.promote_to() == QUEEN);
            int piece = std::max(0, Position::get_piece_type(position.piece_on(move.from())));
            int capture = std::max(0, Position::get_piece_type(position.piece_on(move.to())));

            score += engine.capture_history[move.to()][piece][capture];
        } else {
            score += 2 * engine.quiet_history[move.from()][move.to()];

            // continuation history
            int piece = position.piece_on(move.from()) - 1;
            int ply = current_depth + MAX_CH_PLY;
            score += engine.cont_history[ply - 1][move.to()][piece];
            score += engine.cont_history[ply - 2][move.to()][piece];
        }

        scored_moves.add(ScoredMove(move, score));
    }
}

static inline Move pop_best(ScoredMoveList& moves) {
    int best_idx = 0;
    for (int i = 1; i < moves.size; i++) {
        if (moves.list[i].score > moves.list[best_idx].score) {
            best_idx = i;
        }
    }

    Move best_move = moves.list[best_idx].move;
    moves.list[best_idx] = moves.list[--moves.size];
    return best_move;
}


Move MovePicker::next_move() {
    switch (stage) {
        case STAGE_HASH_MOVE: {
            stage++;
            if (hash_move) {
                return hash_move;
            }

            [[fallthrough]];
        }

        case STAGE_GEN_TACTICS: {
            get_scored_moves<GEN_TACTIC>();
            stage++;

            [[fallthrough]];
        }

        case STAGE_GOOD_TACTICS: {
            while (scored_moves.size) {
                Move move = pop_best(scored_moves);

                if (move == hash_move || move == killer) {
                    continue;
                }

                // under-promotions or negative static exchange evaluation
                if ((move.promote_to() && move.promote_to() != QUEEN) || !position.SEE(move)) {
                    bad_tactics.add(move);
                } else {
                    return move;
                }
            }

            if (only_tactics) {
                stage = STAGE_BAD_TACTICS;
                return next_move();
            }

            stage++;

            [[fallthrough]];
        }

        case STAGE_KILLER: {
            stage++;
            if (killer && is_pseudo_legal(position, killer)) {
                return killer;
            }

            [[fallthrough]];
        }

        case STAGE_GEN_QUIETS: {
            get_scored_moves<GEN_QUIET>();
            stage++;

            [[fallthrough]];
        }

        case STAGE_QUIETS: {
            while (scored_moves.size) {
                Move move = pop_best(scored_moves);

                if (move == hash_move || move == killer) {
                    continue;
                }
                return move;
            }

            stage++;

            [[fallthrough]];
        }

        case STAGE_BAD_TACTICS: {
            while (index < bad_tactics.size) {
                Move move = bad_tactics.moves[index++];

                if (move == hash_move || move == killer) {
                    continue;
                }
                return move;
            }

            return Move();
        }

        default:
            return Move();
    }
}