#include "movepick.hh"

MovePicker::MovePicker(Position& position, Engine& engine, int current_depth, Move hash_move, bool only_tactics)
: position(position), engine(engine), current_depth(current_depth), hash_move(hash_move), only_tactics(only_tactics) {
    killer = engine.killers[current_depth];
    if (killer == hash_move) {
        killer = Move();
    }
}

template<MoveGenType Type>
void MovePicker::get_sorted_moves() {
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

    std::sort(scored_moves.begin(), scored_moves.end(), [](const ScoredMove& a, const ScoredMove& b) {
        return a.score > b.score;
    });
}


Move MovePicker::next_move() {
    Move move;
    if (stage == STAGE_HASH_MOVE) {
        if (hash_move) {
            move = hash_move;
        } else {
            stage = STAGE_GOOD_TACTICS;
            return next_move();
        }
    } else if (stage == STAGE_GOOD_TACTICS) {
        if (index == 0) {
            get_sorted_moves<GEN_TACTIC>();
        }

        while (index < scored_moves.size) {
            ScoredMove sm = scored_moves.list[index];
            Move m = sm.move;
            // under-promotions or negative static exchange evaluation
            if ((m.promote_to() && m.promote_to() != QUEEN) || !position.SEE(m)) {
                bad_tactics.add(m);
                index++;
            } else {
                break;
            }
        }

        if (index < scored_moves.size) {
            move = scored_moves.list[index++].move;
        } else {
            index = 0;

            if (only_tactics) {
                stage = STAGE_BAD_TACTICS;
                return next_move();
            }

            stage = STAGE_KILLER;
            return next_move();
        }
    } else if (stage == STAGE_KILLER) {
        if (killer && is_pseudo_legal(position, killer)) {
            move = killer;
        } else {
            stage = STAGE_QUIETS;
            return next_move();
        }
    } else if (stage == STAGE_QUIETS) {
        if (index == 0) {
            get_sorted_moves<GEN_QUIET>();
        }

        if (index < scored_moves.size) {
            move = scored_moves.list[index++].move;
        } else {
            stage = STAGE_BAD_TACTICS;
            index = 0;
            return next_move();
        }
    } else if (stage == STAGE_BAD_TACTICS) {
        if (index < bad_tactics.size) {
            move = bad_tactics.moves[index++];
        } else {
            return Move();
        }
    }
    
    if (!move) return next_move();

    if (move == hash_move) {
        if (stage != STAGE_HASH_MOVE) {
            return next_move();
        }
        stage = STAGE_GOOD_TACTICS;
    }

    if (move == killer) {
        if (stage != STAGE_KILLER) {
            return next_move();
        }
        stage = STAGE_QUIETS;
    }

    if (!is_legal(position, move)) return next_move();

    return move;
}