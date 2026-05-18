#include "movepick.hh"

MovePicker::MovePicker(Position& position, Engine& engine, int current_depth, Move hash_move, bool only_tactics)
: position(position), engine(engine), current_depth(current_depth), hash_move(hash_move), only_tactics(only_tactics) {}

template<MoveGenType Type>
void MovePicker::get_sorted_moves() {
    MoveList buffer;
    get_pseudo_legal_moves(position, &buffer, Type);

    scored_moves.size = 0;

    // score moves
    if constexpr (Type == TACTIC) {
        for (Move move : buffer) {
            int score = 0;
            score += (1 << 30) * (move.promote_to() == QUEEN);
            int piece = std::max(0, Position::get_piece_type(position.board[move.from()]));
            int capture = std::max(0, Position::get_piece_type(position.board[move.to()]));
            score += engine.capture_history[move.to()][piece][capture];

            scored_moves.add(ScoredMove(move, score));
        }
    } else {
        for (Move move : buffer) {
            int score = 0;

            score += 16384 * (move == engine.killers[current_depth][0]);
            score += 8192 * (move == engine.killers[current_depth][1]);

            score += 2 * engine.quiet_history[move.from()][move.to()];

            // continuation history
            int piece = position.board[move.from()] - 1;
            int ply = current_depth + MAX_CH_PLY;
            score += engine.cont_history[ply - 1][move.to()][piece];
            score += engine.cont_history[ply - 2][move.to()][piece];

            scored_moves.add(ScoredMove(move, score));
        }
    }

    std::sort(scored_moves.begin(), scored_moves.end(), [](const ScoredMove& a, const ScoredMove& b) {
        return a.score > b.score;
    });
}


Move MovePicker::next_move() {
    Move move;
    if (stage == HASH_MOVE) {
        move = hash_move;
        stage = GOOD_TACTICS;
    } else if (stage == GOOD_TACTICS) {
        if (index == 0) {
            get_sorted_moves<TACTIC>();
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
                stage = BAD_TACTICS;
                return next_move();
            }

            stage = QUIETS;
            get_sorted_moves<QUIET>();

            return next_move();
        }
    } else if (stage == QUIETS) {
        if (index < scored_moves.size) {
            move = scored_moves.list[index++].move;
        } else {
            stage = BAD_TACTICS;
            index = 0;
            return next_move();
        }
    } else if (stage == BAD_TACTICS) {
        if (index < bad_tactics.size) {
            move = bad_tactics.moves[index++];
        } else {
            return Move();
        }
    }
    
    if (move && (move != hash_move || (stage == HASH_MOVE + 1 && index == 0)) && is_legal(position, move)) {
        return move;
    }
    return next_move();
}