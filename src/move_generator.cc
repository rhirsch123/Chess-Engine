#include "move_generator.hh"
#include "evaluation.hh"

MoveGenerator::MoveGenerator(Position& position, Engine& engine, int current_depth)
: position(position), engine(engine), current_depth(current_depth) {
    tactics = position.get_pseudo_legal_moves(TACTIC);
    std::sort(tactics.begin(), tactics.end(), [&](const Move& a, const Move& b) {
        if (a.promote_to()) return a.promote_to() == QUEEN;
        if (b.promote_to()) return b.promote_to() != QUEEN;

        int a_piece = Position::get_piece_type(position.board[a.start_row()][a.start_col()]);
        int a_capture = Position::get_piece_type(position.board[a.end_row()][a.end_col()]);
        int b_piece = Position::get_piece_type(position.board[b.start_row()][b.start_col()]);
        int b_capture = Position::get_piece_type(position.board[b.end_row()][b.end_col()]);
        return engine.capture_history[a.to()][a_piece][a_capture] > engine.capture_history[b.to()][b_piece][b_capture];
    });
}

Move MoveGenerator::next_move() {
    Move move;
    if (stage == GOOD_TACTICS) {
        while (index < tactics.size()) {
            Move move = tactics[index];
            // static exchange evaluation
            if (!Evaluation::safe_move(position, move)) {
                bad_tactics.push_back(move);
                index++;
            } else {
                break;
            }
        }

        if (index < tactics.size()) {
            move = tactics[index++];
        } else {
            stage = QUIETS;
            index = 0;
            quiets = position.get_pseudo_legal_moves(QUIET);
            std::sort(quiets.begin(), quiets.end(), [&](const Move& a, const Move& b) {
                if (a == engine.killers[current_depth][0]) return true;
                if (b == engine.killers[current_depth][0]) return false;
                if (a == engine.killers[current_depth][1]) return true;
                if (b == engine.killers[current_depth][1]) return false;

                return engine.quiet_history[a.from()][a.to()] > engine.quiet_history[b.from()][b.to()];
            });

           return next_move();
        }
    } else if (stage == QUIETS) {
        if (index < quiets.size()) {
            move = quiets[index++];
        } else {
            stage = BAD_TACTICS;
            index = 0;
            return next_move();
        }
    } else if (stage == BAD_TACTICS) {
        if (index < bad_tactics.size()) {
            move = bad_tactics[index++];
        } else {
            return Move();
        }
    }
    
    if (move && position.is_legal(move)) {
        return move;
    }
    return next_move();
}