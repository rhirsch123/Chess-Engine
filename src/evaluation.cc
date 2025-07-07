#include "evaluation.hh"

namespace Evaluation {
    // from perspective of white
    static uint64_t get_above_mask(int row) {
        return (1ULL << (row * 8)) - 1;
    }

    // from perspective of white
    static uint64_t get_below_mask(int row) {
        return ~((1ULL << ((row + 1) * 8)) - 1);
    }

    static uint64_t get_column_mask(int col) {
        return 0x0101010101010101ULL << col;
    }

    static uint64_t get_row_mask(int row) {
        return 0xFFULL << (row * 8);
    }

    static uint64_t get_neighboring_cols(int col) {
        uint64_t neighboring_cols = 0ULL;
        if (col - 1 >= 0) {
            neighboring_cols |= get_column_mask(col - 1);
        }
        if (col + 1 < 8) {
            neighboring_cols |= get_column_mask(col + 1);
        }
        return neighboring_cols;
    }

    static int manhattan_distance(int square1, int square2) {
        int row1 = square1 / 8;
        int col1 = square1 % 8;
        int row2 = square2 / 8;
        int col2 = square2 % 8;

        return std::abs(row1 - row2) + std::abs(col1 - col2);
    }

    static uint64_t get_pawn_attacks(uint64_t pawns, int color) {
        uint64_t attacks;
        if (color == WHITE) {
            // shift up
            attacks = pawns >> 8;
        } else {
            // shift down
            attacks = pawns << 8;
        }

        // shift left and right
        return ((attacks & ~get_column_mask(7)) << 1) | ((attacks & ~get_column_mask(0)) >> 1);
    }

    static uint64_t get_attackers(Position& position, int square, int color, uint64_t blockers = 0ULL) {
        uint64_t attackers = 0ULL;

        if (!blockers) {
            blockers = position.white_pieces | position.black_pieces;
        }

        attackers |= (position.piece_maps[ROOK][color] | position.piece_maps[QUEEN][color]) & position.get_rook_moves(square, blockers);
        attackers |= (position.piece_maps[BISHOP][color] | position.piece_maps[QUEEN][color]) & position.get_bishop_moves(square, blockers);
        attackers |= position.piece_maps[KNIGHT][color] & position.knight_moves[square];
        attackers |= position.piece_maps[KING][color] & position.king_moves[square];

        if (color == WHITE) {
            attackers |= position.piece_maps[PAWN][WHITE] & position.black_pawn_attacks[square];
        } else {
            attackers |= position.piece_maps[PAWN][BLACK] & position.white_pawn_attacks[square];
        }

        return attackers;
    }

    // two uses: check if a move is safe, check if a certain piece could safely go to that square
    bool safe_move(Position& position, Move move, int piece) {
        bool valid_move = !piece;
        if (valid_move) {
            piece = position.board[move.start_row()][move.start_col()];
        }
        if (!piece) {
            return false;
        }

        char move_turn = Position::get_color(piece);
        int turn = !move_turn;
        int capture = position.board[move.end_row()][move.end_col()];
        int exchange = capture ? Position::values[Position::get_piece_type(capture)] : 0;
        int last_attacker = Position::get_piece_type(piece);

        if (exchange >= Position::values[last_attacker]) {
            return true;
        }

        // sliding piece blockers - start out as every piece
        uint64_t blockers = position.white_pieces | position.black_pieces;

        int start_square = move.from();
        int end_square = move.to();
        
        // update blockers with move
        if (valid_move) {
            blockers &= ~(1ULL << start_square);
        }
        blockers |= (1ULL << end_square);

        uint64_t attackers = get_attackers(position, end_square, turn, blockers) & blockers;

        while (attackers) {
            int next_attacker;
            int next_attacker_square;
            for (int weakest_piece = PAWN; weakest_piece <= KING; weakest_piece++) {
                if (weakest_piece == KING) {
                    if (!(get_attackers(position, end_square, !turn, blockers) & blockers)) {
                        if (turn == move_turn) {
                            exchange += Position::values[last_attacker];
                        } else {
                            exchange -= Position::values[last_attacker];
                        }
                    }
                    return exchange >= 0;
                }

                uint64_t lowest_attackers = position.piece_maps[weakest_piece][turn] & attackers;
                if (lowest_attackers) {
                    // choose arbitrary lowest attacker if multiple
                    next_attacker_square = least_set_bit(lowest_attackers);
                    next_attacker = weakest_piece;
                    break;
                }
            }

            if (turn == move_turn) {
                exchange += Position::values[last_attacker];
                if (exchange >= Position::values[next_attacker]) {
                    return true;
                }
            } else {
                exchange -= Position::values[last_attacker];
                if (exchange + Position::values[next_attacker] < 0) {
                    return false;
                }
            }

            last_attacker = next_attacker;

            // remove used piece from sliding piece blockers and possible next attackers
            blockers &= ~(1ULL << next_attacker_square);

            turn = !turn;
            attackers = get_attackers(position, end_square, turn, blockers) & blockers;
        }

        return exchange >= 0;
    }


    static int evaluate_pawn_structure(Position& position, Engine& engine, int color, uint64_t pawn_attacks) {
        int isolated_pawn = 0;
        int doubled_pawn = 0;
        int backwards_pawn = 0;
        int pawn_storm = 0;
        int pawn_chain = 0;

        uint64_t pawns = position.piece_maps[PAWN][color];
        while (pawns) {
            int square = least_set_bit(pawns);
            pawns &= pawns - 1;

            int row = square / 8;
            int col = square % 8;

            // isolated
            uint64_t neighboring_cols = get_neighboring_cols(col);
            if (!(neighboring_cols & position.piece_maps[PAWN][color])) {
                isolated_pawn -= 1;
            }

            uint64_t above_mask = get_above_mask(row);

            // doubled
            int num_in_col = num_set_bits(position.piece_maps[PAWN][color] & get_column_mask(col) & above_mask);
            doubled_pawn -= num_in_col - 1;

            // backwards: no neighboring pawns behind it and can't move forward
            if (color == WHITE) {
                uint64_t below_mask = get_below_mask(row);
                if (!(below_mask & neighboring_cols & position.piece_maps[PAWN][WHITE])) {
                    if (position.board[row - 1][col]) {
                        backwards_pawn -= 1;
                    } else {
                        int forward_square = (row - 1) * 8 + col;
                        int white_attacks = num_set_bits(get_attackers(position, forward_square, WHITE));
                        int black_attacks = num_set_bits(get_attackers(position, forward_square, BLACK));
                        if (black_attacks > white_attacks) {
                            backwards_pawn -= 1;
                        }
                    }
                }

                // king side pawn storms
                if (position.king_positions[3] <= 2 && col <= 2 && row < 5) {
                    pawn_storm += 5 - row;
                } else if (position.king_positions[3] >= 5 && col >= 5 && row < 5) {
                    pawn_storm += 5 - row;
                }
            } else {
                if (!(above_mask & neighboring_cols & position.piece_maps[PAWN][BLACK])) {
                    if (position.board[row + 1][col]) {
                        backwards_pawn -= 1;
                    } else {
                        int forward_square = (row + 1) * 8 + col;
                        int white_attacks = num_set_bits(get_attackers(position, forward_square, WHITE));
                        int black_attacks = num_set_bits(get_attackers(position, forward_square, BLACK));
                        if (white_attacks > black_attacks) {
                            backwards_pawn -= 1;
                        }
                    }
                }

                // king side pawn storms
                if (position.king_positions[1] <= 2 && col <= 2 && row > 2) {
                    pawn_storm += row - 2;
                } else if (position.king_positions[1] >= 5 && col >= 5 && row > 2) {
                    pawn_storm += row - 2;
                }
            }
        }
        
        // number of pawns defending pawns
        pawn_chain += num_set_bits(position.piece_maps[PAWN][color] & pawn_attacks);

        return isolated_pawn * engine.ISOLATED_PAWN_WEIGHT + doubled_pawn * engine.DOUBLED_PAWN_WEIGHT +
            backwards_pawn * engine.BACKWARDS_PAWN_WEIGHT + pawn_storm * engine.PAWN_STORM_WEIGHT +
            pawn_chain * engine.PAWN_CHAIN_WEIGHT;
    }


    static int evaluate_knights(Position& position, Engine& engine, int color, uint64_t opp_pawn_attacks) {
        int knight_mobility = 0;
        int knight_outpost = 0;
        int king_tropism = 0;

        uint64_t knights = position.piece_maps[KNIGHT][color];
        while (knights) {
            int square = least_set_bit(knights);
            knights &= knights - 1;

            int row = square / 8;
            int col = square % 8;
            
            uint64_t friendly_pieces = color == WHITE ? position.white_pieces : position.black_pieces;
            uint64_t attacked_squares = position.knight_moves[square] & (~friendly_pieces) & (~opp_pawn_attacks);
            int mobility = num_set_bits(attacked_squares);
        
            if (mobility <= 2 && ((color == WHITE && row <= 3) || (color == BLACK && row >= 4))) {
                // check trapped
                bool trapped = true;
                while (attacked_squares) {
                    int attack = least_set_bit(attacked_squares);
                    attacked_squares &= attacked_squares - 1;
                    if (safe_move(position, Move(square, attack))) {
                        trapped = false;
                        break;
                    }
                }
                if (trapped) {
                    knight_mobility -= 200;
                }
            }
            knight_mobility += engine.knight_mobility_bonus[mobility];

            // outpost: defended by a pawn, bonus if no pawns can attack it
            if (color == WHITE) {
                if (row + 1 < 8 && ((col - 1 >= 0 && position.board[row + 1][col - 1] == WHITE_PAWN) ||
                   (col + 1 < 8 && position.board[row + 1][col + 1] == WHITE_PAWN))) {
                    knight_outpost += 1;
                    if (!(get_neighboring_cols(col) & get_above_mask(row) & position.piece_maps[PAWN][BLACK])) {
                        knight_outpost += 2;
                    }
                }

                // distance to black king
                king_tropism -= manhattan_distance(square, position.king_positions[2] * 8 + position.king_positions[3]) * engine.near_opp_king_bonus[KNIGHT];
            } else {
                if (row - 1 >= 0 && ((col - 1 >= 0 && position.board[row - 1][col - 1] == BLACK_PAWN) ||
                   (col + 1 < 8 && position.board[row - 1][col + 1] == BLACK_PAWN))) {
                    knight_outpost += 1;
                    if (!(get_neighboring_cols(col) & get_below_mask(row) & position.piece_maps[PAWN][WHITE])) {
                        knight_outpost += 2;
                    }
                }

                // distance to white king
                king_tropism -= manhattan_distance(square, position.king_positions[0] * 8 + position.king_positions[1]) * engine.near_opp_king_bonus[KNIGHT];
            }
        }

        return knight_mobility * engine.KNIGHT_MOBILITY_WEIGHT + knight_outpost * engine.KNIGHT_OUTPOST_WEIGHT +
            king_tropism * engine.KING_TROPISM_WEIGHT;
    }


    static int evaluate_bishops(Position& position, Engine& engine, int color, uint64_t opp_pawn_attacks) {
        int bishop_pair = 0;
        int bishop_mobility = 0;
        int bishop_pawn_complex = 0;
        int king_tropism = 0;

        uint64_t bishops = position.piece_maps[BISHOP][color];
        // bishop pair
        if (num_set_bits(bishops) >= 2) {
            bishop_pair += 1;
        }
        while (bishops) {
            int square = least_set_bit(bishops);
            bishops &= bishops - 1;

            int row = square / 8;
            int col = square % 8;

            uint64_t friendly_pieces = color == WHITE ? position.white_pieces : position.black_pieces;
            uint64_t attacked_squares = position.get_bishop_moves(square, position.white_pieces | position.black_pieces) & (~friendly_pieces) & (~opp_pawn_attacks);
            int mobility = num_set_bits(attacked_squares);

            if (mobility <= 2 && ((color == WHITE && row <= 3) || (color == BLACK && row >= 4))) {
                // check trapped
                bool trapped = true;
                while (attacked_squares) {
                    int attack = least_set_bit(attacked_squares);
                    attacked_squares &= attacked_squares - 1;
                    if (safe_move(position, Move(square, attack))) {
                        trapped = false;
                        break;
                    }
                }
                if (trapped) {
                    bishop_mobility -= 200;
                }
            }
            bishop_mobility += engine.bishop_mobility_bonus[mobility];

            // dark/light square pawn complex
            int num_light_pawns = num_set_bits(LIGHT_SQUARES & (position.piece_maps[PAWN][WHITE] | position.piece_maps[PAWN][BLACK]));
            int num_dark_pawns = num_set_bits(DARK_SQUARES & (position.piece_maps[PAWN][WHITE] | position.piece_maps[PAWN][BLACK]));
            if ((row + col) % 2 == 0) { // light squared bishop
                bishop_pawn_complex += num_dark_pawns - num_light_pawns;
            } else { // dark squared bishop
                bishop_pawn_complex += num_light_pawns - num_dark_pawns;
            }

            // distance to opp king
            if (color == WHITE) {
                king_tropism -= manhattan_distance(square, position.king_positions[2] * 8 + position.king_positions[3]) * engine.near_opp_king_bonus[BISHOP];
            } else {
                king_tropism -= manhattan_distance(square, position.king_positions[0] * 8 + position.king_positions[1]) * engine.near_opp_king_bonus[BISHOP];
            }
        }

        return bishop_pair * engine.BISHOP_PAIR_WEIGHT + bishop_mobility * engine.BISHOP_MOBILITY_WEIGHT +
            bishop_pawn_complex * engine.BISHOP_PAWN_COMPLEX_WEIGHT + king_tropism * engine.KING_TROPISM_WEIGHT;
    }


    static int evaluate_rooks(Position& position, Engine& engine, int color, uint64_t opp_pawn_attacks) {
        int rook_mobility = 0;
        int rook_behind_king = 0;
        int king_tropism = 0;

        uint64_t rooks = position.piece_maps[ROOK][color];
        while (rooks) {
            int square = least_set_bit(rooks);
            rooks &= rooks - 1;

            int row = square / 8;
            int col = square % 8;

            uint64_t friendly_pieces = color == WHITE ? position.white_pieces : position.black_pieces;
            uint64_t attacked_squares = position.get_rook_moves(square, position.white_pieces | position.black_pieces) & (~friendly_pieces) & (~opp_pawn_attacks);
            int mobility = num_set_bits(attacked_squares);
                    
            if (mobility <= 2 && ((color == WHITE && row <= 3) || (color == BLACK && row >= 4))) {
                // check trapped
                bool trapped = true;
                while (attacked_squares) {
                    int attack = least_set_bit(attacked_squares);
                    attacked_squares &= attacked_squares - 1;
                    if (safe_move(position, Move(square, attack))) {
                        trapped = false;
                        break;
                    }
                }
                if (trapped) {
                    rook_mobility -= 300;
                }
            }
            
            rook_mobility += engine.rook_mobility_bonus[mobility];

            // trapped by king
            int king_col = color == WHITE ? position.king_positions[1] : position.king_positions[3];
            if ((king_col <= 3 && col < king_col) ||
                (king_col >= 4 && col > king_col)) {

                rook_behind_king -= 1;
            }

            if (color == WHITE) {
                // distance to black king
                king_tropism -= manhattan_distance(square, position.king_positions[2] * 8 + position.king_positions[3]) * engine.near_opp_king_bonus[ROOK];
            } else {
                // distance to white king
                king_tropism -= manhattan_distance(square, position.king_positions[0] * 8 + position.king_positions[1]) * engine.near_opp_king_bonus[ROOK];
            }
        }

        return rook_mobility * engine.ROOK_MOBILITY_WEIGHT + rook_behind_king * engine.ROOK_BEHIND_KING_WEIGHT +
            king_tropism * engine.KING_TROPISM_WEIGHT;
    }


    static int evaluate_queens(Position& position, Engine& engine, int color, uint64_t opp_pawn_attacks) {
        int queen_out_early = 0;
        int queen_mobility = 0;
        int queen_position = 0;
        int king_tropism = 0;

        if (position.half_moves < 15) {
            // don't bring queen out early
            if (position.piece_maps[QUEEN][color]) {
                int queen = least_set_bit(position.piece_maps[QUEEN][color]);
                int row = queen / 8;
                if ((color == WHITE && row < 6) || (color == BLACK && row > 1)) {
                    queen_out_early -= 1;
                }
            }
        } else {
            uint64_t queens = position.piece_maps[QUEEN][color];
            while (queens) {
                int square = least_set_bit(queens);
                queens &= queens - 1;

                int row = square / 8;
                int col = square % 8;
                
                uint64_t all_pieces = position.white_pieces | position.black_pieces;
                uint64_t friendly_pieces = color == WHITE ? position.white_pieces : position.black_pieces;
                uint64_t attacked_squares = (position.get_rook_moves(square, all_pieces) | 
                                            position.get_bishop_moves(square, all_pieces)) & (~friendly_pieces) & (~opp_pawn_attacks);
                int mobility = num_set_bits(attacked_squares);

                if (mobility <= 2 && ((color == WHITE && row <= 3) || (color == BLACK && row >= 4))) {
                    // check trapped
                    bool trapped = true;
                    while (attacked_squares) {
                        int attack = least_set_bit(attacked_squares);
                        attacked_squares &= attacked_squares - 1;
                        if (safe_move(position, Move(square, attack))) {
                            trapped = false;
                            break;
                        }
                    }
                    if (trapped) {
                        queen_mobility -= 500;
                    }
                }
                
                queen_mobility += engine.queen_mobility_bonus[mobility];

                // distance to king
                if (color == WHITE) {
                    king_tropism -= manhattan_distance(square, position.king_positions[2] * 8 + position.king_positions[3]) * engine.near_opp_king_bonus[QUEEN];
                } else {
                    king_tropism -= manhattan_distance(square, position.king_positions[0] * 8 + position.king_positions[1]) * engine.near_opp_king_bonus[QUEEN];
                }
            }
        }

        return queen_out_early * engine.QUEEN_OUT_EARLY_WEIGHT + queen_mobility * engine.QUEEN_MOBILITY_WEIGHT +
            king_tropism * engine.KING_TROPISM_WEIGHT;
    }


    static int evaluate_king_safety(Position& position, Engine& engine) {
        int king_position = 0;
        int king_attackers = 0;
        int open_king = 0;
        int king_open_column = 0;

        // king position
        if (position.king_positions[1] >= 6 ||
            position.king_positions[1] <= 2) {
            king_position += 1;
        } else if (!position.can_castle[0] && !position.can_castle[1]) {
            king_position -= 2;
        }
    
        if (position.king_positions[3] >= 6 ||
            position.king_positions[3] <= 2) {
            king_position -= 1;
        } else if (!position.can_castle[2] && !position.can_castle[3]) {
            king_position += 2;
        }

        // count attackers of squares arround the king
        for (auto dir : position.directions.queen) {
            int white_row = position.king_positions[0] + dir.first;
            int white_col = position.king_positions[1] + dir.second;
            if (white_row >= 0 && white_row < 8 && white_col >= 0 && white_col < 8) {
                king_attackers -= num_set_bits(get_attackers(position, white_row * 8 + white_col, BLACK));
            }

            int black_row = position.king_positions[2] + dir.first;
            int black_col = position.king_positions[3] + dir.second;
            if (black_row >= 0 && black_row < 8 && black_col >= 0 && black_col < 8) {
                king_attackers += num_set_bits(get_attackers(position, black_row * 8 + black_col, WHITE));
            }
        }

        // open space around king is bad - count how many forward moves it would have if it were a queen
        uint64_t all_pieces = position.white_pieces | position.black_pieces;
        int white_square = position.king_positions[0] * 8 + position.king_positions[1];
        uint64_t attacked_squares = (position.get_rook_moves(white_square, all_pieces) | 
            position.get_bishop_moves(white_square, all_pieces)) & (~position.white_pieces);
        uint64_t above_mask = get_above_mask(position.king_positions[0]);
        open_king -= num_set_bits(attacked_squares & above_mask);

        int black_square = position.king_positions[2] * 8 + position.king_positions[3];
        attacked_squares = (position.get_rook_moves(black_square, all_pieces) | 
            position.get_bishop_moves(black_square, all_pieces)) & (~position.black_pieces);
        uint64_t below_mask = get_below_mask(position.king_positions[2]);
        open_king += num_set_bits(attacked_squares & below_mask);

        // open columns in front of king
        for (int c = position.king_positions[1] - 1; c <= position.king_positions[1] + 1; c++) {
            if (c < 0 || c >= 8) {
                continue;
            }
            int weight = 0;
            // no white pawn in column
            if (!(get_column_mask(c) & position.piece_maps[PAWN][WHITE])) {
                weight += 2;
            }
            // no black pawn in column
            if (!(get_column_mask(c) & position.piece_maps[PAWN][BLACK])) {
                weight += 1;
            }

            if (c == position.king_positions[1]) {
                weight *= 2;
            }
            king_open_column -= weight;
        }
        for (int c = position.king_positions[3] - 1; c <= position.king_positions[3] + 1; c++) {
            if (c < 0 || c >= 8) {
                continue;
            }
            int weight = 0;

            // no black pawn in column
            if (!(get_column_mask(c) & position.piece_maps[PAWN][BLACK])) {
                weight += 2;
            }
            // no white pawn in column
            if (!(get_column_mask(c) & position.piece_maps[PAWN][WHITE])) {
                weight += 1;
            }

            if (c == position.king_positions[3]) {
                weight *= 2;
            }
            king_open_column += weight;
        }

        return king_position * engine.KING_POSITION_WEIGHT + king_attackers * engine.KING_ATTACKERS_WEIGHT +
            open_king * engine.OPEN_KING_WEIGHT + king_open_column * engine.KING_OPEN_COLUMN_WEIGHT;
    }


    // heuristic evalution of position, given weight parameters from engine
    // linear combination of features with weights tuned through reinforcement learning
    int evaluation(Position& position, Engine& engine) {
        int material = position.white_material - position.black_material;
        
        int major_material = position.white_material + position.black_material - 
            num_set_bits(position.piece_maps[PAWN][WHITE] | position.piece_maps[PAWN][BLACK]) * Position::values[PAWN];

        if (major_material > (Position::values[ROOK] * 4)) { // opening, middlegame
            int eval = material * engine.MATERIAL_WEIGHT;

            uint64_t white_pawn_attacks = get_pawn_attacks(position.piece_maps[PAWN][WHITE], WHITE);
            uint64_t black_pawn_attacks = get_pawn_attacks(position.piece_maps[PAWN][BLACK], BLACK);

            eval += evaluate_pawn_structure(position, engine, WHITE, white_pawn_attacks);
            eval -= evaluate_pawn_structure(position, engine, BLACK, black_pawn_attacks);

            eval += evaluate_knights(position, engine, WHITE, black_pawn_attacks);
            eval -= evaluate_knights(position, engine, BLACK, white_pawn_attacks);

            eval += evaluate_bishops(position, engine, WHITE, black_pawn_attacks);
            eval -= evaluate_bishops(position, engine, BLACK, white_pawn_attacks);
            
            eval += evaluate_rooks(position, engine, WHITE, black_pawn_attacks);
            eval -= evaluate_rooks(position, engine, BLACK, white_pawn_attacks);

            eval += evaluate_queens(position, engine, WHITE, black_pawn_attacks);
            eval -= evaluate_queens(position, engine, BLACK, white_pawn_attacks);

            eval += evaluate_king_safety(position, engine);

            // center control
            int center_control = 0;
            uint64_t all_pieces = position.white_pieces | position.black_pieces;
            for (int row = 2; row <= 5; row++) {
                for (int col = 2; col <= 5; col++) {
                    int square = row * 8 + col;
                    int white_attacks = num_set_bits(get_attackers(position, square, WHITE, all_pieces));
                    int black_attacks = num_set_bits(get_attackers(position, square, BLACK, all_pieces));
                    center_control += white_attacks - black_attacks;
                }
            }
            if (position.half_moves <= 15) {
                center_control *= 2;
            }
            eval += center_control * engine.CENTER_CONTROL_WEIGHT;

            // penalize certain "equal" trades - piece for three pawns, two minor pieces for rook and pawn, etc
            int white_piece_count = num_set_bits(position.white_pieces & ~position.piece_maps[PAWN][WHITE]);
            int black_piece_count = num_set_bits(position.black_pieces & ~position.piece_maps[PAWN][BLACK]);
            eval += (white_piece_count - black_piece_count) * engine.PIECE_DIFFERENCE_WEIGHT;
            
            if (engine.playing == "white") {
                return eval;
            } else {
                return -eval;
            }
            
        } else { // endgame
            uint64_t all_pieces = position.white_pieces | position.black_pieces;

            if (false && position.white_material + position.black_material == 
                num_set_bits(position.piece_maps[PAWN][WHITE] | position.piece_maps[PAWN][BLACK])) {
                // todo: king and pawn endgame

            } else if (position.white_material + position.black_material > (Position::values[ROOK] * 2) ||
                    position.piece_maps[PAWN][WHITE] || position.piece_maps[PAWN][BLACK]) {

                int isolated_pawn = 0;
                int doubled_pawn = 0;
                int passed_pawn = 0;
                int backwards_pawn = 0;
                int king_position = 0;
                int knight_mobility = 0;
                int bishop_pair = 0;
                int bishop_mobility = 0;
                int rook_mobility = 0;
                int queen_mobility = 0;
                int piece_difference = 0;

                // some endgames have a material advantage, but tend to be difficult or impossible to win
                int drawness = 0;
                // opposite colored bishops
                if (major_material == (Position::values[BISHOP] * 2) &&
                (((LIGHT_SQUARES & position.piece_maps[BISHOP][WHITE]) && (DARK_SQUARES & position.piece_maps[BISHOP][BLACK])) ||
                ((LIGHT_SQUARES & position.piece_maps[BISHOP][BLACK]) && (DARK_SQUARES & position.piece_maps[BISHOP][WHITE])))) {
                    drawness = 10;
                }

                // todo: rook vs rook and minor piece


                // king position: minimize weighted average distance to pawns
                int MY_PASSED_WEIGHT = 4;
                int OPP_PASSED_WEIGHT = 6;
                int MY_BACKWARDS_WEIGHT = 3;
                int OPP_BACKWARDS_WEIGHT = 3;
                int MY_NORMAL_WEIGHT = 2;
                int OPP_NORMAL_WEIGHT = 2;

                int white_king_distance = 0;
                int black_king_distance = 0;
                int white_sum_weights = 0;
                int black_sum_weights = 0;

                uint64_t white_pawn_attacks = 0ULL;
                uint64_t white_pawns = position.piece_maps[PAWN][WHITE];
                while (white_pawns) {
                    int square = least_set_bit(white_pawns);
                    white_pawns &= white_pawns - 1;

                    white_pawn_attacks |= position.white_pawn_attacks[square];

                    int row = square / 8;
                    int col = square % 8;

                    int white_dist = std::abs(position.king_positions[0] - row) +
                        std::abs(position.king_positions[1] - col);
                    int black_dist = std::abs(position.king_positions[2] - row) +
                        std::abs(position.king_positions[3] - col);

                    uint64_t neighboring_cols = 0ULL;
                    if (col - 1 >= 0) {
                        neighboring_cols |= get_column_mask(col - 1);
                    }
                    if (col + 1 < 8) {
                        neighboring_cols |= get_column_mask(col + 1);
                    }

                    // isolated
                    if (!(neighboring_cols & position.piece_maps[PAWN][WHITE])) {
                        isolated_pawn -= 1;
                    }

                    uint64_t above_mask = get_above_mask(row);
                    uint64_t below_mask = get_below_mask(row);

                    // doubled
                    int num_in_col = num_set_bits(position.piece_maps[PAWN][WHITE] & get_column_mask(col) & above_mask);
                    doubled_pawn -= num_in_col - 1;

                    int forward_square = (row - 1) * 8 + col;

                    if (!(above_mask & neighboring_cols & position.piece_maps[PAWN][BLACK]) && 
                        !(above_mask & get_column_mask(col) & (position.piece_maps[PAWN][WHITE] | position.piece_maps[PAWN][BLACK]))) {
                        // passed pawn - no pawns in front, no opponent pawns adjacent in front
                        passed_pawn += 6 - row;

                        white_king_distance += MY_PASSED_WEIGHT * white_dist;
                        black_king_distance += OPP_PASSED_WEIGHT * black_dist;
                        white_sum_weights += MY_PASSED_WEIGHT;
                        black_sum_weights += OPP_PASSED_WEIGHT;
                    } else if (!(below_mask & neighboring_cols & position.piece_maps[PAWN][WHITE]) && 
                        (position.board[row - 1][col] || (num_set_bits(get_attackers(position, forward_square, BLACK)) >
                        num_set_bits(get_attackers(position, forward_square, WHITE))))) {

                        // backwards pawn
                        backwards_pawn -= 1;

                        white_king_distance += MY_BACKWARDS_WEIGHT * white_dist;
                        black_king_distance += OPP_BACKWARDS_WEIGHT * black_dist;
                        white_sum_weights += MY_BACKWARDS_WEIGHT;
                        black_sum_weights += OPP_BACKWARDS_WEIGHT;
                    } else {
                        white_king_distance += MY_NORMAL_WEIGHT * white_dist;
                        black_king_distance += OPP_NORMAL_WEIGHT * black_dist;
                        white_sum_weights += MY_NORMAL_WEIGHT;
                        black_sum_weights += OPP_NORMAL_WEIGHT;
                    }
                }
                uint64_t black_pawn_attacks = 0ULL;
                uint64_t black_pawns = position.piece_maps[PAWN][BLACK];
                while (black_pawns) {
                    int square = least_set_bit(black_pawns);
                    black_pawns &= black_pawns - 1;

                    black_pawn_attacks |= position.black_pawn_attacks[square];

                    int row = square / 8;
                    int col = square % 8;

                    int white_dist = std::abs(position.king_positions[0] - row) +
                        std::abs(position.king_positions[1] - col);
                    int black_dist = std::abs(position.king_positions[2] - row) +
                        std::abs(position.king_positions[3] - col);

                    uint64_t neighboring_cols = 0ULL;
                    if (col - 1 >= 0) {
                        neighboring_cols |= get_column_mask(col - 1);
                    }
                    if (col + 1 < 8) {
                        neighboring_cols |= get_column_mask(col + 1);
                    }

                    // isolated
                    if (!(neighboring_cols & position.piece_maps[PAWN][BLACK])) {
                        isolated_pawn += 1;
                    }

                    uint64_t above_mask = get_above_mask(row);
                    uint64_t below_mask = get_below_mask(row);

                    // doubled
                    int num_in_col = num_set_bits(position.piece_maps[PAWN][BLACK] & get_column_mask(col) & above_mask);
                    doubled_pawn += num_in_col - 1;

                    int forward_square = (row + 1) * 8 + col;

                    if (!(below_mask & neighboring_cols & position.piece_maps[PAWN][WHITE]) && 
                        !(below_mask & get_column_mask(col) & (position.piece_maps[PAWN][WHITE] | position.piece_maps[PAWN][BLACK]))) {
                        // passed pawn - no pawns in front, no opponent pawns adjacent in front
                        passed_pawn -= row + 1;

                        white_king_distance += OPP_PASSED_WEIGHT * white_dist;
                        black_king_distance += MY_PASSED_WEIGHT * black_dist;
                        white_sum_weights += OPP_PASSED_WEIGHT;
                        black_sum_weights += MY_PASSED_WEIGHT;
                    } else if (!(above_mask & neighboring_cols & position.piece_maps[PAWN][BLACK]) &&
                        (position.board[row + 1][col] || (num_set_bits(get_attackers(position, forward_square, WHITE)) >
                        num_set_bits(get_attackers(position, forward_square, BLACK))))) {
                        
                        // backwards pawn
                        backwards_pawn += 1;

                        white_king_distance += OPP_BACKWARDS_WEIGHT * white_dist;
                        black_king_distance += MY_BACKWARDS_WEIGHT * black_dist;
                        white_sum_weights += OPP_BACKWARDS_WEIGHT;
                        black_sum_weights += MY_BACKWARDS_WEIGHT;
                    } else {
                        white_king_distance += OPP_NORMAL_WEIGHT * white_dist;
                        black_king_distance += MY_NORMAL_WEIGHT * black_dist;
                        white_sum_weights += OPP_NORMAL_WEIGHT;
                        black_sum_weights += MY_NORMAL_WEIGHT;
                    }
                }

                // king position
                king_position -= white_king_distance / white_sum_weights;
                king_position += black_king_distance / black_sum_weights;


                // knight
                uint64_t white_knights = position.piece_maps[KNIGHT][WHITE];
                while (white_knights) {
                    int square = least_set_bit(white_knights);
                    white_knights &= white_knights - 1;

                    uint64_t attacked_squares = position.knight_moves[square] & (~position.white_pieces) & (~black_pawn_attacks);
                    knight_mobility += engine.knight_mobility_bonus[num_set_bits(attacked_squares)];
                }
                uint64_t black_knights = position.piece_maps[KNIGHT][BLACK];
                while (black_knights) {
                    int square = least_set_bit(black_knights);
                    black_knights &= black_knights - 1;

                    uint64_t attacked_squares = position.knight_moves[square] & (~position.black_pieces) & (~white_pawn_attacks);
                    knight_mobility -= engine.knight_mobility_bonus[num_set_bits(attacked_squares)];
                }

                // bishop
                uint64_t white_bishops = position.piece_maps[BISHOP][WHITE];
                if (num_set_bits(white_bishops) >= 2) {
                    bishop_pair += 1;
                }
                while (white_bishops) {
                    int square = least_set_bit(white_bishops);
                    white_bishops &= white_bishops - 1;

                    uint64_t attacked_squares = position.get_bishop_moves(square, all_pieces) & (~position.white_pieces);
                    bishop_mobility += engine.bishop_mobility_bonus[num_set_bits(attacked_squares)];
                }
                uint64_t black_bishops = position.piece_maps[BISHOP][BLACK];
                if (num_set_bits(black_bishops) >= 2) {
                    bishop_pair -= 1;
                }
                while (black_bishops) {
                    int square = least_set_bit(black_bishops);
                    black_bishops &= black_bishops - 1;

                    uint64_t attacked_squares = position.get_bishop_moves(square, all_pieces) & (~position.black_pieces);
                    bishop_mobility -= engine.bishop_mobility_bonus[num_set_bits(attacked_squares)];
                }

                // rook
                uint64_t white_rooks = position.piece_maps[ROOK][WHITE];
                while (white_rooks) {
                    int square = least_set_bit(white_rooks);
                    white_rooks &= white_rooks - 1;

                    uint64_t attacked_squares = position.get_rook_moves(square, all_pieces) & (~position.white_pieces) & (~black_pawn_attacks);
                    rook_mobility += engine.rook_mobility_bonus[num_set_bits(attacked_squares)];
                }
                uint64_t black_rooks = position.piece_maps[ROOK][BLACK];
                while (black_rooks) {
                    int square = least_set_bit(black_rooks);
                    black_rooks &= black_rooks - 1;

                    uint64_t attacked_squares = position.get_rook_moves(square, all_pieces) & (~position.black_pieces) & (~white_pawn_attacks);
                    rook_mobility -= engine.rook_mobility_bonus[num_set_bits(attacked_squares)];
                }

                // queen
                uint64_t white_queen = position.piece_maps[QUEEN][WHITE];
                while (white_queen) {
                    int square = least_set_bit(white_queen);
                    white_queen &= white_queen - 1;

                    uint64_t attacked_squares = (position.get_rook_moves(square, all_pieces) | 
                                                position.get_bishop_moves(square, all_pieces)) & (~position.white_pieces) & (~black_pawn_attacks);
                    queen_mobility += engine.queen_mobility_bonus[num_set_bits(attacked_squares)];
                }
                uint64_t black_queen = position.piece_maps[QUEEN][BLACK];
                while (black_queen) {
                    int square = least_set_bit(black_queen);
                    black_queen &= black_queen - 1;

                    uint64_t attacked_squares = (position.get_rook_moves(square, all_pieces) | 
                                                position.get_bishop_moves(square, all_pieces)) & (~position.black_pieces) & (~white_pawn_attacks);
                    queen_mobility -= engine.queen_mobility_bonus[num_set_bits(attacked_squares)];
                }

                int white_piece_count = num_set_bits(position.white_pieces & ~position.piece_maps[PAWN][WHITE]);
                int black_piece_count = num_set_bits(position.black_pieces & ~position.piece_maps[PAWN][BLACK]);
                piece_difference = white_piece_count - black_piece_count;


                int eval = material * engine.MATERIAL_WEIGHT + isolated_pawn * engine.ISOLATED_PAWN_WEIGHT +
                    doubled_pawn * engine.DOUBLED_PAWN_WEIGHT + backwards_pawn * engine.BACKWARDS_PAWN_WEIGHT +
                    passed_pawn * engine.PASSED_PAWN_WEIGHT + knight_mobility * engine.KNIGHT_MOBILITY_WEIGHT +
                    bishop_pair * engine.BISHOP_PAIR_WEIGHT + bishop_mobility * engine.BISHOP_MOBILITY_WEIGHT +
                    rook_mobility * engine.ROOK_MOBILITY_WEIGHT + queen_mobility * engine.QUEEN_MOBILITY_WEIGHT +
                    king_position * engine.KING_POSITION_WEIGHT + piece_difference * engine.PIECE_DIFFERENCE_WEIGHT;
                
                eval /= drawness + 1;
                
                if (engine.playing == "white") {
                    return eval;
                } else {
                    return -eval;
                }

            } else {
                // almost no pieces left - look for checkmate

                int drawness = 0;
                // rook against minor piece
                if (position.white_material == Position::values[ROOK] && position.piece_maps[ROOK][WHITE] &&
                (position.black_material == Position::values[KNIGHT] || position.black_material == Position::values[BISHOP]) &&
                    !position.piece_maps[PAWN][BLACK]) {

                    drawness = 100;
                }
                if (position.black_material == Position::values[ROOK] && position.piece_maps[ROOK][BLACK] &&
                (position.white_material == Position::values[KNIGHT] || position.white_material == Position::values[BISHOP]) &&
                    !position.piece_maps[PAWN][WHITE]) {

                    drawness = 100;
                }

                // todo: bishop and one pawn in wrong corner

                // todo: one pawn with king opposition


                int king_position = 0;

                if (engine.playing == "white") {
                    if (position.white_material > position.black_material) {
                        // force king to corner
                        if (position.king_positions[2] <= 3) {
                            king_position -= position.king_positions[2];
                        } else {
                            king_position -= 7 - position.king_positions[2];
                        }
                        if (position.king_positions[3] <= 3) {
                            king_position -= position.king_positions[3];
                        } else {
                            king_position -= 7 - position.king_positions[3];
                        }

                        // bring king closer to opp king
                        king_position -= std::abs(position.king_positions[0] - position.king_positions[2]) +
                                         std::abs(position.king_positions[1] - position.king_positions[3]);
                    } else {
                        
                    }
                } else {
                    if (position.black_material > position.white_material) {
                        // force king to corner
                        if (position.king_positions[0] <= 3) {
                            king_position -= position.king_positions[0];
                        } else {
                            king_position -= 7 - position.king_positions[0];
                        }
                        if (position.king_positions[1] <= 3) {
                            king_position -= position.king_positions[1];
                        } else {
                            king_position -= 7 - position.king_positions[1];
                        }

                        // bring king closer to opp king
                        king_position -= std::abs(position.king_positions[0] - position.king_positions[2]) +
                                        std::abs(position.king_positions[1] - position.king_positions[3]);
                    } else {

                    }
                }

                float eval;
                if (engine.playing == "white") {
                    eval = material * engine.MATERIAL_WEIGHT + king_position * engine.KING_POSITION_WEIGHT;
                } else {
                    eval = -material * engine.MATERIAL_WEIGHT + king_position * engine.KING_POSITION_WEIGHT;
                }

                eval /= drawness + 1;
                return eval;
            }
        }
    }
}
