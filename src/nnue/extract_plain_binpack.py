import chess
import numpy as np

# extract binary training data from the .plain version of a binpack file

def encode_board(board: chess.Board):
	vec = np.zeros(768, dtype=np.int8)
	for square, piece in board.piece_map().items():
		color_offset = 0 if piece.color == chess.WHITE else 6
		s = (7 - square // 8) * 8 + square % 8
		index = s * 12 + (piece.piece_type - 1) + color_offset
		vec[index] = 1
	
	return vec


pos_file = "positions.bin"
eval_file = "evals.bin"

# clear
with open(pos_file, 'w'):
    pass
with open(eval_file, 'w'):
    pass

input_file = "binpack.plain"

positions = []
evals = []

num_positions = 0

with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("fen "):
            fen = line[4:]
            if fen == "8/8/8/8/8/8/8/8 w - - 0 1":
                break
        elif line.startswith("move "):
            move = line[5:]
        elif line.startswith("score "):
            score = float(line[6:])
        elif line.startswith("ply "):
            ply = int(line[4:])
        elif line.startswith("result "):
            result = int(line[7:])
        elif line == "e":
            board = chess.Board(fen=fen)
            # avoid openings, absolute draws (could be threefold, 50 move rule), tactical positions
            if ply > 5 and score != 0 and not board.is_check() and not board.is_capture(chess.Move.from_uci(move)):
                num_positions += 1

                positions.append(encode_board(board))

                if board.turn == chess.BLACK:
                    score = -score
                    result = -result
                score = min(score, 3000)
                score = max(score, -3000)
                evals.append(score)

                if num_positions % 10000 == 0:
                     print(f'position: {num_positions}')
                
                if num_positions % 100000 == 0:
                    positions = np.array(positions, dtype=np.int8)
                    evals = np.array(evals, dtype=np.int16)

                    # shuffle
                    perm = np.random.permutation(100000)
                    positions = positions[perm]
                    evals = evals[perm]

                    with open(pos_file, "ab") as f:
                        positions.tofile(f)
                    with open(eval_file, "ab") as f:
                        evals.tofile(f)

                    positions = []
                    evals = []

print(f'positions: {num_positions}')

positions = np.array(positions, dtype=np.int8)
evals = np.array(evals, dtype=np.int16)

# shuffle
perm = np.random.permutation(num_positions % 100000)
positions = positions[perm]
evals = evals[perm]

with open(pos_file, "ab") as f:
    positions.tofile(f)
with open(eval_file, "ab") as f:
    evals.tofile(f)