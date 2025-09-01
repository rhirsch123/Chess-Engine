import chess
import chess.pgn
import re
import numpy as np

# extract binary training data from a pgn commented with evaluations

eval_pattern = re.compile(r"([+-]?\d+(?:\.\d+)?)")

def encode_board(board: chess.Board):
	vec = np.zeros(768, dtype=np.int8)
	for square, piece in board.piece_map().items():
		color_offset = 0 if piece.color == chess.WHITE else 6
		s = (7 - square // 8) * 8 + square % 8
		index = s * 12 + (piece.piece_type - 1) + color_offset
		vec[index] = 1
	
	return vec


positions = []
evals = []

pgn = open('games.pgn')

num_games = 0
num_positions = 0
while True:
	if num_games % 1000 == 0:
		print(f'game: {num_games}')

	game = chess.pgn.read_game(pgn)
	if not game:
		break

	board = game.board()

	nodes = list(game.mainline())
	moves = list(game.mainline_moves())
	n = len(moves)
	for i in range(n):
		move = moves[i]
		board.push(move)

		# train on quiet positions
		if i <= 5 or board.is_check() or i == n - 1 or moves[i + 1].promotion or board.is_capture(moves[i + 1]):
			continue

		evaluation = None
		node = nodes[i]
		if node.comment:
			m = eval_pattern.match(node.comment.strip())
			if m:
				evaluation = m.group(1)

				evaluation = int(float(evaluation) * 100)
				if board.turn == chess.WHITE:
					evaluation = -evaluation
				evaluation = min(evaluation, 3000)
				evaluation = max(evaluation, -3000)

		if evaluation != None:
			num_positions += 1

			positions.append(encode_board(board))
			evals.append(evaluation)

	
	num_games += 1

print(f'games: {num_games}')
print(f'positions: {num_positions}')


positions = np.array(positions, dtype=np.int8)
evals = np.array(evals, dtype=np.int16)

# shuffle
perm = np.random.permutation(num_positions)
datpositionsa = positions[perm]
evals = evals[perm]

positions.tofile('positions.bin')
evals.tofile('evals.bin')