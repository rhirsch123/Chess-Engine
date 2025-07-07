import pygame
import time
import os

FIFO_C_TO_P = "cpp_to_python"
FIFO_P_TO_C = "python_to_cpp"

if not os.path.exists(FIFO_C_TO_P):
    os.mkfifo(FIFO_C_TO_P)
if not os.path.exists(FIFO_P_TO_C):
    os.mkfifo(FIFO_P_TO_C)

WIDTH, HEIGHT = 600, 600
SQUARE_SIZE = WIDTH // 8
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
HIGHLIGHT_COLOR = (200, 200, 0)
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

board = []
with open(FIFO_P_TO_C, 'w') as fifo:
    fifo.write("get position\n")

with open(FIFO_C_TO_P, 'r') as fifo:
    for r in range(8):
        row = []
        for c in range(8):
            piece = fifo.readline().strip()
            if (piece == '0'):
                row.append(None)
            elif (piece == '1'):
                row.append('wP')
            elif (piece == '2'):
                row.append('wN')
            elif (piece == '3'):
                row.append('wB')
            elif (piece == '4'):
                row.append('wR')
            elif (piece == '5'):
                row.append('wQ')
            elif (piece == '6'):
                row.append('wK')
            elif (piece == '7'):
                row.append('bP')
            elif (piece == '8'):
                row.append('bN')
            elif (piece == '9'):
                row.append('bB')
            elif (piece == '10'):
                row.append('bR')
            elif (piece == '11'):
                row.append('bQ')
            elif (piece == '12'):
                row.append('bK')
        
        board.append(row)

    turn = fifo.readline().strip()

selected_square = None
game_end = False

pieces = {}
piece_names = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'wP']
for name in piece_names:
    pieces[name] = pygame.transform.scale(pygame.image.load(f"../icons/{name}.png"), (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(board, playing='white'):
    if playing == 'white':
        for row in range(8):
            for col in range(8):
                color = WHITE if (row + col) % 2 == 0 else BROWN
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

                if selected_square == (row, col):
                    pygame.draw.rect(screen, HIGHLIGHT_COLOR, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

                piece = board[row][col]
                if piece:
                    screen.blit(pieces[piece], (col * SQUARE_SIZE, row * SQUARE_SIZE))
    elif playing == 'black':
        for row in range(8):
            for col in range(8):
                flipped_row = 7 - row
                flipped_col = 7 - col
                color = WHITE if (flipped_row + flipped_col) % 2 == 0 else BROWN
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

                if selected_square == (flipped_row, flipped_col):
                    pygame.draw.rect(screen, HIGHLIGHT_COLOR, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

                piece = board[flipped_row][flipped_col]
                if piece:
                    screen.blit(pieces[piece], (col * SQUARE_SIZE, row * SQUARE_SIZE))


class Move:
    def __init__(self, start, end, promote_to=None):
        self.start = start
        self.end = end
        self.promote_to = promote_to

    def __eq__(self, other):
        if not other:
            return False
        return self.start == other.start and self.end == other.end and self.promote_to == other.promote_to
    
    def __str__(self):
        s = f'{self.start[0]}{self.start[1]}{self.end[0]}{self.end[1]}'
        if self.promote_to:
            s += self.promote_to
        else:
            s += 'X'
        return s

    def is_promotion(self):
        r, c = self.start
        return (self.end[0] == 0 and board[r][c] == 'wP') or (self.end[0] == 7 and board[r][c] == 'bP')


def move_from_str(move):
    # format: <row1><col1><row2><col2><promotion or 'X'>
    promote_to = move[4]
    if (promote_to == 'X'):
        promote_to = None
    return Move((int(move[0]), int(move[1])), (int(move[2]), int(move[3])), promote_to)

def update_board(move):
    global turn
    turn = "black" if turn == "white" else "white"

    start = move.start
    end = move.end
    piece = board[start[0]][start[1]]

    if move.promote_to:
        board[end[0]][end[1]] = piece[0] + move.promote_to
    elif piece[1] == 'K' and abs(start[1] - end[1]) > 1:
        # check castle
        board[end[0]][end[1]] = piece
        direction = start[1] - end[1]
        if direction > 0:
            rook = board[end[0]][0]
            board[end[0]][end[1] + 1] = rook
            board[end[0]][0] = None
        else:
            rook = board[end[0]][7]
            board[end[0]][end[1] - 1] = rook
            board[end[0]][7] = None
    elif piece[1] == 'P' and start[1] != end[1] and not board[end[0]][end[1]]:
        # en passant
        board[start[0]][end[1]] = None
        board[end[0]][end[1]] = piece
    else:
        board[end[0]][end[1]] = piece

    board[start[0]][start[1]] = None


def get_playing():
    with open(FIFO_P_TO_C, 'w') as fifo:
        fifo.write("get playing\n")
    
    with open(FIFO_C_TO_P, "r") as fifo:
        playing = fifo.readline().strip()
    return playing

def get_legal_moves():
    with open(FIFO_P_TO_C, 'w') as fifo:
        fifo.write("get legal moves\n")
    
    legal_moves = []
    with open(FIFO_C_TO_P, "r") as fifo:
        move = fifo.readline().strip()
        while (move):
            legal_moves.append(move_from_str(move))
            move = fifo.readline().strip()
    return legal_moves


def send_move(move):
    with open(FIFO_P_TO_C, 'w') as fifo:
        fifo.write("update\n")
        fifo.write(str(move))

def get_move():
    with open(FIFO_P_TO_C, 'w') as fifo:
        fifo.write("make move\n")
    
    with open(FIFO_C_TO_P, "r") as fifo:
        move = fifo.readline().strip()
    return move_from_str(move)

def get_terminal_state():
    with open(FIFO_P_TO_C, 'w') as fifo:
        fifo.write("get terminal state\n")
    
    with open(FIFO_C_TO_P, "r") as fifo:
        terminal = fifo.readline().strip()
    return terminal


def handle_click(pos, playing='white'):
    global board, selected_square, game_end
    
    if playing == 'white':
        col, row = pos[0] // SQUARE_SIZE, pos[1] // SQUARE_SIZE
    elif playing == 'black':
        col, row = 7 - (pos[0]) // SQUARE_SIZE, 7 - (pos[1]) // SQUARE_SIZE
    clicked_piece = board[row][col]

    if selected_square == (row, col):
        selected_square = None
    elif selected_square == None and clicked_piece:
        selected_square = (row, col)
    elif selected_square:
        legal_moves = get_legal_moves()
        move = Move(selected_square, (row, col))

        if move.is_promotion():
            move.promote_to = 'Q'
            if move in legal_moves:
                piece = input('Promote to (Q, R, B, N): ')
                move.promote_to = piece.upper()

        if move in legal_moves:
            update_board(move)
            send_move(move)
            selected_square = None
            time.sleep(0.2)
            terminal = get_terminal_state()
            if terminal and not game_end:
                print(terminal)
                game_end = True
            
        elif clicked_piece:
            selected_square = (row, col)


def main():
    global board, game_end
    playing = get_playing()
    engine_playing = 'black' if playing == 'white' else 'white'

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                with open(FIFO_P_TO_C, 'w') as fifo:
                    fifo.write("exit\n")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(pygame.mouse.get_pos(), playing)
                            
        draw_board(board, playing)
        pygame.display.flip()

        if turn == engine_playing and not game_end:
            engine_move = get_move()
            if engine_move:
                update_board(engine_move)
            
            time.sleep(0.2)
            terminal = get_terminal_state()
            if terminal and not game_end:
                print(terminal)
                game_end = True
        

    pygame.quit()

if __name__ == "__main__":
    main()
