import pygame
import time
import os

FIFO_C_TO_P = "cpp_to_python"
FIFO_P_TO_C = "python_to_cpp"

if not os.path.exists(FIFO_C_TO_P):
    os.mkfifo(FIFO_C_TO_P)
if not os.path.exists(FIFO_P_TO_C):
    os.mkfifo(FIFO_P_TO_C)

WIDTH = 600
SQUARE_SIZE = WIDTH // 8
HEIGHT = WIDTH + 2 * SQUARE_SIZE

LIGHT = (220, 190, 150)
DARK = (160, 120, 85)
LIGHT_HIGHLIGHT = (240, 200, 120)
DARK_HIGHLIGHT = (175, 135, 30)

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
last_move = ((-1, -1), (-1, -1))
getting_promotion = False
promotion_move = None
game_end = False

with open(FIFO_P_TO_C, 'w') as fifo:
    fifo.write("get time\n")

with open(FIFO_C_TO_P, 'r') as fifo:
    minutes = float(fifo.readline().strip())
    increment = float(fifo.readline().strip())
player_seconds = minutes * 60
engine_seconds = minutes * 60
timed_game = minutes > 0

with open(FIFO_P_TO_C, 'w') as fifo:
    fifo.write("get playing\n")
with open(FIFO_C_TO_P, 'r') as fifo:
    playing = fifo.readline().strip()


pieces = {}
piece_names = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'wP']
for name in piece_names:
    pieces[name] = pygame.transform.scale(pygame.image.load(f"../icons/{name}.png"), (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(board):
    if timed_game:
        pygame.draw.rect(screen, (0, 0, 0), (0, 0, WIDTH, SQUARE_SIZE))
        pygame.draw.rect(screen, (0, 0, 0), (0, SQUARE_SIZE * 9, WIDTH, SQUARE_SIZE))

        font = pygame.font.Font(None, 36)
        engine_str = f"{int(engine_seconds) // 60}:{(int(engine_seconds) % 60):02}"
        text = font.render(engine_str, True, (255, 255, 255))
        text_rect = text.get_rect(center=(WIDTH // 2, SQUARE_SIZE / 2))
        screen.blit(text, text_rect)

        font = pygame.font.Font(None, 36)
        player_str = f"{int(player_seconds) // 60}:{(int(player_seconds) % 60):02}"
        text = font.render(player_str, True, (255, 255, 255))
        text_rect = text.get_rect(center=(WIDTH // 2, SQUARE_SIZE * 9.5))
        screen.blit(text, text_rect)
    
    if playing == 'white':
        for row in range(8):
            for col in range(8):
                highlight = selected_square == (row, col) or (row, col) in last_move
                if (row + col) % 2 == 0:
                    color = LIGHT_HIGHLIGHT if highlight else LIGHT
                else:
                    color = DARK_HIGHLIGHT if highlight else DARK

                row_pos = (row + 1) * SQUARE_SIZE
                col_pos = col * SQUARE_SIZE

                pygame.draw.rect(screen, color, (col_pos, row_pos, SQUARE_SIZE, SQUARE_SIZE))

                piece = board[row][col]
                if piece:
                    screen.blit(pieces[piece], (col_pos, row_pos))
    elif playing == 'black':
        for row in range(8):
            for col in range(8):
                flipped_row = 7 - row
                flipped_col = 7 - col

                highlight = selected_square == (flipped_row, flipped_col) or (flipped_row, flipped_col) in last_move
                if (flipped_row + flipped_col) % 2 == 0:
                    color = LIGHT_HIGHLIGHT if highlight else LIGHT
                else:
                    color = DARK_HIGHLIGHT if highlight else DARK

                row_pos = (row + 1) * SQUARE_SIZE
                col_pos = col * SQUARE_SIZE

                pygame.draw.rect(screen, color, (col_pos, row_pos, SQUARE_SIZE, SQUARE_SIZE))

                piece = board[flipped_row][flipped_col]
                if piece:
                    screen.blit(pieces[piece], (col_pos, row_pos))


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
    global turn, player_seconds, engine_seconds

    if turn == playing:
        player_seconds += increment
    else:
        engine_seconds += increment

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


def get_legal_moves():
    with open(FIFO_P_TO_C, 'w') as fifo:
        fifo.write("get legal moves\n")
    
    legal_moves = []
    with open(FIFO_C_TO_P, 'r') as fifo:
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
    
    with open(FIFO_C_TO_P, 'r') as fifo:
        move = fifo.readline().strip()
    return move_from_str(move)

def get_terminal_state():
    with open(FIFO_P_TO_C, 'w') as fifo:
        fifo.write("get terminal state\n")
    
    with open(FIFO_C_TO_P, 'r') as fifo:
        terminal = fifo.readline().strip()
    return terminal


def draw_promotion():
    pygame.draw.rect(screen, (255, 255, 255), (2 * SQUARE_SIZE, 3 * SQUARE_SIZE, SQUARE_SIZE * 4, SQUARE_SIZE * 4))
    queen = pygame.transform.scale(pygame.image.load(f"../icons/bQ.png"), (SQUARE_SIZE * 2, SQUARE_SIZE * 2))
    rook = pygame.transform.scale(pygame.image.load(f"../icons/bR.png"), (SQUARE_SIZE * 2, SQUARE_SIZE * 2))
    bishop = pygame.transform.scale(pygame.image.load(f"../icons/bB.png"), (SQUARE_SIZE * 2, SQUARE_SIZE * 2))
    knight = pygame.transform.scale(pygame.image.load(f"../icons/bN.png"), (SQUARE_SIZE * 2, SQUARE_SIZE * 2))
    screen.blit(queen, (2 * SQUARE_SIZE, 3 * SQUARE_SIZE))
    screen.blit(rook, (4 * SQUARE_SIZE, 3 * SQUARE_SIZE))
    screen.blit(bishop, (2 * SQUARE_SIZE, 5 * SQUARE_SIZE))
    screen.blit(knight, (4 * SQUARE_SIZE, 5 * SQUARE_SIZE))


def handle_click(pos):
    global board, selected_square, last_move, getting_promotion, promotion_move, game_end
    
    if playing == 'white':
        col, row = pos[0] // SQUARE_SIZE, pos[1] // SQUARE_SIZE
        row -= 1
    elif playing == 'black':
        col, row = 7 - (pos[0]) // SQUARE_SIZE, 7 - (pos[1]) // SQUARE_SIZE
        row += 1

    if row < 0 or row > 7:
        return
    
    if getting_promotion:
        if row >= 2 and row <= 5 and col >= 2 and col <= 5:
            getting_promotion = False
            if (playing == 'white' and row < 4 and col < 4) or (playing == 'black' and row > 3 and col > 3):
                promotion = 'Q'
            elif (playing == 'white' and row < 4 and col > 3) or (playing == 'black' and row > 3 and col < 4):
                promotion = 'R'
            elif (playing == 'white' and row > 3 and col < 4) or (playing == 'black' and row < 4 and col > 3):
                promotion = 'B'
            else:
                promotion = 'N'
            
            promotion_move.promote_to = promotion
            update_board(promotion_move)
            send_move(promotion_move)
            last_move = (selected_square, promotion_move.end)
            selected_square = None
            time.sleep(0.2)
            terminal = get_terminal_state()
            if terminal and not game_end:
                print(terminal)
                game_end = True
        return

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
                getting_promotion = True
                promotion_move = move
                return

        if move in legal_moves:
            update_board(move)
            send_move(move)
            last_move = (selected_square, (row, col))
            selected_square = None
            time.sleep(0.2)
            terminal = get_terminal_state()
            if terminal and not game_end:
                print(terminal)
                game_end = True
            
        elif clicked_piece:
            selected_square = (row, col)


def main():
    global board, last_move, player_seconds, engine_seconds, game_end

    engine_playing = 'black' if playing == 'white' else 'white'
    player_timer_start = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                with open(FIFO_P_TO_C, 'w') as fifo:
                    fifo.write("exit\n")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(pygame.mouse.get_pos())
                            
        draw_board(board)
        if getting_promotion:
            draw_promotion()
        pygame.display.flip()

        if not game_end:
            if timed_game and turn == playing:
                if not player_timer_start:
                    player_start = time.time()
                    player_timer_start = True
                else:
                    player_timer_start = True
                    player_seconds -= time.time() - player_start
                    player_start = time.time()
                
            elif turn == engine_playing:
                if timed_game:
                    player_timer_start = False

                    engine_start = time.time()

                engine_move = get_move()
                
                if timed_game:
                    engine_elapsed = time.time() - engine_start
                    engine_seconds -= engine_elapsed

                if engine_move:
                    update_board(engine_move)
                
                last_move = (engine_move.start, engine_move.end)
                
                time.sleep(0.2)
                terminal = get_terminal_state()
                if terminal and not game_end:
                    print(terminal)
                    game_end = True
        

    pygame.quit()

if __name__ == "__main__":
    main()