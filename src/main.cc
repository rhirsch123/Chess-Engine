#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "position.hh"
#include "movegen.hh"
#include "move.hh"
#include "engine.hh"

#define FIFO_C_TO_P "cpp_to_python"
#define FIFO_P_TO_C "python_to_cpp"


static int get_int(char c) {
    return c - '0';
}

static int char_to_piece(char c) {
        if (c == 'Q') {
            return QUEEN;
        } else if (c == 'R') {
            return ROOK;
        } else if (c == 'N') {
            return KNIGHT;
        } else if (c == 'B') {
            return BISHOP; 
        } else if (c == 'P') {
            return PAWN;
        } else if (c == 'K') {
            return KING;
        }
        return -1;
    }

Move move_from_str(std::string move) {
    // format: <row1><col1><row2><col2><promotion or 'X'>
    int r1 = get_int(move[0]);
    int c1 = get_int(move[1]);
    int r2 = get_int(move[2]);
    int c2 = get_int(move[3]);

    char promote_to = move[4];
    int promotion_piece;
    if (promote_to == 'X') {
        promotion_piece = 0;
    } else {
        promotion_piece = char_to_piece(promote_to);
    }
    return Move(r1 * 8 + c1, r2 * 8 + c2, promotion_piece);
}

std::string read_from_pipe() {
    int fd = open(FIFO_P_TO_C, O_RDONLY);
    char buffer[128];
    int bytesRead = read(fd, buffer, sizeof(buffer) - 1);
    close(fd);
    if (bytesRead > 0) {
        buffer[bytesRead] = '\0';
        return std::string(buffer);
    }
    return "";
}

void write_to_pipe(std::string message) {
    int fd = open(FIFO_C_TO_P, O_WRONLY);
    write(fd, message.c_str(), message.size());
    close(fd);
}

// run graphics in python. communicate between processes with named pipes
int main() {
    std::string playing = "white";

    Position position;
    Engine engine;

    int pid = fork();
    if (pid < 0) {
        perror("fork failed");
        return 1;
    }

    if (pid == 0) { // child process
        execlp((char *) "python3", (char *) "python3", (char *) "display.py", NULL);
        perror("execlp failed");
        exit(1);
    }

    // parent process
    mkfifo(FIFO_C_TO_P, 0666);
    mkfifo(FIFO_P_TO_C, 0666);
    while (true) {
        std::string input = read_from_pipe();
        int newline = input.find('\n');
        std::string command = input.substr(0, newline);
        std::string rest = input.substr(newline + 1);
        if (command == "get position") {
            std::ostringstream buffer;
            for (int i = 0; i < 64; i++) {
                buffer << (int) position.piece_on(i) << '\n';
            }
            buffer << (position.turn == WHITE ? "white\n" : "black\n");
            write_to_pipe(buffer.str());

        } else if (command == "get playing") {
            write_to_pipe(playing);

        } else if (command == "get legal moves") {
            std::ostringstream buffer;

            MoveList legals;
            get_legal_moves(position, &legals);
            for (Move move : legals) {
                buffer << move.toString() << '\n';
            }
            
            write_to_pipe(buffer.str());

        } else if (command == "update") {
            position.make_move(move_from_str(rest));

        } else if (command == "make move") {
            Move engine_move = engine.get_move(position);

            write_to_pipe(engine_move.toString());
            position.make_move(engine_move);
        
        } else if (command == "get terminal state") {
            usleep(10000);
            TerminalState state = position.get_terminal_state();
            if (state == WHITE_MATE) {
                write_to_pipe("white wins");
            } else if (state == BLACK_MATE) {
                write_to_pipe("black wins");
            } else if (state == STALEMATE) {
                write_to_pipe("stalemate");
            } else if (state == FIFTY_MOVE_RULE) {
                write_to_pipe("50 move rule");
            } else if (state == THREEFOLD_REPETITION) {
                write_to_pipe("threefold repetition");
            } else if (state == INSUFFICIENT_MATERIAL) {
                write_to_pipe("insufficient material");
            } else {
                write_to_pipe("");
            }

        } else if (command == "exit") {
            waitpid(pid, NULL, 0);
            unlink(FIFO_C_TO_P);
            unlink(FIFO_P_TO_C);
            return 0;
        }
    }
}