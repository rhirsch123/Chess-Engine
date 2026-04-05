CC = g++
CFLAGS = -std=c++17 -O3 -flto -funroll-loops -march=native

SRCS = position.cc engine.cc zobrist.cc bitboards.cc movegen.cc movepick.cc polyglot.cc nnue/nnue.cc
OBJS = $(SRCS:.cc=.o)

TARGETS = main uci perft bench

all: $(TARGETS)

main: main.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

uci: uci.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

perft: perft.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

bench: bench.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cc
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o */*.o $(TARGETS)