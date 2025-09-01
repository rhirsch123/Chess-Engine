CC = g++
CFLAGS = -std=c++17 -g -O3 -flto

SRCS = position.cc move_generator.cc engine.cc evaluation.cc zobrist.cc bitboards.cc polyglot.cc nnue/nnue.cc
OBJS = $(SRCS:.cc=.o)

TARGETS = main test finetune uci

all: $(TARGETS)

main: main.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

test: test.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

finetune: finetune.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

uci: uci.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cc
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o */*.o $(TARGETS)