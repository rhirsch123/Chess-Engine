CC = g++
CFLAGS = -std=c++17 -g -O3 -flto
SRCS = main.cc position.cc move_generator.cc engine.cc evaluation.cc zobrist.cc bitboards.cc polyglot.cc
OBJS = $(SRCS:.cc=.o)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cc
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)