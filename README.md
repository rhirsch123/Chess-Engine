# Chess Engine

### Usage
To play engine with built-in GUI:  
pip install pygame  
make  
./main

Or install other tools and use the uci executable

### Strength
~3400 ELO

### Features
- NNUE (Efficiently Updatable Neural Network) used to evaluate positions
  - Optimized inference with SIMD instructions
  - Trained in pytorch on data from [Lc0](https://github.com/LeelaChessZero/lc0)
- Bitboard representation with magic bitboards for fast move generation
- Negamax search with alpha-beta pruning and iterative deepening
- Move ordering heuristics: hash move, history, etc.
- Search heuristics: late move reduction, null move pruning, futility pruning, etc.
- Zobrist hashing
- Transposition table: store information learned about a position to avoid re-searching it.
- GUI: run in python
- UCI support
