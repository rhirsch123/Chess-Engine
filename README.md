# Chess Engine

### Usage
To play engine with GUI:  
pip install pygame  
Optional: edit main.cc file to play white/black, set time format, set custum position, etc.  
make  
./main  

### Strength
~3200 ELO

### Features
- [NNUE](https://www.chessprogramming.org/NNUE) (Efficiently Updatable Neural Network) used to evaluate positions
  - Trained on data from [Lc0](https://github.com/LeelaChessZero/lc0) (deep NN based engine)
- [Bitboard](https://www.chessprogramming.org/Bitboards) representation (including magic bitboards for sliding pieces) for fast move generation
- [Negamax](https://www.chessprogramming.org/Negamax) search with [alpha-beta pruning](https://www.chessprogramming.org/Alpha-Beta) and [iterative deepening](https://www.chessprogramming.org/Iterative_Deepening).
- [Move ordering](https://www.chessprogramming.org/Move_Ordering) heuristics: history, hash move, MVV-LVA, etc.
- Search heuristics: [late move reduction](https://www.chessprogramming.org/Late_Move_Reductions), [null move pruning](https://www.chessprogramming.org/Null_Move_Pruning), [futility pruning](https://www.chessprogramming.org/Futility_Pruning), etc.
- [Zobrist hashing](https://www.chessprogramming.org/Zobrist_Hashing)
- [Transposition table](https://www.chessprogramming.org/Transposition_Table): store information learned about a position to avoid re-searching it.
- [Opening book](https://www.chessprogramming.org/Opening_Book) support: play random opening from database so it is not deterministic.
- GUI: run in python
- UCI support
