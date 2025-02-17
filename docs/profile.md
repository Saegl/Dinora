# Profiling

Quick example on how to do profiling with `cProfile` and `snakeviz`

```python
import cProfile
engine = Engine(...)

with cProfile.Profile() as profiler:
    move = engine.get_best_move(chess.Board(), MoveTime(1000))
    profiler.dump_stats("mcts.prof")
```

Then to open flamegraph:

```bash
snakeviz mcts.prof
```
