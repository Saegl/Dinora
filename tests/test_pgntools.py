from io import StringIO

from dinora.pgntools import (
    load_chess_games,
    load_game_states,
)

pgn_games = """
[Event "CCRL 40/4"]
[Site "CCRL"]
[Date "2006.05.24"]
[Round "1.1.8"]
[White "Aristarch 4.50"]
[Black "Chess Tiger 15"]
[Result "1/2-1/2"]

1. Nf3 c5 2. e4 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 e6 
6. Be3 a6 7. g4 h6 8. Qf3 Nbd7 9. Be2 Qc7 10. Qh3 d5 
11. exd5 Bb4 12. Bd2 Qb6 13. dxe6 Qxd4 14. exd7+ Bxd7 
15. O-O-O Qb6 16. Be3 Bc5 17. Qg3 Bxe3+ 18. fxe3 O-O 19. 
g5 hxg5 20. Qxg5 Rfe8 21. Rd3 Nh7 22. Nd5 Nxg5 23. Nxb6 Bc6 
24. Rg1 Nh3 25. Rg3 Nf2 26. Nxa8 Nxd3+ 27. cxd3 Rxa8 28. Rg5 g6 
29. Rc5 Kg7 30. h4 Rh8 31. h5 f5 32. d4 gxh5 33. d5 Be8 
34. Rc7+ Bf7 35. d6 Kf6 36. d7 Rd8 37. Rxb7 Ke7 38. Bxa6 Bd5 
39. Rb6 Rxd7 40. Be2 Rc7+ 41. Kd2 Bxa2 42. Rh6 Rd7+ 43. Ke1 Bb3 
44. Rh7+ Ke6 45. Rxd7 Kxd7 46. Bxh5 Ke6 47. Bd1 Ba2 48. Kf2 Ke5 
49. Kf3 Bf7 50. Bc2 Bd5+ 51. Kg3 Bc4 52. Bb1 Bd5 53. Bd3 Bb3 
54. Bb5 Ke4 55. Kf2 f4 56. Bc6+ Ke5 57. e4 1/2-1/2


[Event "CCRL 40/4"]
[Site "CCRL"]
[Date "2006.05.24"]
[Round "1.1.10"]
[White "Gandalf 6"]
[Black "Aristarch 4.50"]
[Result "1-0"]

1. Nf3 Nf6 2. c4 c5 3. Nc3 d5 4. cxd5 Nxd5 5. d4 e6 
6. e4 Nxc3 7. bxc3 cxd4 8. cxd4 Nc6 9. Rb1 Qa5+ 10. Bd2 Qxa2 
11. Bc3 Qa3 12. Rb3 Qe7 13. Bb5 Bd7 14. O-O a6 15. Qa1 f6 16. Bd3 Nd8 
17. d5 Qd6 18. Bb4 Qb6 19. Qb1 Bxb4 20. Rxb4 Qd6 21. Rb6 Qf4 
22. e5 fxe5 23. Re1 O-O 24. Bxh7+ Kh8 25. Re4 Qf6 26. Rh4 Rc8 
27. Rh5 Qf4 28. Bc2+ Kg8 29. dxe6 Bxe6 30. Rxe6 Rxc2 31. Qxc2 Nxe6 
32. Qh7+ Kf7 33. Rf5+ Ke7 34. Rxf4 Rxf4 35. Nxe5 Kd6 36. g3 Rd4 
37. f4 b5 38. Qg6 Rd5 39. Qe8 Rd1+ 40. Kf2 Nd4 41. Qg6+ Kc5 
42. Nd3+ Rxd3 43. Qxd3 b4 44. Qxa6 1-0

[Event "CCRL 40/4"]
[Site "CCRL"]
[Date "2006.05.25"]
[Round "1.1.19"]
[White "Spike 1.1"]
[Black "Aristarch 4.50"]
[Result "0-1"]

1. e4 e5 2. Nf3 d6 3. d4 Nf6 4. Nc3 exd4 5. Nxd4 Be7 6. Be2 O-O 7. O-O a6 
8. Bf4 Nbd7 9. Qd2 Ne5 10. Rad1 Re8 11. Bg3 h6 12. f4 Nc6 13. Nxc6 bxc6 
14. e5 Nd7 15. Kh1 Rb8 16. b3 d5 17. Na4 Nc5 18. Nxc5 Bxc5 
19. Qc3 Bb6 20. Qxc6 Bb7 21. Qa4 d4 22. c3 c5 23. f5 Qg5 
24. f6 gxf6 25. Rxf6 Rxe5 26. Rf2 Bc7 27. Bc4 Re7 28. Bxc7 Rxc7 
29. cxd4 cxd4 30. Rxd4 Kh8 31. Rd1 Qc5 32. Kg1 Bc6 33. Qxa6 Rb6 
34. Qxb6 Qxb6 35. Rd6 Qe3 36. Rd1 f5 37. Rd8+ Kh7 38. Bg8+ Kg7 
39. Bc4 f4 40. Rg8+ Kh7 41. Rd8 f3 42. g3 Be4 43. Rd1 h5 44. a3 Rg7 
45. Be6 h4 46. g4 Bc2 47. Rf1 Bd3 0-1
"""

# 56 full moves + 1 ply = 56 * 2 + 1 = 113
FIRST_GAME_PLIES = 113
# 43 full moves + 1 ply = 43 * 2 + 1 = 87
SECOND_GAME_PLIES = 87
# 47 full moves = 47 * 2 = 94
THIRD_GAME_PLIES = 94


def test_load_chess_games_count():
    handle = StringIO(pgn_games)
    games = list(load_chess_games(handle))

    assert len(games) == 3


def test_load_game_states_count():
    handle = StringIO(pgn_games)
    states = list(load_game_states(handle))

    sum_plies = FIRST_GAME_PLIES + SECOND_GAME_PLIES + THIRD_GAME_PLIES
    assert len(states) == sum_plies
