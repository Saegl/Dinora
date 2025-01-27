# Installation & Usage

**There is no simple installation process for end users!**  
This is mostly intended for python developers  
Only tested OS is linux, might work on Windows (after fixing torch cuda deps and
converting run.sh to run.bat, make PR if you want to fix windows)


You need [uv](https://docs.astral.sh/uv/)  
Clone repo  
Download blob of model weights from latest release into `models/` directory  
```wget https://github.com/Saegl/dinora/releases/download/v0.2.2/alphanet_classic.ckpt```  

run ```$ uv run python -m dinora```  
After dependencies downloaded this will run [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface)  
To check that everything works, in UCI repl you could type  
```uci``` - Will show available options  
```go``` - Will start infinite search and print moves, cancel with `Ctrl-C`  

Additionally, you can check that cuda works
```
$ uv run python
>>> import torch
>>> torch.cuda.is_available() # Should be true
```  

If you have any problems during installation open issue  

## GUI

Install any UCI compatible chess GUI program.
I personally recommend [CuteChess](https://cutechess.com/) from
[releases](https://github.com/cutechess/cutechess/releases)

To configure Dinora chess engine in CuteChess,  
browse Tools > Settings > Engines > Add a new engine >

```
Name: Dinora Chess Engine
Command: ./run.sh
Working Directory: <folder where you cloned the repo>
Protocol: UCI
```


