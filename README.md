# ai-wargame

##Overview
This project is a 2-player strategy game built with Python, where an attacker and a defender compete on a 5x5 board, each with different unit types. The code includes concepts such as adversarial search, minimax algorithm, alpha-beta pruning, and heuristics to create a computer player capable of making its strategic moves. OOP principles were used to handle features such as keeping track of unit types, actions taken and restrictions, and units' health levels.  

## Run
This program runs in the command-line. 

1. Move to the project's directory:
```bash
cd ai-wargame
```
2. Run the game:
```bash
python3 src/ai_wargame.py
```
If you would like to change some game parameters, you can view the different types
of arguments you would like to change with the command:
```bash
python3 src/ai_wargame.py -h
```
### Options
The different changes you can make:
- Help: -h, --help
- Max search time (default: None): --max_time MAX_TIME
- Max turns before end of game (default: None): --max_turns MAX_TURNS
- Alpha-beta on/off (default: None): --alpha_beta ALPHA_BETA
- Game type: auto|attacker|defender|manual (default:manual): --game_type {auto,attacker,defender,manual}



