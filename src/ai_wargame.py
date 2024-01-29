from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import random
from typing import Tuple, Iterable, ClassVar

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000

class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Virus = 1
    Tech = 2
    Firewall = 3
    Program = 4


class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

    def __str__(self):
        if self == GameType.AttackerVsDefender:
            return "Attacker vs Defender"
        elif self == GameType.AttackerVsComp:
            return "Attacker vs Comp"
        elif self == GameType.CompVsDefender:
            return "Comp vs Defender"
        elif self == GameType.CompVsComp:
            return "Comp vs Comp"
        else:
            return super().__str__()


##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 1, 3],  # AI
        [9, 1, 6, 1, 6],  # Virus
        [1, 6, 1, 1, 1],  # Tech
        [1, 1, 1, 1, 1],  # Firewall
        [3, 3, 3, 1, 3],  # Program
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [0, 0, 0, 0, 0],  # Virus
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Firewall
        [0, 0, 0, 0, 0],  # Program
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount


##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 5:
            coord_char = "01234"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 5:
            coord_char = "ABCDE"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDE".find(s[0:1].upper())
            coord.col = "01234".find(s[1:2].lower())
            return coord
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDE".find(s[0:1].upper())
            coords.src.col = "01234".find(s[1:2].lower())
            coords.dst.row = "ABCDE".find(s[2:3].upper())
            coords.dst.col = "01234".find(s[3:4].lower())
            return coords
        else:
            return None

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))


##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth : int | None = 4
    min_depth : int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = True
    max_turns: int | None = 100
    heuristic: int | None = 0

##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth : dict[int,int] = field(default_factory=dict)
    total_seconds: float = 0.0

##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True
    h_score: int = -2000000000
    states_evaluated: int = 0

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.
        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords: CoordPair, bot) -> bool:
        """Validate a move expressed as a CoordPair."""

        # Checks if its a valid coord within the board
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(
                coords.dst):  # Not src/dst coord within the board
            return False

        # Checks if unit source is the player's unit
        if self.get(coords.src) is None or self.get(coords.src).player != self.next_player:
            return False

        # Checks if the destination coordinate is an adjacent coord
        if not self.is_adjacent(coords):
            if not bot:
                print("You can only move to adjacent coordinates.")
            return False
        
        #Uses the function is_in_combat() to check whether the unit is in combat or not. If unit is in combat, it cannot move
        if self.is_in_combat(coords) and ((self.board[coords.src.row][coords.src.col].type == UnitType.AI) or (self.board[coords.src.row][coords.src.col].type == UnitType.Firewall) or (self.board[coords.src.row][coords.src.col].type == UnitType.Program)):
            if not bot:
                print("This unit cannot be moved while engaged in combat")
            return False

        # Checks if unit is a Tech or Virus. If it is, unit can move up, down, right, left
        if (self.board[coords.src.row][coords.src.col].type == UnitType.Tech) or (
                self.board[coords.src.row][coords.src.col].type == UnitType.Virus):
            return True
        
        #Checks if unit from attacker is AI, Firewall or Program. If it is, unit can only move up or left.
        if (self.board[coords.src.row][coords.src.col].player == Player.Attacker and coords.src.col < coords.dst.col) or (self.board[coords.src.row][coords.src.col].player == Player.Attacker and coords.src.row < coords.dst.row):
            if not bot:
                print("The attacker’s AI, Firewall and Program can only move up or left")
            return False
        
        #Checks if unit from defender is AI, Firewall or Program. If it is, unit can only move down or right.
        if (self.board[coords.src.row][coords.src.col].player == Player.Defender and coords.src.col > coords.dst.col) or (self.board[coords.src.row][coords.src.col].player == Player.Defender and coords.src.row > coords.dst.row):
            if not bot:
                print("The defender’s AI, Firewall and Program can only move down or right")
            return False

        return True

    def is_valid_action(self, coords: CoordPair) -> bool:
        """Validate an action expressed as a CoordPair."""
        unit = self.get(coords.src)
        if unit is None or unit.player != self.next_player:
            return False

        elif not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False

        else:
            return True

    def is_in_combat(self, coords: CoordPair) -> bool:
        """Validate if unit is engaged in combat."""
        for i in coords.src.iter_adjacent():
            adjacent_unit = self.get(i)
            if (adjacent_unit is not None) and (adjacent_unit.player != self.get(coords.src).player):
                return True
        return False

    def perform_move(self, coords: CoordPair) -> Tuple[bool, str, int]:
        """Validate and perform a move expressed as a CoordPair."""
        # Flag: movement = 0
        if self.is_empty(coords.dst) and self.is_valid_move(coords, False):
                self.set(coords.dst,self.get(coords.src))
                self.set(coords.src,None)
                return (True,"", 0)
                    
        elif not self.is_empty(coords.dst) and self.is_valid_action(coords):
            (success, message, actionType) = self.action(coords)
            if success:
                return (True, message, actionType)
            else:
                return (False, message, -1)
        else:
            return (False, "", -1)

    def action(self, coords: CoordPair) -> Tuple[bool, str, int]:
        """Validate and perform an action expressed as a CoordPair."""
        # Flags: self-destruct = 1, repair = 2, attack = 3
        target = self.get(coords.dst)
        source = self.get(coords.src)
        if coords.dst == coords.src:
            self.self_destruct(coords)
            return (True, f"{target.to_string()} self-destructed", 1)
        elif self.is_adjacent(coords):
            if self.is_ally(coords.dst):
                if target.health == 9:
                    return (False, f"{target.to_string()} already has max health", -1)
                else:
                    if self.repair(coords) == 0:
                        return (False, f"{source.to_string()} cannot repair {target.to_string()}", -1)
                    return (True, f"{target.to_string()} was repaired by {source.to_string()}", 2)
            else:
                self.attack(coords)
                return (True, f"{target.to_string()} was attacked by {source.to_string()}", 3)
        else:
            return (False, "", -1)

    def self_destruct(self, coords: CoordPair):
        """Perform a self-destruct action."""
        unit = self.get(coords.dst)
        for coord in coords.dst.iter_range(1):
            if self.get(coord) is not None:
                self.get(coord).mod_health(-2)
                self.remove_dead(coord)
            else:
                continue
        unit.mod_health(-unit.health)
        self.remove_dead(coords.dst)

    def repair(self, coords: CoordPair) -> int:
        """Perform a repair action."""
        source = self.get(coords.src)
        target = self.get(coords.dst)
        amount = source.repair_amount(target)
        target.mod_health(amount)
        return amount

    def attack(self, coords: CoordPair):
        """Perform an attack action."""
        source = self.get(coords.src)
        target = self.get(coords.dst)
        target.mod_health(-source.damage_amount(target))
        source.mod_health(-target.damage_amount(source))
        self.remove_dead(coords.dst)
        self.remove_dead(coords.src)

    def is_adjacent(self, coords: CoordPair) -> bool:
        """Check if destination coordinate is adjacent to source coordinate."""
        for coord in coords.src.iter_adjacent():
            if coord == coords.dst:
                return True
            else:
                continue
        return False

    def is_ally(self, target: Coord) -> bool:
        """Check if target unit is an ally or not."""
        for coord, unit in self.player_units(self.next_player):
            if coord == target:
                return True;
        return False

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within our board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move."""
        while True:
            mv = self.read_move()
            (success, result, actionType) = self.perform_move(mv)
            if success:
                print(f"Player {self.next_player.name}: ", end='')
                print(result)
                self.next_turn()
                print("\n" + str(self))  # Print the board to the terminal
                move = (mv, actionType)
                return move
            else:
                print(result)
                print("The move is not valid! Try again.")
                move = (None, actionType)

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success, result, actionType) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ",end='')
                if result == "":
                    print(f"move from {mv.src.to_string()} to {mv.dst.to_string()}")
                else:
                    print(result)
                self.next_turn()
                print("\n" + str(self))
                move = (mv, actionType)
                return move
            else:
                return (None, actionType)
        
    
    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
        start_time = datetime.now()
        
        # Variables needed for minimax()
        alpha_beta = self.options.alpha_beta
        maxPlayer = True
        alpha = MIN_HEURISTIC_SCORE
        beta = MAX_HEURISTIC_SCORE
        
        depth = self.depth()

        for i in range(depth):
            self.stats.evaluations_per_depth[i + 1] = 0

        (score, move) = self.minimax(depth, maxPlayer, alpha_beta, alpha, beta)
        self.h_score = score
        
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds = elapsed_seconds

        print(f"Evals per depth: ",end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{depth}={self.stats.evaluations_per_depth[k]} ",end='')
            depth -= 1
        print()
        self.states_evaluated = sum(self.stats.evaluations_per_depth.values())
        return move
    
    def depth(self):
        depth = 0
        if self.turns_played < self.options.max_turns * 0.5:
            depth = self.options.min_depth
        else:
            depth = self.options.max_depth
        return depth
    
    def minimax(self, depth, maxPlayer, alpha_beta, alpha, beta) -> Tuple[int, CoordPair]:
        """ 
        Search function using minimax and alpha-beta pruning, which takes the search depth, if the current player is the maximizing player, if 
        alpa-beta pruning is on, alpha's and beta's initial values as inputs. It outputs the best move found by the search and its associated score.
        """
        if depth == 0 or self.is_finished(): # Base Case
            self.stats.evaluations_per_depth[depth] = self.stats.evaluations_per_depth.get(depth, 0) + 1
            return (self.evaluate(), None)
        
        moves = list(self.generate_moves()) # Generate a list of possible moves to search
        self.stats.evaluations_per_depth[depth] += len(moves)
        best_move = None

        if maxPlayer: # Maximizing player 
            best_score = MIN_HEURISTIC_SCORE
            for move in moves:
                new_game = self.apply_move(move) # Apply a move to generate a new game state to search
                if new_game == None:
                    continue
                (score, _ ) = new_game.minimax(depth - 1, False, alpha_beta, alpha, beta) # Recursive call to analyze next game state
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if alpha_beta and beta <= alpha: # Alpha-Beta pruning
                    break
        else: # Minimizing player
            best_score = MAX_HEURISTIC_SCORE
            for move in moves:
                new_game = self.apply_move(move)
                if new_game == None:
                    continue
                (score, _ ) = new_game.minimax(depth - 1, True, alpha_beta, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
                if alpha_beta and beta <= alpha:
                    break
        
        return (best_score, best_move) # Best move chosen after the search and its associated score

    def e0(self) -> int:
        """
        e0 heuristic function: the score is the difference in the number of units for each player. All units except AI are given an arbitrary weight
        of 3 and AI is given an arbitrary weight of 9999 (since it is the most important unit of the game). A negative result means a disadvantage 
        for the current player, a null result means the current player has no more no less advantage then the other player.
        """
        VP1 = TP1 = FP1 = PP1 = AIP1 = 0
        VP2 = TP2 = FP2 = PP2 = AIP2 = 0

        for (_, unit) in self.player_units(self.next_player):
            if unit.type == UnitType.Virus:
                VP1 += 1
            elif unit.type == UnitType.Tech:
                TP1 += 1
            elif unit.type == UnitType.Firewall:
                FP1 += 1
            elif unit.type == UnitType.Program:
                PP1 += 1
            elif unit.type == UnitType.AI:
                AIP1 += 1

        for (_, unit) in self.player_units(self.next_player.next()):
            if unit.type == UnitType.Virus:
                VP2 += 1
            elif unit.type == UnitType.Tech:
                TP2 += 1
            elif unit.type == UnitType.Firewall:
                FP2 += 1
            elif unit.type == UnitType.Program:
                PP2 += 1
            elif unit.type == UnitType.AI:
                AIP2 += 1

        heuristic_value = (3 * (VP1 + TP1 + FP1 + PP1) + 9999 * AIP1) - (3 * (VP2 + TP2 + FP2 + PP2) + 9999 * AIP2)

        return heuristic_value
    
    """
    This heuristic emphasizes the protection of the AI while also focusing on attacking the enemy AI. It is a slight refinement of e0.
    The AI is given a very high weight (10000) to make its survival the top priority. Firewalls are given a slightly higher weight (5) than other units because they absorb attacks well.
    Viruses have a weight of 2 because they pose a significant threat with their ability to destroy the AI in one attack.
    """

    def e1(self):
        weights = {
            UnitType.AI: 10000,
            UnitType.Virus: 3,
            UnitType.Tech: 1,
            UnitType.Firewall: 5,
            UnitType.Program: 1
        }

        value_p1 = 0  # attacker
        value_p2 = 0  # defender

        for row in self.board:
            for cell in row:
                if cell:
                    unit_type = cell.type  # get the unit type from the Unit instance
                    player = cell.player  # get the player from the Unit instance

                    if player == Player.Attacker:
                        value_p1 += weights[unit_type]
                    else:
                        value_p2 += weights[unit_type]

        return value_p1 - value_p2
    
    def e2(self) -> int:
        """
        e2 heuristic function: the score is determined by each unit's (belonging to the current player) advantage over an adjacent unit of the
        opposing player. It first checks if there is an enemy unit adjacent to the current unit, if so, then their combined difference in health 
        and in damage makes up for this unit's advantage. The total advantage of all current player's unit is returned. 
        """
        score = 0
        for (src, unit ) in self.player_units(self.next_player): # Go through each unit belonging to the current player
            for dst in src.iter_adjacent():
                target = self.get(dst)
                if target == None:
                    continue
                if not self.is_ally(dst):
                    score += (unit.health - target.health) + (unit.damage_amount(target) - target.damage_amount(unit)) # Compute the current unit's advantage
        return score # Total advantage of all current player's unit
    
    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        dumb_move = True
        move_candidates = list(self.generate_moves())
        while (dumb_move):
            random.shuffle(move_candidates)
            if move_candidates[0].src != move_candidates[0].dst:
                dumb_move = False
        
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)
    
    def evaluate(self):
        """Evaluate a game state based on a heuristic function and returns a score associated to it."""
        if self.options.heuristic == 0:
            e = self.e0()
        elif self. options.heuristic == 1:
            e = self.e1
        else:
            e = self.e2
        return e

    def generate_moves(self) -> Iterable[CoordPair]:
        """Generate all valid possible moves based on the calling game state."""
        move = CoordPair()
        for (src, unit ) in self.player_units(self.next_player):
            move.src = src
            # Moving to an adjacent empty cell
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move, True) and self.is_empty(move.dst):
                    yield move.clone()
            # Self destruct action
            move.dst = src
            if self.is_valid_action(move) and not self.is_empty(move.dst) and unit.type != UnitType.AI: # AI self destruct is not considered a valid move
                yield move.clone()

            # Repair and attack actions
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_action(move) and not self.is_empty(move.dst):
                    yield move.clone()

    def apply_move(self, move: CoordPair) -> Game:
        """
        This function applies a input move on the calling game state to simulate a new game state. It calls perform_move to apply the move 
        and returns a new game state resulting in the application of the move.
        """
        new_game = self.clone()
        (success, _ , _ ) = new_game.perform_move(move) # Perform given move on calling game state

        if success:
            return new_game # Return new game state
        else:
            return None

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord, unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker    
        return Player.Defender


###########################################################################################################
class GameTrace:
    def __init__(self, filename):
        self.file = open(filename, 'w')

    def write_parameters(self, options):
        self.file.write(f"Timeout: {options.max_time} seconds\n")
        self.file.write(f"Max Turns: {options.max_turns}\n")
        self.file.write(f"Alpha-Beta: {'On' if options.alpha_beta else 'Off'}\n")
        self.file.write(f"Play Mode: {str(options.game_type)}\n")
        if options.game_type == "manual":
            self.file.write("Attacker: H & Defender: H\n")
        elif options.game_type == "auto":
            self.file.write("Attacker: AI & Defender: AI\n")
        elif options.game_type == "defender":
            self.file.write("Attacker: AI & Defender: H\n")
        else:
            self.file.write("Attacker: H & Defender: AI\n")
        self.file.write(f"Heuristic used by AI: e{options.heuristic}\n")
        self.file.flush()  # Flush the buffer

    def write_board(self, game):
        self.file.write(str(game) + "\n")
        self.file.flush()  # Flush the buffer

    def write_action(self, game, turn, player, action, time):
        self.file.write(f"Turn #{turn} - {player.next().name}\n")
        (coords, actionType) = action
        string = ""
        match actionType:
            case 0:
                string = "move"
            case 1:
                string = "self-destruct"
            case 2:
                string = "repair"
            case 3:
                string = "attack"

        self.file.write(f"Action: {string} from {coords.src} to {coords.dst}\n")
        if time != None:
            self.file.write(f"Time for this action: {time} sec\n")
            self.file.write(f"Heuristic score: {game.h_score}\n")
            self.file.write(f"Cumulative evals: {game.states_evaluated}\n")
            depth = game.depth()
            self.file.write(f"Cumulative evals by depth: ")
            for k in sorted(game.stats.evaluations_per_depth.keys()):
                self.file.write(f"{depth}={game.stats.evaluations_per_depth[k]} ")
                depth -= 1
            self.file.write("\n")
            depth = game.depth()
            self.file.write(f"Cumulative % evals by depth: ")
            for k in sorted(game.stats.evaluations_per_depth.keys()):
                percentage = (game.stats.evaluations_per_depth[k] / game.states_evaluated) * 100
                self.file.write(f"{depth}={percentage:.1f}% ")
                depth -= 1
            self.file.write("\n")

        self.file.write("\n")
        self.file.flush()  # Flush the buffer

    def write_game_result(self, winner, turns_played):
        self.file.write(f"The winner of the game ({winner.name}) wins in {turns_played} turns\n")
        self.file.flush()  # Flush the buffer

    def close(self):
        self.file.close()


##############################################################################################################
def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_time', type=float, help='maximum search time', default=5)
    parser.add_argument('--max_turns', type=int, help='maximum turns before end of game', default=100)
    parser.add_argument('--alpha_beta', type=bool, help='alpha-beta on/off', default=False)
    parser.add_argument('--game_type', type=str, choices=["auto", "attacker", "defender", "manual"], default="auto",
                        help='game type: auto|attacker|defender|manual')
    args = parser.parse_args()


    # Initialize the GameTrace
    filename = f"gameTrace-{'true' if args.alpha_beta else 'false'}-{args.max_time}-{args.max_turns}.txt"
    trace = GameTrace(filename)
    # trace.write_parameters(args)

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)
    options.max_time = args.max_time
    options.max_turns = args.max_turns
    options.alpha_beta = args.alpha_beta
    trace.write_parameters(options)

    # override class defaults via command line options
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.max_turns is not None:
        options.max_turns = args.max_turns
    if args.alpha_beta is not None:
        options.alpha_beta = args.alpha_beta

    # create a new game
    game = Game(options=options)
    end = False
    stats = Stats()

    print()
    print(game)
    trace.write_board(game)  # Write the current board state to the trace
    # the main game loop
    while not end:
        end = game.is_finished()
        winner = game.has_winner()
        if winner is not None:
            trace.write_game_result(winner, game.turns_played)  # Write the game result
            trace.close()  # Close the trace file
            print(f"{winner.name} wins!\n{game.turns_played} turns played")
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            move = game.human_turn()
            trace.write_action(game, game.turns_played, game.next_player, move, None)  # Log the action to the trace file
            trace.write_board(game)  # Write the current board state to the trace
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            trace.write_action(game, game.turns_played, game.next_player, move, game.stats.total_seconds)
            trace.write_board(game)
            if stats.total_seconds > options.max_time:
                trace.write_board(game)  # Write the current board state
                trace.write_game_result(game.next_player.next(), game.turns_played)  # Write the game result
                trace.close()  # Close the trace file
                print(f"{game.next_player.name} took too long! Winner is {game.next_player.next().name}")
                break
            if move is not None:
                continue
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)
    trace.close()
    #############################################################################################################

if __name__ == '__main__':
    main()
