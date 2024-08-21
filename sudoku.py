import os
import random

# +----------------------------------------------+
# |                    PART 1                    |
# +----------------------------------------------+

def backtrack(c, board):
    """
    Helper function that generates sudoku puzzle using recursive backtracking
    """
    if c == 81:  # base case
        return True

    y = c // 9  # current row
    x = c % 9  # current column
    y0 = y - y % 3  # y-coordinate of the top left entry of the subgrid
    x0 = x - x % 3  # x-coordinate of the top left entry of the subgrid
    l = list(range(1, 10))
    random.shuffle(l)

    # try to set current grid cell to n
    for n in l:
        valid = True
        for i in board[y]:  # checking rows
            if n == i:
                valid = False
        for i in map(lambda row: board[row][x], range(9)):  # checking columns
            if n == i:
                valid = False
        for row in map(lambda j: board[j], range(y0, y0 + 3)):  # checking subgrids
            for j in range(x0, x0 + 3):
                if n == row[j]:
                    valid = False

        board[y][x] = n
        if valid and backtrack(c + 1, board):  # placing n is a valid sudoku move, moving on to next grid cell
            return True

    # no suitable number, revert to previous position
    board[y][x] = 0
    return False

def generate_puzzle(given: int) -> list:
    """
    Generates a Sudoku puzzle randomly
    :param given: number of cells which should be filled initially
    :return: the Sudoku puzzle in normal representation
    """
    board = [[0 for _ in range(9)] for _ in range(9)]
    # generate board
    backtrack(0, board)

    # randomly removing (81 - n) grid cell entries leaving n entries
    remove = list(range(81))
    random.shuffle(remove)
    for i in range(81 - given):
        c = remove[i]
        x = c % 9
        y = c // 9
        board[x][y] = 0
    return board

def pretty_print(board: list) -> None:
    """
    Prints sudoku board such that it's human-readable and pretty
    :param board the Sudoku board in normal representation
    """
    print("Remove Me")

GIVEN_NUMBERS = 81
SUDOKU = generate_puzzle(GIVEN_NUMBERS)

print(" -------  SUDOKU ------- ")
pretty_print(SUDOKU)
print("Part 1 Done!")
exit(0)

# +----------------------------------------------+
# |                    PART 2                    |
# +----------------------------------------------+

TEMPLATE = []
# Create TEMPLATE here

def spawn_candidate() -> list:
    """
    Creates a new candidate randomly
    :return: A new candidate
    """
    print("Remove Me")

def convert_to_normal(candidate: list) -> list:
    """
    Converts solution representation to normal representation
    :param candidate: the solution representation to convert
    :return: A new copy of the normal representation
    """
    print("Remove Me")

print("Testing solution representation...")
pretty_print(convert_to_normal(spawn_candidate()))
print("Testing solution representation Done!")
exit(0)

def fitness(candidate: list) -> int:
    """
    Calculates the fitness of a solution as the number of unique entries
    in each subgrid and column
    Because of how our candidate is modeled, the rows will always be valid
    :param candidate: the candidate solution
    :return: The fitness of the candidate, with max of 81 (aka complete solution)
    """
    print("Remove Me")

print("Testing fitness function...")
assert (fitness(TEMPLATE) == 162)
print("Testing fitness Done!")
exit(0)

print("spawning initial population...")
POPULATION_SIZE = 1500
population = []
# creating the population
for _ in range(POPULATION_SIZE):
    population.append(spawn_candidate())

def sort_by_fitness() -> None:
    """
    Sort the entire population in increasing fitness
    Used for ranked selection
    """
    population.sort(key=lambda x: fitness(x))

def rank_select() -> list:
    """
    Selects a candidate from the population with probability weighted by rank
    Assumes populace is sorted by fitness increasing fitness
    :return: the selected candidate
    """
    print("Remove Me")

def random_crossover(a: list, b: list) -> list:
    """
    Creates two children based on two parent solutions
    Each child inherits an entire row from parent a or b, based on a coin flip
    Refer to slides for further details
    :param a: parent a
    :param b: parent b
    :return: a list containing the two children
    """
    print("Remove Me")

def mutate(candidate: list) -> None:
    """
    Picks a random row and swaps two entries in that row
    The candidate is modified directly
    :param candidate: the candidate to mutate
    """
    print("Remove Me")

def tournament_eliminate() -> None:
    """
    Eliminates a candidate by choosing two random candidates from the population
    and killing the weaker one
    """
    print("Remove Me")

# +----------------------------------------------+
# |                    PART 3                    |
# +----------------------------------------------+
MAX_GENERATIONS = 1500
SURVIVOR_PERCENTAGE = 0.65
OFFSPRING_PERCENTAGE = 0.65
NEWCOMERS_PERCENTAGE = 1 - SURVIVOR_PERCENTAGE
MUTATION_PROBABILITY = 0.9

def print_entire_page() -> None:
    """
    Pretty print stats for each generation animation style
    Assumes population is sorted
    """
    SCREEN_SIZE = os.get_terminal_size()
    SCREEN_HEIGHT = 16
    SCREEN_WIDTH = 25
    if SCREEN_WIDTH > SCREEN_SIZE.columns or SCREEN_HEIGHT > SCREEN_SIZE.lines:
        print(f"Terminal (width=${SCREEN_SIZE.columns}, height=${SCREEN_SIZE.lines}) is too small to render animation!")
        exit(0)

    rem = SCREEN_SIZE.lines - SCREEN_HEIGHT
    if best_fitness == 2 * 81:
        print("Solution found!")
        pretty_print(convert_to_normal(population[-1]))
        for _ in range(rem + 1):
            print('')

    else:
        print(f"Best Candidate Fitness: {best_fitness}")
        print(f"Worst Candidates Fitness: {worst_fitness}")
        pretty_print(convert_to_normal(population[-1]))
        for _ in range(rem):
            print('')

print("Starting simulation")

# +----------------------------------------------+
# |                    PART 3                    |
# +----------------------------------------------+

N_WAY_MUTATION = 8
