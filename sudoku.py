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
    :return: the Sudoku puzzle as a list
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
    """
    for row in range(9):
        if row % 3 == 0:
            print(" ------- ------- ------- ")
        for col in range(9):
            if col % 3 == 0:
                print("|", end=" ")
            print(board[row][col], end=" ")
        print("|")
    print(" ------- ------- ------- ")

GIVEN_NUMBERS = 17
SUDOKU = generate_puzzle(GIVEN_NUMBERS)

print(" -------  SUDOKU ------- ")
pretty_print(SUDOKU)
print("Part 1 Done!")
# exit(0)

# +----------------------------------------------+
# |                    PART 2                    |
# +----------------------------------------------+

TEMPLATE = []
for row in SUDOKU:
    missing = []
    for i in range(1, 10):
        if i not in row:
            missing.append(i)
    TEMPLATE.append(missing)

def spawn_candidate() -> list:
    """
    Creates a new candidate randomly
    :return: A new candidate
    """
    candidate = []
    for row in TEMPLATE:
        row_copy = row.copy()
        random.shuffle(row_copy)
        candidate.append(row_copy)
    return candidate

def get_board_from_candidate(candidate: list) -> list:
    """
    Converts solution representation to normal representation
    :param candidate: the solution to convert
    :return: A new copy of the normal representation
    """
    board = []
    for i in range(9):
        l = []
        idx = 0
        for x in SUDOKU[i]:
            if x == 0:
                l.append(candidate[i][idx])
                idx += 1
            else:
                l.append(x)
        board.append(l)
    return board

print("Testing solution representation...")
pretty_print(get_board_from_candidate(spawn_candidate()))
print("Testing solution representation Done!")
# exit(0)

def fitness(candidate: list) -> int:
    """
    Calculates the fitness of a solution as the number of unique entries
    in each subgrid and column
    Because of how our candidate is modeled, the rows will always be valid
    :param candidate: the candidate solution
    :return: The fitness of the candidate, with max of 81 (aka complete solution)
    """
    result = 0
    board = get_board_from_candidate(candidate)

    # check columns first
    for col in range(9):
        col_list = []
        for j in range(9):
            col_list.append(board[j][col])
        result += len(set(col_list))

    for subgrid_row in range(0, 9, 3):
        for subgrid_col in range(0, 9, 3):
            subgrid_list = []
            for dx in range(3):
                for dy in range(3):
                    subgrid_list.append(board[subgrid_row + dx][subgrid_col + dy])
            result += len(set(subgrid_list))
    return result

print("Testing fitness function...")
# assert (fitness(TEMPLATE) == 162)
print("Testing fitness Done!")
# exit(0)

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
    S = POPULATION_SIZE * (POPULATION_SIZE + 1) // 2
    n = random.randint(0, S)
    count = 0
    idx = 1
    while count < n:
        count += idx
        if count < n:
            idx += 1
    return population[idx - 1]

def random_crossover(a: list, b: list) -> list:
    """
    Creates two children based on two parent solutions
    Each child inherits an entire row from parent a or b, based on a coin flip
    Refer to slides for further details
    :param a: parent a
    :param b: parent b
    :return: a list containing the two children
    """
    child_a = []
    child_b = []
    for i in range(9):
        if random.random() < 0.5:
            child_a.append(a[i].copy())
            child_b.append(b[i].copy())
        else:
            child_a.append(b[i].copy())
            child_b.append(a[i].copy())
    return [child_a, child_b]

def mutate(candidate: list) -> None:
    """
    Picks a random row and swaps two entries in that row
    The candidate is modified directly
    :param candidate: the candidate to mutate
    """
    selected_row = candidate[random.randint(0, 8)]
    if len(selected_row) == 0:
        return
    pos_1 = random.randint(0, len(selected_row) - 1)
    pos_2 = random.randint(0, len(selected_row) - 1)
    selected_row[pos_1], selected_row[pos_2] = selected_row[pos_2], selected_row[pos_1]

def tournament_eliminate() -> None:
    """
    Eliminates a candidate by choosing two random candidates from the population
    and killing the weaker one
    """
    candidates = random.sample(range(len(population)), k=2)
    candidates.sort(key=lambda x: fitness(population[x]))
    population.pop(candidates[0])

# +----------------------------------------------+
# |                    PART 3                    |
# +----------------------------------------------+
MAX_GENERATIONS = 1500
SURVIVOR_PERCENTAGE = 0.65
NEWCOMERS_PERCENTAGE = 0.35

CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.9
N_WAY_MUTATION = 8

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
        pretty_print(get_board_from_candidate(population[-1]))
        for _ in range(rem + 1):
            print('')

    else:
        print(f"Best Candidate Fitness: {best_fitness}")
        print(f"Worst Candidates Fitness: {worst_fitness}")
        pretty_print(get_board_from_candidate(population[-1]))
        for _ in range(rem):
            print('')

print("Starting simulation")

MAX_STAGNATE = 1000
stagnate_count = 0
previous_best_fitness = 0
# +----------------------------------------------+
# |                    PART 3                    |
# +----------------------------------------------+
for _ in range(MAX_GENERATIONS):
    sort_by_fitness()

    best_fitness = fitness(population[-1])
    worst_fitness = fitness(population[0])

    if previous_best_fitness >= best_fitness:
        stagnate_count += 1
    else:
        stagnate_count = 0

    if stagnate_count >= MAX_STAGNATE:
        # kill off population
        population = []
        for _ in range(POPULATION_SIZE):
            population.append(spawn_candidate())
        stagnate_count = 0

    previous_best_fitness = best_fitness

    print_entire_page()

    if best_fitness == 162:
        exit(0)

    # start by creating offspring
    new_generation = []
    while len(new_generation) < int(POPULATION_SIZE * SURVIVOR_PERCENTAGE) + 1:
        parent_a = rank_select()
        parent_b = rank_select()

        if random.random() < CROSSOVER_PROBABILITY:
            child_a, child_b = random_crossover(parent_a, parent_b)

            if random.random() < MUTATION_PROBABILITY:
                for _ in range(random.randint(1, N_WAY_MUTATION)):
                    mutate(child_a)
            if random.random() < MUTATION_PROBABILITY:
                for _ in range(random.randint(1, N_WAY_MUTATION)):
                    mutate(child_b)

            new_generation.append(child_a)
            new_generation.append(child_b)
        else:
            new_generation.append(parent_a)
            new_generation.append(parent_b)

    population.extend(new_generation)

    # then kill
    while len(population) > POPULATION_SIZE * (1 - NEWCOMERS_PERCENTAGE):
        tournament_eliminate()

    # add in new spawns
    while len(population) < POPULATION_SIZE:
        population.append(spawn_candidate())

# def display_results():
#     plt.plot(list(range(1, len(fitness_scores) + 1)), fitness_scores)
#     plt.show()
#
# display_results()