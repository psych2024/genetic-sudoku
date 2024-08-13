import random
from matplotlib import pyplot as plt

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

GIVEN_NUMBERS = 30
SUDOKU = generate_puzzle(GIVEN_NUMBERS)

print(" -------  SUDOKU ------- ")
pretty_print(SUDOKU)

TEMPLATE = []
for row in SUDOKU:
    missing = []
    for i in range(1, 10):
        if i not in row:
            missing.append(i)
    TEMPLATE.append(missing)

POPULATION_SIZE = 1200
MAX_FITNESS = 2 * 81
N_WAY_MUTATION = 5

SURVIVOR_PERCENTAGE = 0.6
NEWCOMERS_PERCENTAGE = 0.3

MUTATION_PROBABILITY = 0.9
CROSSOVER_PROBABILITY = 0.8
MAX_GENERATIONS = 400

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

# spawing initial population
populace = []

def spawn_candidate() -> list:
    candidate = []
    for row in TEMPLATE:
        row_copy = row.copy()
        random.shuffle(row_copy)
        candidate.append(row_copy)
    return candidate

def random_crossover(a: list, b: list) -> list:
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

def sort_by_fitness():
    populace.sort(key=lambda x: fitness(x))

def rank_selection() -> list:
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
    return populace[idx - 1]

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

def tournament_eliminate() -> list:
    candidates = random.sample(range(len(populace)), k=2)
    candidates.sort(key=lambda x: fitness(populace[x]))
    return populace.pop(candidates[0])

print("Starting simulation")
fitness_scores = []

# first we create the candidates
for _ in range(POPULATION_SIZE):
    populace.append(spawn_candidate())

# start solving
for _ in range(MAX_GENERATIONS):
    sort_by_fitness()

    best_fitness = fitness(populace[-1])
    fitness_scores.append(best_fitness)
    print(f"Best Candidate Fitness: {best_fitness}")
    # pretty_print(get_board_from_candidate(populace[-1]))

    if best_fitness == MAX_FITNESS:
        print("Solution found!")
        pretty_print(get_board_from_candidate(populace[-1]))
        break

    # start by creating offspring
    new_generation = []
    while len(new_generation) < int(POPULATION_SIZE * SURVIVOR_PERCENTAGE) + 1:
        parent_a = rank_selection()
        parent_b = rank_selection()

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

    populace.extend(new_generation)

    # then kill
    while len(populace) > POPULATION_SIZE * (1 - NEWCOMERS_PERCENTAGE):
        tournament_eliminate()

    # add in new spawns
    while len(populace) < POPULATION_SIZE:
        populace.append(spawn_candidate())

def display_results():
    plt.plot(list(range(1, len(fitness_scores) + 1)), fitness_scores)
    plt.show()

display_results()