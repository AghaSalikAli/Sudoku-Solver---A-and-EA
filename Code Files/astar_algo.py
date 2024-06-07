import queue
import time

def is_goal_state(sudoku):
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                return False

    for i in range(9):
        if len(set(sudoku[i])) != 9:
            return False
        if len(set([sudoku[j][i] for j in range(9)])) != 9:
            return False

    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            if len(set([sudoku[x][y] for x in range(i, i + 3) for y in range(j, j + 3)])) != 9:
                return False

    return True

def heuristic(sudoku, number):
    if number == 1:  # Number of empty cells Heuristic
        count = 0
        for i in range(9):
            for j in range(9):
                if sudoku[i][j] == 0:
                    count += 1
        return count
    elif number == 2:    # Constraint count Heuristic
        constraint_count = 0
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                possible_values = set(range(1, 10)) - \
                    set(sudoku[i]) - \
                    set([sudoku[x][j] for x in range(9)]) - \
                    set([sudoku[x][y] for x in range(i//33, (i//3+1)*3)
                        for y in range(j//33, (j//3+1)*3)])
                constraint_count += len(possible_values)
    return constraint_count

def a_star_algorithm(sudoku, heuristic_num, update_ui=None):
    q = queue.PriorityQueue()
    q.put((heuristic(sudoku, heuristic_num), sudoku))
    start_time = time.time()
    configurations_explored = 0



    while not q.empty():
        configurations_explored += 1
        current_state = q.get()[1]



        # Update UI with current state if update_ui function is provided
        if configurations_explored % 100 == 0:
            update_ui(current_state)

        if is_goal_state(current_state):
            end_time = time.time()
            return current_state, end_time - start_time, configurations_explored

        row, col = None, None
        for i in range(9):
            for j in range(9):
                if current_state[i][j] == 0:
                    row, col = i, j
                    break
            if row is not None:
                break

        possible_values = set(range(1, 10)) - \
                          set(current_state[row]) - \
                          set(current_state[i][col] for i in range(9)) - \
                          {current_state[x][y] for x in range(row // 3 * 3, (row // 3 + 1) * 3)
                           for y in range(col // 3 * 3, (col // 3 + 1) * 3)}

        for value in possible_values:
            new_state = [row[:] for row in current_state]
            new_state[row][col] = value
            new_cost = heuristic(new_state, heuristic_num) + 1
            q.put((new_cost, new_state))

    return None, None, configurations_explored


def a_star_analysis(sudoku, heuristic_num):
    q = queue.PriorityQueue()
    q.put((heuristic(sudoku, heuristic_num), sudoku))
    start_time = time.time()
    configurations_explored = 0



    while not q.empty():
        configurations_explored += 1
        current_state = q.get()[1]


        if is_goal_state(current_state):
            end_time = time.time()
            return current_state, end_time - start_time, configurations_explored

        row, col = None, None
        for i in range(9):
            for j in range(9):
                if current_state[i][j] == 0:
                    row, col = i, j
                    break
            if row is not None:
                break

        possible_values = set(range(1, 10)) - \
                          set(current_state[row]) - \
                          set(current_state[i][col] for i in range(9)) - \
                          {current_state[x][y] for x in range(row // 3 * 3, (row // 3 + 1) * 3)
                           for y in range(col // 3 * 3, (col // 3 + 1) * 3)}

        for value in possible_values:
            new_state = [row[:] for row in current_state]
            new_state[row][col] = value
            new_cost = heuristic(new_state, heuristic_num) + 1
            q.put((new_cost, new_state))

    return None, None, configurations_explored


