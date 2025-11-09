'''
Fifteen Puzzle

Limits: 2 sec., 256 MiB

Perhaps everyone played the famous fifteen puzzle game in their childhood.
The rules are very simple: the game board consists of eight squares, numbered from 1 to 8,
and one empty space. In one move, you can slide an adjacent square (horizontally or vertically) into the empty space.
The goal of the game is to reach the following position (0 represents the empty space):

123
456
780
Petya also loves this game, but often he can't reach the target position even after several hours. Help Petya.

Input

The initial game position is given as 3 lines of 3 digits each, without any spaces. The numbers range from 0 to 8.

Output

In a single line, print one integer — the minimum number of moves required to reach the target position, or -1 if the target position cannot be reached.


'''

import heapq

def manhattan_distance(state, goal):
    """
    Calculate the sum of Manhattan distances for all tiles
    Manhattan distance = |x1 - x2| + |y1 - y2|
    """
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != '0':  # Don't count the empty space
                value = state[i][j]
                # Find goal position of this value
                for gi in range(3):
                    for gj in range(3):
                        if goal[gi][gj] == value:
                            distance += abs(i - gi) + abs(j - gj)
                            break
    return distance

def get_neighbors(state):
    """
    Get all possible states from current state by moving adjacent tiles
    """
    neighbors = []

    # Find position of empty space (0)
    zero_i, zero_j = 0, 0
    for i in range(3):
        for j in range(3):
            if state[i][j] == '0':
                zero_i, zero_j = i, j
                break

    # Possible moves: up, down, left, right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for di, dj in moves:
        ni, nj = zero_i + di, zero_j + dj

        # Check if move is valid
        if 0 <= ni < 3 and 0 <= nj < 3:
            # Create new state by swapping
            new_state = [row[:] for row in state]  # Deep copy
            new_state[zero_i][zero_j], new_state[ni][nj] = new_state[ni][nj], new_state[zero_i][zero_j]
            neighbors.append(new_state)

    return neighbors

def state_to_tuple(state):
    """Convert 2D list to tuple for hashing"""
    return tuple(tuple(row) for row in state)

def is_solvable(state):
    """
    Check if the puzzle is solvable
    A puzzle is solvable if the number of inversions is even
    """
    flat = []
    for row in state:
        for val in row:
            if val != '0':
                flat.append(int(val))

    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1

    return inversions % 2 == 0

def a_star(start, goal):
    """
    A* algorithm implementation
    f(n) = g(n) + h(n)
    g(n) = cost from start to current node
    h(n) = heuristic (Manhattan distance)
    """

    # Check if puzzle is solvable
    if not is_solvable(start):
        return -1

    start_tuple = state_to_tuple(start)
    goal_tuple = state_to_tuple(goal)

    if start_tuple == goal_tuple:
        return 0

    h_start = manhattan_distance(start, goal)
    # Priority queue: (f_score, g_score, state)
    pq = [(h_start, 0, start)]

    visited = set()
    g_scores = {start_tuple: 0}

    while pq:
        f, g, current = heapq.heappop(pq)
        current_tuple = state_to_tuple(current)

        if current_tuple in visited:
            continue

        visited.add(current_tuple)

        # Check if we reached the goal
        if current_tuple == goal_tuple:
            return g

        # Explore neighbors
        for neighbor in get_neighbors(current):
            neighbor_tuple = state_to_tuple(neighbor)

            if neighbor_tuple in visited:
                continue

            # g(neighbor) = g(current) + 1 (each move costs 1)
            tentative_g = g + 1

            # Only process if we found a better path
            if neighbor_tuple not in g_scores or tentative_g < g_scores[neighbor_tuple]:
                g_scores[neighbor_tuple] = tentative_g
                h = manhattan_distance(neighbor, goal)
                f = tentative_g + h
                heapq.heappush(pq, (f, tentative_g, neighbor))

    return -1  # No solution found

def main():
    # Read input
    matrix = []
    for _ in range(3):
        matrix.append(list(input().strip()))

    # Goal state
    goal = [
        ['1', '2', '3'],
        ['4', '5', '6'],
        ['7', '8', '0']
    ]

    # Run A* algorithm
    result = a_star(matrix, goal)
    print(result)

if __name__ == "__main__":
    main()
