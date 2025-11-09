'''
Help the Vacuum Cleaner

Limits: 3 sec., 1024 MiB

A robot vacuum cleaner is cleaning a room. The room is represented by a map of n x m cells. Each cell can be either dirty or clean.
The vacuum cleaner starts at the top-left cell.
The vacuum cleaner can move to an adjacent cell (one that shares an edge with the current cell) or clean the cell it is currently on.
Each action takes 1 second.

Find the minimum time required for the vacuum cleaner to clean all the dirty cells.

Input

The first line contains two integers, n and m — the length and width of the room.
The next n lines each contain m characters, either * or .. * means the cell is dirty, and . means the cell is clean.

Output

Print a single integer — the time it will take for the vacuum cleaner to clean the room.

Constraints

1 ≤ n, m
n * m ≤ 20
'''

from collections import deque

def bfs_distance(grid, start, n, m):
    """Знаходить найкоротші відстані від start до всіх клітинок"""
    dist = [[float('inf')] * m for _ in range(n)]
    dist[start[0]][start[1]] = 0

    queue = deque([start])

    while queue:
        x, y = queue.popleft()

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < n and 0 <= ny < m and dist[nx][ny] == float('inf'):
                dist[nx][ny] = dist[x][y] + 1
                queue.append((nx, ny))

    return dist

def solve():
    n, m = map(int, input().split())
    grid = []
    for _ in range(n):
        grid.append(input().strip())

    # Знаходимо всі брудні клітинки
    dirty = []
    for i in range(n):
        for j in range(m):
            if grid[i][j] == '*':
                dirty.append((i, j))

    # Якщо немає брудних клітинок
    if not dirty:
        print(0)
        return

    # Створюємо стартову позицію (0, 0)
    start = (0, 0)

    # Обчислюємо відстані від стартової позиції та між усіма брудними клітинками
    # D[i][j] = відстань від i-ї позиції до j-ї брудної клітинки
    # D[0][j] = відстань від старту до j-ї брудної клітинки
    positions = [start] + dirty
    D = [[float('inf')] * (len(dirty) + 1) for _ in range(len(dirty) + 1)]

    for i, pos in enumerate(positions):
        dist_map = bfs_distance(grid, pos, n, m)
        for j, dirty_pos in enumerate(dirty):
            D[i][j + 1] = dist_map[dirty_pos[0]][dirty_pos[1]]

    # DP з бітовими масками
    # dp[mask][i] = мінімальна відстань для відвідування підмножини mask брудних клітинок,
    # закінчуючи в i-й брудній клітинці
    BITMASK = 1 << len(dirty)
    dp = [[float('inf')] * (len(dirty) + 1) for _ in range(BITMASK)]

    # Ініціалізація: йдемо зі старту до кожної брудної клітинки
    for j in range(1, len(dirty) + 1):
        mask = 1 << (j - 1)
        dp[mask][j] = D[0][j]

    # Заповнюємо DP
    for mask in range(BITMASK):
        # Перевірка: чи є хоч один валідний стан в цій масці
        has_valid = False
        for i in range(1, len(dirty) + 1):
            if dp[mask][i] < float('inf'):
                has_valid = True
                break

        if not has_valid:
            continue

        for i in range(1, len(dirty) + 1):
            # Перевірка: чи i-й біт встановлений в масці
            bit_i = 1 << (i - 1)
            if not (mask & bit_i):
                continue

            curr_d = dp[mask][i]

            if curr_d >= float('inf'):
                continue

            for j in range(1, len(dirty) + 1):
                bit = 1 << (j - 1)

                if mask & bit:
                    continue

                nm = mask | bit
                nd = D[i][j] + curr_d

                if nd < dp[nm][j]:
                    dp[nm][j] = nd

    # Знаходимо мінімальну відстань для відвідування всіх брудних клітинок
    full_mask = (1 << len(dirty)) - 1
    result = min(dp[full_mask][1:])

    # Додаємо час на очищення кожної клітинки (по 1 секунді на кожну)
    print(result + len(dirty))

if __name__ == "__main__":
    solve()



