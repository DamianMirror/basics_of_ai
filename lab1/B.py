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


import sys
from collections import deque

def solve():
    n, m = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(n)]

    # Find dirty cells
    dirty = []
    for i in range(n):
        row = grid[i]
        for j in range(m):
            if row[j] == '*':
                dirty.append((i, j))

    k = len(dirty)

    if k == 0:
        return 0

    INF = 999999

    def bfs(si, sj):
        d = [[INF]*m for _ in range(n)]
        d[si][sj] = 0
        q = deque([(si, sj)])
        while q:
            i, j = q.popleft()
            cd = d[i][j]
            # Unroll loop for PyPy
            ni, nj = i, j+1
            if nj < m and d[ni][nj] == INF:
                d[ni][nj] = cd + 1
                q.append((ni, nj))
            ni, nj = i+1, j
            if ni < n and d[ni][nj] == INF:
                d[ni][nj] = cd + 1
                q.append((ni, nj))
            ni, nj = i, j-1
            if nj >= 0 and d[ni][nj] == INF:
                d[ni][nj] = cd + 1
                q.append((ni, nj))
            ni, nj = i-1, j
            if ni >= 0 and d[ni][nj] == INF:
                d[ni][nj] = cd + 1
                q.append((ni, nj))
        return d

    # Build distance matrix
    dist = [[0]*k for _ in range(k)]
    from_start = [0]*k

    # From start
    sd = bfs(0, 0)
    for i in range(k):
        di, dj = dirty[i]
        from_start[i] = sd[di][dj]

    # Between dirty cells
    for i in range(k):
        di, dj = dirty[i]
        dd = bfs(di, dj)
        for j in range(k):
            if i != j:
                dj_i, dj_j = dirty[j]
                dist[i][j] = dd[dj_i][dj_j]

    # DP with bitmask
    full = (1<<k)-1
    dp = [[INF]*k for _ in range(1<<k)]

    for i in range(k):
        dp[1<<i][i] = from_start[i] + 1

    for mask in range(1, 1<<k):
        dp_mask = dp[mask]
        for i in range(k):
            bit_i = 1<<i
            if not (mask & bit_i):
                continue
            cur = dp_mask[i]
            if cur >= INF:
                continue
            dist_i = dist[i]
            for j in range(k):
                bit_j = 1<<j
                if mask & bit_j:
                    continue
                nmask = mask | bit_j
                nval = cur + dist_i[j] + 1
                if nval < dp[nmask][j]:
                    dp[nmask][j] = nval

    return min(dp[full])

print(solve())