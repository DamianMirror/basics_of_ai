'''
Trilandia
Limits: 2 sec., 256 MiB

Trilandian scientists have reached a new level of research,
so they decided to develop a robot that could explore surfaces of any kind at any point in the Universe.
For this, they are developing a remote control system for the robot.

You need to solve only a small subtask for the robot — namely, to find the minimum distance between two sectors.
When the robot is in sector number 𝑖 and needs to move to sector number 𝑗, it must travel a certain distance,
which we will consider as the number of edges it needs to cross to move to another sector. Each sector represents a triangle in space.
You can move from one sector to another only if they share a common edge.

Input
The first line contains one integer 𝑛 — the number of sectors.
The second line contains two integers 𝑖 and 𝑗 (sectors are numbered from 1 to 𝑛).
The next 3 ⋅ 𝑛 lines contain the description of the sectors:
Each sector is described by three lines, each containing three real numbers with precision up to 10⁻³ — the coordinates of the vertices in space.

Output
In a single line, print one number — the minimum distance between the 𝑖-th and 𝑗-th sectors, or -1 if there is no path between the given sectors.

Constraints
1 ≤ 𝑛 ≤ 35000.
'''

from collections import defaultdict, deque
from itertools import combinations

def main():
    n_sectors = int(input())
    start, finish = map(int, input().split())

    # keys - edges, values - triangles they belong to
    edge_to_triangles = defaultdict(list)
    for i in range(1, n_sectors + 1):
        points = [tuple(map(int, input().split())) for _ in range(3)]
        for p1, p2 in combinations(points, 2):
            edge_to_triangles[frozenset([p1, p2])].append(i)

    print(edge_to_triangles.values())

    # graph of connected triangles
    triangle_graph = defaultdict(set)
    for triangles in edge_to_triangles.values():
        if len(triangles) > 1:
            for t1, t2 in combinations(triangles, 2):
                triangle_graph[t1].add(t2)
                triangle_graph[t2].add(t1)

    #bfs
    queue = deque([(start, 0)])
    visited = set()

    while queue:
        node, distance = queue.popleft()
        if node == finish:
            print(distance)
            return

        if node in visited:
            continue
        visited.add(node)

        for neigh in triangle_graph[node]:
            if neigh not in visited:
                queue.append((neigh, distance + 1))

main()