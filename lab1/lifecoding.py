
from collections import defaultdict
import heapq

'''
test

5 6 1
1 3 5
2 1 4
4 5 6
2 5 3
4 1 6
1 5 0
'''


def main():
    n, m, s = map(int, input().split())

    print(n, m, s)

    graph = defaultdict(dict)
    for _ in range(m):
        u,v, w = map(int, input().split())
        graph[u][v] = w

    print(graph)

    #dejkstra

    start = s
    heap = [(0, start)]
    visited = set()
    disctances = {start: 0}

    while heap:
        curr_dist, node = heapq.heappop(heap)

        if node in visited:
            continue

        visited.add(node)

        for neigbour, weight in graph[node].items():
            if neigbour in visited:
                continue

            new_weight = curr_dist + weight

            if neigbour not in disctances or disctances[neigbour] > new_weight:

                disctances[neigbour] = new_weight
                heapq.heappush(heap, (new_weight, neigbour))

    print(disctances)

main()