'''
Festive Cake

Limits: 2 sec., 256 MiB

Petro, 53 years old, a pastry chef, decided to bake a cake specially for the celebration of the 47th day of the year.
Moreover, considering the scale of the celebration of this remarkable event in Lviv, Petro plans to bake many cakes and connect some of them with waffle bridges.
Rumor has it that the composition will also include several chocolate fountains.
According to the plan, there should be n cakes, connected by m bidirectional bridges. However, Petro has not yet decided the lengths of all bridges:
for some bridges he already knows the length, and for others he is still undecided.
Petro believes that the celebration will be especially successful if the shortest distance between cake number 1 and cake number n through the bridges is exactly w.
Help him determine the lengths of the bridges that would satisfy this condition. In case of success, Petro will treat you with his sweets.

Input

The first line contains three integers n, m, w — the number of cakes, the number of bridges, and the desired shortest distance, respectively.
The next m lines each contain three integers u_i, v_i, w_i. This means the i-th bridge connects cakes numbered u_i and v_i and has length w_i. If w_i = −1,
then the length of this bridge is unknown.
It is guaranteed that there exists a path between any pair of cakes via bridges, and no bridge connects a cake to itself.

Output

Output m lines, one integer each — the lengths of the bridges in the order they were given in the input.
If it is impossible to achieve the desired minimum distance, output -1 in a single line.
Note: each bridge length must be a positive integer between 1 and 200 inclusive.

Constraints

1 ≤ n ≤ 100
1 ≤ u_i, v_i ≤ n, u_i ≠ v_i
1 ≤ m, w ≤ 200
w_i = −1 or 1 ≤ w_i ≤ 200
'''
from _collections import defaultdict
import heapq
import numpy as np


def find_min(graph, target_cakes):
    min_value = float('inf')
    min_edge = None

    for u, v in target_cakes:
        if u in graph and v in graph[u]:  # перевірка, чи ребро існує
            weight = graph[u][v]
            if weight < min_value:
                min_value = weight
                min_edge = (u, v)

    return min_edge


def main():
    n_cakes, n_bridges, target_distance = map(int, input().split())
    target_cakes = []
    order = []

    graph = defaultdict(dict)
    for _ in range(n_bridges):
        u, v, dist = map(int, input().split())
        order.append((u, v))
        if dist == -1:
            target_cakes.append((u, v))
        graph[u][v] = np.abs(dist)
        graph[v][u] = np.abs(dist)

    start = 1
    finish = n_cakes


    while True:
        heap = [(0, start)]  # element and weight
        visited = set()
        distances = {start: 0, finish: np.inf}
        parents = {start: None}

        while heap:
            current_dist, node = heapq.heappop(heap)

            if node == finish:
                distances[finish] = current_dist
                break

            if node in visited:
                continue

            visited.add(node)

            for neighbor , weight in graph[node].items():
                if neighbor in visited:
                    continue
                new_dist = current_dist + weight
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    parents[neighbor] = node
                    heapq.heappush(heap, (new_dist, neighbor))

        if distances[finish] >= target_distance:
            break
        else:
            min_index = find_min(graph, target_cakes)
            new_weight = target_distance - distances[finish] + 1
            graph[min_index[0]][min_index[1]] = new_weight
            graph[min_index[1]][min_index[0]] = new_weight


    if distances[finish] > target_distance:
        print(-1)
    else:
        for u, v in order:
            print(graph[u][v])


main()