'''
Dangerous Friends

Limits: 3 sec., 256 MiB

Your city has been closed for quarantine — from now on no one can leave or enter it. At the moment of quarantine introduction,
there were n people living in the city, and some pairs of them are friends and have daily contact with each other.
Since all the city’s residents are very responsible citizens, on the very first day of quarantine everyone took the appropriate test and publicly announced their results.
Thus, you know who in the city is sick on the first day after the quarantine began. However, the disease spreads very quickly,
and therefore there is a high risk of infection when contacting an already sick person.
Even though you are definitely healthy, as a responsible person you decided to protect yourself.
To do this, you will cut off communication with some of your friends in order to prevent the possibility of infection.
Now you need to decide with which friends you will stop contacting from today.
So help yourself — write a program for this!

Input

The first line contains three integers n, m, k — the number of people in the city (including you),
the number of pairs of friends among these people, and the number of sick people at the initial moment.
The next m lines each contain two integers u_i, v_i — people with these numbers are friends.
The last line contains k numbers — the indices of the people who are sick at the initial moment.
Note that your index number is always equal to 1, and you are definitely healthy at the start of the quarantine.

Output

In the first line print a single integer — the minimum number of friends you need to stop communicating with.
In the second line print the indices of these friends in ascending order.

Constraints

2 ≤ n
1 ≤ m
0 ≤ k ≤ n − 1
5 tests: n, m ≤ 100
5 tests: n, m ≤ 10^3
10 tests: n, m ≤ 10^5
5 tests: n, m ≤ 5 ⋅ 10^5
'''

'''
I will create a list 'infected' with False for the people who won't be infected and True for people who potentionaly can be infected,
and using bfs starting from the infected people i will mark all the targets. in the end i'll extract all the friend of number 1, who were marked.
'''
from collections import defaultdict, deque


def main():
    n_people, n_friend_pairs, n_sick_people = map(int, input().split())

    graph = defaultdict(list)
    for _ in range(n_friend_pairs):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)

    sick_people_list = list(map(int, input().split()))

    # print(graph)

    # BFS
    infected = [False] * (n_people + 1)

    q = deque(sick_people_list)

    for s in sick_people_list:
        infected[s] = True

    while q:
        node = q.popleft()
        for neigh in graph[node]:
            if neigh == 1:
                continue
            if not infected[neigh]:
                infected[neigh] = True
                q.append(neigh)

    # check which of person 1's friends are infected
    dangerous = [f for f in graph[1] if infected[f]]

    print(len(dangerous))
    for i in dangerous:
        print(i)

main()