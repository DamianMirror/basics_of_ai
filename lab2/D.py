'''
Clustering Geographic Locations
Limits: 2 sec., 256 MiB
You are given N points of geographic locations placed on a 2D plane where people often travel. You need to assign these points to K or fewer clusters using the classic K-means algorithm.
The implementation of this algorithm includes random generation of initial cluster centers, but in this problem you will be given K different indices. The points at these indices will be the initial cluster centers.
Another parameter of the K-means algorithm is the number of iterations of its application; for simplicity, let's say that exactly 100 iterations will be sufficient.
Input
The first line contains 2 natural numbers N and K.
The next N lines contain N different pairs of integers xᵢ, yᵢ - coordinates of the points.
In the last line, K different natural numbers idxⱼ are given - points with these indices are the initial cluster centers.
Output
In N lines, output 1 natural number each - which cluster the i-th point belongs to.
Constraints

1 ≤ K ≤ N ≤ 10³
-10⁹ ≤ xᵢ, yᵢ ≤ 10⁹
1 ≤ idxⱼ ≤ N

Notes
If at any iteration of the algorithm there are clusters to which no point belongs,
then you need to move all their centers to the point (0, 0). For example, when k = 5,
but the points were assigned to the first three clusters,
then the centers of the last 2 clusters should be moved to (0, 0).

The solution is considered correct if the points belonging to one cluster in your answer
match the points belonging to the corresponding cluster in the reference solution.
Cluster numbers may differ, the main thing is that the distribution of points across clusters is identical.
'''

def main():
    N, K = map(int, input().split())

    points = [tuple(map(int, input().split())) for _ in range(N)]

    initial_indices = list(map(int, input().split()))



if __name__ == "__main__":
    main()