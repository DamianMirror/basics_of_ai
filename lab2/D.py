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

import math


def euclidean_distance_squared(p1, p2):
    """Calculate Euclidean distance between two points"""
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def kmeans(points, initial_centers_idx, k, iterations=100):
    """
    K-means clustering algorithm

    Args:
        points: list of (x, y) tuples
        initial_centers_idx: list of indices for initial cluster centers (1-indexed)
        k: number of clusters
        iterations: number of iterations to run

    Returns:
        list of cluster assignments for each point (0-indexed cluster numbers)
    """
    n = len(points)

    # Initialize cluster centers using given indices (convert from 1-indexed to 0-indexed)
    centers = [points[idx - 1] for idx in initial_centers_idx]

    # Initialize cluster assignments
    clusters = [0] * n

    # Run for specified number of iterations
    for iteration in range(iterations):
        # Step 1: Assign each point to nearest cluster center
        changed = False  # Track if any assignments changed

        for i in range(n):
            min_dist_sq = float('inf')
            closest_cluster = clusters[i]  # Start with current assignment

            for j in range(k):
                dist_sq = euclidean_distance_squared(points[i], centers[j])
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_cluster = j

            if clusters[i] != closest_cluster:
                clusters[i] = closest_cluster
                changed = True

        # Early stopping: if no assignments changed, we've converged
        if not changed:
            break

        # Step 2: Update cluster centers
        new_centers = []

        for j in range(k):
            # Find all points belonging to cluster j
            cluster_points = [points[i] for i in range(n) if clusters[i] == j]

            if len(cluster_points) == 0:
                # If no points in cluster, move center to (0, 0)
                new_centers.append((0, 0))
            else:
                # Calculate mean of all points in cluster
                mean_x = sum(p[0] for p in cluster_points) / len(cluster_points)
                mean_y = sum(p[1] for p in cluster_points) / len(cluster_points)
                new_centers.append((mean_x, mean_y))

        centers = new_centers

    return clusters

def main():
    N, K = map(int, input().split())

    points = [tuple(map(int, input().split())) for _ in range(N)]

    initial_indices = list(map(int, input().split()))

    # Run K-means
    clusters = kmeans(points, initial_indices, K, iterations=100)

    # Output cluster assignments (convert to 1-indexed)
    for cluster in clusters:
        print(cluster + 1)


if __name__ == "__main__":
    main()