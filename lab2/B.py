'''
Center of Segments
Limits: 2 sec., 512 MiB
Given a set of segments. You need to find a point such that the sum of distances from it to all segments is minimal.
Input
The first line contains one integer N - the number of segments.
The next N lines contain 6 integers separated by spaces - x, y, z coordinates of the start and x, y, z coordinates of the end of the segment.
Output
One number - the sum of distances from the found point to the given segments with precision up to 4 decimal places.
Constraints

1 ≤ N ≤ 100
-100 ≤ x, y, z ≤ 100
'''
from dataclasses import dataclass
import math


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Point3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)


@dataclass
class Segment:
    start: Point3D
    end: Point3D


def distance_point_to_segment(point, segment):
    """
    Calculate minimum distance from point to line segment in 3D
    """
    p = point
    a = segment.start
    b = segment.end

    ab = b - a
    ap = p - a

    ab_len_sq = ab.dot(ab)

    if ab_len_sq == 0:
        return ap.norm()

    t = ap.dot(ab) / ab_len_sq
    t = max(0, min(1, t))

    closest = a + ab * t

    return (p - closest).norm()


def compute_total_distance(point, segments):
    """
    Compute total distance from point to all segments
    """
    return sum(distance_point_to_segment(point, seg) for seg in segments)


def compute_gradient(point, segments, epsilon=1e-6):
    """
    Compute numerical gradient of the objective function
    """
    current_distance = compute_total_distance(point, segments)

    # Numerical gradient for x
    point_dx = Point3D(point.x + epsilon, point.y, point.z)
    dist_dx = compute_total_distance(point_dx, segments)
    grad_x = (dist_dx - current_distance) / epsilon

    # Numerical gradient for y
    point_dy = Point3D(point.x, point.y + epsilon, point.z)
    dist_dy = compute_total_distance(point_dy, segments)
    grad_y = (dist_dy - current_distance) / epsilon

    # Numerical gradient for z
    point_dz = Point3D(point.x, point.y, point.z + epsilon)
    dist_dz = compute_total_distance(point_dz, segments)
    grad_z = (dist_dz - current_distance) / epsilon

    return Point3D(grad_x, grad_y, grad_z)


def gradient_descent(segments, max_iterations=100000, learning_rate=0.001, tolerance=1e-6):
    """
    Find optimal point using gradient descent with distance-based convergence
    """
    # Initialize at centroid of all segment endpoints
    all_points = []
    for seg in segments:
        all_points.append(seg.start)
        all_points.append(seg.end)

    sum_x = sum(p.x for p in all_points)
    sum_y = sum(p.y for p in all_points)
    sum_z = sum(p.z for p in all_points)
    n = len(all_points)

    point = Point3D(sum_x / n, sum_y / n, sum_z / n)

    # Calculate initial distance
    prev_distance = compute_total_distance(point, segments)

    # Gradient descent loop
    for iteration in range(max_iterations):
        # Compute gradient
        gradient = compute_gradient(point, segments)

        # Update point
        point = point - gradient * learning_rate

        # Calculate new distance
        current_distance = compute_total_distance(point, segments)

        # Check for convergence based on distance change
        distance_change = abs(current_distance - prev_distance)
        if distance_change < tolerance:
            break

        # Update previous distance for next iteration
        prev_distance = current_distance

    # Calculate final minimum distance
    min_distance = compute_total_distance(point, segments)

    return point, min_distance


def main():
    n = int(input())
    segments = []

    for _ in range(n):
        coords = list(map(int, input().split()))
        start = Point3D(coords[0], coords[1], coords[2])
        end = Point3D(coords[3], coords[4], coords[5])
        segments.append(Segment(start, end))

    # Run gradient descent
    optimal_point, min_distance = gradient_descent(segments)

    # Output result with 4 decimal places
    print(f"{min_distance:.4f}")


if __name__ == "__main__":
    main()