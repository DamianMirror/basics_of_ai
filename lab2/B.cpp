#include <iostream>
#include <vector>
#include <cmath>      // For sqrt, abs
#include <iomanip>    // For std::fixed, std::setprecision
#include <algorithm>  // For std::min, std::max
#include <utility>    // For std::pair

struct Point3D {
    double x = 0.0, y = 0.0, z = 0.0;

    // Operator overloading for vector math
    Point3D operator+(const Point3D& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    Point3D operator-(const Point3D& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    Point3D operator*(double scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }

    Point3D operator/(double scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }

    double dot(const Point3D& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    double norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }
};

// Segment struct
struct Segment {
    Point3D start;
    Point3D end;
};

double distance_point_to_segment(const Point3D& point, const Segment& segment) {
    Point3D p = point;
    Point3D a = segment.start;
    Point3D b = segment.end;

    Point3D ab = b - a;
    Point3D ap = p - a;

    double ab_len_sq = ab.dot(ab);

    // If segment is just a point
    if (ab_len_sq == 0) {
        return ap.norm();
    }

    // Project p onto the line ab, but clamp t to [0, 1]
    double t = ap.dot(ab) / ab_len_sq;
    t = std::max(0.0, std::min(1.0, t));

    Point3D closest = a + ab * t;
    return (p - closest).norm();
}

double compute_total_distance(const Point3D& point, const std::vector<Segment>& segments) {
    double total_dist = 0.0;
    for (const auto& seg : segments) {
        total_dist += distance_point_to_segment(point, seg);
    }
    return total_dist;
}

Point3D compute_gradient(const Point3D& point, const std::vector<Segment>& segments) {
    const double epsilon = 1e-6;
    double current_distance = compute_total_distance(point, segments);

    // Gradient for x
    Point3D point_dx = {point.x + epsilon, point.y, point.z};
    double dist_dx = compute_total_distance(point_dx, segments);
    double grad_x = (dist_dx - current_distance) / epsilon;

    // Gradient for y
    Point3D point_dy = {point.x, point.y + epsilon, point.z};
    double dist_dy = compute_total_distance(point_dy, segments);
    double grad_y = (dist_dy - current_distance) / epsilon;

    // Gradient for z
    Point3D point_dz = {point.x, point.y, point.z + epsilon};
    double dist_dz = compute_total_distance(point_dz, segments);
    double grad_z = (dist_dz - current_distance) / epsilon;

    return {grad_x, grad_y, grad_z};
}

std::pair<Point3D, double> gradient_descent(const std::vector<Segment>& segments) {
    // Hyperparameters
    const int max_iterations = 100000;
    const double learning_rate = 0.001;
    const double tolerance = 1e-6;

    // Initialize at centroid of all segment endpoints
    Point3D sum_points; // Initializes to (0,0,0)
    int n = 0;
    for (const auto& seg : segments) {
        sum_points = sum_points + seg.start;
        sum_points = sum_points + seg.end;
        n += 2;
    }

    Point3D point = sum_points / n;
    double prev_distance = compute_total_distance(point, segments);

    // Gradient descent loop
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        Point3D gradient = compute_gradient(point, segments);
        point = point - (gradient * learning_rate);

        double current_distance = compute_total_distance(point, segments);

        // Check for convergence
        double distance_change = std::abs(current_distance - prev_distance);
        if (distance_change < tolerance) {
            break;
        }

        prev_distance = current_distance;
    }

    double min_distance = compute_total_distance(point, segments);
    return {point, min_distance};
}

int main() {
    int n;
    std::cin >> n;

    std::vector<Segment> segments(n);
    for (int i = 0; i < n; ++i) {
        std::cin >> segments[i].start.x >> segments[i].start.y >> segments[i].start.z
                 >> segments[i].end.x   >> segments[i].end.y   >> segments[i].end.z;
    }

    // Run gradient descent
    std::pair<Point3D, double> result = gradient_descent(segments);
    double min_distance = result.second;

    // Output result with 4 decimal places
    std::cout << std::fixed << std::setprecision(4) << min_distance << std::endl;

    return 0;
}