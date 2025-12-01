'''
Lidar robot localization 1
Limits: 10 sec., 512 MiB
Given a log of sensor data and commands for a robot moving around a room.
You need to determine the robot's coordinates after it finishes moving and the robot's average speed.
The robot moves exclusively on the floor (2D space). The robot moves by changing its coordinates without
changing its orientation in space. The robot has one sensor - a 2D lidar. The lidar scans the space around
the robot. One scan occurs as follows: The lidar emits K rays, the angle between adjacent rays is 360/K
degrees, and for each ray determines the distance to the wall in that direction. The lidar has an error
Sl in distance (the distance determined by the lidar is considered a random variable with a normal
distribution with standard deviation Sl), and has no error in angle. Data from the lidar arrives in order
of increasing angle counterclockwise. The first measurement is in the direction of the oX axis.
Scanning occurs instantaneously.
One robot movement is its displacement by vector (X, Y) with standard deviation for each axis (Sx, Sy).
Movement along the X axis and Y axis are independent of each other. The robot's life cycle consists of
alternating scans and movements.
The room is defined as a polygon without self-intersections.
Test examples
https://drive.google.com/file/d/1oKJG0GzDMqiUCQ1ETId_lzJM9FVppJST/view?usp=sharing
Answer: 2.99802 3.03753
https://drive.google.com/file/d/1vY0ifjkmSohlEPi31H95I2mYPEU4Mf4m/view?usp=sharing
Answer: 1.9406 8.00725
https://drive.google.com/file/d/19xgVFV5x9-4pigDZ-E7YT7gIMOYlG_hb/view?usp=sharing
Answer: 1.59266 8.63027
https://drive.google.com/file/d/1CV8pNotTxMZ6HYmOuOL9WuYNE5Ck40Nu/view?usp=sharing
Answer: 3.62952 15.1485
Input
The first line contains number N - the number of sides of the room.
The next line contains N pairs of integers - X, Y coordinates of the room.
The next line contains integers M - the number of robot movements, K - lidar step.
The next line contains 3 real numbers Sl, Sx, Sy.
The next line contains the number 1 or 0. If 1, it is followed by two real numbers - x and y -
the robot's initial coordinates. If 0 - the initial position is unknown.
Then follow M pairs of lines describing the robot's scans and movements.
The first line of each pair contains K real numbers - lidar measurements.
The second line contains two real numbers - robot displacement.
The last line contains K real numbers - lidar measurements after the movement ends.
Output
In one line output two numbers - the robot's coordinates after the movement ends.
Constraints
3 ≤ N ≤ 100
−1000 ≤ X, Y ≤ 1000
0 ≤ M ≤ 1000
36 ≤ K ≤ 3600
0.001 ≤ SL, SX, SY ≤ 0.2
The area of the room is at least 0.1 of the area of the rectangle circumscribed around it.
'''

import math
import random


def main():
    # ==========================================
    # 1. HARVEST DATA (Read all inputs)
    # ==========================================

    # Room geometry
    N = int(input())
    coords = list(map(float, input().split()))
    room = [(coords[i], coords[i + 1]) for i in range(0, 2 * N, 2)]

    # Settings
    M, K = map(int, input().split())
    Sl, Sx, Sy = map(float, input().split())

    # Initial position info
    init_line = list(map(float, input().split()))
    known_start = (init_line[0] == 1)
    start_pos = (init_line[1], init_line[2]) if known_start else (0, 0)


    # Read the Sequence: Scan0 -> (Move1, Scan1) -> (Move2, Scan2)...
    scans = []
    moves = []

    # First scan (Scan 0)
    scans.append(list(map(float, input().split())))

    # Subsequent M moves and scans
    for _ in range(M):
        moves.append(tuple(map(float, input().split())))
        scans.append(list(map(float, input().split())))

    # ==========================================
    # 2. WORK (Particle Filter Logic)
    # ==========================================

    # --- Helpers ---
    angles = [i * (2 * math.pi / K) for i in range(K)]

    def is_inside(x, y):
        """Check if point is inside the room polygon."""
        inside = False
        j = N - 1
        for i in range(N):
            xi, yi = room[i]
            xj, yj = room[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def get_ray_dist(px, py, angle_idx):
        """Ray casting: distance to nearest wall."""
        ray_a = angles[angle_idx]
        dx, dy = math.cos(ray_a), math.sin(ray_a)
        min_dist = 2000.0  # Max coordinate is 1000, so 2000 is safe infinity

        for i in range(N):
            p1 = room[i]
            p2 = room[(i + 1) % N]

            # Segment vector (p1 -> p2)
            sdx, sdy = p2[0] - p1[0], p2[1] - p1[1]
            # Vector from particle to p1
            v1x, v1y = p1[0] - px, p1[1] - py

            # Cross product denominator
            denom = dx * sdy - dy * sdx

            if denom != 0:
                t = (v1x * sdy - v1y * sdx) / denom # Ray Distance
                u = (v1x * dy - v1y * dx) / denom # Wall Fraction
                if t > 0 and 0 <= u <= 1:
                    if t < min_dist: min_dist = t
        return min_dist

    # --- Initialization ---
    NUM_PARTICLES = 400
    particles = []

    # Bounding box for random generation
    xs = [p[0] for p in room]
    ys = [p[1] for p in room]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if known_start:
        for _ in range(NUM_PARTICLES):
            # Create particles with small noise around known start
            particles.append([random.gauss(start_pos[0], 0.5), random.gauss(start_pos[1], 0.5)])
    else:
        # Create particles uniformly inside the room
        while len(particles) < NUM_PARTICLES:
            rx = random.uniform(min_x, max_x)
            ry = random.uniform(min_y, max_y)
            if is_inside(rx, ry):
                particles.append([rx, ry])

    # Optimization: Check only every 10th ray to save time
    check_indices = list(range(0, K, max(1, K // 10)))

    # --- Main Loop ---
    # We have M+1 scans and M moves.
    # Logic: Weight(Scan i) -> Resample -> Move(Move i)

    for i in range(M + 1):
        current_scan = scans[i]

        # 1. Weighting
        weights = []
        for p in particles:
            if not is_inside(p[0], p[1]):
                weights.append(0.0)
                continue

            sq_error = 0.0
            for idx in check_indices:
                sim_dist = get_ray_dist(p[0], p[1], idx)
                obs_dist = current_scan[idx]
                sq_error += (sim_dist - obs_dist) ** 2

            # Gaussian likelihood
            weights.append(math.exp(-sq_error / (2 * Sl ** 2)))

        # Normalize weights
        total = sum(weights)
        if total == 0:
            weights = [1.0 / NUM_PARTICLES] * NUM_PARTICLES
        else:
            weights = [w / total for w in weights]

        # 2. Resampling
        # Pick new particles based on weights
        new_particles = random.choices(particles, weights=weights, k=NUM_PARTICLES)
        particles = [[p[0], p[1]] for p in new_particles]  # Deep copy

        # 3. Moving (only if we have a move left)
        if i < M:
            dx, dy = moves[i]
            for p in particles:
                p[0] += random.gauss(dx, Sx)
                p[1] += random.gauss(dy, Sy)

    # --- Output ---
    avg_x = sum(p[0] for p in particles) / NUM_PARTICLES
    avg_y = sum(p[1] for p in particles) / NUM_PARTICLES
    print(f"{avg_x:.5f} {avg_y:.5f}")


if __name__ == "__main__":
    main()