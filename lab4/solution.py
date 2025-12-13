import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
OUTPUT_DIR = 'output_videos'
MIN_MATCH_COUNT = 10
MATCH_BUFFER = 20  # <--- NEW: Added to coin radius to get match zone
MIN_FRAMES_TO_CONFIRM = 15


class GlobalCoinTracker:
    def __init__(self):
        self.global_coins = []
        self.candidates = []
        self.accumulated_homography = np.eye(3)
        self.prev_frame_gray = None
        self.prev_kps = None
        self.prev_des = None
        self.camera_path = []
        self.next_id = 1
        self.orb = cv2.ORB_create(nfeatures=10000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def update_camera_movement(self, curr_frame_gray):
        kps, des = self.orb.detectAndCompute(curr_frame_gray, None)
        h, w = curr_frame_gray.shape[:2]

        if self.prev_frame_gray is None:
            self.prev_frame_gray = curr_frame_gray
            self.prev_kps = kps
            self.prev_des = des
            self.camera_path.append((w / 2, h / 2))
            return True

        if des is None or self.prev_des is None:
            return False

        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.prev_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                self.accumulated_homography = np.matmul(self.accumulated_homography, M)
                self.prev_frame_gray = curr_frame_gray
                self.prev_kps = kps
                self.prev_des = des

                center_local = np.array([[[w / 2, h / 2]]], dtype=np.float32)
                center_global = cv2.perspectiveTransform(center_local, self.accumulated_homography)
                gx, gy = center_global[0][0]
                self.camera_path.append((gx, gy))
                return True

        return False

    def process_and_get_ids(self, current_frame_coins):
        frame_ids = []

        # Reset matched status
        for cand in self.candidates:
            cand['matched_this_frame'] = False

        for (cx, cy, r) in current_frame_coins:
            point = np.array([[[cx, cy]]], dtype=np.float32)
            global_point = cv2.perspectiveTransform(point, self.accumulated_homography)
            gx, gy = global_point[0][0]

            matched_id = None

            # 1. Check CONFIRMED coins
            for i, (saved_x, saved_y, saved_r, uid) in enumerate(self.global_coins):
                dist = np.sqrt((gx - saved_x) ** 2 + (gy - saved_y) ** 2)

                # --- DYNAMIC RADIUS CHECK ---
                # Check against the saved coin's specific radius + buffer
                if dist < (saved_r + MATCH_BUFFER):
                    matched_id = uid
                    # Update average position AND radius
                    new_gx = (saved_x * 0.9) + (gx * 0.1)
                    new_gy = (saved_y * 0.9) + (gy * 0.1)
                    new_r = (saved_r * 0.9) + (r * 0.1)  # Smooth the radius too
                    self.global_coins[i] = (new_gx, new_gy, new_r, uid)
                    break

            if matched_id is not None:
                frame_ids.append(matched_id)
                continue

            # 2. Check CANDIDATES
            matched_candidate_idx = -1
            for i, cand in enumerate(self.candidates):
                dist = np.sqrt((gx - cand['x']) ** 2 + (gy - cand['y']) ** 2)

                # --- DYNAMIC RADIUS CHECK ---
                if dist < (cand['r'] + MATCH_BUFFER):
                    matched_candidate_idx = i
                    break

            if matched_candidate_idx != -1:
                cand = self.candidates[matched_candidate_idx]
                cand['seen_count'] += 1
                cand['missing_count'] = 0
                cand['matched_this_frame'] = True

                # Update Candidate
                cand['x'] = (cand['x'] * 0.5) + (gx * 0.5)
                cand['y'] = (cand['y'] * 0.5) + (gy * 0.5)
                cand['r'] = (cand['r'] * 0.5) + (r * 0.5)

                if cand['seen_count'] >= MIN_FRAMES_TO_CONFIRM:
                    new_id = self.next_id
                    self.next_id += 1
                    self.global_coins.append((cand['x'], cand['y'], cand['r'], new_id))
                    frame_ids.append(new_id)
                    self.candidates.pop(matched_candidate_idx)
                else:
                    frame_ids.append(None)
            else:
                # 3. New Candidate
                new_cand = {
                    'x': gx, 'y': gy, 'r': r,
                    'seen_count': 1,
                    'missing_count': 0,
                    'matched_this_frame': True
                }
                self.candidates.append(new_cand)
                frame_ids.append(None)

        # 4. Cleanup
        active_candidates = []
        for cand in self.candidates:
            if not cand.get('matched_this_frame', False):
                cand['missing_count'] += 1
            if cand['missing_count'] < 10:
                active_candidates.append(cand)
        self.candidates = active_candidates

        return frame_ids

    def get_count(self):
        return len(self.global_coins)

    def save_global_map(self, test_num, output_dir):
        if not self.global_coins and not self.camera_path:
            return

        plt.figure(figsize=(10, 10))
        if self.camera_path:
            path_arr = np.array(self.camera_path)
            plt.plot(path_arr[:, 0], path_arr[:, 1], 'orange', linewidth=2, label='Camera Path')
            plt.scatter(path_arr[0, 0], path_arr[0, 1], c='green', marker='^', s=100, label='Start')
            plt.scatter(path_arr[-1, 0], path_arr[-1, 1], c='red', marker='v', s=100, label='End')

        for (gx, gy, r, uid) in self.global_coins:
            # Visualize the actual coin size
            circle = plt.Circle((gx, gy), r, color='blue', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            # Visualize the capture zone (dashed)
            zone = plt.Circle((gx, gy), r + MATCH_BUFFER, color='cyan', fill=False, linestyle='--', linewidth=1)
            plt.gca().add_patch(zone)

            plt.text(gx, gy, str(uid), color='red', fontsize=12, ha='center', va='center', weight='bold')

        plt.title(f'Global Map - Test {test_num} (Total: {len(self.global_coins)})')
        plt.xlabel('Global X')
        plt.ylabel('Global Y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.gca().invert_yaxis()

        save_path = Path(output_dir) / f'map_test_{test_num}.png'
        plt.savefig(save_path)
        plt.close()
        print(f"  Map saved to: {save_path}")


def remove_overlapping_circles(circles):
    """
    FILTERS OVERLAPPING CIRCLES.
    Removes a circle if it physically intersects with a larger circle.
    """
    if not circles:
        return []

    circles = sorted(circles, key=lambda x: x[2], reverse=True)

    filtered = []
    for (x, y, r) in circles:
        is_overlapping = False
        for (fx, fy, fr) in filtered:
            dist = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)

            # --- OVERLAP CHECK ---
            if dist < (fr + r - 15):
                is_overlapping = True
                break

        if not is_overlapping:
            filtered.append((x, y, r))

    return filtered


def detect_coins_in_frame_hough(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=36,
        param1=70,
        param2=35,
        minRadius=17,
        maxRadius=90
    )

    detected = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        detected = [(int(x), int(y), int(r)) for (x, y, r) in circles]

    detected = remove_overlapping_circles(detected)
    return detected


def process_video_and_save(video_path, test_num, expected_count, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_dir).mkdir(exist_ok=True)
    save_path = Path(output_dir) / f'result_test_{test_num}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

    tracker = GlobalCoinTracker()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tracker.update_camera_movement(gray)
        coins = detect_coins_in_frame_hough(frame)
        current_ids = tracker.process_and_get_ids(coins)

        # --- VISUALIZATION ---
        output_frame = frame.copy()

        # Draw Camera Path
        if len(tracker.camera_path) > 1:
            H_inv = np.linalg.inv(tracker.accumulated_homography)
            global_pts = np.array(tracker.camera_path, dtype=np.float32).reshape(-1, 1, 2)
            local_pts = cv2.perspectiveTransform(global_pts, H_inv)
            for j in range(1, len(local_pts)):
                p1 = (int(local_pts[j - 1][0][0]), int(local_pts[j - 1][0][1]))
                p2 = (int(local_pts[j][0][0]), int(local_pts[j][0][1]))
                cv2.line(output_frame, p1, p2, (0, 165, 255), 2)

        cv2.circle(output_frame, (width // 2, height // 2), 5, (0, 0, 255), -1)

        # Draw Coins
        for i, ((x, y, r), coin_id) in enumerate(zip(coins, current_ids)):
            # Dynamic Match Zone for Visualization
            dynamic_radius = r + MATCH_BUFFER

            if coin_id is not None:
                # Confirmed
                cv2.circle(output_frame, (x, y), dynamic_radius, (255, 255, 0), 1)
                cv2.circle(output_frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output_frame, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(output_frame, str(coin_id), (x - 10, y - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # Candidate
                cv2.circle(output_frame, (x, y), r, (0, 255, 255), 1)
                cv2.putText(output_frame, "?", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        info_bg_height = 180
        overlay = output_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, info_bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output_frame, 0.3, 0, output_frame)

        y_offset = 25
        cv2.putText(output_frame, f"Test {test_num} - Dynamic Radius", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(output_frame, f"Frame: {frame_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(output_frame, f"Visible: {len(coins)}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        total = tracker.get_count()
        cv2.putText(output_frame, f"Total CONFIRMED: {total}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        exp_text = f"Expected: {expected_count}" if expected_count else "Expected: N/A"
        cv2.putText(output_frame, exp_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
        y_offset += 25
        cv2.putText(output_frame, f"Radius: Coin Size + {MATCH_BUFFER}px", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        out.write(output_frame)

    cap.release()
    out.release()

    tracker.save_global_map(test_num, output_dir)

    return tracker.get_count()


# --- HELPER FUNCTIONS ---

def load_solutions(solutions_file='solutions'):
    solutions = {}
    if not os.path.exists(solutions_file):
        return {i: 0 for i in range(10)}

    with open(solutions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    solutions[int(parts[0])] = int(parts[1])
    return solutions


def process_all_tests(tests_dir='tests'):
    tests_path = Path(tests_dir)
    results = {}
    solutions = load_solutions()

    print(f"Processing videos and saving to '{OUTPUT_DIR}'...")

    for i in range(10):
        video_path = tests_path / f'test_{i}.mp4'
        if video_path.exists():
            print(f"Processing {video_path.name}...")
            expected_val = solutions.get(i, 0)
            coin_count = process_video_and_save(video_path, test_num=i, expected_count=expected_val,
                                                output_dir=OUTPUT_DIR)
            results[i] = coin_count
            print(f"Test {i}: {coin_count} coins detected (Expected: {expected_val})")
        else:
            print(f"Video not found: {video_path}")

    return results, solutions


def compare_results(results, solutions):
    print("\n" + "=" * 60)
    print("Results Comparison:")
    print("=" * 60)
    print(f"{'Test':<10} {'Detected':<15} {'Expected':<15} {'Match':<10}")
    print("-" * 60)
    total_correct = 0
    for test_num in sorted(results.keys()):
        detected = results[test_num]
        expected = solutions.get(test_num, "N/A")
        match = "PASS" if detected == expected else "FAIL"
        if detected == expected:
            total_correct += 1
        print(f"{test_num:<10} {detected:<15} {expected:<15} {match:<10}")
    print("-" * 60)
    print(f"Accuracy: {total_correct}/{len(results)} tests passed")
    return total_correct


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print("Starting Coin Detection (Dynamic Radius Version)...")
    print()

    results, solutions = process_all_tests()
    correct_count = compare_results(results, solutions)

    if correct_count >= 7:
        print("\n*** SUCCESS! Got 7 or more correct! ***")
    else:
        print(f"\nNeed {7 - correct_count} more correct answers.")