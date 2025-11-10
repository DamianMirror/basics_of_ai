'''
House Price Prediction
Limits: 1.5 sec., 256 MiB
You, as a future (and possibly current) specialist, have been asked to write a program for predicting house prices in Khmelnytskyi. To solve this task, you chose the K Nearest Neighbors (KNN) algorithm. Perhaps not the best choice, but due to its simplicity in implementation, you will quickly complete this task!
Given N houses whose prices are already known, M houses whose prices need to be predicted, and a number K. The i-th house is characterized by six features:

f_i1 - number of floors
f_i2 - number of rooms
f_i3 - distance to city center
f_i4 - distance to nearest school
f_i5 - distance to nearest shopping center
f_i6 - total area

Distance is given in meters, area in square meters.
Also, for the i-th of N houses, you know its price c_i. To predict the price of each of the M houses, you need to find K most similar houses to it and take their average cost. The similarity metric for houses A and B is defined by the function d = √(Σ(f_ai - f_bi)²) for i from 1 to 6, where f_ai, f_bi are the i-th features of houses A and B respectively. The smaller d, the more similar the houses.
If for some house A there are two houses B and C with the same similarity, then the one with the smaller sequence number is considered more similar to house A (for example B, if B < C).
Your answer will be considered correct if each predicted price differs from the actual corresponding house price by no more than 10⁻⁴.
Input
The first line contains 3 natural numbers N, M, and K - the number of houses with known prices, the number of houses whose prices need to be predicted, and the number of nearest neighbors to consider when predicting prices.
Next, in 2N lines, information about houses with known prices is given. In odd lines, natural numbers f_i1, f_i2, f_i3, f_i4, f_i5, f_i6 are given - 6 features for the i-th house. In even lines, natural number c_i is given - the price for the i-th house.
At the end, in M lines, information about houses whose prices you need to predict is given: f_j1, f_j2, f_j3, f_j4, f_j5, f_j6 - 6 features for the j-th house.
Output
In M lines, output 1 floating-point number - the predicted price for each of the M houses.
Constraints

1 ≤ N, M ≤ 10³
K ≤ N
1 ≤ f_i1, f_i2, f_i3, f_i4, f_i5, f_i6 ≤ 10⁸
1 ≤ c_i ≤ 10⁸
'''

import heapq

def euclidean_distance_squared(features1, features2):
    """Calculate squared Euclidean distance"""
    dist_sq = 0
    for i in range(6):  # We know there are exactly 6 features
        diff = features1[i] - features2[i]
        dist_sq += diff * diff
    return dist_sq

def knn(known_houses, target_features, K):
    if K == len(known_houses):
        return sum(price for _, price in known_houses) / K

    distances = []
    for idx in range(len(known_houses)):
        features, price = known_houses[idx]
        dist_sq = euclidean_distance_squared(features, target_features)
        distances.append((dist_sq, idx, price))

    # Use heap only if K is very small compared to N
    if K < len(known_houses) // 10:  # K < N/10
        k_nearest = heapq.nsmallest(K, distances, key=lambda x: (x[0], x[1]))
        total_price = sum(price for _, _, price in k_nearest)
    else:
        # Use sort for larger K
        distances.sort(key=lambda x: (x[0], x[1]))
        total_price = sum(distances[i][2] for i in range(K))

    return total_price / K

def main():
    N, M, K = map(int, input().split())

    known_houses = []

    for _ in range(N):
        features = list(map(int, input().split()))
        price = int(input())
        known_houses.append((features, price))

    target_houses = [list(map(int, input().split())) for _ in range(M)]

    for target_features in target_houses:
        predicted_price = knn(known_houses, target_features, K)
        print(f"{predicted_price:.6f}")


if __name__ == "__main__":
    main()