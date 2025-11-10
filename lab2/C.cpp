#include <iostream>
#include <vector>
#include <string>
#include <sstream>      // For std::stringstream
#include <algorithm>    // For std::sort
#include <tuple>        // For std::tuple
#include <iomanip>      // For std::setprecision


long long euclidean_distance_squared(const std::vector<int>& features1, const std::vector<int>& features2) {
    long long dist = 0;
    for (size_t i = 0; i < features1.size(); ++i) {
        long long diff = (long long)features1[i] - features2[i];
        dist += diff * diff;
    }
    return dist;
}

double knn(const std::vector<std::pair<std::vector<int>, int>>& known_houses,
           const std::vector<int>& target_features, int K) {

    std::vector<std::tuple<long long, int, int>> distances;

    for (size_t i = 0; i < known_houses.size(); ++i) {
        const auto& features = known_houses[i].first;
        const auto& price = known_houses[i].second;

        long long dist = euclidean_distance_squared(features, target_features);
        distances.emplace_back(dist, (int)i, price);
    }

    std::partial_sort(distances.begin(), distances.begin() + K, distances.end());

    double price_sum = 0.0;
    for (int i = 0; i < K; ++i) {
        price_sum += std::get<2>(distances[i]);
    }

    return price_sum / K;
}

std::vector<int> read_feature_line() {
    std::vector<int> features;
    std::string line;

    std::getline(std::cin >> std::ws, line);

    std::stringstream ss(line);
    int feature;
    while (ss >> feature) {
        features.push_back(feature);
    }
    return features;
}

int main() {
    int N, M, K;
    std::cin >> N >> M >> K;

    std::vector<std::pair<std::vector<int>, int>> known_houses(N);
    for (int i = 0; i < N; ++i) {
        known_houses[i].first = read_feature_line();
        std::cin >> known_houses[i].second;
    }

    std::vector<std::vector<int>> target_houses(M);
    for (int i = 0; i < M; ++i) {
        target_houses[i] = read_feature_line();
    }

    std::cout << std::fixed << std::setprecision(6);

    for (const auto& target_features : target_houses) {
        double predicted_price = knn(known_houses, target_features, K);
        std::cout << predicted_price << "\n";
    }

    return 0;
}