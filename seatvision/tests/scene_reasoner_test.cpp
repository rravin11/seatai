#include <cassert>
#include <chrono>
#include <iostream>

#include "seatvision/product/scene_reasoner.hpp"
#include "seatvision/product/tracker.hpp"

using namespace seatvision::product;

int main() {
    ReasonerConfig config;
    config.minimum_seat_track_age = 1;
    config.occupied_confirmation = std::chrono::milliseconds(0);
    config.claimed_confirmation = std::chrono::milliseconds(0);
    config.available_confirmation = std::chrono::milliseconds(0);
    SemanticPolicy policy{{"human"}, {"seatable_surface"}, {}};
    MultiObjectTracker tracker(config);
    DynamicSeatReasoner reasoner(config);
    const auto t0 = Clock::now();

    auto tracks = tracker.update(7, t0, {{1, "seatable_surface", 0.95F, {100, 100, 200, 200}, std::nullopt}}, policy);
    auto scene = reasoner.update(7, t0, tracks);
    assert(scene.seats.size() == 1);
    assert(scene.seats.front().state == Occupancy::Available);

    tracks = tracker.update(7, t0 + std::chrono::milliseconds(1), {
        {1, "seatable_surface", 0.95F, {100, 100, 200, 200}, std::nullopt},
        {0, "human", 0.98F, {115, 165, 150, 160}, std::nullopt},
    }, policy);
    scene = reasoner.update(7, t0 + std::chrono::milliseconds(1), tracks);
    assert(scene.seats.size() == 1);
    assert(scene.seats.front().state == Occupancy::Occupied);

    tracks = tracker.update(7, t0 + std::chrono::milliseconds(2), {
        {1, "seatable_surface", 0.95F, {100, 100, 200, 200}, std::nullopt},
        {43, "unseen_future_object", 0.90F, {120, 170, 80, 80}, std::nullopt},
    }, policy);
    scene = reasoner.update(7, t0 + std::chrono::milliseconds(2), tracks);
    assert(scene.seats.front().state == Occupancy::Claimed);
    std::cout << "dynamic seat discovery and unknown-object reasoning: PASS\n";
}
