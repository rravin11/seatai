#include "seatvision/product/scene_reasoner.hpp"

#include <algorithm>
#include <format>

namespace seatvision::product {

DynamicSeatReasoner::DynamicSeatReasoner(ReasonerConfig config) : config_(std::move(config)) {}

float DynamicSeatReasoner::intersection_over_union(const cv::Rect& a, const cv::Rect& b) {
    const int intersect = (a & b).area();
    const int total = a.area() + b.area() - intersect;
    return total > 0 ? static_cast<float>(intersect) / total : 0.0F;
}

float DynamicSeatReasoner::overlap_ratio(const cv::Rect& container, const cv::Rect& candidate) {
    const int overlap = (container & candidate).area();
    return candidate.area() > 0 ? static_cast<float>(overlap) / candidate.area() : 0.0F;
}

Occupancy DynamicSeatReasoner::choose_evidence(const DynamicSeat& seat, const std::vector<Track>& tracks, float& confidence) {
    float person_evidence{};
    float object_evidence{};
    for (const auto& track : tracks) {
        if (track.missed != 0) continue;  // Never treat stale tracks as current evidence.
        if (track.role == EntityRole::Seat || track.role == EntityRole::Ignored) continue;
        const float overlap = overlap_ratio(seat.support_region, track.box);
        const float relation = std::max(overlap, intersection_over_union(seat.support_region, track.box));
        if (relation < 0.12F) continue;
        const float evidence = relation * track.confidence;
        if (track.role == EntityRole::Person) person_evidence = std::max(person_evidence, evidence);
        else if (track.role == EntityRole::Object) object_evidence = std::max(object_evidence, evidence);
    }
    if (person_evidence > 0.12F) { confidence = person_evidence; return Occupancy::Occupied; }
    if (object_evidence > 0.10F) { confidence = object_evidence; return Occupancy::Claimed; }
    confidence = 0.55F;
    return Occupancy::Available;
}

std::chrono::milliseconds DynamicSeatReasoner::confirmation_for(Occupancy state) const {
    switch (state) {
        case Occupancy::Occupied: return config_.occupied_confirmation;
        case Occupancy::Claimed: return config_.claimed_confirmation;
        case Occupancy::Available: return config_.available_confirmation;
        default: return std::chrono::milliseconds(500);
    }
}

SceneSnapshot DynamicSeatReasoner::update(int camera_id, Timestamp timestamp, const std::vector<Track>& tracks) {
    std::unordered_map<std::string, bool> alive;
    for (const auto& track : tracks) {
        if (track.camera_id != camera_id || track.role != EntityRole::Seat || track.age < config_.minimum_seat_track_age) continue;
        const std::string key = std::format("camera-{}:seat-track-{}", camera_id, track.id);
        alive[key] = true;
        auto [it, inserted] = seats_.try_emplace(key);
        auto& memory = it->second;
        if (inserted) {
            memory.seat = {key, camera_id, track.id, track.box, Occupancy::Unknown, track.confidence, timestamp, timestamp};
            memory.candidate_since = timestamp;
        }
        // A lower-center chair region is a conservative default seat surface. A segmentation
        // backend can replace this geometry with its mask-derived support surface without
        // changing the rest of the reasoning pipeline.
        const int top = track.box.y + static_cast<int>(track.box.height * 0.38F);
        memory.seat.support_region = cv::Rect(track.box.x, top, track.box.width, std::max(1, track.box.br().y - top));
        float confidence{};
        const Occupancy observation = choose_evidence(memory.seat, tracks, confidence);
        if (observation != memory.candidate) { memory.candidate = observation; memory.candidate_since = timestamp; }
        if (timestamp - memory.candidate_since >= confirmation_for(observation) && memory.seat.state != observation) {
            const Occupancy prior = memory.seat.state;
            memory.seat.state = observation;
            pending_events_.push_back({"seat_state_changed", memory.seat.id, camera_id, prior, observation, confidence, timestamp});
        }
        memory.seat.confidence = confidence;
        memory.seat.last_updated = timestamp;
    }
    std::vector<DynamicSeat> seats;
    for (auto it = seats_.begin(); it != seats_.end();) {
        if (it->second.seat.camera_id != camera_id) { ++it; continue; }
        if (!alive.contains(it->first)) {
            // Do not emit 'available' from absence. Preserve the discovered entity briefly
            // and explicitly mark it occluded while the chair detector has no evidence.
            it->second.seat.state = Occupancy::Occluded;
            it->second.seat.confidence = 0.0F;
            if (timestamp - it->second.seat.last_updated > std::chrono::seconds(8)) { it = seats_.erase(it); continue; }
        }
        seats.push_back(it->second.seat);
        ++it;
    }
    return {camera_id, timestamp, tracks, seats};
}

std::vector<SeatEvent> DynamicSeatReasoner::take_events() {
    std::vector<SeatEvent> result;
    result.swap(pending_events_);
    return result;
}

}  // namespace seatvision::product
