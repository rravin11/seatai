#include "seatvision/product/tracker.hpp"

#include "seatvision/product/semantics.hpp"

#include <algorithm>
#include <tuple>

namespace seatvision::product {
namespace {
float iou(const cv::Rect& a, const cv::Rect& b) {
    const int intersection = (a & b).area();
    const int combined = a.area() + b.area() - intersection;
    return combined > 0 ? static_cast<float>(intersection) / combined : 0.0F;
}
}  // namespace

MultiObjectTracker::MultiObjectTracker(ReasonerConfig config) : config_(std::move(config)) {}

EntityRole MultiObjectTracker::classify(const std::string& label, const SemanticPolicy& semantics) const {
    return classify_label(label, semantics);
}

std::vector<Track> MultiObjectTracker::update(int camera_id, Timestamp timestamp, const std::vector<Detection>& detections,
                                              const SemanticPolicy& semantics) {
    auto& tracks = tracks_by_camera_[camera_id];
    for (auto& track : tracks) ++track.missed;
    std::vector<bool> used(tracks.size());
    for (const auto& detection : detections) {
        const EntityRole role = classify(detection.label, semantics);
        if (role == EntityRole::Ignored) continue;
        int best{-1}; float best_iou = config_.association_iou;
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            if (used[i] || tracks[i].label != detection.label) continue;
            const float candidate = iou(tracks[i].box, detection.box);
            if (candidate > best_iou) { best_iou = candidate; best = static_cast<int>(i); }
        }
        if (best >= 0) {
            auto& track = tracks[best];
            track.box = detection.box; track.confidence = detection.score; track.mask = detection.mask;
            ++track.age; track.missed = 0; track.last_seen = timestamp; used[best] = true;
        } else {
            tracks.push_back({next_id_++, camera_id, detection.label, role, detection.box, detection.score, 1, 0, timestamp, detection.mask});
            used.push_back(true);
        }
    }
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [&](const Track& track) { return track.missed > config_.track_ttl_frames; }), tracks.end());
    return tracks;
}

}  // namespace seatvision::product
