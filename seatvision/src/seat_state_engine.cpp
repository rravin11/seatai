#include "seatvision/seat_state_engine.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace seatvision {
namespace {
std::chrono::milliseconds dwell_for(SeatState state) {
    using namespace std::chrono_literals;
    switch (state) {
        case SeatState::Occupied: return 750ms;
        case SeatState::Claimed: return 1200ms;
        case SeatState::Available: return 2500ms;
        default: return 500ms;
    }
}
}  // namespace

SeatStateEngine::SeatStateEngine(std::vector<SeatZone> zones) : zones_(std::move(zones)) {
    const auto now = std::chrono::steady_clock::now();
    for (const auto& zone : zones_) memory_[zone.id].candidate_since = now;
}

bool SeatStateEngine::overlaps(const std::vector<cv::Point>& zone, const cv::Rect& box) {
    const cv::Point center(box.x + box.width / 2, box.y + box.height / 2);
    if (cv::pointPolygonTest(zone, center, false) >= 0) return true;
    for (const cv::Point& point : zone) if (box.contains(point)) return true;
    return false;
}

SeatState SeatStateEngine::observe(const std::vector<cv::Point>& zone, const std::vector<Detection>& detections,
                                   float& confidence) const {
    float person = 0.0F, item = 0.0F;
    for (const auto& detection : detections) {
        if (!overlaps(zone, detection.box)) continue;
        if (detection.kind == ObjectKind::Person) person = std::max(person, detection.confidence);
        if (detection.kind == ObjectKind::Backpack || detection.kind == ObjectKind::Bottle) item = std::max(item, detection.confidence);
    }
    if (person > 0.0F) { confidence = person; return SeatState::Occupied; }
    if (item > 0.0F) { confidence = item; return SeatState::Claimed; }
    confidence = 0.65F;
    return SeatState::Available;
}

std::vector<SeatEstimate> SeatStateEngine::update(int camera_id, const cv::Size& frame_size,
                                                   const std::vector<Detection>& detections) {
    const auto now = std::chrono::steady_clock::now();
    std::vector<SeatEstimate> estimates;
    for (const auto& seat : zones_) {
        if (seat.camera_id != camera_id) continue;
        std::vector<cv::Point> polygon;
        for (const auto& point : seat.normalized_polygon)
            polygon.emplace_back(cvRound(point.x * frame_size.width), cvRound(point.y * frame_size.height));
        float confidence{};
        const SeatState observed = observe(polygon, detections, confidence);
        auto& memory = memory_[seat.id];
        if (observed != memory.candidate) { memory.candidate = observed; memory.candidate_since = now; }
        if (now - memory.candidate_since >= dwell_for(observed)) memory.stable = observed;
        memory.confidence = confidence;
        estimates.push_back({seat.id, memory.stable, memory.confidence});
    }
    return estimates;
}

const char* SeatStateEngine::state_name(SeatState state) {
    switch (state) {
        case SeatState::Available: return "AVAILABLE";
        case SeatState::Occupied: return "OCCUPIED";
        case SeatState::Claimed: return "CLAIMED";
        default: return "UNCERTAIN";
    }
}

cv::Scalar SeatStateEngine::state_color(SeatState state) {
    switch (state) {
        case SeatState::Available: return {70, 190, 70};
        case SeatState::Occupied: return {40, 70, 230};
        case SeatState::Claimed: return {40, 180, 240};
        default: return {150, 150, 150};
    }
}

void SeatStateEngine::draw(cv::Mat& frame, int camera_id) const {
    for (const auto& seat : zones_) {
        if (seat.camera_id != camera_id) continue;
        std::vector<cv::Point> polygon;
        for (const auto& p : seat.normalized_polygon) polygon.emplace_back(cvRound(p.x * frame.cols), cvRound(p.y * frame.rows));
        const auto& memory = memory_.at(seat.id);
        const auto color = state_color(memory.stable);
        cv::polylines(frame, polygon, true, color, 2, cv::LINE_AA);
        const auto anchor = polygon.front();
        cv::putText(frame, seat.id + " " + state_name(memory.stable), anchor + cv::Point(0, -8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv::LINE_AA);
    }
}

}  // namespace seatvision
