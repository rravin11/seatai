#include "seatvision/product/renderer.hpp"

#include <format>

#include <opencv2/imgproc.hpp>

namespace seatvision::product {

cv::Scalar SceneRenderer::color_for(EntityRole role) {
    switch (role) {
        case EntityRole::Person: return {255, 100, 40};
        case EntityRole::Seat: return {80, 220, 80};
        case EntityRole::Object: return {40, 190, 245};
        case EntityRole::Ignored: return {130, 130, 130};
        default: return {200, 200, 200};
    }
}

cv::Scalar SceneRenderer::color_for(Occupancy state) {
    switch (state) {
        case Occupancy::Available: return {80, 210, 80};
        case Occupancy::Occupied: return {40, 60, 230};
        case Occupancy::Claimed: return {30, 190, 240};
        case Occupancy::Occluded: return {160, 140, 130};
        default: return {170, 170, 170};
    }
}

const char* SceneRenderer::name_for(Occupancy state) {
    switch (state) {
        case Occupancy::Available: return "AVAILABLE";
        case Occupancy::Occupied: return "OCCUPIED";
        case Occupancy::Claimed: return "ITEM";
        case Occupancy::Occluded: return "OCCLUDED";
        default: return "DISCOVERING";
    }
}

cv::Mat SceneRenderer::render(const Frame& frame, const SceneSnapshot& scene, double display_fps) const {
    cv::Mat output = frame.image.clone();
    for (const auto& track : scene.tracks) {
        const auto color = color_for(track.role);
        cv::rectangle(output, track.box, color, 2, cv::LINE_AA);
        cv::putText(output, std::format("{} #{} {:.0f}%", track.label, track.id, track.confidence * 100),
                    track.box.tl() + cv::Point(0, -5), cv::FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv::LINE_AA);
    }
    for (const auto& seat : scene.seats) {
        const auto color = color_for(seat.state);
        cv::rectangle(output, seat.support_region, color, 3, cv::LINE_AA);
        cv::putText(output, std::format("{} {} {:.0f}%", seat.id, name_for(seat.state), seat.confidence * 100),
                    seat.support_region.tl() + cv::Point(0, 22), cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv::LINE_AA);
    }
    cv::putText(output, std::format("CAM {} | {:.1f} display FPS | dynamic seats: {}", frame.camera_id, display_fps, scene.seats.size()),
                {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.65, {255, 255, 255}, 2, cv::LINE_AA);
    return output;
}

}  // namespace seatvision::product
