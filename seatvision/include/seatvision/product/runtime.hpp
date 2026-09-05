#pragma once

#include <memory>
#include <future>
#include <optional>
#include <unordered_map>

#include "seatvision/product/camera_stream.hpp"
#include "seatvision/product/config.hpp"
#include "seatvision/product/detector.hpp"
#include "seatvision/product/scene_reasoner.hpp"
#include "seatvision/product/tracker.hpp"

namespace seatvision::product {

class Runtime {
public:
    explicit Runtime(RuntimeConfig config);
    int run();

private:
    struct CameraPipeline {
        struct InferenceResult {
            std::uint64_t sequence{};
            Timestamp timestamp{};
            std::vector<Detection> detections;
        };
        std::unique_ptr<CameraStream> stream;
        std::unique_ptr<Detector> detector;
        std::uint64_t consumed_sequence{};
        std::uint64_t latest_inferred_sequence{};
        // The compositor retains one image per camera. A camera arriving a few
        // milliseconds later must never make its neighbor disappear from view.
        std::optional<Frame> display_frame;
        std::vector<Detection> last_detections;
        std::vector<Track> last_tracks;
        SceneSnapshot latest_scene;
        std::optional<std::future<InferenceResult>> inference;
        bool logged_first_inference{false};
    };

    RuntimeConfig config_;
    MultiObjectTracker tracker_;
    DynamicSeatReasoner reasoner_;
    std::unordered_map<int, CameraPipeline> cameras_;
};

}  // namespace seatvision::product
