#include "seatvision/product/runtime.hpp"

#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "seatvision/product/renderer.hpp"
#include "seatvision/product/event_sink.hpp"

namespace seatvision::product {
namespace {

cv::Mat waiting_tile(int camera_id, const cv::Size size) {
    cv::Mat tile(size, CV_8UC3, cv::Scalar(20, 20, 20));
    const std::string label = "CAM " + std::to_string(camera_id) + " | waiting for first frame";
    cv::putText(tile, label, {24, size.height / 2}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
    return tile;
}

}  // namespace

Runtime::Runtime(RuntimeConfig config)
    : config_(std::move(config)), tracker_(config_.reasoner), reasoner_(config_.reasoner) {
    for (const auto& camera : config_.cameras) {
        CameraPipeline pipeline;
        pipeline.stream = std::make_unique<CameraStream>(camera);
        pipeline.detector = create_detector(config_.detector);
        cameras_.emplace(camera.sensor_id, std::move(pipeline));
    }
}

int Runtime::run() {
    // Open all Argus sensors before asking either one for frames. Opening a
    // second sensor while a first worker is already streaming can deadlock
    // nvargus-daemon on some JetPack releases.
    for (auto& [id, pipeline] : cameras_) {
        if (!pipeline.stream->prepare()) { std::cerr << "Unable to configure camera " << id << '\n'; return 2; }
    }
    for (auto& [id, pipeline] : cameras_) {
        if (!pipeline.stream->start()) { std::cerr << "Unable to start camera " << id << '\n'; return 2; }
        std::cout << "Camera " << id << " started with " << pipeline.detector->name() << std::endl;
    }
    SceneRenderer renderer;
    EventSink events(config_.event_log);
    const cv::Size preview_size{config_.preview_tile_width, config_.preview_tile_height};
    if (config_.show_window) {
        cv::namedWindow("SeatVision Product", cv::WINDOW_NORMAL);
        cv::resizeWindow("SeatVision Product", preview_size.width * static_cast<int>(config_.cameras.size()), preview_size.height);
    }
    auto fps_since = Clock::now(); int displayed{}; double display_fps{};
    bool running = true;
    while (running) {
        bool display_dirty = false;
        // Iterate in configuration order rather than unordered-map order: CAM 0
        // always occupies the same panel, independent of frame arrival timing.
        for (const auto& camera : config_.cameras) {
            const int camera_id = camera.sensor_id;
            auto& pipeline = cameras_.at(camera_id);
            if (pipeline.inference.has_value() && pipeline.inference->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                const auto result = pipeline.inference->get(); pipeline.inference.reset();
                if (!pipeline.logged_first_inference) {
                    std::cout << "Camera " << camera_id << " first TensorRT result: " << result.detections.size() << " detections" << std::endl;
                    pipeline.logged_first_inference = true;
                }
                pipeline.last_detections = result.detections;
                pipeline.last_tracks = tracker_.update(camera_id, result.timestamp, pipeline.last_detections, config_.semantics);
                pipeline.latest_scene = reasoner_.update(camera_id, result.timestamp, pipeline.last_tracks);
                for (const auto& event : reasoner_.take_events()) events.publish(event);
                display_dirty = true;
            }
            const auto frame = pipeline.stream->latest_after(pipeline.consumed_sequence);
            if (!frame.has_value()) continue;
            pipeline.consumed_sequence = frame->sequence;
            pipeline.display_frame = *frame;
            display_dirty = true;
            if (!pipeline.inference.has_value() && frame->sequence % static_cast<std::uint64_t>(config_.inference_interval) == 0) {
                Frame inference_frame = *frame;
                Detector* detector = pipeline.detector.get();
                pipeline.inference.emplace(std::async(std::launch::async, [detector, inference_frame = std::move(inference_frame)]() mutable {
                    return CameraPipeline::InferenceResult{inference_frame.sequence, inference_frame.captured_at, detector->infer(inference_frame.image)};
                }));
            }
            // State is always computed at the observation timestamp. It is only rendered while fresh,
            // preventing a delayed model result from being represented as current perception.
        }
        if (!display_dirty) {
            if (config_.show_window) {
                const int key = cv::waitKey(1);
                if (key == 'q' || key == 27) running = false;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            continue;
        }
        ++displayed;
        const auto now = Clock::now();
        if (now - fps_since >= std::chrono::seconds(1)) { display_fps = displayed / std::chrono::duration<double>(now - fps_since).count(); displayed = 0; fps_since = now; }
        if (config_.show_window) {
            std::vector<cv::Mat> tiles;
            tiles.reserve(config_.cameras.size());
            for (const auto& camera : config_.cameras) {
                const int camera_id = camera.sensor_id;
                auto& pipeline = cameras_.at(camera_id);
                if (!pipeline.display_frame.has_value()) {
                    tiles.push_back(waiting_tile(camera_id, preview_size));
                    continue;
                }
                SceneSnapshot display_scene = pipeline.latest_scene;
                if (Clock::now() - display_scene.timestamp > std::chrono::milliseconds(850)) {
                    display_scene = {camera_id, pipeline.display_frame->captured_at, {}, {}};
                }
                cv::Mat tile = renderer.render(*pipeline.display_frame, display_scene, display_fps);
                if (tile.size() != preview_size) cv::resize(tile, tile, preview_size, 0.0, 0.0, cv::INTER_AREA);
                tiles.push_back(std::move(tile));
            }
            cv::Mat mosaic; cv::hconcat(tiles, mosaic); cv::imshow("SeatVision Product", mosaic);
            const int key = cv::waitKey(1); if (key == 'q' || key == 27) running = false;
        }
    }
    for (auto& [_, pipeline] : cameras_) pipeline.stream->stop();
    return 0;
}

}  // namespace seatvision::product
