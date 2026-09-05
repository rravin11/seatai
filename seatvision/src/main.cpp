#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "seatvision/argus_camera.hpp"
#include "seatvision/async_detector.hpp"
#include "seatvision/seat_state_engine.hpp"

namespace {
using seatvision::Detection;
using seatvision::ObjectKind;

struct Options { std::string model{"models/yolov8n.onnx"}; int width{1280}; int height{720}; int fps{30}; int detect_every{5}; };

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* name) -> const char* { if (i + 1 >= argc) throw std::runtime_error(std::string("Missing ") + name); return argv[++i]; };
        if (arg == "--model") options.model = value("model path");
        else if (arg == "--width") options.width = std::atoi(value("width"));
        else if (arg == "--height") options.height = std::atoi(value("height"));
        else if (arg == "--fps") options.fps = std::atoi(value("fps"));
        else if (arg == "--detect-every") options.detect_every = std::max(1, std::atoi(value("interval")));
        else if (arg == "--help") { std::cout << "seatvision [--model models/yolov8n.onnx] [--width 1280] [--height 720] [--fps 30] [--detect-every 5]\n"; std::exit(0); }
        else throw std::runtime_error("Unknown option: " + arg);
    }
    return options;
}

cv::Scalar detection_color(ObjectKind kind) {
    if (kind == ObjectKind::Person) return {255, 90, 40};
    if (kind == ObjectKind::Chair) return {80, 220, 80};
    if (kind == ObjectKind::Backpack) return {40, 200, 240};
    if (kind == ObjectKind::Bottle) return {220, 100, 230};
    return {180, 180, 180};
}

void draw_detections(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& detection : detections) {
        const auto color = detection_color(detection.kind);
        cv::rectangle(frame, detection.box, color, 2, cv::LINE_AA);
        cv::putText(frame, detection.label + " " + cv::format("%.0f%%", detection.confidence * 100),
                    detection.box.tl() + cv::Point(0, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv::LINE_AA);
    }
}

std::vector<seatvision::SeatZone> demo_zones() {
    // Replace with calibrated polygons once the cameras are mounted. These are intentionally visible in the demo.
    return {
        {"cam0-seat-a", 0, {{0.04F, 0.58F}, {0.28F, 0.58F}, {0.30F, 0.96F}, {0.03F, 0.96F}}},
        {"cam0-seat-b", 0, {{0.37F, 0.58F}, {0.61F, 0.58F}, {0.63F, 0.96F}, {0.36F, 0.96F}}},
        {"cam1-seat-a", 1, {{0.04F, 0.58F}, {0.28F, 0.58F}, {0.30F, 0.96F}, {0.03F, 0.96F}}},
        {"cam1-seat-b", 1, {{0.37F, 0.58F}, {0.61F, 0.58F}, {0.63F, 0.96F}, {0.36F, 0.96F}}},
    };
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        seatvision::ArgusCamera camera0(0, options.width, options.height, options.fps);
        seatvision::ArgusCamera camera1(1, options.width, options.height, options.fps);
        if (!camera0.open() || !camera1.open()) { std::cerr << "Failed to open both Argus cameras.\n"; return 1; }

        seatvision::AsyncYoloDetector detector0(options.model, 0.35F, 0.45F);
        seatvision::AsyncYoloDetector detector1(options.model, 0.35F, 0.45F);
        seatvision::SeatStateEngine states(demo_zones());
        cv::namedWindow("SeatVision MVP", cv::WINDOW_NORMAL);
        std::cout << "SeatVision running. Press q or Esc to exit. Detection: " << (detector0.ready() ? "YOLO enabled" : "camera-only") << '\n';

        cv::Mat frame0, frame1; std::uint64_t frame_number{};
        auto fps_at = std::chrono::steady_clock::now(); int displayed{}; double fps{};
        while (camera0.read(frame0) && camera1.read(frame1)) {
            ++frame_number;
            if (detector0.ready() && frame_number % static_cast<std::uint64_t>(options.detect_every) == 0) { detector0.submit(frame0); detector1.submit(frame1); }
            const auto result0 = detector0.latest(); const auto result1 = detector1.latest();
            std::vector<Detection> detections0 = result0 ? result0->detections : std::vector<Detection>{};
            std::vector<Detection> detections1 = result1 ? result1->detections : std::vector<Detection>{};
            states.update(0, frame0.size(), detections0); states.update(1, frame1.size(), detections1);
            draw_detections(frame0, detections0); draw_detections(frame1, detections1);
            states.draw(frame0, 0); states.draw(frame1, 1);
            ++displayed; const auto now = std::chrono::steady_clock::now();
            if (now - fps_at >= std::chrono::seconds(1)) { fps = displayed / std::chrono::duration<double>(now - fps_at).count(); displayed = 0; fps_at = now; }
            cv::putText(frame0, "CAM 0 | display " + cv::format("%.1f fps", fps), {12, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.75, {255,255,255}, 2, cv::LINE_AA);
            cv::putText(frame1, "CAM 1 | q: quit", {12, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.75, {255,255,255}, 2, cv::LINE_AA);
            cv::Mat mosaic; cv::hconcat(frame0, frame1, mosaic); cv::imshow("SeatVision MVP", mosaic);
            const int key = cv::waitKey(1); if (key == 'q' || key == 27) break;
        }
    } catch (const std::exception& error) { std::cerr << "SeatVision error: " << error.what() << '\n'; return 1; }
    return 0;
}
