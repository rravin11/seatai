#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void report_bus_messages(GstBus* bus) {
    while (GstMessage* message = gst_bus_pop_filtered(
               bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING))) {
        GError* error = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(message, &error, &debug);
        std::cerr << "GStreamer " << (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR ? "error" : "warning")
                  << ": " << (error == nullptr ? "unknown" : error->message)
                  << (debug == nullptr ? "" : std::string(" | ") + debug) << '\n';
        if (error != nullptr) g_error_free(error);
        g_free(debug);
        gst_message_unref(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int sensor_id = argc > 1 ? std::atoi(argv[1]) : 0;
    const int wanted_frames = argc > 2 ? std::atoi(argv[2]) : 60;
    gst_init(&argc, &argv);

    std::ostringstream description;
    description << "nvarguscamerasrc sensor-id=" << sensor_id
                << " ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12"
                << " ! nvvidconv ! video/x-raw,format=BGRx"
                << " ! appsink name=seatvision_probe_sink drop=true max-buffers=1 sync=false";
    std::cout << "Pipeline: " << description.str() << '\n';

    GError* parse_error = nullptr;
    GstElement* pipeline = gst_parse_launch(description.str().c_str(), &parse_error);
    if (pipeline == nullptr || parse_error != nullptr) {
        std::cerr << "Unable to create pipeline: "
                  << (parse_error == nullptr ? "unknown error" : parse_error->message) << '\n';
        if (parse_error != nullptr) g_error_free(parse_error);
        return 2;
    }
    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "seatvision_probe_sink");
    if (sink == nullptr || !GST_IS_APP_SINK(sink)) {
        std::cerr << "Pipeline did not create an appsink.\n";
        if (sink != nullptr) gst_object_unref(sink);
        gst_object_unref(pipeline);
        return 2;
    }

    const auto change = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (change == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Unable to start pipeline.\n";
        gst_object_unref(sink);
        gst_object_unref(pipeline);
        return 2;
    }
    GstBus* bus = gst_element_get_bus(pipeline);
    int received{};
    bool format_reported{};
    while (received < wanted_frames) {
        report_bus_messages(bus);
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 3 * GST_SECOND);
        if (sample == nullptr) {
            std::cerr << "Timed out waiting for a sample after " << received << " frames.\n";
            report_bus_messages(bus);
            break;
        }
        const GstCaps* caps = gst_sample_get_caps(sample);
        GstVideoInfo info{};
        if (caps == nullptr || !gst_video_info_from_caps(&info, caps)) {
            std::cerr << "Received a sample with invalid caps.\n";
            gst_sample_unref(sample);
            break;
        }
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstVideoFrame frame{};
        if (buffer == nullptr || !gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ)) {
            std::cerr << "Unable to map frame " << received << ".\n";
            gst_sample_unref(sample);
            break;
        }
        if (!format_reported) {
            gchar* cap_text = gst_caps_to_string(caps);
            std::cout << "First sample: caps=" << cap_text
                      << ", width=" << GST_VIDEO_FRAME_WIDTH(&frame)
                      << ", height=" << GST_VIDEO_FRAME_HEIGHT(&frame)
                      << ", stride=" << GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0)
                      << ", buffer_bytes=" << gst_buffer_get_size(buffer) << '\n';
            g_free(cap_text);
            format_reported = true;
        }
        gst_video_frame_unmap(&frame);
        gst_sample_unref(sample);
        ++received;
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    if (received != wanted_frames) return 1;
    std::cout << "PASS sensor=" << sensor_id << " frames=" << received << '\n';
    return 0;
}
