#include "seatvision/argus_camera.hpp"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <cstring>
#include <sstream>

namespace seatvision {

ArgusCamera::ArgusCamera(int sensor_id, int width, int height, int fps, bool rotate_180)
    : sensor_id_(sensor_id), width_(width), height_(height), fps_(fps), rotate_180_(rotate_180) {}

ArgusCamera::~ArgusCamera() {
    if (appsink_ != nullptr) gst_object_unref(appsink_);
    if (pipeline_ != nullptr) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
    }
}

bool ArgusCamera::open() {
    static std::once_flag init_once;
    std::call_once(init_once, [] { gst_init(nullptr, nullptr); });
    std::ostringstream pipeline;
    pipeline << "nvarguscamerasrc sensor-id=" << sensor_id_
             << " ! video/x-raw(memory:NVMM),width=" << width_ << ",height=" << height_
             << ",framerate=" << fps_ << "/1,format=NV12"
             << " ! nvvidconv" << (rotate_180_ ? " flip-method=2" : "")
             << " ! video/x-raw,format=BGRx"
             << " ! appsink name=seatvision_sink drop=true max-buffers=1 sync=false";
    GError* error = nullptr;
    pipeline_ = gst_parse_launch(pipeline.str().c_str(), &error);
    if (pipeline_ == nullptr || error != nullptr) {
        if (error != nullptr) g_error_free(error);
        return false;
    }
    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "seatvision_sink");
    if (appsink_ == nullptr || !GST_IS_APP_SINK(appsink_)) return false;
    gst_app_sink_set_emit_signals(GST_APP_SINK(appsink_), false);
    gst_app_sink_set_drop(GST_APP_SINK(appsink_), true);
    gst_app_sink_set_max_buffers(GST_APP_SINK(appsink_), 1);
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) return false;
    return true;
}

bool ArgusCamera::read(cv::Mat& frame) {
    if (appsink_ == nullptr) return false;
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_), 200000000);
    if (sample == nullptr) return false;
    const GstCaps* caps = gst_sample_get_caps(sample);
    GstVideoInfo info{};
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstVideoFrame video_frame{};
    const bool mapped = caps != nullptr && gst_video_info_from_caps(&info, caps) && buffer != nullptr &&
                        gst_video_frame_map(&video_frame, &info, buffer, GST_MAP_READ);
    if (mapped) {
        cv::Mat bgrx(GST_VIDEO_FRAME_HEIGHT(&video_frame), GST_VIDEO_FRAME_WIDTH(&video_frame), CV_8UC4,
                     GST_VIDEO_FRAME_PLANE_DATA(&video_frame, 0), GST_VIDEO_FRAME_PLANE_STRIDE(&video_frame, 0));
        cv::cvtColor(bgrx, frame, cv::COLOR_BGRA2BGR);
        gst_video_frame_unmap(&video_frame);
    }
    gst_sample_unref(sample);
    return mapped && !frame.empty();
}

}  // namespace seatvision
