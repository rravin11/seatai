#include "seatvision_dataset/batch.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "seatvision_dataset/detector.hpp"
#include "seatvision_dataset/evaluation.hpp"

namespace seatvision::dataset {
namespace {

using Path = std::filesystem::path;

struct ImageInput {
    Path absolute;
    Path relative;
};

struct GroundTruthIndex {
    std::unordered_map<std::string, std::vector<GroundTruthBox>> boxes_by_image;
    std::unordered_set<std::string> image_keys;
    std::map<std::string, std::size_t> unsupported_labels;
    std::size_t ignored_crowd_annotations{};
};

struct BatchRecord {
    std::string image_key;
    int width{};
    int height{};
    double inference_ms{};
    std::vector<Detection> detections;
    std::string annotated_relative;
};

std::string normalized_label(std::string value) {
    for (char& character : value) {
        if (character == '_') character = ' ';
        else character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    std::string compact;
    bool previous_space{};
    for (const char character : value) {
        const bool space = std::isspace(static_cast<unsigned char>(character));
        if (space && (compact.empty() || previous_space)) continue;
        compact.push_back(space ? ' ' : character);
        previous_space = space;
    }
    if (!compact.empty() && compact.back() == ' ') compact.pop_back();
    return compact;
}

std::string key_for(const Path& path) {
    return path.lexically_normal().generic_string();
}

Path comparable_path(const Path& path) {
    std::error_code error;
    const Path canonical = std::filesystem::weakly_canonical(path, error);
    if (!error) return canonical.lexically_normal();
    error.clear();
    const Path absolute = std::filesystem::absolute(path, error);
    return error ? path.lexically_normal() : absolute.lexically_normal();
}

bool is_within(const Path& candidate, const Path& parent) {
    auto candidate_part = candidate.begin();
    for (auto parent_part = parent.begin(); parent_part != parent.end(); ++parent_part, ++candidate_part) {
        if (candidate_part == candidate.end() || *candidate_part != *parent_part) return false;
    }
    return true;
}

bool is_safe_relative_path(const Path& path) {
    if (path.empty() || path.is_absolute()) return false;
    return std::none_of(path.begin(), path.end(), [](const auto& part) { return part == ".."; });
}

bool is_image_file(const Path& path) {
    static const std::unordered_set<std::string> extensions{
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"};
    return extensions.contains(normalized_label(path.extension().string()));
}

std::vector<ImageInput> discover_images(const Path& input, const bool recursive) {
    std::error_code error;
    if (std::filesystem::is_regular_file(input, error)) {
        if (!is_image_file(input)) throw std::runtime_error("Input is not a supported image: " + input.string());
        return {{comparable_path(input), input.filename()}};
    }
    error.clear();
    if (!std::filesystem::is_directory(input, error)) {
        throw std::runtime_error("Input must be an image file or directory: " + input.string());
    }
    const Path root = comparable_path(input);
    const auto options = std::filesystem::directory_options::skip_permission_denied;
    std::vector<ImageInput> images;
    if (recursive) {
        for (std::filesystem::recursive_directory_iterator iterator(root, options, error), end;
             iterator != end; iterator.increment(error)) {
            if (error) {
                std::cerr << "Skipping inaccessible input path: " << error.message() << '\n';
                error.clear();
                continue;
            }
            if (!iterator->is_regular_file(error) || error || !is_image_file(iterator->path())) {
                error.clear();
                continue;
            }
            const Path discovered = iterator->path();
            const Path relative = discovered.lexically_relative(root);
            if (!is_safe_relative_path(relative)) {
                std::cerr << "Skipping unsafe input path: " << discovered << '\n';
                continue;
            }
            images.push_back({discovered, relative});
        }
    } else {
        for (std::filesystem::directory_iterator iterator(root, options, error), end;
             iterator != end; iterator.increment(error)) {
            if (error) {
                std::cerr << "Skipping inaccessible input path: " << error.message() << '\n';
                error.clear();
                continue;
            }
            if (!iterator->is_regular_file(error) || error || !is_image_file(iterator->path())) {
                error.clear();
                continue;
            }
            const Path discovered = iterator->path();
            const Path relative = discovered.lexically_relative(root);
            if (!is_safe_relative_path(relative)) {
                std::cerr << "Skipping unsafe input path: " << discovered << '\n';
                continue;
            }
            images.push_back({discovered, relative});
        }
    }
    std::sort(images.begin(), images.end(), [](const auto& left, const auto& right) {
        return key_for(left.relative) < key_for(right.relative);
    });
    return images;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

std::string html_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
            case '&': escaped += "&amp;"; break;
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '"': escaped += "&quot;"; break;
            case '\'': escaped += "&#39;"; break;
            default: escaped += character;
        }
    }
    return escaped;
}

std::string url_encode(const std::string& value) {
    std::ostringstream output;
    output << std::uppercase << std::hex;
    for (const unsigned char character : value) {
        if (std::isalnum(character) || character == '/' || character == '.' || character == '-' || character == '_') {
            output << static_cast<char>(character);
        } else {
            output << '%' << std::setw(2) << std::setfill('0') << static_cast<int>(character)
                   << std::setfill(' ');
        }
    }
    return output.str();
}

std::string detections_text(const std::vector<Detection>& detections) {
    std::map<std::string, std::size_t> counts;
    for (const auto& detection : detections) ++counts[detection.label];
    if (counts.empty()) return "No detections";
    std::ostringstream output;
    bool first = true;
    for (const auto& [label, count] : counts) {
        if (!first) output << ", ";
        output << label << " x" << count;
        first = false;
    }
    return output.str();
}

cv::Scalar color_for(const int class_id) {
    static const std::array<cv::Scalar, 8> palette{
        cv::Scalar{255, 100, 40}, cv::Scalar{80, 220, 80}, cv::Scalar{40, 190, 245}, cv::Scalar{230, 90, 210},
        cv::Scalar{80, 210, 220}, cv::Scalar{190, 130, 255}, cv::Scalar{60, 170, 255}, cv::Scalar{170, 220, 90}};
    return palette[static_cast<std::size_t>(std::max(0, class_id)) % palette.size()];
}

cv::Mat annotate(const cv::Mat& image, const std::vector<Detection>& detections, const DatasetConfig& config) {
    cv::Mat result = image.clone();
    for (const auto& detection : detections) {
        const cv::Scalar color = color_for(detection.class_id);
        cv::rectangle(result, detection.box, color, 2, cv::LINE_AA);
        std::ostringstream label;
        label << detection.label << ' ' << std::fixed << std::setprecision(0) << detection.score * 100.0F << "%"
              << " [" << name_for(classify_label(detection.label, config.semantics)) << ']';
        const std::string text = label.str();
        int baseline{};
        const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        const int label_top = std::max(0, detection.box.y - text_size.height - baseline - 6);
        const cv::Rect background{detection.box.x, label_top,
                                  std::min(text_size.width + 8, result.cols - detection.box.x),
                                  text_size.height + baseline + 6};
        if (background.width > 0 && background.height > 0) {
            cv::rectangle(result, background, color, cv::FILLED);
            cv::putText(result, text, {detection.box.x + 4, label_top + text_size.height + 2},
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
        }
    }
    return result;
}

void write_detection(std::ostream& output, const Detection& detection, const cv::Size size,
                     const DatasetConfig& config) {
    const auto& box = detection.box;
    const double image_width = std::max(1, size.width);
    const double image_height = std::max(1, size.height);
    output << "{\"class_id\":" << detection.class_id
           << ",\"label\":\"" << json_escape(detection.label) << "\""
           << ",\"semantic_role\":\"" << name_for(classify_label(detection.label, config.semantics)) << "\""
           << ",\"score\":" << std::fixed << std::setprecision(6) << detection.score
           << ",\"bbox_xyxy\":[" << box.x << ',' << box.y << ',' << box.x + box.width << ',' << box.y + box.height << ']'
           << ",\"bbox_xywh\":[" << box.x << ',' << box.y << ',' << box.width << ',' << box.height << ']'
           << ",\"bbox_normalized_xywh\":[" << box.x / image_width << ',' << box.y / image_height << ','
           << box.width / image_width << ',' << box.height / image_height << "]}";
}

void write_prediction(std::ostream& output, const BatchRecord& record, const DatasetConfig& config) {
    output << "{\"source\":\"" << json_escape(record.image_key) << "\""
           << ",\"width\":" << record.width
           << ",\"height\":" << record.height
           << ",\"inference_ms\":" << std::fixed << std::setprecision(3) << record.inference_ms
           << ",\"annotated\":";
    if (record.annotated_relative.empty()) output << "null";
    else output << '"' << json_escape(record.annotated_relative) << '"';
    output << ",\"detections\":[";
    for (std::size_t index = 0; index < record.detections.size(); ++index) {
        if (index != 0) output << ',';
        write_detection(output, record.detections[index], {record.width, record.height}, config);
    }
    output << "]}\n";
}

void write_error(std::ostream& output, const std::string& source, const std::string& stage, const std::string& detail) {
    output << "{\"source\":\"" << json_escape(source) << "\",\"stage\":\"" << json_escape(stage)
           << "\",\"error\":\"" << json_escape(detail) << "\"}\n";
}

double percentile(std::vector<double> values, const double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = (values.size() - 1) * fraction;
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    return values[lower] + (values[upper] - values[lower]) * (position - lower);
}

std::unordered_map<std::string, std::string> model_labels(
    const ModelConfig& config, const std::unordered_map<std::string, std::string>& aliases) {
    std::unordered_map<std::string, std::string> labels;
    for (const auto& label : config.labels) labels.emplace(normalized_label(label), label);
    for (const auto& [source, target] : aliases) {
        const auto target_it = labels.find(normalized_label(target));
        if (target_it == labels.end()) {
            throw std::runtime_error("Alias target is not a configured model label: " + target);
        }
        labels[normalized_label(source)] = target_it->second;
    }
    return labels;
}

GroundTruthIndex load_coco_annotations(const Path& path,
                                       const std::unordered_map<std::string, std::string>& labels) {
    cv::FileStorage storage(path.string(), cv::FileStorage::READ);
    if (!storage.isOpened()) throw std::runtime_error("Cannot open COCO annotation file: " + path.string());
    const cv::FileNode categories = storage["categories"];
    const cv::FileNode images = storage["images"];
    const cv::FileNode annotations = storage["annotations"];
    if (categories.empty() || images.empty() || annotations.empty()) {
        throw std::runtime_error("COCO annotations require categories, images, and annotations arrays.");
    }

    std::unordered_map<int, std::string> category_names;
    for (const auto& category : categories) {
        int id{};
        std::string name;
        category["id"] >> id;
        category["name"] >> name;
        category_names.emplace(id, std::move(name));
    }
    GroundTruthIndex index;
    std::unordered_map<int, std::string> image_keys;
    for (const auto& image : images) {
        int id{};
        std::string file_name;
        image["id"] >> id;
        image["file_name"] >> file_name;
        if (file_name.empty()) continue;
        const std::string key = key_for(Path{file_name});
        image_keys.emplace(id, key);
        index.image_keys.insert(key);
    }
    for (const auto& annotation : annotations) {
        int image_id{};
        int category_id{};
        int iscrowd{};
        annotation["image_id"] >> image_id;
        annotation["category_id"] >> category_id;
        const cv::FileNode crowd = annotation["iscrowd"];
        if (!crowd.empty()) crowd >> iscrowd;
        const auto image_it = image_keys.find(image_id);
        const auto category_it = category_names.find(category_id);
        if (image_it == image_keys.end() || category_it == category_names.end()) continue;
        const std::string raw_label = category_it->second;
        const auto model_label = labels.find(normalized_label(raw_label));
        if (model_label == labels.end()) {
            ++index.unsupported_labels[raw_label];
            continue;
        }
        if (iscrowd != 0) {
            ++index.ignored_crowd_annotations;
            continue;
        }
        const cv::FileNode bbox = annotation["bbox"];
        if (bbox.empty() || bbox.size() != 4) continue;
        std::array<float, 4> values{};
        std::size_t value_index{};
        for (const auto& value : bbox) value >> values[value_index++];
        if (values[2] <= 0.0F || values[3] <= 0.0F) continue;
        index.boxes_by_image[image_it->second].push_back(
            {model_label->second,
             {cvFloor(values[0]), cvFloor(values[1]), std::max(1, cvCeil(values[2])),
              std::max(1, cvCeil(values[3]))}});
    }
    return index;
}

void write_manifest(const Path& path, const DatasetConfig& config, const BatchOptions& options,
                    const std::size_t discovered_images) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot write run manifest: " + path.string());
    output << "{\n"
           << "  \"schema_version\": \"seatvision-dataset-run/v1\",\n"
           << "  \"detector_engine\": \"" << json_escape(config.model.engine.string()) << "\",\n"
           << "  \"confidence_threshold\": " << std::fixed << std::setprecision(3)
           << config.model.confidence_threshold << ",\n"
           << "  \"input\": \"" << json_escape(comparable_path(options.input).string()) << "\",\n"
           << "  \"images_discovered\": " << discovered_images << ",\n"
           << "  \"recursive\": " << (options.recursive ? "true" : "false") << ",\n"
           << "  \"annotated_output\": " << (options.save_annotated ? "true" : "false") << "\n"
           << "}\n";
}

void write_metrics(const Path& path, const EvaluationSummary& evaluation, const GroundTruthIndex& ground_truth,
                   const std::size_t processed_without_ground_truth, const std::size_t matched_annotation_images) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot write metrics file: " + path.string());
    output << "{\n"
           << "  \"schema_version\": \"seatvision-detection-metrics/v1\",\n"
           << "  \"metric_note\": \"Box detection only; static photos cannot establish seat occupancy.\",\n"
           << "  \"iou_threshold_for_precision_recall_f1\": 0.50,\n"
           << "  \"evaluated_images\": " << evaluation.evaluated_images << ",\n"
           << "  \"processed_images_without_ground_truth\": " << processed_without_ground_truth << ",\n"
           << "  \"matched_annotation_images\": " << matched_annotation_images << ",\n"
           << "  \"annotation_images_not_seen_in_input\": "
           << (ground_truth.image_keys.size() >= matched_annotation_images
                   ? ground_truth.image_keys.size() - matched_annotation_images
                   : 0)
           << ",\n"
           << "  \"ignored_crowd_annotations\": " << ground_truth.ignored_crowd_annotations << ",\n"
           << "  \"unsupported_ground_truth_labels\": {";
    bool first = true;
    for (const auto& [label, count] : ground_truth.unsupported_labels) {
        if (!first) output << ',';
        output << "\n    \"" << json_escape(label) << "\": " << count;
        first = false;
    }
    if (!ground_truth.unsupported_labels.empty()) output << '\n';
    output << "  },\n  \"per_label\": {";
    std::map<std::string, LabelMetrics> sorted{evaluation.labels.begin(), evaluation.labels.end()};
    first = true;
    for (const auto& [label, metrics] : sorted) {
        if (!first) output << ',';
        output << "\n    \"" << json_escape(label) << "\": {"
               << "\"ground_truth\":" << metrics.ground_truth
               << ",\"predictions\":" << metrics.predictions
               << ",\"true_positives\":" << metrics.true_positives
               << ",\"false_positives\":" << metrics.false_positives
               << ",\"false_negatives\":" << metrics.false_negatives
               << ",\"precision\":" << std::fixed << std::setprecision(6) << metrics.precision
               << ",\"recall\":" << metrics.recall
               << ",\"f1\":" << metrics.f1
               << ",\"ap50\":";
        if (metrics.ap50.has_value()) output << *metrics.ap50; else output << "null";
        output << ",\"ap50_95\":";
        if (metrics.ap50_95.has_value()) output << *metrics.ap50_95; else output << "null";
        output << '}';
        first = false;
    }
    if (!sorted.empty()) output << '\n';
    output << "  }\n}\n";
}

void write_summary(const Path& path, const std::string& detector_name, const std::vector<BatchRecord>& records,
                   const std::map<std::string, std::size_t>& label_counts, const std::size_t failures,
                   const std::size_t images_discovered, const bool wrote_metrics) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot write batch summary: " + path.string());
    std::vector<double> timings;
    std::size_t detection_count{};
    for (const auto& record : records) {
        timings.push_back(record.inference_ms);
        detection_count += record.detections.size();
    }
    const double total_ms = std::accumulate(timings.begin(), timings.end(), 0.0);
    output << "{\n"
           << "  \"schema_version\": \"seatvision-dataset-summary/v1\",\n"
           << "  \"detector\": \"" << json_escape(detector_name) << "\",\n"
           << "  \"images_discovered\": " << images_discovered << ",\n"
           << "  \"images_processed\": " << records.size() << ",\n"
           << "  \"failures\": " << failures << ",\n"
           << "  \"total_detections\": " << detection_count << ",\n"
           << "  \"inference_latency_ms\": {\"mean\": " << std::fixed << std::setprecision(3)
           << (timings.empty() ? 0.0 : total_ms / timings.size()) << ",\"p50\":" << percentile(timings, 0.50)
           << ",\"p95\":" << percentile(timings, 0.95) << "},\n"
           << "  \"ground_truth_metrics_written\": " << (wrote_metrics ? "true" : "false") << ",\n"
           << "  \"detected_labels\": {";
    bool first = true;
    for (const auto& [label, count] : label_counts) {
        if (!first) output << ',';
        output << "\n    \"" << json_escape(label) << "\": " << count;
        first = false;
    }
    if (!label_counts.empty()) output << '\n';
    output << "  }\n}\n";
}

void write_report(const Path& path, const std::vector<BatchRecord>& records, const std::size_t failures,
                  const std::size_t gallery_limit, const bool has_ground_truth) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot write HTML report: " + path.string());
    output << "<!doctype html><html><head><meta charset=\"utf-8\"><title>SeatVision dataset report</title>"
           << "<style>body{font-family:system-ui,sans-serif;margin:2rem;background:#101418;color:#e7edf3}"
           << ".notice{padding:1rem;background:#253244;border-left:4px solid #5bb8ff}.grid{display:grid;"
           << "grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}.card{background:#192129;padding:.75rem;"
           << "border-radius:.4rem}.card img{width:100%;height:auto;background:#000}code{color:#a9d7ff}</style></head><body>"
           << "<h1>SeatVision still-image validation</h1><div class=\"notice\"><strong>Generic COCO baseline.</strong> "
           << "This validates detector plumbing and supports visual review. It does not certify seat surfaces or "
           << "declare a still image occupied, available, or clear.</div>"
           << "<p>Processed <strong>" << records.size() << "</strong> images; issues: <strong>" << failures
           << "</strong>. "
           << (has_ground_truth ? "See <code>metrics.json</code> for labeled detector metrics."
                                : "Supply COCO annotations on a later run to measure accuracy.")
           << "</p><div class=\"grid\">";
    const std::size_t count = std::min(gallery_limit, records.size());
    for (std::size_t index = 0; index < count; ++index) {
        const auto& record = records[index];
        output << "<article class=\"card\"><strong>" << html_escape(record.image_key) << "</strong><br>"
               << "<small>" << std::fixed << std::setprecision(1) << record.inference_ms << " ms; "
               << html_escape(detections_text(record.detections)) << "</small>";
        if (!record.annotated_relative.empty()) {
            output << "<a href=\"" << url_encode(record.annotated_relative) << "\"><img loading=\"lazy\" src=\""
                   << url_encode(record.annotated_relative) << "\" alt=\"" << html_escape(record.image_key) << "\"></a>";
        }
        output << "</article>";
    }
    output << "</div></body></html>\n";
}

}  // namespace

int run_batch(const DatasetConfig& config, const BatchOptions& options) {
    if (options.input.empty()) throw std::runtime_error("An input image or directory is required.");
    if (options.output.empty()) throw std::runtime_error("An output run directory is required.");
    const Path input = comparable_path(options.input);
    const Path output = comparable_path(options.output);
    std::error_code error;
    if (std::filesystem::is_directory(input, error) && is_within(output, input)) {
        throw std::runtime_error("Output must be outside the input directory to avoid re-ingesting generated annotations.");
    }
    error.clear();
    if (std::filesystem::exists(output, error)) {
        if (error || !std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)) {
            throw std::runtime_error("Output directory must be new or empty: " + output.string());
        }
    }

    auto images = discover_images(input, options.recursive);
    if (options.limit > 0 && images.size() > options.limit) images.resize(options.limit);
    if (images.empty()) throw std::runtime_error("No supported images found under: " + input.string());

    std::filesystem::create_directories(output);
    std::ofstream predictions(output / "predictions.jsonl");
    std::ofstream errors(output / "errors.jsonl");
    if (!predictions || !errors) throw std::runtime_error("Cannot create run outputs in: " + output.string());
    write_manifest(output / "manifest.json", config, options, images.size());

    const auto labels = model_labels(config.model, options.label_aliases);
    std::optional<GroundTruthIndex> ground_truth;
    if (options.coco_annotations.has_value()) ground_truth = load_coco_annotations(*options.coco_annotations, labels);
    DetectionEvaluator evaluator;
    std::size_t matched_annotation_images{};
    std::size_t processed_without_ground_truth{};
    std::size_t failure_count{};
    std::map<std::string, std::size_t> label_counts;
    std::vector<BatchRecord> records;
    records.reserve(images.size());

    auto detector = create_detector(config.model);
    std::cout << "Dataset validation started: " << images.size() << " image(s) using " << detector->name() << '\n';
    for (std::size_t index = 0; index < images.size(); ++index) {
        const auto& image_input = images[index];
        const std::string image_key = key_for(image_input.relative);
        const cv::Mat image = cv::imread(image_input.absolute.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            write_error(errors, image_key, "decode", "OpenCV could not decode this image.");
            ++failure_count;
            continue;
        }
        std::vector<Detection> detections;
        double inference_ms{};
        try {
            const auto started = std::chrono::steady_clock::now();
            detections = detector->infer(image);
            inference_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        } catch (const std::exception& exception) {
            write_error(errors, image_key, "inference", exception.what());
            ++failure_count;
            continue;
        }
        BatchRecord record{image_key, image.cols, image.rows, inference_ms, detections, {}};
        if (options.save_annotated) {
            if (!is_safe_relative_path(image_input.relative)) {
                throw std::runtime_error("Internal error: image path escaped the input root.");
            }
            Path annotated_relative = Path{"annotated"} / image_input.relative;
            annotated_relative.replace_extension(".jpg");
            const Path annotated_path = output / annotated_relative;
            try {
                std::filesystem::create_directories(annotated_path.parent_path());
                if (!cv::imwrite(annotated_path.string(), annotate(image, detections, config))) {
                    throw std::runtime_error("OpenCV imwrite returned false.");
                }
                record.annotated_relative = key_for(annotated_relative);
            } catch (const std::exception& exception) {
                write_error(errors, image_key, "annotation_write", exception.what());
                ++failure_count;
            }
        }
        for (const auto& detection : detections) ++label_counts[detection.label];
        write_prediction(predictions, record, config);
        if (ground_truth.has_value()) {
            const auto annotation = ground_truth->boxes_by_image.find(image_key);
            if (ground_truth->image_keys.contains(image_key)) {
                ++matched_annotation_images;
                evaluator.add_image(image_key, detections,
                                    annotation == ground_truth->boxes_by_image.end() ? std::vector<GroundTruthBox>{}
                                                                                      : annotation->second);
            } else {
                ++processed_without_ground_truth;
            }
        }
        records.push_back(std::move(record));
        if ((index + 1) % 25 == 0 || index + 1 == images.size()) {
            std::cout << "  processed " << index + 1 << '/' << images.size() << '\n';
        }
    }
    predictions.close();
    errors.close();

    if (ground_truth.has_value()) {
        write_metrics(output / "metrics.json", evaluator.summarize(), *ground_truth, processed_without_ground_truth,
                      matched_annotation_images);
    }
    write_summary(output / "summary.json", detector->name(), records, label_counts, failure_count, images.size(),
                  ground_truth.has_value());
    write_report(output / "report.html", records, failure_count, options.gallery_limit, ground_truth.has_value());
    std::cout << "Dataset validation complete: " << records.size() << " processed, " << failure_count
              << " issue(s). Results: " << output << '\n';
    return records.empty() ? 1 : 0;
}

}  // namespace seatvision::dataset
