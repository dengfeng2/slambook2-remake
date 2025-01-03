#include "dataset.h"

#include <opencv2/opencv.hpp>
#include <fmt/format.h>
#include <glog/logging.h>
#include <fstream>

#include "camera.h"
#include "frame.h"

namespace myslam {
    bool Dataset::Init() {
        // read camera intrinsics and extrinsic(s)
        std::ifstream calib_file(dataset_path_ + "/calib.txt");
        if (!calib_file.is_open()) {
            LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt";
            return false;
        }

        std::string line;
        while (std::getline(calib_file, line)) {
            std::istringstream iss(line);
            std::string label;
            iss >> label;
            if (label == "P0:" || label == "P1:") {
                Eigen::Matrix3d k;
                Eigen::Vector3d t;
                iss >> k(0, 0) >> k(0, 1) >> k(0, 2) >> t(0)
                    >> k(1, 0) >> k(1, 1) >> k(1, 2) >> t(1)
                    >> k(2, 0) >> k(2, 1) >> k(2, 2) >> t(2);

                auto camera = std::make_shared<Camera>(k, t);
                cameras_.push_back(camera);
                LOG(INFO) << "Camera " << label << " extrinsic(s): " << t.transpose();
            }
        }
        calib_file.close();
        return true;
    }

    std::shared_ptr<Frame> Dataset::NextFrame() {
        cv::Mat image_left = cv::imread(fmt::format("{}/image_{}/{:06}.png", dataset_path_, 0, current_image_index_),
                                        cv::IMREAD_GRAYSCALE);
        cv::Mat image_right = cv::imread(fmt::format("{}/image_{}/{:06}.png", dataset_path_, 1, current_image_index_),
                                         cv::IMREAD_GRAYSCALE);
        if (image_left.empty() || image_right.empty()) {
            LOG(WARNING) << "cannot find images at index " << current_image_index_;
            return nullptr;
        }
        cv::Mat image_left_resized, image_right_resized;
        cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5,
                   cv::INTER_NEAREST);
        cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,
                   cv::INTER_NEAREST);

        auto new_frame = Frame::Create(image_left_resized, image_right_resized);
        current_image_index_++;
        return new_frame;
    }
} // namespace myslam
