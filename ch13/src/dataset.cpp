#include "dataset.h"
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <fstream>
#include <utility>
#include <fmt/format.h>
#include <memory>

#include "camera.h"
#include "frame.h"

using namespace std;

namespace myslam {

    Dataset::Dataset(std::string dataset_path)
            : dataset_path_(std::move(dataset_path)) {}

    bool Dataset::Init() {
        // read camera intrinsics and extrinsics
        ifstream fin(dataset_path_ + "/calib.txt");
        if (!fin) {
            LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt!";
            return false;
        }

        for (int i = 0; i < 4; ++i) {
            char camera_name[3];
            fin >> camera_name[0] >> camera_name[1] >> camera_name[2];

            double projection_data[12];
            for (double &k: projection_data) {
                fin >> k;
            }
            Eigen::Matrix<double, 3, 3> K;
            K << projection_data[0], projection_data[1], projection_data[2],
                    projection_data[4], projection_data[5], projection_data[6],
                    projection_data[8], projection_data[9], projection_data[10];
            Eigen::Vector3d t;
            t << projection_data[3], projection_data[7], projection_data[11];
            t = K.inverse() * t;
            K = K * 0.5;
            std::shared_ptr<Camera> new_camera = std::make_shared<Camera>(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                                          t.norm(), Sophus::SE3d(Sophus::SO3d(), t));
            cameras_.push_back(new_camera);
            LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
        }
        fin.close();
        current_image_index_ = 0;
        return true;
    }

    std::shared_ptr<Frame> Dataset::NextFrame() {
        // read images
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

        auto new_frame = Frame::CreateFrame();
        new_frame->left_img_ = image_left_resized;
        new_frame->right_img_ = image_right_resized;
        current_image_index_++;
        return new_frame;
    }

}  // namespace myslam
