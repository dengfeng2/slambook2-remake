#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>
#include <utility>
#include <sophus/se3.hpp>
#include <mutex>
#include <utility>
#include <exception>
#include "feature.h"


namespace myslam {
    class Frame {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        static std::shared_ptr<Frame> Create(cv::Mat left_img, cv::Mat right_img) {
            static unsigned long factory_id = 0;
            return std::shared_ptr<Frame>(new Frame(factory_id++, std::move(left_img), std::move(right_img)));
        }

        unsigned long id() const { return id_; }

        cv::Mat left_img() const { return left_img_; }

        cv::Mat right_img() const { return right_img_; }

        std::vector<std::shared_ptr<Feature> > left_features() const { return left_features_; }

        std::vector<std::shared_ptr<Feature> > right_features() const { return right_features_; }

        void AddLeftFeature(std::shared_ptr<Feature> feature) {
            left_features_.push_back(std::move(feature));
        }

        void AddRightFeature(std::shared_ptr<Feature> feature) {
            right_features_.push_back(std::move(feature));
        }

        void AddMapPoint(size_t index, const std::shared_ptr<MapPoint> &map_point) const {
            left_features_[index]->set_map_point(map_point);
            right_features_[index]->set_map_point(map_point);
        }

        Sophus::SE3d Pose() {
            std::unique_lock<std::mutex> lock(pose_mutex_);
            return pose_;
        }

        void SetPose(const Sophus::SE3d &pose) {
            std::unique_lock<std::mutex> lock(pose_mutex_);
            pose_ = pose;
        }

        void SetKeyFrame() {
            is_keyframe_ = true;
        }

        bool is_keyframe() const { return is_keyframe_; }


    private:
        Frame(unsigned long id, cv::Mat left_img, cv::Mat right_img) : id_(id), left_img_(std::move(left_img)),
                                                                       right_img_(std::move(right_img)) {
        }

        const unsigned long id_;
        const cv::Mat left_img_, right_img_; // stereo images

        std::mutex pose_mutex_;
        std::vector<std::shared_ptr<Feature> > left_features_;
        std::vector<std::shared_ptr<Feature> > right_features_;
        bool is_keyframe_ = false;
        Sophus::SE3d pose_;
    };
}

#endif //FRAME_H
