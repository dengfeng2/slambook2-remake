#ifndef FEATURE_H
#define FEATURE_H
#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>

namespace myslam {
    class MapPoint;
    class Frame;

    class Feature {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        static std::shared_ptr<Feature> Create(const cv::KeyPoint &keypoint, std::shared_ptr<Frame> frame, bool is_left);

        cv::KeyPoint keypoint() const { return keypoint_; }

        std::shared_ptr<Frame> GetFrame() const;

        bool is_left() const{return is_left_;}

        std::shared_ptr<MapPoint> map_point() const {
            return map_point_.lock();
        }

        void reset_map_point() {
            map_point_.reset();
        }

        void set_map_point(const std::shared_ptr<MapPoint> &map_point) { map_point_ = map_point; }
        bool is_outlier() const { return is_outlier_; }
        void set_outlier(bool outlier) { is_outlier_ = outlier; }

    private:
        Feature(unsigned long id, const cv::KeyPoint &keypoint, std::shared_ptr<Frame> frame, bool is_left);

        const unsigned long id_;
        const cv::KeyPoint keypoint_;
        const bool is_left_;
        bool is_outlier_ = false; // abnormal
        std::weak_ptr<Frame> frame_;
        std::weak_ptr<MapPoint> map_point_;
    };
} // myslam

#endif //FEATURE_H
