#ifndef SLAMBOOK2_REMAKE_FEATURE_H
#define SLAMBOOK2_REMAKE_FEATURE_H
#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace myslam {

struct Frame;
struct MapPoint;

/**
 * 2D 特征点
 * 在三角化之后会被关联一个地图点
 */
struct Feature {

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    std::weak_ptr<Frame> frame_;         // 持有该feature的frame
    cv::KeyPoint position_;              // 2D提取位置
    std::weak_ptr<MapPoint> map_point_;  // 关联地图点

    bool is_outlier_ = false;       // 是否为异常点
    bool is_on_left_image_ = true;  // 标识是否提在左图，false为右图

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
            : frame_(frame), position_(kp) {}
};
}  // namespace myslam

#endif //SLAMBOOK2_REMAKE_FEATURE_H
