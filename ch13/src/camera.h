#ifndef CAMERA_H
#define CAMERA_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace myslam {
    struct Camera {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Camera(Eigen::Matrix3d _k, Eigen::Vector3d _t): k(std::move(_k)), t(std::move(_t)) {
        }

        const Eigen::Matrix3d k;
        const Eigen::Vector3d t;

        cv::Matx34d Kt() const {
            cv::Matx34d Kt(
                k(0, 0), k(0, 1), k(0, 2), t(0),
                k(1, 0), k(1, 1), k(1, 2), t(1),
                k(2, 0), k(2, 1), k(2, 2), t(2));
            return Kt;
        }
    };
} // namespace myslam

#endif //CAMERA_H
