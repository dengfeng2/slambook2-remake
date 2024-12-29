#ifndef SLAMBOOK2_REMAKE_CAMERA_H
#define SLAMBOOK2_REMAKE_CAMERA_H

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace myslam {
    class Camera {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Camera(double fx, double fy,
               double cx, double cy,
               double baseline, const Sophus::SE3d &pose) : fx_(fx),
                                                            fy_(fy),
                                                            cx_(cx),
                                                            cy_(cy),
                                                            baseline_(
                                                                    baseline),
                                                            pose_(pose) {
            pose_inv_ = pose_.inverse();
        }

        [[nodiscard]] const Sophus::SE3d &pose() const { return pose_; }

        [[nodiscard]] Eigen::Matrix<double, 3, 3> K() const {
            Eigen::Matrix<double, 3, 3> k;
            k << fx_, 0, cx_,
                    0, fy_, cy_,
                    0, 0, 1;
            return k;
        }

        // coordinate transform: world, camera, pixel
        Eigen::Vector3d world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w);

        Eigen::Vector3d camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_c_w);

        [[nodiscard]] Eigen::Vector2d camera2pixel(const Eigen::Vector3d &p_c) const;

        [[nodiscard]] Eigen::Vector3d pixel2camera(const Eigen::Vector2d &p_p, double depth = 1) const;

        Eigen::Vector3d pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &T_c_w, double depth = 1);

        Eigen::Vector2d world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w);


    private:
        double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0,
                baseline_ = 0;  // Camera intrinsics
        Sophus::SE3d pose_;             // extrinsic, from stereo camera to single camera
        Sophus::SE3d pose_inv_;         // inverse of extrinsics
    };

} // namespace myslam

#endif //SLAMBOOK2_REMAKE_CAMERA_H
