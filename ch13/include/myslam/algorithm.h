#ifndef SLAMBOOK2_REMAKE_ALGORITHM_H
#define SLAMBOOK2_REMAKE_ALGORITHM_H

#include <vector>
#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace myslam {

    /**
     * linear triangulation with SVD
     * @param poses     poses,
     * @param points    points in normalized plane
     * @param pt_world  triangulated point in the world
     * @return true if success
     */
    inline bool triangulation(const std::vector<Sophus::SE3d> &poses,
                              const std::vector<Eigen::Vector3d> &points, Eigen::Vector3d &pt_world) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(2 * poses.size(), 4);
        Eigen::Vector<double, Eigen::Dynamic> b(2 * poses.size());
        b.setZero();
        for (long i = 0; i < poses.size(); ++i) {
            Eigen::Matrix<double, 3, 4> m = poses[i].matrix3x4();
            A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
            A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
        }
        auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

        if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
            // 解质量不好，放弃
            return true;
        }
        return false;
    }
}  // namespace myslam

#endif //SLAMBOOK2_REMAKE_ALGORITHM_H
