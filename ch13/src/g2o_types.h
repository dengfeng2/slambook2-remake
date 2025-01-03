#ifndef G2O_TYPES_H
#define G2O_TYPES_H
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <utility>

namespace myslam {
    /// vertex and edges used in g2o ba
    /// 位姿顶点
    class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        void setToOriginImpl() override { _estimate = Sophus::SE3d(); }

        /// left multiplication on SE3
        void oplusImpl(const double *update) override {
            Eigen::Matrix<double, 6, 1> update_eigen;
            update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
            _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
        }

        bool read(std::istream &in) override { return true; }

        bool write(std::ostream &out) const override { return true; }
    };

    class VertexXYZ : public g2o::BaseVertex<3, Eigen::Vector3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        void setToOriginImpl() override { _estimate = Eigen::Vector3d::Zero(); }

        void oplusImpl(const double *update) override {
            _estimate[0] += update[0];
            _estimate[1] += update[1];
            _estimate[2] += update[2];
        }

        bool read(std::istream &in) override { return true; }

        bool write(std::ostream &out) const override { return true; }
    };

    /// 仅估计位姿的一元边
    class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeProjectionPoseOnly(Eigen::Vector3d pos, Eigen::Matrix3d K)
            : _pos3d(std::move(pos)), _K(std::move(K)) {
        }

        void computeError() override {
            const VertexPose *v = dynamic_cast<VertexPose *>(_vertices[0]);
            Sophus::SE3d T = v->estimate();
            Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
            pos_pixel /= pos_pixel[2];
            _error = _measurement - pos_pixel.head<2>();
        }

        void linearizeOplus() override {
            const VertexPose *v = dynamic_cast<VertexPose *>(_vertices[0]);
            Sophus::SE3d T = v->estimate();
            Eigen::Vector3d pos_cam = T * _pos3d;
            double fx = _K(0, 0);
            double fy = _K(1, 1);
            double X = pos_cam[0];
            double Y = pos_cam[1];
            double Z = pos_cam[2];
            double Zinv = 1.0 / (Z + 1e-18);
            double Zinv2 = Zinv * Zinv;
            _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
                    -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
                    fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
                    -fy * X * Zinv;
        }

        bool read(std::istream &in) override { return true; }

        bool write(std::ostream &out) const override { return true; }

    private:
        Eigen::Vector3d _pos3d;
        Eigen::Matrix3d _K;
    };

    class EdgeProjection : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPose, VertexXYZ> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeProjection(const Eigen::Matrix3d &K, const Eigen::Vector3d &t) : _K(K), _t(t) {
        }

        void computeError() override {
            const VertexPose *v0 = dynamic_cast<VertexPose *>(_vertices[0]);
            const VertexXYZ *v1 = dynamic_cast<VertexXYZ *>(_vertices[1]);
            Sophus::SE3d T = v0->estimate();
            Eigen::Vector3d pos_pixel = _K * ((T * v1->estimate()) + _t);
            pos_pixel /= pos_pixel[2];
            _error = _measurement - pos_pixel.head<2>();
        }

        void linearizeOplus() override {
            const VertexPose *v0 = dynamic_cast<VertexPose *>(_vertices[0]);
            const VertexXYZ *v1 = dynamic_cast<VertexXYZ *>(_vertices[1]);
            Sophus::SE3d T = v0->estimate();
            Eigen::Vector3d pw = v1->estimate();
            Eigen::Vector3d pos_cam = (T * pw) + _t;
            double fx = _K(0, 0);
            double fy = _K(1, 1);
            double X = pos_cam[0];
            double Y = pos_cam[1];
            double Z = pos_cam[2];
            double Zinv = 1.0 / (Z + 1e-18);
            double Zinv2 = Zinv * Zinv;
            _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
                    -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
                    fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
                    -fy * X * Zinv;

            _jacobianOplusXj = _jacobianOplusXi.block<2, 3>(0, 0) * T.rotationMatrix();
        }

        bool read(std::istream &in) override { return true; }

        bool write(std::ostream &out) const override { return true; }

    private:
        Eigen::Matrix3d _K;
        Eigen::Vector3d _t;
    };
} // namespace myslam

#endif //G2O_TYPES_H
