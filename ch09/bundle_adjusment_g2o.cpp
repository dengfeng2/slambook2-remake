#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>

#include <sophus/se3.hpp>
#include "common.h"

using namespace std;
/// 姿态和内参的结构
string bal_file = "problem-16-22106-pre.txt";

struct PoseAndIntrinsics {
    PoseAndIntrinsics() = default;

    // /// set from given data address
    explicit PoseAndIntrinsics(const double *data_addr) {
        rotation = Sophus::SO3d::exp(Eigen::Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Eigen::Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    Sophus::SO3d rotation;
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double focal = 0;
    double k1 = 0, k2 = 0;
};

/// 位姿加相机内参的顶点，9维，前三维为so3，接下去为t, f, k1, k2
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoseAndIntrinsics() = default;

    void setToOriginImpl() final {
        _estimate = PoseAndIntrinsics();
    }

    void oplusImpl(const double *update) final {
        _estimate.rotation = Sophus::SO3d::exp(Eigen::Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Eigen::Vector3d(update[3], update[4], update[5]);
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    /// 根据估计值投影一个点
    Eigen::Vector2d project(const Eigen::Vector3d &point) const {
        Eigen::Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc / pc[2];
        double r2 = pc.squaredNorm();
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return {
            _estimate.focal * distortion * pc[0],
            _estimate.focal * distortion * pc[1]
        };
    }

    bool read(istream &in) final { return false; }

    bool write(ostream &out) const final { return false; }
};

class VertexPoint : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() = default;

    void setToOriginImpl() final {
        _estimate = Eigen::Vector3d(0, 0, 0);
    }

    void oplusImpl(const double *update) final {
        _estimate += Eigen::Vector3d(update[0], update[1], update[2]);
    }

    bool read(istream &in) final { return false; }

    bool write(ostream &out) const final { return false; }
};

class EdgeProjection : // 误差向量的维度，观察数据的类型，与边相连的两个顶点的类型
        public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    void computeError() final {
        auto v0 = dynamic_cast<VertexPoseAndIntrinsics *>(_vertices[0]);
        auto v1 = dynamic_cast<VertexPoint *>(_vertices[1]);
        auto proj = v0->project(v1->estimate());
        _error = proj - _measurement;
    }

    // use numeric derivatives
    bool read(istream &in) final { return false; }

    bool write(ostream &out) const final { return false; }
};

void SolveBA(BALProblem &bal_problem);

int main(int argc, char *argv[]) {
    BALProblem bal_problem(bal_file);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<9, 3> >;
    using LinearSolverType = g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        make_unique<BlockSolverType>(make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;

    auto &camera_param = bal_problem.mutable_camera_param();
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        auto *v = new VertexPoseAndIntrinsics();
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera_param.data() + 9 * i));
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }

    auto &world_point = bal_problem.mutable_world_points();
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        auto *v = new VertexPoint();
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Eigen::Vector3d(world_point[3 * i], world_point[3 * i + 1], world_point[3 * i + 2]));
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    const auto &observations = bal_problem.observations();
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        auto *edge = new EdgeProjection();
        auto camera_idx = bal_problem.camera_index()[i];
        auto point_idx = bal_problem.point_index()[i];
        edge->setVertex(0, vertex_pose_intrinsics[camera_idx]);
        edge->setVertex(1, vertex_points[point_idx]);
        edge->setMeasurement(Eigen::Vector2d(observations[2 * i], observations[2 * i + 1]));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }
    optimizer.initializeOptimization();
    optimizer.optimize(40);

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        auto r = estimate.rotation.log();
        camera_param[9 * i + 0] = r[0];
        camera_param[9 * i + 1] = r[1];
        camera_param[9 * i + 2] = r[2];
        camera_param[9 * i + 3] = estimate.translation[0];
        camera_param[9 * i + 4] = estimate.translation[1];
        camera_param[9 * i + 5] = estimate.translation[2];
        camera_param[9 * i + 6] = estimate.focal;
        camera_param[9 * i + 7] = estimate.k1;
        camera_param[9 * i + 8] = estimate.k2;
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        auto vertex = vertex_points[i];
        auto point = vertex->estimate();

        world_point[3 * i] = point(0);
        world_point[3 * i + 1] = point(1);
        world_point[3 * i + 2] = point(2);
    }
}
