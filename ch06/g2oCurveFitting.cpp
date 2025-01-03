#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <cmath>
#include <chrono>
#include <random>

using namespace std;

// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 重置
    void setToOriginImpl() final {
        _estimate << 0, 0, 0;
    }

    // 更新
    void oplusImpl(const double *update) final {
        _estimate += Eigen::Vector3d(update);
    }

    // 存盘和读盘：留空
    bool read(istream &in) final {
        return false;
    }

    bool write(ostream &out) const final {
        return false;
    }
};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit CurveFittingEdge(double x) : _x(x) {
    }

    // 计算曲线模型误差
    void computeError() final {
        const auto *v = dynamic_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }

    // 计算雅可比矩阵
    void linearizeOplus() final {
        const auto *v = dynamic_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }

    bool read(istream &in) final {
        return false;
    }

    bool write(ostream &out) const final {
        return false;
    }

    double _x; // x 值， y 值为 _measurement
};

int main(int argc, char *argv[]) {
    double ar = 1.0, br = 2.0, cr = 1.0; // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0; // 估计参数值
    int N = 100; // 数据点
    double w_sigma = 1.0; // 噪声Sigma值
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> normal_dist(0.0, w_sigma);

    vector<double> x_data, y_data; // 数据
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + normal_dist(gen));
    }

    // 构建图优化，先设定g2o
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1> >; // 每个误差项优化变量维度为3，误差值维度为1
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>; // 线性求解器类型

    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
            make_unique<BlockSolverType>(make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(true); // 打开调试输出

    // 往图中增加顶点
    auto *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));
    v->setId(0);
    optimizer.addVertex(v);

    // 往图中增加边
    for (int i = 0; i < N; i++) {
        auto *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v); // 设置连接的顶点
        edge->setMeasurement(y_data[i]); // 观测数值
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // 信息矩阵：协方差矩阵之逆
        optimizer.addEdge(edge);
    }

    // 执行优化
    cout << "start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double> >(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // 输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " << abc_estimate.transpose() << endl;

    return 0;
}
