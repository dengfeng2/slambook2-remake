#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <sophus/se3.hpp>

using namespace std;
string filename = "sphere.g2o";
/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 本节使用李代数表达位姿图，节点和边的方式为自定义
 * **********************************************/

using Matrix6d = Eigen::Matrix<double, 6, 6>;

// 给定误差求J_R^{-1}的近似
Matrix6d JRInv(const Sophus::SE3d &e) {
    Matrix6d J;
    J.block(0, 0, 3, 3) = Sophus::SO3d::hat(e.so3().log());
    J.block(0, 3, 3, 3) = Sophus::SO3d::hat(e.translation());
    J.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = Sophus::SO3d::hat(e.so3().log());
    // J = J * 0.5 + Matrix6d::Identity();
    J = Matrix6d::Identity(); // try Identity if you want
    return J;
}

// 李代数顶点
using Vector6d = Eigen::Matrix<double, 6, 1>;

class VertexSE3LieAlgebra : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool read(istream &is) final {
        double data[7];
        is >> data[0] >> data[1] >> data[2] >> data[3] >> data[4] >> data[5] >> data[6];
        setEstimate(Sophus::SE3d(
            Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
            Eigen::Vector3d(data[0], data[1], data[2])
        ));
        return true;
    }

    bool write(ostream &os) const final {
        os << id() << " ";
        Eigen::Quaterniond q = _estimate.unit_quaternion();
        os << _estimate.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;
        return true;
    }

    void setToOriginImpl() final {
        _estimate = Sophus::SE3d();
    }

    // 左乘更新
    void oplusImpl(const double *update) final {
        Vector6d upd;
        upd << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(upd) * _estimate;
    }
};

// 两个李代数节点之边
class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, Sophus::SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool read(istream &is) final {
        double data[7];
        is >> data[0] >> data[1] >> data[2] >> data[3] >> data[4] >> data[5] >> data[6];
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        setMeasurement(Sophus::SE3d(q, Eigen::Vector3d(data[0], data[1], data[2])));
        for (int i = 0; i < information().rows() && is.good(); i++)
            for (int j = i; j < information().cols() && is.good(); j++) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool write(ostream &os) const final {
        auto *v1 = dynamic_cast<VertexSE3LieAlgebra *>(_vertices[0]);
        auto *v2 = dynamic_cast<VertexSE3LieAlgebra *>(_vertices[1]);
        os << v1->id() << " " << v2->id() << " ";
        Sophus::SE3d m = _measurement;
        Eigen::Quaterniond q = m.unit_quaternion();
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // information matrix
        for (int i = 0; i < information().rows(); i++)
            for (int j = i; j < information().cols(); j++) {
                os << information()(i, j) << " ";
            }
        os << endl;
        return true;
    }

    // 误差计算与书中推导一致
    void computeError() final {
        Sophus::SE3d v1 = (dynamic_cast<VertexSE3LieAlgebra *>(_vertices[0]))->estimate();
        Sophus::SE3d v2 = (dynamic_cast<VertexSE3LieAlgebra *>(_vertices[1]))->estimate();
        _error = (_measurement.inverse() * v1.inverse() * v2).log();
    }

    // 雅可比计算
    void linearizeOplus() final {
        Sophus::SE3d v1 = (dynamic_cast<VertexSE3LieAlgebra *>(_vertices[0]))->estimate();
        Sophus::SE3d v2 = (dynamic_cast<VertexSE3LieAlgebra *>(_vertices[1]))->estimate();
        Matrix6d J = JRInv(Sophus::SE3d::exp(_error));
        // 尝试把J近似为I？
        _jacobianOplusXi = -J * v2.inverse().Adj();
        _jacobianOplusXj = J * v2.inverse().Adj();
    }
};

int main(int argc, char *argv[]) {

    ifstream fin(filename);
    if (!fin) {
        cout << "file " << filename << " does not exist." << endl;
        return 1;
    }

    // 设定g2o
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6> >;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        make_unique<BlockSolverType>(make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(true); // 打开调试输出

    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量

    vector<VertexSE3LieAlgebra *> vectices;
    vector<EdgeSE3LieAlgebra *> edges;
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // 顶点
            auto *v = new VertexSE3LieAlgebra();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            vectices.push_back(v);
            if (index == 0)
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 边
            auto *e = new EdgeSE3LieAlgebra();
            int idx1, idx2; // 关联的两个顶点
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
            edges.push_back(e);
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    cout << "saving optimization results ..." << endl;

    // 因为用了自定义顶点且没有向g2o注册，这里保存自己来实现
    // 伪装成 SE3 顶点和边，让 g2o_viewer 可以认出
    ofstream fout("result_lie.g2o");
    for (VertexSE3LieAlgebra *v: vectices) {
        fout << "VERTEX_SE3:QUAT ";
        v->write(fout);
    }
    for (EdgeSE3LieAlgebra *e: edges) {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();
    return 0;
}
