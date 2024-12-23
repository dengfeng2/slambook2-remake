#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>
#include <sophus/se3.hpp>
#include <utility>

using namespace std;

void find_feature_matches(
    const cv::Mat &img_1, const cv::Mat &img_2,
    std::vector<cv::KeyPoint> &keypoints_1,
    std::vector<cv::KeyPoint> &keypoints_2,
    std::vector<cv::DMatch> &matches);

// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

void pose_estimation_3d3d(
    const vector<cv::Point3d> &pts1,
    const vector<cv::Point3d> &pts2,
    cv::Mat &R, cv::Mat &t
);

void bundleAdjustment(
    const vector<cv::Point3d> &pts1,
    const vector<cv::Point3d> &pts2,
    cv::Mat &R, cv::Mat &t
);

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    void setToOriginImpl() final {
        _estimate = Sophus::SE3d();
    }

    /// left multiplication on SE3
    void oplusImpl(const double *update) final {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    bool read(istream &in) final {
        return false;
    }

    bool write(ostream &out) const final {
        return false;
    }
};

/// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    explicit EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d point) : _point(std::move(point)) {
    }

    void computeError() final {
        const auto *pose = dynamic_cast<const VertexPose *>(_vertices[0]);
        _error = _measurement - pose->estimate() * _point;
    }

    void linearizeOplus() final {
        auto *pose = dynamic_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = pose->estimate();
        Eigen::Vector3d xyz_trans = T * _point;
        _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
    }

    bool read(istream &in) final {
        return false;
    }

    bool write(ostream &out) const final {
        return false;
    }

protected:
    Eigen::Vector3d _point;
};

int main(int argc, char *argv[]) {
    string img1_filename = "1.png";
    string img2_filename = "2.png";
    string depth1_filename = "1_depth.png";
    string depth2_filename = "2_depth.png";
    //-- 读取图像
    cv::Mat img_1 = cv::imread(img1_filename, cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(img2_filename, cv::IMREAD_COLOR);

    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 建立3D点
    cv::Mat depth1 = cv::imread(depth1_filename, cv::IMREAD_UNCHANGED); // 深度图为16位无符号数，单通道图像
    cv::Mat depth2 = cv::imread(depth1_filename, cv::IMREAD_UNCHANGED); // 深度图为16位无符号数，单通道图像
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<cv::Point3d> pts1, pts2;

    for (cv::DMatch m: matches) {
        ushort d1 = depth1.ptr<unsigned short>(static_cast<int>(keypoints_1[m.queryIdx].pt.y))[static_cast<int>(
            keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(static_cast<int>(keypoints_2[m.trainIdx].pt.y))[static_cast<int>(
            keypoints_2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0) // bad depth
            continue;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        double dd1 = d1 / 5000.0;
        double dd2 = d2 / 5000.0;
        pts1.emplace_back(p1.x * dd1, p1.y * dd1, dd1);
        pts2.emplace_back(p2.x * dd2, p2.y * dd2, dd2);
    }

    cout << "3d-3d pairs: " << pts1.size() << endl;
    cv::Mat R, t;
    pose_estimation_3d3d(pts1, pts2, R, t);
    cout << "ICP via SVD results: " << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "R_inv = " << R.t() << endl;
    cout << "t_inv = " << -R.t() * t << endl;

    cout << "calling bundle adjustment" << endl;

    bundleAdjustment(pts1, pts2, R, t);

    // verify p1 = R * p2 + t
    for (int i = 0; i < 5; i++) {
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "(R*p2+t) = " <<
                R * (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
                << endl;
        cout << endl;
    }
}

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches) {
    //-- 初始化
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<cv::DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
    return {
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    };
}

void pose_estimation_3d3d(const vector<cv::Point3d> &pts1,
                          const vector<cv::Point3d> &pts2,
                          cv::Mat &R, cv::Mat &t) {
    cv::Point3d p1, p2; // center of mass
    auto N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }

    p1 /= static_cast<double>(N);
    p2 /= static_cast<double>(N);
    vector<cv::Point3d> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << "W=" << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    cout << "U=" << U << endl;
    cout << "V=" << V << endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    if (R_.determinant() < 0) {
        R_ = -R_;
    }
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (cv::Mat_<double>(3, 3) <<
         R_(0, 0), R_(0, 1), R_(0, 2),
         R_(1, 0), R_(1, 1), R_(1, 2),
         R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

void bundleAdjustment(
    const vector<cv::Point3d> &pts1,
    const vector<cv::Point3d> &pts2,
    cv::Mat &R, cv::Mat &t) {
    // 构建图优化，先设定g2o
    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>; // 线性求解器类型
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        make_unique<BlockSolverType>(make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(true); // 打开调试输出

    // vertex
    VertexPose *pose = new VertexPose(); // camera pose
    pose->setId(0);
    pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(pose);

    // edges
    for (size_t i = 0; i < pts1.size(); i++) {
        EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
            Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setVertex(0, pose);
        edge->setMeasurement(Eigen::Vector3d(
            pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double> >(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

    cout << endl << "after optimization:" << endl;
    cout << "T=\n" << pose->estimate().matrix() << endl;

    // convert to cv::Mat
    Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
    Eigen::Vector3d t_ = pose->estimate().translation();
    R = (cv::Mat_<double>(3, 3) <<
         R_(0, 0), R_(0, 1), R_(0, 2),
         R_(1, 0), R_(1, 1), R_(1, 2),
         R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}