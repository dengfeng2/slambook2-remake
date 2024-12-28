#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <octomap/octomap.h>    // for octomap
#include <Eigen/Geometry>

using namespace std;


int main(int argc, char *argv[]) {
    vector<cv::Mat> colorImgs, depthImgs; // 彩色图和深度图
    vector<Eigen::Isometry3d> poses; // 相机位姿

    ifstream fin("./data/pose.txt");
    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        colorImgs.push_back(cv::imread("./data/color/" + std::to_string(i + 1) + ".png"));
        depthImgs.push_back(cv::imread("./data/depth/" + std::to_string(i + 1) + ".png", -1)); // 使用-1读取原始图像

        double data[7] = {0};
        for (double &d: data) {
            fin >> d;
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    // 计算点云并拼接
    // 相机内参
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0;

    cout << "正在将图像转换为 Octomap ..." << endl;

    // octomap tree
    octomap::OcTree tree(0.01); // 参数为分辨率

    for (int i = 0; i < 5; i++) {
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];

        octomap::Pointcloud cloud; // the point cloud in octomap

        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0) continue; // 为0表示没有测量到
                Eigen::Vector3d point;
                point[2] = d / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;
                // 将世界坐标系的点放入点云
                cloud.push_back(static_cast<float>(pointWorld[0]), static_cast<float>(pointWorld[1]),
                                static_cast<float>(pointWorld[2]));
            }

        // 将点云存入八叉树地图，给定原点，这样可以计算投射线
        tree.insertPointCloud(cloud, octomap::point3d(T(0, 3), T(1, 3), T(2, 3)));
    }

    // 更新中间节点的占据信息并写入磁盘
    tree.updateInnerOccupancy();
    cout << "saving octomap ... " << endl;
    tree.writeBinary("octomap.bt");
    return 0;
}
