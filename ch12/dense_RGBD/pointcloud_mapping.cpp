#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

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

    cout << "正在将图像转换为点云..." << endl;

    // 新建一个点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (int i = 0; i < 5; i++) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr current(new pcl::PointCloud<pcl::PointXYZRGB>);
        cout << "转换图像中: " << i + 1 << endl;
        const cv::Mat &color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        const Eigen::Isometry3d &T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0) continue; // 为0表示没有测量到
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;

                pcl::PointXYZRGB p;
                p.x = static_cast<float>(pointWorld[0]);
                p.y = static_cast<float>(pointWorld[1]);
                p.z = static_cast<float>(pointWorld[2]);
                p.b = color.data[v * color.step + u * color.channels()];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + u * color.channels() + 2];
                current->points.push_back(p);
            }
        // depth filter and statistical removal
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> statistical_filter;
        statistical_filter.setMeanK(50);
        statistical_filter.setStddevMulThresh(1.0);
        statistical_filter.setInputCloud(current);
        statistical_filter.filter(*tmp);
        (*pointCloud) += *tmp;
    }

    pointCloud->is_dense = false;
    cout << "点云共有" << pointCloud->size() << "个点." << endl;

    // voxel filter
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    float resolution = 0.03f;
    voxel_filter.setLeafSize(resolution, resolution, resolution); // resolution
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
    voxel_filter.setInputCloud(pointCloud);
    voxel_filter.filter(*tmp);
    tmp->swap(*pointCloud);

    cout << "滤波之后，点云共有" << pointCloud->size() << "个点." << endl;

    pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
    return 0;
}
