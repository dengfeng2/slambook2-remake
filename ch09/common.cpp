#include "common.h"
#include <fstream>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

using namespace std;


Eigen::Vector3d median_of_points(const std::vector<double> &points) {
    auto num = points.size() / 3;
    std::vector<double> xs(num);
    std::vector<double> ys(num);
    std::vector<double> zs(num);
    for (int i = 0; i < num; i++) {
        xs[i] = points[3 * i];
        ys[i] = points[3 * i + 1];
        zs[i] = points[3 * i + 2];
    }
    int n = static_cast<int>(num) / 2;
    std::nth_element(xs.begin(), xs.begin() + n, xs.end());
    std::nth_element(ys.begin(), ys.begin() + n, ys.end());
    std::nth_element(zs.begin(), zs.begin() + n, zs.end());
    return {xs[n], ys[n], zs[n]};
}

BALProblem::BALProblem(const std::string &filename) {
    fstream file(filename);
    file >> num_cameras_ >> num_points_ >> num_observations_;

    point_index_.resize(num_observations_);
    camera_index_.resize(num_observations_);
    observations_.resize(2 * num_observations_);

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translation
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    camera_params_.resize(9 * num_cameras_);
    world_points_.resize(3 * num_points_);

    for (int i = 0; i < num_observations_; i++) {
        file >> camera_index_[i] >> point_index_[i] >> observations_[2 * i] >> observations_[2 * i + 1];
    }
    for (int i = 0; i < 9 * num_cameras_; i++) {
        file >> camera_params_[i];
    }
    for (int i = 0; i < 3 * num_points_; i++) {
        file >> world_points_[i];
    }
    file.close();
}

void BALProblem::Normalize() {
    Eigen::Vector3d median = median_of_points(world_points_); // 1.65443, 3.7886, -29.7204

    vector<double> normalized_median;
    for (int i = 0; i < num_points_; ++i) {
        Eigen::Map<Eigen::Vector3d> point(world_points_.data() + 3 * i, 3);
        normalized_median.push_back((point - median).lpNorm<1>());
    }
    auto n = num_points_ / 2;
    std::nth_element(normalized_median.begin(), normalized_median.begin() + n, normalized_median.end());

    const double median_absolute_deviation = normalized_median[n]; // 22.6081

    // Scale so that the median absolute deviation of the resulting
    // reconstruction is 100
    const double scale = 100.0 / median_absolute_deviation;

    // X = scale * (X - median)
    for (int i = 0; i < num_points_; ++i) {
        Eigen::Map<Eigen::Vector3d> point(world_points_.data() + 3 * i, 3);
        point = scale * (point - median);
    }

    for (int i = 0; i < num_cameras_; ++i) {
        Eigen::Map<Eigen::Vector3d> rotation_vector(camera_params_.data() + 9 * i, 3);
        double angle = rotation_vector.norm();
        auto axis = rotation_vector.normalized();
        Eigen::Map<Eigen::Vector3d> old_t(camera_params_.data() + 9 * i + 3, 3);
        // c = -R't
        Eigen::Vector3d center = -1.0 * (Eigen::AngleAxisd(angle, -axis) * old_t);
        center = scale * (center - median);
        // t = -Rc
        Eigen::Vector3d new_t = -1.0 * (Eigen::AngleAxisd(angle, axis) * center);
        camera_params_[9 * i + 3] = new_t[0];
        camera_params_[9 * i + 4] = new_t[1];
        camera_params_[9 * i + 5] = new_t[2];
    }
}

void BALProblem::Perturb(double rotation_sigma,
                         double translation_sigma,
                         double point_sigma) {
    assert(point_sigma >= 0.0);
    assert(rotation_sigma >= 0.0);
    assert(translation_sigma >= 0.0);

    random_device rd;
    mt19937 gen(rd());
    if (point_sigma > 0) {
        normal_distribution<double> norm_dist(0.0, point_sigma);
        for (int i = 0; i < num_points_; ++i) {
            world_points_[i] += norm_dist(gen);
        }
    }

    for (int i = 0; i < num_cameras_; ++i) {
        Eigen::Map<const Eigen::Vector3d> rotation_vector(camera_params_.data() + 9 * i, 3);
        double angle = rotation_vector.norm();
        auto axis = rotation_vector.normalized();
        Eigen::Map<const Eigen::Vector3d> old_t(camera_params_.data() + 9 * i + 3, 3);
        // c = -R't
        Eigen::Vector3d center = -1.0 * (Eigen::AngleAxisd(angle, -axis) * old_t);
        if (rotation_sigma > 0) {
            normal_distribution<double> norm_dist(0.0, rotation_sigma);
            center += Eigen::Vector3d{norm_dist(gen), norm_dist(gen), norm_dist(gen)};
        }
        Eigen::Vector3d new_t = -1.0 * (Eigen::AngleAxisd(angle, axis) * center);
        if (translation_sigma > 0) {
            normal_distribution<double> norm_dist(0.0, translation_sigma);
            new_t += Eigen::Vector3d{norm_dist(gen), norm_dist(gen), norm_dist(gen)};
        }
        camera_params_[9 * i + 3] = new_t[0];
        camera_params_[9 * i + 4] = new_t[1];
        camera_params_[9 * i + 5] = new_t[2];
    }
}

void BALProblem::WriteToPLYFile(const std::string &filename) const {
    ofstream ofs(filename);
    ofs << "ply"
            << '\n' << "format ascii 1.0"
            << '\n' << "element vertex " << num_cameras_ + num_points_
            << '\n' << "property float x"
            << '\n' << "property float y"
            << '\n' << "property float z"
            << '\n' << "property uchar red"
            << '\n' << "property uchar green"
            << '\n' << "property uchar blue"
            << '\n' << "end_header" << std::endl;

    for (int i = 0; i < num_cameras_; ++i) {
        Eigen::Map<const Eigen::Vector3d> rotation_vector(camera_params_.data() + 9 * i, 3);
        double angle = rotation_vector.norm();
        auto axis = rotation_vector.normalized();
        Eigen::Map<const Eigen::Vector3d> old_t(camera_params_.data() + 9 * i + 3, 3);
        // c = -R't
        Eigen::Vector3d center = -1.0 * (Eigen::AngleAxisd(angle, -axis) * old_t);
        ofs << center(0) << ' ' << center(1) << ' ' << center(2) << " 0 255 0\n";
    }
    for (int i = 0; i < num_points_; i++) {
        ofs << world_points_[3 * i] << ' ' << world_points_[3 * i + 1] << ' ' << world_points_[3 * i + 2] <<
                " 255 255 255\n";
    }
    ofs.close();
}
