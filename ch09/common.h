#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <vector>

class BALProblem {
public:
    explicit BALProblem(const std::string &filename);

    int num_cameras() const { return num_cameras_; }
    int num_points() const { return num_points_; }
    int num_observations() const { return num_observations_; }

    void Normalize();

    void Perturb(double rotation_sigma,
                 double translation_sigma,
                 double point_sigma);

    void WriteToPLYFile(const std::string &filename) const;

    const std::vector<int> &camera_index() const { return camera_index_; }
    const std::vector<int> &point_index() const { return point_index_; }
    const std::vector<double> &observations() const { return observations_; }
    std::vector<double> &mutable_camera_param() { return camera_params_; }
    std::vector<double> &mutable_world_points() { return world_points_; }

private:
    int num_cameras_;
    int num_points_;
    int num_observations_;
    std::vector<int> point_index_;
    std::vector<int> camera_index_;;
    std::vector<double> observations_;

    std::vector<double> camera_params_;
    std::vector<double> world_points_;
};

#endif //COMMON_H
