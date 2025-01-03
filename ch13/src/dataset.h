#ifndef DATASET_H
#define DATASET_H

#include <memory>
#include <Eigen/Core>
#include <vector>

namespace myslam {
    struct Camera;
    class Frame;

    class Dataset {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        explicit Dataset(std::string dataset_path): dataset_path_(std::move(dataset_path)) {
        }

        bool Init();

        std::shared_ptr<Camera> GetCamera(int camera_id) const {
            return cameras_.at(camera_id);
        }

        std::shared_ptr<Frame> NextFrame();

    private:
        const std::string dataset_path_;
        int current_image_index_{0};
        std::vector<std::shared_ptr<Camera> > cameras_;
    };
} // namespace myslam

#endif //DATASET_H
