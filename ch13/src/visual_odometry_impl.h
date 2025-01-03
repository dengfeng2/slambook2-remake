#ifndef VISUAL_ODOMETRY_IMPL_H
#define VISUAL_ODOMETRY_IMPL_H

#include <string>
#include <memory>

#include "visual_odometry.h"

namespace myslam {
    class Dataset;

    class Frontend;

    class Viewer;

    class Backend;

    class VisualOdometryImpl : public VisualOdometry {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        explicit VisualOdometryImpl(std::string dataset_path) : dataset_path_(std::move(dataset_path)) {
        };

        bool Init() override;

        void Run() override;

        bool Step() override;

        FrontendStatus GetFrontendStatus() const override;

    private:
        const std::string dataset_path_;
        std::shared_ptr<Dataset> dataset_;
        std::shared_ptr<Frontend> frontend_;
        std::shared_ptr<Viewer> viewer_;
        std::shared_ptr<Backend> backend_;
    };
} // namespace myslam

#endif //VISUAL_ODOMETRY_IMPL_H
