#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H

#include <Eigen/Core>
#include <memory>
#include <string>

namespace myslam {
    enum class FrontendStatus { INITIALIZING, TRACKING_GOOD, TRACKING_BAD, LOST };

    class VisualOdometry {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        static std::shared_ptr<VisualOdometry> Create(const std::string &dataset_path);

        virtual ~VisualOdometry() = default;

        virtual bool Init() = 0;

        virtual void Run() = 0;

        virtual bool Step() = 0;

        virtual FrontendStatus GetFrontendStatus() const = 0;
    };
} // namespace myslam
#endif //VISUAL_ODOMETRY_H
