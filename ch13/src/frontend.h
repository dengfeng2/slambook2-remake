#ifndef FRONTEND_H
#define FRONTEND_H

#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "backend.h"
#include "visual_odometry.h"

namespace myslam {
    struct Camera;

    class Frame;

    class Map;

    class Viewer;

    class Backend;

    class Frontend {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Frontend(std::shared_ptr<Camera> camera_left, std::shared_ptr<Camera> camera_right, std::shared_ptr<Map> map,
                 std::shared_ptr<Viewer> viewer, std::shared_ptr<Backend> backend_);

        bool AddFrame(std::shared_ptr<Frame> frame);

        FrontendStatus GetStatus() const { return status_; }

    private:
        bool StereoInit();

        bool Track();

        bool Reset();

        // 检测左图特征点
        size_t DetectFeatures() const;

        // 使用光流法检测右图特征点
        int FindFeaturesInRight() const;

        int TrackLastFrame() const;

        size_t EstimateCurrentPose() const;

        size_t InsertKeyFrame() const;

        std::shared_ptr<Frame> current_frame_;
        std::shared_ptr<Frame> last_frame_;
        Sophus::SE3d relative_motion_; // 相对上一帧的运动
        FrontendStatus status_{FrontendStatus::INITIALIZING};

        const int num_features_init_{50};
        const int num_features_{150};
        const std::shared_ptr<Camera> camera_left_;
        const std::shared_ptr<Camera> camera_right_;
        const std::shared_ptr<Map> map_;
        const std::shared_ptr<Viewer> viewer_;
        const std::shared_ptr<Backend> backend_;
        cv::Ptr<cv::GFTTDetector> gftt_;
        const int num_features_tracking_{50};
        const int num_features_tracking_bad_{20};
        const int num_features_needed_for_keyframe_{80};
    };
}

#endif //FRONTEND_H
