#ifndef SLAMBOOK2_REMAKE_FRAME_H
#define SLAMBOOK2_REMAKE_FRAME_H
#include <Eigen/Core>
#include <memory>
#include <mutex>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

namespace myslam {
    struct Feature;

    /**
     * 帧
     * 每一帧分配独立id，关键帧分配关键帧ID
     */
    struct Frame {

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        unsigned long id_ = 0;           // id of this frame
        unsigned long keyframe_id_ = 0;  // id of key frame
        bool is_keyframe_ = false;       // 是否为关键帧
        double time_stamp_;              // 时间戳，暂不使用
        Sophus::SE3d pose_;                       // Tcw 形式Pose
        std::mutex pose_mutex_;          // Pose数据锁
        cv::Mat left_img_, right_img_;   // stereo images

        // extracted features in left image
        std::vector<std::shared_ptr<Feature>> features_left_;
        // corresponding features in right image, set to nullptr if no corresponding
        std::vector<std::shared_ptr<Feature>> features_right_;

        explicit Frame(long id) : id_(id) {}

        Frame(long id, double time_stamp, const Sophus::SE3d &pose, const cv::Mat &left, const cv::Mat &right)
                : id_(id), time_stamp_(time_stamp), pose_(pose), left_img_(left), right_img_(right) {}

        // set and get pose, thread safe
        Sophus::SE3d Pose() {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            return pose_;
        }

        void SetPose(const Sophus::SE3d &pose) {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            pose_ = pose;
        }

        /// 设置关键帧并分配并键帧id
        void SetKeyFrame() {
            static long keyframe_factory_id = 0;
            is_keyframe_ = true;
            keyframe_id_ = keyframe_factory_id++;
        }

        /// 工厂构建模式，分配id
        static std::shared_ptr<Frame> CreateFrame() {
            static long factory_id = 0;
            auto id = factory_id++;
            return std::make_shared<Frame>(id);
        }
    };

}  // namespace myslam

#endif //SLAMBOOK2_REMAKE_FRAME_H
