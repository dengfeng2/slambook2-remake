#ifndef SLAMBOOK2_REMAKE_VIEWER_H
#define SLAMBOOK2_REMAKE_VIEWER_H

#include <thread>
#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>

namespace myslam {
    struct Map;
    struct Frame;

/**
 * 可视化
 */
    class Viewer {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Viewer();

        void SetMap(std::shared_ptr<Map> map) { map_ = map; }

        void Close();

        // 增加一个当前帧
        void AddCurrentFrame(std::shared_ptr<Frame> current_frame);

        // 更新地图
        void UpdateMap();

    private:
        void ThreadLoop();

        void DrawFrame(std::shared_ptr<Frame> frame, const float* color);

        void DrawMapPoints();

        void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

        /// plot the features in current frame into an image
        cv::Mat PlotFrameImage();

        std::shared_ptr<Frame> current_frame_ = nullptr;
        std::shared_ptr<Map> map_ = nullptr;

        std::thread viewer_thread_;
        bool viewer_running_ = true;

        std::unordered_map<unsigned long, std::shared_ptr<Frame>> active_keyframes_;
        std::unordered_map<unsigned long, std::shared_ptr<Map>> active_landmarks_;
        bool map_updated_ = false;

        std::mutex viewer_data_mutex_;
    };
}  // namespace myslam


#endif //SLAMBOOK2_REMAKE_VIEWER_H
