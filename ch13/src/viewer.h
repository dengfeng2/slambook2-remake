#ifndef VIEWER_H
#define VIEWER_H
#include <Eigen/Core>
#include <memory>
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>
#include "map.h"
namespace myslam {
    class Frame;
    class Map;

    class Viewer {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Viewer(std::shared_ptr<Map> map);

        void AddCurrentFrame(std::shared_ptr<Frame> frame);

        void UpdateMap();

        void Close();

    private:
        void ThreadLoop();
        static void DrawFrame(const std::shared_ptr<Frame> &frame, const float* color);
        void DrawMapPoints() const;
        cv::Mat PlotFrameImage() const;
        std::thread viewer_thread_;
        bool viewer_running_ = true;

        std::mutex mutex_;
        std::shared_ptr<Map> map_;
        std::shared_ptr<Frame> current_frame_;
        Map::LandmarksType active_landmarks_;
        Map::KeyframesType active_keyframes_;
    };
}  // namespace myslam
#endif //VIEWER_H
