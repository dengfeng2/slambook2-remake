#ifndef BACKEND_H
#define BACKEND_H
#include <Eigen/Core>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "map.h"

namespace myslam {
    struct Camera;
    class Frame;

    class Backend {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Backend(std::shared_ptr<Camera> camera_left, std::shared_ptr<Camera> camera_right, std::shared_ptr<Map> map);

        void Stop();

        void UpdateMap();

    private:

        void BackendLoop();

        void Optimize(const Map::KeyframesType& keyframes, const Map::LandmarksType& landmarks) const;

        const std::shared_ptr<Camera> camera_left_;
        const std::shared_ptr<Camera> camera_right_;
        const std::shared_ptr<Map> map_;

        std::mutex mutex_;
        std::thread backend_thread_;
        std::condition_variable map_update_;
        std::atomic<bool> backend_running_{false};
    };
}  // namespace myslam
#endif //BACKEND_H
