#ifndef MAP_H
#define MAP_H

#include <memory>
#include <Eigen/Core>

namespace myslam {
    struct MapPoint;

    class Frame;

    class Map {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        using LandmarksType = std::unordered_map<unsigned long, std::shared_ptr<MapPoint> >;
        using KeyframesType = std::unordered_map<unsigned long, std::shared_ptr<Frame> >;

        void InsertMapPoint(const std::shared_ptr<MapPoint> &map_point);

        void InsertKeyFrame(const std::shared_ptr<Frame> &frame);

        LandmarksType GetActiveLandmarks() const { return active_landmarks_; }

        KeyframesType GetActiveKeyFrames() const { return active_keyframes_; }

    private:
        void RemoveOldKeyframe(const std::shared_ptr<Frame> &current_frame);

        void CleanMap();

        const int num_active_keyframes_ = 7;

        LandmarksType active_landmarks_;  // active landmarks
        KeyframesType active_keyframes_;  // all key-frames
    };
}
#endif //MAP_H
