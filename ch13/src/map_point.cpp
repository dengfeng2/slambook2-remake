#include "map_point.h"

#include <utility>

#include "feature.h"

namespace myslam {
    std::shared_ptr<MapPoint> MapPoint::Create(Eigen::Vector3d pos) {
        static unsigned long factory_id = 0;
        return std::shared_ptr<MapPoint>(new MapPoint(factory_id++, std::move(pos)));
    }

    MapPoint::MapPoint(unsigned long id, Eigen::Vector3d pos) : id_(id), pos_(std::move(pos)) {
    }

    Eigen::Vector3d MapPoint::Pos() {
        std::unique_lock<std::mutex> lock(pose_mutex_);
        return pos_;
    }

    void MapPoint::SetPos(const Eigen::Vector3d &pos) {
        std::unique_lock<std::mutex> lock(pose_mutex_);
        pos_ = pos;
    }

    void MapPoint::AddObservation(const std::shared_ptr<Feature> &feature) {
        std::unique_lock<std::mutex> lock(pose_mutex_);
        observations_.push_back(feature);
    }

    void MapPoint::RemoveObservation(const std::shared_ptr<Feature> &feat) {
        std::unique_lock<std::mutex> lock(pose_mutex_);
        for (auto iter = observations_.begin(); iter != observations_.end(); ++iter) {
            if (iter->lock() == feat) {
                observations_.erase(iter);
                feat->reset_map_point();
                break;
            }
        }
    }

    bool MapPoint::IsNoObservation() const {
        return observations_.empty();
    }

} // namespace myslam
