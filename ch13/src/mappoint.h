#ifndef SLAMBOOK2_REMAKE_MAPPOINT_H
#define SLAMBOOK2_REMAKE_MAPPOINT_H
#include <memory>
#include <mutex>
#include <list>
#include <Eigen/Core>

namespace myslam {

    struct Feature;

    /**
     * 路标点类
     * 特征点在三角化之后形成路标点
     */
    struct MapPoint {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        unsigned long id_ = 0;  // ID
        bool is_outlier_ = false;
        Eigen::Vector3d pos_ = Eigen::Vector3d::Zero();  // Position in world
        std::mutex data_mutex_;
        int observed_times_ = 0;  // being observed by feature matching algo.
        std::list<std::weak_ptr<Feature>> observations_;

        MapPoint() = default;

        MapPoint(long id, Eigen::Vector3d position);

        Eigen::Vector3d Pos() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return pos_;
        }

        void SetPos(const Eigen::Vector3d &pos) {
            std::unique_lock<std::mutex> lck(data_mutex_);
            pos_ = pos;
        };

        void AddObservation(std::shared_ptr<Feature> feature) {
            std::unique_lock<std::mutex> lck(data_mutex_);
            observations_.push_back(feature);
            observed_times_++;
        }

        void RemoveObservation(std::shared_ptr<Feature> feat);

        std::list<std::weak_ptr<Feature>> GetObs() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return observations_;
        }

        // factory function
        static std::shared_ptr<MapPoint> CreateNewMappoint();
    };
}  // namespace myslam



#endif //SLAMBOOK2_REMAKE_MAPPOINT_H
