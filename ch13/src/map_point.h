#ifndef MAP_POINT_H
#define MAP_POINT_H

#include <Eigen/Core>
#include <list>
#include <mutex>
#include <memory>

namespace myslam {
    class Feature;

    class MapPoint {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        static std::shared_ptr<MapPoint> Create(Eigen::Vector3d pos);

        unsigned long id() const { return id_; }

        Eigen::Vector3d Pos();

        void SetPos(const Eigen::Vector3d &pos);

        void AddObservation(const std::shared_ptr<Feature> &feature);

        void RemoveObservation(const std::shared_ptr<Feature> &feature);

        bool IsNoObservation() const;

    private:
        MapPoint(unsigned long id, Eigen::Vector3d pos);

        const unsigned long id_;
        Eigen::Vector3d pos_;
        std::mutex pose_mutex_;
        std::list<std::weak_ptr<Feature> > observations_;;
    };
} // namespace myslam
#endif //MAP_POINT_H
