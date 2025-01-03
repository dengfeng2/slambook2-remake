#include "backend.h"

#include <glog/logging.h>

#include "camera.h"
#include "frame.h"
#include "map.h"
#include "g2o_types.h"
#include "map_point.h"

namespace myslam {
    Backend::Backend(std::shared_ptr<Camera> camera_left, std::shared_ptr<Camera> camera_right, std::shared_ptr<Map> map):camera_left_(std::move(camera_right)), camera_right_(std::move(camera_right)), map_(std::move(map)) {
        backend_running_.store(true);
        backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
    }

    void Backend::UpdateMap() {
        std::unique_lock<std::mutex> lock(mutex_);
        map_update_.notify_one();
    }

    void Backend::Stop() {
        backend_running_.store(false);
        map_update_.notify_one();
        backend_thread_.join();
    }

    void Backend::BackendLoop() {
        while (backend_running_.load()) {
            std::unique_lock<std::mutex> lock(mutex_);
            map_update_.wait(lock);

            Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
            Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
            Optimize(active_kfs, active_landmarks);
        }
    }

    void Backend::Optimize(const Map::KeyframesType &keyframes,
                       const Map::LandmarksType &landmarks) const {
    // setup g2o
    using LinearSolverType = g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::make_unique<g2o::BlockSolver_6_3>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    std::map<unsigned long, VertexPose *> vertices;
    unsigned long max_kf_id = 0;
    for (auto &keyframe : keyframes) {
        const auto kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
        vertex_pose->setId(kf->key_frame_id());
        vertex_pose->setEstimate(kf->Pose());
        optimizer.addVertex(vertex_pose);
        max_kf_id = std::max(max_kf_id, kf->key_frame_id());

        vertices.insert({kf->key_frame_id(), vertex_pose});
    }


    std::map<unsigned long, VertexXYZ *> vertices_landmarks;

    // edges
    int index = 1;
    double chi2_th = 5.991;
    std::map<EdgeProjection *, std::shared_ptr<Feature>> edges_and_features;

    for (auto &landmark : landmarks) {
        auto landmark_id = landmark.second->id();
        auto observations = landmark.second->GetObs();
        for (auto &obs : observations) {
            if (obs.expired()) {continue;}
            auto feat = obs.lock();
            if (feat->is_outlier()) {continue;}

            auto frame = feat->GetFrame();
            assert(frame != nullptr);
            EdgeProjection *edge = nullptr;
            if (feat->is_left()) {
                edge = new EdgeProjection(camera_left_->k, camera_left_->t);
            } else {
                edge = new EdgeProjection(camera_right_->k, camera_right_->t);
            }

            // ??landmark????????????????
            if (vertices_landmarks.find(landmark_id) == vertices_landmarks.end()) {
                VertexXYZ *v = new VertexXYZ();
                v->setEstimate(landmark.second->Pos());
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);
                vertices_landmarks.insert({landmark_id, v});
                optimizer.addVertex(v);
            }


            if (vertices.find(frame->key_frame_id()) != vertices.end() &&
                vertices_landmarks.find(landmark_id) != vertices_landmarks.end()) {
                    edge->setId(index);
                    edge->setVertex(0, vertices.at(frame->key_frame_id()));    // pose
                    edge->setVertex(1, vertices_landmarks.at(landmark_id));  // landmark
                    edge->setMeasurement({feat->keypoint().pt.x, feat->keypoint().pt.y});
                    edge->setInformation(Eigen::Matrix2d::Identity());
                    auto rk = new g2o::RobustKernelHuber();
                    rk->setDelta(chi2_th);
                    edge->setRobustKernel(rk);
                    edges_and_features.insert({edge, feat});
                    optimizer.addEdge(edge);
                    index++;
                }
            else {delete edge;}

        }
    }

    // do optimization and eliminate the outliers
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5) {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features) {
            if (ef.first->chi2() > chi2_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }
        double inlier_ratio = static_cast<double>(cnt_inlier) / (cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5) {
            break;
        } else {
            chi2_th *= 2;
            iteration++;
        }
    }

    for (auto &ef : edges_and_features) {
        if (ef.first->chi2() > chi2_th) {
            ef.second->set_outlier(true);
            // remove the observation
            ef.second->map_point()->RemoveObservation(ef.second);
        } else {
            ef.second->set_outlier(false);
        }
    }

    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
              << cnt_inlier;

    // Set pose and landmark position
    for (auto &v : vertices) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
    }
    for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->SetPos(v.second->estimate());
    }
}


}  // namespace myslam
