#include "backend.h"

#include <glog/logging.h>

#include "camera.h"
#include "frame.h"
#include "map.h"
#include "g2o_types.h"
#include "map_point.h"

namespace myslam {
    Backend::Backend(std::shared_ptr<Camera> camera_left, std::shared_ptr<Camera> camera_right,
                     std::shared_ptr<Map> map)
            : camera_left_(std::move(camera_left)), camera_right_(std::move(camera_right)), map_(std::move(map)) {
        backend_running_.store(true);
        backend_thread_ = std::thread([this] { BackendLoop(); });
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
            Map::LandmarksType active_landmarks = map_->GetActiveLandmarks();
            Optimize(active_kfs, active_landmarks);
        }
    }

    void Backend::Optimize(const Map::KeyframesType &keyframes,
                           const Map::LandmarksType &landmarks) const {
        // setup g2o
        using LinearSolverType = g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
                std::make_unique<g2o::BlockSolver_6_3>(std::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        std::map<unsigned long, VertexPose *> vertices;
        unsigned long max_kf_id = 0;
        for (const auto &[key_frame_id, kf]: keyframes) {
            auto *vertex_pose = new VertexPose();  // camera vertex_pose
            vertex_pose->setId(kf->id());
            vertex_pose->setEstimate(kf->Pose());
            optimizer.addVertex(vertex_pose);

            max_kf_id = std::max(max_kf_id, kf->id());
            vertices.insert({kf->id(), vertex_pose});
        }

        std::map<unsigned long, VertexXYZ *> vertices_landmarks;
        for (const auto &[lmk_id, lmk]: landmarks) {
            auto *v = new VertexXYZ();
            v->setId(lmk->id() + max_kf_id + 1);
            v->setEstimate(lmk->Pos());
            v->setMarginalized(true);
            optimizer.addVertex(v);

            vertices_landmarks.insert({lmk->id(), v});
        }

        // edges
        double chi2_th = 5.991;
        std::map<EdgeProjection *, std::shared_ptr<Feature>> edges_and_features;

        int edgeIndex = 0;
        for (auto &keyframe : keyframes) {
            const auto kf = keyframe.second;
            for (const auto &feat : kf->left_features()) {
                if (feat && !feat->is_outlier() && feat->map_point()) {
                    auto mp = feat->map_point();
                    auto *edge = new EdgeProjection(camera_left_->k, camera_left_->t);
                    edge->setId(edgeIndex++);
                    edge->setVertex(0, vertices.at(kf->id()));
                    edge->setVertex(1, vertices_landmarks.at(mp->id()));
                    edge->setMeasurement({feat->keypoint().pt.x, feat->keypoint().pt.y});
                    edge->setInformation(Eigen::Matrix2d::Identity());
                    auto *rk = new g2o::RobustKernelHuber();
                    rk->setDelta(chi2_th);
                    edge->setRobustKernel(rk);
                    optimizer.addEdge(edge);

                    edges_and_features.insert({edge, feat});
                }
            }
            for (const auto &feat : kf->right_features()) {
                if (feat && !feat->is_outlier() && feat->map_point()) {
                    auto mp = feat->map_point();
                    auto *edge = new EdgeProjection(camera_right_->k, camera_right_->t);
                    edge->setId(edgeIndex++);
                    edge->setVertex(0, vertices.at(kf->id()));
                    edge->setVertex(1, vertices_landmarks.at(mp->id()));
                    edge->setMeasurement({feat->keypoint().pt.x, feat->keypoint().pt.y});
                    edge->setInformation(Eigen::Matrix2d::Identity());
                    auto *rk = new g2o::RobustKernelHuber();
                    rk->setDelta(chi2_th);
                    edge->setRobustKernel(rk);
                    optimizer.addEdge(edge);

                    edges_and_features.insert({edge, feat});
                }
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
            for (auto &ef: edges_and_features) {
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

        for (auto& [edge, feat]: edges_and_features) {
            if (edge->chi2() > chi2_th) {
                feat->set_outlier();
                feat->map_point()->RemoveObservation(feat);
                feat->reset_map_point();
            } else {
                CHECK_EQ(feat->is_outlier(), false);
            }
        }

        LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/" << cnt_inlier << " for chi2_th=" << chi2_th;

        // Set pose and landmark position
        for (auto &[key_frame_id, vertex_pose]: vertices) {
            auto dis = (vertex_pose->estimate() * keyframes.at(key_frame_id)->Pose().inverse()).log().norm();
            keyframes.at(key_frame_id)->SetPose(vertex_pose->estimate());
            LOG(INFO) << "After backend, the distance of pose between frontend and backend is " << dis;
        }
        for (auto &[lmk_id, vertex_xyz]: vertices_landmarks) {
            landmarks.at(lmk_id)->SetPos(vertex_xyz->estimate());
        }
    }


}  // namespace myslam
