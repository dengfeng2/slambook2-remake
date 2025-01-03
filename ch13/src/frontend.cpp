#include "frontend.h"
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "camera.h"
#include "frame.h"
#include "map_point.h"
#include "map.h"
#include "viewer.h"
#include "g2o_types.h"
#include "backend.h"

namespace myslam {
    Frontend::Frontend(std::shared_ptr<Camera> camera_left, std::shared_ptr<Camera> camera_right,
                       std::shared_ptr<Map> map, std::shared_ptr<Viewer> viewer, std::shared_ptr<Backend> backend)
        : camera_left_(std::move(camera_left)), camera_right_(std::move(camera_right)), map_(std::move(map)), viewer_(std::move(viewer)), backend_(std::move(backend)) {
        gftt_ = cv::GFTTDetector::create(num_features_, 0.01, 20);
    }

    bool Frontend::AddFrame(std::shared_ptr<Frame> frame) {
        current_frame_ = std::move(frame);
        bool ret = false;
        switch (status_) {
            case FrontendStatus::INITIALIZING:
                ret = StereoInit();
                break;
            case FrontendStatus::TRACKING_GOOD:
            case FrontendStatus::TRACKING_BAD:
                ret = Track();
                break;
            case FrontendStatus::LOST:
                ret = Reset();
                break;
        }
        last_frame_ = current_frame_;
        LOG(INFO) << "=============[Frontend::AddFrame] " << current_frame_->id() << " is key frame:" << (
            current_frame_->is_keyframe() ? "[Y]key_id="+std::to_string(current_frame_->key_frame_id()) : "[N]") << "===============";
        return ret;
    }

    bool Frontend::StereoInit() {
        auto num_features_left = DetectFeatures();
        auto num_matched_features = FindFeaturesInRight();
        if (num_matched_features < num_features_init_) {
            LOG(INFO) << "detect frame " << current_frame_->id() << ", left feature num is " << num_features_left <<
                    ", but matched feature num is " << num_matched_features;
            return false;
        }
        BuildInitMap();

        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        return true;
    }

    bool Frontend::Track() {
        if (last_frame_) {
            current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
        }
        auto num_track_last = TrackLastFrame();
        auto tracking_inlier_num = EstimateCurrentPose();
        if (tracking_inlier_num > num_features_tracking_) {
            status_ = FrontendStatus::TRACKING_GOOD;
        } else if (tracking_inlier_num > num_features_tracking_bad_) {
            status_ = FrontendStatus::TRACKING_BAD;
        } else {
            status_ = FrontendStatus::LOST;
            LOG(INFO) << "detect frame " << current_frame_->id() << ", left track feature num is " << num_track_last <<
                    ", but matched feature num is " << tracking_inlier_num;
        }
        if (tracking_inlier_num < num_features_needed_for_keyframe_) {
            InsertKeyFrame();
            // else: still have enough features, don't insert keyframe
        }

        relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

        if (viewer_) {viewer_->AddCurrentFrame(current_frame_);};
        return true;
    }

    bool Frontend::Reset() {
        LOG(INFO) << "Reset is not implemented. ";
        return true;
    }

    size_t Frontend::DetectFeatures() const {
        cv::Mat mask(current_frame_->left_img().size(), CV_8UC1, 255);
        for (auto &feat: current_frame_->left_features()) {
            cv::rectangle(mask, feat->keypoint().pt - cv::Point2f(10, 10),
                          feat->keypoint().pt + cv::Point2f(10, 10), 0, cv::FILLED);
        }
        std::vector<cv::KeyPoint> keyPoints;
        gftt_->detect(current_frame_->left_img(), keyPoints, mask);
        for (const auto &kp: keyPoints) {
            current_frame_->AddLeftFeature(Feature::Create(kp, current_frame_, true));
        }
        LOG(INFO) << "Detect " << keyPoints.size() << " new features";
        return keyPoints.size();
    }

    int Frontend::FindFeaturesInRight() const {
        std::vector<cv::Point2f> kps_left, kps_right;
        for (const auto &feat: current_frame_->left_features()) {
            kps_left.push_back(feat->keypoint().pt);
            if (auto mp = feat->map_point(); mp) {
                auto px = camera_right_->k * (current_frame_->Pose() * mp->Pos());
                kps_right.emplace_back(px[0] / px[2], px[1] / px[2]);
            } else {
                kps_right.push_back(feat->keypoint().pt);
            }
        }
        std::vector<uchar> status;
        cv::Mat error;
        cv::calcOpticalFlowPyrLK(
            current_frame_->left_img(), current_frame_->right_img(), kps_left,
            kps_right, status, error, cv::Size(11, 11), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;
        for (int i = 0; i < status.size(); ++i) {
            if (status[i]) {
                const cv::KeyPoint kp(kps_right[i], 7);
                current_frame_->AddRightFeature(Feature::Create(kp, current_frame_, false));
                num_good_pts++;
            } else {
                current_frame_->AddRightFeature(nullptr);
            }
        }
        LOG(INFO) << "Find " << num_good_pts << " in the right image.";
        return num_good_pts;
    }

    void Frontend::BuildInitMap() const {
        std::vector<cv::Point2f> kps_left;
        std::vector<cv::Point2f> kps_right;
        for (size_t i = 0; i < current_frame_->left_features().size(); ++i) {
            if (!current_frame_->right_features()[i]) { continue; }
            kps_left.push_back(current_frame_->left_features()[i]->keypoint().pt);
            kps_right.push_back(current_frame_->right_features()[i]->keypoint().pt);
        }
        cv::Mat points4D;
        cv::triangulatePoints(camera_left_->Kt(), camera_right_->Kt(), kps_left, kps_right, points4D);

        int cnt_init_landmarks = 0;
        for (size_t i = 0; i < current_frame_->left_features().size(); ++i) {
            if (!current_frame_->right_features()[i]) { continue; }
            cv::Vec4f homoPoint = points4D.col(cnt_init_landmarks);
            homoPoint /= homoPoint[3];
            Eigen::Vector3d world_point = Eigen::Vector3d(homoPoint[0], homoPoint[1], homoPoint[2]);
            auto new_world_point = MapPoint::Create(world_point);
            new_world_point->AddObservation(current_frame_->left_features()[i]);
            new_world_point->AddObservation(current_frame_->right_features()[i]);
            current_frame_->AddMapPoint(i, new_world_point);
            map_->InsertMapPoint(new_world_point);
            cnt_init_landmarks++;
        }

        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);
        //backend_->UpdateMap();

        LOG(INFO) << "Initial map created with " << cnt_init_landmarks
                << " map points";
    }

    int Frontend::TrackLastFrame() const {
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_last, kps_current;
        for (auto &feat: last_frame_->left_features()) {
            if (auto mp = feat->map_point(); mp) {
                // use project point
                auto px = camera_left_->k * (current_frame_->Pose() * mp->Pos());
                kps_last.push_back(feat->keypoint().pt);
                kps_current.emplace_back(px[0] / px[2], px[1] / px[2]);
            } else {
                kps_last.push_back(feat->keypoint().pt);
                kps_current.push_back(feat->keypoint().pt);
            }
        }

        std::vector<uchar> status;
        cv::Mat error;
        cv::calcOpticalFlowPyrLK(
            last_frame_->left_img(), current_frame_->left_img(), kps_last,
            kps_current, status, error, cv::Size(11, 11), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);
        int num_good_pts = 0;

        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                cv::KeyPoint kp(kps_current[i], 7);
                auto feat = Feature::Create(kp, current_frame_, true);
                feat->set_map_point(last_frame_->left_features()[i]->map_point());
                current_frame_->AddLeftFeature(feat);
                num_good_pts++;
            }
        }

        LOG(INFO) << "Find " << num_good_pts << " in the last image.";
        return num_good_pts;
    }

    size_t Frontend::EstimateCurrentPose() const {
        // setup g2o
        using LinearSolverType = g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<g2o::BlockSolver_6_3>(std::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        // vertex
        auto *vertex_pose = new VertexPose(); // camera vertex_pose
        vertex_pose->setId(0);
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.addVertex(vertex_pose);

        // edges
        int index = 1;
        std::vector<EdgeProjectionPoseOnly *> edges;
        std::vector<std::shared_ptr<Feature> > features;
        for (auto &feat: current_frame_->left_features()) {
            if (auto mp = feat->map_point(); mp) {
                features.push_back(feat);
                auto *edge = new EdgeProjectionPoseOnly(mp->Pos(), camera_left_->k);
                edge->setId(index);
                edge->setVertex(0, vertex_pose);
                auto pt = feat->keypoint().pt;
                edge->setMeasurement({pt.x, pt.y});
                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber);
                edges.push_back(edge);
                optimizer.addEdge(edge);
                index++;
            }
        }
        // estimate the Pose the determine the outliers
        constexpr double chi2_th = 5.991;
        int cnt_outlier = 0;
        for (int iteration = 0; iteration < 4; ++iteration) {
            vertex_pose->setEstimate(current_frame_->Pose());
            optimizer.initializeOptimization();
            optimizer.optimize(10);
            cnt_outlier = 0;

            // count the outliersc
            for (size_t i = 0; i < edges.size(); ++i) {
                auto e = edges[i];
                if (features[i]->is_outlier()) {
                    e->computeError();
                }
                if (e->chi2() > chi2_th) {
                    features[i]->set_outlier(true);
                    e->setLevel(1);
                    cnt_outlier++;
                } else {
                    features[i]->set_outlier(false);
                    e->setLevel(0);
                };

                if (iteration == 2) {
                    e->setRobustKernel(nullptr);
                }
            }
        }

        LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
                << features.size() - cnt_outlier;
        // Set pose and outlier
        current_frame_->SetPose(vertex_pose->estimate());

        LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

        for (auto &feat: features) {
            if (feat->is_outlier()) {
                feat->reset_map_point();
                feat->set_outlier(false); // maybe we can still use it in future
            }
        }
        return features.size() - cnt_outlier;
    }

    void Frontend::InsertKeyFrame() const {
        // current frame is a new keyframe
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);

        LOG(INFO) << "Set frame " << current_frame_->id() << " as keyframe "
                << current_frame_->key_frame_id();

        SetObservationsForKeyFrame();
        auto new_num_features_left = DetectFeatures();
        auto num_matched_features = FindFeaturesInRight();

        auto new_num_world_point = TriangulateNewPoints();
        LOG(INFO) << "insert key frame " << current_frame_->id() << ", left new feature num is " <<
                new_num_features_left <<
                ", matched feature num is " << num_matched_features << ", new feature num is " << new_num_world_point;
        // update backend because we have a new keyframe
        //backend_->UpdateMap();

        if (viewer_) {viewer_->UpdateMap();}
    }

    void Frontend::SetObservationsForKeyFrame() const {
        for (auto &feat: current_frame_->left_features()) {
            if (auto mp = feat->map_point(); mp) { mp->AddObservation(feat); }
        }
    }

    size_t Frontend::TriangulateNewPoints() const {
        std::vector<int> indices;
        std::vector<cv::Point2f> kps_left;
        std::vector<cv::Point2f> kps_right;
        for (int i = 0; i < current_frame_->left_features().size(); ++i) {
            if (!current_frame_->left_features()[i]->map_point() && current_frame_->right_features()[i]) {
                kps_left.push_back(current_frame_->left_features()[i]->keypoint().pt);
                kps_right.push_back(current_frame_->right_features()[i]->keypoint().pt);
                indices.push_back(i);
            }
        }
        cv::Mat points4D;
        cv::triangulatePoints(camera_left_->Kt(), camera_right_->Kt(), kps_left, kps_right, points4D);

        for (int i = 0; i < indices.size(); i++) {
            cv::Vec4f homoPoint = points4D.col(i);
            auto idx = indices[i];
            homoPoint /= homoPoint[3];
            Eigen::Vector3d pworld = current_frame_->Pose().inverse() * Eigen::Vector3d(homoPoint[0], homoPoint[1], homoPoint[2]);
            auto new_world_point = MapPoint::Create(pworld);
            new_world_point->AddObservation(current_frame_->left_features()[idx]);
            new_world_point->AddObservation(current_frame_->right_features()[idx]);
            current_frame_->AddMapPoint(idx, new_world_point);
            map_->InsertMapPoint(new_world_point);
        }
        LOG(INFO) << "new landmarks: " << indices.size();
        return indices.size();
    }
}
