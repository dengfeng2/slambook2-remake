#include "map.h"

#include <glog/logging.h>
#include "frame.h"
#include "map_point.h"

namespace myslam {
    void Map::InsertKeyFrame(const std::shared_ptr<Frame> &frame) {
        current_frame_ = frame;
        active_keyframes_[frame->key_frame_id()] = frame;

        if (active_keyframes_.size() > num_active_keyframes_) {
            RemoveOldKeyframe();
        }
    }

    void Map::InsertMapPoint(const std::shared_ptr<MapPoint> &map_point) {
        active_landmarks_[map_point->id()] = map_point;
    }

    void Map::RemoveOldKeyframe() {
        if (current_frame_ == nullptr) { return; }
        double max_dis = 0, min_dis = 9999;
        long max_kf_id = 0, min_kf_id = 0;
        auto Twc = current_frame_->Pose().inverse();
        for (auto &kf: active_keyframes_) {
            if (kf.second == current_frame_) { continue; }
            auto dis = (kf.second->Pose() * Twc).log().norm();
            if (dis > max_dis) {
                max_dis = dis;
                max_kf_id = kf.first;
            }
            if (dis < min_dis) {
                min_dis = dis;
                min_kf_id = kf.first;
            }
        }
        constexpr double min_dis_th = 0.2;
        auto kf_id_to_remove = min_dis < min_dis_th ? min_kf_id : max_kf_id;

        LOG(INFO) << "remove keyframe " << kf_id_to_remove;
        // remove keyframe and landmark observation
        for (auto feat: active_keyframes_[kf_id_to_remove]->left_features()) {
            if (auto mp = feat->map_point(); mp) {
                mp->RemoveObservation(feat);
            }
        }
        for (auto feat: active_keyframes_[kf_id_to_remove]->right_features()) {
            if (feat == nullptr) {continue;}
            if (auto mp = feat->map_point();mp) {
                mp->RemoveObservation(feat);
            }
        }
        active_keyframes_.erase(kf_id_to_remove);

        CleanMap();
    }

    void Map::CleanMap() {
        int cnt_landmark_removed = 0;
        for (auto iter = active_landmarks_.begin(); iter != active_landmarks_.end();) {
            if (iter->second->IsNoObservation()) {
                iter = active_landmarks_.erase(iter);
                cnt_landmark_removed++;
            } else {
                ++iter;
            }
        }
        LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
    }
}
