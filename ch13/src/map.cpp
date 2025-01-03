#include "map.h"

#include <glog/logging.h>
#include "frame.h"
#include "map_point.h"

namespace myslam {
    void Map::InsertKeyFrame(const std::shared_ptr<Frame> &frame) {
        CHECK_EQ(active_keyframes_.find(frame->id()) == active_keyframes_.end(), true);
        frame->SetKeyFrame();
        active_keyframes_[frame->id()] = frame;

        if (active_keyframes_.size() > num_active_keyframes_) {
            RemoveOldKeyframe(frame);
        }
    }

    void Map::InsertMapPoint(const std::shared_ptr<MapPoint> &map_point) {
        CHECK_EQ(active_landmarks_.find(map_point->id()) == active_landmarks_.end(), true);
        active_landmarks_[map_point->id()] = map_point;
    }

    void Map::RemoveOldKeyframe(const std::shared_ptr<Frame> &current_frame) {
        CHECK_NOTNULL(current_frame);
        double max_dis = 0, min_dis = 9999;
        unsigned long max_kf_id = 0, min_kf_id = 0;
        auto Twc = current_frame->Pose().inverse();
        for (auto &[frame_id, kf]: active_keyframes_) {
            if (frame_id == current_frame->id()) { continue; }
            auto dis = (kf->Pose() * Twc).log().norm();
            if (dis > max_dis) {
                max_dis = dis;
                max_kf_id = frame_id;
            }
            if (dis < min_dis) {
                min_dis = dis;
                min_kf_id = frame_id;
            }
        }
        constexpr double min_dis_th = 0.2;
        auto kf_id_to_remove = min_dis < min_dis_th ? min_kf_id : max_kf_id;

        LOG(INFO) << "remove keyframe " << kf_id_to_remove;
        // remove keyframe and landmark observation
        for (const auto &feat: active_keyframes_[kf_id_to_remove]->left_features()) {
            CHECK_NOTNULL(feat);
            if (auto mp = feat->map_point(); mp) {
                mp->RemoveObservation(feat);
            }
        }
        for (const auto &feat: active_keyframes_[kf_id_to_remove]->right_features()) {
            if (feat == nullptr) { continue; }
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
