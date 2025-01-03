#include "visual_odometry_impl.h"

#include <glog/logging.h>

#include "dataset.h"
#include "frontend.h"
#include "map.h"
#include "viewer.h"
#include "backend.h"

namespace myslam {

    std::shared_ptr<VisualOdometry> VisualOdometry::Create(const std::string &dataset_path) {
        return std::make_shared<VisualOdometryImpl>(dataset_path);
    }

    bool VisualOdometryImpl::Init() {
        dataset_ = std::make_shared<Dataset>(dataset_path_);
        CHECK_EQ(dataset_->Init(), true);
        auto map = std::make_shared<Map>();
        viewer_ = std::make_shared<Viewer>(map);
        backend_ = std::make_shared<Backend>(dataset_->GetCamera(0), dataset_->GetCamera(1), map);
        frontend_ = std::make_shared<Frontend>(dataset_->GetCamera(0), dataset_->GetCamera(1), map, viewer_, backend_);

        return true;
    }

    void VisualOdometryImpl::Run() {
        while (true) {
            LOG(INFO) << "VO is running";
            if (!Step()) {
                break;
            }
        }

        backend_->Stop();
        viewer_->Close();

        LOG(INFO) << "VO exit";
    }

    bool VisualOdometryImpl::Step() {
        auto new_frame = dataset_->NextFrame();
        if (new_frame == nullptr) return false;

        auto t1 = std::chrono::steady_clock::now();
        bool success = frontend_->AddFrame(new_frame);
        auto t2 = std::chrono::steady_clock::now();
        auto time_used =
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
        return success;
    }

    FrontendStatus VisualOdometryImpl::GetFrontendStatus() const { return frontend_->GetStatus(); }

}  // namespace myslam
