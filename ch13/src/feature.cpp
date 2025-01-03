#include "feature.h"
#include <memory>
#include "frame.h"

namespace myslam {
    std::shared_ptr<Feature> Feature::Create(const cv::KeyPoint &keypoint, std::shared_ptr<Frame> frame, bool is_left) {
        static unsigned long factory_id = 0;
        return std::shared_ptr<Feature>(new Feature(factory_id++, keypoint, frame, is_left));
    }

    Feature::Feature(unsigned long id, const cv::KeyPoint &keypoint, std::shared_ptr<Frame> frame, bool is_left): id_(id), keypoint_(keypoint), frame_(frame), is_left_(is_left) {
    }

    std::shared_ptr<Frame> Feature::GetFrame() const {
        return frame_.lock();
    }

}
