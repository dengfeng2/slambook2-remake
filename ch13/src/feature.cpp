#include "feature.h"
#include <memory>
#include "frame.h"

namespace myslam {
    std::shared_ptr<Feature> Feature::Create(const cv::KeyPoint &keypoint) {
        static unsigned long factory_id = 0;
        return std::shared_ptr<Feature>(new Feature(factory_id++, keypoint));
    }

    Feature::Feature(unsigned long id, const cv::KeyPoint &keypoint) : id_(id), keypoint_(keypoint) {
    }

}
