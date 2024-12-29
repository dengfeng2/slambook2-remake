#ifndef SLAMBOOK2_REMAKE_VISUAL_ODOMETRY_IMPL_H
#define SLAMBOOK2_REMAKE_VISUAL_ODOMETRY_IMPL_H
#include "myslam/visual_odometry.h"

namespace myslam {
    struct Map;
    struct Viewer;
    struct Dataset;

    enum class FrontendStatus { INITING, TRACKING_GOOD, TRACKING_BAD, LOST };
    /**
     * VO 对外接口
     */
    class VisualOdometry {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        /// constructor with config file
        VisualOdometry(std::string &config_path);

        /**
         * do initialization things before run
         * @return true if success
         */
        bool Init();

        /**
         * start vo in the dataset
         */
        void Run();

        /**
         * Make a step forward in dataset
         */
        bool Step();

        /// 获取前端状态
        FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); }

    private:
        bool inited_ = false;
        std::string config_file_path_;

        std::shared_ptr<Frontend> frontend_ = nullptr;
        std::shared_ptr<Backend> backend_ = nullptr;
        std::shared_ptr<Map> map_ = nullptr;
        std::shared_ptr<Viewer> viewer_ = nullptr;

        // dataset
        std::shared_ptr<Dataset> dataset_ = nullptr;
    };
}  // namespace myslam
#endif //SLAMBOOK2_REMAKE_VISUAL_ODOMETRY_IMPL_H
