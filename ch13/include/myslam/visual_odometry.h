#ifndef SLAMBOOK2_REMAKE_VISUAL_ODOMETRY_H
#define SLAMBOOK2_REMAKE_VISUAL_ODOMETRY_H

namespace myslam {

    enum class FrontendStatus {
        INITING, TRACKING_GOOD, TRACKING_BAD, LOST
    };

    /**
     * VO 对外接口
     */
    class VisualOdometry {
    public:
        ~VisualOdometry() = default;

        /**
         * do initialization things before run
         * @return true if success
         */
        virtual bool Init() = 0;

        /**
         * start vo in the dataset
         */
        virtual void Run() = 0;

        /**
         * Make a step forward in dataset
         */
        virtual bool Step() = 0;

        /// 获取前端状态
        virtual FrontendStatus GetFrontendStatus() const = 0;

    };
}  // namespace myslam

#endif //SLAMBOOK2_REMAKE_VISUAL_ODOMETRY_H
