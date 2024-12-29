#ifndef SLAMBOOK2_REMAKE_DATASET_H
#define SLAMBOOK2_REMAKE_DATASET_H
#include <memory>
#include <Eigen/Core>

namespace myslam {
    struct Frame;
    struct Camera;

/**
 * 数据集读取
 * 构造时传入配置文件路径，配置文件的dataset_dir为数据集路径
 * Init之后可获得相机和下一帧图像
 */
    class Dataset {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Dataset(std::string  dataset_path);

        /// 初始化，返回是否成功
        bool Init();

        /// create and return the next frame containing the stereo images
        std::shared_ptr<Frame> NextFrame();

        /// get camera by id
        std::shared_ptr<Camera> GetCamera(int camera_id) const {
            return cameras_.at(camera_id);
        }

    private:
        std::string dataset_path_;
        int current_image_index_ = 0;

        std::vector<std::shared_ptr<Camera>> cameras_;
    };
}  // namespace myslam

#endif //SLAMBOOK2_REMAKE_DATASET_H
