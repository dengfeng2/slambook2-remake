#ifndef SLAMBOOK2_REMAKE_CONFIG_H
#define SLAMBOOK2_REMAKE_CONFIG_H

#include <memory>
#include <opencv2/opencv.hpp>

namespace myslam {

    /**
    * 配置类，使用SetParameterFile确定配置文件
    * 然后用Get得到对应值
    * 单例模式
    */
    class Config {

    public:
        ~Config();  // close the file when deconstructing

        // set a new config file
        static bool SetParameterFile(const std::string &filename);

        // access the parameter values
        template<typename T>
        static T Get(const std::string &key) {
            return T(Config::config_->file_[key]);
        }

    private:
        static std::shared_ptr<Config> config_;
        cv::FileStorage file_;

        Config() {}  // private constructor makes a singleton
    };
}  // namespace myslam
#endif //SLAMBOOK2_REMAKE_CONFIG_H
