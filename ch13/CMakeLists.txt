include_directories(include)
add_library(myslam SHARED
        src/camera.cpp
        src/config.cpp
        src/dataset.cpp
        src/visual_odometry.cpp
        src/visual_odometry_impl.cpp
)
target_include_directories(myslam PUBLIC
        ${OpenCV_INCLUDE_DIRS})
target_link_libraries(myslam
        Eigen3::Eigen
        Sophus::Sophus
        ${OpenCV_LIBS}
        glog::glog
        fmt::fmt
)