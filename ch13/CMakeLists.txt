include_directories(include)
add_library(myslam SHARED
        src/dataset.cpp
        src/feature.cpp
        src/map_point.cpp
        src/map.cpp
        src/viewer.cpp
        src/frontend.cpp
        src/backend.cpp
        src/visual_odometry_impl.cpp
)
target_include_directories(myslam PUBLIC
        ${OpenCV_INCLUDE_DIRS})
target_link_libraries(myslam PUBLIC
        Eigen3::Eigen
        Sophus::Sophus
        ${OpenCV_LIBS}
        glog::glog
        fmt::fmt
        g2o::core g2o::types_slam3d g2o::solver_dense g2o::solver_csparse g2o::csparse_extension g2o::solver_eigen
        ${Pangolin_LIBRARIES}
)


add_executable(run_kitti_stereo app/run_kitti_stereo.cpp)
target_link_libraries(run_kitti_stereo PUBLIC myslam)