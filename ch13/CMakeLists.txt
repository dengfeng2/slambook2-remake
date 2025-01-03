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
        g2o::g2o_ceres_ad g2o::stuff g2o::opengl_helper g2o::core g2o::g2o_cli_library g2o::g2o_simulator_library g2o::types_slam2d g2o::types_slam2d_addons g2o::types_sclam2d g2o::types_slam3d g2o::types_slam3d_addons g2o::types_sba g2o::types_icp g2o::types_sim3 g2o::solver_pcg g2o::solver_dense g2o::solver_slam2d_linear g2o::solver_structure_only g2o::solver_csparse g2o::csparse_extension g2o::solver_cholmod g2o::solver_eigen g2o::g2o_hierarchical_library
        ${Pangolin_LIBRARIES}
)


add_executable(run_kitti_stereo app/run_kitti_stereo.cpp)
target_link_libraries(run_kitti_stereo myslam)