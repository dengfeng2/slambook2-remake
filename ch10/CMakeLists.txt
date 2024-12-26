add_executable(pose_graph_g2o_SE3 pose_graph_g2o_SE3.cpp)
target_link_libraries(pose_graph_g2o_SE3 g2o::core g2o::stuff g2o::types_slam3d)

add_executable(pose_graph_g2o_lie_algebra pose_graph_g2o_lie_algebra.cpp)
target_link_libraries(pose_graph_g2o_lie_algebra Eigen3::Eigen Sophus::Sophus g2o::core g2o::stuff g2o::types_slam3d)