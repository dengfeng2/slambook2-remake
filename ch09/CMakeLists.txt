add_library(bal_common common.cpp)
target_link_libraries(bal_common Eigen3::Eigen Sophus::Sophus)

add_executable(bundle_adjustment_ceres bundle_adjustment_ceres.cpp)
target_link_libraries(bundle_adjustment_ceres Ceres::ceres bal_common)

file(COPY problem-16-22106-pre.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(bundle_adjusment_g2o bundle_adjusment_g2o.cpp)
target_link_libraries(bundle_adjusment_g2o bal_common  g2o::g2o_ceres_ad g2o::stuff g2o::opengl_helper g2o::core g2o::g2o_cli_library g2o::g2o_simulator_library g2o::types_slam2d g2o::types_slam2d_addons g2o::types_sclam2d g2o::types_slam3d g2o::types_slam3d_addons g2o::types_sba g2o::types_icp g2o::types_sim3 g2o::solver_pcg g2o::solver_dense g2o::solver_slam2d_linear g2o::solver_structure_only g2o::solver_csparse g2o::csparse_extension g2o::solver_cholmod g2o::solver_eigen g2o::g2o_hierarchical_library)