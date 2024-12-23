add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton Eigen3::Eigen)

add_executable(ceresCurveFitting ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting Eigen3::Eigen Ceres::ceres)

add_executable(g2oCurveFitting g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting Eigen3::Eigen Ceres::ceres g2o::core)