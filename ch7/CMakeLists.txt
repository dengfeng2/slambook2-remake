add_executable(orb_cv orb_cv.cpp)
target_include_directories(orb_cv PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(orb_self orb_self.cpp)
target_include_directories(orb_self PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orb_self ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_include_directories(pose_estimation_2d2d PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_include_directories(triangulation PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_include_directories(pose_estimation_3d2d PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} Eigen3::Eigen Sophus::Sophus g2o::core)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_include_directories(pose_estimation_3d3d PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS} Eigen3::Eigen Sophus::Sophus g2o::core)

file(COPY 1.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY 2.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY 1_depth.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY 2_depth.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR})