add_executable(dense_mapping dense_mapping.cpp)
target_include_directories(dense_mapping PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(dense_mapping ${OpenCV_LIBS} Eigen3::Eigen Sophus::Sophus)