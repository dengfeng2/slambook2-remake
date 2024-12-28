add_executable(pointcloud_mapping pointcloud_mapping.cpp)
target_include_directories(pointcloud_mapping PUBLIC ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(pointcloud_mapping ${OpenCV_LIBS} Eigen3::Eigen ${PCL_LIBRARIES})

add_executable(surfel_mapping surfel_mapping.cpp)
target_include_directories(surfel_mapping PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(surfel_mapping ${PCL_LIBRARIES})

add_executable(octomap_mapping octomap_mapping.cpp)
target_include_directories(octomap_mapping PUBLIC ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS})
target_link_libraries(octomap_mapping ${OpenCV_LIBS} Eigen3::Eigen ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})