add_executable(eigenMatrix eigenMatrix.cpp)
target_link_libraries(eigenMatrix Eigen3::Eigen)

add_executable(eigenGeometry eigenGeometry.cpp)
target_link_libraries(eigenGeometry Eigen3::Eigen)

add_executable(coordinateTransform coordinateTransform.cpp)
target_link_libraries(coordinateTransform Eigen3::Eigen)

add_executable(plotTrajectory plotTrajectory.cpp)
target_link_libraries(plotTrajectory Eigen3::Eigen ${Pangolin_LIBRARIES})
configure_file(trajectory.txt ${CMAKE_CURRENT_BINARY_DIR}/trajectory.png COPYONLY)

add_executable(visualizeGeometry visualizeGeometry.cpp)
target_link_libraries(visualizeGeometry Eigen3::Eigen ${Pangolin_LIBRARIES})