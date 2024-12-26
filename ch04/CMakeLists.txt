set(CMAKE_BUILD_TYPE "Debug")

add_executable(useSophus useSophus.cpp)
target_link_libraries(useSophus PUBLIC Sophus::Sophus)

add_executable(trajectoryError trajectoryError.cpp)
target_link_libraries(trajectoryError PUBLIC Sophus::Sophus ${Pangolin_LIBRARIES})
configure_file(estimated.txt ${CMAKE_CURRENT_BINARY_DIR}/estimated.txt COPYONLY)
configure_file(groundtruth.txt ${CMAKE_CURRENT_BINARY_DIR}/groundtruth.txt COPYONLY)