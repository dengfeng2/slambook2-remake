add_executable(imageBasics imageBasics.cpp)
target_include_directories(imageBasics PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(imageBasics ${OpenCV_LIBS})
configure_file(ubuntu.png ${CMAKE_CURRENT_BINARY_DIR}/ubuntu.png COPYONLY)

add_executable(undistortedImage undistortedImage.cpp)
target_include_directories(undistortedImage PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(undistortedImage ${OpenCV_LIBS})
configure_file(distorted.png ${CMAKE_CURRENT_BINARY_DIR}/distorted.png COPYONLY)