add_executable(feature_training feature_training.cpp)
target_include_directories(feature_training PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(feature_training ${OpenCV_LIBS} ${DBoW3_LIBS})

add_executable(loop_closure loop_closure.cpp)
target_include_directories(loop_closure PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(loop_closure ${OpenCV_LIBS} ${DBoW3_LIBS})

add_executable(gen_vocab_large gen_vocab_large.cpp)
target_include_directories(gen_vocab_large PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gen_vocab_large ${OpenCV_LIBS} ${DBoW3_LIBS})

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY vocab_larger.yml.gz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})