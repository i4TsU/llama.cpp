add_executable(qwen3vl-preprocess qwen3vl-preprocess.cpp)
target_include_directories(qwen3vl-preprocess PRIVATE ../../common ../../includes ../../tools/mtmd)
target_link_libraries(qwen3vl-preprocess PRIVATE common llama mtmd ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

add_executable(qwen3vl-inference qwen3vl-inference.cpp)
target_include_directories(qwen3vl-inference PRIVATE ../../common ../../includes)
target_link_libraries(qwen3vl-inference PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
