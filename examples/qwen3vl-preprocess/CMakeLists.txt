# qwen3vl-preprocess tools
#
# These tools demonstrate pre-computing image embeddings separately from inference,
# allowing the vision encoder to be run once and the embeddings reused.

find_package(Threads REQUIRED)

# Preprocess tool - requires mtmd for vision encoding
set(TARGET llama-qwen3vl-preprocess)
add_executable(${TARGET} qwen3vl-preprocess.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-qwen3vl-preprocess)
target_link_libraries(${TARGET} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Inference tool - only needs the language model, no mtmd dependency
set(TARGET llama-qwen3vl-inference)
add_executable(${TARGET} qwen3vl-inference.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-qwen3vl-inference)
target_link_libraries(${TARGET} PRIVATE common llama Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
