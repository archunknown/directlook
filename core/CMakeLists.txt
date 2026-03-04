find_package(OpenCV REQUIRED)

# Vinculación explícita al directorio extraído de ONNX Runtime
set(ONNXRUNTIME_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/models/onnxruntime-linux-x64-1.16.3/include)
set(ONNXRUNTIME_LIB_DIR ${CMAKE_SOURCE_DIR}/models/onnxruntime-linux-x64-1.16.3/lib)

add_executable(directlook-daemon src/daemon.cpp)

target_include_directories(directlook-daemon PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_directories(directlook-daemon PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(directlook-daemon PRIVATE ${OpenCV_LIBS} onnxruntime)