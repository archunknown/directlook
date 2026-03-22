cmake_minimum_required(VERSION 3.10)
project(DirectLookDaemon)

find_package(OpenCV REQUIRED)

# --- ONNX Runtime: rutas condicionales por plataforma ---
if(WIN32)
    set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../models/onnxruntime-win-x64-1.24.2)
else()
    set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../models/onnxruntime-linux-x64-1.16.3)
endif()

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR     ${ONNXRUNTIME_ROOT}/lib)

if(WIN32)
    set(IPC_SOURCES src/ipc_windows.cpp)
    set(SINK_SOURCES src/video_sink_windows.cpp)
else()
    set(IPC_SOURCES src/ipc_unix.cpp)
    set(SINK_SOURCES src/video_sink_unix.cpp)
endif()

add_executable(directlook-daemon src/daemon.cpp src/vision_pipeline.cpp src/cpu_monitor.cpp ${IPC_SOURCES} ${SINK_SOURCES})

target_include_directories(directlook-daemon PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../ipc
)

target_link_directories(directlook-daemon PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(directlook-daemon PRIVATE ${OpenCV_LIBS} onnxruntime)

# Copiar modelos al directorio de salida
add_custom_command(TARGET directlook-daemon POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/../modelos
    ${CMAKE_CURRENT_BINARY_DIR}/modelos
)

# Windows: linkear psapi.lib para GetProcessMemoryInfo
if(WIN32)
    target_link_libraries(directlook-daemon PRIVATE psapi)
endif()