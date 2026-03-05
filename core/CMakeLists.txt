find_package(OpenCV REQUIRED)

# --- ONNX Runtime: rutas condicionales por plataforma ---
if(WIN32)
    set(ONNXRUNTIME_ROOT ${CMAKE_SOURCE_DIR}/models/onnxruntime-win-x64-1.24.2)
else()
    set(ONNXRUNTIME_ROOT ${CMAKE_SOURCE_DIR}/models/onnxruntime-linux-x64-1.16.3)
endif()

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR     ${ONNXRUNTIME_ROOT}/lib)

add_executable(directlook-daemon src/daemon.cpp)

target_include_directories(directlook-daemon PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/ipc
)

target_link_directories(directlook-daemon PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(directlook-daemon PRIVATE ${OpenCV_LIBS} onnxruntime)

# Windows: linkear psapi.lib para GetProcessMemoryInfo
if(WIN32)
    target_link_libraries(directlook-daemon PRIVATE psapi)
endif()