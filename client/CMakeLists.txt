find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

add_executable(directlook-client src/tray.cpp)

target_include_directories(directlook-client PRIVATE ${GTK3_INCLUDE_DIRS})
target_link_libraries(directlook-client PRIVATE ${GTK3_LIBRARIES})