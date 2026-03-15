cmake_minimum_required(VERSION 3.10)
project(DirectLookClient)

if(WIN32)
    # Configuración pura para Windows (Win32 API)
    add_executable(directlook-client WIN32 src/tray.cpp)
else()
    # Configuración original para Linux (GTK3)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0 ayatana-appindicator3-0.1 libnotify)

    add_executable(directlook-client src/tray.cpp)

    target_include_directories(directlook-client PRIVATE ${GTK3_INCLUDE_DIRS})
    target_link_libraries(directlook-client PRIVATE ${GTK3_LIBRARIES})
endif()