cmake_minimum_required(VERSION 3.10)
project(ffmpegApp)

set(CMAKE_CXX_STANDARD 17)

# --- 新增：静态优先开关 ---
option(APP_PREFER_STATIC "Prefer static third-party libs" ON)

if (APP_PREFER_STATIC)
    # 对子项目（llama.cpp / whisper.cpp）优先静态
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    # 某些版本会用这些变量，统一关掉共享库
    set(LLAMA_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)

    # 查找库时优先 .a
    set(_OLD_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

# 设置第三方库路径
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(FFMPEG_DIR ${THIRD_PARTY_DIR}/ffmpeg)

# 查找 FFmpeg 组件（会优先找 .a）
find_path(FFMPEG_INCLUDE_DIR libavcodec/avcodec.h PATHS ${FFMPEG_DIR}/include)
find_library(AVCODEC_LIBRARY avcodec PATHS ${FFMPEG_DIR}/lib)
find_library(AVFORMAT_LIBRARY avformat PATHS ${FFMPEG_DIR}/lib)
find_library(AVUTIL_LIBRARY avutil PATHS ${FFMPEG_DIR}/lib)
find_library(SWSCALE_LIBRARY swscale PATHS ${FFMPEG_DIR}/lib)
find_library(SWRESAMPLE_LIBRARY swresample PATHS ${FFMPEG_DIR}/lib)

if (APP_PREFER_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_OLD_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

# 加载 llama.cpp / whisper.cpp
add_subdirectory(${THIRD_PARTY_DIR}/llama.cpp llama_cpp_build EXCLUDE_FROM_ALL)
add_subdirectory(${THIRD_PARTY_DIR}/whisper.cpp whisper_build EXCLUDE_FROM_ALL)

add_executable(my_app
    ./translation/translation.cpp
    ./video/video.cpp
    ./whisper/whisper_bridge.cpp
    ./whisper/common/common-whisper.cpp
    ./whisper/common/common.cpp
    ./whisper/grammar-parser.cpp
    ./common/progress_ipc.cpp
    main.cpp
)

target_include_directories(my_app PRIVATE
    ${FFMPEG_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/spdlog
    ${THIRD_PARTY_DIR}/whisper.cpp/include
)

target_link_libraries(my_app
    ${AVCODEC_LIBRARY}
    ${AVFORMAT_LIBRARY}
    ${AVUTIL_LIBRARY}
    ${SWSCALE_LIBRARY}
    ${SWRESAMPLE_LIBRARY}
    llama
    whisper
)

# 可选：减小体积
if (APPLE)
    target_link_options(my_app PRIVATE -Wl,-dead_strip)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)