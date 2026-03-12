cmake_minimum_required(VERSION 3.10)
project(ffmpegApp)

set(CMAKE_CXX_STANDARD 17)

option(APP_PREFER_STATIC "Prefer static third-party libs" ON)

if (APP_PREFER_STATIC)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    set(LLAMA_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)

    set(_OLD_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

if (WIN32)
  set(FFMPEG_DIR ${THIRD_PARTY_DIR}/ffmpeg-win64)
elseif(APPLE)
  set(FFMPEG_DIR ${THIRD_PARTY_DIR}/ffmpeg-mac)
endif()

# 查找 FFmpeg 组件（会优先找 .a）
find_path(FFMPEG_INCLUDE_DIR libavcodec/avcodec.h
  PATHS ${FFMPEG_DIR}/include
  NO_DEFAULT_PATH
)
find_library(AVCODEC_LIBRARY avcodec PATHS ${FFMPEG_DIR}/lib NO_DEFAULT_PATH)
find_library(AVFORMAT_LIBRARY avformat PATHS ${FFMPEG_DIR}/lib NO_DEFAULT_PATH)
find_library(AVUTIL_LIBRARY avutil PATHS ${FFMPEG_DIR}/lib NO_DEFAULT_PATH)
find_library(SWSCALE_LIBRARY swscale PATHS ${FFMPEG_DIR}/lib NO_DEFAULT_PATH)
find_library(SWRESAMPLE_LIBRARY swresample PATHS ${FFMPEG_DIR}/lib NO_DEFAULT_PATH)

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

if(WIN32)
  target_link_libraries(my_app PRIVATE
    "-Wl,--start-group"
    ${AVFORMAT_LIBRARY}
    ${AVCODEC_LIBRARY}
    ${SWRESAMPLE_LIBRARY}
    ${SWSCALE_LIBRARY}
    ${AVUTIL_LIBRARY}
    "-Wl,--end-group"
    Ws2_32
    llama
    whisper
  )
else()
target_link_libraries(my_app
    ${AVCODEC_LIBRARY}
    ${AVFORMAT_LIBRARY}
    ${AVUTIL_LIBRARY}
    ${SWSCALE_LIBRARY}
    ${SWRESAMPLE_LIBRARY}
    llama
    whisper
)
endif()

# if (APPLE)
#     target_link_options(my_app PRIVATE -Wl,-dead_strip)
# endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)