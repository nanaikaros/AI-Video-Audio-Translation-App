cmake_minimum_required(VERSION 3.14)
project(AIVideoTranslationApp)
# cmake --preset default -DWITH_STATIC_LIB=ON -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>"
# cmake --build build --config Release
if(MSVC)
  # 优先使用新接口（CMake >= 3.15），并强制写入缓存
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)

  # 兼容老 CMake：把已存在的 /MD(/MDd) 标志替换为 /MT(/MTd)
  foreach(var CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_C_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    string(REPLACE "/MDd" "/MTd" ${var} "${${var}}")
    string(REPLACE "/MD"  "/MT"  ${var} "${${var}}")
    set(${var} "${${var}}" CACHE STRING "" FORCE)
  endforeach()

  add_compile_options(
    "$<$<CONFIG:Debug>:/MTd>"
    "$<$<CONFIG:Release>:/MT>"
  )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_SYSTEM_DEPS "Use local ffmpeg llama whisper" OFF)  
option(USE_SYSTEM_GGML "Use system-installed GGML library" OFF)

set(GGML_SRC_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/ggml)
set(WHISPER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/whisper)
set(LLAMA_SRC_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/src/llama)
set(PADDLE_SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src/paddle)

set(WHISPER_USE_SYSTEM_GGML ON CACHE BOOL "" FORCE)
set(LLAMA_USE_SYSTEM_GGML   ON CACHE BOOL "" FORCE)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)

#
# build the ggml library
#
add_subdirectory(${GGML_SRC_DIR} ggml_build)
if(WIN32)
    # The following adds a _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR macro and is a workaround for
    # the Windows C++ standard library which does not support constexpr mutexes.
    # From the release notes://github.com/microsoft/STL/wiki/Changelog
    #  Disable constexpr mutex constructor on Windows
    #  Fixed mutex's constructor to be constexpr. #3824 #4000 #4339
    #  Note: Programs that aren't following the documented restrictions on binary compatibility may encounter
    #  null dereferences in mutex machinery. You must follow this rule:
    #  When you mix binaries built by different supported versions of the toolset, the Redistributable version
    #  must be at least as new as the latest toolset used by any app component.
    #  You can define _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR as an escape hatch.
    #
    # Specifically to whisper.cpp this would cause a crash when using the Java bindings.
    # resulting in a Invalid memory access error.
    target_compile_definitions(ggml-base PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/paddle)


if(APPLE)
  set(PADDLE_LIB "${CMAKE_CURRENT_SOURCE_DIR}/src/paddle/paddle_inference/paddle_inference_install_dir")
else()
  set(PADDLE_LIB "${CMAKE_CURRENT_SOURCE_DIR}/src/paddle/paddle_inference")
endif()

link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/third_party/install/yaml-cpp/lib")
link_directories("${PADDLE_LIB}/paddle/lib")

add_subdirectory(${PADDLE_SRC_DIR} paddle_build) 
add_subdirectory(${WHISPER_SRC_DIR} whisper_build)
add_subdirectory(${LLAMA_SRC_DIR} llama_build)


add_executable(my_app
    src/translation/translation.cpp
    src/video/video.cpp
    src/video/mp4.cpp
    src/tool/fileloader.hpp
    src/whisper/whisper_bridge.cpp
    src/paddle/ocr_main.cpp
    common/progress_ipc.cpp
    main.cpp
)

if(WIN32)
  # set_target_properties(my_app PROPERTIES
  #   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  #   RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
  #   RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
  #   RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo"
  #   RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin/MinSizeRel"
  # )
  set_target_properties(my_app PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set_target_properties(whisper PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set_target_properties(llama   PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(my_app PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(my_app PRIVATE 
  llama
  whisper
  paddle
  glog
  gflags
  xxhash
  yaml-cpp
  paddle_inference
  phi
  phi_core
  common
)

# vcpkg
find_package(FFMPEG REQUIRED)
target_include_directories(my_app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${FFMPEG_INCLUDE_DIRS}
)

if(WIN32)
  target_link_libraries(my_app PRIVATE Ws2_32 ${FFMPEG_LIBRARIES} gflags_static)
elseif(APPLE)
  target_link_libraries(my_app PRIVATE ${FFMPEG_LIBRARIES})
endif()