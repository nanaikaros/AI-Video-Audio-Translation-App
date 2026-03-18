cmake_minimum_required(VERSION 3.10)
project(AIVideoTranslationApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_SYSTEM_DEPS "Use local ffmpeg llama whisper" OFF)  
option(USE_SYSTEM_GGML "Use system-installed GGML library" OFF)

set(GGML_SRC_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/ggml)
set(WHISPER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/whisper)
set(LLAMA_SRC_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/src/llama)

set(WHISPER_USE_SYSTEM_GGML ON CACHE BOOL "" FORCE)
set(LLAMA_USE_SYSTEM_GGML   ON CACHE BOOL "" FORCE)

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


add_subdirectory(${WHISPER_SRC_DIR} whisper_build)
add_subdirectory(${LLAMA_SRC_DIR} llama_build)

add_executable(my_app
    src/translation/translation.cpp
    src/video/video.cpp
    src/whisper/whisper_bridge.cpp
    common/progress_ipc.cpp
    main.cpp
)

target_include_directories(my_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(WIN32)
  set_target_properties(my_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin/MinSizeRel"
  )
endif()

target_link_libraries(my_app PRIVATE 
  llama
  whisper
)

# vcpkg
find_package(FFMPEG REQUIRED)
target_include_directories(my_app PRIVATE ${FFMPEG_INCLUDE_DIRS})
if(WIN32)
  target_link_libraries(my_app PRIVATE Ws2_32 ${FFMPEG_LIBRARIES})
elseif(APPLE)
  target_link_libraries(my_app PRIVATE ${FFMPEG_LIBRARIES})
endif()