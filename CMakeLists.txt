cmake_minimum_required(VERSION 3.10)
project(AIVideoTranslationApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_SYSTEM_DEPS "Use Homebrew-installed llama/whisper" ON)

set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

if (WIN32)
  # todo: win
elseif(APPLE)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(AVCODEC REQUIRED IMPORTED_TARGET libavcodec)
  pkg_check_modules(AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
  pkg_check_modules(AVUTIL REQUIRED IMPORTED_TARGET libavutil)
  pkg_check_modules(SWSCALE REQUIRED IMPORTED_TARGET libswscale)
  pkg_check_modules(SWRESAMPLE REQUIRED IMPORTED_TARGET libswresample)
endif()

add_executable(my_app
    src/translation/translation.cpp
    src/video/video.cpp
    src/whisper/whisper_bridge.cpp
    src/whisper/common/common-whisper.cpp
    src/whisper/common/common.cpp
    src/whisper/grammar-parser.cpp
    src/common/progress_ipc.cpp
    main.cpp
)

target_include_directories(my_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (USE_SYSTEM_DEPS)
  if (APPLE)
    execute_process(COMMAND brew --prefix llama.cpp OUTPUT_VARIABLE LLAMA_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND brew --prefix whisper-cpp OUTPUT_VARIABLE WHISPER_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)

    find_path(LLAMA_INCLUDE_DIR llama.h PATHS ${LLAMA_PREFIX}/include REQUIRED)
    find_path(WHISPER_INCLUDE_DIR whisper.h PATHS ${WHISPER_PREFIX}/include REQUIRED)
    find_path(GGML_INCLUDE_DIR ggml.h PATHS ${LLAMA_PREFIX}/include ${WHISPER_PREFIX}/include REQUIRED)
    
    find_library(LLAMA_LIBRARY llama PATHS ${LLAMA_PREFIX}/lib REQUIRED)
    find_library(WHISPER_LIBRARY whisper PATHS ${WHISPER_PREFIX}/lib REQUIRED)
    
    find_library(GGML_LIBRARY ggml PATHS ${LLAMA_PREFIX}/lib ${WHISPER_PREFIX}/lib)
    find_library(GGML_BASE_LIBRARY ggml-base PATHS ${LLAMA_PREFIX}/lib ${WHISPER_PREFIX}/lib)
    find_library(GGML_CPU_LIBRARY ggml-cpu PATHS ${LLAMA_PREFIX}/lib ${WHISPER_PREFIX}/lib)
    find_library(GGML_BLAS_LIBRARY ggml-blas PATHS ${LLAMA_PREFIX}/lib ${WHISPER_PREFIX}/lib)
    find_library(GGML_METAL_LIBRARY ggml-metal PATHS ${LLAMA_PREFIX}/lib ${WHISPER_PREFIX}/lib)

    target_include_directories(my_app PRIVATE
      ${LLAMA_INCLUDE_DIR}
      ${WHISPER_INCLUDE_DIR}
      ${GGML_INCLUDE_DIR}
    )

    target_link_libraries(my_app PRIVATE
      ${LLAMA_LIBRARY}
      ${WHISPER_LIBRARY}
      ${GGML_LIBRARY}
      ${GGML_BASE_LIBRARY}
      ${GGML_CPU_LIBRARY}
      ${GGML_BLAS_LIBRARY}
      ${GGML_METAL_LIBRARY}
    )
  elseif(WIN32)
    message(FATAL_ERROR "System dependencies are not supported on Windows yet.")
  endif()

else()
  add_subdirectory(${THIRD_PARTY_DIR}/llama.cpp llama_cpp_build EXCLUDE_FROM_ALL)
  add_subdirectory(${THIRD_PARTY_DIR}/whisper.cpp whisper_build EXCLUDE_FROM_ALL)

  target_link_libraries(my_app PRIVATE
    llama
    whisper
  )
endif()

if(WIN32)
  target_link_libraries(my_app PRIVATE
    Ws2_32
    ${AVCODEC_LIBRARY}
    ${AVFORMAT_LIBRARY}
    ${AVUTIL_LIBRARY}
    ${SWSCALE_LIBRARY}
    ${SWRESAMPLE_LIBRARY}
  )
elseif(APPLE)
  target_link_libraries(my_app PRIVATE
    PkgConfig::AVCODEC
    PkgConfig::AVFORMAT
    PkgConfig::AVUTIL
    PkgConfig::SWSCALE
    PkgConfig::SWRESAMPLE
  )
endif()