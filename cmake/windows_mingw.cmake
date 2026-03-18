set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_VERSION 1)

# 指定交叉编译器（Homebrew 安装的 mingw-w64）
set(CMAKE_C_COMPILER   x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_RC_COMPILER  x86_64-w64-mingw32-windres)

# 查找路径：优先在源码的 third_party 下寻找 include/lib
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SOURCE_DIR}/third_party)

# 查找策略：程序用主机工具，库/头文件从 find_root_path 中找
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)