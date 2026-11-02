cmake_minimum_required(VERSION 3.26...4.4)
project(miniserve_native LANGUAGES CXX CUDA)

# 项目的生产目标是 Linux 原生扩展；Windows 只作为 WSL 宿主，不生成替代性的 DLL。
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "MiniServe V1 native extensions must be built in Linux/WSL2, not Windows.")
endif()

# 架构必须由 Profile 显式传入。省略时让 CMake 自动猜测会降低构建可复现性，
# 也可能只留下 PTX 而没有目标机器需要的原生 cubin。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES explicitly: 75 for Profile L or 120 for Profile D.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 必须从“正在构建 wheel 的同一个 Python”询问 Torch CMake 路径，防止系统 Python
# 与 venv 各装一份 torch 后发生头文件、ABI 或运行时库错配。
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  RESULT_VARIABLE TORCH_PREFIX_RESULT
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_PREFIX_RESULT EQUAL 0)
  message(FATAL_ERROR "The build interpreter cannot import torch.")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
# TORCH_CXX_FLAGS 包含当前 wheel 的 ABI 约束，不能靠手写编译选项猜测。
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ops.cpp 注册公共 schema，vector_add.cu 注册 CUDA 实现；它们共同链接成一个
# 由 torch.ops.load_library 动态加载的共享库，而不是 Python CPython extension。
add_library(miniserve_ops SHARED
  csrc/ops.cpp
  csrc/vector_add.cu
  csrc/rmsnorm.cu
  csrc/paged_decode_attention.cu
  csrc/paged_kv_append.cu
)
target_link_libraries(miniserve_ops PRIVATE ${TORCH_LIBRARIES})
target_compile_options(miniserve_ops PRIVATE
  # PyTorch/CUDA 头文件中会使用扩展 constexpr 语义；只把该选项传给 NVCC。
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
set_property(TARGET miniserve_ops PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(miniserve_ops PROPERTIES
  # 去掉 Linux 默认的 lib 前缀，使严格加载器能匹配 _miniserve_ops*.so。
  PREFIX ""
  OUTPUT_NAME "_miniserve_ops"
  # The installed extension lives in site-packages/miniserve while the
  # PyTorch shared libraries live in the sibling site-packages/torch/lib.
  # Use a relative RUNPATH so ldd and a clean Python process can resolve the
  # dependencies without hard-coding this machine's virtual-environment path.
  INSTALL_RPATH "$ORIGIN/../torch/lib"
)

install(
  # wheel 中的目标目录必须与 Python 包 miniserve 相同，加载器才能从包目录或
  # distribution metadata 唯一定位共享库。
  TARGETS miniserve_ops
  LIBRARY DESTINATION miniserve
  RUNTIME DESTINATION miniserve
  ARCHIVE DESTINATION miniserve
)
