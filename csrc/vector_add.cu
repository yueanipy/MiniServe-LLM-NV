#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace {

// 256 是易于理解的教学基线：它是 warp 大小 32 的整数倍，同时不会让单个 block
// 过大。这里暂不把 block size 当作性能调优结论。
constexpr int kThreadsPerBlock = 256;

// 每个 CUDA thread 只负责一个线性元素。__restrict__ 告诉编译器三个指针互不别名，
// 但正确性不能依赖这一优化提示；真正的输入合同仍由 host wrapper 检查。
template <typename scalar_t>
__global__ void vector_add_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ output,
    int64_t numel) {
  // Grid 中每个 thread 的全局线性编号：block 起点 + block 内 thread 偏移。
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  // blocks 使用向上取整，最后一个 block 往往有多余 thread。没有这个判断时，
  // size=257 等非 256 对齐输入会越界读写 GPU 内存。
  if (index < numel) {
    output[index] = a[index] + b[index];
  }
}

// 这是 Dispatcher 在 CUDA tensor 上调用的 host wrapper。它负责验证合同、选择
// 正确设备与 stream、分派 dtype 并启动 kernel；kernel 本身只做最小数据面计算。
at::Tensor vector_add_cuda(const at::Tensor& a, const at::Tensor& b) {
  // 禁止 CPU fallback：Phase 0 的目的就是证明真实 CUDA 路径可用。
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "vector_add expects CUDA tensors");

  // 多 GPU 环境里，两个输入必须位于同一设备，否则指针不能由同一 kernel 安全访问。
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same CUDA device");

  // 本基线不实现 broadcasting、类型提升或非连续 stride；这些都必须显式拒绝，
  // 不能让调用方误以为行为等同于完整的 torch.add。
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have identical shapes");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have identical dtypes");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a and b must be contiguous");
  TORCH_CHECK(
      a.is_floating_point(),
      "vector_add supports floating-point tensors only");

  // CUDAGuard 临时切换到输入所在设备，并在函数返回时恢复调用者原来的设备。
  // 这比依赖“当前设备碰巧正确”可靠，也为未来多 GPU 测试保留正确语义。
  c10::cuda::CUDAGuard device_guard(a.device());

  // empty_like 同时继承 shape、dtype、layout 和 device；输出尚未初始化，随后由
  // kernel 完整写入。零元素 tensor 没有合法 launch，因此直接返回空输出。
  at::Tensor output = at::empty_like(a);
  const int64_t numel = a.numel();
  if (numel == 0) {
    return output;
  }

  // 整数向上取整：(N + T - 1) / T。尾部多出的 thread 由 kernel 边界判断拦截。
  const int blocks = static_cast<int>(
      (numel + kThreadsPerBlock - 1) / kThreadsPerBlock);

  // 必须使用 PyTorch 当前 CUDA stream，而不是默认 stream。这样自定义算子才能和
  // 调用者前后的异步 PyTorch 操作保持正确依赖，不需要额外全局同步。
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.device().index());

  // 根据运行时 dtype 实例化 scalar_t。当前显式覆盖 float/double/half/bfloat16；
  // Profile L 测试重点是 FP32/FP16，BF16 不因硬件“似乎支持”就自动成为正式路径。
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      a.scalar_type(),
      "miniserve_vector_add_cuda",
      [&] {
        vector_add_kernel<scalar_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
      });

  // kernel launch 是异步的；这里检查 launch 配置/立即错误，但不做 device-wide
  // synchronize。数值测试或调用者在真正读取结果时再建立同步边界。
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

// 把同一个 canonical schema 的 CUDA dispatch key 绑定到上面的 host wrapper。
// 没有注册 CPU 实现是有意设计：CPU 输入应明确失败，而不是静默换路径。
TORCH_LIBRARY_IMPL(miniserve, CUDA, m) {
  m.impl("vector_add", &vector_add_cuda);
}
