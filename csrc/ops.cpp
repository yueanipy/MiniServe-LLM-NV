#include <torch/library.h>

// 这个文件只声明算子的“公共合同”，不包含任何 CUDA 计算逻辑。
// PyTorch Dispatcher 会先根据这里的 schema 校验参数，再按设备类型选择实现。
// 将 schema 与 CUDA 实现分开，可以避免 Python、测试和未来其他 backend 各自定义
// 一套不一致的接口。
TORCH_LIBRARY(miniserve, m) {
  // 两个输入与一个输出都使用通用 Tensor；shape、dtype、device 等更细的不变量
  // 由具体 backend 在运行时检查。命名空间最终对应 torch.ops.miniserve。
  m.def("vector_add(Tensor a, Tensor b) -> Tensor");
  // RMSNorm 只规范化最后一维；weight 必须是一维且长度等于 hidden size。
  m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor");
  // 当前层的 paged K/V pool 与 POD metadata；seq_lens 已包含本轮当前 token。
  m.def(
      "paged_decode_attention(Tensor query, Tensor key_cache, Tensor value_cache, "
      "Tensor block_tables, Tensor sequence_lengths, float scale) -> Tensor");
  // GraphManager 专用 schema：shape/dtype 合同相同，metadata value 已在 capture 外校验。
  m.def(
      "paged_decode_attention_graph(Tensor query, Tensor key_cache, Tensor value_cache, "
      "Tensor block_tables, Tensor sequence_lengths, float scale) -> Tensor");
  m.def(
      "paged_kv_append_graph(Tensor(a!) key_pool, Tensor(b!) value_pool, int layer_index, "
      "Tensor key, Tensor value, Tensor slot_mapping) -> Tensor(a!)");
}
