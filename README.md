# MiniServe-LLM-NV

一个用于学习和验证的单 NVIDIA GPU 大模型推理服务。包含真正的 batched decode、paged KV / prefix cache、CUDA 算子、CUDA Graph，以及 OpenAI 风格 HTTP / SSE 接口。

这是精简源码仓库：不包含模型权重、本机模型锁、实验原始数据、运行日志、编译产物或历史交接记录。保留 9 个代表性测试文件，不上传完整实验测试集。

## 成果与实测

### 已实现的工程能力

- **真正的 GPU 批执行**：将多个请求的 decode 合成一次模型 forward，按请求 ID 回填；配合 sequence / RNG / KV 的提交边界，处理重排、取消和失败，区别于仅在调度器中选出多个请求。
- **Paged KV 与前缀复用**：按物理块统计容量、共享引用和缓存所有权，支持缓存驱逐与回收，减少重复前缀计算。
- **可运行的服务链路**：模型锁与权重校验、CUDA 算子、调度与执行、HTTP / SSE、终态和资源释放串成完整链路。
- **分层验证**：精简发布前在本机通过 131 项 CPU、24 项真实 CUDA 测试，以及新源码构建、真实 HTTP 与 SSE 请求验证。

### 批处理的服务收益

以下是项目历史阶段的同版本 serial / batched 对照：单 RTX 5060 Ti、Qwen3-1.7B FP16、eager、固定 2,340 个 KV blocks，prefix / chunk 关闭，**8 并发，每场景 3 组配对筛选**。

| 请求场景 | 串行总输出吞吐 | 批处理总输出吞吐 | 吞吐倍率 | 请求中位耗时：串行 → 批处理 |
| --- | ---: | ---: | ---: | ---: |
| 输入128、输出32 token | 42.25 token/s | **152.53 token/s** | **3.61倍（增加261%）** | **5.91 → 1.64秒（减少72%）** |
| 输入512、输出128 token | 26.55 token/s | **118.07 token/s** | **4.45倍（增加345%）** | **37.43 → 8.20秒（减少78%）** |

token/s 是所有请求合计的服务输出吞吐，不是单个用户的速度。吞吐倍率取配对比值的中位数，耗时取各次运行请求 p50 的中位数。批处理需显式启用；当前默认仍为 serial CUDA Graph，不能直接套用此表。

### 缓存、显存利用与可靠性

| 项目 | 历史实测观察 |
| --- | --- |
| 共享前缀、8并发、150秒窗口 | prefix off / on 的窗口成功请求中位数 **144 → 184，增加约28%** |
| 混合负载、8并发、120秒窗口 | prefix off / on 的窗口成功请求中位数 **176 → 248，增加约41%**；长请求首 token 等待 p50 **1.247 → 0.394秒** |
| 固定448 MiB KV预算的GPU容量测试 | 连续 / 分页方式的同时活跃请求 **4 → 10**；扣除各自常驻基线后的 PyTorch 峰值增量 **711.17 → 711.54 MiB**，显存增量基本不变，承载请求更多 |
| 显式1.7B压力配置 | 完成计划 **1,800秒**混合压力和 **53次SSE断连**，最终 **2,340/2,340 KV块释放**；受控过载 timeout 与负载发生器限制保留，不算正常成功或生产SLA |

分页的收益主要是显存利用率和容量，不等于总显存必然下降：低负载会预分配缓存池，实测新增分配反而多约304 MiB。共享前缀收益依赖输入复用，不能与批处理倍率相乘。

**结果边界：**上述性能与容量数字来自原完整工程的历史候选和特定配置，不是这个精简发布版重新执行的完整矩阵，也不是“最初版 → 最终版”的统一A/B结论。它们是有限筛选观察，不认证可靠P99、任意输入收益或通用提速。原始实验数据未随源码发布，不能据此视为可由本仓库直接审计的公开基准。Graph / sampling / C1 的负结果仍保留，项目不声称性能领先于 vLLM。

## 环境

- Linux 或 Windows + WSL2，Python 3.12。
- 已有 NVIDIA 驱动及可用的 `nvidia-smi`。
- 实测构建环境：RTX 5060 Ti、CUDA Toolkit 13.0、`sm_120`。需要 `nvcc`、`cuobjdump`、GCC/G++、Python venv；`nsys` 仅 profiling 时需要。
- 其他显卡架构需要调整构建参数并自行验证，不保证即装即用。在 WSL 中使用宿主机驱动，不额外安装 Linux 显卡驱动。

## 1. 安装和构建

在 Linux / WSL 终端中执行：

```bash
git clone https://github.com/yueanipy/MiniServe-LLM-NV.git
cd MiniServe-LLM-NV
bash scripts/bootstrap_profile_d.sh
source scripts/activate_profile_d.sh
bash scripts/build_native.sh D
python -m miniserve.toolchain.verify_binary --profile D
```

默认 CUDA 路径是 `/usr/local/cuda-13.0`，不同时先设置 `export MINISERVE_CUDA_HOME=/your/cuda/path`。依赖版本在 `requirements/profile-d.txt` 中固定。首次安装会联网下载依赖，首次构建需要一定时间。构建产物写入用户缓存及被 Git 忽略的 `evidence/`，不会进入源码提交。

## 2. 下载模型

默认推荐从 Qwen3-0.6B 开始。以下命令下载公开检查点，并在本机生成带逐文件 SHA-256 的锁；权重保存在 Hugging Face 用户缓存，不放入仓库：

```bash
mkdir -p local
python scripts/download_model.py \
  --repo-id Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --lock-path local/qwen3-0.6b.json
```

`local/` 已被忽略。锁中包含本机 snapshot 路径，不应上传，也不能直接拿别人的路径使用。运行时验证模型文件，不会静默跟随 `main`。模型本身的许可和使用条件以对应上游仓库为准。

## 3. 启动服务

```bash
source scripts/activate_profile_d.sh
miniserve-serve \
  --model-lock local/qwen3-0.6b.json \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 64 --num-kv-blocks 8
```

在另一个终端测试：

```bash
curl -f http://127.0.0.1:8000/health/ready
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"你好"}],"max_tokens":8,"temperature":0}'
```

流式输出在请求中增加 `"stream":true`，并使用 `curl -N`。按 Ctrl+C 退出服务。默认上下文较小，输入 token 数加 `max_tokens` 必须不超过 `max_model_len`；扩大上下文或 KV 池前先检查显存。

默认是 serial CUDA Graph、reference sampling、prefix cache 开启、整段 prefill。真实 batched decode 需显式设置 `--decode-execution eager --decode-batching batched`。请先用 `miniserve-serve --help` 查看参数；不要将显式实验配置的结论当作默认配置结果。

## 4. 如何验证

不需要模型和 GPU 的逻辑测试：

```bash
python -m compileall -q miniserve scripts
CUDA_VISIBLE_DEVICES= python -m pytest tests/unit tests/correctness -m unit_cpu -q
```

真实 CUDA 算子测试，需要先完成构建：

```bash
python -m pytest tests/cuda_extension -m cuda_extension -q
```

最后执行上面的真实 HTTP 请求，检查健康状态、返回文本及 `usage`。CPU 通过不等于 GPU 或真实模型已通过；CUDA 被跳过也不算通过。本仓库的精简测试不替代原完整项目验收。

发布前本机已完成：精简 CPU 测试 131 项、真实 CUDA 测试 24 项、从本源码构建新 wheel，以及真实服务 HTTP 200 / SSE `[DONE]` 验证。这些是该发布环境的结果，不保证所有机器都得到相同结论。本仓库不附带云端 CI；上述验证在本机执行，不需要额外 GitHub 工作流权限。

## 目录

```text
miniserve/    推理、调度、KV、模型、HTTP 和 benchmark 源码
csrc/        CUDA / C++ 算子
scripts/     环境、构建、模型下载
requirements/ 固定依赖
tests/       精简 CPU 与 CUDA 检查
```

## 支持与限制

- 已测模型主线包括 Qwen3-0.6B 和 SmolLM2；Qwen3-1.7B 是显式实验目标，不保证任意模型适配。
- 这是学习与系统工程项目，不是生产服务，没有鉴权、TLS、多 GPU 或量化保证。默认只监听本机，请勿直接暴露到公网。
- 固定 Qwen3 输入仍有一个完整词表元素超过原数值容差；显式 chunked prefill 可能在 FP16 近并列 token 上与 HF 分叉，短输入的 HF recompute/cache 也存在差异。不承诺通用 HF 等价、相同随机分布或相同 seed 跨引擎输出。
- Graph / sampling / 可选 FlashInfer C1 的历史实验不能作为通用提速证明。C1 为默认关闭的失败研究路径，不作为部署推荐；本精简仓库不配置其可选依赖。
- 不声称比 vLLM 快。不同模型、并发、上下文和执行方式必须分别实测，不能把历史筛选数字作为这个发布版的普遍性能保证。

## 来源与许可

从本地已验收候选 `709caad8f1acd5925a06d412fc13d438058c3877` 导出运行时源码，单独建立干净 Git 历史；本发布仅精简材料、环境检查和说明，不宣称新增算法或新的完整验收。
项目使用 PyTorch、Transformers、Hugging Face 模型及可选第三方组件；其能力和许可归各上游所有。开发和验证使用了 AI coding agent 辅助，不等同于所有实现均由个人独立编写。

源码许可：Apache-2.0，见 [LICENSE](LICENSE)。
