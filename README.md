# MiniServe-LLM-NV

一个用于学习和验证的单 NVIDIA GPU 大模型推理服务。包含真正的 batched decode、paged KV / prefix cache、CUDA 算子、CUDA Graph，以及 OpenAI 风格 HTTP / SSE 接口。

这是精简源码仓库：不包含模型权重、本机模型锁、实验原始数据、运行日志、编译产物或历史交接记录。保留 9 个代表性测试文件，不上传完整实验测试集。

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
