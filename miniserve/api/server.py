"""Phase 10 最小 OpenAI-compatible FastAPI 服务与 localhost CLI。"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from miniserve.engine.events import EngineEventKind
from miniserve.engine.state import RequestStatus

from .adapter import to_engine_request
from .runtime import ServiceRuntime
from .schemas import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    StreamChoice,
    StreamDelta,
)
from .streaming import IncrementalDetokenizer, SlowConsumerError


class ApiError(RuntimeError):
    def __init__(self, status_code: int, message: str, *, error_type: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


def _error_response(status_code: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )


def _finish_reason(reason: str | None) -> str:
    if reason in {"stop", "length"}:
        return reason
    raise ApiError(500, f"engine request did not finish normally: {reason}", error_type="server_error")


async def _prepare_request(runtime: ServiceRuntime, request: ChatCompletionRequest):
    if not runtime.ready:
        raise ApiError(503, "model worker is not ready", error_type="service_unavailable")
    if request.model != runtime.model_id:
        raise ApiError(404, f"model is not served: {request.model}", error_type="model_not_found")
    if not await runtime.try_admit():
        raise ApiError(429, "max_pending_requests capacity reached", error_type="rate_limit_error")
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    arrival_ns = time.monotonic_ns()
    try:
        engine_request = await asyncio.to_thread(
            to_engine_request,
            request,
            runtime.tokenizer,
            request_id=request_id,
            arrival_ns=arrival_ns,
            max_model_len=runtime.max_model_len,
            vocab_size=runtime.vocab_size,
            additional_stop_token_ids=runtime.stop_token_ids,
            default_temperature=runtime.default_temperature,
            default_top_p=runtime.default_top_p,
            default_top_k=runtime.default_top_k,
        )
        runtime.validate_sampling_scope(engine_request)
    except ValueError as exc:
        await runtime.release_admission()
        # Prompt/token/sampling/context contracts are client-controlled input errors.
        # Convert them to the same OpenAI-style 400 envelope as Pydantic validation.
        raise ApiError(400, str(exc), error_type="invalid_request_error") from exc
    except Exception:
        await runtime.release_admission()
        raise
    channel = None
    submitted = False
    try:
        channel = runtime.bridge.register(request_id)  # 合同：先注册结果通道。
        runtime.engine.submit(engine_request)  # 然后才能向 Worker 提交。
        submitted = True
        # submit 的 callback 只会排入当前 event loop；本协程返回前先创建时间线。
        runtime.metrics.request_started(
            request_id,
            arrival_ns,
            prompt_tokens=len(engine_request.prompt_token_ids),
        )
    except queue.Full as exc:
        if channel is not None:
            channel.close()
        await runtime.release_admission()
        raise ApiError(429, "engine command queue is full", error_type="rate_limit_error") from exc
    except RuntimeError as exc:
        if channel is not None:
            channel.close(cancel_reason="request_setup_failed" if submitted else None)
        await runtime.release_admission()
        if not runtime.ready:
            raise ApiError(503, "model worker is not ready", error_type="service_unavailable") from exc
        raise
    except Exception:
        if channel is not None:
            channel.close(cancel_reason="request_setup_failed" if submitted else None)
        await runtime.release_admission()
        raise
    assert channel is not None
    return request_id, engine_request, channel


def create_app(runtime: ServiceRuntime, *, max_body_bytes: int = 1_048_576) -> FastAPI:
    if max_body_bytes <= 0:
        raise ValueError("max_body_bytes must be positive.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.shutdown()

    app = FastAPI(title="MiniServe-LLM-NV", version="1.0", lifespan=lifespan)
    app.state.runtime = runtime

    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, exc: ApiError):
        return _error_response(exc.status_code, str(exc), exc.error_type)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, exc: RequestValidationError):
        return _error_response(400, str(exc), "invalid_request_error")

    @app.middleware("http")
    async def enforce_body_limit(request: Request, call_next):
        content_length = request.headers.get("content-length")
        try:
            declared_size = None if content_length is None else int(content_length)
        except ValueError:
            return _error_response(400, "invalid content-length", "invalid_request_error")
        if declared_size is not None:
            if declared_size < 0:
                return _error_response(400, "invalid content-length", "invalid_request_error")
            if declared_size > max_body_bytes:
                return _error_response(413, "request body is too large", "invalid_request_error")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_body_bytes:
                return _error_response(413, "request body is too large", "invalid_request_error")
            body.extend(chunk)
        # Starlette 下游解析需要再次读取 body；缓存已受上面的逐 chunk 上限保护。
        request._body = bytes(body)
        return await call_next(request)

    @app.get("/health/live")
    async def health_live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def health_ready():
        fatal = runtime.fatal_error
        if fatal is not None:
            # fatal 分类对外可见：坏 context 不是普通未就绪，必须显式 fail-stop。
            return JSONResponse(
                status_code=503,
                content={
                    "status": "fatal",
                    "error_type": type(fatal).__name__,
                    "message": str(fatal)[:500],
                },
            )
        if not runtime.ready:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready", "model": runtime.model_id}

    @app.get("/metrics")
    async def service_metrics():
        # Prometheus 文本只含固定标签和累计值，不导出 request_id/prompt 等高基数数据。
        return Response(
            content=runtime.metrics.render_prometheus(
                pending_requests=runtime.pending_requests,
                ready=runtime.ready,
            ),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(api_request: ChatCompletionRequest, raw_request: Request):
        request_id, engine_request, channel = await _prepare_request(runtime, api_request)
        created = int(time.time())
        detokenizer = IncrementalDetokenizer(runtime.tokenizer)

        if not api_request.stream:
            terminal = None
            completed = False
            abort_reason = "response_aborted"
            try:
                async for event in channel:
                    runtime.metrics.observe_event(event)
                    if event.kind is EngineEventKind.TOKEN:
                        detokenizer.push(event.token_id)
                    else:
                        terminal = event
                completed = True
            except SlowConsumerError as exc:
                abort_reason = "slow_consumer"
                raise ApiError(429, str(exc), error_type="slow_consumer") from exc
            finally:
                if not completed:
                    runtime.metrics.request_aborted(
                        request_id, time.monotonic_ns(), abort_reason
                    )
                channel.close(cancel_reason=None if completed else "response_aborted")
                await runtime.release_admission()
            assert terminal is not None
            if terminal.status is not RequestStatus.FINISHED:
                raise ApiError(500, terminal.reason or "engine failed", error_type="server_error")
            content = runtime.tokenizer.decode(
                detokenizer.token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            completion_tokens = len(detokenizer.token_ids)
            return ChatCompletionResponse(
                id=request_id,
                created=created,
                model=runtime.model_id,
                choices=[
                    ChatCompletionChoice(
                        message=AssistantMessage(content=content),
                        finish_reason=_finish_reason(terminal.reason),
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=len(engine_request.prompt_token_ids),
                    completion_tokens=completion_tokens,
                    total_tokens=len(engine_request.prompt_token_ids) + completion_tokens,
                ),
            )

        async def generate_sse():
            completed = False
            abort_reason = "client_disconnect"
            try:
                first = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=runtime.model_id,
                    choices=[StreamChoice(delta=StreamDelta(role="assistant"))],
                )
                yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"
                async for event in channel:
                    runtime.metrics.observe_event(event)
                    if await raw_request.is_disconnected():
                        return
                    if event.kind is EngineEventKind.TOKEN:
                        delta = detokenizer.push(event.token_id)
                        if delta:
                            chunk = ChatCompletionChunk(
                                id=request_id,
                                created=created,
                                model=runtime.model_id,
                                choices=[StreamChoice(delta=StreamDelta(content=delta))],
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    else:
                        if event.status is not RequestStatus.FINISHED:
                            payload = {"error": {"message": event.reason or "engine failed", "type": "server_error"}}
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        else:
                            tail = detokenizer.finish()
                            if tail:
                                chunk = ChatCompletionChunk(
                                    id=request_id,
                                    created=created,
                                    model=runtime.model_id,
                                    choices=[StreamChoice(delta=StreamDelta(content=tail))],
                                )
                                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                            final = ChatCompletionChunk(
                                id=request_id,
                                created=created,
                                model=runtime.model_id,
                                choices=[
                                    StreamChoice(
                                        delta=StreamDelta(),
                                        finish_reason=_finish_reason(event.reason),
                                    )
                                ],
                                usage=CompletionUsage(
                                    prompt_tokens=len(engine_request.prompt_token_ids),
                                    completion_tokens=len(detokenizer.token_ids),
                                    total_tokens=(
                                        len(engine_request.prompt_token_ids)
                                        + len(detokenizer.token_ids)
                                    ),
                                ),
                            )
                            yield f"data: {final.model_dump_json(exclude_none=True)}\n\n"
                        completed = True
                        yield "data: [DONE]\n\n"
            except SlowConsumerError as exc:
                abort_reason = "slow_consumer"
                payload = {"error": {"message": str(exc), "type": "slow_consumer"}}
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if not completed:
                    runtime.metrics.request_aborted(
                        request_id, time.monotonic_ns(), abort_reason
                    )
                channel.close(cancel_reason=None if completed else "client_disconnect")
                await runtime.release_admission()

        return StreamingResponse(generate_sse(), media_type="text/event-stream")

    return app


def _resolve_cli_model_lock(model: str | None, model_lock: str | None) -> str:
    """把文档中的 --model 入口映射到仓库内经过哈希校验的 lock。"""

    if model is not None and model_lock is not None:
        raise ValueError("Use either --model or --model-lock, not both.")
    known = {
        "Qwen/Qwen3-0.6B": "model-locks/qwen3-0.6b-profile-l.json",
        "HuggingFaceTB/SmolLM2-360M-Instruct": (
            "model-locks/smollm2-360m-profile-l.json"
        ),
    }
    if model is None:
        return model_lock or known["Qwen/Qwen3-0.6B"]
    try:
        return known[model]
    except KeyError as exc:
        raise ValueError(
            f"Unknown --model {model!r}; use --model-lock for an explicitly reviewed lock."
        ) from exc


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser so prefill defaults are directly testable."""

    parser = argparse.ArgumentParser(description="Run MiniServe Phase 10 on localhost.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model")
    model_group.add_argument("--model-lock")
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--num-kv-blocks", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-pending-requests", type=int, default=4)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Explicit whole-prefill budget; defaults to --max-model-len.",
    )
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument(
        "--attention-backend",
        choices=("cuda_paged", "flashinfer_paged", "torch_paged_ref"),
        default="cuda_paged",
    )
    parser.add_argument("--rmsnorm-backend", choices=("cuda", "torch"), default="cuda")
    parser.add_argument("--decode-execution", choices=("cuda_graph", "eager"), default="cuda_graph")
    parser.add_argument("--decode-batching", choices=("serial", "batched"), default="serial")
    parser.add_argument("--batched-graph", action="store_true")
    parser.add_argument("--sampling-execution", choices=("reference", "batched_greedy"), default="reference")
    parser.add_argument("--no-prefix-cache", action="store_true")
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument(
        "--chunked-prefill",
        dest="enable_chunked_prefill",
        action="store_true",
        help="Explicitly enable chunked prefill (off by default).",
    )
    chunk_group.add_argument(
        "--no-chunked-prefill",
        dest="enable_chunked_prefill",
        action="store_false",
        help="Keep chunked prefill disabled (the default).",
    )
    parser.set_defaults(enable_chunked_prefill=False)
    parser.add_argument("--prefill-chunk-size", type=int, default=None)
    parser.add_argument("--allow-remote-unauthenticated", action="store_true")
    return parser


def main() -> None:
    from miniserve.observability.logging import configure_logging

    configure_logging()
    parser = build_cli_parser()
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote_unauthenticated:
        raise SystemExit("Refusing unauthenticated remote bind; pass --allow-remote-unauthenticated explicitly.")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("WARNING: binding an unauthenticated LLM service to a remote interface.")
    from .bootstrap import build_profile_l_runtime

    import uvicorn

    try:
        selected_lock = _resolve_cli_model_lock(args.model, args.model_lock)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        service = build_profile_l_runtime(
            selected_lock,
            max_model_len=args.max_model_len,
            num_kv_blocks=args.num_kv_blocks,
            max_num_seqs=args.max_num_seqs,
            max_pending_requests=args.max_pending_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            kv_block_size=args.kv_block_size,
            attention_backend=args.attention_backend,
            rmsnorm_backend=args.rmsnorm_backend,
            decode_execution=args.decode_execution,
            decode_batching=args.decode_batching,
            batched_graph=args.batched_graph,
            sampling_execution=args.sampling_execution,
            enable_prefix_cache=not args.no_prefix_cache,
            enable_chunked_prefill=args.enable_chunked_prefill,
            prefill_chunk_size=args.prefill_chunk_size,
        )
    except ValueError as exc:
        parser.error(str(exc))
    uvicorn.run(create_app(service), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
