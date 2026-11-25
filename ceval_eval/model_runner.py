from __future__ import annotations

import hashlib
import os
from importlib.metadata import version
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return requested


def _resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if requested not in mapping:
        raise ValueError(f"Unsupported dtype: {requested}")
    return mapping[requested]


class MockRunner:
    """Deterministic offline backend for smoke tests and CI."""

    def __init__(self) -> None:
        self.device = "mock"
        self.dtype = "none"

    def generate(
        self,
        prompts: list[str],
        system: str,
        gen_cfg: dict[str, Any],
        num_return_sequences: int = 1,
    ) -> list[list[dict[str, Any]]]:
        outputs: list[list[dict[str, Any]]] = []
        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            answer = "ABCD"[digest[0] % 4]
            sample_outputs = [
                {
                    "text": f"答案：{answer}",
                    "tokens": 4,
                    "truncated": False,
                }
                for _ in range(num_return_sequences)
            ]
            outputs.append(sample_outputs)
        return outputs

    def meta(self) -> dict[str, Any]:
        return {
            "backend": "mock",
            "model": "mock",
            "device": "mock",
            "dtype": "none",
        }


class HFModelRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        model_cfg = config.get("model", {})
        self.model_name = model_cfg.get("local_path") or model_cfg["name_or_path"]
        self.revision = model_cfg.get("revision")
        self.device = _resolve_device(model_cfg.get("device", "auto"))
        self.dtype = _resolve_dtype(model_cfg.get("torch_dtype", "auto"), self.device)
        self.max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 128))
        cache_dir = model_cfg.get("cache_dir") or os.environ.get("HF_HOME")
        local_files_only = bool(model_cfg.get("local_files_only", False))

        revision = model_cfg.get("revision")
        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        }
        model_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
            "local_files_only": local_files_only,
        }
        if revision:
            tokenizer_kwargs["revision"] = revision
            model_kwargs["revision"] = revision
        if bool(model_cfg.get("low_cpu_mem_usage", False)):
            model_kwargs["low_cpu_mem_usage"] = True
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
            model_kwargs["cache_dir"] = cache_dir
        attn_implementation = model_cfg.get("attn_implementation")
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _messages(self, prompt: str, system: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _trim_generated(self, token_ids: list[int]) -> tuple[str, int, bool]:
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        kept: list[int] = []
        for token in token_ids:
            if eos is not None and token == eos:
                break
            if pad is not None and token == pad:
                continue
            kept.append(token)
        text = self.tokenizer.decode(kept, skip_special_tokens=True).strip()
        truncated = len(kept) >= self.max_new_tokens
        return text, len(kept), truncated

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        system: str,
        gen_cfg: dict[str, Any],
        num_return_sequences: int = 1,
    ) -> list[list[dict[str, Any]]]:
        if num_return_sequences < 1:
            raise ValueError("num_return_sequences must be >= 1")
        do_sample = bool(gen_cfg.get("do_sample", False))
        if num_return_sequences > 1 and not do_sample:
            raise ValueError("num_return_sequences > 1 requires do_sample=true")

        chat_prompts = [
            self.tokenizer.apply_chat_template(
                self._messages(prompt, system),
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_len = input_ids.shape[1]

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(gen_cfg.get("temperature", 0.7))
            generation_kwargs["top_p"] = float(gen_cfg.get("top_p", 1.0))
        if gen_cfg.get("repetition_penalty") is not None:
            generation_kwargs["repetition_penalty"] = float(
                gen_cfg["repetition_penalty"]
            )

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        generated = generated[:, input_len:]

        outputs: list[list[dict[str, Any]]] = []
        for batch_index in range(len(prompts)):
            sample_outputs = []
            for sample_index in range(num_return_sequences):
                row = batch_index * num_return_sequences + sample_index
                text, token_count, truncated = self._trim_generated(
                    generated[row].tolist()
                )
                sample_outputs.append(
                    {
                        "text": text,
                        "tokens": token_count,
                        "truncated": truncated,
                    }
                )
            outputs.append(sample_outputs)
        return outputs

    def meta(self) -> dict[str, Any]:
        return {
            "backend": "transformers",
            "model": self.model_name,
            "revision": self.revision,
            "device": self.device,
            "dtype": str(self.dtype),
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }


def build_runner(config: dict[str, Any]) -> MockRunner | HFModelRunner:
    if bool(config.get("runtime", {}).get("mock", False)):
        return MockRunner()
    return HFModelRunner(config)
