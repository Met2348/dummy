"""TRT-LLM build script template (not executed in tests)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrtLlmBuildConfig:
    checkpoint_dir: str
    output_dir: str
    dtype: str = "float16"
    max_batch_size: int = 32
    max_input_len: int = 4096
    max_output_len: int = 1024
    max_num_tokens: int = 8192
    use_paged_context_fmha: bool = True
    use_fp8_context_fmha: bool = False
    use_weight_only: bool = False
    quant_format: str | None = None

    def to_cli(self) -> str:
        parts = [
            "trtllm-build",
            f"--checkpoint_dir {self.checkpoint_dir}",
            f"--output_dir {self.output_dir}",
            f"--gemm_plugin {self.dtype}",
            f"--gpt_attention_plugin {self.dtype}",
            f"--max_batch_size {self.max_batch_size}",
            f"--max_input_len {self.max_input_len}",
            f"--max_output_len {self.max_output_len}",
            f"--max_num_tokens {self.max_num_tokens}",
        ]
        if self.use_paged_context_fmha:
            parts.append("--use_paged_context_fmha enable")
        if self.use_fp8_context_fmha:
            parts.append("--use_fp8_context_fmha enable")
        if self.use_weight_only:
            parts.append("--use_weight_only")
        return " \\\n    ".join(parts)


if __name__ == "__main__":
    cfg = TrtLlmBuildConfig(checkpoint_dir="./qwen-ckpt", output_dir="./qwen-engine")
    print(cfg.to_cli())
