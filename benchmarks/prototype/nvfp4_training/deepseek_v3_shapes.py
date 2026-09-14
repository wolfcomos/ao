"""TorchTitan DeepSeek-V3 expert-weight shapes shared by grouped benchmarks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekV3ModelConfig:
    model: str
    experts: int
    expert_parallel_degree: int
    dim: int
    moe_hidden_dim: int
    tokens: int  # per expert per step, see DEEPSEEK_V3_MODEL_CONFIGS

    @property
    def local_experts(self) -> int:
        return self.experts // self.expert_parallel_degree


@dataclass(frozen=True)
class DeepSeekV3WeightShape:
    model: str
    projection: str
    experts: int
    m: int
    n: int


@dataclass(frozen=True)
class DeepSeekV3ActivationShape:
    model: str
    projection: str
    experts: int
    tokens: int
    dim: int


# tokens: per-expert tokens per step for a balanced router, EP-group ranks x local batch
# x seq x top_k / experts, from the DeepSeek-V3 training configurations the benches
# target (torchtitan flavors).
DEEPSEEK_V3_MODEL_CONFIGS = (
    # No such run; 256 keeps the debug rows at 16 tiles for E=4, the launch floor.
    DeepSeekV3ModelConfig("debugmodel", 8, 1, 256, 256, 256),
    # 2 nodes x 4 GPUs, ep 8, local batch 4, seq 4096, top_k 6: 8*4*4096*6/64 = 12288.
    DeepSeekV3ModelConfig("16B", 64, 8, 2048, 1408, 12288),
    # 671B trains at EP=64: 256 routed experts / 64 == 4 local experts per rank, which
    # is why every caller here passes factorized_experts=4 (or 2 in tests).
    # The 671B layout runs 16 nodes x 4 GPUs at ep 32, local batch 8, seq 4096, top_k 8:
    # 32*8*4096*8/256 = 32768 (the EP=64 layout with the same batch would give 65536).
    DeepSeekV3ModelConfig("671B", 256, 64, 7168, 2048, 32768),
)


def get_deepseek_v3_weight_shapes(
    *, factorized_experts: int | None = None
) -> list[DeepSeekV3WeightShape]:
    """Return TorchTitan w1/w3 and w2 shapes, optionally with a smaller E."""
    shapes = []
    for config in DEEPSEEK_V3_MODEL_CONFIGS:
        experts = factorized_experts or config.local_experts
        shapes.extend(
            (
                DeepSeekV3WeightShape(
                    config.model,
                    "gate/up (w1/w3)",
                    experts,
                    config.moe_hidden_dim,
                    config.dim,
                ),
                DeepSeekV3WeightShape(
                    config.model,
                    "down (w2)",
                    experts,
                    config.dim,
                    config.moe_hidden_dim,
                ),
            )
        )
    return shapes


def get_deepseek_v3_activation_shapes(
    side: str, *, factorized_experts: int | None = None
) -> list[DeepSeekV3ActivationShape]:
    """Return the packed ``x`` (tokens, n) or ``dy`` (tokens, m) per weight shape."""
    tokens = {config.model: config.tokens for config in DEEPSEEK_V3_MODEL_CONFIGS}
    return [
        DeepSeekV3ActivationShape(
            shape.model,
            shape.projection,
            shape.experts,
            tokens[shape.model],
            {"x": shape.n, "dy": shape.m}[side],
        )
        for shape in get_deepseek_v3_weight_shapes(
            factorized_experts=factorized_experts
        )
    ]
