from __future__ import annotations


class BetaAnneal:
    """Linear ramp from 0 → beta_max over the first warmup_frac of total steps."""

    def __init__(self, warmup_frac: float = 0.2, beta_max: float = 1.0) -> None:
        self.warmup_frac = warmup_frac
        self.beta_max = beta_max

    def get(self, step: int, total_steps: int) -> float:
        warmup = self.warmup_frac * total_steps
        return self.beta_max * min(1.0, step / warmup) if warmup > 0 else self.beta_max


class DannLambda:
    """Linear ramp from 0 → lambda_max over the first warmup_frac of total steps."""

    def __init__(self, warmup_frac: float = 0.2, lambda_max: float = 1.0) -> None:
        self.warmup_frac = warmup_frac
        self.lambda_max = lambda_max

    def get(self, step: int, total_steps: int) -> float:
        warmup = self.warmup_frac * total_steps
        return self.lambda_max * min(1.0, step / warmup) if warmup > 0 else self.lambda_max
