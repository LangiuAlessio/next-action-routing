"""
LLM ReAct predictor — N1 baseline.

Uses `claude -p` subprocess (OAuth via Max plan, no API key needed).
ReAct framing: model outputs a brief reasoning trace then a ranked JSON
prediction. Interpretability score = fraction of context actions cited
in the reasoning trace.

Vocabulary management: only the top-`max_vocab` most frequent training
actions are shown to the model (keeps prompts short and cheap).
"""

from __future__ import annotations
import json
import re
import subprocess
import time
from collections import Counter
from typing import Iterable

_SYSTEM_PROMPT = (
    "You are a next-action predictor. "
    "Given a user's recent action history, predict the most likely next action. "
    "Output ONLY the requested JSON — no prose, no markdown fences."
)

_PRED_TEMPLATE = """\
History (oldest to newest): {history}

Candidate actions: {vocab}

Briefly reason (1-2 sentences), then output a JSON object on the LAST line:
{{"reasoning": "<brief analysis>", "top5": [{{"action":"A","score":0.7}}, ...]}}

Rules:
- Use only actions from the candidate list
- Scores must be non-negative and roughly sum to 1.0
- Output exactly one JSON object on the last line
"""


def _extract_json(text: str) -> dict:
    """Extract the last {...} block from model output."""
    # Try direct parse first
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Find last {...} block (handles trailing newlines, markdown fences)
    matches = list(re.finditer(r'\{[^{}]*"top5"[^{}]*\}', stripped, re.DOTALL))
    if not matches:
        # broader search
        matches = list(re.finditer(r'\{.*\}', stripped, re.DOTALL))
    if matches:
        try:
            return json.loads(matches[-1].group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in: {stripped[:300]}")


class LLMReAct:
    """
    N1 baseline: ReAct-style LLM predictor via `claude -p` subprocess.

    fit() builds a frequency-ranked vocabulary from training sequences.
    predict_top() calls `claude -p` with a ReAct prompt and parses the
    structured JSON response.
    update() is a no-op (LLMs are stateless per-call in this setup).
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_vocab: int = 30,
        timeout: int = 60,
    ):
        self.model = model
        self.max_vocab = max_vocab
        self.timeout = timeout
        self._freq: Counter = Counter()
        self._top_vocab: list[str] = []
        self._last_reasoning: str = ""

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, sequences: Iterable[list[str]]) -> "LLMReAct":
        for seq in sequences:
            self._freq.update(seq)
        self._top_vocab = [a for a, _ in self._freq.most_common(self.max_vocab)]
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, context: list[str]) -> dict[str, float]:
        vocab = self._top_vocab or list(self._freq.keys())[:self.max_vocab]
        if not vocab:
            return {}

        history_str = " → ".join(context[-6:]) if context else "(empty)"
        vocab_str = ", ".join(vocab)
        prompt = _PRED_TEMPLATE.format(history=history_str, vocab=vocab_str)

        try:
            result = subprocess.run(
                [
                    "claude", "-p", prompt,
                    "--system-prompt", _SYSTEM_PROMPT,
                    "--output-format", "json",
                    "--model", self.model,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd="/tmp",
            )
            outer = json.loads(result.stdout)
            response_text = outer.get("result", "")
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
            self._last_reasoning = ""
            return {}

        try:
            parsed = _extract_json(response_text)
        except ValueError:
            self._last_reasoning = ""
            return {}

        self._last_reasoning = parsed.get("reasoning", "")
        top5 = parsed.get("top5", [])

        raw: dict[str, float] = {}
        vocab_set = set(vocab)
        for item in top5:
            if isinstance(item, dict):
                action = item.get("action", "")
                score = float(item.get("score", 0.0))
                if action in vocab_set and score > 0:
                    raw[action] = score

        total = sum(raw.values())
        if total <= 0:
            return {}
        return {a: v / total for a, v in raw.items()}

    def predict_top(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        dist = self.predict(context)
        return sorted(dist.items(), key=lambda x: -x[1])[:k]

    def update(self, context: list[str], next_action: str) -> None:
        pass  # stateless per-call; online update not applicable

    def interpretability_score(self, context: list[str]) -> float:
        """
        Fraction of reasoning words that are actions present in context.
        Operationalises the grounded-context ratio for ReAct traces.
        """
        if not self._last_reasoning or not context:
            return 0.0
        ctx_set = set(context)
        words = re.findall(r"[\w./-]+", self._last_reasoning.lower())
        if not words:
            return 0.0
        grounded = sum(1 for w in words if w in ctx_set)
        return min(grounded / len(words), 1.0)
