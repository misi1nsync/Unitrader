"""Run a repo-local "skill" with Claude and get structured JSON back.

A skill is a directory under ``unitrader/skills/<name>/`` holding:
  SKILL.md     instructions, used as the system prompt
  schema.json  JSON Schema the response must match (structured outputs)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd

from unitrader import config
from unitrader.features import summarize

log = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent / "skills"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16000


class SkillError(RuntimeError):
    """The skill did not produce a usable result."""


@dataclass
class SkillResult:
    output: dict[str, Any]
    model: str
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> Any:
        return self.output.get("verdict")


@dataclass(frozen=True)
class Skill:
    name: str
    instructions: str
    schema: dict[str, Any]

    @classmethod
    def load(cls, name: str) -> "Skill":
        root = SKILLS_DIR / name
        if not (root / "SKILL.md").is_file():
            raise SkillError(f"unknown skill {name!r} (no {root / 'SKILL.md'})")
        text = (root / "SKILL.md").read_text()
        if text.startswith("---"):
            # Drop YAML frontmatter; it's metadata, not instructions.
            text = text.split("---", 2)[2]
        schema = json.loads((root / "schema.json").read_text())
        return cls(name=name, instructions=text.strip(), schema=schema)


_client: anthropic.Anthropic | None = None


def _default_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def run_skill(
    name: str,
    data: pd.DataFrame | dict[str, Any],
    *,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
    effort: str | None = None,
) -> SkillResult:
    """Run skill ``name`` over market ``data`` and return its schema-valid JSON output.

    A DataFrame of OHLCV bars is summarized into per-symbol features first;
    a dict is sent as-is.
    """
    payload = summarize(data) if isinstance(data, pd.DataFrame) else data
    return _call(name, "Market data summary:\n\n" + json.dumps(payload, separators=(",", ":")),
                 client=client, model=model, effort=effort)


def invoke(skill: str, *, client: anthropic.Anthropic | None = None, model: str | None = None,
           effort: str | None = None, **inputs: Any) -> SkillResult:
    """Run ``skill`` with named JSON-serializable inputs, e.g. invoke("x", signal=..., rules=[...])."""
    name = skill.removesuffix(".md").removesuffix("_skill")
    return _call(name, "Inputs:\n\n" + json.dumps(inputs, separators=(",", ":"), default=str),
                 client=client, model=model, effort=effort)


def _call(
    name: str,
    content: str,
    *,
    client: anthropic.Anthropic | None,
    model: str | None,
    effort: str | None,
) -> SkillResult:
    skill = Skill.load(name)
    client = client or _default_client()

    response = client.beta.messages.create(
        model=model or config.model,
        max_tokens=MAX_TOKENS,
        betas=[FALLBACK_BETA],
        # If a safety classifier declines, re-run on Anthropic's recommended
        # fallback model instead of failing the hourly run.
        fallbacks="default",
        system=skill.instructions,
        output_config={
            "effort": effort or config.effort,
            "format": {"type": "json_schema", "schema": skill.schema},
        },
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "refusal":
        details = response.stop_details
        category = getattr(details, "category", None) if details else None
        raise SkillError(f"{name}: request declined (category={category}, request_id={response._request_id})")
    if response.stop_reason == "max_tokens":
        raise SkillError(f"{name}: response truncated at max_tokens={MAX_TOKENS}")

    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise SkillError(f"{name}: response had no text block (stop_reason={response.stop_reason})")
    try:
        output = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SkillError(f"{name}: response was not valid JSON: {exc}") from exc

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    if response.model != (model or config.model):
        log.warning("%s served by fallback model %s", name, response.model)
    log.info("%s: %s, %d in / %d out tokens", name, response.model, usage["input_tokens"], usage["output_tokens"])
    return SkillResult(output=output, model=response.model, usage=usage)
