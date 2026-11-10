"""
Shared model backends for PulsePoint (action plans and the optional comment tagger).

All calls go over plain HTTP with the standard library — no extra packages.
Importing this module never touches the network.

PRIVACY: the default and recommended backend is a LOCAL Ollama model, so
employee comment text never leaves the machine. Any call that would send
comment text to a hosted provider (OpenAI, Azure OpenAI, Anthropic) is refused
unless the explicit opt-in PULSEPOINT_ALLOW_HOSTED_COMMENTS=1 is set — see
require_local_or_opt_in(). Aggregate-only requests (no comment text) may use
any configured provider.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

LOCAL_PROVIDERS = {"ollama"}
HOSTED_OPT_IN = "PULSEPOINT_ALLOW_HOSTED_COMMENTS"


class HostedCommentsNotAllowed(RuntimeError):
    """Raised when comment text would be sent to a hosted model without the opt-in."""


def hosted_comments_allowed() -> bool:
    return os.environ.get(HOSTED_OPT_IN, "").strip() == "1"


def require_local_or_opt_in(provider: str, sends_comment_text: bool) -> None:
    """Refuse to send comment text to a hosted provider unless explicitly opted in."""
    if sends_comment_text and provider not in LOCAL_PROVIDERS and not hosted_comments_allowed():
        raise HostedCommentsNotAllowed(
            f"Refusing to send employee comment text to the hosted provider '{provider}'. "
            f"Use the local Ollama backend (PULSEPOINT_LLM_PROVIDER=ollama), switch off verbatim quotes "
            f"(aggregate-only), or set {HOSTED_OPT_IN}=1 if your organisation has approved this endpoint.")


def provider_name() -> str:
    # Default to local Ollama — free, and keeps comments on the machine.
    return (os.environ.get("PULSEPOINT_LLM_PROVIDER") or "ollama").strip().lower()


def model_name(provider: str) -> str:
    m = os.environ.get("PULSEPOINT_LLM_MODEL")
    if m:
        return m
    # Defaults checked against each vendor's model docs in November 2026:
    #   OpenAI developers.openai.com/api/docs/models — gpt-6.1-sol (balanced flagship tier)
    #   Anthropic platform.claude.com API primer — claude-sonnet-5-5 (the old default was a
    #   Sonnet; claude-3-5-sonnet-latest points at a family retired in October 2025)
    #   Ollama library — llama3.1 is still published; any capable local model works.
    # Override with PULSEPOINT_LLM_MODEL. Hosted providers remain opt-in.
    return {
        "ollama": "llama3.1",
        "openai": "gpt-6.1-sol",
        "azure": os.environ.get("PULSEPOINT_AZURE_DEPLOYMENT", ""),
        "anthropic": "claude-sonnet-5-5",
    }.get(provider, "")


def configured_provider():
    """Return the provider name if it is usable (keys present), else None.

    Ollama needs no key (local), so it counts as configured by default; cloud
    providers count only when their key/endpoint env vars are set. Used to gate
    the UI so the card degrades gracefully instead of erroring."""
    p = provider_name()
    if p == "ollama":
        return "ollama"
    if p == "openai" and (os.environ.get("OPENAI_API_KEY") or os.environ.get("PULSEPOINT_OPENAI_API_KEY")):
        return "openai"
    if p == "azure" and os.environ.get("PULSEPOINT_AZURE_ENDPOINT") and \
            os.environ.get("PULSEPOINT_AZURE_DEPLOYMENT") and os.environ.get("PULSEPOINT_AZURE_API_KEY"):
        return "azure"
    if p == "anthropic" and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("PULSEPOINT_ANTHROPIC_API_KEY")):
        return "anthropic"
    return None


# ---------------------------------------------------------------------------
# LLM backends — all via stdlib urllib (no extra dependencies). temperature 0.
# ---------------------------------------------------------------------------
def _http_json(url: str, payload: dict, headers: dict, timeout: int = 300) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_ollama(system: str, user: str, model: str) -> str:
    host = os.environ.get("PULSEPOINT_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    out = _http_json(f"{host}/api/chat", {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }, headers={})
    return (out.get("message") or {}).get("content", "")


def _call_openai(system: str, user: str, model: str) -> str:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("PULSEPOINT_OPENAI_API_KEY")
    base = os.environ.get("PULSEPOINT_OPENAI_BASE", "https://api.openai.com/v1").rstrip("/")
    out = _http_json(f"{base}/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }, headers={"Authorization": f"Bearer {key}"})
    return out["choices"][0]["message"]["content"]


def _call_azure(system: str, user: str, model: str) -> str:
    endpoint = os.environ["PULSEPOINT_AZURE_ENDPOINT"].rstrip("/")
    deployment = os.environ["PULSEPOINT_AZURE_DEPLOYMENT"]
    api_version = os.environ.get("PULSEPOINT_AZURE_API_VERSION", "2024-06-01")
    key = os.environ["PULSEPOINT_AZURE_API_KEY"]
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    out = _http_json(url, {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }, headers={"api-key": key})
    return out["choices"][0]["message"]["content"]


def _call_anthropic(system: str, user: str, model: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("PULSEPOINT_ANTHROPIC_API_KEY")
    out = _http_json("https://api.anthropic.com/v1/messages", {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }, headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
    parts = out.get("content") or []
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


BACKENDS = {"ollama": _call_ollama, "openai": _call_openai,
             "azure": _call_azure, "anthropic": _call_anthropic}




def call(provider: str, system: str, user: str, model: str, sends_comment_text: bool) -> str:
    """Single entry point: privacy check, then the provider's backend."""
    require_local_or_opt_in(provider, sends_comment_text)
    if provider not in BACKENDS:
        raise RuntimeError(f"Unknown provider '{provider}'")
    return BACKENDS[provider](system, user, model)
