"""Machine checks shown in /health and on the UI: is Ollama up, how much memory is free."""

import httpx

from generation_eval.config import Settings

# The judge model (qwen2.5:1.5b, ~1.1 GB) plus the frameworks need about this much free memory.
LLM_JUDGE_MEMORY_MB = 1800


def available_memory_mb() -> int | None:
    """MemAvailable from /proc/meminfo (Linux); None elsewhere."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


async def ollama_status(settings: Settings) -> dict:
    status = {"url": settings.ollama_base_url, "reachable": False, "judge_model_installed": False,
              "embedding_model_installed": False}
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            tags = (await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")).json()
    except (httpx.HTTPError, ValueError):
        return status
    installed = {m["name"] for m in tags.get("models", [])}
    status.update(
        reachable=True,
        judge_model_installed=settings.judge_model in installed,
        embedding_model_installed=any(n.split(":")[0] == settings.ragas_embedding_model for n in installed),
    )
    return status
