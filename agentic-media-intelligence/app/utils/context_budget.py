"""
Context budgeting: truncate retrieved context to stay under LLM token limits.
Groq 8B has ~6k TPM; use this to avoid 413 errors while preserving key grounding data.
"""


def budget_context(
    ctx: list,
    *,
    top_text_chunks: int = 5,
    top_graph_facts: int = 8,
    max_chars: int = 18_000,
) -> str:
    """
    Truncate context list to stay under token budget (~4 chars/token).
    Keeps top N text chunks and top M graph facts. Hard cap at max_chars.
    """
    if not ctx:
        return "No context."
    text_section = None
    graph_section = None
    for item in ctx:
        s = str(item)
        if "--- TEXT CHUNKS" in s or "TEXT CHUNKS" in s:
            text_section = s
        elif "--- GRAPH KNOWLEDGE" in s or "GRAPH KNOWLEDGE" in s:
            graph_section = s
    parts = []
    if text_section:
        lines = [
            ln.strip()
            for ln in text_section.split("\n")
            if ln.strip() and not ln.strip().startswith("---")
        ]
        chunk_lines = lines[: top_text_chunks * 4]  # ~4 lines per chunk
        if chunk_lines:
            parts.append("--- TEXT CHUNKS (with entity types) ---\n" + "\n".join(chunk_lines))
    if graph_section:
        glines = [ln for ln in graph_section.split("\n") if ln.strip() and "GRAPH FACT:" in ln][
            :top_graph_facts
        ]
        if glines:
            parts.append("--- GRAPH KNOWLEDGE ---\n" + "\n".join(glines))
    result = "\n\n".join(parts) if parts else "No context."
    if len(result) > max_chars:
        result = result[:max_chars] + "\n\n[Context truncated to stay under token limit.]"
    return result


def budget_context_for_synthesis(ctx: list) -> str:
    """Synthesis-specific budget: ~4500 tokens (stays under Groq 6k TPM with prompt)."""
    return budget_context(ctx, top_text_chunks=5, top_graph_facts=8, max_chars=18_000)
