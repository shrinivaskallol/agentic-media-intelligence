"""
Evidence indexing: align numbered citations [1], [2], ... with retrievable metadata for the UI.

Ordering matches synthesis: vector (text) chunks first, then graph facts — same as context budgeting.
"""

from __future__ import annotations

import html
import re
from typing import Any

# Same defaults as budget_context_for_synthesis
_TOP_VECTORS = 5
_TOP_GRAPH = 8
_MAX_CHARS = 18_000

_GRAPH_FACT_LINE = re.compile(r"GRAPH FACT:\s*(.+)", re.IGNORECASE)
_GRAPH_TAIL_ID = re.compile(r"\[(Graph-[^\]]+)\]\s*$")
_VECTOR_PREFIX = re.compile(r"^\[Vector-([a-f0-9]+)\]\s*", re.IGNORECASE)
# [Entity (hq, region)] --REL_TYPE--> [Entity (hq, region)]
_BRACKET_TRIPLE = re.compile(
    r"\[([^\]]+)\]\s*--([A-Z][A-Z0-9_]*)-->\s*\[([^\]]+)\]",
)

# Natural-language templates for portfolio / business-friendly display (and synthesis context).
_REL_VERB: dict[str, str] = {
    "PARTNERS_WITH": "{head} is a partner of {tail}",
    "MANUFACTURED_BY": "{head} is manufactured by {tail}",
    "SUPPLIES_EQUIPMENT_TO": "{head} supplies equipment to {tail}",
    "SUPPLIES_OPTICS_TO": "{head} supplies optics to {tail}",
    "SECURED_3NM_CAPACITY": "{head} has secured 3nm capacity from {tail}",
    "SUPPLIES_TO": "{head} supplies {tail}",
    "RELATED_TO": "{head} is related to {tail}",
    "RELATES_TO": "{head} relates to {tail}",
}

_VERIFIED_SUFFIX = " (Verified via Knowledge Graph)"
# Strip retriever decoration "[TEXT] [1]" so it does not clash with our global [1], [2] indices
_LEADING_CHUNK_TAGS = re.compile(
    r"^\[(?:TEXT|PERSON|ORG|LOC|PRODUCT)\]\s*\[\d+\]\s*",
    re.IGNORECASE,
)


def _get_field(state: Any, name: str, default: Any) -> Any:
    if state is None:
        return default
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def _split_sections(context: list[Any]) -> tuple[str, str]:
    graph_s, vec_s = "", ""
    for block in context or []:
        s = str(block)
        if "--- GRAPH KNOWLEDGE ---" in s:
            graph_s = s
        elif "--- TEXT CHUNKS" in s or "TEXT CHUNKS" in s:
            vec_s = s
    return graph_s, vec_s


def _clean_entity_label(raw: str) -> str:
    """Strip '(hq, region)' suffix from Neo4j-formatted node labels."""
    s = raw.strip()
    if " (" in s:
        return s.split(" (", 1)[0].strip()
    return s


def _relation_fallback_phrase(rel: str, head: str, tail: str) -> str:
    readable = rel.replace("_", " ").strip().lower()
    return f"{head} {readable} {tail}"


def humanize_graph_fact_payload(payload: str) -> str:
    """
    Turn graph fact lines into short prose for UI and synthesis.

    Supports bracket triples from Neo4j formatting and a minimal ``Head REL Tail`` fallback.
    """
    p = str(payload).strip()
    p = _GRAPH_TAIL_ID.sub("", p).strip()

    head, rel, tail = None, None, None
    m = _BRACKET_TRIPLE.search(p)
    if m:
        head = _clean_entity_label(m.group(1))
        rel = m.group(2).strip()
        tail = _clean_entity_label(m.group(3))
    else:
        tokens = p.split()
        for i in range(1, len(tokens) - 1):
            tok = tokens[i]
            if "_" in tok or (tok.isupper() and len(tok) > 1):
                rel = tok
                head = " ".join(tokens[:i]).strip()
                tail = " ".join(tokens[i + 1 :]).strip()
                if head and tail:
                    break
                head, rel, tail = None, None, None

    if not head or not rel or not tail:
        return p + _VERIFIED_SUFFIX if p else ""

    template = _REL_VERB.get(rel)
    if template:
        core = template.format(head=head, tail=tail)
    else:
        core = _relation_fallback_phrase(rel, head, tail)
    return core + _VERIFIED_SUFFIX


def _meta_for_chunk_id(chunk_id: str, ids: list[str], metas: list[dict[str, Any]]) -> dict[str, Any]:
    cid = chunk_id.lower()
    for i, x in enumerate(ids):
        if str(x).lower() == cid:
            return metas[i] if i < len(metas) else {}
    return {}


def collect_evidence_entries(
    context: list[Any],
    *,
    retrieved_ids: list[str] | None = None,
    retrieved_metadata: list[dict[str, Any]] | None = None,
    top_vectors: int = _TOP_VECTORS,
    top_graph: int = _TOP_GRAPH,
) -> list[dict[str, Any]]:
    """
    Ordered evidence rows for numbering. Vectors first, then graph facts (matches budget_context order).
    Each entry: kind, body (for LLM), snippet, source_name, url, source_id, graph_fact_id.
    """
    ids = list(retrieved_ids or [])
    metas = list(retrieved_metadata or [])
    graph_s, vec_s = _split_sections(list(context or []))
    entries: list[dict[str, Any]] = []

    vec_count = 0
    for ln in vec_s.split("\n"):
        ln = ln.strip()
        if not ln or ln.startswith("---") or "(No vector results)" in ln:
            continue
        m = _VECTOR_PREFIX.match(ln)
        if not m:
            continue
        if vec_count >= top_vectors:
            break
        chunk_id = m.group(1)
        rest = _LEADING_CHUNK_TAGS.sub("", ln[m.end() :].strip(), count=1)
        meta = _meta_for_chunk_id(chunk_id, ids, metas)
        entity = meta.get("entity")
        if entity:
            source_name = f"News / filing ({entity})"
        else:
            source_name = f"Vector chunk {chunk_id[:8]}…"
        body = rest if rest else ln
        entries.append(
            {
                "kind": "vector",
                "body": body,
                "snippet": body,
                "source_name": source_name,
                "url": str(meta.get("url") or ""),
                "source_id": chunk_id,
                "graph_fact_id": "",
            }
        )
        vec_count += 1

    graph_count = 0
    for ln in graph_s.split("\n"):
        ln = ln.strip()
        if not ln or ln.startswith("---") or "(No graph results)" in ln:
            continue
        if "GRAPH FACT:" not in ln:
            continue
        if graph_count >= top_graph:
            break
        gm = _GRAPH_FACT_LINE.search(ln)
        if not gm:
            continue
        payload = gm.group(1).strip()
        fact_id = ""
        tid = _GRAPH_TAIL_ID.search(ln)
        if tid:
            fact_id = tid.group(1)
        human = humanize_graph_fact_payload(payload)
        short_label = f"Structural fact ({fact_id})" if fact_id else "Structural fact"
        entries.append(
            {
                "kind": "graph",
                "body": human,
                "snippet": human,
                "source_name": short_label,
                "url": "",
                "source_id": fact_id or payload[:48],
                "graph_fact_id": fact_id,
            }
        )
        graph_count += 1

    return entries


def build_numbered_context_for_llm(state: Any) -> str:
    """Flatten evidence into numbered lines for the synthesis prompt."""
    ctx = list(_get_field(state, "context", []) or [])
    ids = list(_get_field(state, "retrieved_ids", []) or [])
    metas = list(_get_field(state, "retrieved_metadata", []) or [])
    entries = collect_evidence_entries(ctx, retrieved_ids=ids, retrieved_metadata=metas)
    if not entries:
        return "No relevant data retrieved."
    lines: list[str] = []
    for i, e in enumerate(entries, start=1):
        label = "Graph" if e["kind"] == "graph" else "Vector"
        lines.append(f"[{i}] ({label}) {e['body']}")
    text = "\n\n".join(lines)
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "\n\n[Context truncated to stay under token limit.]"
    return text


def get_citation_map(
    state: Any, *, snippet_max: int = 500
) -> dict[int, dict[str, str]]:
    """
    Map citation index (1-based) to source metadata for the Sources panel.

    Keys are integers; values contain source_name, url, snippet, kind (``vector`` | ``graph``).
    """
    ctx = list(_get_field(state, "context", []) or [])
    ids = list(_get_field(state, "retrieved_ids", []) or [])
    metas = list(_get_field(state, "retrieved_metadata", []) or [])
    entries = collect_evidence_entries(ctx, retrieved_ids=ids, retrieved_metadata=metas)
    out: dict[int, dict[str, str]] = {}
    for i, e in enumerate(entries, start=1):
        snip = e["snippet"]
        if len(snip) > snippet_max:
            snip = snip[:snippet_max] + "…"
        out[i] = {
            "source_name": str(e["source_name"]),
            "url": str(e.get("url") or ""),
            "snippet": snip,
            "kind": str(e["kind"]),
        }
    return out


_CITATION_NUM = re.compile(r"\[(\d+)\]")


def inject_citation_tooltips_html(markdown_summary: str, citation_map: dict[int, dict[str, str]]) -> str:
    """
    Wrap numeric citations [n] in HTML spans with title = snippet (or source_name) for hover hints.
    Safe for st.markdown(..., unsafe_allow_html=True).
    """

    def repl(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        info = citation_map.get(idx, {})
        title = (info.get("snippet") or info.get("source_name") or f"Source {idx}")[:800]
        title_esc = html.escape(title, quote=True)
        return f'<span title="{title_esc}">[{idx}]</span>'

    return _CITATION_NUM.sub(repl, markdown_summary)
