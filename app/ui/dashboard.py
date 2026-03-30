"""
Streamlit dashboard: local workflow vs remote SSE to MCP (port 8000).

Run from repo root:
  uv run streamlit run app/ui/dashboard.py

Requires: .env with API keys; Docker DBs for full retrieval. For **Remote (SSE)** mode,
start ``uv run python src/mcp_server.py`` first so ``/ami/dashboard/stream`` is available.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
os.chdir(_root)

from dotenv import load_dotenv

load_dotenv(_root / ".env", override=True)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graph.entity_workflow import build_workflow
from app.logic.workflow import make_initial_state
from app.ui.workflow_stream import iter_workflow_dashboard_events
from app.utils.citations import get_citation_map, inject_citation_tooltips_html


def _hitl_enabled() -> bool:
    return os.environ.get("AMI_HITL_DIVERSITY", "").strip().lower() in ("1", "true", "yes")


def _build_ui_workflow():
    if _hitl_enabled():
        return build_workflow(checkpointer=MemorySaver())
    return build_workflow()


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


async def _collect_events(app: Any, inputs: Any, config: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    async for ev in iter_workflow_dashboard_events(app, inputs, config):
        out.append(ev)
    return out


def run_local_collect(query: str, thread_id: str) -> list[dict[str, Any]]:
    app = _build_ui_workflow()
    inputs = make_initial_state(query)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        return asyncio.run(_collect_events(app, inputs, config))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_collect_events(app, inputs, config))
        finally:
            loop.close()


def run_local_resume(thread_id: str, lambda_value: float) -> list[dict[str, Any]]:
    app = _build_ui_workflow()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        return asyncio.run(_collect_events(app, Command(resume=float(lambda_value)), config))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                _collect_events(app, Command(resume=float(lambda_value)), config)
            )
        finally:
            loop.close()


def iter_remote_sse(base: str, path: str, params: dict[str, str]):
    """Yield decoded JSON events from MCP custom SSE endpoint (blocking iterator)."""
    url = base.rstrip("/") + path
    with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        with client.stream("GET", url, params=params) as resp:
            resp.raise_for_status()
            buf = b""
            for chunk in resp.iter_bytes():
                buf += chunk
                while b"\n\n" in buf:
                    block, buf = buf.split(b"\n\n", 1)
                    for raw_line in block.split(b"\n"):
                        if raw_line.startswith(b"data: "):
                            yield json.loads(raw_line[6:].decode("utf-8"))


def _render_pulse(ev: dict[str, Any], container) -> None:
    t = ev.get("type")
    if t == "node_start":
        container.markdown(f"▸ **`{ev.get('node')}`**")
    elif t == "tool_start":
        container.caption(f"tool: `{ev.get('name')}`")


def main() -> None:
    st.set_page_config(
        page_title="AMI Research Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Agentic Media Intelligence — Research dashboard")
    st.caption(
        "Local mode runs the LangGraph in-process. Remote mode streams from the MCP server "
        "(same machine: start `uv run python src/mcp_server.py`)."
    )

    with st.sidebar:
        backend = st.radio(
            "Backend",
            ["Remote (SSE → MCP)", "Local (in-process)"],
            index=0,
            help="Remote shows live node updates via GET /ami/dashboard/stream on the MCP port.",
        )
        mcp_base = st.text_input("MCP base URL", value="http://127.0.0.1:8000")
        thread_id = st.text_input("thread_id", value=st.session_state.get("ami_tid", "streamlit-1"))
        if st.button("Reset session state"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.session_state.setdefault("ami_tid", thread_id)
    if thread_id != st.session_state.get("ami_tid"):
        st.session_state["ami_tid"] = thread_id

    query = st.text_area(
        "Research query",
        placeholder="e.g. What connects TSMC EUV procurement to ASML suppliers?",
        height=100,
    )

    st.session_state.setdefault("awaiting_resume", False)
    start = st.button("Start research", type="primary")

    if start and query.strip():
        tid = st.session_state.get("ami_tid", thread_id)
        st.session_state["ami_tid"] = tid

        if backend.startswith("Remote"):
            events: list[dict[str, Any]] = []
            error: str | None = None
            try:
                with st.status("Live pulse (SSE)", expanded=True) as pulse:
                    for ev in iter_remote_sse(
                        mcp_base,
                        "/ami/dashboard/stream",
                        {"query": query.strip(), "thread_id": tid},
                    ):
                        events.append(ev)
                        _render_pulse(ev, pulse)
                        if ev.get("type") == "interrupt":
                            st.session_state["interrupt_snapshot"] = ev
                            st.session_state["awaiting_resume"] = True
                        if ev.get("type") == "done":
                            st.session_state["last_done_status"] = ev.get("status")
                            if ev.get("status") == "interrupted":
                                st.session_state["awaiting_resume"] = True
                                st.session_state["final_state"] = ev.get("state") or {}
                            elif ev.get("status") == "partial":
                                st.session_state["awaiting_resume"] = False
                                st.session_state["final_state"] = ev.get("state") or {}
                            elif ev.get("status") == "completed":
                                st.session_state["awaiting_resume"] = False
                                st.session_state.pop("interrupt_snapshot", None)
                                st.session_state["final_state"] = ev.get("state") or {}
            except httpx.ConnectError:
                error = f"Cannot connect to {mcp_base}. Start MCP: `uv run python src/mcp_server.py`"
            except httpx.HTTPStatusError as e:
                error = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            except Exception as e:
                error = str(e)
            st.session_state["last_events"] = events
            if error:
                st.error(error)
        else:
            with st.spinner("Running workflow in-process (trace appears when finished)…"):
                events = run_local_collect(query.strip(), tid)
            st.session_state["last_events"] = events
            with st.status("Execution trace", expanded=True) as pulse:
                for ev in events:
                    _render_pulse(ev, pulse)
                    if ev.get("type") == "interrupt":
                        st.session_state["interrupt_snapshot"] = ev
                        st.session_state["awaiting_resume"] = True
                    if ev.get("type") == "done":
                        st.session_state["last_done_status"] = ev.get("status")
                        if ev.get("status") == "interrupted":
                            st.session_state["awaiting_resume"] = True
                            st.session_state["final_state"] = ev.get("state") or {}
                        elif ev.get("status") == "partial":
                            st.session_state["awaiting_resume"] = False
                            st.session_state["final_state"] = ev.get("state") or {}
                        elif ev.get("status") == "completed":
                            st.session_state["awaiting_resume"] = False
                            st.session_state.pop("interrupt_snapshot", None)
                            st.session_state["final_state"] = ev.get("state") or {}

    # After Start research updates session_state, so HITL controls render in the same run.
    if st.session_state.get("interrupt_snapshot") or st.session_state.get("awaiting_resume"):
        st.markdown("---")
        st.subheader("Diversity intervention (HITL)")
        st.warning(
            "Set **MMR λ** here (same **thread_id** as in the sidebar), then click **Resume with λ**."
        )
        lam = st.slider("MMR λ (0 = max diversity, 1 = relevance-only)", 0.0, 1.0, 0.5, 0.05)
        if st.button("Resume with λ", type="primary"):
            tid = st.session_state.get("ami_tid", thread_id)
            if backend.startswith("Remote"):
                events_resume: list[dict[str, Any]] = []
                err = None
                try:
                    with st.status("Resuming (SSE)…", expanded=True) as pulse_r:
                        for ev in iter_remote_sse(
                            mcp_base,
                            "/ami/dashboard/resume/stream",
                            {"thread_id": tid, "lambda_value": str(lam)},
                        ):
                            events_resume.append(ev)
                            _render_pulse(ev, pulse_r)
                            if ev.get("type") == "done":
                                st.session_state["awaiting_resume"] = False
                                st.session_state.pop("interrupt_snapshot", None)
                except Exception as e:
                    err = e
                if err:
                    st.error(str(err))
                elif events_resume:
                    st.session_state["last_events"] = events_resume
                    final = events_resume[-1]
                    if final.get("type") == "done" and final.get("status") == "completed":
                        st.session_state["final_state"] = final.get("state") or {}
                        st.session_state["last_done_status"] = "completed"
                    elif final.get("type") == "done":
                        st.session_state["last_done_status"] = final.get("status")
                        st.session_state["final_state"] = final.get("state") or {}
                    st.rerun()
            else:
                with st.spinner("Resuming in-process…"):
                    events_resume = run_local_resume(tid, lam)
                st.session_state["last_events"] = events_resume
                st.session_state["awaiting_resume"] = False
                st.session_state.pop("interrupt_snapshot", None)
                final = events_resume[-1] if events_resume else {}
                if final.get("type") == "done":
                    st.session_state["final_state"] = final.get("state") or {}
                    st.session_state["last_done_status"] = final.get("status")
                st.rerun()

    if st.session_state.get("final_state"):
        st.markdown("---")
        st.subheader("Executive summary")
        fs = st.session_state["final_state"]
        summary_raw = (fs.get("content") or fs.get("response") or "").strip()
        done_status = st.session_state.get("last_done_status")
        cite_map = get_citation_map(fs)

        col_main, col_div = st.columns([4, 1])
        with col_div:
            lam = float(fs.get("mmr_lambda", 1.0) or 1.0)
            lam = max(0.0, min(1.0, lam))
            st.metric(
                "Diversity score",
                f"{1.0 - lam:.2f}",
                help="1 − MMR λ (higher means more diversity weight in retrieval).",
            )
        with col_main:
            if summary_raw:
                st.markdown(
                    inject_citation_tooltips_html(summary_raw, cite_map),
                    unsafe_allow_html=True,
                )
            elif done_status == "interrupted":
                st.info(
                    "Graph paused for **HITL / MMR λ**. Use **Diversity intervention** (section above), "
                    "then **Resume with λ**."
                )
            elif done_status == "partial":
                st.warning(
                    "Run ended **before synthesis** produced a report (stream or graph stopped early, "
                    "or an error occurred after retrieval). Check the **MCP server terminal** for tracebacks."
                )
            else:
                st.warning("No report in final state yet (run may have failed or been cut off).")

        with st.expander("Sources & Evidence"):
            if not cite_map:
                st.caption("No indexed sources to show (retrieval may be empty).")
            else:
                vec_idxs = sorted(
                    i for i, d in cite_map.items() if d.get("kind") == "vector"
                )
                graph_idxs = sorted(
                    i for i, d in cite_map.items() if d.get("kind") == "graph"
                )

                if vec_idxs:
                    st.markdown(
                        "**Semantic News (Vector):** "
                        + ", ".join(f"[{i}]" for i in vec_idxs)
                    )
                    for idx in vec_idxs:
                        info = cite_map[idx]
                        st.markdown(f"**[{idx}]** {info.get('source_name', '')}")
                        url = (info.get("url") or "").strip()
                        if url:
                            st.markdown(f"[Open link]({url})")
                        st.caption(info.get("snippet", ""))

                if vec_idxs and graph_idxs:
                    st.divider()

                if graph_idxs:
                    st.markdown(
                        "**Structural facts (Graph):** "
                        + ", ".join(f"[{i}]" for i in graph_idxs)
                    )
                    for idx in graph_idxs:
                        info = cite_map[idx]
                        st.markdown(f"**[{idx}]** {info.get('source_name', '')}")
                        url = (info.get("url") or "").strip()
                        if url:
                            st.markdown(f"[Open link]({url})")
                        st.caption(info.get("snippet", ""))

        with st.expander("Technical Trace (JSON)", expanded=False):
            st.code(_json_line(st.session_state["final_state"]), language="json")


# Streamlit executes this script on each run; keep `main` at module scope.
main()
