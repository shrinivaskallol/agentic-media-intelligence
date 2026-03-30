"""
FastAPI app exposing the entity workflow as a chat API.
POST /chat: run workflow with query and thread_id for stateful conversations.
GET /health: verify Neo4j and Postgres connections.

Run: uv run uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

from app import configure_logging

configure_logging()
from fastapi import FastAPI, HTTPException

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from pydantic import BaseModel, Field

from app.graph.entity_workflow import build_workflow, get_checkpoint_db_path
from app.tools.db_utils import connect_postgres, get_neo4j_driver


def _make_initial_state(query: str) -> dict:
    """Create initial state for the entity workflow (GraphState)."""
    return {
        "query": query,
        "entities": [],
        "intent": "",
        "context": [],
        "response": "",
        "mmr_lambda": 1.0,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build workflow with SQLite checkpointer at startup; tear down on shutdown."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = get_checkpoint_db_path()
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        app.state.workflow = build_workflow(checkpointer=saver)
        yield


app = FastAPI(title="Agentic Media Intelligence", lifespan=lifespan)


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    query: str = Field(..., description="User natural language query.")
    thread_id: str = Field(..., description="Conversation thread ID for stateful memory.")


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    neo4j: str
    postgres: str


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    """
    Run the entity workflow for the given query.
    Uses thread_id for persistent conversation memory (checkpoints).
    Returns the final state from the graph.
    """
    workflow = app.state.workflow
    inputs = _make_initial_state(request.query)
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        final_state = await workflow.ainvoke(inputs, config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Return state as dict (Pydantic model or dict)
    if hasattr(final_state, "model_dump"):
        return final_state.model_dump()
    return dict(final_state) if final_state else {}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Verify Neo4j and Postgres connections are alive.
    """
    neo4j_ok = postgres_ok = False

    # Check Neo4j
    try:
        from neo4j.exceptions import ServiceUnavailable

        driver = get_neo4j_driver()
        if driver:
            driver.verify_connectivity()
            driver.close()
            neo4j_ok = True
    except (ServiceUnavailable, OSError):
        pass

    # Check Postgres
    try:
        conn = connect_postgres()
        if conn:
            conn.close()
            postgres_ok = True
    except OSError:
        pass

    status = "ok" if (neo4j_ok and postgres_ok) else "degraded"
    return HealthResponse(
        status=status,
        neo4j="ok" if neo4j_ok else "error",
        postgres="ok" if postgres_ok else "error",
    )
