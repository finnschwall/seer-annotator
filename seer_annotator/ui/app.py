"""FastAPI debug UI for inspecting local annotation state."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from ..config import PipelineConfig, Settings
from ..store import Store

_HTML_PATH = Path(__file__).parent / "static" / "index.html"


def create_app(pipeline: PipelineConfig, settings: Settings) -> FastAPI:
    store = Store(settings.runtime.store_path)
    app = FastAPI(title="SEER Annotator Debug UI", docs_url=None, redoc_url=None)

    # ------------------------------------------------------------------
    # API routes
    # ------------------------------------------------------------------

    @app.get("/api/pipeline")
    def get_pipeline() -> dict:
        return {
            "review_id": pipeline.review_id,
            "setup_id": pipeline.setup_id,
            "runs": [
                {"run_id": r.run_id, "name": r.name, "model": r.model_name,
                 "provider": r.model_provider, "config": r.config.model_dump()}
                for r in pipeline.runs
            ],
            "papers": [
                {"paper_id": p.paper_id, "title": p.title, "split": p.split}
                for p in pipeline.papers
            ],
            "questions": [
                {"question_id": q.question_id, "key": q.key, "label": q.label,
                 "question_type": q.question_type, "version_id": q.version_id}
                for q in pipeline.questions
            ],
        }

    @app.get("/api/stats")
    def get_stats() -> dict:
        stats = store.stats()
        costs = store.cost_summary()
        run_names = {r.run_id: r.name for r in pipeline.runs}
        for row in costs:
            row["run_name"] = run_names.get(row["run_id"], str(row["run_id"]))
        return {"stats": stats, "cost_summary": costs}

    @app.get("/api/answers")
    def get_answers(run_id: int | None = None, paper_id: int | None = None) -> list:
        rows = store.all_answers(run_id=run_id, paper_id=paper_id)
        # Enrich with human-readable names
        run_names = {r.run_id: r.name for r in pipeline.runs}
        paper_titles = {p.paper_id: p.title for p in pipeline.papers}
        q_labels = {q.version_id: {"key": q.key, "label": q.label} for q in pipeline.questions}

        enriched = []
        for row in rows:
            payload = None
            if row.get("payload_json"):
                try:
                    payload = json.loads(row["payload_json"])
                except Exception:
                    pass
            enriched.append(
                {
                    **row,
                    "payload": payload,
                    "payload_json": None,  # strip raw JSON from list view
                    "run_name": run_names.get(row["run_id"], str(row["run_id"])),
                    "paper_title": paper_titles.get(row["paper_id"], str(row["paper_id"])),
                    "question_info": q_labels.get(row["version_id"], {}),
                }
            )
        return enriched

    @app.get("/api/answer/{run_id}/{paper_id}/{version_id}")
    def get_answer_detail(run_id: int, paper_id: int, version_id: int) -> dict:
        rows = store.all_answers(run_id=run_id, paper_id=paper_id)
        for row in rows:
            if row["version_id"] == version_id:
                payload = None
                if row.get("payload_json"):
                    try:
                        payload = json.loads(row["payload_json"])
                        # Parse nested raw_response if it's a JSON string
                        if payload and isinstance(payload.get("raw_response"), str):
                            try:
                                payload["raw_response"] = json.loads(payload["raw_response"])
                            except Exception:
                                pass
                    except Exception:
                        pass
                return {**row, "payload": payload}
        raise HTTPException(status_code=404, detail="Answer not found")

    @app.get("/api/ocr/{paper_id}")
    def get_ocr(paper_id: int) -> dict:
        md = store.get_ocr(paper_id)
        paper = next((p for p in pipeline.papers if p.paper_id == paper_id), None)
        return {
            "paper_id": paper_id,
            "title": paper.title if paper else "",
            "abstract": paper.abstract if paper else "",
            "markdown": md,
            "has_ocr": md is not None,
        }

    # ------------------------------------------------------------------
    # Serve the SPA
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    @app.get("/{path:path}", response_class=HTMLResponse)
    def spa(path: str = "") -> HTMLResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        return HTMLResponse(_HTML_PATH.read_text())

    return app


def main() -> None:
    import sys
    import uvicorn

    if len(sys.argv) < 2:
        print("Usage: seer-ui PIPELINE.json [--host 127.0.0.1] [--port 8765]")
        sys.exit(1)

    pipeline_path = sys.argv[1]
    with open(pipeline_path) as f:
        pipeline = PipelineConfig.model_validate(json.load(f))

    settings = Settings.load()
    app = create_app(pipeline=pipeline, settings=settings)
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
