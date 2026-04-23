"""
api/app.py
──────────
FastAPI application — the HTTP layer of the Behavioral Personalization Engine.

Endpoints
─────────
GET  /                     → Redirect to /dashboard
GET  /dashboard            → Render and return the full HTML dashboard
GET  /api/overview         → KPI metrics JSON
GET  /api/customers        → Full customer list (filterable, sortable)
GET  /api/customers/{id}   → Single customer profile
GET  /api/segments         → Segment summaries
GET  /api/campaigns        → Today's campaign decisions
GET  /api/rules            → All timing + segment rules
GET  /api/heatmap          → Session timing heatmap matrix
POST /api/simulate         → Run simulation with custom inputs
GET  /api/export           → Generate & download the HTML dashboard file
POST /api/chat             → Natural-language customer description → segment + strategy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBase, Field

from data.db import CUSTOMERS
from engine.profiler import build_dashboard, run_simulation
from engine.html_generator import render_dashboard
from models.customer import (
    SimulationInputs, CustomerProfile, Segment, SessionTiming, Category,
)
from engine.predictor import predict_customer


# ── Pydantic I/O schemas ──────────────────────────────────────────────────────

class SimulationRequest(PydanticBase):
    discount_aggressiveness: float = Field(30, ge=0,  le=100)
    send_time_optimization:  float = Field(60, ge=0,  le=100)
    segmentation_depth:      int   = Field(4,  ge=1,  le=6)
    content_personalization: float = Field(75, ge=0,  le=100)


class ChatRequest(PydanticBase):
    message: str = Field(..., min_length=1, max_length=2000)


# ── Chat helpers ──────────────────────────────────────────────────────────────

_TIMING_HOURS: dict[str, list[int]] = {
    "morning":    [6, 7, 8],
    "lunch":      [11, 12, 13],
    "afternoon":  [13, 14, 15, 16],
    "evening":    [18, 19, 20],
    "late_night": [22, 23, 0, 1],
    "weekend":    [10, 11, 14, 15],
    "varied":     [9, 12, 15, 19],
}

_CHAT_SYSTEM = """You are a retail customer behavior analyst for a fashion e-commerce platform.

Given a natural language description of a customer's shopping habits, extract their behavioral attributes for segment classification.

Session timing reference:
- morning: shops 6am–9am (pre-work, workout crowd)
- lunch: shops 11am–1pm (lunchtime browser)
- afternoon: shops 1pm–5pm (afternoon casual)
- evening: shops 6pm–9pm (after-work shopper)
- late_night: shops 10pm–2am (night owl)
- weekend: mostly active Saturday/Sunday
- varied: no consistent timing

For attributes not explicitly mentioned, infer reasonable defaults from context clues. Examples:
- "high spender" → avg_order_value ~$300+
- "only buys on sale" → discount_usage ~0.90
- "shops occasionally" → purchase_freq ~0.8/mo
- "loyal customer" → size_consistent ~0.85, engagement_score ~80"""

_CHAT_TOOL = {
    "name": "extract_customer_profile",
    "description": "Extract structured behavioral attributes from a natural language customer description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "timing": {
                "type": "string",
                "enum": ["morning", "lunch", "afternoon", "evening", "late_night", "weekend", "varied"],
            },
            "avg_order_value":  {"type": "number",  "description": "Average order value in USD"},
            "discount_usage":   {"type": "number",  "description": "Fraction 0–1 of orders using discounts"},
            "categories": {
                "type": "array",
                "items": {"type": "string", "enum": ["streetwear", "formal", "athletic"]},
            },
            "purchase_freq":    {"type": "number",  "description": "Average purchases per month"},
            "size_consistent":  {"type": "number",  "description": "Size ordering consistency 0–1"},
            "engagement_score": {"type": "integer", "description": "Overall engagement score 0–100"},
            "interpretation":   {"type": "string",  "description": "One sentence summarising how you read the description"},
        },
        "required": [
            "timing", "avg_order_value", "discount_usage", "categories",
            "purchase_freq", "size_consistent", "engagement_score", "interpretation",
        ],
    },
}


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Behavioral Personalization Engine",
    description="Ingests customer activity data, builds dynamic behavioral profiles, and drives targeted marketing decisions.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_EXPORT_PATH = ROOT / "output" / "dashboard.html"


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse, summary="Full dashboard (HTML)", tags=["Dashboard"])
def get_dashboard():
    data = build_dashboard(CUSTOMERS)
    return HTMLResponse(content=render_dashboard(data))


@app.get("/api/overview", summary="KPI overview metrics", tags=["Analytics"])
def api_overview():
    return build_dashboard(CUSTOMERS).overview.to_dict()


@app.get("/api/customers", summary="List customer profiles", tags=["Customers"])
def api_customers(
    segment: Optional[str] = Query(None),
    sort_by: str = Query("score"),
    limit: int = Query(100, ge=1, le=500),
):
    customers = list(CUSTOMERS)
    if segment:
        customers = [c for c in customers if c.segment.value.lower() == segment.lower()]
    sort_map = {
        "score": lambda c: c.engagement_score,
        "aov":   lambda c: c.avg_order_value,
        "freq":  lambda c: c.purchase_freq,
    }
    customers.sort(key=sort_map.get(sort_by, sort_map["score"]), reverse=True)
    return [c.to_dict() for c in customers[:limit]]


@app.get("/api/customers/{customer_id}", summary="Single customer profile", tags=["Customers"])
def api_customer(customer_id: int):
    c = next((c for c in CUSTOMERS if c.id == customer_id), None)
    if not c:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return c.to_dict()


@app.get("/api/segments", summary="Segment summaries", tags=["Analytics"])
def api_segments():
    return [s.to_dict() for s in build_dashboard(CUSTOMERS).segments]


@app.get("/api/heatmap", summary="Session timing heatmap", tags=["Analytics"])
def api_heatmap():
    return build_dashboard(CUSTOMERS).heatmap.to_dict()


@app.get("/api/campaigns", summary="Campaign decisions", tags=["Campaigns"])
def api_campaigns():
    return [c.to_dict() for c in build_dashboard(CUSTOMERS).campaigns]


@app.get("/api/rules", summary="All rules and content matrix", tags=["Rules"])
def api_rules():
    data = build_dashboard(CUSTOMERS)
    return {
        "timing_rules":   [r.to_dict() for r in data.timing_rules],
        "segment_rules":  [r.to_dict() for r in data.segment_rules],
        "content_matrix": [r.to_dict() for r in data.content_matrix],
    }


@app.post("/api/simulate", summary="Run engagement & revenue simulation", tags=["Simulation"])
def api_simulate(body: SimulationRequest):
    inputs = SimulationInputs(
        discount_aggressiveness=body.discount_aggressiveness,
        send_time_optimization=body.send_time_optimization,
        segmentation_depth=body.segmentation_depth,
        content_personalization=body.content_personalization,
    )
    return run_simulation(inputs).to_dict()


@app.get("/api/export", summary="Download generated HTML dashboard", tags=["Dashboard"])
def api_export():
    data = build_dashboard(CUSTOMERS)
    render_dashboard(data, output_path=_EXPORT_PATH)
    return FileResponse(path=str(_EXPORT_PATH), filename="dashboard.html", media_type="text/html")


@app.post("/api/chat", summary="Chat-based strategy recommendation", tags=["Chat"])
def api_chat(body: ChatRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=[{"type": "text", "text": _CHAT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        tools=[_CHAT_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": body.message}],
    )

    tool_input = next((b.input for b in resp.content if b.type == "tool_use"), None)
    if not tool_input:
        raise HTTPException(status_code=500, detail="Failed to parse customer description")

    profile = CustomerProfile(
        id=0,
        name="Chat Customer",
        initials="CC",
        color="#534AB7",
        segment=Segment.FULL_PRICE,
        timing=SessionTiming(tool_input["timing"]),
        session_hours=_TIMING_HOURS.get(tool_input["timing"], [12]),
        purchase_freq=float(tool_input["purchase_freq"]),
        avg_order_value=float(tool_input["avg_order_value"]),
        categories=[Category(c) for c in tool_input.get("categories", [])],
        size_consistent=float(tool_input["size_consistent"]),
        discount_usage=float(tool_input["discount_usage"]),
        engagement_score=int(tool_input["engagement_score"]),
    )

    prediction = predict_customer(profile)

    return {
        "interpretation": tool_input["interpretation"],
        "interpreted_profile": {
            "timing":           tool_input["timing"],
            "avg_order_value":  round(float(tool_input["avg_order_value"]), 2),
            "discount_usage":   round(float(tool_input["discount_usage"]) * 100),
            "categories":       tool_input.get("categories", []),
            "purchase_freq":    round(float(tool_input["purchase_freq"]), 1),
        },
        "prediction": prediction.to_dict(),
    }
