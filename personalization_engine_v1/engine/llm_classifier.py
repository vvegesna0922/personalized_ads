"""
engine/llm_classifier.py
------------------------
LLM-backed shopping type classifier.

This module classifies customers from behavioral facts and product signals,
without using a pre-filled customer segment as an input. The LLM returns a
validated structured object so the rest of the app can treat the result as
data rather than free-form prose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from models.customer import CustomerProfile


SHOPPING_TYPES = [
    "Full-Price",
    "Night Streetwear",
    "Lunch Shopper",
    "Sale Shopper",
    "Athletic Regular",
    "Formal Rare",
    "Unclear",
]

PRIMARY_CATEGORIES = ["streetwear", "formal", "athletic", "mixed", "unknown"]
PRICE_SENSITIVITY = ["low", "medium", "high", "unknown"]
CHANNELS = ["Email", "Push", "SMS", "unknown"]


@dataclass
class CustomerBehaviorFeatures:
    customer_id: int | None = None
    active_hours: list[int] = field(default_factory=list)
    purchase_freq: float = 0.0
    avg_order_value: float = 0.0
    discount_usage: float = 0.0
    size_consistent: float = 0.0
    engagement_score: int = 0
    product_signals: list[str] = field(default_factory=list)
    recent_items: list[str] = field(default_factory=list)
    channel_metrics: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_customer(cls, customer: CustomerProfile) -> "CustomerBehaviorFeatures":
        return cls(
            customer_id=customer.id,
            active_hours=customer.session_hours,
            purchase_freq=customer.purchase_freq,
            avg_order_value=customer.avg_order_value,
            discount_usage=customer.discount_usage,
            size_consistent=customer.size_consistent,
            engagement_score=customer.engagement_score,
            product_signals=[c.value for c in customer.categories],
            notes=(
                "Product signals are observed category affinities from browsing or purchase data. "
                "Do not use any assigned customer segment."
            ),
        )

    def to_llm_payload(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "active_hours": self.active_hours,
            "purchase_freq_per_month": self.purchase_freq,
            "avg_order_value_usd": self.avg_order_value,
            "discounted_purchase_fraction": self.discount_usage,
            "size_consistency": self.size_consistent,
            "engagement_score_0_100": self.engagement_score,
            "product_signals": self.product_signals,
            "recent_items": self.recent_items,
            "channel_metrics": self.channel_metrics,
            "notes": self.notes,
        }


_SYSTEM = """You classify ecommerce customers into shopping types from observed data.

Use only the provided behavioral facts and product signals. Do not assume an assigned segment exists.
If the evidence is thin or contradictory, choose "Unclear" rather than forcing a weak label.

Shopping type definitions:
- Full-Price: high AOV, low discount dependency, often evening/editorial browsing.
- Night Streetwear: late-night activity with streetwear/drop/sneaker/hoodie signals.
- Lunch Shopper: midday activity, moderate basket size, quick browsing, free-shipping or convenience signals.
- Sale Shopper: high discount dependency, low AOV, purchases clustered around deals.
- Athletic Regular: athletic/product-performance signals, frequent purchasing, often morning activity.
- Formal Rare: formal/premium signals, high AOV, infrequent purchases, often weekend browsing.

Return concise evidence grounded in the metrics."""

_TOOL = {
    "name": "classify_customer_shopping_type",
    "description": "Classify a customer from behavior and product signals without using a pre-filled segment.",
    "input_schema": {
        "type": "object",
        "properties": {
            "shopping_type": {"type": "string", "enum": SHOPPING_TYPES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "primary_category": {"type": "string", "enum": PRIMARY_CATEGORIES},
            "secondary_categories": {
                "type": "array",
                "items": {"type": "string", "enum": PRIMARY_CATEGORIES},
            },
            "price_sensitivity": {"type": "string", "enum": PRICE_SENSITIVITY},
            "best_channel": {"type": "string", "enum": CHANNELS},
            "best_send_time": {"type": "string"},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
            },
            "recommended_action": {"type": "string"},
        },
        "required": [
            "shopping_type",
            "confidence",
            "primary_category",
            "secondary_categories",
            "price_sensitivity",
            "best_channel",
            "best_send_time",
            "evidence",
            "recommended_action",
        ],
    },
}


def _clamp_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalise_result(raw: dict[str, Any]) -> dict[str, Any]:
    shopping_type = raw.get("shopping_type")
    primary_category = raw.get("primary_category")
    price_sensitivity = raw.get("price_sensitivity")
    best_channel = raw.get("best_channel")
    secondary = raw.get("secondary_categories") or []
    evidence = raw.get("evidence") or []

    if shopping_type not in SHOPPING_TYPES:
        shopping_type = "Unclear"
    if primary_category not in PRIMARY_CATEGORIES:
        primary_category = "unknown"
    if price_sensitivity not in PRICE_SENSITIVITY:
        price_sensitivity = "unknown"
    if best_channel not in CHANNELS:
        best_channel = "unknown"

    return {
        "shopping_type": shopping_type,
        "confidence": round(_clamp_confidence(raw.get("confidence")), 2),
        "primary_category": primary_category,
        "secondary_categories": [c for c in secondary if c in PRIMARY_CATEGORIES and c != primary_category],
        "price_sensitivity": price_sensitivity,
        "best_channel": best_channel,
        "best_send_time": str(raw.get("best_send_time") or "unknown"),
        "evidence": [str(item) for item in evidence[:5]] or ["Insufficient evidence provided."],
        "recommended_action": str(raw.get("recommended_action") or "Collect more behavioral history before targeting."),
    }


def classify_customer_with_llm(features: CustomerBehaviorFeatures) -> dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=[_TOOL],
            tool_choice={"type": "any"},
            messages=[{
                "role": "user",
                "content": (
                    "Classify this customer from the provided metrics. "
                    "Do not rely on any assigned segment.\n\n"
                    f"{features.to_llm_payload()}"
                ),
            }],
        )
    except Exception as e:
        raise RuntimeError(f"LLM classification request failed: {e}") from e

    tool_input = next((b.input for b in resp.content if getattr(b, "type", None) == "tool_use"), None)
    if not tool_input:
        raise RuntimeError("Failed to classify customer behavior")

    return {
        "input_features": features.to_llm_payload(),
        "classification": _normalise_result(tool_input),
    }
