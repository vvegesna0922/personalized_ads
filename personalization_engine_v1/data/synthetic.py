"""
data/synthetic.py
─────────────────
Synthetic customer generator for classifier evaluation.

Generates N customers by sampling behavioral features from per-segment
probability distributions. Distributions are deliberately overlapping at
segment boundaries — e.g., ~15% of Athletic Regulars browse at lunch,
~18% of Formal Rare customers shop in the evening instead of weekends —
so the rule-based classifier produces non-trivial accuracy.

Ground truth is the segment used during generation. This lets us measure
precision, recall, and drift rate honestly rather than against hand-crafted
profiles tuned to match the scoring rules.
"""

from __future__ import annotations
import random
from models.customer import CustomerProfile, SessionTiming, Category, Segment


# ── Segment configuration ──────────────────────────────────────────────────────

SEGMENT_PROPORTIONS: dict[Segment, float] = {
    Segment.FULL_PRICE:       0.11,
    Segment.NIGHT_STREETWEAR: 0.19,
    Segment.LUNCH_SHOPPER:    0.17,
    Segment.SALE_SHOPPER:     0.25,
    Segment.ATHLETIC_REGULAR: 0.15,
    Segment.FORMAL_RARE:      0.13,
}

_COLORS: dict[Segment, str] = {
    Segment.FULL_PRICE:       "#534AB7",
    Segment.NIGHT_STREETWEAR: "#0F6E56",
    Segment.LUNCH_SHOPPER:    "#854F0B",
    Segment.SALE_SHOPPER:     "#A32D2D",
    Segment.ATHLETIC_REGULAR: "#185FA5",
    Segment.FORMAL_RARE:      "#3c3489",
}

_FIRST_NAMES = [
    "Alex", "Jordan", "Morgan", "Casey", "Riley", "Taylor", "Quinn", "Avery",
    "Blake", "Cameron", "Dakota", "Emery", "Finley", "Harper", "Jaden",
    "Kendall", "Logan", "Marlowe", "Nico", "Oakley", "Payton", "Reese",
    "Sage", "Tatum", "Uma", "Vesper", "Wren", "Xian", "Yael", "Zara",
    "Aiden", "Brianna", "Colin", "Diana", "Ethan", "Fiona", "Gavin",
    "Helena", "Ivan", "Julia", "Kevin", "Luna", "Mateo", "Nadia", "Oscar",
    "Petra", "Rafael", "Sienna", "Tobias", "Ursula", "Victor", "Willow",
    "Xander", "Yasmine", "Zion", "Aaron", "Bella", "Carlos", "Demi",
    "Eduardo", "Freya", "Gideon", "Hana", "Ignacio", "Juno", "Kiran",
    "Layla", "Marco", "Nina", "Omar", "Piper", "Rosa", "Stefan",
    "Tessa", "Umar", "Viola", "Wade", "Xenia", "Yuki", "Adele",
    "Bruno", "Cara", "Derek", "Elsa", "Franz", "Gina", "Hugo",
    "Iris", "Jake", "Kira", "Liam", "Maya", "Nolan", "Olive",
    "Paulo", "Quin", "Rena", "Sam", "Thea", "Uri", "Val",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Martinez", "Wilson", "Anderson", "Taylor", "Thomas", "Hernandez",
    "Moore", "Martin", "Jackson", "Thompson", "White", "Lopez", "Lee",
    "Gonzalez", "Harris", "Clark", "Lewis", "Robinson", "Walker", "Perez",
    "Hall", "Young", "Allen", "Sanchez", "Wright", "King", "Scott",
    "Green", "Baker", "Adams", "Nelson", "Hill", "Ramirez", "Campbell",
    "Mitchell", "Roberts", "Carter", "Phillips", "Evans", "Turner", "Torres",
    "Parker", "Collins", "Edwards", "Stewart", "Flores", "Morris", "Nguyen",
    "Murphy", "Rivera", "Cook", "Rogers", "Morgan", "Peterson", "Cooper",
    "Reed", "Bailey", "Bell", "Gomez", "Kelly", "Howard", "Ward",
    "Cox", "Diaz", "Richardson", "Wood", "Watson", "Brooks", "Bennett",
    "Patel", "Kim", "Okafor", "Volkov", "Diallo", "Müller", "Tanaka",
]


# ── Sampling helpers ───────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _norm(rng: random.Random, mu: float, sigma: float, lo: float, hi: float) -> float:
    return _clamp(rng.gauss(mu, sigma), lo, hi)

def _beta(rng: random.Random, alpha: float, beta_p: float,
          lo: float = 0.0, hi: float = 1.0) -> float:
    return _clamp(rng.betavariate(alpha, beta_p), lo, hi)

def _choice(rng: random.Random, options: list[tuple], weights: list[float]):
    """Weighted random choice from a list of options."""
    r = rng.random()
    cumulative = 0.0
    for opt, w in zip(options, weights):
        cumulative += w
        if r < cumulative:
            return opt
    return options[-1]

def _session_hours(rng: random.Random, timing: SessionTiming) -> list[int]:
    pools: dict[SessionTiming, list[int]] = {
        SessionTiming.MORNING:    [5, 6, 7, 8],
        SessionTiming.LUNCH:      [11, 12, 13],
        SessionTiming.AFTERNOON:  [13, 14, 15, 16],
        SessionTiming.EVENING:    [18, 19, 20, 21],
        SessionTiming.LATE_NIGHT: [22, 23, 0, 1, 2],
        SessionTiming.VARIED:     [8, 10, 14, 17, 19, 21],
        SessionTiming.WEEKEND:    [10, 11, 14, 15],
    }
    pool = pools[timing]
    n = rng.randint(2, min(3, len(pool)))
    return sorted(rng.sample(pool, n))

def _make_name(rng: random.Random, used: set[str]) -> tuple[str, str]:
    for _ in range(200):
        first = rng.choice(_FIRST_NAMES)
        last  = rng.choice(_LAST_NAMES)
        name  = f"{first} {last}"
        if name not in used:
            used.add(name)
            return name, first[0] + last[0]
    fallback = f"Customer {rng.randint(1000, 9999)}"
    return fallback, "C?"


# ── Per-segment generators ─────────────────────────────────────────────────────
# Each function samples from distributions calibrated so that ~12-18% of
# customers land near a segment boundary and may be misclassified.

def _gen_full_price(rng: random.Random, cid: int, used: set[str]) -> CustomerProfile:
    name, initials = _make_name(rng, used)
    timing = _choice(rng,
        [SessionTiming.EVENING, SessionTiming.MORNING, SessionTiming.LUNCH],
        [0.68, 0.20, 0.12])     # 12% LUNCH → confusable with Lunch Shopper
    aov      = _norm(rng, 228, 55, 120, 420)
    discount = _beta(rng, 1.2, 15, 0.0, 0.30)
    freq     = _norm(rng, 4.0, 1.0, 1.5, 7.5)
    size_c   = _norm(rng, 0.92, 0.05, 0.72, 1.0)
    eng      = int(_norm(rng, 87, 12, 52, 100))
    cats     = _choice(rng,
        [[Category.FORMAL, Category.STREETWEAR],
         [Category.FORMAL],
         [Category.FORMAL, Category.ATHLETIC],
         [Category.STREETWEAR]],
        [0.40, 0.28, 0.20, 0.12])
    return CustomerProfile(
        id=cid, name=name, initials=initials,
        color=_COLORS[Segment.FULL_PRICE],
        segment=Segment.FULL_PRICE, timing=timing,
        session_hours=_session_hours(rng, timing),
        purchase_freq=round(freq, 1),
        avg_order_value=round(aov),
        categories=cats,
        size_consistent=round(size_c, 2),
        discount_usage=round(discount, 2),
        engagement_score=eng,
    )


def _gen_night_streetwear(rng: random.Random, cid: int, used: set[str]) -> CustomerProfile:
    name, initials = _make_name(rng, used)
    timing = _choice(rng,
        [SessionTiming.LATE_NIGHT, SessionTiming.EVENING, SessionTiming.VARIED],
        [0.80, 0.12, 0.08])     # 12% EVENING → confusable with Full-Price
    aov      = _norm(rng, 87, 22, 38, 150)
    discount = _beta(rng, 3, 8, 0.03, 0.62)
    freq     = _norm(rng, 2.4, 0.7, 1.0, 4.8)
    size_c   = _norm(rng, 0.85, 0.07, 0.62, 0.99)
    eng      = int(_norm(rng, 65, 15, 28, 95))
    # 13% have no STREETWEAR → timing signal alone must carry classification
    cats     = _choice(rng,
        [[Category.STREETWEAR],
         [Category.STREETWEAR, Category.ATHLETIC],
         [Category.ATHLETIC]],
        [0.65, 0.22, 0.13])
    return CustomerProfile(
        id=cid, name=name, initials=initials,
        color=_COLORS[Segment.NIGHT_STREETWEAR],
        segment=Segment.NIGHT_STREETWEAR, timing=timing,
        session_hours=_session_hours(rng, timing),
        purchase_freq=round(freq, 1),
        avg_order_value=round(aov),
        categories=cats,
        size_consistent=round(size_c, 2),
        discount_usage=round(discount, 2),
        engagement_score=eng,
    )


def _gen_lunch_shopper(rng: random.Random, cid: int, used: set[str]) -> CustomerProfile:
    name, initials = _make_name(rng, used)
    timing = _choice(rng,
        [SessionTiming.LUNCH, SessionTiming.AFTERNOON, SessionTiming.MORNING],
        [0.78, 0.12, 0.10])     # 10% MORNING → confusable with Athletic Regular
    aov      = _norm(rng, 74, 20, 33, 130)
    discount = _beta(rng, 5, 8, 0.18, 0.68)
    freq     = _norm(rng, 2.8, 0.7, 1.5, 5.2)
    size_c   = _norm(rng, 0.84, 0.07, 0.63, 0.99)
    eng      = int(_norm(rng, 60, 13, 28, 92))
    cats     = _choice(rng,
        [[Category.ATHLETIC, Category.STREETWEAR],
         [Category.ATHLETIC],
         [Category.STREETWEAR],
         [Category.ATHLETIC, Category.STREETWEAR, Category.FORMAL]],
        [0.38, 0.28, 0.22, 0.12])
    return CustomerProfile(
        id=cid, name=name, initials=initials,
        color=_COLORS[Segment.LUNCH_SHOPPER],
        segment=Segment.LUNCH_SHOPPER, timing=timing,
        session_hours=_session_hours(rng, timing),
        purchase_freq=round(freq, 1),
        avg_order_value=round(aov),
        categories=cats,
        size_consistent=round(size_c, 2),
        discount_usage=round(discount, 2),
        engagement_score=eng,
    )


def _gen_sale_shopper(rng: random.Random, cid: int, used: set[str]) -> CustomerProfile:
    name, initials = _make_name(rng, used)
    timing = _choice(rng,
        [SessionTiming.VARIED, SessionTiming.EVENING,
         SessionTiming.LUNCH,  SessionTiming.LATE_NIGHT],
        [0.42, 0.25, 0.18, 0.15])   # LUNCH and LATE_NIGHT → boundary confusion
    aov      = _norm(rng, 42, 14, 17, 84)
    discount = _beta(rng, 10, 1.5, 0.68, 0.99)
    freq     = _norm(rng, 1.3, 0.45, 0.5, 2.6)
    size_c   = _norm(rng, 0.68, 0.08, 0.48, 0.90)
    eng      = int(_norm(rng, 33, 12, 8, 65))
    cats     = _choice(rng,
        [[Category.STREETWEAR, Category.FORMAL],
         [Category.STREETWEAR],
         [Category.ATHLETIC],
         [Category.STREETWEAR, Category.ATHLETIC]],
        [0.30, 0.25, 0.22, 0.23])
    return CustomerProfile(
        id=cid, name=name, initials=initials,
        color=_COLORS[Segment.SALE_SHOPPER],
        segment=Segment.SALE_SHOPPER, timing=timing,
        session_hours=_session_hours(rng, timing),
        purchase_freq=round(freq, 1),
        avg_order_value=round(aov),
        categories=cats,
        size_consistent=round(size_c, 2),
        discount_usage=round(discount, 2),
        engagement_score=eng,
    )


def _gen_athletic_regular(rng: random.Random, cid: int, used: set[str]) -> CustomerProfile:
    name, initials = _make_name(rng, used)
    timing = _choice(rng,
        [SessionTiming.MORNING, SessionTiming.LUNCH, SessionTiming.EVENING],
        [0.78, 0.14, 0.08])     # 14% LUNCH → confusable with Lunch Shopper
    aov      = _norm(rng, 130, 22, 72, 205)
    discount = _beta(rng, 2.5, 8, 0.06, 0.42)
    freq     = _norm(rng, 3.8, 0.8, 2.0, 6.2)
    size_c   = _norm(rng, 0.95, 0.04, 0.80, 1.0)
    eng      = int(_norm(rng, 80, 11, 48, 100))
    cats     = _choice(rng,
        [[Category.ATHLETIC],
         [Category.ATHLETIC, Category.STREETWEAR],
         [Category.ATHLETIC, Category.FORMAL]],
        [0.72, 0.18, 0.10])
    return CustomerProfile(
        id=cid, name=name, initials=initials,
        color=_COLORS[Segment.ATHLETIC_REGULAR],
        segment=Segment.ATHLETIC_REGULAR, timing=timing,
        session_hours=_session_hours(rng, timing),
        purchase_freq=round(freq, 1),
        avg_order_value=round(aov),
        categories=cats,
        size_consistent=round(size_c, 2),
        discount_usage=round(discount, 2),
        engagement_score=eng,
    )


def _gen_formal_rare(rng: random.Random, cid: int, used: set[str]) -> CustomerProfile:
    name, initials = _make_name(rng, used)
    timing = _choice(rng,
        [SessionTiming.WEEKEND, SessionTiming.EVENING, SessionTiming.AFTERNOON],
        [0.72, 0.18, 0.10])     # 18% EVENING → confusable with Full-Price
    aov      = _norm(rng, 435, 75, 240, 660)
    discount = _beta(rng, 1.0, 22, 0.0, 0.14)
    freq     = _norm(rng, 0.65, 0.25, 0.28, 1.55)
    size_c   = _norm(rng, 0.97, 0.03, 0.87, 1.0)
    eng      = int(_norm(rng, 50, 10, 23, 76))
    cats     = _choice(rng,
        [[Category.FORMAL],
         [Category.FORMAL, Category.ATHLETIC],
         [Category.FORMAL, Category.STREETWEAR]],
        [0.78, 0.14, 0.08])
    return CustomerProfile(
        id=cid, name=name, initials=initials,
        color=_COLORS[Segment.FORMAL_RARE],
        segment=Segment.FORMAL_RARE, timing=timing,
        session_hours=_session_hours(rng, timing),
        purchase_freq=round(freq, 1),
        avg_order_value=round(aov),
        categories=cats,
        size_consistent=round(size_c, 2),
        discount_usage=round(discount, 2),
        engagement_score=eng,
    )


_GENERATORS = {
    Segment.FULL_PRICE:       _gen_full_price,
    Segment.NIGHT_STREETWEAR: _gen_night_streetwear,
    Segment.LUNCH_SHOPPER:    _gen_lunch_shopper,
    Segment.SALE_SHOPPER:     _gen_sale_shopper,
    Segment.ATHLETIC_REGULAR: _gen_athletic_regular,
    Segment.FORMAL_RARE:      _gen_formal_rare,
}


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_customers(n: int = 300, seed: int = 42) -> list[CustomerProfile]:
    """
    Generate n synthetic customers sampled from overlapping per-segment
    distributions. Segment counts are proportional to SEGMENT_PROPORTIONS.
    The random seed makes results fully reproducible.
    """
    rng = random.Random(seed)
    segments = list(SEGMENT_PROPORTIONS.keys())
    weights  = [SEGMENT_PROPORTIONS[s] for s in segments]
    total_w  = sum(weights)
    weights  = [w / total_w for w in weights]

    # Allocate counts proportionally, giving remainder to last segment
    counts: dict[Segment, int] = {}
    remaining = n
    for seg, w in zip(segments[:-1], weights[:-1]):
        c = round(w * n)
        counts[seg] = c
        remaining -= c
    counts[segments[-1]] = remaining

    customers: list[CustomerProfile] = []
    used_names: set[str] = set()
    cid = 101   # IDs start after the 40 seed profiles

    for seg in segments:
        gen_fn = _GENERATORS[seg]
        for _ in range(counts[seg]):
            customers.append(gen_fn(rng, cid, used_names))
            cid += 1

    rng.shuffle(customers)
    for i, c in enumerate(customers):
        c.id = i + 101

    return customers
