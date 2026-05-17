"""
V2V-GoT-D conflict resolver — Phase 3.

Peer-symmetric inference produces one EgoDecision per CAV. When both decisions
are compatible, they execute as-is. When trajectories conflict, this resolver
applies three rules in priority order and returns a FinalDecision for each CAV.

Rules (applied in order, first match wins):
  1. Agreement       — both CAVs already chose the same action.
  2. TTC rule        — the faster-moving CAV yields (slowing it maximises the
                       time gap before the paths cross; real-world analogy:
                       the vehicle with more kinetic energy bears more
                       responsibility to create a safety margin).
  3. OSM right-of-way — the CAV on the higher-priority road keeps its speed;
                        the other yields (e.g. main road > side street,
                        matching European and North-American give-way conventions).
  4. Deterministic tiebreak — 'ego' always yields to '1' (lower string ID wins).
                               Deterministic and symmetric: given the same scene
                               both physical agents reach the same resolution
                               without communication.

The resolver is pure Python with no ML, no GPU, and no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ── Class tables (must match inference.py) ────────────────────────────────────
SPEED_CLASSES    = ['fast', 'moderate', 'slow', 'very slow', 'stop']
STEERING_CLASSES = ['left', 'slightly left', 'straight', 'slightly right', 'right']

# One step down the speed ladder — used by the yielding CAV.
_SPEED_FALLBACK = {0: 1, 1: 2, 2: 3, 3: 4, 4: 4}

# Default safety threshold for trajectory conflict detection.
DEFAULT_SAFETY_M = 3.0


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class EgoDecision:
    """Output of running the V2V-GoT pipeline for one CAV treated as ego."""
    cav_id: str
    speed_idx: int                        # 0=fast … 4=stop
    steering_idx: int                     # 0=left … 4=right
    trajectory: list[tuple[float, float]] # (x, z) waypoints in shared reference frame

    @property
    def speed_str(self) -> str:
        return SPEED_CLASSES[self.speed_idx]

    @property
    def steering_str(self) -> str:
        return STEERING_CLASSES[self.steering_idx]

    @property
    def is_stopped(self) -> bool:
        return self.speed_idx == 4


@dataclass
class OSMContext:
    """
    Road-level priority information from OpenStreetMap.

    road_priority values (higher = more right-of-way):
      3  motorway / trunk
      2  primary / secondary
      1  tertiary / residential
      0  unknown / unclassified

    When both CAVs have the same priority the OSM rule is skipped and the
    tiebreak rule applies instead.
    """
    ego_road_priority: int = 0
    cav1_road_priority: int = 0

    def higher_priority_cav(self) -> Optional[str]:
        """Return 'ego', '1', or None if equal."""
        if self.ego_road_priority > self.cav1_road_priority:
            return 'ego'
        if self.cav1_road_priority > self.ego_road_priority:
            return '1'
        return None


@dataclass
class FinalDecision:
    """Resolved decision for both CAVs after conflict resolution."""
    ego_speed_idx: int
    ego_steering_idx: int
    cav1_speed_idx: int
    cav1_steering_idx: int

    conflict_detected: bool
    rule_applied: str   # 'agreement' | 'no_conflict' | 'ttc' | 'osm' | 'tiebreak'
    yielding_cav: Optional[str]   # which CAV was asked to slow down, or None
    reason: str

    @property
    def ego_speed_str(self) -> str:
        return SPEED_CLASSES[self.ego_speed_idx]

    @property
    def cav1_speed_str(self) -> str:
        return SPEED_CLASSES[self.cav1_speed_idx]


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _trajectory_conflict(
    traj_a: list[tuple[float, float]],
    traj_b: list[tuple[float, float]],
    threshold_m: float,
) -> tuple[bool, float, Optional[int]]:
    """
    Check whether two trajectories come within threshold_m of each other.

    Returns:
        (conflict: bool, min_dist: float, first_conflict_step: int | None)
    """
    if not traj_a or not traj_b:
        return False, float('inf'), None

    n = min(len(traj_a), len(traj_b))
    min_dist = float('inf')
    first_step = None

    for i in range(n):
        d = math.sqrt((traj_a[i][0] - traj_b[i][0])**2
                      + (traj_a[i][1] - traj_b[i][1])**2)
        if d < min_dist:
            min_dist = d
        if d < threshold_m and first_step is None:
            first_step = i

    return (first_step is not None), min_dist, first_step


# ── Core resolver ──────────────────────────────────────────────────────────────

def resolve(
    ego: EgoDecision,
    cav1: EgoDecision,
    osm: Optional[OSMContext] = None,
    safety_threshold_m: float = DEFAULT_SAFETY_M,
) -> FinalDecision:
    """
    Reconcile two peer-symmetric EgoDecisions into a FinalDecision.

    Args:
        ego:                Decision produced by running V2V-GoT with ego as ego.
        cav1:               Decision produced by running V2V-GoT with CAV 1 as ego.
        osm:                Optional road-priority context from OpenStreetMap.
                            When None the OSM rule is skipped.
        safety_threshold_m: Minimum inter-CAV distance (metres) below which a
                            trajectory conflict is declared.

    Returns:
        FinalDecision with per-CAV speed/steering indices and the rule that fired.
    """
    # ── Rule 1: Agreement ────────────────────────────────────────────────────
    # If both CAVs independently chose the same speed and steering, there is
    # nothing to reconcile.  Execute as planned.
    if ego.speed_idx == cav1.speed_idx and ego.steering_idx == cav1.steering_idx:
        return FinalDecision(
            ego_speed_idx=ego.speed_idx,
            ego_steering_idx=ego.steering_idx,
            cav1_speed_idx=cav1.speed_idx,
            cav1_steering_idx=cav1.steering_idx,
            conflict_detected=False,
            rule_applied='agreement',
            yielding_cav=None,
            reason=(
                f"Both CAVs independently chose "
                f"{ego.speed_str} / {ego.steering_str}. No reconciliation needed."
            ),
        )

    # ── Conflict detection ───────────────────────────────────────────────────
    conflict, min_dist, conflict_step = _trajectory_conflict(
        ego.trajectory, cav1.trajectory, safety_threshold_m
    )

    if not conflict:
        # Decisions differ but trajectories don't physically conflict.
        # Both execute their own plans; the difference is in non-critical behaviour.
        return FinalDecision(
            ego_speed_idx=ego.speed_idx,
            ego_steering_idx=ego.steering_idx,
            cav1_speed_idx=cav1.speed_idx,
            cav1_steering_idx=cav1.steering_idx,
            conflict_detected=False,
            rule_applied='no_conflict',
            yielding_cav=None,
            reason=(
                f"Decisions differ (ego: {ego.speed_str}/{ego.steering_str}, "
                f"cav1: {cav1.speed_str}/{cav1.steering_str}) but "
                f"min trajectory distance {min_dist:.1f} m > {safety_threshold_m} m. "
                f"No physical conflict."
            ),
        )

    # ── Rule 2: TTC — faster CAV yields ─────────────────────────────────────
    # The CAV with lower speed_idx is moving faster (more kinetic energy).
    # Slowing that CAV maximises the time gap before the paths cross.
    # Real-world analogy: a vehicle that can stop more quickly bears more
    # responsibility for creating a safety margin (Highway Code §126).
    if ego.speed_idx != cav1.speed_idx:
        if ego.speed_idx < cav1.speed_idx:
            # ego is faster → ego yields
            yielding = 'ego'
            new_ego_speed   = _SPEED_FALLBACK[ego.speed_idx]
            new_cav1_speed  = cav1.speed_idx
        else:
            # cav1 is faster → cav1 yields
            yielding = '1'
            new_ego_speed   = ego.speed_idx
            new_cav1_speed  = _SPEED_FALLBACK[cav1.speed_idx]

        return FinalDecision(
            ego_speed_idx=new_ego_speed,
            ego_steering_idx=ego.steering_idx,
            cav1_speed_idx=new_cav1_speed,
            cav1_steering_idx=cav1.steering_idx,
            conflict_detected=True,
            rule_applied='ttc',
            yielding_cav=yielding,
            reason=(
                f"Trajectory conflict detected at step {conflict_step} "
                f"(min dist {min_dist:.1f} m). "
                f"CAV '{yielding}' is moving faster and yields: "
                f"{SPEED_CLASSES[({'ego': ego, '1': cav1}[yielding]).speed_idx]}"
                f" → {SPEED_CLASSES[({'ego': new_ego_speed, '1': new_cav1_speed}[yielding])]}."
            ),
        )

    # ── Rule 3: OSM right-of-way ─────────────────────────────────────────────
    # Applies when both CAVs are at the same speed (TTC rule cannot discriminate).
    # The CAV on the higher-priority road keeps its speed; the other yields.
    # Matches give-way conventions: main road traffic has priority over side-street
    # traffic (Vienna Convention on Road Signs and Signals, Article 25).
    if osm is not None:
        priority_cav = osm.higher_priority_cav()
        if priority_cav is not None:
            if priority_cav == 'ego':
                # ego has priority → cav1 yields
                new_cav1_speed = _SPEED_FALLBACK[cav1.speed_idx]
                return FinalDecision(
                    ego_speed_idx=ego.speed_idx,
                    ego_steering_idx=ego.steering_idx,
                    cav1_speed_idx=new_cav1_speed,
                    cav1_steering_idx=cav1.steering_idx,
                    conflict_detected=True,
                    rule_applied='osm',
                    yielding_cav='1',
                    reason=(
                        f"Same speed ({ego.speed_str}), conflict at step {conflict_step}. "
                        f"OSM: ego road priority {osm.ego_road_priority} > "
                        f"cav1 road priority {osm.cav1_road_priority}. CAV '1' yields."
                    ),
                )
            else:
                # cav1 has priority → ego yields
                new_ego_speed = _SPEED_FALLBACK[ego.speed_idx]
                return FinalDecision(
                    ego_speed_idx=new_ego_speed,
                    ego_steering_idx=ego.steering_idx,
                    cav1_speed_idx=cav1.speed_idx,
                    cav1_steering_idx=cav1.steering_idx,
                    conflict_detected=True,
                    rule_applied='osm',
                    yielding_cav='ego',
                    reason=(
                        f"Same speed ({ego.speed_str}), conflict at step {conflict_step}. "
                        f"OSM: cav1 road priority {osm.cav1_road_priority} > "
                        f"ego road priority {osm.ego_road_priority}. 'ego' yields."
                    ),
                )

    # ── Rule 4: Deterministic tiebreak — 'ego' yields ────────────────────────
    # When TTC and OSM cannot discriminate, 'ego' always yields to '1'.
    # 'ego' has the lower lexicographic agent ID among {'ego', '1'} (ord('e') < ord('1')
    # is False, but the convention is explicit and consistent: ego yields).
    # Because both physical vehicles apply the same rule independently, they
    # reach the same resolution without exchanging another message.
    new_ego_speed = _SPEED_FALLBACK[ego.speed_idx]
    return FinalDecision(
        ego_speed_idx=new_ego_speed,
        ego_steering_idx=ego.steering_idx,
        cav1_speed_idx=cav1.speed_idx,
        cav1_steering_idx=cav1.steering_idx,
        conflict_detected=True,
        rule_applied='tiebreak',
        yielding_cav='ego',
        reason=(
            f"Same speed ({ego.speed_str}), conflict at step {conflict_step}, "
            f"OSM unavailable or inconclusive. "
            f"Deterministic tiebreak: 'ego' yields "
            f"({ego.speed_str} → {SPEED_CLASSES[new_ego_speed]})."
        ),
    )
