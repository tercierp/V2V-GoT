"""Unit tests for v2vgotd.resolver."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from v2vgotd.resolver import (
    EgoDecision, OSMContext, FinalDecision,
    resolve, _trajectory_conflict, _SPEED_FALLBACK,
    SPEED_CLASSES, STEERING_CLASSES,
)

# ── Trajectory fixtures ────────────────────────────────────────────────────────
#
# All coordinates are in ego's reference frame: x = forward, z = lateral.
# Conflict threshold default = 3.0 m.
#
# CONVERGING pair: ego moves forward (+x), cav1 moves backward toward ego.
#   Step 0: |2 - 6| = 4.0 m  (no conflict yet)
#   Step 1: |4 - 4| = 0.0 m  (conflict at step 1)
EGO_CONVERGING  = [(2.0, 0.0), (4.0, 0.0), (6.0, 0.0),
                   (8.0, 0.0), (10.0, 0.0), (12.0, 0.0)]
CAV1_CONVERGING = [(6.0, 0.0), (4.0, 0.0), (2.0, 0.0),
                   (0.0, 0.0), (-2.0, 0.0), (-4.0, 0.0)]

# PARALLEL pair: same forward speed, 20 m apart laterally — never conflicts.
EGO_PARALLEL  = [(2.0,  0.0), (4.0,  0.0), (6.0,  0.0),
                 (8.0,  0.0), (10.0, 0.0), (12.0, 0.0)]
CAV1_PARALLEL = [(2.0, 20.0), (4.0, 20.0), (6.0, 20.0),
                 (8.0, 20.0), (10.0, 20.0), (12.0, 20.0)]


def _decision(cav_id: str, speed_idx: int, steering_idx: int,
              trajectory: list) -> EgoDecision:
    return EgoDecision(
        cav_id=cav_id,
        speed_idx=speed_idx,
        steering_idx=steering_idx,
        trajectory=trajectory,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestTrajectoryConflict(unittest.TestCase):

    def test_no_conflict_far_apart(self):
        conflict, min_dist, step = _trajectory_conflict(
            EGO_PARALLEL, CAV1_PARALLEL, threshold_m=3.0)
        self.assertFalse(conflict)
        self.assertAlmostEqual(min_dist, 20.0, places=1)
        self.assertIsNone(step)

    def test_conflict_converging(self):
        conflict, min_dist, step = _trajectory_conflict(
            EGO_CONVERGING, CAV1_CONVERGING, threshold_m=3.0)
        self.assertTrue(conflict)
        self.assertEqual(step, 1)           # first conflict at step 1
        self.assertAlmostEqual(min_dist, 0.0, places=1)

    def test_empty_trajectory_no_conflict(self):
        conflict, min_dist, step = _trajectory_conflict([], EGO_CONVERGING, 3.0)
        self.assertFalse(conflict)
        self.assertEqual(min_dist, float('inf'))
        self.assertIsNone(step)


class TestRule1Agreement(unittest.TestCase):

    def test_full_agreement_parallel_trajectories(self):
        # Same speed + steering, no physical conflict
        ego  = _decision('ego', speed_idx=0, steering_idx=2, trajectory=EGO_PARALLEL)
        cav1 = _decision('1',   speed_idx=0, steering_idx=2, trajectory=CAV1_PARALLEL)
        result = resolve(ego, cav1)
        self.assertEqual(result.rule_applied, 'agreement')
        self.assertFalse(result.conflict_detected)
        self.assertEqual(result.ego_speed_idx,  0)
        self.assertEqual(result.cav1_speed_idx, 0)
        self.assertIsNone(result.yielding_cav)

    def test_agreement_checked_before_conflict(self):
        # Same speed + steering; trajectories actually converge — agreement still fires
        # because Rule 1 is checked first and short-circuits.
        ego  = _decision('ego', speed_idx=2, steering_idx=2, trajectory=EGO_CONVERGING)
        cav1 = _decision('1',   speed_idx=2, steering_idx=2, trajectory=CAV1_CONVERGING)
        result = resolve(ego, cav1)
        self.assertEqual(result.rule_applied, 'agreement')
        self.assertFalse(result.conflict_detected)


class TestNoConflict(unittest.TestCase):

    def test_different_actions_but_no_physical_conflict(self):
        # ego fast+straight, cav1 slow+straight — 20 m apart laterally
        ego  = _decision('ego', speed_idx=0, steering_idx=2, trajectory=EGO_PARALLEL)
        cav1 = _decision('1',   speed_idx=2, steering_idx=2, trajectory=CAV1_PARALLEL)
        result = resolve(ego, cav1)
        self.assertEqual(result.rule_applied, 'no_conflict')
        self.assertFalse(result.conflict_detected)
        # each keeps its own plan
        self.assertEqual(result.ego_speed_idx,  0)
        self.assertEqual(result.cav1_speed_idx, 2)


class TestRule2TTC(unittest.TestCase):
    """TTC rule: faster CAV (lower speed_idx) yields when trajectories conflict."""

    def test_faster_ego_yields(self):
        # ego fast (idx=0), cav1 slow (idx=2) — converging trajectories
        ego  = _decision('ego', speed_idx=0, steering_idx=2, trajectory=EGO_CONVERGING)
        cav1 = _decision('1',   speed_idx=2, steering_idx=2, trajectory=CAV1_CONVERGING)
        result = resolve(ego, cav1)
        self.assertEqual(result.rule_applied, 'ttc')
        self.assertTrue(result.conflict_detected)
        self.assertEqual(result.yielding_cav, 'ego')
        self.assertEqual(result.ego_speed_idx,  _SPEED_FALLBACK[0])  # fast → moderate
        self.assertEqual(result.cav1_speed_idx, 2)                    # slow unchanged

    def test_faster_cav1_yields(self):
        # cav1 fast (idx=0), ego slow (idx=2) — converging trajectories
        ego  = _decision('ego', speed_idx=2, steering_idx=2, trajectory=EGO_CONVERGING)
        cav1 = _decision('1',   speed_idx=0, steering_idx=2, trajectory=CAV1_CONVERGING)
        result = resolve(ego, cav1)
        self.assertEqual(result.rule_applied, 'ttc')
        self.assertEqual(result.yielding_cav, '1')
        self.assertEqual(result.ego_speed_idx,  2)                    # slow unchanged
        self.assertEqual(result.cav1_speed_idx, _SPEED_FALLBACK[0])  # fast → moderate

    def test_speed_fallback_stop_stays_stop(self):
        # stop (idx=4) cannot go lower
        self.assertEqual(_SPEED_FALLBACK[4], 4)

    def test_speed_fallback_is_monotone(self):
        for idx in range(5):
            self.assertGreaterEqual(_SPEED_FALLBACK[idx], idx)


class TestRule3OSM(unittest.TestCase):
    """
    OSM rule fires when: (a) Rule 1 does not apply (different actions),
    (b) trajectories conflict, and (c) both CAVs have the same speed.
    Different steering_idx is used to bypass Rule 1.
    """

    def test_ego_priority_road_cav1_yields(self):
        # Same speed (idx=1), different steering — OSM gives ego priority
        ego  = _decision('ego', speed_idx=1, steering_idx=2, trajectory=EGO_CONVERGING)
        cav1 = _decision('1',   speed_idx=1, steering_idx=1, trajectory=CAV1_CONVERGING)
        osm  = OSMContext(ego_road_priority=2, cav1_road_priority=1)
        result = resolve(ego, cav1, osm=osm)
        self.assertEqual(result.rule_applied, 'osm')
        self.assertTrue(result.conflict_detected)
        self.assertEqual(result.yielding_cav, '1')
        self.assertEqual(result.ego_speed_idx,  1)                    # ego keeps speed
        self.assertEqual(result.cav1_speed_idx, _SPEED_FALLBACK[1])  # cav1 yields

    def test_cav1_priority_road_ego_yields(self):
        ego  = _decision('ego', speed_idx=1, steering_idx=2, trajectory=EGO_CONVERGING)
        cav1 = _decision('1',   speed_idx=1, steering_idx=1, trajectory=CAV1_CONVERGING)
        osm  = OSMContext(ego_road_priority=0, cav1_road_priority=2)
        result = resolve(ego, cav1, osm=osm)
        self.assertEqual(result.rule_applied, 'osm')
        self.assertEqual(result.yielding_cav, 'ego')
        self.assertEqual(result.ego_speed_idx,  _SPEED_FALLBACK[1])
        self.assertEqual(result.cav1_speed_idx, 1)

    def test_equal_priority_falls_through_to_tiebreak(self):
        ego  = _decision('ego', speed_idx=1, steering_idx=2, trajectory=EGO_CONVERGING)
        cav1 = _decision('1',   speed_idx=1, steering_idx=1, trajectory=CAV1_CONVERGING)
        osm  = OSMContext(ego_road_priority=1, cav1_road_priority=1)
        result = resolve(ego, cav1, osm=osm)
        self.assertEqual(result.rule_applied, 'tiebreak')


class TestRule4Tiebreak(unittest.TestCase):

    def test_ego_yields_when_no_osm(self):
        # Same speed, different steering, no OSM — tiebreak: ego yields
        ego  = _decision('ego', speed_idx=1, steering_idx=2, trajectory=EGO_CONVERGING)
        cav1 = _decision('1',   speed_idx=1, steering_idx=1, trajectory=CAV1_CONVERGING)
        result = resolve(ego, cav1, osm=None)
        self.assertEqual(result.rule_applied, 'tiebreak')
        self.assertEqual(result.yielding_cav, 'ego')
        self.assertEqual(result.ego_speed_idx,  _SPEED_FALLBACK[1])
        self.assertEqual(result.cav1_speed_idx, 1)

    def test_tiebreak_is_deterministic(self):
        ego  = _decision('ego', speed_idx=2, steering_idx=2, trajectory=EGO_CONVERGING)
        cav1 = _decision('1',   speed_idx=2, steering_idx=3, trajectory=CAV1_CONVERGING)
        r1 = resolve(ego, cav1)
        r2 = resolve(ego, cav1)
        self.assertEqual(r1.rule_applied,   r2.rule_applied)
        self.assertEqual(r1.yielding_cav,   r2.yielding_cav)
        self.assertEqual(r1.ego_speed_idx,  r2.ego_speed_idx)
        self.assertEqual(r1.cav1_speed_idx, r2.cav1_speed_idx)


class TestFinalDecisionProperties(unittest.TestCase):

    def test_speed_str_properties(self):
        ego  = _decision('ego', speed_idx=0, steering_idx=2, trajectory=EGO_CONVERGING)
        cav1 = _decision('1',   speed_idx=2, steering_idx=2, trajectory=CAV1_CONVERGING)
        result = resolve(ego, cav1)
        self.assertIn(result.ego_speed_str,  SPEED_CLASSES)
        self.assertIn(result.cav1_speed_str, SPEED_CLASSES)

    def test_both_stopped_is_agreement(self):
        ego  = _decision('ego', speed_idx=4, steering_idx=2, trajectory=[(0.0, 0.0)])
        cav1 = _decision('1',   speed_idx=4, steering_idx=2, trajectory=[(0.1, 0.0)])
        result = resolve(ego, cav1)
        # Both stopped + same steering → agreement
        self.assertEqual(result.rule_applied, 'agreement')
        self.assertEqual(result.ego_speed_idx,  4)
        self.assertEqual(result.cav1_speed_idx, 4)

    def test_reason_non_empty_for_all_speeds(self):
        for speed in range(5):
            ego  = _decision('ego', speed_idx=speed,   steering_idx=2,
                             trajectory=EGO_CONVERGING)
            cav1 = _decision('1',   speed_idx=speed,   steering_idx=3,
                             trajectory=CAV1_CONVERGING)
            result = resolve(ego, cav1)
            self.assertGreater(len(result.reason), 0,
                               f'Empty reason for speed_idx={speed}')

    def test_yielding_cav_always_slows_or_stays(self):
        """The yielding CAV's final speed_idx must be >= its input speed_idx."""
        ego  = _decision('ego', speed_idx=0, steering_idx=2, trajectory=EGO_CONVERGING)
        cav1 = _decision('1',   speed_idx=2, steering_idx=2, trajectory=CAV1_CONVERGING)
        result = resolve(ego, cav1)
        if result.yielding_cav == 'ego':
            self.assertGreaterEqual(result.ego_speed_idx,  0)
        elif result.yielding_cav == '1':
            self.assertGreaterEqual(result.cav1_speed_idx, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
