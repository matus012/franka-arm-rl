"""Eval-only contact sensors (they observe, they do not change the MDP).

The table is a static prop, not a rigid body, so contacts are filtered against its collision
*shape* (gotcha 4a.8). Sensors are updated every physics step; the evaluator reads them after
each control step.
"""

from __future__ import annotations

from isaaclab_newton.sensors import ContactSensorCfg

TABLE_SHAPE = "{ENV_REGEX_NS}/Table/Collisions/Cube"
HAND_BODIES = ("panda_hand", "panda_leftfinger", "panda_rightfinger")


def add_eval_sensors(cfg) -> None:
    """Attach hand-vs-table, cube-vs-hand and cube-vs-table contact sensors to an env cfg."""
    cfg.scene.robot.spawn.activate_contact_sensors = True
    # every robot body vs the table (run 2: the clean-grasp rule counts arm-table contact too). The slam
    # metric (section 6) still uses only the hand + finger rows of this sensor.
    cfg.scene.contact_hand_table = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_*",
        filter_shape_prim_expr=[TABLE_SHAPE],
        history_length=0,
    )
    # cube vs hand + fingers + every arm link (the jam audit found cubes carried on the wrist, link6/7)
    cfg.scene.contact_cube_hand = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{b}" for b in HAND_BODIES] + ["{ENV_REGEX_NS}/Robot/panda_link*"],
        history_length=0,
    )
    cfg.scene.contact_cube_table = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        filter_shape_prim_expr=[TABLE_SHAPE],
        history_length=0,
    )
