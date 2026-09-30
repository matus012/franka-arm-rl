# Video index

All videos:
- 1280x720 per view, real time (50 fps, one frame per 20 ms control step)
- Newton renderer; the blue plane is the table top
- phase name overlaid (HOVER / DESCEND / CLOSE / LIFT / CARRY / HOLD, "(grasp)" once the grasp is confirmed)
- the target drawn as a wireframe cube: magenta, turning green once the cube center is within 3 cm

## Final policy (phase 6: random cube pose, target, mass and friction)
**3x3 grid, close-up:** `videos/p6/p6_closeup_grid_3x3.mp4` (GIF: `videos/final/readme_grid.gif`).
- 9 episodes, all success + clean.
- Each: the hand comes down vertically over its cube, the reflex closes the fingers on two faces, and the cube is
  lifted and carried into the green target box by ~2 s, then held to 8 s.

**Single success, close-up:** `videos/p6/p6_single_success_closeup.mp4`.

**Single success, wide:** `videos/final/final_single_success_wide.mp4`. Same sequence from a wide camera:
- hover over the cube (0-0.7 s), vertical descent, lift (~1.2 s), carry (~1.8 s)
- hold in the green box from ~2.2 s to 8 s

**No near-miss / failure singles.** Failures are 0.5 % of test episodes (5 of 1,000: 4 never left HOVER, 1 stopped in
DESCEND; 6 episodes dropped and re-grasped). None occurred in the recorded episodes.

## Comparisons
**Scripted vs RL:** `videos/final/scripted_vs_rl_side_by_side.mp4`. Left: the hand-written controller (+ reflex);
right: the RL policy. Same world and episode. Both succeed; the RL policy is about twice as fast:
- at 1.22 s the scripted arm is still descending while RL is lifting
- at 3.02 s the scripted arm is lifting while RL is in HOLD

**Heavy, slippery cube:** `videos/final/twist_heavy_side_by_side.mp4`. Left: phase 5 (trained without mass/friction
randomization); right: phase 6 (with it). Cube 0.5 kg, friction 0.5, 4 episodes each (2x2).
- This is the grid's hardest cell (14.5 % vs 31.5 % success).
- On each side one episode ends in the green box; the others drop the cube and retry.
- The reflex's fixed 3 mm squeeze barely holds this cube; see RESULTS.md.

## Curriculum
**Phase 2 -> 6:** `videos/curriculum/curriculum_phase2_to_6.mp4`. The first 4 s of one episode per phase, each policy
on its own world, with a title bar:
- fixed world
- + cube position
- + yaw (the hand turns to the faces)
- + random target
- + mass/friction

Every clip shows descend -> lift -> hold in the green box.

## Audit frames
`results/audit/<phase>_env0_phase<k>_<name>_steps_*.png`: >= 3 close-up frames per phase for phases 2-6 (prefixes
`s16_phase2_`, `p3_`, `p4_`, `p5_`, `p6_`), each described in words in the report.
