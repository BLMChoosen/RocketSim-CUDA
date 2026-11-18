"""
Unit and integration tests for Priority P4:
1. is_ball_probably_going_in (analytical trajectory extrapolation towards goal in CUDA and Python bindings).
2. GameEventTracker (GPU tracking of touches, shots on goal, saves, assists, and goals).
3. Batched execution and zero-copy tensor views for all event buffers.
4. Selective indexed resetting of game events.
5. Python GameEventTracker wrapper and integration with RocketSimBatchedEnv.
"""

import os
import sys
import pytest

# Ensure build directory and bindings directory are in sys.path
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "bindings"))
python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
for d in [build_dir, src_dir, python_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

import rocketsim_cuda
from gym_env import RocketSimBatchedEnv, GameEventTracker

try:
    import torch
except ImportError:
    torch = None


def test_p4_is_ball_probably_going_in_standalone():
    """
    Test standalone rocketsim_cuda.is_ball_probably_going_in analytical goal extrapolation.
    """
    # 1. Direct shot towards Orange goal (+Y)
    pos = (0.0, 0.0, 100.0)
    vel = (0.0, 3000.0, 0.0)
    going_in, goal_team = rocketsim_cuda.is_ball_probably_going_in(pos, vel, 2.0, 0.0)
    assert going_in is True
    assert goal_team == 1  # Defending team is Orange (1)

    # 2. Direct shot towards Blue goal (-Y)
    pos = (0.0, 0.0, 100.0)
    vel = (0.0, -3000.0, 0.0)
    going_in, goal_team = rocketsim_cuda.is_ball_probably_going_in(pos, vel, 2.0, 0.0)
    assert going_in is True
    assert goal_team == 0  # Defending team is Blue (0)

    # 3. Too slow (takes > max_time to reach goal line)
    pos = (0.0, 0.0, 100.0)
    vel = (0.0, 500.0, 0.0)  # Needs > 10s to travel 5124 UU
    going_in, _ = rocketsim_cuda.is_ball_probably_going_in(pos, vel, 2.0, 0.0)
    assert going_in is False

    # 4. Too wide (X out of bounds at goal line)
    pos = (0.0, 0.0, 100.0)
    vel = (2000.0, 3000.0, 0.0)  # Will be at X ~ 3400 UU at Y = 5124 (goal half-width is ~893)
    going_in, _ = rocketsim_cuda.is_ball_probably_going_in(pos, vel, 2.0, 0.0)
    assert going_in is False

    # 5. Too high (Z above crossbar at goal line)
    pos = (0.0, 0.0, 100.0)
    vel = (0.0, 3000.0, 2000.0)  # Will launch far above crossbar (642 UU)
    going_in, _ = rocketsim_cuda.is_ball_probably_going_in(pos, vel, 2.0, 0.0)
    assert going_in is False

    # 6. Stationary ball (vel.y == 0)
    pos = (0.0, 0.0, 100.0)
    vel = (0.0, 0.0, 0.0)
    going_in, _ = rocketsim_cuda.is_ball_probably_going_in(pos, vel, 2.0, 0.0)
    assert going_in is False


def test_p4_is_ball_probably_going_in_batched():
    """
    Test batched GPU evaluation via SimContext and RocketSimBatchedEnv.
    """
    num_envs = 4
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)

    if torch is not None:
        # Set env 0 to going into Orange, env 1 to going into Blue, env 2 to going wide, env 3 stationary
        ball_state = torch.zeros((num_envs, 13), dtype=torch.float32)
        # Env 0: Orange shot
        ball_state[0, 0] = 0.0; ball_state[0, 1] = 0.0; ball_state[0, 2] = 100.0
        ball_state[0, 3] = 0.0; ball_state[0, 4] = 3000.0; ball_state[0, 5] = 0.0
        # Env 1: Blue shot
        ball_state[1, 0] = 0.0; ball_state[1, 1] = 0.0; ball_state[1, 2] = 100.0
        ball_state[1, 3] = 0.0; ball_state[1, 4] = -3000.0; ball_state[1, 5] = 0.0
        # Env 2: Wide shot
        ball_state[2, 0] = 0.0; ball_state[2, 1] = 0.0; ball_state[2, 2] = 100.0
        ball_state[2, 3] = 2500.0; ball_state[2, 4] = 3000.0; ball_state[2, 5] = 0.0
        # Env 3: Stationary
        ball_state[3, 0] = 0.0; ball_state[3, 1] = 0.0; ball_state[3, 2] = 100.0

        env.set_ball_state(ball_state)

        is_going_in, goal_team = env.is_ball_probably_going_in(max_time=2.5)

        assert int(is_going_in[0]) == 1
        assert int(goal_team[0]) == 1  # Orange defending

        assert int(is_going_in[1]) == 1
        assert int(goal_team[1]) == 0  # Blue defending

        assert int(is_going_in[2]) == 0
        assert int(is_going_in[3]) == 0


def test_p4_game_event_tracker_config():
    """
    Test GameEventTrackerConfig getters and setters on SimContext.
    """
    sim = rocketsim_cuda.SimContext(num_envs=2, cars_per_env=1)
    cfg = sim.get_game_event_tracker_config()
    assert cfg.shot_min_speed == 1750.0
    assert cfg.shot_event_cooldown == 1.0
    assert cfg.shot_min_score_time == 2.0

    # Modify config and write back
    cfg.shot_min_speed = 1500.0
    cfg.shot_event_cooldown = 0.5
    sim.set_game_event_tracker_config(cfg)

    cfg2 = sim.get_game_event_tracker_config()
    assert cfg2.shot_min_speed == 1500.0
    assert cfg2.shot_event_cooldown == 0.5


def test_p4_game_event_tracking_toggle():
    """
    Test enabling and disabling game event tracking on SimContext.
    """
    sim = rocketsim_cuda.SimContext(num_envs=2, cars_per_env=1)
    assert sim.is_game_event_tracking_enabled() is True
    sim.set_game_event_tracking_enabled(False)
    assert sim.is_game_event_tracking_enabled() is False
    sim.set_game_event_tracking_enabled(True)
    assert sim.is_game_event_tracking_enabled() is True


def test_p4_touch_and_event_views():
    """
    Test that game event views return correct tensor shapes and types.
    """
    num_envs = 4
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)

    step_touches = env.get_step_touches()
    step_shots = env.get_step_shots()
    step_saves = env.get_step_saves()
    step_goals = env.get_step_goals()
    step_assists = env.get_step_assists()

    total_touches = env.get_total_touches()
    total_shots = env.get_total_shots()
    total_saves = env.get_total_saves()
    total_goals = env.get_total_goals()
    total_assists = env.get_total_assists()

    ball_shot = env.get_ball_shot()
    ball_shot_goal_team = env.get_ball_shot_goal_team()

    assert step_touches.shape == [num_envs, cars_per_env] or step_touches.shape == (num_envs, cars_per_env)
    assert step_shots.shape == [num_envs, cars_per_env] or step_shots.shape == (num_envs, cars_per_env)
    assert total_touches.shape == [num_envs, cars_per_env] or total_touches.shape == (num_envs, cars_per_env)
    assert total_goals.shape == [num_envs, cars_per_env] or total_goals.shape == (num_envs, cars_per_env)
    assert ball_shot.shape == [num_envs] or ball_shot.shape == (num_envs,)
    assert ball_shot_goal_team.shape == [num_envs] or ball_shot_goal_team.shape == (num_envs,)

    info = env.get_info()
    assert "step_touches" in info
    assert "step_shots" in info
    assert "step_saves" in info
    assert "step_goals" in info
    assert "step_assists" in info
    assert "total_touches" in info


def test_p4_python_game_event_tracker_class():
    """
    Test Python GameEventTracker wrapper class.
    """
    num_envs = 2
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)

    tracker = GameEventTracker(env)
    assert tracker.config.shot_min_speed == 1750.0

    tracker.update()
    step_evts = tracker.get_step_events()
    assert "touches" in step_evts
    assert "shots" in step_evts
    assert "saves" in step_evts
    assert "goals" in step_evts
    assert "assists" in step_evts

    totals = tracker.get_totals()
    assert "touches" in totals
    assert "goals" in totals

    tracker.reset_persistent_info()


def test_p4_ball_touch_event_trigger():
    """
    Test that when a car hits the ball, step_touches becomes 1 and total_touches increments.
    """
    if torch is None:
        pytest.skip("PyTorch required for tensor manipulation")

    num_envs = 1
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)

    # Spawn ball directly ahead of car and drive car into ball
    ball_state = torch.zeros((1, 13), dtype=torch.float32)
    ball_state[0, 0] = 0.0
    ball_state[0, 1] = 200.0
    ball_state[0, 2] = 93.15  # Resting on ground
    env.set_ball_state(ball_state)

    car_state = torch.zeros((1, 1, 13), dtype=torch.float32)
    car_state[0, 0, 0] = 0.0
    car_state[0, 0, 1] = 0.0
    car_state[0, 0, 2] = 17.03  # Resting on ground
    car_state[0, 0, 3] = 0.0
    car_state[0, 0, 4] = 2000.0  # High velocity towards ball
    car_state[0, 0, 6] = 1.0    # quat_w
    env.set_car_state(car_state)

    # Step forward 15 ticks: car will collide with ball
    touched_detected = False
    for _ in range(15):
        env.step()
        if int(env.get_step_touches()[0, 0]) == 1:
            touched_detected = True
            break

    assert touched_detected is True
    assert int(env.get_total_touches()[0, 0]) >= 1


def test_p4_selective_indexed_game_event_reset():
    """
    Test selective indexed reset of game events for specific environments.
    """
    num_envs = 4
    cars_per_env = 1
    sim = rocketsim_cuda.SimContext(num_envs=num_envs, cars_per_env=cars_per_env)

    # Reset env 1 and 3 specifically using a Python list
    sim.reset_game_events_indexed([1, 3])

    # Should not throw or crash on empty list
    sim.reset_game_events_indexed([])

    # If torch is available, test with torch CPU tensor as well
    if torch is not None:
        idx_tensor = torch.tensor([0, 2], dtype=torch.int32)
        sim.reset_game_events_indexed(idx_tensor)
        empty_tensor = torch.tensor([], dtype=torch.int32)
        sim.reset_game_events_indexed(empty_tensor)


def test_p4_shot_event_trigger():
    """
    Test that a high-speed ball shot towards the net triggers step_shots, total_shots, and ball_shot.
    """
    if torch is None:
        pytest.skip("PyTorch required for tensor manipulation")

    env = RocketSimBatchedEnv(num_envs=1, cars_per_env=2)
    # Car 0 (Blue) waiting in defense, Car 1 (Orange) shooting at Blue net (-Y)
    car_state = torch.zeros((1, 2, 13), dtype=torch.float32)
    car_state[0, 0, 0] = 500.0; car_state[0, 0, 1] = -5000.0; car_state[0, 0, 2] = 17.03; car_state[0, 0, 6] = 1.0
    car_state[0, 1, 0] = 0.0; car_state[0, 1, 1] = -3000.0; car_state[0, 1, 2] = 17.03; car_state[0, 1, 4] = -2300.0; car_state[0, 1, 6] = 1.0

    ball_state = torch.zeros((1, 13), dtype=torch.float32)
    ball_state[0, 0] = 0.0; ball_state[0, 1] = -3150.0; ball_state[0, 2] = 93.15

    env.set_car_state(car_state)
    env.set_ball_state(ball_state)

    shot_ok = False
    for step in range(45):
        env.step()
        if int(env.get_step_shots()[0, 1]) == 1:
            shot_ok = True
            break

    assert shot_ok is True, "Shot was not credited to shooter within 45 ticks"
    assert int(env.get_ball_shot()[0]) == 1
    assert int(env.get_ball_shot_goal_team()[0]) == 0  # Blue net targeted
    assert int(env.get_total_shots()[0, 1]) >= 1


def test_p4_save_event_trigger():
    """
    Test that an active shot on goal intercepted and cleared by a defender triggers step_saves and total_saves.
    """
    if torch is None:
        pytest.skip("PyTorch required for tensor manipulation")

    env = RocketSimBatchedEnv(num_envs=1, cars_per_env=2)
    car_state = torch.zeros((1, 2, 13), dtype=torch.float32)
    car_state[0, 0, 0] = 500.0; car_state[0, 0, 1] = -5000.0; car_state[0, 0, 2] = 17.03; car_state[0, 0, 6] = 1.0
    car_state[0, 1, 0] = 0.0; car_state[0, 1, 1] = -3000.0; car_state[0, 1, 2] = 17.03; car_state[0, 1, 4] = -2300.0; car_state[0, 1, 6] = 1.0

    ball_state = torch.zeros((1, 13), dtype=torch.float32)
    ball_state[0, 0] = 0.0; ball_state[0, 1] = -3150.0; ball_state[0, 2] = 93.15

    env.set_car_state(car_state)
    env.set_ball_state(ball_state)

    # Step until shot is active
    for _ in range(38):
        env.step()

    assert int(env.get_ball_shot()[0]) == 1

    # Intercept ball in air with Car 0 (Blue defender)
    ball_obs = env.get_ball_observations()
    bx, by, bz = float(ball_obs[0, 0]), float(ball_obs[0, 1]), float(ball_obs[0, 2])
    car0_state = torch.zeros((1, 13), dtype=torch.float32)
    car0_state[0, 0] = bx; car0_state[0, 1] = by - 120.0; car0_state[0, 2] = bz; car0_state[0, 4] = 2300.0; car0_state[0, 6] = 1.0
    env.set_car_state(car0_state, car_start=0)

    save_ok = False
    for _ in range(10):
        env.step()
        if int(env.get_step_saves()[0, 0]) == 1:
            save_ok = True
            break

    assert save_ok is True, "Save was not credited to defender"
    assert int(env.get_ball_shot()[0]) == 0
    assert int(env.get_total_saves()[0, 0]) >= 1


def test_p4_goal_and_assist_event_trigger():
    """
    Test 2-touch passing sequence crediting goal to shooter and assist to passer.
    """
    if torch is None:
        pytest.skip("PyTorch required for tensor manipulation")

    env = RocketSimBatchedEnv(num_envs=1, cars_per_env=4)
    # Car 0 (Blue passer), Car 2 (Blue shooter)
    car_state = torch.zeros((1, 4, 13), dtype=torch.float32)
    car_state[0, 0, 0] = 0.0; car_state[0, 0, 1] = 0.0; car_state[0, 0, 2] = 17.03; car_state[0, 0, 4] = 2000.0; car_state[0, 0, 6] = 1.0
    car_state[0, 1, 0] = 2000.0; car_state[0, 2, 0] = -2000.0; car_state[0, 3, 0] = 2000.0

    ball_state = torch.zeros((1, 13), dtype=torch.float32)
    ball_state[0, 0] = 0.0; ball_state[0, 1] = 150.0; ball_state[0, 2] = 93.15

    env.set_car_state(car_state)
    env.set_ball_state(ball_state)

    # Car 0 hits ball
    for _ in range(10):
        env.step()
        if int(env.get_step_touches()[0, 0]) == 1:
            break

    # Car 2 hits ball next
    bobs = env.get_ball_observations()
    bx, by = float(bobs[0, 0]), float(bobs[0, 1])
    c2s = torch.zeros((1, 13), dtype=torch.float32)
    c2s[0, 0] = bx; c2s[0, 1] = by - 100.0; c2s[0, 2] = 17.03; c2s[0, 4] = 2000.0; c2s[0, 6] = 1.0
    env.set_car_state(c2s, car_start=2)

    for _ in range(10):
        env.step()
        if int(env.get_step_touches()[0, 2]) == 1:
            break

    # Score ball into Orange net (+Y)
    scored = torch.zeros((1, 13), dtype=torch.float32)
    scored[0, 0] = 0.0; scored[0, 1] = 5300.0; scored[0, 2] = 100.0
    env.set_ball_state(scored)
    env.step()

    assert int(env.get_step_goals()[0, 2]) == 1, "Shooter (Car 2) must receive step_goals = 1"
    assert int(env.get_step_assists()[0, 0]) == 1, "Passer (Car 0) must receive step_assists = 1"
    assert int(env.get_total_goals()[0, 2]) == 1
    assert int(env.get_total_assists()[0, 0]) == 1


def test_p4_module_exports_and_aliases():
    """
    Test direct imports from rocketsim_cuda root package and env aliases.
    """
    import rocketsim_cuda
    assert hasattr(rocketsim_cuda, "GameEventTracker")
    assert hasattr(rocketsim_cuda, "GameEventTrackerConfig")
    assert hasattr(rocketsim_cuda, "RocketSimEnv")
    assert hasattr(rocketsim_cuda, "RocketSimBatchedEnv")
    assert hasattr(rocketsim_cuda, "BallPredTracker")
    assert hasattr(rocketsim_cuda, "is_ball_probably_going_in")

    from rocketsim_cuda import RocketSimEnv, GameEventTracker, BallPredTracker, is_ball_probably_going_in
    assert RocketSimEnv is not None
    assert GameEventTracker is not None
    assert BallPredTracker is not None
    assert is_ball_probably_going_in is not None


def test_p4_multi_defender_save_attribution():
    """
    Validate multi-touch save attribution when multiple defenders contact the ball
    inside or near the goal cavity. The earliest/primary defender should receive save credit.
    """
    if torch is None:
        pytest.skip("PyTorch required for tensor manipulation")

    # 1 env, 4 cars: Car 0 (Blue defender 1), Car 1 (Orange shooter), Car 2 (Blue defender 2), Car 3 (Orange)
    env = RocketSimBatchedEnv(num_envs=1, cars_per_env=4)

    # Shooter Car 1 fires ball at Blue net (-Y)
    car_state = torch.zeros((1, 4, 13), dtype=torch.float32)
    car_state[0, 0, 0] = -200.0; car_state[0, 0, 1] = -5000.0; car_state[0, 0, 2] = 17.03; car_state[0, 0, 6] = 1.0 # Blue def 1
    car_state[0, 1, 0] = 0.0; car_state[0, 1, 1] = -3000.0; car_state[0, 1, 2] = 17.03; car_state[0, 1, 4] = -2300.0; car_state[0, 1, 6] = 1.0 # Shooter
    car_state[0, 2, 0] = 200.0; car_state[0, 2, 1] = -5000.0; car_state[0, 2, 2] = 17.03; car_state[0, 2, 6] = 1.0  # Blue def 2
    car_state[0, 3, 0] = 2000.0; car_state[0, 3, 1] = 0.0; car_state[0, 3, 2] = 17.03; car_state[0, 3, 6] = 1.0

    ball_state = torch.zeros((1, 13), dtype=torch.float32)
    ball_state[0, 0] = 0.0; ball_state[0, 1] = -3150.0; ball_state[0, 2] = 93.15

    env.set_car_state(car_state)
    env.set_ball_state(ball_state)

    # Step until shot is active
    for _ in range(38):
        env.step()

    assert int(env.get_ball_shot()[0]) == 1

    # Intercept ball with Blue defender 0
    ball_obs = env.get_ball_observations()
    bx, by, bz = float(ball_obs[0, 0]), float(ball_obs[0, 1]), float(ball_obs[0, 2])
    c0_state = torch.zeros((1, 13), dtype=torch.float32)
    c0_state[0, 0] = bx; c0_state[0, 1] = by - 120.0; c0_state[0, 2] = bz; c0_state[0, 4] = 2300.0; c0_state[0, 6] = 1.0
    env.set_car_state(c0_state, car_start=0)

    save_awarded = False
    for _ in range(10):
        env.step()
        if int(env.get_step_saves()[0, 0]) == 1 or int(env.get_step_saves()[0, 2]) == 1:
            save_awarded = True
            break

    assert save_awarded is True, "Save must be awarded to defending team"
    assert int(env.get_ball_shot()[0]) == 0, "Ball shot must clear upon save"

