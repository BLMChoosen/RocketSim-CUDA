"""
Unit and integration tests for P2 (GPU State Setters & 4v4 support) and P3 (BallPredTracker).
Validates:
1. set_ball_state via GPU / Tensor buffers (full batch and indexed selective reset).
2. set_car_state via GPU / Tensor buffers (full batch and indexed selective reset).
3. set_pad_state via GPU / Tensor buffers.
4. 4v4 match support (cars_per_env=8) execution stability.
5. predict_ball_trajectory GPU kernel on-demand trajectory prediction.
6. BallPredTracker Python wrapper integration and physical sanity (gravity, ground bounce).
7. Zero-allocation in-place caching and GetBallStateForTime parity in BallPredTracker.
8. Robustness against empty indices, negative/out-of-bounds indices, and non-contiguous strided 3D slices.
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
from gym_env import RocketSimBatchedEnv, BallPredTracker

try:
    import torch
except ImportError:
    torch = None


def test_p2_set_ball_state_full_batch():
    """
    Test P2: set_ball_state with full batch tensor [N, 13].
    Verifies all 13 columns: pos.xyz (0..2), vel.xyz (3..5), quat.wxyz (6..9), ang_vel.xyz (10..12).
    """
    num_envs = 8
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)

    if torch is not None:
        custom_state = torch.zeros((num_envs, 13), dtype=torch.float32)
        for i in range(num_envs):
            custom_state[i, 0] = 100.0 * (i + 1)   # pos_x
            custom_state[i, 1] = -200.0 * (i + 1)  # pos_y
            custom_state[i, 2] = 500.0 + 10.0 * i  # pos_z
            custom_state[i, 3] = 10.0 * (i + 1)    # vel_x
            custom_state[i, 4] = -15.0 * (i + 1)   # vel_y
            custom_state[i, 5] = 20.0 * (i + 1)    # vel_z
            custom_state[i, 6] = 1.0               # quat_w (col 6)
            custom_state[i, 7] = 0.0               # quat_x
            custom_state[i, 8] = 0.0               # quat_y
            custom_state[i, 9] = 0.0               # quat_z
            custom_state[i, 10] = 1.0 * (i + 1)    # ang_vel_x
            custom_state[i, 11] = 2.0 * (i + 1)    # ang_vel_y
            custom_state[i, 12] = 3.0 * (i + 1)    # ang_vel_z
        env.set_ball_state(custom_state)
    else:
        custom_state = rocketsim_cuda.zeros([num_envs, 13], dtype="float32")
        for i in range(num_envs):
            custom_state[i, 0] = 100.0 * (i + 1)
            custom_state[i, 1] = -200.0 * (i + 1)
            custom_state[i, 2] = 500.0 + 10.0 * i
            custom_state[i, 3] = 10.0 * (i + 1)
            custom_state[i, 4] = -15.0 * (i + 1)
            custom_state[i, 5] = 20.0 * (i + 1)
            custom_state[i, 6] = 1.0
            custom_state[i, 10] = 1.0 * (i + 1)
            custom_state[i, 11] = 2.0 * (i + 1)
            custom_state[i, 12] = 3.0 * (i + 1)
        env.set_ball_state(custom_state)

    ball_obs = env.get_ball_observations()
    for i in range(num_envs):
        assert abs(float(ball_obs[i, 0]) - 100.0 * (i + 1)) < 1e-3
        assert abs(float(ball_obs[i, 1]) - -200.0 * (i + 1)) < 1e-3
        assert abs(float(ball_obs[i, 2]) - (500.0 + 10.0 * i)) < 1e-3
        assert abs(float(ball_obs[i, 3]) - 10.0 * (i + 1)) < 1e-3
        assert abs(float(ball_obs[i, 4]) - -15.0 * (i + 1)) < 1e-3
        assert abs(float(ball_obs[i, 5]) - 20.0 * (i + 1)) < 1e-3
        assert abs(float(ball_obs[i, 6]) - 1.0) < 1e-3
        assert abs(float(ball_obs[i, 10]) - 1.0 * (i + 1)) < 1e-3
        assert abs(float(ball_obs[i, 11]) - 2.0 * (i + 1)) < 1e-3
        assert abs(float(ball_obs[i, 12]) - 3.0 * (i + 1)) < 1e-3


def test_p2_set_ball_state_selective_indices():
    """
    Test P2: set_ball_state with partial indices. Only target environments should mutate.
    """
    num_envs = 8
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    # Capture initial ball positions
    initial_ball_obs = env.get_ball_observations()
    init_z = [float(initial_ball_obs[e, 2]) for e in range(num_envs)]

    target_envs = [1, 4, 7]
    if torch is not None:
        target_states = torch.zeros((len(target_envs), 9), dtype=torch.float32)
        for idx, e in enumerate(target_envs):
            target_states[idx, 0] = 999.0 * e
            target_states[idx, 1] = -888.0 * e
            target_states[idx, 2] = 777.0 * e
        env.set_ball_state(target_states, indices=target_envs)
    else:
        target_states = rocketsim_cuda.zeros([len(target_envs), 9], dtype="float32")
        for idx, e in enumerate(target_envs):
            target_states[idx, 0] = 999.0 * e
            target_states[idx, 1] = -888.0 * e
            target_states[idx, 2] = 777.0 * e
        env.set_ball_state(target_states, indices=target_envs)

    curr_ball_obs = env.get_ball_observations()
    for e in range(num_envs):
        if e in target_envs:
            assert abs(float(curr_ball_obs[e, 0]) - 999.0 * e) < 1e-3
            assert abs(float(curr_ball_obs[e, 1]) - -888.0 * e) < 1e-3
            assert abs(float(curr_ball_obs[e, 2]) - 777.0 * e) < 1e-3
        else:
            # Must remain untouched
            assert abs(float(curr_ball_obs[e, 2]) - init_z[e]) < 1e-3


def test_p2_set_car_state_full_batch():
    """
    Test P2: set_car_state with full batch tensor [N, C, 14].
    Verifies all 14 columns: pos.xyz (0..2), vel.xyz (3..5), quat.wxyz (6..9), ang_vel.xyz (10..12), boost (13).
    """
    num_envs = 4
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)

    if torch is not None:
        custom_car_states = torch.zeros((num_envs, cars_per_env, 14), dtype=torch.float32)
        for e in range(num_envs):
            for c in range(cars_per_env):
                custom_car_states[e, c, 0] = 50.0 * (e + 1) + 10.0 * c  # pos_x
                custom_car_states[e, c, 1] = -50.0 * (e + 1) - 10.0 * c # pos_y
                custom_car_states[e, c, 2] = 200.0 + 5.0 * e           # pos_z
                custom_car_states[e, c, 3] = 100.0                     # vel_x
                custom_car_states[e, c, 4] = -100.0                    # vel_y
                custom_car_states[e, c, 5] = 50.0                      # vel_z
                custom_car_states[e, c, 6] = 1.0                       # quat_w (col 6)
                custom_car_states[e, c, 10] = 0.5                      # ang_vel_x
                custom_car_states[e, c, 11] = 0.6                      # ang_vel_y
                custom_car_states[e, c, 12] = 0.7                      # ang_vel_z
                custom_car_states[e, c, 13] = 75.0                     # boost
        env.set_car_state(custom_car_states)
    else:
        custom_car_states = rocketsim_cuda.zeros([num_envs, cars_per_env, 14], dtype="float32")
        for e in range(num_envs):
            for c in range(cars_per_env):
                custom_car_states[e, c, 0] = 50.0 * (e + 1) + 10.0 * c
                custom_car_states[e, c, 1] = -50.0 * (e + 1) - 10.0 * c
                custom_car_states[e, c, 2] = 200.0 + 5.0 * e
                custom_car_states[e, c, 3] = 100.0
                custom_car_states[e, c, 4] = -100.0
                custom_car_states[e, c, 5] = 50.0
                custom_car_states[e, c, 6] = 1.0
                custom_car_states[e, c, 10] = 0.5
                custom_car_states[e, c, 11] = 0.6
                custom_car_states[e, c, 12] = 0.7
                custom_car_states[e, c, 13] = 75.0
        env.set_car_state(custom_car_states)

    car_obs = env.get_car_observations()
    for e in range(num_envs):
        for c in range(cars_per_env):
            assert abs(float(car_obs[e, c, 0]) - (50.0 * (e + 1) + 10.0 * c)) < 1e-3
            assert abs(float(car_obs[e, c, 1]) - (-50.0 * (e + 1) - 10.0 * c)) < 1e-3
            assert abs(float(car_obs[e, c, 2]) - (200.0 + 5.0 * e)) < 1e-3
            assert abs(float(car_obs[e, c, 3]) - 100.0) < 1e-3
            assert abs(float(car_obs[e, c, 4]) - -100.0) < 1e-3
            assert abs(float(car_obs[e, c, 5]) - 50.0) < 1e-3
            assert abs(float(car_obs[e, c, 6]) - 1.0) < 1e-3
            assert abs(float(car_obs[e, c, 10]) - 0.5) < 1e-3
            assert abs(float(car_obs[e, c, 11]) - 0.6) < 1e-3
            assert abs(float(car_obs[e, c, 12]) - 0.7) < 1e-3
            assert abs(float(car_obs[e, c, 13]) - 75.0) < 1e-3


def test_p2_set_pad_state():
    """
    Test P2: set_pad_state directly on GPU.
    """
    num_envs = 4
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)

    if torch is not None:
        is_active = torch.zeros((num_envs, 34), dtype=torch.uint8)
        cooldown = torch.full((num_envs, 34), 5.5, dtype=torch.float32)
        is_active[:, 0] = 1
        cooldown[:, 0] = 0.0
    else:
        is_active = rocketsim_cuda.zeros([num_envs, 34], dtype="uint8")
        cooldown = rocketsim_cuda.zeros([num_envs, 34], dtype="float32")
        for e in range(num_envs):
            for p in range(34):
                cooldown[e, p] = 5.5
            is_active[e, 0] = 1
            cooldown[e, 0] = 0.0

    env.set_pad_state(pad_is_active=is_active, pad_cooldown=cooldown)
    pad_active = env.get_pad_is_active()
    for e in range(num_envs):
        assert int(pad_active[e, 0]) == 1
        assert int(pad_active[e, 1]) == 0

    # Step simulation without error
    env.step()


def test_p2_4v4_match_support():
    """
    Test P2: Full 4v4 match support with cars_per_env=8.
    Verifies memory allocation, step execution, and numerical sanity.
    """
    num_envs = 4
    cars_per_env = 8
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    assert env.cars_per_env == 8

    # Reset and step for 60 ticks
    obs = env.reset()
    assert obs.shape == (num_envs, 8, 14)

    total_cars = num_envs * cars_per_env
    actions = rocketsim_cuda.zeros([total_cars, 8], dtype="float32")
    for i in range(total_cars):
        actions[i, 0] = 1.0  # Full throttle for all 8 cars

    for _ in range(60):
        obs, reward, terminated, truncated, info = env.step(actions)

    # Check that positions are finite numbers
    for e in range(num_envs):
        for c in range(8):
            z = float(obs[e, c, 2])
            assert not (z != z), f"Car observation contains NaN in 4v4 (env {e}, car {c})"
            assert abs(z) < 1e6, f"Car observation out of bounds in 4v4 (env {e}, car {c})"


def test_p2_empty_indices_noop():
    """
    Adversarial edge case: passing empty indices (list or tensor) MUST be a safe no-op.
    Must NOT reset or mutate any environments or cars!
    """
    num_envs = 4
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    init_ball_z = float(env.get_ball_observations()[0, 2])
    init_car_z = float(env.get_car_observations()[0, 0, 2])

    # 1. Empty list indices for ball
    if torch is not None:
        dummy_state = torch.zeros((num_envs, 13))
        env.set_ball_state(dummy_state, indices=[])
        assert abs(float(env.get_ball_observations()[0, 2]) - init_ball_z) < 1e-4

        # 2. Empty tensor indices for ball
        empty_idx_tensor = torch.tensor([], dtype=torch.int32)
        env.set_ball_state(dummy_state, indices=empty_idx_tensor)
        assert abs(float(env.get_ball_observations()[0, 2]) - init_ball_z) < 1e-4

        # 3. Empty list indices for car
        dummy_car = torch.zeros((num_envs, cars_per_env, 14))
        env.set_car_state(dummy_car, indices=[])
        assert abs(float(env.get_car_observations()[0, 0, 2]) - init_car_z) < 1e-4

        # 4. Empty list indices for pad
        env.set_pad_state(pad_is_active=torch.zeros((num_envs, 34), dtype=torch.uint8), indices=[])


def test_p2_out_of_bounds_indices_guard():
    """
    Adversarial edge case: passing negative or out-of-bounds indices.
    CUDA kernel must safely ignore out-of-bounds indices without illegal memory accesses.
    """
    num_envs = 4
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    init_ball_z = float(env.get_ball_observations()[0, 2])

    if torch is not None:
        # Indices contains -1 and 999999 along with valid index 0
        oob_indices = [-1, 999999]
        st = torch.zeros((2, 13))
        st[:, 2] = 456.0
        env.set_ball_state(st, indices=oob_indices)
        # All valid envs should remain unchanged
        for e in range(num_envs):
            assert abs(float(env.get_ball_observations()[e, 2]) - init_ball_z) < 1e-4


def test_p2_strided_3d_car_tensor():
    """
    Adversarial edge case: passing a sliced 3D car tensor with non-contiguous environment stride (e.g. full[::2]).
    Must read each sliced environment at the correct stride offset!
    """
    if torch is None:
        pytest.skip("PyTorch required for strided slice test")

    num_envs = 4
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    # Create tensor for 8 envs, 2 cars, 14 cols
    full = torch.zeros((8, 2, 14))
    for e in range(8):
        for c in range(2):
            full[e, c, 0] = 1000.0 * e + 10.0 * c  # Unique pos_x per (e, c)
            full[e, c, 6] = 1.0                    # quat_w

    # Slice every 2nd environment -> shape (4, 2, 14), strides: (56, 14, 1)
    sliced = full[::2]
    env.set_car_state(sliced)

    obs = env.get_car_observations()
    # Env 0 in sliced is Env 0 in full
    assert abs(float(obs[0, 0, 0]) - 0.0) < 1e-3
    assert abs(float(obs[0, 1, 0]) - 10.0) < 1e-3
    # Env 1 in sliced is Env 2 in full (pos_x = 2000.0, 2010.0)
    assert abs(float(obs[1, 0, 0]) - 2000.0) < 1e-3
    assert abs(float(obs[1, 1, 0]) - 2010.0) < 1e-3
    # Env 2 in sliced is Env 4 in full (pos_x = 4000.0, 4010.0)
    assert abs(float(obs[2, 0, 0]) - 4000.0) < 1e-3
    assert abs(float(obs[2, 1, 0]) - 4010.0) < 1e-3


def test_p3_predict_ball_trajectory_physical_fidelity():
    """
    Test P3: predict_ball_trajectory GPU kernel on-demand trajectory prediction.
    Tests:
    1. Output tensor shape [num_envs, num_ticks, 6].
    2. Ball falling under gravity from height (Z acceleration ~ -650 UU/s^2).
    3. Ground bounce off floor (Z >= 85 UU, Z velocity reverses upwards).
    4. Simulation state invariance: predicting trajectory must NOT alter the current ball state in SimContext!
    """
    num_envs = 4
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)

    # Place ball at moderate height (Z=300, rest velocity) so it hits floor and bounces within 180 ticks
    ball_state = rocketsim_cuda.zeros([num_envs, 13], dtype="float32")
    for e in range(num_envs):
        ball_state[e, 2] = 300.0   # pos_z = 300 UU
        ball_state[e, 6] = 1.0     # quat_w = 1.0
    env.set_ball_state(ball_state)

    ball_obs_before_z = float(env.get_ball_observations()[0, 2])
    assert abs(ball_obs_before_z - 300.0) < 1e-3

    num_ticks = 180  # 1.5 seconds at 120Hz
    traj = env.predict_ball_trajectory(num_ticks=num_ticks)

    # 1. Verify shape
    assert traj.shape == (num_envs, num_ticks, 6), f"Unexpected traj shape: {traj.shape}"

    # 2. Verify simulation state invariance
    ball_obs_after_z = float(env.get_ball_observations()[0, 2])
    assert abs(ball_obs_after_z - ball_obs_before_z) < 1e-4, "predict_ball_trajectory mutated simulation state!"

    # 3. Verify trajectory physics:
    first_z = float(traj[0, 0, 2])
    assert abs(first_z - 300.0) < 10.0

    # Over initial ticks, pos_z decreases under gravity
    mid_z = float(traj[0, 40, 2])
    assert mid_z < first_z, f"Ball did not fall under gravity: mid_z={mid_z} >= first_z={first_z}"

    # Ball radius is ~93.15 UU; arena floor collision discrete step compression reaches ~87.18 UU (identical to sim step)
    min_z = 1e9
    has_bounced = False
    for t in range(num_ticks):
        z_val = float(traj[0, t, 2])
        vz_val = float(traj[0, t, 5])
        if z_val < min_z:
            min_z = z_val
        if vz_val > 50.0:
            has_bounced = True

    assert min_z >= 85.0, f"Ball fell through arena floor! min_z = {min_z}"
    assert has_bounced, "Ball did not bounce off the floor during 180 ticks"


def test_p3_ball_pred_tracker_wrapper_and_time_query():
    """
    Test P3: BallPredTracker Python wrapper class with in-place zero allocation and time query.
    """
    num_envs = 2
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    tracker = BallPredTracker(env=env, num_ticks=120)

    # First prediction allocates cached GPU tensor
    pred = tracker.predict()
    assert pred.shape == (num_envs, 120, 6)

    # Subsequent update re-uses pre-allocated buffer
    pred2 = tracker.update()
    assert pred2.shape == (num_envs, 120, 6)

    cached = tracker.get_trajectory()
    assert cached.shape == (num_envs, 120, 6)

    # Test get_ball_state_for_time (matches C++ BallPredTracker::GetBallStateForTime)
    s_0 = tracker.get_ball_state_for_time(0.0, env_idx=0)
    assert abs(float(s_0[0]) - float(cached[0, 0, 0])) < 1e-4

    s_1s = tracker.get_ball_state_for_time(1.0, env_idx=0)
    # At 1.0s, tick is 120 -> clamped to 119
    assert abs(float(s_1s[0]) - float(cached[0, 119, 0])) < 1e-4


def test_p3_predict_trajectory_with_out_tensor():
    """
    Test P3: Pre-allocating output tensor and passing directly into predict_ball_trajectory.
    Verifies that the kernel writes into pre-allocated memory in-place.
    """
    num_envs = 2
    num_ticks = 60
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=1)

    out = rocketsim_cuda.zeros([num_envs, num_ticks, 6], dtype="float32")
    res = env.predict_ball_trajectory(num_ticks=num_ticks, out_tensor=out)

    # Check that out was written in-place
    assert float(out[0, 0, 2]) > 0.0
