#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "rocketsim_cuda/config.h"

namespace rocketsim_cuda {

/**
 * @brief Configuration parameters for GameEventTracker.
 * Strictly matches RocketSim CPU GameEventTrackerConfig.
 */
struct GameEventTrackerConfig {
    float shot_min_speed = 1750.0f;          // Minimum ball speed (UU/s) towards net to be considered a shot
    float shot_touch_min_delay = 0.3f;       // Time (s) since shooting car last touched ball
    float pred_score_extra_margin = 0.0f;    // Added margin for predicting if ball is scoring
    float shot_event_cooldown = 1.0f;        // Minimum time (s) between shot events
    float shot_min_score_time = 2.0f;        // Max extrapolation time (s) for shot prediction
    float goal_max_touch_time = 4.0f;        // Max time (s) between touch and goal to count as scorer
    float pass_max_touch_time = 2.0f;        // Max time (s) between passer touch and shooter touch
};

/**
 * @brief Device SoA structure for GameEventTracker environment-level persistent tracking state.
 * Slices are allocated per-environment in SimContext (length = num_envs, 128-byte aligned).
 */
struct GameEventTrackerEnvStateSoA {
    uint8_t* __restrict__ ball_shot = nullptr;           // [num_envs] 1 if ball is currently predicted as shot towards a goal
    uint8_t* __restrict__ ball_shot_goal_team = nullptr; // [num_envs] defending team (0 = BLUE net, 1 = ORANGE net)
    float*   __restrict__ shot_cooldown = nullptr;       // [num_envs] shot event cooldown timer in seconds
    uint8_t* __restrict__ ball_scored_last = nullptr;    // [num_envs] 1 if ball was scored on the previous update/tick
};

/**
 * @brief Device SoA structure for GameEventTracker car-level events.
 * Slices are allocated per-car in SimContext (length = num_envs * cars_per_env, 128-byte aligned).
 */
struct GameEventTrackerCarEventsSoA {
    // Step events (updated per simulation step)
    uint8_t*  __restrict__ step_touches = nullptr; // 1 if car touched the ball during this step
    uint8_t*  __restrict__ step_shots = nullptr;   // 1 if car executed a shot on goal this step
    uint8_t*  __restrict__ step_saves = nullptr;   // 1 if car made a save this step
    uint8_t*  __restrict__ step_goals = nullptr;   // 1 if car scored a goal this step
    uint8_t*  __restrict__ step_assists = nullptr; // 1 if car assisted a goal this step

    // Cumulative episode/session totals
    uint32_t* __restrict__ total_touches = nullptr; // Lifetime touches
    uint32_t* __restrict__ total_shots = nullptr;   // Lifetime shots on goal
    uint32_t* __restrict__ total_saves = nullptr;   // Lifetime saves
    uint32_t* __restrict__ total_goals = nullptr;   // Lifetime goals scored
    uint32_t* __restrict__ total_assists = nullptr; // Lifetime assists
};

} // namespace rocketsim_cuda
