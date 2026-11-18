#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include "config.h"
#include "types/car_state.cuh"
#include "types/ball_state.cuh"
#include "types/car_controls.cuh"
#include "types/arena_state.cuh"
#include "types/car_config.cuh"
#include "types/arena_config.cuh"
#include "types/game_event_tracker.cuh"

namespace rocketsim_cuda {

class SimContext {
public:
    explicit SimContext(uint32_t num_envs = 1, uint32_t cars_per_env = 1, cudaStream_t stream = nullptr);
    ~SimContext();

    // Disable copy, allow move
    SimContext(const SimContext&) = delete;
    SimContext& operator=(const SimContext&) = delete;
    SimContext(SimContext&& o) noexcept;
    SimContext& operator=(SimContext&& o) noexcept;

    uint32_t GetNumEnvs() const { return m_num_envs; }
    uint32_t GetCarsPerEnv() const { return m_cars_per_env; }
    uint32_t GetTotalCars() const { return m_total_cars; }
    size_t GetAllocatedBytes() const { return m_allocated_bytes; }
    cudaStream_t GetStream() const { return m_stream; }

    size_t GetBallPitchFloats() const;
    size_t GetCarPitchFloats() const;

    const BallStateSoA& GetBallState() const { return m_ball_state; }
    BallStateSoA& GetBallState() { return m_ball_state; }

    const CarStateSoA& GetCarState() const { return m_car_state; }
    CarStateSoA& GetCarState() { return m_car_state; }

    const ArenaStateSoA& GetArenaState() const { return m_arena_state; }
    ArenaStateSoA& GetArenaState() { return m_arena_state; }

    const CarConfigSoA& GetCarConfig() const { return m_car_config; }
    CarConfigSoA& GetCarConfig() { return m_car_config; }

    const MutatorConfigSoA& GetMutatorConfig() const { return m_mutator_config; }
    MutatorConfigSoA& GetMutatorConfig() { return m_mutator_config; }

    const ArenaConfigSoA& GetArenaConfig() const { return m_arena_config; }
    ArenaConfigSoA& GetArenaConfig() { return m_arena_config; }

    const float* GetRewards() const { return m_arena_state.rewards; }
    float* GetRewards() { return m_arena_state.rewards; }

    const uint8_t* GetTerminated() const { return m_arena_state.terminated; }
    uint8_t* GetTerminated() { return m_arena_state.terminated; }

    const uint8_t* GetTruncated() const { return m_arena_state.truncated; }
    uint8_t* GetTruncated() { return m_arena_state.truncated; }

    const CarControlsSoA& GetControls() const { return m_controls; }
    CarControlsSoA& GetControls() { return m_controls; }

    // State initialization kernels
    void ResetToDefault();

    // Asynchronous GPU selective resets (no host synchronization barriers)
    void ResetEnvironmentsIndexed(const int32_t* d_env_indices, uint32_t num_resets);
    void ResetEnvironmentsIndexed(const int64_t* d_env_indices, uint32_t num_resets);
    void ResetEnvironmentsMasked(const uint8_t* d_reset_mask);

    // State transfer helpers for differential testing & serialization
    void CopyBallStateToHost(BallStatePOD* host_out, uint32_t env_start = 0, uint32_t count = 0);
    void CopyCarStateToHost(CarStatePOD* host_out, uint32_t car_start = 0, uint32_t count = 0);
    void CopyBallStateToDevice(const BallStatePOD* host_in, uint32_t env_start = 0, uint32_t count = 0);
    void CopyCarStateToDevice(const CarStatePOD* host_in, uint32_t car_start = 0, uint32_t count = 0);
    void CopyControlsToDevice(const CarControls* host_in, uint32_t car_start = 0, uint32_t count = 0);

    // GPU Tensor State Setters (Zero-Copy VRAM Resets)
    void SetBallStateFromTensor(
        const float* d_tensor,
        uint32_t count = 0,
        uint32_t start_env = 0,
        const int32_t* d_indices = nullptr,
        uint32_t num_cols = 13,
        int64_t stride_env = 13,
        int64_t stride_col = 1);

    void SetCarStateFromTensor(
        const float* d_tensor,
        uint32_t count = 0,
        uint32_t start_car = 0,
        const int32_t* d_indices = nullptr,
        uint32_t num_cols = 14,
        int64_t stride_env = 0,
        int64_t stride_car = 14,
        int64_t stride_col = 1);

    void SetPadStateFromTensor(
        const uint8_t* d_active = nullptr,
        int64_t stride_env_active = MAX_BOOST_PADS,
        int64_t stride_pad_active = 1,
        const float* d_cooldown = nullptr,
        int64_t stride_env_cd = MAX_BOOST_PADS,
        int64_t stride_pad_cd = 1,
        uint32_t count = 0,
        uint32_t start_env = 0,
        const int32_t* d_indices = nullptr);

    // On-Demand Dedicated Ball Trajectory Prediction Kernel
    void PredictBallTrajectory(
        float* d_out,
        uint32_t num_ticks,
        uint32_t num_cols = 6,
        const float* d_init_state = nullptr,
        uint32_t init_cols = 0,
        int64_t stride_init_env = 0,
        int64_t stride_init_col = 1,
        int64_t stride_out_env = 0,
        int64_t stride_out_tick = 0,
        int64_t stride_out_col = 1);

    // Game Event Tracking (Shots on goal, saves, assists, goals, touches)
    const GameEventTrackerConfig& GetGameEventTrackerConfig() const { return m_event_tracker_config; }
    void SetGameEventTrackerConfig(const GameEventTrackerConfig& cfg) { m_event_tracker_config = cfg; }
    bool IsGameEventTrackingEnabled() const { return m_event_tracking_enabled; }
    void SetGameEventTrackingEnabled(bool enabled) { m_event_tracking_enabled = enabled; }
    const GameEventTrackerEnvStateSoA& GetGameEventEnvState() const { return m_event_env_state; }
    GameEventTrackerEnvStateSoA& GetGameEventEnvState() { return m_event_env_state; }
    const GameEventTrackerCarEventsSoA& GetGameEventCarEvents() const { return m_event_car_events; }
    GameEventTrackerCarEventsSoA& GetGameEventCarEvents() { return m_event_car_events; }
    void UpdateGameEvents(float delta_time = DELTA_TIME, uint32_t delta_ticks = 1);
    void ResetGameEvents(int32_t env_idx = -1);
    void ResetGameEventsIndexed(const int32_t* d_env_indices, uint32_t num_resets);

    // Goal Extrapolation (IsBallProbablyGoingIn)
    void EvaluateIsBallProbablyGoingIn(float max_time = 2.0f, float extra_margin = 0.0f);
    const uint8_t* GetIsBallProbablyGoingIn() const { return m_d_is_ball_probably_going_in; }
    uint8_t* GetIsBallProbablyGoingIn() { return m_d_is_ball_probably_going_in; }
    const int32_t* GetBallProbablyGoingInGoalTeam() const { return m_d_ball_probably_going_in_goal_team; }
    int32_t* GetBallProbablyGoingInGoalTeam() { return m_d_ball_probably_going_in_goal_team; }

    // Simulation Step
    void Step(uint32_t batch_size = 0, const float* actions_tensor = nullptr);

private:
    void AllocateArena();
    void FreeArena();

    uint32_t m_num_envs = 0;
    uint32_t m_cars_per_env = 0;
    uint32_t m_total_cars = 0;
    size_t m_allocated_bytes = 0;
    void* m_d_pool = nullptr;
    cudaStream_t m_stream = nullptr;

    BallStateSoA m_ball_state;
    CarStateSoA m_car_state;
    ArenaStateSoA m_arena_state;
    CarControlsSoA m_controls;
    CarConfigSoA m_car_config;
    MutatorConfigSoA m_mutator_config;
    ArenaConfigSoA m_arena_config;

    // Game Event Tracking & Goal Prediction
    GameEventTrackerConfig m_event_tracker_config;
    bool m_event_tracking_enabled = true;
    GameEventTrackerEnvStateSoA m_event_env_state;
    GameEventTrackerCarEventsSoA m_event_car_events;
    uint8_t* m_d_is_ball_probably_going_in = nullptr;
    int32_t* m_d_ball_probably_going_in_goal_team = nullptr;
};

// Global Simulation Step Entry Point (GEMINI.md Section 6.2)
void sim_step_batch(SimContext* ctx, uint32_t batch_size = 0, const CarControlsSoA* controls = nullptr, const float* actions_tensor = nullptr);

} // namespace rocketsim_cuda
