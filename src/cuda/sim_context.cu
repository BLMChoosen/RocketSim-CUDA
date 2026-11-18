#include "rocketsim_cuda/sim_context.cuh"
#include "rocketsim_cuda/physics/ball_dynamics.cuh"
#include "rocketsim_cuda/physics/arena_sdf.cuh"
#include <stdexcept>
#include <iostream>

namespace rocketsim_cuda {

namespace {

inline size_t align_128(size_t size) {
    return (size + 127) & ~size_t(127);
}

__device__ inline void init_single_ball(BallStateSoA& ball_state, uint32_t idx) {
    ball_state.pos_x[idx] = 0.0f;
    ball_state.pos_y[idx] = 0.0f;
    ball_state.pos_z[idx] = BALL_REST_Z;

    ball_state.vel_x[idx] = 0.0f;
    ball_state.vel_y[idx] = 0.0f;
    ball_state.vel_z[idx] = 0.0f;

    ball_state.q_w[idx] = 1.0f;
    ball_state.q_x[idx] = 0.0f;
    ball_state.q_y[idx] = 0.0f;
    ball_state.q_z[idx] = 0.0f;

    ball_state.ang_vel_x[idx] = 0.0f;
    ball_state.ang_vel_y[idx] = 0.0f;
    ball_state.ang_vel_z[idx] = 0.0f;
}

struct KickoffSpawn {
    float x, y, z;
    float qw, qx, qy, qz;
};

// 5 Standard Soccar kickoff spawns for Blue team (facing +Y towards Orange goal, RLConst.h:355-362)
__device__ constexpr KickoffSpawn BLUE_KICKOFF_SPAWNS[5] = {
    // 0: Diagonal Left (yaw = pi/4, 45 deg)
    { -2048.0f, -2560.0f, 17.0f, 0.9238795f, 0.0f, 0.0f,  0.3826834f },
    // 1: Diagonal Right (yaw = 3*pi/4, 135 deg)
    {  2048.0f, -2560.0f, 17.0f, 0.3826834f, 0.0f, 0.0f,  0.9238795f },
    // 2: Off-Center Left (yaw = pi/2, 90 deg)
    {  -256.0f, -3840.0f, 17.0f, 0.7071068f, 0.0f, 0.0f,  0.7071068f },
    // 3: Off-Center Right (yaw = pi/2, 90 deg)
    {   256.0f, -3840.0f, 17.0f, 0.7071068f, 0.0f, 0.0f,  0.7071068f },
    // 4: Goalie / Center (yaw = pi/2, 90 deg)
    {     0.0f, -4608.0f, 17.0f, 0.7071068f, 0.0f, 0.0f,  0.7071068f }
};

// 5 Standard Soccar kickoff spawns for Orange team (X -> -X, Y -> -Y, yaw -> yaw + PI)
__device__ constexpr KickoffSpawn ORANGE_KICKOFF_SPAWNS[5] = {
    // 0: Diagonal Right (mirrored from Blue 0: X=2048, Y=2560, yaw = -3*pi/4)
    {  2048.0f,  2560.0f, 17.0f, 0.3826834f, 0.0f, 0.0f, -0.9238795f },
    // 1: Diagonal Left (mirrored from Blue 1: X=-2048, Y=2560, yaw = -pi/4)
    { -2048.0f,  2560.0f, 17.0f, 0.9238795f, 0.0f, 0.0f, -0.3826834f },
    // 2: Off-Center Right (mirrored from Blue 2: X=256, Y=3840, yaw = -pi/2)
    {   256.0f,  3840.0f, 17.0f, 0.7071068f, 0.0f, 0.0f, -0.7071068f },
    // 3: Off-Center Left (mirrored from Blue 3: X=-256, Y=3840, yaw = -pi/2)
    {  -256.0f,  3840.0f, 17.0f, 0.7071068f, 0.0f, 0.0f, -0.7071068f },
    // 4: Goalie / Center (mirrored from Blue 4: X=0, Y=4608, yaw = -pi/2)
    {     0.0f,  4608.0f, 17.0f, 0.7071068f, 0.0f, 0.0f, -0.7071068f }
};

__device__ inline uint32_t get_kickoff_slot(uint32_t env_idx, uint32_t seed) {
    uint32_t x = env_idx ^ (seed * 0x9E3779B9u);
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = (x >> 16) ^ x;
    return x % 5;
}

__device__ inline void init_single_car(
    CarStateSoA& car_state,
    uint32_t idx,
    uint32_t env_idx,
    uint32_t car_in_env_idx,
    uint32_t cars_per_env,
    uint32_t seed = 0)
{
    uint8_t team = (cars_per_env > 1) ? (car_in_env_idx % 2) : 0;
    if (car_state.team) car_state.team[idx] = team;
    if (car_state.hitbox_type) car_state.hitbox_type[idx] = 0;
    uint32_t team_car_idx = (cars_per_env > 1) ? (car_in_env_idx / 2) : car_in_env_idx;
    uint32_t base_slot = get_kickoff_slot(env_idx, seed);
    uint32_t spawn_slot = (base_slot + team_car_idx) % 5;

    const KickoffSpawn& spawn = (team == 0) ? BLUE_KICKOFF_SPAWNS[spawn_slot] : ORANGE_KICKOFF_SPAWNS[spawn_slot];

    car_state.pos_x[idx] = spawn.x;
    car_state.pos_y[idx] = spawn.y;
    car_state.pos_z[idx] = spawn.z;

    car_state.vel_x[idx] = 0.0f;
    car_state.vel_y[idx] = 0.0f;
    car_state.vel_z[idx] = 0.0f;

    car_state.q_w[idx] = spawn.qw;
    car_state.q_x[idx] = spawn.qx;
    car_state.q_y[idx] = spawn.qy;
    car_state.q_z[idx] = spawn.qz;

    car_state.ang_vel_x[idx] = 0.0f;
    car_state.ang_vel_y[idx] = 0.0f;
    car_state.ang_vel_z[idx] = 0.0f;

    car_state.boost[idx] = BOOST_SPAWN;
    car_state.time_since_boosted[idx] = 0.0f;
    car_state.boosting_time[idx] = 0.0f;
    car_state.is_boosting[idx] = 0;

    car_state.is_on_ground[idx] = 1;
    car_state.wheel_contact_0[idx] = 1;
    car_state.wheel_contact_1[idx] = 1;
    car_state.wheel_contact_2[idx] = 1;
    car_state.wheel_contact_3[idx] = 1;

    car_state.suspension_length_0[idx] = 0.0f;
    car_state.suspension_length_1[idx] = 0.0f;
    car_state.suspension_length_2[idx] = 0.0f;
    car_state.suspension_length_3[idx] = 0.0f;

    car_state.has_jumped[idx] = 0;
    car_state.is_jumping[idx] = 0;
    car_state.jump_time[idx] = 0.0f;
    car_state.has_double_jumped[idx] = 0;
    car_state.air_time[idx] = 0.0f;
    car_state.air_time_since_jump[idx] = 0.0f;

    car_state.has_flipped[idx] = 0;
    car_state.is_flipping[idx] = 0;
    car_state.flip_time[idx] = 0.0f;
    car_state.flip_rel_torque_x[idx] = 0.0f;
    car_state.flip_rel_torque_y[idx] = 0.0f;
    car_state.flip_rel_torque_z[idx] = 0.0f;

    car_state.is_auto_flipping[idx] = 0;
    car_state.auto_flip_timer[idx] = 0.0f;
    car_state.auto_flip_torque_scale[idx] = 0.0f;
    car_state.handbrake_val[idx] = 0.0f;

    car_state.is_supersonic[idx] = 0;
    car_state.supersonic_time[idx] = 0.0f;
    car_state.is_demoed[idx] = 0;
    car_state.demo_respawn_timer[idx] = 0.0f;

    car_state.world_contact_has_contact[idx] = 0;
    car_state.world_contact_normal_x[idx] = 0.0f;
    car_state.world_contact_normal_y[idx] = 0.0f;
    car_state.world_contact_normal_z[idx] = 0.0f;
    car_state.car_contact_other_car_id[idx] = -1;
    car_state.car_contact_cooldown_timer[idx] = 0.0f;

    car_state.ball_hit_is_valid[idx] = 0;
    car_state.ball_hit_rel_pos_x[idx] = 0.0f;
    car_state.ball_hit_rel_pos_y[idx] = 0.0f;
    car_state.ball_hit_rel_pos_z[idx] = 0.0f;
    car_state.ball_hit_extra_hit_force_x[idx] = 0.0f;
    car_state.ball_hit_extra_hit_force_y[idx] = 0.0f;
    car_state.ball_hit_extra_hit_force_z[idx] = 0.0f;
    car_state.ball_hit_tick_count[idx] = 0;
    if (car_state.ball_touched) car_state.ball_touched[idx] = 0;

    car_state.last_controls_throttle[idx] = 0.0f;
    car_state.last_controls_steer[idx] = 0.0f;
    car_state.last_controls_pitch[idx] = 0.0f;
    car_state.last_controls_yaw[idx] = 0.0f;
    car_state.last_controls_roll[idx] = 0.0f;
    car_state.last_controls_boost[idx] = 0;
    car_state.last_controls_jump[idx] = 0;
    car_state.last_controls_handbrake[idx] = 0;

    car_state.wheel_engine_force[idx] = 0.0f;
    car_state.wheel_brake[idx] = 0.0f;
    car_state.wheel_steer_angle[idx] = 0.0f;
    car_state.wheel_lat_friction_0[idx] = 0.0f;
    car_state.wheel_lat_friction_1[idx] = 0.0f;
    car_state.wheel_lat_friction_2[idx] = 0.0f;
    car_state.wheel_lat_friction_3[idx] = 0.0f;
    car_state.wheel_long_friction_0[idx] = 0.0f;
    car_state.wheel_long_friction_1[idx] = 0.0f;
    car_state.wheel_long_friction_2[idx] = 0.0f;
    car_state.wheel_long_friction_3[idx] = 0.0f;

    car_state.pos_bt_x[idx] = car_state.pos_x[idx] * 0.02f;
    car_state.pos_bt_y[idx] = car_state.pos_y[idx] * 0.02f;
    car_state.pos_bt_z[idx] = car_state.pos_z[idx] * 0.02f;
    car_state.vel_bt_x[idx] = car_state.vel_x[idx] * 0.02f;
    car_state.vel_bt_y[idx] = car_state.vel_y[idx] * 0.02f;
    car_state.vel_bt_z[idx] = car_state.vel_z[idx] * 0.02f;
}

__device__ inline void init_single_arena(ArenaStateSoA& arena_state, uint32_t env_idx, uint32_t cars_per_env = 1) {
    if (arena_state.is_goal) arena_state.is_goal[env_idx] = 0;
    if (arena_state.scoring_team) arena_state.scoring_team[env_idx] = 0;
    if (arena_state.is_out_of_bounds) arena_state.is_out_of_bounds[env_idx] = 0;
    if (arena_state.tick_count) arena_state.tick_count[env_idx] = 0;
    if (arena_state.terminated) arena_state.terminated[env_idx] = 0;
    if (arena_state.truncated) arena_state.truncated[env_idx] = 0;
    if (arena_state.rewards) {
        for (uint32_t c = 0; c < cars_per_env; ++c) {
            arena_state.rewards[env_idx * cars_per_env + c] = 0.0f;
        }
    }
    for (uint32_t p = 0; p < MAX_BOOST_PADS; ++p) {
        uint32_t pad_idx = env_idx * MAX_BOOST_PADS + p;
        if (arena_state.pad_is_active) arena_state.pad_is_active[pad_idx] = 1;
        if (arena_state.pad_cooldown) arena_state.pad_cooldown[pad_idx] = 0.0f;
    }
}

__global__ void k_init_ball_state(BallStateSoA ball_state, uint32_t count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    init_single_ball(ball_state, idx);
}

__global__ void k_init_car_state(CarStateSoA car_state, uint32_t count, uint32_t cars_per_env, uint32_t seed = 0) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t cpe = (cars_per_env > 0) ? cars_per_env : 1;
    uint32_t env_idx = idx / cpe;
    uint32_t car_in_env_idx = idx % cpe;
    init_single_car(car_state, idx, env_idx, car_in_env_idx, cars_per_env, seed);
}

__global__ void k_init_arena_state(ArenaStateSoA arena_state, uint32_t count, uint32_t cars_per_env) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    init_single_arena(arena_state, idx, cars_per_env);
}

template <typename TIndex>
__global__ void k_reset_environments_indexed(
    const TIndex* __restrict__ env_indices,
    uint32_t num_resets,
    uint32_t cars_per_env,
    BallStateSoA ball_state,
    CarStateSoA car_state,
    ArenaStateSoA arena_state,
    uint32_t seed = 0)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_resets) return;

    int64_t env_idx = env_indices[i];
    if (env_idx < 0) return;

    uint32_t e = static_cast<uint32_t>(env_idx);
    init_single_ball(ball_state, e);
    init_single_arena(arena_state, e, cars_per_env);

    for (uint32_t c = 0; c < cars_per_env; ++c) {
        uint32_t car_idx = e * cars_per_env + c;
        init_single_car(car_state, car_idx, e, c, cars_per_env, seed);
    }
}

__global__ void k_reset_environments_masked(
    const uint8_t* __restrict__ reset_mask,
    uint32_t num_envs,
    uint32_t cars_per_env,
    BallStateSoA ball_state,
    CarStateSoA car_state,
    ArenaStateSoA arena_state,
    uint32_t seed = 0)
{
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_envs) return;
    if (!reset_mask[e]) return;

    init_single_ball(ball_state, e);
    init_single_arena(arena_state, e, cars_per_env);

    for (uint32_t c = 0; c < cars_per_env; ++c) {
        uint32_t car_idx = e * cars_per_env + c;
        init_single_car(car_state, car_idx, e, c, cars_per_env, seed);
    }
}

__global__ void k_reset_game_events(
    uint32_t num_envs,
    GameEventTrackerEnvStateSoA env_state,
    int32_t target_env)
{
    uint32_t env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx >= num_envs) return;
    if (target_env >= 0 && static_cast<int32_t>(env_idx) != target_env) return;

    env_state.ball_shot[env_idx] = 0;
    env_state.ball_shot_goal_team[env_idx] = 0;
    env_state.shot_cooldown[env_idx] = 0.0f;
    env_state.ball_scored_last[env_idx] = 0;
}

template <typename TIndex>
__global__ void k_reset_game_events_indexed(
    const TIndex* __restrict__ d_indices,
    uint32_t num_resets,
    uint32_t num_envs,
    GameEventTrackerEnvStateSoA env_state)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_resets) return;

    int64_t env_idx = static_cast<int64_t>(d_indices[idx]);
    if (env_idx < 0 || static_cast<uint32_t>(env_idx) >= num_envs) return;

    env_state.ball_shot[env_idx] = 0;
    env_state.ball_shot_goal_team[env_idx] = 0;
    env_state.shot_cooldown[env_idx] = 0.0f;
    env_state.ball_scored_last[env_idx] = 0;
}

__global__ void k_reset_game_events_masked(
    const uint8_t* __restrict__ d_reset_mask,
    uint32_t num_envs,
    GameEventTrackerEnvStateSoA env_state)
{
    uint32_t env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx >= num_envs) return;
    if (d_reset_mask[env_idx] == 0) return;

    env_state.ball_shot[env_idx] = 0;
    env_state.ball_shot_goal_team[env_idx] = 0;
    env_state.shot_cooldown[env_idx] = 0.0f;
    env_state.ball_scored_last[env_idx] = 0;
}

__global__ void k_export_ball_pod(BallStateSoA src, BallStatePOD* dst, uint32_t count, uint32_t start_env) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst[i] = src.get_pod(start_env + i);
}

__global__ void k_import_ball_pod(BallStateSoA dst, const BallStatePOD* src, uint32_t count, uint32_t start_env) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst.set_pod(start_env + i, src[i]);
}

__global__ void k_export_car_pod(CarStateSoA src, CarStatePOD* dst, uint32_t count, uint32_t start_car) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst[i] = src.get_pod(start_car + i);
}

__global__ void k_import_car_pod(CarStateSoA dst, const CarStatePOD* src, uint32_t count, uint32_t start_car) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst.set_pod(start_car + i, src[i]);
}

__global__ void k_import_controls(CarControlsSoA dst, const CarControls* src, uint32_t count, uint32_t start_car) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst.set(start_car + i, src[i]);
}

} // namespace

SimContext::SimContext(uint32_t num_envs, uint32_t cars_per_env, cudaStream_t stream)
    : m_num_envs(num_envs), m_cars_per_env(cars_per_env), m_stream(stream) {
    m_total_cars = m_num_envs * m_cars_per_env;
    AllocateArena();
    ResetToDefault();
}

SimContext::~SimContext() {
    FreeArena();
}

SimContext::SimContext(SimContext&& o) noexcept
    : m_num_envs(o.m_num_envs),
      m_cars_per_env(o.m_cars_per_env),
      m_total_cars(o.m_total_cars),
      m_allocated_bytes(o.m_allocated_bytes),
      m_d_pool(o.m_d_pool),
      m_stream(o.m_stream),
      m_ball_state(o.m_ball_state),
      m_car_state(o.m_car_state),
      m_arena_state(o.m_arena_state),
      m_controls(o.m_controls),
      m_car_config(o.m_car_config),
      m_mutator_config(o.m_mutator_config),
      m_arena_config(o.m_arena_config) {
    o.m_d_pool = nullptr;
    o.m_allocated_bytes = 0;
    o.m_num_envs = 0;
    o.m_cars_per_env = 0;
    o.m_total_cars = 0;
    o.m_arena_state = ArenaStateSoA{};
    o.m_car_config = CarConfigSoA{};
    o.m_mutator_config = MutatorConfigSoA{};
    o.m_arena_config = ArenaConfigSoA{};
}

SimContext& SimContext::operator=(SimContext&& o) noexcept {
    if (this != &o) {
        FreeArena();
        m_num_envs = o.m_num_envs;
        m_cars_per_env = o.m_cars_per_env;
        m_total_cars = o.m_total_cars;
        m_allocated_bytes = o.m_allocated_bytes;
        m_d_pool = o.m_d_pool;
        m_stream = o.m_stream;
        m_ball_state = o.m_ball_state;
        m_car_state = o.m_car_state;
        m_arena_state = o.m_arena_state;
        m_controls = o.m_controls;
        m_car_config = o.m_car_config;
        m_mutator_config = o.m_mutator_config;
        m_arena_config = o.m_arena_config;

        o.m_d_pool = nullptr;
        o.m_allocated_bytes = 0;
        o.m_num_envs = 0;
        o.m_cars_per_env = 0;
        o.m_total_cars = 0;
        o.m_arena_state = ArenaStateSoA{};
        o.m_car_config = CarConfigSoA{};
        o.m_mutator_config = MutatorConfigSoA{};
        o.m_arena_config = ArenaConfigSoA{};
    }
    return *this;
}

size_t SimContext::GetBallPitchFloats() const {
    return align_128(m_num_envs * sizeof(float)) / sizeof(float);
}

size_t SimContext::GetCarPitchFloats() const {
    size_t car_count = (m_total_cars > 0) ? m_total_cars : 1;
    return align_128(car_count * sizeof(float)) / sizeof(float);
}

void SimContext::AllocateArena() {
    if (m_num_envs == 0) return;

    size_t env_count = m_num_envs;
    size_t car_count = (m_total_cars > 0) ? m_total_cars : 1;

    // Helper to calculate aligned bytes
    auto calc_slice = [](size_t count, size_t elem_size) -> size_t {
        return align_128(count * elem_size);
    };

    // Calculate total pool bytes across Ball, Car, Controls, and Staging buffers
    size_t total = 0;

    // --- Ball SoA slices (length = env_count) ---
    // 13 float arrays: pos(3), vel(3), quat(4), ang_vel(3)
    total += calc_slice(env_count, sizeof(float)) * 13;
    // Gamemode extensions: hs(3 floats), ds(1 int32, 2 floats, 1 uint8, 1 uint64)
    total += calc_slice(env_count, sizeof(float)) * 3;
    total += calc_slice(env_count, sizeof(int32_t));
    total += calc_slice(env_count, sizeof(float)) * 2;
    total += calc_slice(env_count, sizeof(uint8_t));
    total += calc_slice(env_count, sizeof(uint64_t));

    // --- Car SoA slices (length = car_count) ---
    // Rigid body: 13 float arrays (pos, vel, quat, ang_vel)
    total += calc_slice(car_count, sizeof(float)) * 13;
    // Boost: 3 floats, 1 uint8
    total += calc_slice(car_count, sizeof(float)) * 3;
    total += calc_slice(car_count, sizeof(uint8_t));
    // Wheels: 5 uint8 (ground + 4 contacts), 4 floats (suspension lengths)
    total += calc_slice(car_count, sizeof(uint8_t)) * 5;
    total += calc_slice(car_count, sizeof(float)) * 4;
    // Jump: 2 uint8, 4 floats
    total += calc_slice(car_count, sizeof(uint8_t)) * 2;
    total += calc_slice(car_count, sizeof(float)) * 4;
    // Flip: 2 uint8, 4 floats
    total += calc_slice(car_count, sizeof(uint8_t)) * 2;
    total += calc_slice(car_count, sizeof(float)) * 4;
    // Auto-flip & handbrake: 1 uint8, 3 floats
    total += calc_slice(car_count, sizeof(uint8_t));
    total += calc_slice(car_count, sizeof(float)) * 3;
    // Supersonic, demo, team & hitbox_type: 4 uint8, 2 floats
    total += calc_slice(car_count, sizeof(uint8_t)) * 4;
    total += calc_slice(car_count, sizeof(float)) * 2;
    // Contacts: 1 uint8, 3 floats, 1 int32, 1 float
    total += calc_slice(car_count, sizeof(uint8_t));
    total += calc_slice(car_count, sizeof(float)) * 3;
    total += calc_slice(car_count, sizeof(int32_t));
    total += calc_slice(car_count, sizeof(float));
    // Ball hit: 1 uint8, 6 floats, 1 uint64
    total += calc_slice(car_count, sizeof(uint8_t));
    total += calc_slice(car_count, sizeof(float)) * 6;
    total += calc_slice(car_count, sizeof(uint64_t));
    total += calc_slice(car_count, sizeof(uint8_t)); // ball_touched
    // Last controls: 5 floats, 3 uint8
    total += calc_slice(car_count, sizeof(float)) * 5;
    total += calc_slice(car_count, sizeof(uint8_t)) * 3;
    // Persistent wheel & Bullet unit dynamics: 17 floats
    total += calc_slice(car_count, sizeof(float)) * 17;

    // --- Controls SoA slices (length = car_count) ---
    total += calc_slice(car_count, sizeof(float)) * 5;
    total += calc_slice(car_count, sizeof(uint8_t)) * 3;

    // --- Arena SoA slices (length = env_count) ---
    total += calc_slice(env_count, sizeof(uint8_t));                  // is_goal
    total += calc_slice(env_count, sizeof(uint8_t));                  // scoring_team
    total += calc_slice(env_count, sizeof(uint8_t));                  // is_out_of_bounds
    total += calc_slice(env_count, sizeof(uint32_t));                 // tick_count
    total += calc_slice(env_count, sizeof(uint8_t));                  // terminated
    total += calc_slice(env_count, sizeof(uint8_t));                  // truncated
    total += calc_slice(car_count, sizeof(float));                    // rewards
    total += calc_slice(env_count * MAX_BOOST_PADS, sizeof(uint8_t)); // pad_is_active
    total += calc_slice(env_count * MAX_BOOST_PADS, sizeof(float));   // pad_cooldown

    // --- Game Event Tracking SoA slices (GEMINI.md SoA layout) ---
    total += calc_slice(env_count, sizeof(uint8_t));                  // ball_shot
    total += calc_slice(env_count, sizeof(uint8_t));                  // ball_shot_goal_team
    total += calc_slice(env_count, sizeof(float));                    // shot_cooldown
    total += calc_slice(env_count, sizeof(uint8_t));                  // ball_scored_last
    total += calc_slice(env_count, sizeof(uint8_t));                  // is_ball_probably_going_in
    total += calc_slice(env_count, sizeof(int32_t));                  // ball_probably_going_in_goal_team

    total += calc_slice(car_count, sizeof(uint8_t)) * 5;              // step touches, shots, saves, goals, assists
    total += calc_slice(car_count, sizeof(uint32_t)) * 5;             // total touches, shots, saves, goals, assists

    // --- Staging POD buffers (pre-allocated to eliminate runtime allocations) ---
    total += calc_slice(env_count, sizeof(BallStatePOD));
    total += calc_slice(car_count, sizeof(CarStatePOD));
    total += calc_slice(car_count, sizeof(CarControls));

    m_allocated_bytes = total;

    // Single pre-allocated device memory arena (GEMINI.md Section 2.2)
    cudaError_t err = cudaMalloc(&m_d_pool, m_allocated_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed in SimContext: ") + cudaGetErrorString(err));
    }

    // Zero entire memory pool
    cudaMemsetAsync(m_d_pool, 0, m_allocated_bytes, m_stream);

    // Carve out 128-byte aligned slices
    size_t offset = 0;
    auto assign_slice = [&](size_t count, size_t elem_size) -> void* {
        void* ptr = static_cast<char*>(m_d_pool) + offset;
        offset += calc_slice(count, elem_size);
        return ptr;
    };

    // Assign Ball pointers
    m_ball_state.pos_x = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.pos_y = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.pos_z = static_cast<float*>(assign_slice(env_count, sizeof(float)));

    m_ball_state.vel_x = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.vel_y = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.vel_z = static_cast<float*>(assign_slice(env_count, sizeof(float)));

    m_ball_state.q_w   = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.q_x   = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.q_y   = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.q_z   = static_cast<float*>(assign_slice(env_count, sizeof(float)));

    m_ball_state.ang_vel_x = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.ang_vel_y = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.ang_vel_z = static_cast<float*>(assign_slice(env_count, sizeof(float)));

    m_ball_state.hs_y_target_dir     = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.hs_cur_target_speed = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.hs_time_since_hit   = static_cast<float*>(assign_slice(env_count, sizeof(float)));

    m_ball_state.ds_charge_level        = static_cast<int32_t*>(assign_slice(env_count, sizeof(int32_t)));
    m_ball_state.ds_accumulated_hit_force = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.ds_y_target_dir        = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.ds_has_damaged         = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_ball_state.ds_last_damage_tick    = static_cast<uint64_t*>(assign_slice(env_count, sizeof(uint64_t)));

    // Assign Car pointers
    m_car_state.pos_x = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.pos_y = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.pos_z = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.vel_x = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.vel_y = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.vel_z = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.q_w   = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.q_x   = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.q_y   = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.q_z   = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.ang_vel_x = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ang_vel_y = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ang_vel_z = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.boost              = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.time_since_boosted = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.boosting_time      = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.is_boosting        = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));

    m_car_state.is_on_ground     = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.wheel_contact_0  = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.wheel_contact_1  = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.wheel_contact_2  = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.wheel_contact_3  = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));

    m_car_state.suspension_length_0 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.suspension_length_1 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.suspension_length_2 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.suspension_length_3 = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.has_jumped        = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.is_jumping        = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.jump_time         = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.has_double_jumped = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.air_time          = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.air_time_since_jump = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.has_flipped       = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.is_flipping       = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.flip_time         = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.flip_rel_torque_x = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.flip_rel_torque_y = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.flip_rel_torque_z = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.is_auto_flipping     = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.auto_flip_timer        = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.auto_flip_torque_scale = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.handbrake_val          = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.is_supersonic    = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.supersonic_time    = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.is_demoed        = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.demo_respawn_timer = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.team             = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.hitbox_type      = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_config.hitbox_type     = m_car_state.hitbox_type;

    m_car_state.world_contact_has_contact = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.world_contact_normal_x    = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.world_contact_normal_y    = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.world_contact_normal_z    = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.car_contact_other_car_id  = static_cast<int32_t*>(assign_slice(car_count, sizeof(int32_t)));
    m_car_state.car_contact_cooldown_timer= static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.ball_hit_is_valid         = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.ball_hit_rel_pos_x        = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ball_hit_rel_pos_y        = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ball_hit_rel_pos_z        = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ball_hit_extra_hit_force_x= static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ball_hit_extra_hit_force_y= static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ball_hit_extra_hit_force_z= static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ball_hit_tick_count       = static_cast<uint64_t*>(assign_slice(car_count, sizeof(uint64_t)));
    m_car_state.ball_touched              = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));

    m_car_state.last_controls_throttle    = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.last_controls_steer       = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.last_controls_pitch       = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.last_controls_yaw         = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.last_controls_roll        = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.last_controls_boost       = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.last_controls_jump        = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.last_controls_handbrake   = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));

    m_car_state.wheel_engine_force        = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_brake               = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_steer_angle         = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_lat_friction_0      = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_lat_friction_1      = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_lat_friction_2      = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_lat_friction_3      = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_long_friction_0     = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_long_friction_1     = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_long_friction_2     = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_long_friction_3     = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.pos_bt_x                 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.pos_bt_y                 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.pos_bt_z                 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.vel_bt_x                 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.vel_bt_y                 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.vel_bt_z                 = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    // Assign Controls pointers
    m_controls.throttle  = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_controls.steer     = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_controls.pitch     = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_controls.yaw       = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_controls.roll      = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_controls.boost     = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_controls.jump      = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_controls.handbrake = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));

    // Assign Arena pointers
    m_arena_state.is_goal          = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_arena_state.scoring_team     = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_arena_state.is_out_of_bounds = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_arena_state.tick_count       = static_cast<uint32_t*>(assign_slice(env_count, sizeof(uint32_t)));
    m_arena_state.terminated       = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_arena_state.truncated        = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_arena_state.rewards          = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_arena_state.pad_is_active    = static_cast<uint8_t*>(assign_slice(env_count * MAX_BOOST_PADS, sizeof(uint8_t)));
    m_arena_state.pad_cooldown     = static_cast<float*>(assign_slice(env_count * MAX_BOOST_PADS, sizeof(float)));

    // Assign Game Event Tracker & Goal Prediction SoA pointers
    m_event_env_state.ball_shot           = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_event_env_state.ball_shot_goal_team = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_event_env_state.shot_cooldown       = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_event_env_state.ball_scored_last    = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));

    m_d_is_ball_probably_going_in         = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_d_ball_probably_going_in_goal_team  = static_cast<int32_t*>(assign_slice(env_count, sizeof(int32_t)));

    m_event_car_events.step_touches       = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_event_car_events.step_shots         = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_event_car_events.step_saves         = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_event_car_events.step_goals         = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_event_car_events.step_assists       = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));

    m_event_car_events.total_touches      = static_cast<uint32_t*>(assign_slice(car_count, sizeof(uint32_t)));
    m_event_car_events.total_shots        = static_cast<uint32_t*>(assign_slice(car_count, sizeof(uint32_t)));
    m_event_car_events.total_saves        = static_cast<uint32_t*>(assign_slice(car_count, sizeof(uint32_t)));
    m_event_car_events.total_goals        = static_cast<uint32_t*>(assign_slice(car_count, sizeof(uint32_t)));
    m_event_car_events.total_assists      = static_cast<uint32_t*>(assign_slice(car_count, sizeof(uint32_t)));
}

void SimContext::FreeArena() {
    if (m_d_pool) {
        cudaFree(m_d_pool);
        m_d_pool = nullptr;
    }
    m_allocated_bytes = 0;
    m_d_is_ball_probably_going_in = nullptr;
    m_d_ball_probably_going_in_goal_team = nullptr;
    m_event_env_state = {};
    m_event_car_events = {};
}

void SimContext::ResetToDefault() {
    if (m_num_envs == 0) return;

    constexpr uint32_t threads = 128;
    uint32_t ball_blocks = (m_num_envs + threads - 1) / threads;
    k_init_ball_state<<<ball_blocks, threads, 0, m_stream>>>(m_ball_state, m_num_envs);

    if (m_total_cars > 0) {
        uint32_t car_blocks = (m_total_cars + threads - 1) / threads;
        k_init_car_state<<<car_blocks, threads, 0, m_stream>>>(m_car_state, m_total_cars, m_cars_per_env, 0);
    }

    uint32_t arena_blocks = (m_num_envs + threads - 1) / threads;
    k_init_arena_state<<<arena_blocks, threads, 0, m_stream>>>(m_arena_state, m_num_envs, m_cars_per_env);

    ResetGameEvents(-1);

    if (m_stream) {
        cudaStreamSynchronize(m_stream);
    } else {
        cudaDeviceSynchronize();
    }
}

void SimContext::ResetEnvironmentsIndexed(const int32_t* d_env_indices, uint32_t num_resets) {
    if (!d_env_indices || num_resets == 0) return;
    constexpr uint32_t threads = 128;
    uint32_t blocks = (num_resets + threads - 1) / threads;
    k_reset_environments_indexed<<<blocks, threads, 0, m_stream>>>(
        d_env_indices, num_resets, m_cars_per_env, m_ball_state, m_car_state, m_arena_state, 0
    );
    ResetGameEventsIndexed(d_env_indices, num_resets);
    // Asynchronous execution on m_stream with NO host synchronization barriers
}

void SimContext::ResetEnvironmentsIndexed(const int64_t* d_env_indices, uint32_t num_resets) {
    if (!d_env_indices || num_resets == 0) return;
    constexpr uint32_t threads = 128;
    uint32_t blocks = (num_resets + threads - 1) / threads;
    k_reset_environments_indexed<<<blocks, threads, 0, m_stream>>>(
        d_env_indices, num_resets, m_cars_per_env, m_ball_state, m_car_state, m_arena_state, 0
    );
    if (m_event_env_state.ball_shot) {
        k_reset_game_events_indexed<<<blocks, threads, 0, m_stream>>>(
            d_env_indices, num_resets, m_num_envs, m_event_env_state
        );
    }
    // Asynchronous execution on m_stream with NO host synchronization barriers
}

void SimContext::ResetEnvironmentsMasked(const uint8_t* d_reset_mask) {
    if (!d_reset_mask || m_num_envs == 0) return;
    constexpr uint32_t threads = 128;
    uint32_t blocks = (m_num_envs + threads - 1) / threads;
    k_reset_environments_masked<<<blocks, threads, 0, m_stream>>>(
        d_reset_mask, m_num_envs, m_cars_per_env, m_ball_state, m_car_state, m_arena_state, 0
    );
    if (m_event_env_state.ball_shot) {
        k_reset_game_events_masked<<<blocks, threads, 0, m_stream>>>(
            d_reset_mask, m_num_envs, m_event_env_state
        );
    }
    // Asynchronous execution on m_stream with NO host synchronization barriers
}

void SimContext::CopyBallStateToHost(BallStatePOD* host_out, uint32_t env_start, uint32_t count) {
    if (count == 0) count = m_num_envs - env_start;
    if (count == 0 || host_out == nullptr) return;

    // Use pre-allocated staging buffer at end of pool
    char* staging_base = static_cast<char*>(m_d_pool) + m_allocated_bytes
        - align_128(m_num_envs * sizeof(BallStatePOD))
        - align_128(m_total_cars * sizeof(CarStatePOD))
        - align_128(m_total_cars * sizeof(CarControls));
    BallStatePOD* d_staging = reinterpret_cast<BallStatePOD*>(staging_base);

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_export_ball_pod<<<blocks, threads, 0, m_stream>>>(m_ball_state, d_staging, count, env_start);

    cudaMemcpyAsync(host_out, d_staging, count * sizeof(BallStatePOD), cudaMemcpyDeviceToHost, m_stream);
    if (m_stream) cudaStreamSynchronize(m_stream); else cudaDeviceSynchronize();
}

void SimContext::CopyCarStateToHost(CarStatePOD* host_out, uint32_t car_start, uint32_t count) {
    if (count == 0) count = m_total_cars - car_start;
    if (count == 0 || host_out == nullptr) return;

    char* staging_base = static_cast<char*>(m_d_pool) + m_allocated_bytes
        - align_128(m_total_cars * sizeof(CarStatePOD))
        - align_128(m_total_cars * sizeof(CarControls));
    CarStatePOD* d_staging = reinterpret_cast<CarStatePOD*>(staging_base);

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_export_car_pod<<<blocks, threads, 0, m_stream>>>(m_car_state, d_staging, count, car_start);

    cudaMemcpyAsync(host_out, d_staging, count * sizeof(CarStatePOD), cudaMemcpyDeviceToHost, m_stream);
    if (m_stream) cudaStreamSynchronize(m_stream); else cudaDeviceSynchronize();
}

void SimContext::CopyBallStateToDevice(const BallStatePOD* host_in, uint32_t env_start, uint32_t count) {
    if (count == 0) count = m_num_envs - env_start;
    if (count == 0 || host_in == nullptr) return;

    char* staging_base = static_cast<char*>(m_d_pool) + m_allocated_bytes
        - align_128(m_num_envs * sizeof(BallStatePOD))
        - align_128(m_total_cars * sizeof(CarStatePOD))
        - align_128(m_total_cars * sizeof(CarControls));
    BallStatePOD* d_staging = reinterpret_cast<BallStatePOD*>(staging_base);

    cudaMemcpyAsync(d_staging, host_in, count * sizeof(BallStatePOD), cudaMemcpyHostToDevice, m_stream);

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_import_ball_pod<<<blocks, threads, 0, m_stream>>>(m_ball_state, d_staging, count, env_start);

    if (m_stream) cudaStreamSynchronize(m_stream); else cudaDeviceSynchronize();
}

void SimContext::CopyCarStateToDevice(const CarStatePOD* host_in, uint32_t car_start, uint32_t count) {
    if (count == 0) count = m_total_cars - car_start;
    if (count == 0 || host_in == nullptr) return;

    char* staging_base = static_cast<char*>(m_d_pool) + m_allocated_bytes
        - align_128(m_total_cars * sizeof(CarStatePOD))
        - align_128(m_total_cars * sizeof(CarControls));
    CarStatePOD* d_staging = reinterpret_cast<CarStatePOD*>(staging_base);

    cudaMemcpyAsync(d_staging, host_in, count * sizeof(CarStatePOD), cudaMemcpyHostToDevice, m_stream);

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_import_car_pod<<<blocks, threads, 0, m_stream>>>(m_car_state, d_staging, count, car_start);

    if (m_stream) cudaStreamSynchronize(m_stream); else cudaDeviceSynchronize();
}

void SimContext::CopyControlsToDevice(const CarControls* host_in, uint32_t car_start, uint32_t count) {
    if (count == 0) count = m_total_cars - car_start;
    if (count == 0 || host_in == nullptr) return;

    char* staging_base = static_cast<char*>(m_d_pool) + m_allocated_bytes
        - align_128(m_total_cars * sizeof(CarControls));
    CarControls* d_staging = reinterpret_cast<CarControls*>(staging_base);

    cudaMemcpyAsync(d_staging, host_in, count * sizeof(CarControls), cudaMemcpyHostToDevice, m_stream);

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_import_controls<<<blocks, threads, 0, m_stream>>>(m_controls, d_staging, count, car_start);

    if (m_stream) cudaStreamSynchronize(m_stream); else cudaDeviceSynchronize();
}

// ============================================================================
// State Setters & Ball Trajectory Prediction Kernels
// ============================================================================

namespace {

__global__ void k_set_ball_state_tensor(
    BallStateSoA dst,
    uint32_t num_envs,
    const float* __restrict__ src_tensor,
    uint32_t count,
    uint32_t start_env,
    const int32_t* __restrict__ d_indices,
    uint32_t num_cols,
    int64_t stride_env,
    int64_t stride_col)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int32_t raw_idx = d_indices ? d_indices[i] : static_cast<int32_t>(start_env + i);
    if (raw_idx < 0 || static_cast<uint32_t>(raw_idx) >= num_envs) return;
    uint32_t env_idx = static_cast<uint32_t>(raw_idx);

    const float* row = src_tensor + (i * stride_env);

    dst.pos_x[env_idx] = row[0 * stride_col];
    dst.pos_y[env_idx] = row[1 * stride_col];
    dst.pos_z[env_idx] = row[2 * stride_col];

    dst.vel_x[env_idx] = row[3 * stride_col];
    dst.vel_y[env_idx] = row[4 * stride_col];
    dst.vel_z[env_idx] = row[5 * stride_col];

    if (num_cols >= 13) {
        dst.q_w[env_idx] = row[6 * stride_col];
        dst.q_x[env_idx] = row[7 * stride_col];
        dst.q_y[env_idx] = row[8 * stride_col];
        dst.q_z[env_idx] = row[9 * stride_col];

        dst.ang_vel_x[env_idx] = row[10 * stride_col];
        dst.ang_vel_y[env_idx] = row[11 * stride_col];
        dst.ang_vel_z[env_idx] = row[12 * stride_col];
    } else if (num_cols >= 9) {
        dst.q_w[env_idx] = 1.0f;
        dst.q_x[env_idx] = 0.0f;
        dst.q_y[env_idx] = 0.0f;
        dst.q_z[env_idx] = 0.0f;

        dst.ang_vel_x[env_idx] = row[6 * stride_col];
        dst.ang_vel_y[env_idx] = row[7 * stride_col];
        dst.ang_vel_z[env_idx] = row[8 * stride_col];
    } else {
        dst.q_w[env_idx] = 1.0f;
        dst.q_x[env_idx] = 0.0f;
        dst.q_y[env_idx] = 0.0f;
        dst.q_z[env_idx] = 0.0f;

        dst.ang_vel_x[env_idx] = 0.0f;
        dst.ang_vel_y[env_idx] = 0.0f;
        dst.ang_vel_z[env_idx] = 0.0f;
    }
}

__global__ void k_set_car_state_tensor(
    CarStateSoA dst,
    uint32_t total_cars,
    uint32_t cars_per_env,
    const float* __restrict__ src_tensor,
    uint32_t count,
    uint32_t start_car,
    const int32_t* __restrict__ d_indices,
    uint32_t num_cols,
    int64_t stride_env,
    int64_t stride_car,
    int64_t stride_col)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int32_t raw_idx = d_indices ? d_indices[i] : static_cast<int32_t>(start_car + i);
    if (raw_idx < 0 || static_cast<uint32_t>(raw_idx) >= total_cars) return;
    uint32_t car_idx = static_cast<uint32_t>(raw_idx);

    int64_t row_offset;
    if (stride_env != 0 && cars_per_env > 0) {
        int64_t e = i / cars_per_env;
        int64_t c = i % cars_per_env;
        row_offset = e * stride_env + c * stride_car;
    } else {
        row_offset = i * stride_car;
    }
    const float* row = src_tensor + row_offset;

    float px = row[0 * stride_col];
    float py = row[1 * stride_col];
    float pz = row[2 * stride_col];

    float vx = row[3 * stride_col];
    float vy = row[4 * stride_col];
    float vz = row[5 * stride_col];

    dst.pos_x[car_idx] = px;
    dst.pos_y[car_idx] = py;
    dst.pos_z[car_idx] = pz;

    dst.pos_bt_x[car_idx] = px * 0.02f;
    dst.pos_bt_y[car_idx] = py * 0.02f;
    dst.pos_bt_z[car_idx] = pz * 0.02f;

    dst.vel_x[car_idx] = vx;
    dst.vel_y[car_idx] = vy;
    dst.vel_z[car_idx] = vz;

    dst.vel_bt_x[car_idx] = vx * 0.02f;
    dst.vel_bt_y[car_idx] = vy * 0.02f;
    dst.vel_bt_z[car_idx] = vz * 0.02f;

    float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
    float wx = 0.0f, wy = 0.0f, wz = 0.0f;
    float b_val = BOOST_SPAWN;

    if (num_cols >= 14) {
        qw = row[6 * stride_col];
        qx = row[7 * stride_col];
        qy = row[8 * stride_col];
        qz = row[9 * stride_col];
        wx = row[10 * stride_col];
        wy = row[11 * stride_col];
        wz = row[12 * stride_col];
        b_val = row[13 * stride_col];
    } else if (num_cols >= 13) {
        qw = row[6 * stride_col];
        qx = row[7 * stride_col];
        qy = row[8 * stride_col];
        qz = row[9 * stride_col];
        wx = row[10 * stride_col];
        wy = row[11 * stride_col];
        wz = row[12 * stride_col];
    } else if (num_cols >= 9) {
        wx = row[6 * stride_col];
        wy = row[7 * stride_col];
        wz = row[8 * stride_col];
    }

    dst.q_w[car_idx] = qw;
    dst.q_x[car_idx] = qx;
    dst.q_y[car_idx] = qy;
    dst.q_z[car_idx] = qz;

    dst.ang_vel_x[car_idx] = wx;
    dst.ang_vel_y[car_idx] = wy;
    dst.ang_vel_z[car_idx] = wz;

    dst.boost[car_idx] = b_val;

    // Reset transient mechanics on state reset
    dst.is_on_ground[car_idx] = (pz <= 17.5f) ? 1 : 0;
    dst.has_jumped[car_idx] = 0;
    dst.is_jumping[car_idx] = 0;
    dst.jump_time[car_idx] = 0.0f;
    dst.has_double_jumped[car_idx] = 0;
    dst.has_flipped[car_idx] = 0;
    dst.is_flipping[car_idx] = 0;
    dst.flip_time[car_idx] = 0.0f;
    dst.air_time[car_idx] = 0.0f;
    dst.air_time_since_jump[car_idx] = 0.0f;
    dst.is_boosting[car_idx] = 0;
    dst.boosting_time[car_idx] = 0.0f;
    dst.time_since_boosted[car_idx] = 0.0f;
    dst.is_auto_flipping[car_idx] = 0;
    dst.auto_flip_timer[car_idx] = 0.0f;
    dst.is_demoed[car_idx] = 0;
    dst.demo_respawn_timer[car_idx] = 0.0f;
    if (dst.ball_touched) dst.ball_touched[car_idx] = 0;
    if (dst.ball_hit_is_valid) dst.ball_hit_is_valid[car_idx] = 0;
}

__global__ void k_set_pad_state_tensor(
    ArenaStateSoA dst,
    uint32_t num_envs,
    const uint8_t* __restrict__ d_active,
    int64_t stride_env_active,
    int64_t stride_pad_active,
    const float* __restrict__ d_cooldown,
    int64_t stride_env_cd,
    int64_t stride_pad_cd,
    uint32_t count,
    uint32_t start_env,
    const int32_t* __restrict__ d_indices)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int32_t raw_idx = d_indices ? d_indices[i] : static_cast<int32_t>(start_env + i);
    if (raw_idx < 0 || static_cast<uint32_t>(raw_idx) >= num_envs) return;
    uint32_t env_idx = static_cast<uint32_t>(raw_idx);

    for (uint32_t p = 0; p < MAX_BOOST_PADS; ++p) {
        uint32_t pad_linear_idx = env_idx * MAX_BOOST_PADS + p;
        if (d_active != nullptr) {
            dst.pad_is_active[pad_linear_idx] = d_active[i * stride_env_active + p * stride_pad_active];
        }
        if (d_cooldown != nullptr) {
            dst.pad_cooldown[pad_linear_idx] = d_cooldown[i * stride_env_cd + p * stride_pad_cd];
        }
    }
}

__global__ void k_predict_ball_trajectory(
    uint32_t num_envs,
    uint32_t num_ticks,
    uint32_t num_cols,
    BallStateSoA ball_state,
    const float* __restrict__ d_init_state,
    uint32_t init_cols,
    int64_t stride_init_env,
    int64_t stride_init_col,
    float* __restrict__ d_out,
    int64_t stride_out_env,
    int64_t stride_out_tick,
    int64_t stride_out_col,
    float dt,
    MutatorConfigSoA mutator_config)
{
    uint32_t env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx >= num_envs) return;

    MutatorConfig mut_cfg = mutator_config.get(env_idx);

    Vec3 pos, vel, ang_vel;
    Quat quat;

    if (d_init_state != nullptr) {
        const float* init_row = d_init_state + env_idx * stride_init_env;
        pos = Vec3(init_row[0 * stride_init_col], init_row[1 * stride_init_col], init_row[2 * stride_init_col]);
        vel = Vec3(init_row[3 * stride_init_col], init_row[4 * stride_init_col], init_row[5 * stride_init_col]);
        if (init_cols >= 13) {
            quat = Quat(init_row[6 * stride_init_col], init_row[7 * stride_init_col], init_row[8 * stride_init_col], init_row[9 * stride_init_col]);
            ang_vel = Vec3(init_row[10 * stride_init_col], init_row[11 * stride_init_col], init_row[12 * stride_init_col]);
        } else if (init_cols >= 9) {
            quat = Quat::identity();
            ang_vel = Vec3(init_row[6 * stride_init_col], init_row[7 * stride_init_col], init_row[8 * stride_init_col]);
        } else {
            quat = Quat::identity();
            ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        }
    } else {
        pos = Vec3(ball_state.pos_x[env_idx], ball_state.pos_y[env_idx], ball_state.pos_z[env_idx]);
        vel = Vec3(ball_state.vel_x[env_idx], ball_state.vel_y[env_idx], ball_state.vel_z[env_idx]);
        ang_vel = Vec3(ball_state.ang_vel_x[env_idx], ball_state.ang_vel_y[env_idx], ball_state.ang_vel_z[env_idx]);
        quat = Quat(ball_state.q_w[env_idx], ball_state.q_x[env_idx], ball_state.q_y[env_idx], ball_state.q_z[env_idx]);
    }

    for (uint32_t t = 0; t < num_ticks; ++t) {
        float* out_tick = d_out + env_idx * stride_out_env + t * stride_out_tick;
        out_tick[0 * stride_out_col] = pos.x;
        out_tick[1 * stride_out_col] = pos.y;
        out_tick[2 * stride_out_col] = pos.z;
        out_tick[3 * stride_out_col] = vel.x;
        out_tick[4 * stride_out_col] = vel.y;
        out_tick[5 * stride_out_col] = vel.z;

        if (num_cols >= 9) {
            if (num_cols >= 13) {
                out_tick[6 * stride_out_col] = quat.w;
                out_tick[7 * stride_out_col] = quat.x;
                out_tick[8 * stride_out_col] = quat.y;
                out_tick[9 * stride_out_col] = quat.z;
                out_tick[10 * stride_out_col] = ang_vel.x;
                out_tick[11 * stride_out_col] = ang_vel.y;
                out_tick[12 * stride_out_col] = ang_vel.z;
            } else {
                out_tick[6 * stride_out_col] = ang_vel.x;
                out_tick[7 * stride_out_col] = ang_vel.y;
                out_tick[8 * stride_out_col] = ang_vel.z;
            }
        }

        if (t + 1 < num_ticks) {
            step_ball_state(pos, vel, ang_vel, quat, dt, mut_cfg);
        }
    }
}

} // namespace

void SimContext::SetBallStateFromTensor(
    const float* d_tensor,
    uint32_t count,
    uint32_t start_env,
    const int32_t* d_indices,
    uint32_t num_cols,
    int64_t stride_env,
    int64_t stride_col)
{
    if (d_tensor == nullptr) return;
    if (count == 0) {
        if (d_indices != nullptr) return;
        if (start_env >= m_num_envs) return;
        count = m_num_envs - start_env;
    }
    if (count == 0) return;

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_set_ball_state_tensor<<<blocks, threads, 0, m_stream>>>(
        m_ball_state,
        m_num_envs,
        d_tensor,
        count,
        start_env,
        d_indices,
        num_cols,
        stride_env,
        stride_col
    );
}

void SimContext::SetCarStateFromTensor(
    const float* d_tensor,
    uint32_t count,
    uint32_t start_car,
    const int32_t* d_indices,
    uint32_t num_cols,
    int64_t stride_env,
    int64_t stride_car,
    int64_t stride_col)
{
    if (d_tensor == nullptr) return;
    if (count == 0) {
        if (d_indices != nullptr) return;
        if (start_car >= m_total_cars) return;
        count = m_total_cars - start_car;
    }
    if (count == 0) return;

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_set_car_state_tensor<<<blocks, threads, 0, m_stream>>>(
        m_car_state,
        m_total_cars,
        m_cars_per_env,
        d_tensor,
        count,
        start_car,
        d_indices,
        num_cols,
        stride_env,
        stride_car,
        stride_col
    );
}

void SimContext::SetPadStateFromTensor(
    const uint8_t* d_active,
    int64_t stride_env_active,
    int64_t stride_pad_active,
    const float* d_cooldown,
    int64_t stride_env_cd,
    int64_t stride_pad_cd,
    uint32_t count,
    uint32_t start_env,
    const int32_t* d_indices)
{
    if (d_active == nullptr && d_cooldown == nullptr) return;
    if (count == 0) {
        if (d_indices != nullptr) return;
        if (start_env >= m_num_envs) return;
        count = m_num_envs - start_env;
    }
    if (count == 0) return;

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_set_pad_state_tensor<<<blocks, threads, 0, m_stream>>>(
        m_arena_state,
        m_num_envs,
        d_active,
        stride_env_active,
        stride_pad_active,
        d_cooldown,
        stride_env_cd,
        stride_pad_cd,
        count,
        start_env,
        d_indices
    );
}

void SimContext::PredictBallTrajectory(
    float* d_out,
    uint32_t num_ticks,
    uint32_t num_cols,
    const float* d_init_state,
    uint32_t init_cols,
    int64_t stride_init_env,
    int64_t stride_init_col,
    int64_t stride_out_env,
    int64_t stride_out_tick,
    int64_t stride_out_col)
{
    if (d_out == nullptr || num_ticks == 0 || m_num_envs == 0) return;

    if (stride_out_env == 0) stride_out_env = static_cast<int64_t>(num_ticks * num_cols);
    if (stride_out_tick == 0) stride_out_tick = static_cast<int64_t>(num_cols);

    constexpr uint32_t threads = 128;
    uint32_t blocks = (m_num_envs + threads - 1) / threads;
    k_predict_ball_trajectory<<<blocks, threads, 0, m_stream>>>(
        m_num_envs,
        num_ticks,
        num_cols,
        m_ball_state,
        d_init_state,
        init_cols,
        stride_init_env,
        stride_init_col,
        d_out,
        stride_out_env,
        stride_out_tick,
        stride_out_col,
        DELTA_TIME,
        m_mutator_config
    );
}

__global__ void k_evaluate_is_ball_probably_going_in(
    uint32_t num_envs,
    const BallStateSoA ball_state,
    float max_time,
    float extra_margin,
    uint8_t* __restrict__ out_going_in,
    int32_t* __restrict__ out_goal_team,
    float goal_base_threshold_y,
    float ball_radius)
{
    uint32_t env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx >= num_envs) return;

    Vec3 ball_pos(ball_state.pos_x[env_idx], ball_state.pos_y[env_idx], ball_state.pos_z[env_idx]);
    Vec3 ball_vel(ball_state.vel_x[env_idx], ball_state.vel_y[env_idx], ball_state.vel_z[env_idx]);

    int32_t goal_team = -1;
    bool going_in = is_ball_probably_going_in(
        ball_pos,
        ball_vel,
        max_time,
        extra_margin,
        &goal_team,
        goal_base_threshold_y,
        ball_radius
    );

    out_going_in[env_idx] = going_in ? 1 : 0;
    out_goal_team[env_idx] = goal_team;
}

__device__ __forceinline__ bool get_shooter_passer_gpu(
    uint32_t env_idx,
    uint32_t cars_per_env,
    uint8_t target_team,
    const CarStateSoA& car_state,
    uint64_t current_tick,
    uint64_t max_shooter_ticks,
    uint64_t max_passer_ticks,
    int32_t& shooter_out,
    int32_t& passer_out,
    bool find_passer)
{
    shooter_out = -1;
    passer_out = -1;

    uint64_t best_shooter_tick = 0;
    for (uint32_t c = 0; c < cars_per_env; c++) {
        uint32_t car_idx = env_idx * cars_per_env + c;
        if (car_state.team[car_idx] != target_team) continue;
        if (!car_state.ball_hit_is_valid[car_idx]) continue;

        uint64_t hit_tick = car_state.ball_hit_tick_count[car_idx];
        if (hit_tick + max_shooter_ticks >= current_tick) {
            if (shooter_out < 0 || hit_tick > best_shooter_tick) {
                shooter_out = static_cast<int32_t>(car_idx);
                best_shooter_tick = hit_tick;
            }
        }
    }

    if (shooter_out >= 0 && find_passer) {
        uint64_t best_passer_tick = 0;
        for (uint32_t c = 0; c < cars_per_env; c++) {
            uint32_t car_idx = env_idx * cars_per_env + c;
            if (car_state.team[car_idx] != target_team) continue;
            if (!car_state.ball_hit_is_valid[car_idx]) continue;
            if (static_cast<int32_t>(car_idx) == shooter_out) continue;

            uint64_t hit_tick = car_state.ball_hit_tick_count[car_idx];
            if (hit_tick + max_passer_ticks >= best_shooter_tick) {
                if (passer_out < 0 || hit_tick > best_passer_tick) {
                    passer_out = static_cast<int32_t>(car_idx);
                    best_passer_tick = hit_tick;
                }
            }
        }
    }

    return shooter_out >= 0;
}

__global__ void k_update_game_events(
    uint32_t num_envs,
    uint32_t cars_per_env,
    const BallStateSoA ball_state,
    const CarStateSoA car_state,
    const ArenaStateSoA arena_state,
    GameEventTrackerEnvStateSoA env_state,
    GameEventTrackerCarEventsSoA car_events,
    GameEventTrackerConfig config,
    float delta_time,
    uint32_t delta_ticks,
    float goal_base_threshold_y,
    float ball_radius)
{
    uint32_t env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx >= num_envs) return;

    uint32_t current_tick = arena_state.tick_count[env_idx];
    float tickrate = 1.0f / DELTA_TIME; // 120.0f
    float score_threshold_y = goal_base_threshold_y + ball_radius;

    Vec3 ball_pos(ball_state.pos_x[env_idx], ball_state.pos_y[env_idx], ball_state.pos_z[env_idx]);
    Vec3 ball_vel(ball_state.vel_x[env_idx], ball_state.vel_y[env_idx], ball_state.vel_z[env_idx]);

    // 1. Reset per-step car events and detect touches
    for (uint32_t c = 0; c < cars_per_env; c++) {
        uint32_t car_idx = env_idx * cars_per_env + c;
        car_events.step_shots[car_idx] = 0;
        car_events.step_saves[car_idx] = 0;
        car_events.step_goals[car_idx] = 0;
        car_events.step_assists[car_idx] = 0;

        uint8_t touched = car_state.ball_touched ? car_state.ball_touched[car_idx] : 0;
        car_events.step_touches[car_idx] = touched;
        if (touched) {
            car_events.total_touches[car_idx] += 1;
        }
    }

    bool scored = is_ball_scored(ball_pos, score_threshold_y);
    bool was_scored_last = env_state.ball_scored_last[env_idx] != 0;

    if (scored && !was_scored_last) {
        // Goal scored event!
        // RS_TEAM_FROM_Y(-ball_pos.y):
        // If ball_pos.y > 0 (Orange net scored into), scoring team is BLUE (0).
        // If ball_pos.y < 0 (Blue net scored into), scoring team is ORANGE (1).
        uint8_t scoring_team = (ball_pos.y > 0.0f) ? 0 : 1;

        int32_t shooter_idx = -1;
        int32_t passer_idx = -1;
        uint64_t max_shooter_ticks = static_cast<uint64_t>(config.goal_max_touch_time * tickrate);
        uint64_t max_passer_ticks = static_cast<uint64_t>(config.pass_max_touch_time * tickrate);

        if (get_shooter_passer_gpu(
            env_idx, cars_per_env, scoring_team, car_state, current_tick,
            max_shooter_ticks, max_passer_ticks, shooter_idx, passer_idx, true))
        {
            if (shooter_idx >= 0) {
                car_events.step_goals[shooter_idx] = 1;
                car_events.total_goals[shooter_idx] += 1;
            }
            if (passer_idx >= 0) {
                car_events.step_assists[passer_idx] = 1;
                car_events.total_assists[passer_idx] += 1;
            }
        }
    } else if (!scored) {
        uint8_t ball_shot = env_state.ball_shot[env_idx];
        if (!ball_shot) {
            float shot_cd = env_state.shot_cooldown[env_idx];
            if (shot_cd > 0.0f) {
                shot_cd = fmaxf(shot_cd - delta_time, 0.0f);
                env_state.shot_cooldown[env_idx] = shot_cd;
            } else {
                float speed_sq = ball_vel.length_sq();
                if (speed_sq >= config.shot_min_speed * config.shot_min_speed) {
                    int32_t goal_team = 0;
                    if (is_ball_probably_going_in(
                        ball_pos, ball_vel, config.shot_min_score_time, config.pred_score_extra_margin,
                        &goal_team, goal_base_threshold_y, ball_radius))
                    {
                        uint8_t shooter_team = (goal_team == 0) ? 1 : 0;
                        uint64_t shot_min_touch_delay_ticks = static_cast<uint64_t>(config.shot_touch_min_delay * tickrate);
                        uint64_t max_shooter_ticks = delta_ticks + shot_min_touch_delay_ticks;
                        uint64_t max_passer_ticks = static_cast<uint64_t>(config.pass_max_touch_time * tickrate);

                        int32_t shooter_idx = -1;
                        int32_t passer_idx = -1;
                        if (get_shooter_passer_gpu(
                            env_idx, cars_per_env, shooter_team, car_state, current_tick,
                            max_shooter_ticks, max_passer_ticks, shooter_idx, passer_idx, true))
                        {
                            uint64_t ticks_since_hit = (current_tick >= car_state.ball_hit_tick_count[shooter_idx])
                                ? (current_tick - car_state.ball_hit_tick_count[shooter_idx])
                                : 0;
                            if (ticks_since_hit >= shot_min_touch_delay_ticks) {
                                env_state.ball_shot[env_idx] = 1;
                                env_state.ball_shot_goal_team[env_idx] = static_cast<uint8_t>(goal_team);
                                env_state.shot_cooldown[env_idx] = config.shot_event_cooldown;

                                car_events.step_shots[shooter_idx] = 1;
                                car_events.total_shots[shooter_idx] += 1;
                            }
                        }
                    }
                }
            }
        } else {
            // Ball is currently shot
            int32_t dummy_team = 0;
            bool will_score = is_ball_probably_going_in(
                ball_pos, ball_vel, config.shot_min_score_time, config.pred_score_extra_margin,
                &dummy_team, goal_base_threshold_y, ball_radius
            );

            if (!will_score) {
                // Ball stopped going in (saved or missed)
                uint8_t goal_team = env_state.ball_shot_goal_team[env_idx];
                int32_t saver_idx = -1;
                int32_t dummy_passer = -1;
                if (get_shooter_passer_gpu(
                    env_idx, cars_per_env, goal_team, car_state, current_tick,
                    delta_ticks, 0, saver_idx, dummy_passer, false))
                {
                    if (saver_idx >= 0) {
                        car_events.step_saves[saver_idx] = 1;
                        car_events.total_saves[saver_idx] += 1;
                    }
                }
                env_state.ball_shot[env_idx] = 0;
            }
        }
    }

    env_state.ball_scored_last[env_idx] = scored ? 1 : 0;
}

void SimContext::EvaluateIsBallProbablyGoingIn(float max_time, float extra_margin) {
    if (m_num_envs == 0 || m_d_is_ball_probably_going_in == nullptr) return;

    constexpr uint32_t threads = 128;
    uint32_t blocks = (m_num_envs + threads - 1) / threads;

    float goal_base_threshold_y = m_mutator_config.goal_base_threshold_y
        ? m_mutator_config.goal_base_threshold_y[0]
        : SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y;
    float ball_radius = m_mutator_config.ball_radius
        ? m_mutator_config.ball_radius[0]
        : BALL_RADIUS;

    k_evaluate_is_ball_probably_going_in<<<blocks, threads, 0, m_stream>>>(
        m_num_envs,
        m_ball_state,
        max_time,
        extra_margin,
        m_d_is_ball_probably_going_in,
        m_d_ball_probably_going_in_goal_team,
        goal_base_threshold_y,
        ball_radius
    );
}

void SimContext::UpdateGameEvents(float delta_time, uint32_t delta_ticks) {
    if (m_num_envs == 0 || !m_event_tracking_enabled || m_event_env_state.ball_shot == nullptr) return;

    constexpr uint32_t threads = 128;
    uint32_t blocks = (m_num_envs + threads - 1) / threads;

    float goal_base_threshold_y = m_mutator_config.goal_base_threshold_y
        ? m_mutator_config.goal_base_threshold_y[0]
        : SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y;
    float ball_radius = m_mutator_config.ball_radius
        ? m_mutator_config.ball_radius[0]
        : BALL_RADIUS;

    k_update_game_events<<<blocks, threads, 0, m_stream>>>(
        m_num_envs,
        m_cars_per_env,
        m_ball_state,
        m_car_state,
        m_arena_state,
        m_event_env_state,
        m_event_car_events,
        m_event_tracker_config,
        delta_time,
        delta_ticks,
        goal_base_threshold_y,
        ball_radius
    );
}

void SimContext::ResetGameEvents(int32_t env_idx) {
    if (m_num_envs == 0 || m_event_env_state.ball_shot == nullptr) return;

    constexpr uint32_t threads = 128;
    uint32_t blocks = (m_num_envs + threads - 1) / threads;

    k_reset_game_events<<<blocks, threads, 0, m_stream>>>(
        m_num_envs,
        m_event_env_state,
        env_idx
    );
}

void SimContext::ResetGameEventsIndexed(const int32_t* d_env_indices, uint32_t num_resets) {
    if (!d_env_indices || num_resets == 0 || m_event_env_state.ball_shot == nullptr) return;

    constexpr uint32_t threads = 128;
    uint32_t blocks = (num_resets + threads - 1) / threads;

    k_reset_game_events_indexed<<<blocks, threads, 0, m_stream>>>(
        d_env_indices,
        num_resets,
        m_num_envs,
        m_event_env_state
    );
}

} // namespace rocketsim_cuda
