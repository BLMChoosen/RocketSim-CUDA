#pragma once
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/quat.cuh"
#include "rocketsim_cuda/physics/integrator.cuh"
#include "rocketsim_cuda/physics/arena_sdf.cuh"
#include "rocketsim_cuda/physics/contact_solver.cuh"
#include "rocketsim_cuda/types/arena_config.cuh"
#include "rocketsim_cuda/types/ball_state.cuh"

namespace rocketsim_cuda {

/**
 * @brief Step a single ball state forward by dt with analytical arena SDF collisions.
 * Matches 1:1 with CPU RocketSim Ball simulation.
 */
__device__ __forceinline__ void step_ball_state(
    Vec3& pos,
    Vec3& vel,
    Vec3& ang_vel,
    Quat& quat,
    float dt,
    const MutatorConfig& mut_cfg = MutatorConfig())
{
    // Check if sleeping (zero velocity on ground)
    if (vel.length_sq() == 0.0f && ang_vel.length_sq() == 0.0f && pos.z <= BALL_REST_Z + 0.1f) {
        return;
    }

    // 1. Damping
    apply_rigid_body_damping(vel, ang_vel, mut_cfg.ball_drag, 0.0f, dt);

    // 2. Gravity
    vel = vel + mut_cfg.gravity * dt;

    // 3. Arena / Ground collision resolution (updates velocities and pushes out penetration)
    resolve_ball_arena_collision(pos, vel, ang_vel, mut_cfg.ball_radius, mut_cfg.ball_world_restitution, mut_cfg.ball_world_friction, mut_cfg.gravity.z, dt);

    // 4. Linear pos integration with post-collision velocity (in Bullet units for exact rounding parity)
    pos = (pos * 0.02f + vel * (0.02f * dt)) * 50.0f;

    // 5. Rotation integration with post-collision angular velocity
    quat = bullet_integrate_quaternion(quat, ang_vel, dt);

    // Velocity Clamping
    float b_speed = vel.length();
    if (b_speed > mut_cfg.ball_max_speed) {
        vel = vel * (mut_cfg.ball_max_speed / b_speed);
    }
}

} // namespace rocketsim_cuda
