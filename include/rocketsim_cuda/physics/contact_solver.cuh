#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/mat3.cuh"
#include "rocketsim_cuda/physics/arena_sdf.cuh"
#include "rocketsim_cuda/types/car_state.cuh"
#include "rocketsim_cuda/types/ball_state.cuh"
#include "rocketsim_cuda/types/arena_state.cuh"
#include "rocketsim_cuda/types/car_config.cuh"
#include "rocketsim_cuda/types/arena_config.cuh"
#include "rocketsim_cuda/physics/suspension.cuh"

namespace rocketsim_cuda {

// Octane Hitbox Geometry
__device__ __forceinline__ Vec3 get_octane_hitbox_offset() {
    return Vec3(13.8757f, 0.0f, 20.755f);
}

__device__ __forceinline__ Vec3 get_octane_hitbox_half() {
    return Vec3(60.2535f, 43.3497f, 19.32955f);
}

/**
 * @brief Evaluates RocketSim BALL_CAR_EXTRA_IMPULSE_FACTOR_CURVE (RLConst.h:496-503).
 * Piecewise curve points: (0, 0.65), (500, 0.65), (2300, 0.55), (4600, 0.30).
 */
__device__ __forceinline__ float evaluate_ball_car_extra_impulse_factor(float rel_speed) {
    if (rel_speed <= 500.0f) {
        return 0.65f;
    } else if (rel_speed <= 2300.0f) {
        float t = (rel_speed - 500.0f) / 1800.0f;
        return 0.65f - 0.10f * t;
    } else if (rel_speed <= 4600.0f) {
        float t = (rel_speed - 2300.0f) / 2300.0f;
        return 0.55f - 0.25f * t;
    } else {
        return 0.30f;
    }
}

/**
 * @brief Tests collision between an Octane oriented bounding box (OBB) and the ball sphere.
 * 
 * @param car_pos Center of mass position of the car in world space.
 * @param car_basis 3x3 rotation matrix of the car (columns: forward, right, up).
 * @param ball_pos Center position of the ball in world space.
 * @param ball_radius Radius of the ball sphere (default BALL_RADIUS = 91.25f).
 * @param out_normal_world Contact normal pointing from the car toward the ball in world space.
 * @param out_contact_pt_world Contact point on the car surface in world space.
 * @param out_penetration Penetration depth (positive when intersecting).
 * @return true if car and ball are colliding, false otherwise.
 */
__device__ __forceinline__ bool test_car_ball_collision(
    const Vec3& car_pos,
    const Mat3& car_basis,
    const Vec3& hitbox_offset,
    const Vec3& hitbox_half,
    const Vec3& ball_pos,
    float ball_radius,
    Vec3& out_normal_world,
    Vec3& out_contact_pt_world,
    float& out_penetration)
{
    Vec3 hitbox_center = car_pos + car_basis * hitbox_offset;

    constexpr float BOX_MARGIN = 2.0f; // Bullet CONVEX_DISTANCE_MARGIN = 0.04 BT = 2.0 UU
    Vec3 inner_half = hitbox_half - Vec3(BOX_MARGIN, BOX_MARGIN, BOX_MARGIN);

    // 1. Transform ball into car hitbox local frame
    Vec3 d_world = ball_pos - hitbox_center;
    Vec3 p_local = car_basis.transpose() * d_world;

    // 2. Clamp to inner box half-extents (matching btSphereBoxCollisionAlgorithm)
    Vec3 q_inner(
        fmaxf(-inner_half.x, fminf(inner_half.x, p_local.x)),
        fmaxf(-inner_half.y, fminf(inner_half.y, p_local.y)),
        fmaxf(-inner_half.z, fminf(inner_half.z, p_local.z))
    );

    Vec3 diff_local = p_local - q_inner;
    float dist_sq = diff_local.length_sq();
    float total_radius = ball_radius + BOX_MARGIN;

    if (dist_sq > total_radius * total_radius) {
        return false;
    }

    Vec3 normal_local;
    Vec3 q_box;
    if (dist_sq > 1e-8f) {
        float dist = sqrtf(dist_sq);
        normal_local = diff_local * (1.0f / dist);
        out_penetration = total_radius - dist;
        q_box = q_inner + normal_local * BOX_MARGIN;
    } else {
        // Center is inside inner box: project to closest face (mirroring btSphereBoxCollisionAlgorithm::getSpherePenetration)
        float min_dist = inner_half.x - p_local.x;
        normal_local = Vec3(1.0f, 0.0f, 0.0f);
        Vec3 closest_pt = p_local;
        closest_pt.x = inner_half.x;

        float face_dist = inner_half.x + p_local.x;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_pt = p_local;
            closest_pt.x = -inner_half.x;
            normal_local = Vec3(-1.0f, 0.0f, 0.0f);
        }

        face_dist = inner_half.y - p_local.y;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_pt = p_local;
            closest_pt.y = inner_half.y;
            normal_local = Vec3(0.0f, 1.0f, 0.0f);
        }

        face_dist = inner_half.y + p_local.y;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_pt = p_local;
            closest_pt.y = -inner_half.y;
            normal_local = Vec3(0.0f, -1.0f, 0.0f);
        }

        face_dist = inner_half.z - p_local.z;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_pt = p_local;
            closest_pt.z = inner_half.z;
            normal_local = Vec3(0.0f, 0.0f, 1.0f);
        }

        face_dist = inner_half.z + p_local.z;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_pt = p_local;
            closest_pt.z = -inner_half.z;
            normal_local = Vec3(0.0f, 0.0f, -1.0f);
        }

        out_penetration = total_radius + min_dist;
        q_box = closest_pt + normal_local * BOX_MARGIN;
    }

    out_normal_world = car_basis * normal_local;
    out_contact_pt_world = hitbox_center + car_basis * q_box;
    return true;
}

__device__ __forceinline__ bool test_car_ball_collision(
    const Vec3& car_pos,
    const Mat3& car_basis,
    const Vec3& ball_pos,
    float ball_radius,
    Vec3& out_normal_world,
    Vec3& out_contact_pt_world,
    float& out_penetration)
{
    return test_car_ball_collision(
        car_pos, car_basis,
        get_octane_hitbox_offset(), get_octane_hitbox_half(),
        ball_pos, ball_radius,
        out_normal_world, out_contact_pt_world, out_penetration
    );
}

/**
 * @brief Resolves analytical Car-Ball collision with bilateral impulse exchange,
 * RocketSim extra hit impulse curve, velocity limits, and hit tracking state updates.
 */
__device__ __forceinline__ bool resolve_car_ball_collision(
    uint32_t env_idx,
    uint32_t car_idx,
    BallStateSoA& ball_state,
    CarStateSoA& car_state,
    ArenaStateSoA& arena_state,
    float dt,
    const MutatorConfig& mut_cfg,
    const CarConfig& car_cfg)
{
    // Load car pose & velocities
    Vec3 car_pos(car_state.pos_x[car_idx], car_state.pos_y[car_idx], car_state.pos_z[car_idx]);
    Vec3 car_vel(car_state.vel_x[car_idx], car_state.vel_y[car_idx], car_state.vel_z[car_idx]);
    Vec3 car_omega(car_state.ang_vel_x[car_idx], car_state.ang_vel_y[car_idx], car_state.ang_vel_z[car_idx]);
    Quat car_quat(car_state.q_w[car_idx], car_state.q_x[car_idx], car_state.q_y[car_idx], car_state.q_z[car_idx]);
    Mat3 car_basis = Mat3::from_quat(car_quat);

    // Load ball pose & velocities
    Vec3 ball_pos(ball_state.pos_x[env_idx], ball_state.pos_y[env_idx], ball_state.pos_z[env_idx]);
    Vec3 ball_vel(ball_state.vel_x[env_idx], ball_state.vel_y[env_idx], ball_state.vel_z[env_idx]);
    Vec3 ball_omega(ball_state.ang_vel_x[env_idx], ball_state.ang_vel_y[env_idx], ball_state.ang_vel_z[env_idx]);

    Vec3 normal_world;
    Vec3 contact_pt_world;
    float penetration = 0.0f;

    float ball_radius = mut_cfg.ball_radius;
    float ball_mass = mut_cfg.ball_mass;
    float car_mass = mut_cfg.car_mass;
    Vec3 hitbox_offset = car_cfg.hitbox_pos_offset;
    Vec3 hitbox_half = car_cfg.get_hitbox_half();

    if (!test_car_ball_collision(car_pos, car_basis, hitbox_offset, hitbox_half, ball_pos, ball_radius, normal_world, contact_pt_world, penetration)) {
        return false;
    }

    // Hit detected: determine non-consecutive hit for extra hit impulse
    uint64_t current_tick = arena_state.tick_count ? arena_state.tick_count[env_idx] : 0;
    bool is_first_hit = (car_state.ball_hit_is_valid[car_idx] == 0);
    uint64_t last_hit_tick = car_state.ball_hit_tick_count[car_idx];
    bool is_non_consecutive = is_first_hit || (current_tick > last_hit_tick + 1) || (last_hit_tick > current_tick);

    // Compute RocketSim Piecewise Extra Hit Impulse (RLConst.h:135-139, 496-503, Ball.cpp:261-285)
    Vec3 car_forward = car_basis.forward;
    Vec3 rel_pos = ball_pos - car_pos;
    Vec3 rel_vel = ball_vel - car_vel;
    float rel_speed = fminf(rel_vel.length(), 4600.0f); // BALL_CAR_EXTRA_IMPULSE_MAXDELTAVEL_UU

    Vec3 added_vel(0.0f, 0.0f, 0.0f);
    if (is_non_consecutive && rel_speed > 0.0f) {
        float z_scale = 0.35f; // BALL_CAR_EXTRA_IMPULSE_Z_SCALE
        Vec3 scaled_rel_pos(rel_pos.x, rel_pos.y, rel_pos.z * z_scale);
        float scaled_len = scaled_rel_pos.length();
        Vec3 hit_dir = (scaled_len > 1e-6f) ? (scaled_rel_pos * (1.0f / scaled_len)) : Vec3(0.0f, 0.0f, 1.0f);

        // BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE = 0.65f -> (1 - 0.65f) = 0.35f
        Vec3 fwd_adj = car_forward * (hit_dir.dot(car_forward) * (1.0f - 0.65f));
        Vec3 adj_hit_dir = hit_dir - fwd_adj;
        float adj_len = adj_hit_dir.length();
        hit_dir = (adj_len > 1e-6f) ? (adj_hit_dir * (1.0f / adj_len)) : hit_dir;

        float factor = evaluate_ball_car_extra_impulse_factor(rel_speed);
        added_vel = hit_dir * (rel_speed * factor * mut_cfg.ball_hit_extra_force_scale);
    }

    // Contact point lever arms
    Vec3 r_c = contact_pt_world - car_pos;
    Vec3 r_b = normal_world * (-ball_radius);

    // Contact point velocities
    Vec3 v_c_pt = car_vel + car_omega.cross(r_c);
    Vec3 v_b_pt = ball_vel + ball_omega.cross(r_b);
    Vec3 v_rel_pt = v_b_pt - v_c_pt;
    float vn = v_rel_pt.dot(normal_world);

    // Inertia and mass parameters
    float inv_m_c = 1.0f / car_mass;
    float inv_m_b = 1.0f / ball_mass;
    Vec3 inv_I_c_local = car_cfg.calculate_inv_inertia(car_mass);
    float inv_I_b = 1.0f / (0.4f * ball_mass * ball_radius * ball_radius);

    // Bilateral impulse exchange (restitution e = 0.0f, friction mu = 2.0f)
    Vec3 J_total(0.0f, 0.0f, 0.0f);
    if (vn < 0.0f) {
        // Car rotational term for normal direction
        Vec3 u_c_n = r_c.cross(normal_world);
        Vec3 u_c_n_local = car_basis.transpose() * u_c_n;
        Vec3 w_c_n_local(u_c_n_local.x * inv_I_c_local.x, u_c_n_local.y * inv_I_c_local.y, u_c_n_local.z * inv_I_c_local.z);
        float rot_term_c = u_c_n_local.dot(w_c_n_local);

        // Ball rotational term for normal direction
        Vec3 u_b_n = r_b.cross(normal_world);
        float rot_term_b = u_b_n.length_sq() * inv_I_b;

        float Kn = inv_m_c + inv_m_b + rot_term_c + rot_term_b;
        float Jn = (Kn > 1e-8f) ? (-vn / Kn) : 0.0f; // e = 0.0f
        Vec3 normal_impulse = normal_world * Jn;

        // Tangential friction (mu = 2.0f)
        Vec3 v_t = v_rel_pt - normal_world * vn;
        float vt_mag = v_t.length();
        Vec3 tangent_impulse(0.0f, 0.0f, 0.0f);
        if (vt_mag > 1e-5f) {
            Vec3 t = v_t * (1.0f / vt_mag);

            Vec3 u_c_t = r_c.cross(t);
            Vec3 u_c_t_local = car_basis.transpose() * u_c_t;
            Vec3 w_c_t_local(u_c_t_local.x * inv_I_c_local.x, u_c_t_local.y * inv_I_c_local.y, u_c_t_local.z * inv_I_c_local.z);
            float rot_t_c = u_c_t_local.dot(w_c_t_local);

            Vec3 u_b_t = r_b.cross(t);
            float rot_t_b = u_b_t.length_sq() * inv_I_b;

            float Kt = inv_m_c + inv_m_b + rot_t_c + rot_t_b;
            float Jt_desired = (Kt > 1e-8f) ? (vt_mag / Kt) : 0.0f;
            float Jt = fminf(Jt_desired, 2.0f * Jn); // mu = 2.0f
            tangent_impulse = t * (-Jt);
        }

        J_total = normal_impulse + tangent_impulse;
    }

    // Apply impulse to ball
    ball_vel = ball_vel + J_total * inv_m_b;
    Vec3 tau_ball = r_b.cross(J_total);
    ball_omega = ball_omega + tau_ball * inv_I_b;

    // Apply impulse to car (equal and opposite)
    car_vel = car_vel - J_total * inv_m_c;
    Vec3 tau_car = r_c.cross(J_total * (-1.0f));
    Vec3 tau_car_local = car_basis.transpose() * tau_car;
    Vec3 delta_omega_car_local(tau_car_local.x * inv_I_c_local.x, tau_car_local.y * inv_I_c_local.y, tau_car_local.z * inv_I_c_local.z);
    car_omega = car_omega + car_basis * delta_omega_car_local;

    // Apply RocketSim extra hit impulse to ball
    ball_vel = ball_vel + added_vel;

    // Resolve interpenetration: distribute split-impulse penetration push
    if (penetration > 0.0f) {
        float p_push = penetration * 0.8f; // erp2 = 0.8f
        float mass_sum = car_mass + ball_mass;
        float ball_frac = car_mass / mass_sum;
        float car_frac  = ball_mass / mass_sum;

        ball_pos = ball_pos + normal_world * (p_push * ball_frac);
        car_pos  = car_pos  - normal_world * (p_push * car_frac);
    }

    // Velocity Clamping
    float b_speed = ball_vel.length();
    if (b_speed > mut_cfg.ball_max_speed) {
        ball_vel = ball_vel * (mut_cfg.ball_max_speed / b_speed);
    }
    float b_ang_speed = ball_omega.length();
    if (b_ang_speed > BALL_MAX_ANG_SPEED) {
        ball_omega = ball_omega * (BALL_MAX_ANG_SPEED / b_ang_speed);
    }
    float c_speed = car_vel.length();
    if (c_speed > CAR_MAX_SPEED) {
        car_vel = car_vel * (CAR_MAX_SPEED / c_speed);
    }
    float c_ang_speed = car_omega.length();
    if (c_ang_speed > CAR_MAX_ANG_SPEED) {
        car_omega = car_omega * (CAR_MAX_ANG_SPEED / c_ang_speed);
    }

    // Relative position on ball surface
    Vec3 rel_pos_on_ball = normal_world * (-ball_radius);

    // Populate Hit State Tracking
    car_state.ball_hit_is_valid[car_idx] = 1;
    car_state.ball_hit_rel_pos_x[car_idx] = rel_pos_on_ball.x;
    car_state.ball_hit_rel_pos_y[car_idx] = rel_pos_on_ball.y;
    car_state.ball_hit_rel_pos_z[car_idx] = rel_pos_on_ball.z;
    car_state.ball_hit_extra_hit_force_x[car_idx] = added_vel.x;
    car_state.ball_hit_extra_hit_force_y[car_idx] = added_vel.y;
    car_state.ball_hit_extra_hit_force_z[car_idx] = added_vel.z;
    car_state.ball_hit_tick_count[car_idx] = current_tick;

    // Write back updated ball state
    ball_state.pos_x[env_idx] = ball_pos.x;
    ball_state.pos_y[env_idx] = ball_pos.y;
    ball_state.pos_z[env_idx] = ball_pos.z;

    ball_state.vel_x[env_idx] = ball_vel.x;
    ball_state.vel_y[env_idx] = ball_vel.y;
    ball_state.vel_z[env_idx] = ball_vel.z;

    ball_state.ang_vel_x[env_idx] = ball_omega.x;
    ball_state.ang_vel_y[env_idx] = ball_omega.y;
    ball_state.ang_vel_z[env_idx] = ball_omega.z;

    // Write back updated car state
    car_state.pos_x[car_idx] = car_pos.x;
    car_state.pos_y[car_idx] = car_pos.y;
    car_state.pos_z[car_idx] = car_pos.z;

    car_state.vel_x[car_idx] = car_vel.x;
    car_state.vel_y[car_idx] = car_vel.y;
    car_state.vel_z[car_idx] = car_vel.z;

    car_state.ang_vel_x[car_idx] = car_omega.x;
    car_state.ang_vel_y[car_idx] = car_omega.y;
    car_state.ang_vel_z[car_idx] = car_omega.z;

    return true;
}

__device__ __forceinline__ bool resolve_car_ball_collision(
    uint32_t env_idx,
    uint32_t car_idx,
    BallStateSoA& ball_state,
    CarStateSoA& car_state,
    ArenaStateSoA& arena_state,
    float dt)
{
    uint8_t ht = car_state.hitbox_type ? car_state.hitbox_type[car_idx] : 0;
    return resolve_car_ball_collision(
        env_idx, car_idx, ball_state, car_state, arena_state, dt,
        MutatorConfig(), get_car_config(ht)
    );
}

/**
 * @brief Resolves ball collision against arena SDF and floor.
 */
__device__ __forceinline__ void resolve_ball_arena_collision(
    Vec3& pos,
    Vec3& vel,
    Vec3& ang_vel,
    float radius = BALL_RADIUS,
    float restitution = BALL_RESTITUTION,
    float friction = BALL_FRICTION,
    float gravity_z = GRAVITY_Z,
    float dt = 1.0f / 120.0f)
{
    float dist = 0.0f;
    Vec3 normal(0.0f, 0.0f, 1.0f);
    arena_sdf_and_normal(pos, dist, normal);

    if (dist < radius) {
        float penetration = radius - dist;
        pos = pos + normal * penetration;

        float vn = normal.dot(vel);
        if (vn < 0.0f) {
            // Low-speed restitution threshold matching Bullet (0.2 BT units = 10.0 UU/s)
            float e = (fabsf(vn) < 10.0f) ? 0.0f : restitution;
            float delta_vn = -(1.0f + e) * vn;
            Vec3 normal_impulse = normal * delta_vn;

            // Surface contact slip velocity (contact point r = -radius * normal)
            // v_contact = vel + ang_vel x r = vel - (ang_vel x normal) * radius
            Vec3 v_contact = vel - ang_vel.cross(normal) * radius;
            Vec3 v_slip = v_contact - normal * normal.dot(v_contact);
            float slip_speed = v_slip.length();

            Vec3 tangent_impulse(0.0f, 0.0f, 0.0f);
            if (slip_speed > 1e-4f) {
                // Effective tangential mass ratio for rolling solid sphere: 2/7
                float j_slip_per_m = (2.0f / 7.0f) * slip_speed;
                float j_coulomb_max_per_m = friction * delta_vn;

                float j_fric_mag = fminf(j_slip_per_m, j_coulomb_max_per_m);
                Vec3 t_dir = v_slip * (1.0f / slip_speed);
                tangent_impulse = t_dir * (-j_fric_mag);

                // Rotational torque coupling:
                // Delta omega = (r x J_t) / I = (-radius * normal x J_t) / (0.4 * M * radius^2)
                //             = (2.5 / radius) * (tangent_impulse x normal)
                Vec3 delta_omega = tangent_impulse.cross(normal) * (2.5f / radius);
                ang_vel = ang_vel + delta_omega;
            }

            vel = vel + normal_impulse + tangent_impulse;
        }
    }
}

/**
 * @brief Resolves chassis-arena and chassis-ground penetration and records world contact normals.
 */
__device__ __forceinline__ void resolve_chassis_arena_collision(
    uint32_t car_idx,
    CarStateSoA& car_state,
    Vec3& pos,
    Vec3& vel,
    Vec3& omega,
    const Mat3& basis,
    float dt,
    const Vec3& hitbox_offset = get_octane_hitbox_offset(),
    const Vec3& hitbox_half = get_octane_hitbox_half(),
    float car_mass = CAR_MASS,
    const Vec3& inv_inertia_bt = Vec3(1.0f / 54.0f, 1.0f / 96.0f, 1.0f / 132.0f))
{
    bool has_contact = false;
    Vec3 sum_normal(0.0f, 0.0f, 0.0f);
    int num_contacts = 0;
    int contact_indices[8];
    float contact_dists[8];
    Vec3 contact_normals[8];
    Vec3 contact_corners[8];

    constexpr float CHASSIS_MARGIN_UU = 0.04f * 50.0f; // 2.0f UU (CONVEX_DISTANCE_MARGIN)

    // 1. Identify penetrating corners against Arena SDF
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float sx = (i & 1) ? 1.0f : -1.0f;
        float sy = (i & 2) ? 1.0f : -1.0f;
        float sz = (i & 4) ? 1.0f : -1.0f;

        Vec3 local_corner = hitbox_offset + Vec3(
            sx * hitbox_half.x,
            sy * hitbox_half.y,
            sz * hitbox_half.z
        );

        Vec3 world_corner = pos + basis * local_corner;
        float dist = 0.0f;
        Vec3 normal(0.0f, 0.0f, 1.0f);
        arena_sdf_and_normal(world_corner, dist, normal);

        if (dist <= CHASSIS_MARGIN_UU) {
            has_contact = true;
            sum_normal = sum_normal + normal;
            contact_indices[num_contacts] = i;
            contact_dists[num_contacts] = dist;
            contact_normals[num_contacts] = normal;
            contact_corners[num_contacts] = world_corner;
            num_contacts++;
        }
    }

    // 2. Resolve contact impulses matching Bullet resolveSingleCollision
    if (num_contacts > 0) {
        float contact_scale = 1.0f / static_cast<float>(num_contacts);
        Vec3 vel_bt = vel * 0.02f;

        for (int k = 0; k < num_contacts; ++k) {
            float dist = contact_dists[k];
            Vec3 normal = contact_normals[k];
            Vec3 world_corner = contact_corners[k];

            float dist_bt = dist * 0.02f;
            float penetration_bt = 0.04f - dist_bt;
            if (penetration_bt > 0.0f) {
                pos = pos + normal * (penetration_bt * 50.0f * contact_scale);
            }

            Vec3 rel_pos_bt = (world_corner - pos) * 0.02f;
            Vec3 pt_vel_bt = vel_bt + omega.cross(rel_pos_bt);
            float rel_vel = normal.dot(pt_vel_bt);

            if (rel_vel < 0.0f || penetration_bt > 0.0f) {
                float pos_error = (0.2f * penetration_bt) / dt;
                float vel_error = -(1.0f + 0.3f) * rel_vel; // CARWORLD_COLLISION_RESTITUTION = 0.3f

                Vec3 c0 = rel_pos_bt.cross(normal);
                Vec3 c0_loc = basis.transpose() * c0;
                float denom = (1.0f / car_mass) + (c0_loc.x * c0_loc.x * inv_inertia_bt.x
                                                + c0_loc.y * c0_loc.y * inv_inertia_bt.y
                                                + c0_loc.z * c0_loc.z * inv_inertia_bt.z);

                float normal_impulse_bt = fmaxf(0.0f, (pos_error + vel_error) / denom) * contact_scale;
                Vec3 imp_bt = normal * normal_impulse_bt;

                // Coulomb friction: CARWORLD_COLLISION_FRICTION = 0.3f
                Vec3 tangent_vel = pt_vel_bt - normal * rel_vel;
                float tangent_speed = tangent_vel.length();
                if (tangent_speed > 1e-4f && normal_impulse_bt > 0.0f) {
                    Vec3 tangent_dir = tangent_vel * (1.0f / tangent_speed);
                    float friction_impulse_bt = fminf(tangent_speed * car_mass, normal_impulse_bt * 0.3f);
                    imp_bt = imp_bt - tangent_dir * friction_impulse_bt;
                }

                vel_bt = vel_bt + imp_bt * (1.0f / car_mass);
                Vec3 ang_imp_loc = basis.transpose() * (rel_pos_bt.cross(imp_bt));
                Vec3 d_omega_loc(
                    ang_imp_loc.x * inv_inertia_bt.x,
                    ang_imp_loc.y * inv_inertia_bt.y,
                    ang_imp_loc.z * inv_inertia_bt.z
                );
                omega = omega + basis * d_omega_loc;
            }
        }
        vel = vel_bt * 50.0f;
    }

    if (has_contact) {
        Vec3 avg_normal = (sum_normal.length_sq() > 1e-6f) ? sum_normal.normalized() : Vec3(0.0f, 0.0f, 1.0f);
        car_state.world_contact_has_contact[car_idx] = 1;
        car_state.world_contact_normal_x[car_idx]    = avg_normal.x;
        car_state.world_contact_normal_y[car_idx]    = avg_normal.y;
        car_state.world_contact_normal_z[car_idx]    = avg_normal.z;
    } else {
        car_state.world_contact_has_contact[car_idx] = 0;
        car_state.world_contact_normal_x[car_idx]    = 0.0f;
        car_state.world_contact_normal_y[car_idx]    = 0.0f;
        car_state.world_contact_normal_z[car_idx]    = 0.0f;
    }
}

// --- Car-Car Collision Detection & Resolution (ODE dBoxBox2 + Bullet & RocketSim Bump Mechanics) ---

__device__ __forceinline__ void dLineClosestApproachDevice(
    const Vec3& pa, const Vec3& ua,
    const Vec3& pb, const Vec3& ub,
    float& alpha, float& beta)
{
    Vec3 p = pb - pa;
    float uaub = ua.dot(ub);
    float q1 = ua.dot(p);
    float q2 = -ub.dot(p);
    float d = 1.0f - uaub * uaub;
    if (d <= 0.0001f) {
        alpha = 0.0f;
        beta = 0.0f;
    } else {
        float inv_d = 1.0f / d;
        alpha = (q1 + uaub * q2) * inv_d;
        beta = (uaub * q1 + q2) * inv_d;
    }
}

__device__ __forceinline__ int intersectRectQuad2Device(const float h[2], const float p[8], float ret[16]) {
    int nq = 4, nr = 0;
    float buffer[16];
    const float* q = p;
    float* r = ret;
    for (int dir = 0; dir <= 1; dir++) {
        for (int sign = -1; sign <= 1; sign += 2) {
            const float* pq = q;
            float* pr = r;
            nr = 0;
            for (int i = nq; i > 0; i--) {
                if (sign * pq[dir] < h[dir]) {
                    pr[0] = pq[0];
                    pr[1] = pq[1];
                    pr += 2;
                    nr++;
                    if (nr >= 8) { q = r; goto done; }
                }
                const float* nextq = (i > 1) ? pq + 2 : q;
                if ((sign * pq[dir] < h[dir]) ^ (sign * nextq[dir] < h[dir])) {
                    float denom = nextq[dir] - pq[dir];
                    if (fabsf(denom) > 1e-6f) {
                        pr[1 - dir] = pq[1 - dir] + (nextq[1 - dir] - pq[1 - dir]) / denom * (sign * h[dir] - pq[dir]);
                    } else {
                        pr[1 - dir] = pq[1 - dir];
                    }
                    pr[dir] = sign * h[dir];
                    pr += 2;
                    nr++;
                    if (nr >= 8) { q = r; goto done; }
                }
                pq += 2;
            }
            q = r;
            r = (q == ret) ? buffer : ret;
            nq = nr;
        }
    }
done:
    if (q != ret) {
        for (int k = 0; k < nr * 2; k++) ret[k] = q[k];
    }
    return nr;
}

/**
 * @brief Port of Bullet btBoxBoxDetector::dBoxBox2 for car-car OBB contact detection.
 * Cites: libsrc/bullet3-3.24/BulletCollision/CollisionDispatch/btBoxBoxDetector.cpp:272-728
 */
__device__ __forceinline__ bool test_car_car_collision_dBoxBox(
    const Vec3& car1_pos, const Mat3& R1,
    const Vec3& car2_pos, const Mat3& R2,
    Vec3& out_normal, Vec3& out_contact_pt, float& out_penetration, int& out_code)
{
    Vec3 p1 = car1_pos + R1 * get_octane_hitbox_offset();
    Vec3 p2 = car2_pos + R2 * get_octane_hitbox_offset();
    Vec3 A = get_octane_hitbox_half();
    Vec3 B = get_octane_hitbox_half();

    Vec3 p = p2 - p1;
    Vec3 pp = R1.transpose() * p;

    float R11 = R1.forward.dot(R2.forward);
    float R12 = R1.forward.dot(R2.right);
    float R13 = R1.forward.dot(R2.up);

    float R21 = R1.right.dot(R2.forward);
    float R22 = R1.right.dot(R2.right);
    float R23 = R1.right.dot(R2.up);

    float R31 = R1.up.dot(R2.forward);
    float R32 = R1.up.dot(R2.right);
    float R33 = R1.up.dot(R2.up);

    float Q11 = fabsf(R11);
    float Q12 = fabsf(R12);
    float Q13 = fabsf(R13);

    float Q21 = fabsf(R21);
    float Q22 = fabsf(R22);
    float Q23 = fabsf(R23);

    float Q31 = fabsf(R31);
    float Q32 = fabsf(R32);
    float Q33 = fabsf(R33);

    float s = -1e30f;
    int invert_normal = 0;
    int code = 0;
    Vec3 normalR(0.0f, 0.0f, 0.0f);
    Vec3 normalC(0.0f, 0.0f, 0.0f);
    bool use_normalR = false;

    // Separating axes 1..3: u1, u2, u3 (Box 1 axes)
    {
        float expr1 = pp.x;
        float s2 = fabsf(expr1) - (A.x + B.x * Q11 + B.y * Q12 + B.z * Q13);
        if (s2 > 0.0f) return false;
        if (s2 > s) { s = s2; normalR = R1.forward; use_normalR = true; invert_normal = (expr1 < 0.0f); code = 1; }
    }
    {
        float expr1 = pp.y;
        float s2 = fabsf(expr1) - (A.y + B.x * Q21 + B.y * Q22 + B.z * Q23);
        if (s2 > 0.0f) return false;
        if (s2 > s) { s = s2; normalR = R1.right; use_normalR = true; invert_normal = (expr1 < 0.0f); code = 2; }
    }
    {
        float expr1 = pp.z;
        float s2 = fabsf(expr1) - (A.z + B.x * Q31 + B.y * Q32 + B.z * Q33);
        if (s2 > 0.0f) return false;
        if (s2 > s) { s = s2; normalR = R1.up; use_normalR = true; invert_normal = (expr1 < 0.0f); code = 3; }
    }

    // Separating axes 4..6: v1, v2, v3 (Box 2 axes)
    {
        float expr1 = R2.forward.dot(p);
        float s2 = fabsf(expr1) - (A.x * Q11 + A.y * Q21 + A.z * Q31 + B.x);
        if (s2 > 0.0f) return false;
        if (s2 > s) { s = s2; normalR = R2.forward; use_normalR = true; invert_normal = (expr1 < 0.0f); code = 4; }
    }
    {
        float expr1 = R2.right.dot(p);
        float s2 = fabsf(expr1) - (A.x * Q12 + A.y * Q22 + A.z * Q32 + B.y);
        if (s2 > 0.0f) return false;
        if (s2 > s) { s = s2; normalR = R2.right; use_normalR = true; invert_normal = (expr1 < 0.0f); code = 5; }
    }
    {
        float expr1 = R2.up.dot(p);
        float s2 = fabsf(expr1) - (A.x * Q13 + A.y * Q23 + A.z * Q33 + B.z);
        if (s2 > 0.0f) return false;
        if (s2 > s) { s = s2; normalR = R2.up; use_normalR = true; invert_normal = (expr1 < 0.0f); code = 6; }
    }

    constexpr float fudge2 = 1.0e-5f;
    constexpr float fudge_factor = 1.05f;
    constexpr float SIMD_EPSILON = 1.0e-5f;

    Q11 += fudge2; Q12 += fudge2; Q13 += fudge2;
    Q21 += fudge2; Q22 += fudge2; Q23 += fudge2;
    Q31 += fudge2; Q32 += fudge2; Q33 += fudge2;

    auto test_cross = [&](float expr1, float expr2, float n1, float n2, float n3, int cc) -> bool {
        float s2 = fabsf(expr1) - expr2;
        if (s2 > SIMD_EPSILON) return false;
        float l = sqrtf(n1 * n1 + n2 * n2 + n3 * n3);
        if (l > SIMD_EPSILON) {
            s2 /= l;
            if (s2 * fudge_factor > s) {
                s = s2;
                use_normalR = false;
                normalC = Vec3(n1 / l, n2 / l, n3 / l);
                invert_normal = (expr1 < 0.0f);
                code = cc;
            }
        }
        return true;
    };

    // Separating axes 7..9: u1 x (v1, v2, v3)
    if (!test_cross(pp.z * R21 - pp.y * R31, A.y * Q31 + A.z * Q21 + B.y * Q13 + B.z * Q12, 0.0f, -R31, R21, 7)) return false;
    if (!test_cross(pp.z * R22 - pp.y * R32, A.y * Q32 + A.z * Q22 + B.x * Q13 + B.z * Q11, 0.0f, -R32, R22, 8)) return false;
    if (!test_cross(pp.z * R23 - pp.y * R33, A.y * Q33 + A.z * Q23 + B.x * Q12 + B.y * Q11, 0.0f, -R33, R23, 9)) return false;

    // Separating axes 10..12: u2 x (v1, v2, v3)
    if (!test_cross(pp.x * R31 - pp.z * R11, A.x * Q31 + A.z * Q11 + B.y * Q23 + B.z * Q22, R31, 0.0f, -R11, 10)) return false;
    if (!test_cross(pp.x * R32 - pp.z * R12, A.x * Q32 + A.z * Q12 + B.x * Q23 + B.z * Q21, R32, 0.0f, -R12, 11)) return false;
    if (!test_cross(pp.x * R33 - pp.z * R13, A.x * Q33 + A.z * Q13 + B.x * Q22 + B.y * Q21, R33, 0.0f, -R13, 12)) return false;

    // Separating axes 13..15: u3 x (v1, v2, v3)
    if (!test_cross(pp.y * R11 - pp.x * R21, A.x * Q21 + A.y * Q11 + B.y * Q33 + B.z * Q32, -R21, R11, 0.0f, 13)) return false;
    if (!test_cross(pp.y * R12 - pp.x * R22, A.x * Q22 + A.y * Q12 + B.x * Q33 + B.z * Q31, -R22, R12, 0.0f, 14)) return false;
    if (!test_cross(pp.y * R13 - pp.x * R23, A.x * Q23 + A.y * Q13 + B.x * Q32 + B.y * Q31, -R23, R13, 0.0f, 15)) return false;

    if (!code) return false;

    Vec3 normal;
    if (use_normalR) {
        normal = normalR;
    } else {
        normal = R1 * normalC;
    }
    if (invert_normal) {
        normal = normal * (-1.0f);
    }
    out_penetration = -s;
    out_normal = normal;
    out_code = code;

    // Compute contact point
    if (code > 6) {
        // Edge-edge contact (btBoxBoxDetector.cpp:430-476)
        Vec3 pa = p1;
        float sign0 = (normal.dot(R1.forward) > 0.0f) ? 1.0f : -1.0f;
        float sign1 = (normal.dot(R1.right) > 0.0f) ? 1.0f : -1.0f;
        float sign2 = (normal.dot(R1.up) > 0.0f) ? 1.0f : -1.0f;
        pa = pa + R1.forward * (sign0 * A.x) + R1.right * (sign1 * A.y) + R1.up * (sign2 * A.z);

        Vec3 pb = p2;
        float signB0 = (normal.dot(R2.forward) > 0.0f) ? -1.0f : 1.0f;
        float signB1 = (normal.dot(R2.right) > 0.0f) ? -1.0f : 1.0f;
        float signB2 = (normal.dot(R2.up) > 0.0f) ? -1.0f : 1.0f;
        pb = pb + R2.forward * (signB0 * B.x) + R2.right * (signB1 * B.y) + R2.up * (signB2 * B.z);

        int edgeA_idx = (code - 7) / 3;
        int edgeB_idx = (code - 7) % 3;
        Vec3 ua = (edgeA_idx == 0) ? R1.forward : ((edgeA_idx == 1) ? R1.right : R1.up);
        Vec3 ub = (edgeB_idx == 0) ? R2.forward : ((edgeB_idx == 1) ? R2.right : R2.up);

        float alpha = 0.0f, beta = 0.0f;
        dLineClosestApproachDevice(pa, ua, pb, ub, alpha, beta);
        pa = pa + ua * alpha;
        pb = pb + ub * beta;

        out_contact_pt = pb;
    } else {
        // Face contact (btBoxBoxDetector.cpp:484-685)
        Mat3 Ra = (code <= 3) ? R1 : R2;
        Mat3 Rb = (code <= 3) ? R2 : R1;
        Vec3 pa = (code <= 3) ? p1 : p2;
        Vec3 pb = (code <= 3) ? p2 : p1;
        Vec3 Sa = (code <= 3) ? A : B;
        Vec3 Sb = (code <= 3) ? B : A;

        Vec3 normal2 = (code <= 3) ? normal : (normal * (-1.0f));
        Vec3 nr = Rb.transpose() * normal2;
        Vec3 anr(fabsf(nr.x), fabsf(nr.y), fabsf(nr.z));

        int lanr = 0, a1 = 1, a2 = 2;
        if (anr.y > anr.x) {
            if (anr.y > anr.z) {
                a1 = 0; lanr = 1; a2 = 2;
            } else {
                a1 = 0; a2 = 1; lanr = 2;
            }
        } else {
            if (anr.x > anr.z) {
                lanr = 0; a1 = 1; a2 = 2;
            } else {
                a1 = 0; a2 = 1; lanr = 2;
            }
        }

        Vec3 Rb_lanr = (lanr == 0) ? Rb.forward : ((lanr == 1) ? Rb.right : Rb.up);
        float nr_lanr_val = (lanr == 0) ? nr.x : ((lanr == 1) ? nr.y : nr.z);
        float Sb_lanr = (lanr == 0) ? Sb.x : ((lanr == 1) ? Sb.y : Sb.z);

        Vec3 center = (nr_lanr_val < 0.0f)
            ? (pb - pa + Rb_lanr * Sb_lanr)
            : (pb - pa - Rb_lanr * Sb_lanr);

        int codeN = (code <= 3) ? (code - 1) : (code - 4);
        int code1 = (codeN == 0) ? 1 : 0;
        int code2 = (codeN == 2) ? 1 : 2;

        Vec3 Ra_code1 = (code1 == 0) ? Ra.forward : ((code1 == 1) ? Ra.right : Ra.up);
        Vec3 Ra_code2 = (code2 == 0) ? Ra.forward : ((code2 == 1) ? Ra.right : Ra.up);
        Vec3 Rb_a1    = (a1 == 0) ? Rb.forward : ((a1 == 1) ? Rb.right : Rb.up);
        Vec3 Rb_a2    = (a2 == 0) ? Rb.forward : ((a2 == 1) ? Rb.right : Rb.up);

        float c1 = center.dot(Ra_code1);
        float c2 = center.dot(Ra_code2);
        float m11 = Ra_code1.dot(Rb_a1);
        float m12 = Ra_code1.dot(Rb_a2);
        float m21 = Ra_code2.dot(Rb_a1);
        float m22 = Ra_code2.dot(Rb_a2);

        float Sb_a1 = (a1 == 0) ? Sb.x : ((a1 == 1) ? Sb.y : Sb.z);
        float Sb_a2 = (a2 == 0) ? Sb.x : ((a2 == 1) ? Sb.y : Sb.z);

        float k1 = m11 * Sb_a1;
        float k2 = m21 * Sb_a1;
        float k3 = m12 * Sb_a2;
        float k4 = m22 * Sb_a2;

        float quad[8];
        quad[0] = c1 - k1 - k3; quad[1] = c2 - k2 - k4;
        quad[2] = c1 - k1 + k3; quad[3] = c2 - k2 + k4;
        quad[4] = c1 + k1 + k3; quad[5] = c2 + k2 + k4;
        quad[6] = c1 + k1 - k3; quad[7] = c2 + k2 - k4;

        float rect[2];
        rect[0] = (code1 == 0) ? Sa.x : ((code1 == 1) ? Sa.y : Sa.z);
        rect[1] = (code2 == 0) ? Sa.x : ((code2 == 1) ? Sa.y : Sa.z);

        float ret[16];
        int n = intersectRectQuad2Device(rect, quad, ret);

        if (n < 1) {
            out_contact_pt = center + pa;
        } else {
            float det = m11 * m22 - m12 * m21;
            float det1 = (fabsf(det) > 1e-6f) ? (1.0f / det) : 1.0f;
            float inv_m11 = m11 * det1;
            float inv_m12 = m12 * det1;
            float inv_m21 = m21 * det1;
            float inv_m22 = m22 * det1;

            float maxdepth = -1e30f;
            Vec3 deepest_pt = center;
            float Sa_codeN = (codeN == 0) ? Sa.x : ((codeN == 1) ? Sa.y : Sa.z);

            for (int j = 0; j < n; j++) {
                float kk1 = inv_m22 * (ret[j * 2] - c1) - inv_m12 * (ret[j * 2 + 1] - c2);
                float kk2 = -inv_m21 * (ret[j * 2] - c1) + inv_m11 * (ret[j * 2 + 1] - c2);
                Vec3 pt = center + Rb_a1 * kk1 + Rb_a2 * kk2;
                float dep = Sa_codeN - normal2.dot(pt);
                if (dep > maxdepth) {
                    maxdepth = dep;
                    deepest_pt = pt;
                }
            }

            if (code <= 3) {
                out_contact_pt = deepest_pt + pa;
            } else {
                out_contact_pt = deepest_pt + pa - normal * maxdepth;
            }
        }
    }

    return true;
}

/**
 * @brief Evaluates bump velocity amount curves (RLConst.h:505-520).
 */
__device__ __forceinline__ float evaluate_bump_vel_amount(float speed, bool ground_hit) {
    if (ground_hit) {
        if (speed <= 0.0f) return 5.0f / 6.0f;
        if (speed < 1400.0f) {
            float t = speed / 1400.0f;
            return (5.0f / 6.0f) + (1100.0f - (5.0f / 6.0f)) * t;
        }
        if (speed < 2200.0f) {
            float t = (speed - 1400.0f) / 800.0f;
            return 1100.0f + (1530.0f - 1100.0f) * t;
        }
        return 1530.0f;
    } else {
        if (speed <= 0.0f) return 5.0f / 6.0f;
        if (speed < 1400.0f) {
            float t = speed / 1400.0f;
            return (5.0f / 6.0f) + (1390.0f - (5.0f / 6.0f)) * t;
        }
        if (speed < 2200.0f) {
            float t = (speed - 1400.0f) / 800.0f;
            return 1390.0f + (1945.0f - 1390.0f) * t;
        }
        return 1945.0f;
    }
}

/**
 * @brief Evaluates bump upward velocity amount curve (RLConst.h:521-527).
 */
__device__ __forceinline__ float evaluate_bump_upward_vel_amount(float speed) {
    if (speed <= 0.0f) return 2.0f / 6.0f;
    if (speed < 1400.0f) {
        float t = speed / 1400.0f;
        return (2.0f / 6.0f) + (278.0f - (2.0f / 6.0f)) * t;
    }
    if (speed < 2200.0f) {
        float t = (speed - 1400.0f) / 800.0f;
        return 278.0f + (417.0f - 278.0f) * t;
    }
    return 417.0f;
}

/**
 * @brief Resolves collision and bump impulses between two cars.
 * Cites: src/Sim/Arena/Arena.cpp:323-405, src/Sim/Car/Car.cpp:184-203, RLConst.h:40-41, 144-145, 505-527
 */
__device__ __forceinline__ bool resolve_car_car_collision(
    uint32_t car1_idx,
    uint32_t car2_idx,
    CarStateSoA& car_state,
    float dt)
{
    Vec3 pos1(car_state.pos_x[car1_idx], car_state.pos_y[car1_idx], car_state.pos_z[car1_idx]);
    Vec3 vel1(car_state.vel_x[car1_idx], car_state.vel_y[car1_idx], car_state.vel_z[car1_idx]);
    Vec3 omega1(car_state.ang_vel_x[car1_idx], car_state.ang_vel_y[car1_idx], car_state.ang_vel_z[car1_idx]);
    Quat quat1(car_state.q_w[car1_idx], car_state.q_x[car1_idx], car_state.q_y[car1_idx], car_state.q_z[car1_idx]);
    Mat3 basis1 = Mat3::from_quat(quat1);

    Vec3 pos2(car_state.pos_x[car2_idx], car_state.pos_y[car2_idx], car_state.pos_z[car2_idx]);
    Vec3 vel2(car_state.vel_x[car2_idx], car_state.vel_y[car2_idx], car_state.vel_z[car2_idx]);
    Vec3 omega2(car_state.ang_vel_x[car2_idx], car_state.ang_vel_y[car2_idx], car_state.ang_vel_z[car2_idx]);
    Quat quat2(car_state.q_w[car2_idx], car_state.q_x[car2_idx], car_state.q_y[car2_idx], car_state.q_z[car2_idx]);
    Mat3 basis2 = Mat3::from_quat(quat2);

    Vec3 normal_world;
    Vec3 contact_pt_world;
    float penetration = 0.0f;
    int code = 0;

    if (!test_car_car_collision_dBoxBox(pos1, basis1, pos2, basis2, normal_world, contact_pt_world, penetration, code)) {
        return false;
    }

    // Check front bumper contact (Arena.cpp:358-360: localPoint.x > BUMP_MIN_FORWARD_DIST = 64.5 UU)
    Vec3 local1 = basis1.transpose() * (contact_pt_world - pos1);
    bool hitWithBumper1 = (local1.x > 64.5f);

    Vec3 local2 = basis2.transpose() * (contact_pt_world - pos2);
    bool hitWithBumper2 = (local2.x > 64.5f);

    Vec3 bump_impulse_car1(0.0f, 0.0f, 0.0f);
    Vec3 bump_impulse_car2(0.0f, 0.0f, 0.0f);

    int32_t car1_other_id = car_state.car_contact_other_car_id ? car_state.car_contact_other_car_id[car1_idx] : -1;
    float car1_cooldown = car_state.car_contact_cooldown_timer ? car_state.car_contact_cooldown_timer[car1_idx] : 0.0f;
    int32_t car2_id = (int32_t)car2_idx + 1;

    // Car 1 bumps Car 2
    if (!(car1_other_id == car2_id && car1_cooldown > 0.0f)) {
        Vec3 delta_pos = pos2 - pos1;
        if (vel1.dot(delta_pos) > 0.0f) {
            float dist = delta_pos.length();
            Vec3 dir_to_other = (dist > 1e-4f) ? (delta_pos * (1.0f / dist)) : Vec3(0.0f, 0.0f, 1.0f);
            float speed1 = vel1.length();
            Vec3 vel_dir = (speed1 > 1e-4f) ? (vel1 * (1.0f / speed1)) : dir_to_other;

            float speed_towards = vel1.dot(dir_to_other);
            float other_away_speed = vel2.dot(vel_dir);

            if (speed_towards > other_away_speed && hitWithBumper1) {
                bool ground_hit = (car_state.is_on_ground[car2_idx] != 0);
                float base_scale = evaluate_bump_vel_amount(speed_towards, ground_hit);
                Vec3 hit_up_dir = ground_hit ? basis2.up : Vec3(0.0f, 0.0f, 1.0f);
                float up_scale = evaluate_bump_upward_vel_amount(speed_towards);

                bump_impulse_car2 = vel_dir * base_scale + hit_up_dir * up_scale;

                if (car_state.car_contact_other_car_id) car_state.car_contact_other_car_id[car1_idx] = car2_id;
                if (car_state.car_contact_cooldown_timer) car_state.car_contact_cooldown_timer[car1_idx] = 0.25f;
            }
        }
    }

    // Car 2 bumps Car 1
    int32_t car2_other_id = car_state.car_contact_other_car_id ? car_state.car_contact_other_car_id[car2_idx] : -1;
    float car2_cooldown = car_state.car_contact_cooldown_timer ? car_state.car_contact_cooldown_timer[car2_idx] : 0.0f;
    int32_t car1_id = (int32_t)car1_idx + 1;

    if (!(car2_other_id == car1_id && car2_cooldown > 0.0f)) {
        Vec3 delta_pos = pos1 - pos2;
        if (vel2.dot(delta_pos) > 0.0f) {
            float dist = delta_pos.length();
            Vec3 dir_to_other = (dist > 1e-4f) ? (delta_pos * (1.0f / dist)) : Vec3(0.0f, 0.0f, 1.0f);
            float speed2 = vel2.length();
            Vec3 vel_dir = (speed2 > 1e-4f) ? (vel2 * (1.0f / speed2)) : dir_to_other;

            float speed_towards = vel2.dot(dir_to_other);
            float other_away_speed = vel1.dot(vel_dir);

            if (speed_towards > other_away_speed && hitWithBumper2) {
                bool ground_hit = (car_state.is_on_ground[car1_idx] != 0);
                float base_scale = evaluate_bump_vel_amount(speed_towards, ground_hit);
                Vec3 hit_up_dir = ground_hit ? basis1.up : Vec3(0.0f, 0.0f, 1.0f);
                float up_scale = evaluate_bump_upward_vel_amount(speed_towards);

                bump_impulse_car1 = vel_dir * base_scale + hit_up_dir * up_scale;

                if (car_state.car_contact_other_car_id) car_state.car_contact_other_car_id[car2_idx] = car1_id;
                if (car_state.car_contact_cooldown_timer) car_state.car_contact_cooldown_timer[car2_idx] = 0.25f;
            }
        }
    }

    // Bilateral contact impulse resolution
    Vec3 r1 = contact_pt_world - pos1;
    Vec3 r2 = contact_pt_world - pos2;

    Vec3 v1_pt = vel1 + omega1.cross(r1);
    Vec3 v2_pt = vel2 + omega2.cross(r2);
    Vec3 v_rel = v2_pt - v1_pt;
    float vn = v_rel.dot(normal_world);

    Vec3 inv_I_local = get_octane_inv_inertia_local();
    float inv_m = 1.0f / CAR_MASS;

    if (vn < 0.0f) {
        Vec3 u1_n = r1.cross(normal_world);
        Vec3 u1_n_local = basis1.transpose() * u1_n;
        Vec3 w1_n_local(u1_n_local.x * inv_I_local.x, u1_n_local.y * inv_I_local.y, u1_n_local.z * inv_I_local.z);
        float rot1 = u1_n_local.dot(w1_n_local);

        Vec3 u2_n = r2.cross(normal_world);
        Vec3 u2_n_local = basis2.transpose() * u2_n;
        Vec3 w2_n_local(u2_n_local.x * inv_I_local.x, u2_n_local.y * inv_I_local.y, u2_n_local.z * inv_I_local.z);
        float rot2 = u2_n_local.dot(w2_n_local);

        float Kn = inv_m + inv_m + rot1 + rot2;
        constexpr float e = 0.1f; // CARCAR_COLLISION_RESTITUTION (RLConst.h:41)
        float Jn = (Kn > 1e-8f) ? (-(1.0f + e) * vn / Kn) : 0.0f;
        Vec3 normal_impulse = normal_world * Jn;

        constexpr float mu = 0.09f; // CARCAR_COLLISION_FRICTION (RLConst.h:40)
        Vec3 v_t = v_rel - normal_world * vn;
        float vt_mag = v_t.length();
        Vec3 tangent_impulse(0.0f, 0.0f, 0.0f);
        if (vt_mag > 1e-5f) {
            Vec3 t_dir = v_t * (1.0f / vt_mag);

            Vec3 u1_t = r1.cross(t_dir);
            Vec3 u1_t_local = basis1.transpose() * u1_t;
            Vec3 w1_t_local(u1_t_local.x * inv_I_local.x, u1_t_local.y * inv_I_local.y, u1_t_local.z * inv_I_local.z);
            float rot1_t = u1_t_local.dot(w1_t_local);

            Vec3 u2_t = r2.cross(t_dir);
            Vec3 u2_t_local = basis2.transpose() * u2_t;
            Vec3 w2_t_local(u2_t_local.x * inv_I_local.x, u2_t_local.y * inv_I_local.y, u2_t_local.z * inv_I_local.z);
            float rot2_t = u2_t_local.dot(w2_t_local);

            float Kt = inv_m + inv_m + rot1_t + rot2_t;
            float Jt_desired = (Kt > 1e-8f) ? (vt_mag / Kt) : 0.0f;
            float Jt = fminf(Jt_desired, mu * Jn);
            tangent_impulse = t_dir * (-Jt);
        }

        Vec3 J_total = normal_impulse + tangent_impulse;

        vel1 = vel1 - J_total * inv_m;
        Vec3 tau1 = r1.cross(J_total * (-1.0f));
        Vec3 tau1_local = basis1.transpose() * tau1;
        Vec3 dw1_local(tau1_local.x * inv_I_local.x, tau1_local.y * inv_I_local.y, tau1_local.z * inv_I_local.z);
        omega1 = omega1 + basis1 * dw1_local;

        vel2 = vel2 + J_total * inv_m;
        Vec3 tau2 = r2.cross(J_total);
        Vec3 tau2_local = basis2.transpose() * tau2;
        Vec3 dw2_local(tau2_local.x * inv_I_local.x, tau2_local.y * inv_I_local.y, tau2_local.z * inv_I_local.z);
        omega2 = omega2 + basis2 * dw2_local;
    }

    if (penetration > 0.0f) {
        float push = penetration * 0.5f * 0.8f;
        pos1 = pos1 - normal_world * push;
        pos2 = pos2 + normal_world * push;
    }

    // Apply bump impulse cache (Car::_FinishPhysicsTick)
    vel1 = vel1 + bump_impulse_car1;
    vel2 = vel2 + bump_impulse_car2;

    // Velocity limits
    float spd1 = vel1.length();
    if (spd1 > CAR_MAX_SPEED) vel1 = vel1 * (CAR_MAX_SPEED / spd1);
    float ang_spd1 = omega1.length();
    if (ang_spd1 > CAR_MAX_ANG_SPEED) omega1 = omega1 * (CAR_MAX_ANG_SPEED / ang_spd1);

    float spd2 = vel2.length();
    if (spd2 > CAR_MAX_SPEED) vel2 = vel2 * (CAR_MAX_SPEED / spd2);
    float ang_spd2 = omega2.length();
    if (ang_spd2 > CAR_MAX_ANG_SPEED) omega2 = omega2 * (CAR_MAX_ANG_SPEED / ang_spd2);

    // Write back SoA
    car_state.pos_x[car1_idx] = pos1.x;
    car_state.pos_y[car1_idx] = pos1.y;
    car_state.pos_z[car1_idx] = pos1.z;
    car_state.pos_bt_x[car1_idx] = pos1.x * 0.02f;
    car_state.pos_bt_y[car1_idx] = pos1.y * 0.02f;
    car_state.pos_bt_z[car1_idx] = pos1.z * 0.02f;
    car_state.vel_x[car1_idx] = vel1.x;
    car_state.vel_y[car1_idx] = vel1.y;
    car_state.vel_z[car1_idx] = vel1.z;
    car_state.ang_vel_x[car1_idx] = omega1.x;
    car_state.ang_vel_y[car1_idx] = omega1.y;
    car_state.ang_vel_z[car1_idx] = omega1.z;

    car_state.pos_x[car2_idx] = pos2.x;
    car_state.pos_y[car2_idx] = pos2.y;
    car_state.pos_z[car2_idx] = pos2.z;
    car_state.pos_bt_x[car2_idx] = pos2.x * 0.02f;
    car_state.pos_bt_y[car2_idx] = pos2.y * 0.02f;
    car_state.pos_bt_z[car2_idx] = pos2.z * 0.02f;
    car_state.vel_x[car2_idx] = vel2.x;
    car_state.vel_y[car2_idx] = vel2.y;
    car_state.vel_z[car2_idx] = vel2.z;
    car_state.ang_vel_x[car2_idx] = omega2.x;
    car_state.ang_vel_y[car2_idx] = omega2.y;
    car_state.ang_vel_z[car2_idx] = omega2.z;

    return true;
}

} // namespace rocketsim_cuda

