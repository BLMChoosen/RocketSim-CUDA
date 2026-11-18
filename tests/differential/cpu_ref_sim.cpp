#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "cpu_ref_sim.h"

// RocketSim CPU & Bullet headers
#include "RocketSim.h"
#include "Sim/Arena/Arena.h"
#include "Sim/Car/Car.h"
#include "Sim/Ball/Ball.h"
#include "RLConst.h"
#include "BulletLink.h"
#include "BulletCollision/CollisionShapes/btStaticPlaneShape.h"
#include "BulletCollision/CollisionShapes/btTriangleMesh.h"
#include "BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h"
#include "BulletCollision/CollisionDispatch/btInternalEdgeUtility.h"
#include "LinearMath/btQuaternion.h"
#include "LinearMath/btMatrix3x3.h"
#include "Sim/BoostPad/BoostPad.h"

#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <mutex>

namespace rocketsim_cuda {

namespace {
    static bool s_rocketSimInitialized = false;

    void EnsureRocketSimInit() {
        if (!s_rocketSimInitialized) {
            RocketSim::Init("", true);
            s_rocketSimInitialized = true;
        }
    }

    static std::unordered_map<const CPURefSim*, RocketSim::Arena*> s_simArenaMap;
    static std::mutex s_simArenaMapMutex;
}

bool GetCPURefSimBoostPadState(const CPURefSim* sim, int padIdx, bool& isActive, float& cooldown) {
    std::lock_guard<std::mutex> lock(s_simArenaMapMutex);
    auto it = s_simArenaMap.find(sim);
    if (it != s_simArenaMap.end() && it->second) {
        auto arena = it->second;
        if (padIdx >= 0 && padIdx < static_cast<int>(arena->_boostPads.size())) {
            auto state = arena->_boostPads[padIdx]->GetState();
            isActive = state.isActive;
            cooldown = state.cooldown;
            return true;
        }
    }
    return false;
}

CPURefSim::CPURefSim(int numCars, bool addFloor, float tickRate, int spawnSeed, int hitboxType)
    : m_numCars(numCars), m_addFloor(addFloor), m_tickRate(tickRate), m_spawnSeed(spawnSeed), m_hitboxType(hitboxType) {
    EnsureRocketSimInit();
    InitArena();
}

CPURefSim::~CPURefSim() {
    CleanupArena();
}

CPURefSim::CPURefSim(CPURefSim&& other) noexcept
    : m_arena(other.m_arena),
      m_arenaMesh(other.m_arenaMesh),
      m_arenaMeshShape(other.m_arenaMeshShape),
      m_arenaMeshInfoMap(other.m_arenaMeshInfoMap),
      m_cars(std::move(other.m_cars)),
      m_numCars(other.m_numCars),
      m_addFloor(other.m_addFloor),
      m_tickRate(other.m_tickRate),
      m_spawnSeed(other.m_spawnSeed),
      m_hitboxType(other.m_hitboxType) {
    {
        std::lock_guard<std::mutex> lock(s_simArenaMapMutex);
        s_simArenaMap.erase(&other);
        s_simArenaMap[this] = m_arena;
    }
    other.m_arena = nullptr;
    other.m_arenaMesh = nullptr;
    other.m_arenaMeshShape = nullptr;
    other.m_arenaMeshInfoMap = nullptr;
    other.m_cars.clear();
}

CPURefSim& CPURefSim::operator=(CPURefSim&& other) noexcept {
    if (this != &other) {
        CleanupArena();
        m_arena = other.m_arena;
        m_arenaMesh = other.m_arenaMesh;
        m_arenaMeshShape = other.m_arenaMeshShape;
        m_arenaMeshInfoMap = other.m_arenaMeshInfoMap;
        m_cars = std::move(other.m_cars);
        m_numCars = other.m_numCars;
        m_addFloor = other.m_addFloor;
        m_tickRate = other.m_tickRate;
        m_spawnSeed = other.m_spawnSeed;
        m_hitboxType = other.m_hitboxType;
        {
            std::lock_guard<std::mutex> lock(s_simArenaMapMutex);
            s_simArenaMap.erase(&other);
            s_simArenaMap[this] = m_arena;
        }
        other.m_arena = nullptr;
        other.m_arenaMesh = nullptr;
        other.m_arenaMeshShape = nullptr;
        other.m_arenaMeshInfoMap = nullptr;
        other.m_cars.clear();
    }
    return *this;
}

void CPURefSim::InitArena() {
    // GameMode::THE_VOID bypasses mesh loading entirely
    m_arena = RocketSim::Arena::Create(RocketSim::GameMode::THE_VOID, {}, m_tickRate);
    if (!m_arena) {
        throw std::runtime_error("Failed to create RocketSim CPU Arena in GameMode::THE_VOID");
    }

    if (m_addFloor) {
        auto fnAddStaticPlane = [&](const btVector3& normal, const btVector3& posBT) {
            btCollisionShape* planeShape = new btStaticPlaneShape(normal, 0);
            btRigidBody* rb = m_arena->_AddStaticCollisionShape(planeShape, posBT);
            if (rb) {
                rb->setRestitution(RocketSim::RLConst::ARENA_COLLISION_BASE_RESTITUTION);
                rb->setFriction(RocketSim::RLConst::ARENA_COLLISION_BASE_FRICTION);
                rb->setRollingFriction(0.0f);
            }
        };

        // Floor (z = 0)
        fnAddStaticPlane(btVector3(0, 0, 1), btVector3(0, 0, 0));

        // Ceiling (z = 2048 UU)
        fnAddStaticPlane(btVector3(0, 0, -1), btVector3(0, 0, RocketSim::RLConst::ARENA_HEIGHT * UU_TO_BT));

        // Procedural collision mesh for curved transition ramps and walls
        // Matching the analytical arena SDF (R = 260 UU fillets at floor and ceiling)
        m_arenaMesh = new btTriangleMesh();

        constexpr float R = 260.0f;
        constexpr float H = RocketSim::RLConst::ARENA_HEIGHT; // 2048.0f
        constexpr int NUM_SEGS = 64;
        constexpr float PI = 3.14159265358979323846f;
        constexpr float HALF_PI = PI * 0.5f;

        // 8 outer perimeter vertices (counter-clockwise)
        struct Vec2 { float x, y; };
        const Vec2 V[8] = {
            { 4096.0f,  3968.0f}, // V0: side +X to chamfer +X+Y
            { 2944.0f,  5120.0f}, // V1: chamfer +X+Y to back +Y
            {-2944.0f,  5120.0f}, // V2: back +Y to chamfer -X+Y
            {-4096.0f,  3968.0f}, // V3: chamfer -X+Y to side -X
            {-4096.0f, -3968.0f}, // V4: side -X to chamfer -X-Y
            {-2944.0f, -5120.0f}, // V5: chamfer -X-Y to back -Y
            { 2944.0f, -5120.0f}, // V6: back -Y to chamfer +X-Y
            { 4096.0f, -3968.0f}  // V7: chamfer +X-Y to side +X
        };

        constexpr float INV_SQRT2 = 0.7071067811865475f;
        const Vec2 N_side[8] = {
            {-1.0f, 0.0f},
            {-INV_SQRT2, -INV_SQRT2},
            {0.0f, -1.0f},
            {INV_SQRT2, -INV_SQRT2},
            {1.0f, 0.0f},
            {INV_SQRT2, INV_SQRT2},
            {0.0f, 1.0f},
            {-INV_SQRT2, INV_SQRT2}
        };

        constexpr float MITER_SCALE = 2.0f - 1.4142135623730951f; // 2 - sqrt(2)
        Vec2 miter_scaled[8];
        for (int k = 0; k < 8; k++) {
            int next_side = (k + 1) % 8;
            miter_scaled[k].x = (N_side[k].x + N_side[next_side].x) * MITER_SCALE;
            miter_scaled[k].y = (N_side[k].y + N_side[next_side].y) * MITER_SCALE;
        }

        auto get_offset_vertex = [&](int k, float d, float z) -> btVector3 {
            float vx = (V[k].x + miter_scaled[k].x * d) * UU_TO_BT;
            float vy = (V[k].y + miter_scaled[k].y * d) * UU_TO_BT;
            float vz = z * UU_TO_BT;
            return btVector3(vx, vy, vz);
        };

        auto add_quad = [&](const btVector3& v00, const btVector3& v10,
                            const btVector3& v01, const btVector3& v11) {
            // Inward winding for counter-clockwise perimeter
            m_arenaMesh->addTriangle(v00, v01, v10, true);
            m_arenaMesh->addTriangle(v10, v01, v11, true);
        };

        // 1. Bottom curved ramp (fillet from floor z=0 to wall z=R)
        for (int i = 0; i < NUM_SEGS; i++) {
            float theta0 = HALF_PI * (float(i) / float(NUM_SEGS));
            float theta1 = HALF_PI * (float(i + 1) / float(NUM_SEGS));

            float d0 = R * (1.0f - sinf(theta0));
            float z0 = R * (1.0f - cosf(theta0));
            float d1 = R * (1.0f - sinf(theta1));
            float z1 = R * (1.0f - cosf(theta1));

            for (int k = 0; k < 8; k++) {
                int k_start = (k + 7) % 8;
                int k_end = k;
                btVector3 v00 = get_offset_vertex(k_start, d0, z0);
                btVector3 v10 = get_offset_vertex(k_end, d0, z0);
                btVector3 v01 = get_offset_vertex(k_start, d1, z1);
                btVector3 v11 = get_offset_vertex(k_end, d1, z1);
                add_quad(v00, v10, v01, v11);
            }
        }

        // 2. Vertical wall section (from z = R to z = H - R, d = 0)
        for (int k = 0; k < 8; k++) {
            int k_start = (k + 7) % 8;
            int k_end = k;
            btVector3 v00 = get_offset_vertex(k_start, 0.0f, R);
            btVector3 v10 = get_offset_vertex(k_end, 0.0f, R);
            btVector3 v01 = get_offset_vertex(k_start, 0.0f, H - R);
            btVector3 v11 = get_offset_vertex(k_end, 0.0f, H - R);
            add_quad(v00, v10, v01, v11);
        }

        // 3. Top curved ramp (fillet from wall z=H-R to ceiling z=H)
        for (int i = 0; i < NUM_SEGS; i++) {
            float theta0 = HALF_PI * (float(i) / float(NUM_SEGS));
            float theta1 = HALF_PI * (float(i + 1) / float(NUM_SEGS));

            float d0 = R * (1.0f - cosf(theta0));
            float z0 = (H - R) + R * sinf(theta0);
            float d1 = R * (1.0f - cosf(theta1));
            float z1 = (H - R) + R * sinf(theta1);

            for (int k = 0; k < 8; k++) {
                int k_start = (k + 7) % 8;
                int k_end = k;
                btVector3 v00 = get_offset_vertex(k_start, d0, z0);
                btVector3 v10 = get_offset_vertex(k_end, d0, z0);
                btVector3 v01 = get_offset_vertex(k_start, d1, z1);
                btVector3 v11 = get_offset_vertex(k_end, d1, z1);
                add_quad(v00, v10, v01, v11);
            }
        }

        m_arenaMeshShape = new btBvhTriangleMeshShape(m_arenaMesh, true);
        m_arenaMeshInfoMap = new btTriangleInfoMap();
        btGenerateInternalEdgeInfo(m_arenaMeshShape, m_arenaMeshInfoMap);
        m_arenaMeshShape->setTriangleInfoMap(m_arenaMeshInfoMap);

        btRigidBody* rbMesh = m_arena->_AddStaticCollisionShape(m_arenaMeshShape, btVector3(0, 0, 0));
        if (rbMesh) {
            rbMesh->setRestitution(RocketSim::RLConst::ARENA_COLLISION_BASE_RESTITUTION);
            rbMesh->setFriction(RocketSim::RLConst::ARENA_COLLISION_BASE_FRICTION);
            rbMesh->setRollingFriction(0.0f);
        }
    }

    // Enable arena boost pad simulation without loading collision meshes
    m_arena->gameMode = RocketSim::GameMode::SOCCAR;

    // Populate standard 34 Soccar boost pads
    using namespace RocketSim::RLConst::BoostPads;
    m_arena->_boostPads.reserve(LOCS_AMOUNT_BIG + LOCS_AMOUNT_SMALL_SOCCAR);

    for (int i = 0; i < (LOCS_AMOUNT_BIG + LOCS_AMOUNT_SMALL_SOCCAR); i++) {
        RocketSim::BoostPadConfig padConfig;
        padConfig.isBig = (i < LOCS_AMOUNT_BIG);
        padConfig.pos = padConfig.isBig
            ? LOCS_BIG_SOCCAR[i]
            : LOCS_SMALL_SOCCAR[i - LOCS_AMOUNT_BIG];

        RocketSim::BoostPad* pad = RocketSim::BoostPad::_AllocBoostPad();
        pad->_Setup(padConfig);

        m_arena->_boostPads.push_back(pad);
        m_arena->_boostPadGrid.Add(pad);
    }

    {
        std::lock_guard<std::mutex> lock(s_simArenaMapMutex);
        s_simArenaMap[this] = m_arena;
    }

    m_cars.clear();
    const RocketSim::CarConfig* carCfg = &RocketSim::CAR_CONFIG_OCTANE;
    switch (m_hitboxType) {
        case 1: carCfg = &RocketSim::CAR_CONFIG_DOMINUS; break;
        case 2: carCfg = &RocketSim::CAR_CONFIG_PLANK; break;
        case 3: carCfg = &RocketSim::CAR_CONFIG_BREAKOUT; break;
        case 4: carCfg = &RocketSim::CAR_CONFIG_HYBRID; break;
        case 5: carCfg = &RocketSim::CAR_CONFIG_MERC; break;
        case 6: carCfg = &RocketSim::CAR_CONFIG_PSYCLOPS; break;
        default: carCfg = &RocketSim::CAR_CONFIG_OCTANE; break;
    }

    for (int i = 0; i < m_numCars; i++) {
        RocketSim::Team team = (m_numCars > 1 && (i % 2 != 0)) ? RocketSim::Team::ORANGE : RocketSim::Team::BLUE;
        RocketSim::Car* car = m_arena->AddCar(team, *carCfg);
        if (!car) {
            throw std::runtime_error("Failed to add car to RocketSim CPU Arena");
        }
        // Enforce deterministic initial spawn position and boost
        car->Respawn(RocketSim::GameMode::THE_VOID, m_spawnSeed + i, RocketSim::RLConst::BOOST_SPAWN_AMOUNT);
        m_cars.push_back(car);
    }
}

void CPURefSim::CleanupArena() {
    {
        std::lock_guard<std::mutex> lock(s_simArenaMapMutex);
        s_simArenaMap.erase(this);
    }
    if (m_arena) {
        delete m_arena;
        m_arena = nullptr;
    }
    if (m_arenaMeshShape) {
        delete m_arenaMeshShape;
        m_arenaMeshShape = nullptr;
    }
    if (m_arenaMeshInfoMap) {
        delete m_arenaMeshInfoMap;
        m_arenaMeshInfoMap = nullptr;
    }
    if (m_arenaMesh) {
        delete m_arenaMesh;
        m_arenaMesh = nullptr;
    }
    m_cars.clear();
}

void CPURefSim::Reset() {
    CleanupArena();
    InitArena();
}

void CPURefSim::SetHitboxType(int ht) {
    if (m_hitboxType == ht) return;
    m_hitboxType = ht;
    CleanupArena();
    InitArena();
}

void CPURefSim::ResetToRandomKickoff(int seed) {
    if (m_arena) {
        m_arena->ResetToRandomKickoff(seed);
    }
}

uint64_t CPURefSim::GetTickCount() const {
    return m_arena ? m_arena->tickCount : 0;
}

void CPURefSim::Step(const CarControls* controls, int numCars) {
    if (!m_arena) return;

    if (controls) {
        int applyCount = (std::min)(numCars, static_cast<int>(m_cars.size()));
        for (int i = 0; i < applyCount; i++) {
            m_cars[i]->controls.throttle  = controls[i].throttle;
            m_cars[i]->controls.steer     = controls[i].steer;
            m_cars[i]->controls.pitch     = controls[i].pitch;
            m_cars[i]->controls.yaw       = controls[i].yaw;
            m_cars[i]->controls.roll      = controls[i].roll;
            m_cars[i]->controls.boost     = (controls[i].boost != 0);
            m_cars[i]->controls.jump      = (controls[i].jump != 0);
            m_cars[i]->controls.handbrake = (controls[i].handbrake != 0);
            m_cars[i]->controls.ClampFix();
        }
    }

    m_arena->Step(1);
}

void CPURefSim::GetBallState(BallStatePOD& out) const {
    if (!m_arena || !m_arena->ball) return;

    RocketSim::BallState bs = m_arena->ball->GetState();
    out.pos = Vec3(bs.pos.x, bs.pos.y, bs.pos.z);
    out.vel = Vec3(bs.vel.x, bs.vel.y, bs.vel.z);
    out.ang_vel = Vec3(bs.angVel.x, bs.angVel.y, bs.angVel.z);

    btQuaternion bq = m_arena->ball->_rigidBody.getWorldTransform().getRotation();
    out.quat = Quat(bq.w(), bq.x(), bq.y(), bq.z());
}

void CPURefSim::GetCarState(int carIdx, CarStatePOD& out) const {
    if (carIdx < 0 || carIdx >= static_cast<int>(m_cars.size())) return;
    RocketSim::Car* car = m_cars[carIdx];
    RocketSim::CarState cs = car->GetState();

    out.pos = Vec3(cs.pos.x, cs.pos.y, cs.pos.z);
    out.vel = Vec3(cs.vel.x, cs.vel.y, cs.vel.z);
    out.ang_vel = Vec3(cs.angVel.x, cs.angVel.y, cs.angVel.z);

    btQuaternion cq = car->_rigidBody.getWorldTransform().getRotation();
    out.quat = Quat(cq.w(), cq.x(), cq.y(), cq.z());

    out.boost = cs.boost;
    out.is_on_ground = cs.isOnGround ? 1 : 0;
    out.has_jumped = cs.hasJumped ? 1 : 0;
    out.has_double_jumped = cs.hasDoubleJumped ? 1 : 0;
    out.has_flipped = cs.hasFlipped ? 1 : 0;
    out.is_demoed = cs.isDemoed ? 1 : 0;
    out.is_supersonic = cs.isSupersonic ? 1 : 0;
    out.demo_respawn_timer = cs.demoRespawnTimer;
    out.ball_touched = (cs.ballHitInfo.isValid && m_arena && cs.ballHitInfo.tickCountWhenHit == m_arena->tickCount) ? 1 : 0;
    out.team = (car->team == RocketSim::Team::BLUE) ? 0 : 1;
    out.hitbox_type = static_cast<uint8_t>(m_hitboxType);

    for (int w = 0; w < 4; w++) {
        out.wheels_with_contact[w] = cs.wheelsWithContact[w] ? 1 : 0;
        float restLen = car->_bulletVehicle.m_wheelInfo[w].getSuspensionRestLength();
        float curLen = car->_bulletVehicle.m_wheelInfo[w].m_raycastInfo.m_suspensionLength;
        out.suspension_lengths[w] = (restLen - curLen) * BT_TO_UU;
    }

    out.last_controls.throttle  = cs.lastControls.throttle;
    out.last_controls.steer     = cs.lastControls.steer;
    out.last_controls.pitch     = cs.lastControls.pitch;
    out.last_controls.yaw       = cs.lastControls.yaw;
    out.last_controls.roll      = cs.lastControls.roll;
    out.last_controls.boost     = cs.lastControls.boost ? 1 : 0;
    out.last_controls.jump      = cs.lastControls.jump ? 1 : 0;
    out.last_controls.handbrake = cs.lastControls.handbrake ? 1 : 0;

    out.is_boosting             = cs.isBoosting ? 1 : 0;
    out.boosting_time           = cs.boostingTime;
    out.handbrake_val           = cs.handbrakeVal;
    out.is_jumping              = cs.isJumping ? 1 : 0;
    out.jump_time               = cs.jumpTime;
    out.air_time_since_jump     = cs.airTimeSinceJump;
    out.flip_time               = cs.flipTime;
    out.is_flipping             = cs.isFlipping ? 1 : 0;

    out.time_since_boosted      = cs.timeSinceBoosted;
    out.air_time                = cs.airTime;
    out.flip_rel_torque         = Vec3(cs.flipRelTorque.x, cs.flipRelTorque.y, cs.flipRelTorque.z);
    out.is_auto_flipping        = cs.isAutoFlipping ? 1 : 0;
    out.auto_flip_timer         = cs.autoFlipTimer;
    out.auto_flip_torque_scale  = cs.autoFlipTorqueScale;
    out.supersonic_time         = cs.supersonicTime;
    out.world_contact_has_contact = cs.worldContact.hasContact ? 1 : 0;
    out.world_contact_normal    = Vec3(cs.worldContact.contactNormal.x, cs.worldContact.contactNormal.y, cs.worldContact.contactNormal.z);
    out.car_contact_other_car_id   = static_cast<int32_t>(cs.carContact.otherCarID);
    out.car_contact_cooldown_timer = cs.carContact.cooldownTimer;
    out.ball_hit_is_valid       = cs.ballHitInfo.isValid ? 1 : 0;
    out.ball_hit_rel_pos        = Vec3(cs.ballHitInfo.relativePosOnBall.x, cs.ballHitInfo.relativePosOnBall.y, cs.ballHitInfo.relativePosOnBall.z);
    out.ball_hit_extra_hit_force = Vec3(cs.ballHitInfo.extraHitVel.x, cs.ballHitInfo.extraHitVel.y, cs.ballHitInfo.extraHitVel.z);
    out.ball_hit_tick_count     = cs.ballHitInfo.tickCountWhenHit;
}

void CPURefSim::GetWheelDebug(int carIdx, int w, float& susp_force, float& v_rel, float& pushback, float& comp_bt) const {
    susp_force = 0.0f; v_rel = 0.0f; pushback = 0.0f; comp_bt = 0.0f;
    if (carIdx < 0 || carIdx >= static_cast<int>(m_cars.size())) return;
    RocketSim::Car* car = m_cars[carIdx];
    const auto& wi = car->_bulletVehicle.m_wheelInfo[w];
    susp_force = wi.m_wheelsSuspensionForce;
    v_rel = wi.m_suspensionRelativeVelocity;
    pushback = wi.m_extraPushback;
    comp_bt = wi.getSuspensionRestLength() - wi.m_raycastInfo.m_suspensionLength;
}

void CPURefSim::SetBallState(const BallStatePOD& in) {
    if (!m_arena || !m_arena->ball) return;

    RocketSim::BallState bs;
    bs.pos = RocketSim::Vec(in.pos.x, in.pos.y, in.pos.z);
    bs.vel = RocketSim::Vec(in.vel.x, in.vel.y, in.vel.z);
    bs.angVel = RocketSim::Vec(in.ang_vel.x, in.ang_vel.y, in.ang_vel.z);
    btMatrix3x3 basis(btQuaternion(in.quat.x, in.quat.y, in.quat.z, in.quat.w));
    bs.rotMat = RocketSim::RotMat(basis);

    // If ball is spawned in mid-air with zero velocity, impart microscopic vel so Arena::Step won't force ISLAND_SLEEPING
    if (bs.pos.z > 95.0f && bs.vel.LengthSq() == 0.0f && bs.angVel.LengthSq() == 0.0f) {
        bs.vel.z = -1e-6f;
    }

    m_arena->ball->SetState(bs);
    m_arena->ball->_rigidBody.activate(true);
    m_arena->ball->_rigidBody.setActivationState(ACTIVE_TAG);
}

void CPURefSim::SetCarState(int carIdx, const CarStatePOD& in) {
    if (carIdx < 0 || carIdx >= static_cast<int>(m_cars.size())) return;
    RocketSim::Car* car = m_cars[carIdx];
    RocketSim::CarState cs = car->GetState();

    cs.pos = RocketSim::Vec(in.pos.x, in.pos.y, in.pos.z);
    cs.vel = RocketSim::Vec(in.vel.x, in.vel.y, in.vel.z);
    cs.angVel = RocketSim::Vec(in.ang_vel.x, in.ang_vel.y, in.ang_vel.z);
    btMatrix3x3 basis(btQuaternion(in.quat.x, in.quat.y, in.quat.z, in.quat.w));
    cs.rotMat = RocketSim::RotMat(basis);
    cs.boost = in.boost;
    cs.isOnGround = (in.is_on_ground != 0);
    cs.hasJumped = (in.has_jumped != 0);
    cs.hasDoubleJumped = (in.has_double_jumped != 0);
    cs.hasFlipped = (in.has_flipped != 0);
    cs.isDemoed = (in.is_demoed != 0);
    cs.isSupersonic = (in.is_supersonic != 0);
    cs.demoRespawnTimer = in.demo_respawn_timer;

    cs.isBoosting = (in.is_boosting != 0);
    cs.boostingTime = in.boosting_time;
    cs.handbrakeVal = in.handbrake_val;
    cs.isJumping = (in.is_jumping != 0);
    cs.jumpTime = in.jump_time;
    cs.airTimeSinceJump = in.air_time_since_jump;
    cs.flipTime = in.flip_time;
    cs.isFlipping = (in.is_flipping != 0);

    cs.timeSinceBoosted = in.time_since_boosted;
    cs.airTime = in.air_time;
    cs.flipRelTorque = RocketSim::Vec(in.flip_rel_torque.x, in.flip_rel_torque.y, in.flip_rel_torque.z);
    cs.isAutoFlipping = (in.is_auto_flipping != 0);
    cs.autoFlipTimer = in.auto_flip_timer;
    cs.autoFlipTorqueScale = in.auto_flip_torque_scale;
    cs.supersonicTime = in.supersonic_time;
    cs.worldContact.hasContact = (in.world_contact_has_contact != 0);
    cs.worldContact.contactNormal = RocketSim::Vec(in.world_contact_normal.x, in.world_contact_normal.y, in.world_contact_normal.z);
    cs.carContact.otherCarID = (in.car_contact_other_car_id >= 0) ? static_cast<uint32_t>(in.car_contact_other_car_id) : 0;
    cs.carContact.cooldownTimer = in.car_contact_cooldown_timer;
    if (in.ball_hit_is_valid) {
        cs.ballHitInfo.isValid = true;
        cs.ballHitInfo.relativePosOnBall = RocketSim::Vec(in.ball_hit_rel_pos.x, in.ball_hit_rel_pos.y, in.ball_hit_rel_pos.z);
        cs.ballHitInfo.extraHitVel = RocketSim::Vec(in.ball_hit_extra_hit_force.x, in.ball_hit_extra_hit_force.y, in.ball_hit_extra_hit_force.z);
        cs.ballHitInfo.tickCountWhenHit = in.ball_hit_tick_count;
    }

    if (in.ball_touched && m_arena) {
        cs.ballHitInfo.isValid = true;
        cs.ballHitInfo.tickCountWhenHit = m_arena->tickCount;
    }
    car->team = (in.team == 0) ? RocketSim::Team::BLUE : RocketSim::Team::ORANGE;

    car->SetState(cs);
    car->_rigidBody.activate(true);
    car->_rigidBody.setActivationState(ACTIVE_TAG);
}

void CPURefSim::DumpCarInternalState(int carIdx) const {
    if (carIdx < 0 || carIdx >= static_cast<int>(m_cars.size())) return;
    RocketSim::Car* car = m_cars[carIdx];
    auto is = car->_internalState;
    std::cout << "  [CPU Car " << carIdx << " Internal] "
              << "og=" << (int)is.isOnGround
              << " hasJmp=" << (int)is.hasJumped
              << " isJmp=" << (int)is.isJumping
              << " jmpTime=" << is.jumpTime
              << " hasDblJmp=" << (int)is.hasDoubleJumped
              << " hasFlip=" << (int)is.hasFlipped
              << " isFlip=" << (int)is.isFlipping
              << " flipTime=" << is.flipTime
              << " isAutoFlip=" << (int)is.isAutoFlipping
              << " wheelsContact=(" << (int)is.wheelsWithContact[0] << "," << (int)is.wheelsWithContact[1]
              << "," << (int)is.wheelsWithContact[2] << "," << (int)is.wheelsWithContact[3] << ")"
              << " worldContact=" << (int)is.worldContact.hasContact
              << " lastJmp=" << (int)is.lastControls.jump
              << "\n";
}

bool CPURefSim::IsBallProbablyGoingIn(float maxTime, float extraMargin, int* goalTeamOut) const {
    if (!m_arena) return false;
    RocketSim::Team team = RocketSim::Team::BLUE;
    bool res = m_arena->IsBallProbablyGoingIn(maxTime, extraMargin, goalTeamOut ? &team : nullptr);
    if (goalTeamOut) {
        *goalTeamOut = (team == RocketSim::Team::BLUE) ? 0 : 1;
    }
    return res;
}

} // namespace rocketsim_cuda
