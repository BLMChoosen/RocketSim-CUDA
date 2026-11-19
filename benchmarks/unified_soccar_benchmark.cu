#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <thread>
#include <iomanip>
#include <intrin.h>
#include <cuda_runtime.h>

#include "RocketSim.h"
#include "Sim/Arena/Arena.h"
#include "Sim/Car/Car.h"
#include "Sim/Ball/Ball.h"
#include "RLConst.h"

#include "cpu_ref_sim.h"
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/sim_context.cuh"

namespace {

__device__ inline float benchmark_lcg_float(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return (float)(state & 0x00FFFFFF) / 16777216.0f;
}

__global__ void update_car_inputs_kernel(
    rocketsim_cuda::CarControlsSoA controls,
    int* d_change_counters,
    uint32_t* d_rng_states,
    uint32_t total_cars
) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_cars) return;

    int counter = d_change_counters[idx] - 1;
    if (counter <= 0) {
        uint32_t rng = d_rng_states[idx];
        float throttle = benchmark_lcg_float(rng) * 2.0f - 1.0f;
        float steer    = benchmark_lcg_float(rng) * 2.0f - 1.0f;
        float pitch    = benchmark_lcg_float(rng) * 2.0f - 1.0f;
        float yaw      = benchmark_lcg_float(rng) * 2.0f - 1.0f;
        float roll     = benchmark_lcg_float(rng) * 2.0f - 1.0f;
        bool boost     = benchmark_lcg_float(rng) > 0.4f;
        bool jump      = benchmark_lcg_float(rng) > 0.6f;
        bool handbrake = benchmark_lcg_float(rng) > 0.8f;

        counter = 2 + (int)(benchmark_lcg_float(rng) * 59.0f);

        d_rng_states[idx] = rng;
        d_change_counters[idx] = counter;

        controls.throttle[idx] = throttle;
        controls.steer[idx]    = steer;
        controls.pitch[idx]    = pitch;
        controls.yaw[idx]      = yaw;
        controls.roll[idx]     = roll;
        controls.boost[idx]    = boost ? 1 : 0;
        controls.jump[idx]     = jump ? 1 : 0;
        controls.handbrake[idx]= handbrake ? 1 : 0;
    } else {
        d_change_counters[idx] = counter;
    }
}

__global__ void init_car_inputs_kernel(
    int* d_change_counters,
    uint32_t* d_rng_states,
    uint32_t total_cars,
    uint32_t seed
) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_cars) return;

    uint32_t rng = seed + idx * 1999u + 7u;
    d_rng_states[idx] = rng;
    d_change_counters[idx] = 2 + (int)(benchmark_lcg_float(rng) * 59.0f);
}

} // namespace

int main(int argc, char** argv) {
    std::cout << "======================================================================\n";
    std::cout << "    ROCKETSIM CPU vs ROCKETSIM-CUDA CANONICAL COMPARISON BENCHMARK     \n";
    std::cout << "======================================================================\n";

    // Initialize RocketSim CPU
    RocketSim::Init("", true);

    const int total_ticks_single = 1000000; // 1M ticks simulated
    const int total_ticks_multi_per_thread = 1000000; // 1M ticks per thread
    const unsigned int num_cpu_threads = 12; // 12 logical cores on Ryzen 5 5500

    // -------------------------------------------------------------------------
    // TEST 1: Single-Thread RocketSim CPU Performance (Soccar 2v2)
    // -------------------------------------------------------------------------
    std::cout << "\n----------------------------------------------------------------------\n";
    std::cout << ">>> Running Benchmark [1/3]: Single-Thread CPU (Soccar, 2v2, 1M ticks)...\n";
    std::cout << "----------------------------------------------------------------------\n";
    
    double single_elapsed_sec = 0.0;
    double single_tps = 0.0;
    uint64_t single_total_cycles = 0;
    double single_avg_cycles = 0.0;

    {
        rocketsim_cuda::CPURefSim sim(4, true); // 4 cars (2v2), procedural Soccar arena

        rocketsim_cuda::CarControls controls[4] = {};
        int change_counter[4] = {0, 0, 0, 0};
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist_signed(-1.0f, 1.0f);
        std::uniform_int_distribution<int> dist_interval(2, 60);

        // Warm-up 2,000 ticks
        std::cout << "    Warming up (2,000 ticks)...\n";
        for (int t = 0; t < 2000; t++) {
            sim.Step(nullptr, 4);
        }

        std::cout << "    Simulating 1,000,000 ticks...\n";
        uint64_t c0 = __rdtsc();
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < total_ticks_single; t++) {
            for (int c = 0; c < 4; c++) {
                if (--change_counter[c] <= 0) {
                    controls[c].throttle  = dist_signed(rng);
                    controls[c].steer     = dist_signed(rng);
                    controls[c].pitch     = dist_signed(rng);
                    controls[c].yaw       = dist_signed(rng);
                    controls[c].roll      = dist_signed(rng);
                    controls[c].boost     = (dist_signed(rng) > 0.4f) ? 1 : 0;
                    controls[c].jump      = (dist_signed(rng) > 0.6f) ? 1 : 0;
                    controls[c].handbrake = (dist_signed(rng) > 0.8f) ? 1 : 0;
                    controls[c].clamp_fix();
                    change_counter[c]     = dist_interval(rng);
                }
            }
            sim.Step(controls, 4);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t c1 = __rdtsc();

        single_elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
        single_total_cycles = c1 - c0;
        single_avg_cycles = static_cast<double>(single_total_cycles) / total_ticks_single;
        single_tps = total_ticks_single / single_elapsed_sec;

        std::cout << "    [+] Single-Thread Completed!\n";
        std::cout << "        Ticks simulated:   " << total_ticks_single << "\n";
        std::cout << "        Elapsed Time:      " << std::fixed << std::setprecision(3) << (single_elapsed_sec * 1000.0) << " ms (" << single_elapsed_sec << " s)\n";
        std::cout << "        Avg CPU Cycles/tick: " << std::fixed << std::setprecision(1) << single_avg_cycles << "\n";
        std::cout << "        Single-Thread TPS: " << std::fixed << std::setprecision(0) << single_tps << " TPS\n";
    }

    // -------------------------------------------------------------------------
    // TEST 2: Multi-Thread RocketSim CPU Performance (12 Threads, Soccar 2v2)
    // -------------------------------------------------------------------------
    std::cout << "\n----------------------------------------------------------------------\n";
    std::cout << ">>> Running Benchmark [2/3]: Multi-Thread CPU (12 Threads, 1M ticks/thread = 12M total)...\n";
    std::cout << "----------------------------------------------------------------------\n";

    double multi_elapsed_sec = 0.0;
    double multi_tps = 0.0;
    uint64_t total_multi_ticks = static_cast<uint64_t>(total_ticks_multi_per_thread) * num_cpu_threads;

    {
        std::vector<std::thread> workers;
        workers.reserve(num_cpu_threads);

        std::cout << "    Spawning " << num_cpu_threads << " worker threads...\n";
        auto t0 = std::chrono::high_resolution_clock::now();

        for (unsigned int th = 0; th < num_cpu_threads; th++) {
            workers.emplace_back([th, total_ticks_multi_per_thread]() {
                rocketsim_cuda::CPURefSim sim(4, true, 120.0f, th * 10);

                rocketsim_cuda::CarControls controls[4] = {};
                int change_counter[4] = {0, 0, 0, 0};
                std::mt19937 rng(1337 + th * 7919);
                std::uniform_real_distribution<float> dist_signed(-1.0f, 1.0f);
                std::uniform_int_distribution<int> dist_interval(2, 60);

                // Warm-up 1,000 ticks
                for (int t = 0; t < 1000; t++) {
                    sim.Step(nullptr, 4);
                }

                for (int t = 0; t < total_ticks_multi_per_thread; t++) {
                    for (int c = 0; c < 4; c++) {
                        if (--change_counter[c] <= 0) {
                            controls[c].throttle  = dist_signed(rng);
                            controls[c].steer     = dist_signed(rng);
                            controls[c].pitch     = dist_signed(rng);
                            controls[c].yaw       = dist_signed(rng);
                            controls[c].roll      = dist_signed(rng);
                            controls[c].boost     = (dist_signed(rng) > 0.4f) ? 1 : 0;
                            controls[c].jump      = (dist_signed(rng) > 0.6f) ? 1 : 0;
                            controls[c].handbrake = (dist_signed(rng) > 0.8f) ? 1 : 0;
                            controls[c].clamp_fix();
                            change_counter[c]     = dist_interval(rng);
                        }
                    }
                    sim.Step(controls, 4);
                }
            });
        }

        for (auto& w : workers) {
            w.join();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        multi_elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
        multi_tps = static_cast<double>(total_multi_ticks) / multi_elapsed_sec;

        std::cout << "    [+] Multi-Thread Completed!\n";
        std::cout << "        Threads:           " << num_cpu_threads << "\n";
        std::cout << "        Ticks/Thread:      " << total_ticks_multi_per_thread << "\n";
        std::cout << "        Total Ticks:       " << total_multi_ticks << "\n";
        std::cout << "        Elapsed Time:      " << std::fixed << std::setprecision(3) << (multi_elapsed_sec * 1000.0) << " ms (" << multi_elapsed_sec << " s)\n";
        std::cout << "        Multi-Thread TPS:  " << std::fixed << std::setprecision(0) << multi_tps << " TPS\n";
    }

    // -------------------------------------------------------------------------
    // TEST 3: RocketSim-CUDA Performance (RTX 5060, Soccar 2v2)
    // -------------------------------------------------------------------------
    std::cout << "\n----------------------------------------------------------------------\n";
    std::cout << ">>> Running Benchmark [3/3]: RocketSim-CUDA (RTX 5060, Soccar 2v2)...\n";
    std::cout << "----------------------------------------------------------------------\n";

    const uint32_t cuda_envs = 32768; // 32k environments
    const uint32_t cuda_cars_per_env = 4; // 2v2 (4 cars)
    const uint32_t cuda_total_cars = cuda_envs * cuda_cars_per_env; // 131,072 cars
    const uint32_t cuda_steps = 1000; // 1,000 steps

    double cuda_elapsed_ms = 0.0;
    double cuda_physical_sps = 0.0;
    double cuda_agent_sps = 0.0;
    double cuda_step_latency_ms = 0.0;

    {
        rocketsim_cuda::SimContext ctx(cuda_envs, cuda_cars_per_env);
        ctx.ResetToDefault();

        // Allocate input management buffers on GPU
        int* d_change_counters = nullptr;
        uint32_t* d_rng_states = nullptr;
        cudaMalloc(&d_change_counters, cuda_total_cars * sizeof(int));
        cudaMalloc(&d_rng_states, cuda_total_cars * sizeof(uint32_t));

        uint32_t block_size = 256;
        uint32_t grid_size = (cuda_total_cars + block_size - 1) / block_size;
        init_car_inputs_kernel<<<grid_size, block_size>>>(d_change_counters, d_rng_states, cuda_total_cars, 42u);
        cudaDeviceSynchronize();

        // Warm-up 100 steps
        std::cout << "    Warming up GPU (" << cuda_envs << " envs, " << cuda_total_cars << " cars, 100 steps)...\n";
        for (int i = 0; i < 100; i++) {
            update_car_inputs_kernel<<<grid_size, block_size>>>(ctx.GetControls(), d_change_counters, d_rng_states, cuda_total_cars);
            ctx.Step(cuda_envs);
        }
        cudaDeviceSynchronize();

        std::cout << "    Simulating " << cuda_steps << " batch steps (" << (static_cast<uint64_t>(cuda_envs) * cuda_steps) << " arena ticks)...\n";

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (uint32_t s = 0; s < cuda_steps; s++) {
            update_car_inputs_kernel<<<grid_size, block_size>>>(ctx.GetControls(), d_change_counters, d_rng_states, cuda_total_cars);
            ctx.Step(cuda_envs);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_change_counters);
        cudaFree(d_rng_states);

        cuda_elapsed_ms = ms;
        cuda_step_latency_ms = cuda_elapsed_ms / static_cast<double>(cuda_steps);
        double total_arena_ticks = static_cast<double>(cuda_envs) * static_cast<double>(cuda_steps);
        cuda_physical_sps = total_arena_ticks / (cuda_elapsed_ms / 1000.0);
        cuda_agent_sps = cuda_physical_sps * cuda_cars_per_env;

        std::cout << "    [+] RocketSim-CUDA Completed!\n";
        std::cout << "        Environments:      " << cuda_envs << " (32,768 arenas)\n";
        std::cout << "        Total Cars:        " << cuda_total_cars << " (131,072 active cars)\n";
        std::cout << "        Steps Simulated:   " << cuda_steps << "\n";
        std::cout << "        Total Arena Ticks: " << static_cast<uint64_t>(total_arena_ticks) << "\n";
        std::cout << "        Total Car Ticks:   " << static_cast<uint64_t>(total_arena_ticks * 4.0) << "\n";
        std::cout << "        Elapsed Time:      " << std::fixed << std::setprecision(2) << cuda_elapsed_ms << " ms\n";
        std::cout << "        Batch Step Latency: " << std::fixed << std::setprecision(4) << cuda_step_latency_ms << " ms\n";
        std::cout << "        Physical SPS (TPS):" << std::fixed << std::setprecision(0) << cuda_physical_sps << " SPS\n";
        std::cout << "        Agent SPS (Car-TPS):" << std::fixed << std::setprecision(0) << cuda_agent_sps << " Car-SPS\n";
    }

    // -------------------------------------------------------------------------
    // SUMMARY REPORT OUTPUT (Formatted for User Request)
    // -------------------------------------------------------------------------
    std::cout << "\n\n";
    std::cout << "======================================================================\n";
    std::cout << "                       FINAL BENCHMARK SUMMARY                        \n";
    std::cout << "======================================================================\n";
    std::cout << "Single-Thread CPU TPS:  " << std::fixed << std::setprecision(0) << single_tps << " TPS\n";
    std::cout << "Multi-Thread CPU TPS:   " << std::fixed << std::setprecision(0) << multi_tps << " TPS\n";
    std::cout << "RocketSim-CUDA SPS:     " << std::fixed << std::setprecision(0) << cuda_physical_sps << " SPS (Physical Arena TPS)\n";
    std::cout << "RocketSim-CUDA Car-SPS: " << std::fixed << std::setprecision(0) << cuda_agent_sps << " Car-SPS (Agent Car TPS)\n";
    std::cout << "Speedup vs Single-Thread: " << std::fixed << std::setprecision(1) << (cuda_physical_sps / single_tps) << "x (Physical) / " 
              << (cuda_agent_sps / single_tps) << "x (Agent)\n";
    std::cout << "Speedup vs Multi-Thread:  " << std::fixed << std::setprecision(1) << (cuda_physical_sps / multi_tps) << "x (Physical) / " 
              << (cuda_agent_sps / multi_tps) << "x (Agent)\n";
    std::cout << "======================================================================\n";

    return 0;
}
