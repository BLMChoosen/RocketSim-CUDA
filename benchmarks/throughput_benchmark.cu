#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/sim_context.cuh"

using namespace rocketsim_cuda;

int main(int argc, char** argv) {
    uint32_t envs = 32768;
    uint32_t steps = 1000;
    uint32_t cars = 1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--envs" && i + 1 < argc) {
            envs = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--steps" && i + 1 < argc) {
            steps = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--cars" && i + 1 < argc) {
            cars = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        }
    }

    std::cout << "======================================================================\n"
              << "                 ROCKETSIM-CUDA THROUGHPUT BENCHMARK                  \n"
              << "======================================================================\n"
              << "  Environments: " << envs << "\n"
              << "  Cars/Env:     " << cars << " (Total Cars: " << (envs * cars) << ")\n"
              << "  Steps:        " << steps << "\n"
              << "======================================================================\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    SimContext ctx(envs, cars);
    auto t1 = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[+] SimContext allocation & init: " << init_ms << " ms ("
              << (ctx.GetAllocatedBytes() / (1024.0 * 1024.0)) << " MB)\n";

    // Warm-up
    ctx.ResetToDefault();
    for (int i = 0; i < 20; ++i) {
        ctx.Step(envs);
    }
    cudaDeviceSynchronize();

    std::cout << "[+] Running benchmark across " << steps << " simulation steps...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (uint32_t i = 0; i < steps; ++i) {
        ctx.Step(envs);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avg_step_latency_ms = elapsed_ms / static_cast<double>(steps);
    double total_sim_steps = static_cast<double>(envs) * static_cast<double>(steps);
    double sps = total_sim_steps / (elapsed_ms / 1000.0);
    double car_sps = sps * static_cast<double>(cars);

    std::cout << "----------------------------------------------------------------------\n"
              << "  Benchmark Results:\n"
              << "    Total Elapsed Time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n"
              << "    Batch Step Latency: " << std::setprecision(4) << avg_step_latency_ms << " ms\n"
              << "    Physical SPS:       " << static_cast<uint64_t>(sps) << " SPS\n"
              << "    Agent SPS:          " << static_cast<uint64_t>(car_sps) << " Car-SPS\n"
              << "----------------------------------------------------------------------\n";

    if (avg_step_latency_ms < 0.15) {
        std::cout << "[+] Target Met: Batch step latency (" << avg_step_latency_ms << " ms) < 0.15 ms target.\n";
    } else {
        std::cout << "[!] Batch step latency: " << avg_step_latency_ms << " ms.\n";
    }

    return 0;
}
