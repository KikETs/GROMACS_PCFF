#include "gromacs/utility/vectypes.h"

#include <chrono>
#include <cstdio>
#include <vector>

int main()
{
    constexpr int numAtoms   = 7907;
    constexpr int iterations = 200000;
    std::vector<gmx::RVec> fast(numAtoms), slow1(numAtoms), slow2(numAtoms), physical(numAtoms);

    for (int i = 0; i < numAtoms; ++i)
    {
        fast[i]  = gmx::RVec{ 1.0F + i * 0.001F, 2.0F, 3.0F };
        slow1[i] = gmx::RVec{ 0.1F, 0.2F + i * 0.0001F, 0.3F };
        slow2[i] = gmx::RVec{ 0.01F, 0.02F, 0.03F + i * 0.00001F };
    }

    auto runOld = [&]() {
        float checksum = 0.0F;
        auto  begin    = std::chrono::steady_clock::now();
        for (int iter = 0; iter < iterations; ++iter)
        {
            for (int atom = 0; atom < numAtoms; ++atom)
            {
                gmx::RVec restoredPhysical = fast[atom];
                gmx::RVec restoredCombined = fast[atom];
                restoredPhysical += slow1[atom];
                restoredCombined += 2.0F * slow1[atom];
                restoredPhysical += slow2[atom];
                restoredCombined += 4.0F * slow2[atom];
                physical[atom] = restoredPhysical;
            }
            checksum += physical[(iter % numAtoms)][XX] * 1.0e-12F;
        }
        auto end = std::chrono::steady_clock::now();
        return std::pair<double, float>{ std::chrono::duration<double>(end - begin).count(), checksum };
    };

    auto runFast = [&]() {
        float checksum = 0.0F;
        auto  begin    = std::chrono::steady_clock::now();
        for (int iter = 0; iter < iterations; ++iter)
        {
            for (int atom = 0; atom < numAtoms; ++atom)
            {
                physical[atom][XX] = fast[atom][XX] + slow1[atom][XX] + slow2[atom][XX];
                physical[atom][YY] = fast[atom][YY] + slow1[atom][YY] + slow2[atom][YY];
                physical[atom][ZZ] = fast[atom][ZZ] + slow1[atom][ZZ] + slow2[atom][ZZ];
            }
            checksum += physical[(iter % numAtoms)][XX] * 1.0e-12F;
        }
        auto end = std::chrono::steady_clock::now();
        return std::pair<double, float>{ std::chrono::duration<double>(end - begin).count(), checksum };
    };

    auto runFlat = [&]() {
        float      checksum = 0.0F;
        const auto flatSize = numAtoms * DIM;
        auto*      fastRaw  = reinterpret_cast<const float*>(fast.data());
        auto*      slow1Raw = reinterpret_cast<const float*>(slow1.data());
        auto*      slow2Raw = reinterpret_cast<const float*>(slow2.data());
        auto*      physRaw  = reinterpret_cast<float*>(physical.data());
        auto       begin    = std::chrono::steady_clock::now();
        for (int iter = 0; iter < iterations; ++iter)
        {
            for (int i = 0; i < flatSize; ++i)
            {
                physRaw[i] = fastRaw[i] + slow1Raw[i] + slow2Raw[i];
            }
            checksum += physRaw[(iter % numAtoms) * DIM] * 1.0e-12F;
        }
        auto end = std::chrono::steady_clock::now();
        return std::pair<double, float>{ std::chrono::duration<double>(end - begin).count(), checksum };
    };

    const auto oldResult  = runOld();
    const auto fastResult = runFast();
    const auto flatResult = runFlat();
    std::printf("old_seconds %.6f checksum %.9g\n", oldResult.first, oldResult.second);
    std::printf("fast_seconds %.6f checksum %.9g\n", fastResult.first, fastResult.second);
    std::printf("flat_seconds %.6f checksum %.9g\n", flatResult.first, flatResult.second);
    std::printf("speedup %.6f\n", oldResult.first / fastResult.first);
    std::printf("flat_speedup %.6f\n", oldResult.first / flatResult.first);
}
