#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct RVec
{
    float v[3];
    float& operator[](int d) { return v[d]; }
    const float& operator[](int d) const { return v[d]; }
};

template<typename T>
struct ArrayRef
{
    T* data;
    T& operator[](int index) const { return data[index]; }
};

template<typename Body>
__attribute__((noinline)) void updateForAtoms(int numAtoms, const Body& body)
{
#pragma omp parallel for num_threads(8) schedule(static)
    for (int atom = 0; atom < numAtoms; ++atom)
    {
        body(atom);
    }
}

template<int NumKicks>
__attribute__((noinline)) void oldUpdate(int          numAtoms,
                                         const RVec* invMass,
                                         const RVec* force0,
                                         const RVec* force1,
                                         const RVec* force2,
                                         float       scale0,
                                         float       scale1,
                                         float       scale2,
                                         float       driftDt,
                                         RVec*       position,
                                         RVec*       velocity)
{
    const ArrayRef<const RVec> invMassRef{ invMass };
    const ArrayRef<const RVec> force0Ref{ force0 };
    const ArrayRef<const RVec> force1Ref{ force1 };
    const ArrayRef<const RVec> force2Ref{ force2 };
    const ArrayRef<RVec>       positionRef{ position };
    const ArrayRef<RVec>       velocityRef{ velocity };
    updateForAtoms(numAtoms, [&](const int atom)
    {
        const float invMassX = invMassRef[atom][0];
        const float invMassY = invMassRef[atom][1];
        const float invMassZ = invMassRef[atom][2];
        float       vx       = velocityRef[atom][0];
        float       vy       = velocityRef[atom][1];
        float       vz       = velocityRef[atom][2];
        vx += scale0 * invMassX * force0Ref[atom][0];
        vy += scale0 * invMassY * force0Ref[atom][1];
        vz += scale0 * invMassZ * force0Ref[atom][2];
        if constexpr (NumKicks >= 2)
        {
            vx += scale1 * invMassX * force1Ref[atom][0];
            vy += scale1 * invMassY * force1Ref[atom][1];
            vz += scale1 * invMassZ * force1Ref[atom][2];
        }
        if constexpr (NumKicks >= 3)
        {
            vx += scale2 * invMassX * force2Ref[atom][0];
            vy += scale2 * invMassY * force2Ref[atom][1];
            vz += scale2 * invMassZ * force2Ref[atom][2];
        }
        velocityRef[atom][0] = vx;
        velocityRef[atom][1] = vy;
        velocityRef[atom][2] = vz;
        positionRef[atom][0] += driftDt * vx;
        positionRef[atom][1] += driftDt * vy;
        positionRef[atom][2] += driftDt * vz;
    });
}

template<int NumKicks>
__attribute__((noinline)) void newRange(int                    beginAtom,
                                        int                    endAtom,
                                        const RVec* __restrict invMass,
                                        const RVec* __restrict force0,
                                        const RVec* __restrict force1,
                                        const RVec* __restrict force2,
                                        float                  scale0,
                                        float                  scale1,
                                        float                  scale2,
                                        float                  driftDt,
                                        RVec* __restrict       position,
                                        RVec* __restrict       velocity)
{
    for (int atom = beginAtom; atom < endAtom; ++atom)
    {
        for (int d = 0; d < 3; ++d)
        {
            const float inverseMass = invMass[atom][d];
            float       v           = velocity[atom][d];
            v += scale0 * inverseMass * force0[atom][d];
            if constexpr (NumKicks >= 2)
            {
                v += scale1 * inverseMass * force1[atom][d];
            }
            if constexpr (NumKicks >= 3)
            {
                v += scale2 * inverseMass * force2[atom][d];
            }
            velocity[atom][d] = v;
            position[atom][d] += driftDt * v;
        }
    }
}

template<int NumKicks>
__attribute__((noinline)) void newUpdate(int                    numAtoms,
                                         const RVec* __restrict invMass,
                                         const RVec* __restrict force0,
                                         const RVec* __restrict force1,
                                         const RVec* __restrict force2,
                                         float                  scale0,
                                         float                  scale1,
                                         float                  scale2,
                                         float                  driftDt,
                                         RVec* __restrict       position,
                                         RVec* __restrict       velocity)
{
#pragma omp parallel for num_threads(8) schedule(static)
    for (int atom = 0; atom < numAtoms; ++atom)
    {
        for (int d = 0; d < 3; ++d)
        {
            const float inverseMass = invMass[atom][d];
            float       v           = velocity[atom][d];
            v += scale0 * inverseMass * force0[atom][d];
            if constexpr (NumKicks >= 2)
            {
                v += scale1 * inverseMass * force1[atom][d];
            }
            if constexpr (NumKicks >= 3)
            {
                v += scale2 * inverseMass * force2[atom][d];
            }
            velocity[atom][d] = v;
            position[atom][d] += driftDt * v;
        }
    }
}

template<int NumKicks, typename Update>
double runCase(Update&&             update,
               int                 iterations,
               const std::vector<RVec>& invMass,
               const std::array<std::vector<RVec>, 3>& force,
               const std::vector<RVec>& initialPosition,
               const std::vector<RVec>& initialVelocity,
               double*             checksum)
{
    std::vector<RVec> position = initialPosition;
    std::vector<RVec> velocity = initialVelocity;
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        update(static_cast<int>(position.size()),
               invMass.data(),
               force[0].data(),
               force[1].data(),
               force[2].data(),
               0.00025F,
               0.0005F,
               0.001F,
               0.0005F,
               position.data(),
               velocity.data());
    }
    const auto stop = std::chrono::steady_clock::now();
    double sum = 0;
    for (int atom = 0; atom < static_cast<int>(position.size()); atom += 97)
    {
        sum += position[atom][0] + position[atom][1] + position[atom][2];
        sum += velocity[atom][0] + velocity[atom][1] + velocity[atom][2];
    }
    *checksum = sum;
    return std::chrono::duration<double>(stop - start).count();
}

template<int NumKicks>
void benchmarkKickCount(int                                    iterations,
                        const std::vector<RVec>&               invMass,
                        const std::array<std::vector<RVec>, 3>& force,
                        const std::vector<RVec>&               initialPosition,
                        const std::vector<RVec>&               initialVelocity)
{
    for (int repeat = 0; repeat < 7; ++repeat)
    {
        double oldChecksum = 0;
        double newChecksum = 0;
        const double oldSeconds = runCase<NumKicks>(oldUpdate<NumKicks>,
                                                    iterations,
                                                    invMass,
                                                    force,
                                                    initialPosition,
                                                    initialVelocity,
                                                    &oldChecksum);
        const double newSeconds = runCase<NumKicks>(newUpdate<NumKicks>,
                                                    iterations,
                                                    invMass,
                                                    force,
                                                    initialPosition,
                                                    initialVelocity,
                                                    &newChecksum);
        std::printf("kicks=%d repeat=%d old=%.9f new=%.9f speedup=%.6f checksum_delta=%.9g\n",
                    NumKicks,
                    repeat,
                    oldSeconds,
                    newSeconds,
                    oldSeconds / newSeconds,
                    oldChecksum - newChecksum);
    }
}

int main(int argc, char** argv)
{
    const int numAtoms   = (argc > 1) ? std::atoi(argv[1]) : 7907;
    const int iterations = (argc > 2) ? std::atoi(argv[2]) : 20000;
    std::vector<RVec> invMass(numAtoms);
    std::array<std::vector<RVec>, 3> force{ std::vector<RVec>(numAtoms),
                                             std::vector<RVec>(numAtoms),
                                             std::vector<RVec>(numAtoms) };
    std::vector<RVec> initialPosition(numAtoms);
    std::vector<RVec> initialVelocity(numAtoms);
    for (int atom = 0; atom < numAtoms; ++atom)
    {
        for (int d = 0; d < 3; ++d)
        {
            invMass[atom][d]         = 0.01F + 0.000001F * static_cast<float>((atom + d) % 101);
            initialPosition[atom][d] = 0.1F * static_cast<float>((atom + d) % 37);
            initialVelocity[atom][d] = 0.0001F * static_cast<float>((atom + 3 * d) % 53);
            for (int level = 0; level < 3; ++level)
            {
                force[level][atom][d] = 0.001F * static_cast<float>((atom + d + level) % 29 - 14);
            }
        }
    }

    benchmarkKickCount<1>(iterations, invMass, force, initialPosition, initialVelocity);
    benchmarkKickCount<2>(iterations, invMass, force, initialPosition, initialVelocity);
    benchmarkKickCount<3>(iterations, invMass, force, initialPosition, initialVelocity);
}
