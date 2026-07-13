#include "gmxpre.h"

#include "exactrespa_nonbonded_gpu_internal.h"

#if GMX_GPU_CUDA

#include <cmath>

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/gputraits.cuh"
#include "gromacs/gpu_utils/typecasts_cuda_hip.h"
#include "gromacs/nbnxm/pairlist.h"
#include "gromacs/pbcutil/ishift.h"

namespace gmx
{
namespace
{

constexpr int c_exactRespaGpuThreadsPerBlock = 256;

__device__ float respaSwitchInGpu(const float r, const float off, const float on)
{
    if (on <= off)
    {
        return (r >= on) ? 1.0F : 0.0F;
    }
    if (r <= off)
    {
        return 0.0F;
    }
    if (r >= on)
    {
        return 1.0F;
    }

    const float x = (r - off) / (on - off);
    return x * x * (3.0F - 2.0F * x);
}

struct SplitWeights
{
    float inner  = 0.0F;
    float middle = 0.0F;
    float outer  = 1.0F;
};

__device__ SplitWeights computeSplitWeightsGpu(const ExactRespaGpuRuntimeParams& params, const float r)
{
    SplitWeights weights;
    if (params.hasMiddle != 0)
    {
        const float switchIntoMiddle = respaSwitchInGpu(r, params.innerOff, params.innerOn);
        const float switchIntoOuter  = respaSwitchInGpu(r, params.outerOn, params.outerOff);
        weights.inner               = 1.0F - switchIntoMiddle;
        weights.middle              = switchIntoMiddle * (1.0F - switchIntoOuter);
        weights.outer               = switchIntoOuter;
    }
    else
    {
        const float switchIntoOuter = respaSwitchInGpu(r, params.outerOn, params.outerOff);
        weights.inner              = 1.0F - switchIntoOuter;
        weights.middle             = 0.0F;
        weights.outer              = switchIntoOuter;
    }
    return weights;
}

__device__ float repulsionPowerTermGpu(const float rinv,
                                       const float rinvsq,
                                       const float rinvsix,
                                       const float repulsionPower)
{
    if (repulsionPower == 12.0F)
    {
        return rinvsix * rinvsix;
    }
    if (repulsionPower == 9.0F)
    {
        return rinvsix * rinvsq * rinv;
    }
    return powf(rinv, repulsionPower);
}

__device__ void atomicAddForce(float3* forces,
                               const int numAtoms,
                               const int level,
                               const int atomIndex,
                               const float3 force)
{
    float3* target = &forces[level * numAtoms + atomIndex];
    atomicAdd(&target->x, force.x);
    atomicAdd(&target->y, force.y);
    atomicAdd(&target->z, force.z);
}

__device__ void atomicAddShiftForce(float3* shiftForces,
                                    const int level,
                                    const int shiftIndex,
                                    const int centralShiftIndex,
                                    const float3 force)
{
    float3* targetShift   = &shiftForces[level * c_numShiftVectors + shiftIndex];
    float3* centralTarget = &shiftForces[level * c_numShiftVectors + centralShiftIndex];
    atomicAdd(&targetShift->x, force.x);
    atomicAdd(&targetShift->y, force.y);
    atomicAdd(&targetShift->z, force.z);
    atomicAdd(&centralTarget->x, -force.x);
    atomicAdd(&centralTarget->y, -force.y);
    atomicAdd(&centralTarget->z, -force.z);
}

__device__ void atomicAddVirial(float* virials, const int level, const float3 dx, const float3 force)
{
    const int offset = level * DIM * DIM;
    atomicAdd(&virials[offset + XX * DIM + XX], -0.5F * dx.x * force.x);
    atomicAdd(&virials[offset + XX * DIM + YY], -0.5F * dx.x * force.y);
    atomicAdd(&virials[offset + XX * DIM + ZZ], -0.5F * dx.x * force.z);
    atomicAdd(&virials[offset + YY * DIM + XX], -0.5F * dx.y * force.x);
    atomicAdd(&virials[offset + YY * DIM + YY], -0.5F * dx.y * force.y);
    atomicAdd(&virials[offset + YY * DIM + ZZ], -0.5F * dx.y * force.z);
    atomicAdd(&virials[offset + ZZ * DIM + XX], -0.5F * dx.z * force.x);
    atomicAdd(&virials[offset + ZZ * DIM + YY], -0.5F * dx.z * force.y);
    atomicAdd(&virials[offset + ZZ * DIM + ZZ], -0.5F * dx.z * force.z);
}

__device__ void accumulateLocalVirial(float* virial, const float3 dx, const float3 force)
{
    virial[XX * DIM + XX] += -0.5F * dx.x * force.x;
    virial[XX * DIM + YY] += -0.5F * dx.x * force.y;
    virial[XX * DIM + ZZ] += -0.5F * dx.x * force.z;
    virial[YY * DIM + XX] += -0.5F * dx.y * force.x;
    virial[YY * DIM + YY] += -0.5F * dx.y * force.y;
    virial[YY * DIM + ZZ] += -0.5F * dx.y * force.z;
    virial[ZZ * DIM + XX] += -0.5F * dx.z * force.x;
    virial[ZZ * DIM + YY] += -0.5F * dx.z * force.y;
    virial[ZZ * DIM + ZZ] += -0.5F * dx.z * force.z;
}

__device__ void accumulateLevelContribution(const ExactRespaGpuRuntimeParams& params,
                                            const int                        level,
                                            const float                      scalar,
                                            const float                      rinvsq,
                                            const float3                     dx,
                                            const int                        ai,
                                            const int                        aj,
                                            const int                        shiftIndex,
                                            float3*                          levelForces,
                                            float3*                          levelShiftForces,
                                            float*                           levelVirials,
                                            float*                           outerLevelVirial)
{
    if ((params.activeLevelMask & (1 << level)) == 0 || scalar == 0.0F)
    {
        return;
    }

    const float3 force = make_float3(scalar * rinvsq * dx.x,
                                     scalar * rinvsq * dx.y,
                                     scalar * rinvsq * dx.z);

    atomicAddForce(levelForces, params.numAtoms, level, ai, force);
    atomicAddForce(
            levelForces, params.numAtoms, level, aj, make_float3(-force.x, -force.y, -force.z));

    if ((params.shiftLevelMask & (1 << level)) != 0 && shiftIndex != params.centralShiftIndex)
    {
        atomicAddShiftForce(levelShiftForces, level, shiftIndex, params.centralShiftIndex, force);
    }

    if ((params.directVirialLevelMask & (1 << level)) != 0)
    {
        if (outerLevelVirial != nullptr && level == params.outerLevel)
        {
            accumulateLocalVirial(outerLevelVirial, dx, force);
        }
        else
        {
            atomicAddVirial(levelVirials, level, dx, force);
        }
    }
}

__global__ void exactRespaNonbondedKernel(const ExactRespaGpuRuntimeParams params,
                                          const ExactRespaGpuPairEntry*    pairEntries,
                                          const float3*                    coordinates,
                                          const float3*                    shiftVectors,
                                          const int*                       atomTypes,
                                          const float*                     atomCharges,
                                          const float*                     nbfp,
                                          const float*                     coulombTable,
                                          float3*                          levelForces,
                                          float3*                          levelShiftForces,
                                          float*                           levelEnergies,
                                          float*                           levelVirials)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int threadIndex = static_cast<int>(threadIdx.x);
    const bool havePair = index < params.numPairs;
    const bool accumulateOuterEnergy =
            (params.accumulateEnergyMask & (1 << params.outerLevel)) != 0;
    const bool accumulateOuterDirectVirial =
            (params.directVirialLevelMask & (1 << params.outerLevel)) != 0;

    float threadLjEnergy              = 0.0F;
    float threadCoulombEnergy         = 0.0F;
    float threadExcludedCoulombEnergy = 0.0F;
    float threadVirial[DIM * DIM]     = {};
    __shared__ float sharedEnergy[3][c_exactRespaGpuThreadsPerBlock];
    __shared__ float sharedVirial[DIM * DIM][c_exactRespaGpuThreadsPerBlock];

    if (havePair)
    {
        const ExactRespaGpuPairEntry entry      = pairEntries[index];
        const int                    ai         = entry.ai;
        const int                    aj         = entry.aj;
        const int                    shiftIndex = entry.shiftIndex;
        const bool                   excluded   = (entry.excluded != 0);

        const float3 xi    = coordinates[ai];
        const float3 xj    = coordinates[aj];
        const float3 shift = shiftVectors[shiftIndex];
        const float3 dx =
                make_float3(xi.x + shift.x - xj.x, xi.y + shift.y - xj.y, xi.z + shift.z - xj.z);

        float rsq = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
        rsq       = fmaxf(rsq, c_nbnxnMinDistanceSquared);

        const float rinv   = rsqrtf(rsq);
        const float rinvsq = rinv * rinv;
        const float r      = rsq * rinv;

        if (excluded)
        {
            float correctionScalar  = 0.0F;
            float fullCoulombEnergy = 0.0F;
            if (rsq < params.coulombCutoff2 && params.coulombUsesEwaldTable != 0
                && params.suppressEwaldExcludedAndSelf == 0)
            {
                const float qq = atomCharges[ai] * atomCharges[aj] * params.epsfac;
                if (qq != 0.0F)
                {
                    const float scaledR    = r * params.coulombTableScale;
                    const int   tableIndex = static_cast<int>(scaledR);
                    const float frac       = scaledR - tableIndex;
                    const float halfsp     = 0.5F / params.coulombTableScale;
                    const float baseF      = coulombTable[tableIndex * 4];
                    const float fexcl      = baseF + frac * coulombTable[tableIndex * 4 + 1];
                    const float vcorr =
                            coulombTable[tableIndex * 4 + 2] - halfsp * frac * (baseF + fexcl);
                    correctionScalar  = -qq * fexcl * r;
                    fullCoulombEnergy = -qq * vcorr;
                }
            }

            if ((params.activeLevelMask & (1 << params.outerLevel)) != 0)
            {
                accumulateLevelContribution(params,
                                            params.outerLevel,
                                            correctionScalar,
                                            rinvsq,
                                            dx,
                                            ai,
                                            aj,
                                            shiftIndex,
                                            levelForces,
                                            levelShiftForces,
                                            levelVirials,
                                            accumulateOuterDirectVirial ? threadVirial : nullptr);
            }

            if (accumulateOuterEnergy)
            {
                threadCoulombEnergy         = fullCoulombEnergy;
                threadExcludedCoulombEnergy = fullCoulombEnergy;
            }
        }
        else
        {
            const auto splitWeights = computeSplitWeightsGpu(params, r);

            float rawLjScalar = 0.0F;
            if (rsq < params.vdwCutoff2)
            {
                const int   typeI         = atomTypes[ai];
                const int   typeJ         = atomTypes[aj];
                const float c6            = nbfp[typeI * params.ntype2 + typeJ * 2];
                const float cRepulsive    = nbfp[typeI * params.ntype2 + typeJ * 2 + 1];
                const float rinvsix       = rinvsq * rinvsq * rinvsq;
                const float repulsiveTerm =
                        repulsionPowerTermGpu(rinv, rinvsq, rinvsix, params.repulsionPower);
                rawLjScalar = cRepulsive * repulsiveTerm - c6 * rinvsix;
                if (accumulateOuterEnergy)
                {
                    threadLjEnergy = cRepulsive * repulsiveTerm * params.invRepulsionPower
                                     - c6 * rinvsix * (1.0F / 6.0F);
                }
            }

            float bareCoulombScalar = 0.0F;
            float correctionScalar  = 0.0F;
            float fullCoulombEnergy = 0.0F;
            if (rsq < params.coulombCutoff2)
            {
                const float qq = atomCharges[ai] * atomCharges[aj] * params.epsfac;
                if (qq != 0.0F)
                {
                    bareCoulombScalar = qq * rinv;
                    if (params.coulombUsesEwaldTable != 0)
                    {
                        const float scaledR = r * params.coulombTableScale;
                        const int   tableIndex = static_cast<int>(scaledR);
                        const float frac       = scaledR - tableIndex;
                        const float halfsp     = 0.5F / params.coulombTableScale;
                        const float baseF      = coulombTable[tableIndex * 4];
                        const float fexcl      = baseF + frac * coulombTable[tableIndex * 4 + 1];
                        const float vcorr =
                                coulombTable[tableIndex * 4 + 2] - halfsp * frac * (baseF + fexcl);
                        correctionScalar  = -qq * fexcl * r;
                        fullCoulombEnergy = qq * (rinv - params.ewaldShift - vcorr);
                    }
                    else
                    {
                        fullCoulombEnergy = qq * rinv;
                    }
                }
            }

            const float innerScalar =
                    bareCoulombScalar * splitWeights.inner + rawLjScalar * splitWeights.inner;
            const float middleScalar = bareCoulombScalar * splitWeights.middle
                                       + rawLjScalar * splitWeights.middle;
            const float bareOuterScalar =
                    bareCoulombScalar * splitWeights.outer + rawLjScalar * splitWeights.outer;
            const float outerScalar = correctionScalar + bareOuterScalar;

            if ((params.activeLevelMask & (1 << params.innerLevel)) != 0)
            {
                accumulateLevelContribution(params,
                                            params.innerLevel,
                                            innerScalar,
                                            rinvsq,
                                            dx,
                                            ai,
                                            aj,
                                            shiftIndex,
                                            levelForces,
                                            levelShiftForces,
                                            levelVirials,
                                            accumulateOuterDirectVirial ? threadVirial : nullptr);
            }
            if (params.hasMiddle != 0 && params.middleLevel >= 0
                && (params.activeLevelMask & (1 << params.middleLevel)) != 0)
            {
                accumulateLevelContribution(params,
                                            params.middleLevel,
                                            middleScalar,
                                            rinvsq,
                                            dx,
                                            ai,
                                            aj,
                                            shiftIndex,
                                            levelForces,
                                            levelShiftForces,
                                            levelVirials,
                                            accumulateOuterDirectVirial ? threadVirial : nullptr);
            }
            if ((params.activeLevelMask & (1 << params.outerLevel)) != 0)
            {
                accumulateLevelContribution(params,
                                            params.outerLevel,
                                            outerScalar,
                                            rinvsq,
                                            dx,
                                            ai,
                                            aj,
                                            shiftIndex,
                                            levelForces,
                                            levelShiftForces,
                                            levelVirials,
                                            accumulateOuterDirectVirial ? threadVirial : nullptr);
            }

            if (accumulateOuterEnergy)
            {
                threadCoulombEnergy = fullCoulombEnergy;
            }
        }
    }

    if (accumulateOuterDirectVirial)
    {
        for (int component = 0; component < DIM * DIM; ++component)
        {
            sharedVirial[component][threadIndex] = threadVirial[component];
        }
        __syncthreads();

        for (int stride = c_exactRespaGpuThreadsPerBlock / 2; stride > 0; stride >>= 1)
        {
            if (threadIndex < stride)
            {
                for (int component = 0; component < DIM * DIM; ++component)
                {
                    sharedVirial[component][threadIndex] += sharedVirial[component][threadIndex + stride];
                }
            }
            __syncthreads();
        }

        if (threadIndex == 0)
        {
            const int offset = params.outerLevel * DIM * DIM;
            for (int component = 0; component < DIM * DIM; ++component)
            {
                if (sharedVirial[component][0] != 0.0F)
                {
                    atomicAdd(&levelVirials[offset + component], sharedVirial[component][0]);
                }
            }
        }
    }

    if (accumulateOuterEnergy)
    {
        sharedEnergy[0][threadIndex] = threadLjEnergy;
        sharedEnergy[1][threadIndex] = threadCoulombEnergy;
        sharedEnergy[2][threadIndex] = threadExcludedCoulombEnergy;
        __syncthreads();

        for (int stride = c_exactRespaGpuThreadsPerBlock / 2; stride > 0; stride >>= 1)
        {
            if (threadIndex < stride)
            {
                sharedEnergy[0][threadIndex] += sharedEnergy[0][threadIndex + stride];
                sharedEnergy[1][threadIndex] += sharedEnergy[1][threadIndex + stride];
                sharedEnergy[2][threadIndex] += sharedEnergy[2][threadIndex + stride];
            }
            __syncthreads();
        }

        if (threadIndex == 0)
        {
            if (sharedEnergy[0][0] != 0.0F)
            {
                atomicAdd(&levelEnergies[c_exactRespaGpuLjEnergyOffset + params.outerLevel],
                          sharedEnergy[0][0]);
            }
            if (sharedEnergy[1][0] != 0.0F)
            {
                atomicAdd(&levelEnergies[c_exactRespaGpuCoulombEnergyOffset + params.outerLevel],
                          sharedEnergy[1][0]);
            }
            if (sharedEnergy[2][0] != 0.0F)
            {
                atomicAdd(&levelEnergies[c_exactRespaGpuExcludedCoulombEnergyOffset + params.outerLevel],
                          sharedEnergy[2][0]);
            }
        }
    }
}

} // namespace

void launchExactRespaNonbondedGpuKernel(const ExactRespaGpuRuntimeParams&         params,
                                        const DeviceBuffer<ExactRespaGpuPairEntry>& d_pairEntries,
                                        const DeviceBuffer<Float3>&                d_coordinates,
                                        const DeviceBuffer<Float3>&                d_shiftVectors,
                                        const DeviceBuffer<int>&                   d_atomTypes,
                                        const DeviceBuffer<float>&                 d_atomCharges,
                                        const DeviceBuffer<float>&                 d_nbfp,
                                        const DeviceBuffer<float>&                 d_coulombTable,
                                        DeviceBuffer<Float3>                       d_levelForces,
                                        DeviceBuffer<Float3>                       d_levelShiftForces,
                                        DeviceBuffer<float>                        d_levelEnergies,
                                        DeviceBuffer<float>                        d_levelVirials,
                                        const DeviceStream&                        deviceStream)
{
    if (params.numPairs == 0)
    {
        return;
    }

    const int blocks = (params.numPairs + c_exactRespaGpuThreadsPerBlock - 1) / c_exactRespaGpuThreadsPerBlock;
    const float3* d_coordinatesFloat3 = asFloat3(d_coordinates);
    const float3* d_shiftVectorsFloat3 = asFloat3(d_shiftVectors);
    float3* d_levelForcesFloat3 = asFloat3(d_levelForces);
    float3* d_levelShiftForcesFloat3 = asFloat3(d_levelShiftForces);
    exactRespaNonbondedKernel<<<blocks, c_exactRespaGpuThreadsPerBlock, 0, deviceStream.stream()>>>(
            params,
            d_pairEntries,
            d_coordinatesFloat3,
            d_shiftVectorsFloat3,
            d_atomTypes,
            d_atomCharges,
            d_nbfp,
            d_coulombTable,
            d_levelForcesFloat3,
            d_levelShiftForcesFloat3,
            d_levelEnergies,
            d_levelVirials);
    CU_RET_ERR(cudaGetLastError(), "Launching exact r-RESPA GPU nonbonded kernel failed. ");
}

} // namespace gmx

#endif
