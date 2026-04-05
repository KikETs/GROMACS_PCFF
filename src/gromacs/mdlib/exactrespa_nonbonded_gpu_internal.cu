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
                                            float*                           levelVirials)
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
        atomicAddVirial(levelVirials, level, dx, force);
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
                                          float*                           levelLjEnergies,
                                          float*                           levelCoulombEnergies,
                                          float*                           levelExcludedCoulombEnergies,
                                          float*                           levelVirials)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= params.numPairs)
    {
        return;
    }

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

    const float factorLj      = excluded ? 0.0F : 1.0F;
    const float factorCoulomb = excluded ? 0.0F : 1.0F;
    const auto  splitWeights  = computeSplitWeightsGpu(params, r);

    float rawLjScalar = 0.0F;
    float rawLjEnergy = 0.0F;
    if (factorLj != 0.0F && rsq < params.vdwCutoff2)
    {
        const int   typeI         = atomTypes[ai];
        const int   typeJ         = atomTypes[aj];
        const float c6            = nbfp[typeI * params.ntype2 + typeJ * 2];
        const float cRepulsive    = nbfp[typeI * params.ntype2 + typeJ * 2 + 1];
        const float rinvsix       = rinvsq * rinvsq * rinvsq;
        const float repulsiveTerm = (params.repulsionPower == 12.0F) ? (rinvsix * rinvsix)
                                                                     : powf(rinv, params.repulsionPower);
        rawLjScalar = cRepulsive * repulsiveTerm - c6 * rinvsix;
        rawLjEnergy = cRepulsive * repulsiveTerm * params.invRepulsionPower - c6 * rinvsix * (1.0F / 6.0F);
    }

    float bareCoulombScalar = 0.0F;
    float correctionScalar  = 0.0F;
    float fullCoulombEnergy = 0.0F;
    if (rsq < params.coulombCutoff2)
    {
        const float qq = atomCharges[ai] * atomCharges[aj] * params.epsfac;
        if (qq != 0.0F)
        {
            const float scaledR = r * params.coulombTableScale;
            const int   tableIndex = static_cast<int>(scaledR);
            const float frac       = scaledR - tableIndex;
            const float halfsp     = 0.5F / params.coulombTableScale;
            const float baseF      = coulombTable[tableIndex * 4];
            const float fexcl      = baseF + frac * coulombTable[tableIndex * 4 + 1];
            const float vcorr = coulombTable[tableIndex * 4 + 2] - halfsp * frac * (baseF + fexcl);
            bareCoulombScalar = factorCoulomb * qq * rinv;
            correctionScalar  = -qq * fexcl / rinv;
            fullCoulombEnergy = qq * (factorCoulomb * (rinv - params.ewaldShift) - vcorr);
        }
    }

    const float innerCorrectionScalar  = excluded ? 0.0F : correctionScalar * splitWeights.inner;
    const float middleCorrectionScalar = excluded ? 0.0F : correctionScalar * splitWeights.middle;
    const float outerCorrectionScalar  = excluded ? correctionScalar : correctionScalar * splitWeights.outer;
    const float innerScalar =
            bareCoulombScalar * splitWeights.inner + factorLj * rawLjScalar * splitWeights.inner + innerCorrectionScalar;
    const float middleScalar = bareCoulombScalar * splitWeights.middle
                               + factorLj * rawLjScalar * splitWeights.middle + middleCorrectionScalar;
    const float bareOuterScalar =
            bareCoulombScalar * splitWeights.outer + factorLj * rawLjScalar * splitWeights.outer;
    const float outerScalar = outerCorrectionScalar + bareOuterScalar;

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
                                levelVirials);
    if (params.hasMiddle != 0 && params.middleLevel >= 0)
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
                                    levelVirials);
    }
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
                                levelVirials);

    if ((params.accumulateEnergyMask & (1 << params.outerLevel)) != 0)
    {
        atomicAdd(&levelLjEnergies[params.outerLevel], factorLj * rawLjEnergy);
        atomicAdd(&levelCoulombEnergies[params.outerLevel], fullCoulombEnergy);
        if (excluded)
        {
            atomicAdd(&levelExcludedCoulombEnergies[params.outerLevel], fullCoulombEnergy);
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
                                        DeviceBuffer<float>                        d_levelLjEnergies,
                                        DeviceBuffer<float>                        d_levelCoulombEnergies,
                                        DeviceBuffer<float>                        d_levelExcludedCoulombEnergies,
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
            d_levelLjEnergies,
            d_levelCoulombEnergies,
            d_levelExcludedCoulombEnergies,
            d_levelVirials);
    CU_RET_ERR(cudaGetLastError(), "Launching exact r-RESPA GPU nonbonded kernel failed. ");
}

} // namespace gmx

#endif
