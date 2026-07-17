/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2019- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 *
 * \brief Implements Leap-Frog using CUDA
 *
 * This file contains CUDA implementation of back-end specific code for Leap-Frog.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "leapfrog_gpu_internal.h"

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/typecasts_cuda_hip.h"
#include "gromacs/gpu_utils/vectype_ops_cuda.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pbcutil/pbc_aiuc_cuda.cuh"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/template_mp.h"
#include "gromacs/utility/vec.h"

namespace gmx
{

/*!\brief Number of CUDA threads in a block
 *
 * \todo Check if using smaller block size will lead to better performance.
 */
constexpr static int c_threadsPerBlock = 256;
//! Maximum number of threads in a block (for __launch_bounds__)
constexpr static int c_maxThreadsPerBlock = c_threadsPerBlock;

/*! \brief Main kernel for Leap-Frog integrator.
 *
 *  The coordinates and velocities are updated on the GPU. Also saves the intermediate values of the coordinates for
 *   further use in constraints.
 *
 *  Each GPU thread works with a single particle. Empty declaration is needed to
 *  avoid "no previous prototype for function" clang warning.
 *
 *  \todo Check if the force should be set to zero here.
 *  \todo This kernel can also accumulate incidental temperatures for each atom.
 *
 * \tparam        numTempScaleValues               The number of different T-couple values.
 * \tparam        parrinelloRahmanVelocityScaling  The properties of the Parrinello-Rahman velocity scaling matrix.
 * \param[in]     numAtoms                         Total number of atoms.
 * \param[in,out] gm_x                             Coordinates to update upon integration.
 * \param[out]    gm_x0                            A copy of the coordinates before the integration (for constraints).
 * \param[in,out] gm_v                             Velocities to update.
 * \param[in]     gm_f                             Atomic forces.
 * \param[in]     gm_inverseMasses                 Reciprocal masses.
 * \param[in]     dt                               Timestep.
 * \param[in]     gm_lambdas                       Temperature scaling factors (one per group)
 * \param[in]     gm_tempScaleGroups               Mapping of atoms into groups.
 * \param[in]     prVelocityScalingMatrixDiagonal  Diagonal elements of Parrinello-Rahman velocity scaling matrix
 */
template<NumTempScaleValues numTempScaleValues, ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling>
__launch_bounds__(c_maxThreadsPerBlock) __global__
        void leapFrogKernel(const int numAtoms,
                            float3* __restrict__ gm_x,
                            float3* __restrict__ gm_x0,
                            float3* __restrict__ gm_v,
                            const float3* __restrict__ gm_f,
                            const float* __restrict__ gm_inverseMasses,
                            const float dt,
                            const float* __restrict__ gm_lambdas,
                            const unsigned short* __restrict__ gm_tempScaleGroups,
                            const float3 prVelocityScalingMatrixDiagonal)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex < numAtoms)
    {
        float3 x    = gm_x[threadIndex];
        float3 v    = gm_v[threadIndex];
        float3 f    = gm_f[threadIndex];
        float  im   = gm_inverseMasses[threadIndex];
        float  imdt = im * dt;

        gm_x0[threadIndex] = x;

        if constexpr (numTempScaleValues != NumTempScaleValues::None
                      || parrinelloRahmanVelocityScaling != ParrinelloRahmanVelocityScaling::No)
        {
            float3 vp = v;

            if constexpr (numTempScaleValues != NumTempScaleValues::None)
            {
                float lambda = 1.0F;
                if constexpr (numTempScaleValues == NumTempScaleValues::Single)
                {
                    lambda = gm_lambdas[0];
                }
                else if constexpr (numTempScaleValues == NumTempScaleValues::Multiple)
                {
                    int tempScaleGroup = gm_tempScaleGroups[threadIndex];
                    lambda             = gm_lambdas[tempScaleGroup];
                }
                vp *= lambda;
            }

            if constexpr (parrinelloRahmanVelocityScaling == ParrinelloRahmanVelocityScaling::Diagonal)
            {
                vp.x -= prVelocityScalingMatrixDiagonal.x * v.x;
                vp.y -= prVelocityScalingMatrixDiagonal.y * v.y;
                vp.z -= prVelocityScalingMatrixDiagonal.z * v.z;
            }

            v = vp;
        }

        v += f * imdt;

        x += v * dt;
        gm_v[threadIndex] = v;
        gm_x[threadIndex] = x;
    }
}

__launch_bounds__(c_maxThreadsPerBlock) __global__
        void leapFrogDriftOnlyKernel(const int numAtoms,
                                     float3* __restrict__ gm_x,
                                     const float3* __restrict__ gm_v,
                                     const float dt)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex < numAtoms)
    {
        const float3 x = gm_x[threadIndex];
        const float3 v = gm_v[threadIndex];
        gm_x[threadIndex] = make_float3(__fadd_rn(x.x, __fmul_rn(v.x, dt)),
                                        __fadd_rn(x.y, __fmul_rn(v.y, dt)),
                                        __fadd_rn(x.z, __fmul_rn(v.z, dt)));
    }
}

template<bool initialPhase>
__launch_bounds__(c_maxThreadsPerBlock) __global__
        void exactRespaKickKernel(const int numAtoms,
                                 float3* __restrict__ gm_x,
                                 float3* __restrict__ gm_v,
                                 const float3* __restrict__ gm_level0Force,
                                 const float3* __restrict__ gm_level1Force,
                                 const float3* __restrict__ gm_level2Force,
                                 const float* __restrict__ gm_inverseMasses,
                                 const int* __restrict__ gm_stateToNbnxm,
                                 const int   highestActiveLevel,
                                 const float level0HalfDt,
                                 const float level1HalfDt,
                                 const float level2HalfDt,
                                 const float driftDt,
                                 const float velocityScaleFirst,
                                 const float velocityScaleSecond)
{
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= numAtoms)
    {
        return;
    }

    const int    nbnxmAtom = gm_stateToNbnxm[atom];
    const float  invMass   = gm_inverseMasses[atom];
    float3       velocity  = gm_v[atom];
    const float3 force0    = gm_level0Force[nbnxmAtom];

    if constexpr (initialPhase)
    {
        const auto applyVelocityScale = [&](const float scale)
        {
            if (scale != 1.0F)
            {
                velocity.x = __fmul_rn(velocity.x, scale);
                velocity.y = __fmul_rn(velocity.y, scale);
                velocity.z = __fmul_rn(velocity.z, scale);
            }
        };
        applyVelocityScale(velocityScaleFirst);
        applyVelocityScale(velocityScaleSecond);
    }

    const auto applyKick = [&](const float3 force, const float halfDt)
    {
        const float scale = __fmul_rn(invMass, halfDt);
        velocity.x        = __fadd_rn(velocity.x, __fmul_rn(scale, force.x));
        velocity.y        = __fadd_rn(velocity.y, __fmul_rn(scale, force.y));
        velocity.z        = __fadd_rn(velocity.z, __fmul_rn(scale, force.z));
    };

    if constexpr (initialPhase)
    {
        if (highestActiveLevel >= 2)
        {
            applyKick(gm_level2Force[atom], level2HalfDt);
        }
        if (highestActiveLevel >= 1)
        {
            applyKick(gm_level1Force[nbnxmAtom], level1HalfDt);
        }
        applyKick(force0, level0HalfDt);
    }
    else
    {
        applyKick(force0, level0HalfDt);
        if (highestActiveLevel >= 1)
        {
            applyKick(gm_level1Force[nbnxmAtom], level1HalfDt);
        }
        if (highestActiveLevel >= 2)
        {
            applyKick(gm_level2Force[atom], level2HalfDt);
        }
    }

    gm_v[atom] = velocity;
    if constexpr (initialPhase)
    {
        const float3 position = gm_x[atom];
        gm_x[atom] = make_float3(__fadd_rn(position.x, __fmul_rn(velocity.x, driftDt)),
                                 __fadd_rn(position.y, __fmul_rn(velocity.y, driftDt)),
                                 __fadd_rn(position.z, __fmul_rn(velocity.z, driftDt)));
    }
}

__launch_bounds__(c_maxThreadsPerBlock) __global__
        void exactRespaKineticEnergyPartialKernel(const int numAtoms,
                                                  const float3* __restrict__ gm_v,
                                                  const float* __restrict__ gm_masses,
                                                  float* __restrict__ gm_partialKineticEnergy)
{
    const int thread = threadIdx.x;
    int       atom   = blockIdx.x * blockDim.x + thread;
    float     kineticEnergy = 0.0F;

    while (atom < numAtoms)
    {
        const float3 velocity = gm_v[atom];
        const float  halfMass = __fmul_rn(0.5F, gm_masses[atom]);
        kineticEnergy = __fadd_rn(
                kineticEnergy, __fmul_rn(__fmul_rn(halfMass, velocity.x), velocity.x));
        kineticEnergy = __fadd_rn(
                kineticEnergy, __fmul_rn(__fmul_rn(halfMass, velocity.y), velocity.y));
        kineticEnergy = __fadd_rn(
                kineticEnergy, __fmul_rn(__fmul_rn(halfMass, velocity.z), velocity.z));
        atom += gridDim.x * blockDim.x;
    }

    __shared__ float reduction[c_threadsPerBlock];
    reduction[thread] = kineticEnergy;
    __syncthreads();

    for (int stride = c_threadsPerBlock / 2; stride > 0; stride /= 2)
    {
        if (thread < stride)
        {
            reduction[thread] = __fadd_rn(reduction[thread], reduction[thread + stride]);
        }
        __syncthreads();
    }

    if (thread == 0)
    {
        gm_partialKineticEnergy[blockIdx.x] = reduction[0];
    }
}

__launch_bounds__(c_maxThreadsPerBlock) __global__
        void exactRespaKineticEnergyFinalizeKernel(const float* __restrict__ gm_partialKineticEnergy,
                                                   float* __restrict__ gm_kineticEnergy)
{
    const int thread = threadIdx.x;
    __shared__ float reduction[c_threadsPerBlock];
    reduction[thread] = thread < c_exactRespaKineticReductionBlocks
                                ? gm_partialKineticEnergy[thread]
                                : 0.0F;
    __syncthreads();

    for (int stride = c_threadsPerBlock / 2; stride > 0; stride /= 2)
    {
        if (thread < stride)
        {
            reduction[thread] = __fadd_rn(reduction[thread], reduction[thread + stride]);
        }
        __syncthreads();
    }

    if (thread == 0)
    {
        gm_kineticEnergy[0] = reduction[0];
    }
}

__launch_bounds__(c_maxThreadsPerBlock) __global__
        void exactRespaScaleVelocityKernel(const int numAtoms,
                                           float3* __restrict__ gm_v,
                                           const float velocityScale)
{
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom < numAtoms)
    {
        const float3 velocity = gm_v[atom];
        gm_v[atom] = make_float3(__fmul_rn(velocity.x, velocityScale),
                                 __fmul_rn(velocity.y, velocityScale),
                                 __fmul_rn(velocity.z, velocityScale));
    }
}

void launchLeapFrogKernel(const int                             numAtoms,
                          DeviceBuffer<Float3>                  d_x,
                          DeviceBuffer<Float3>                  d_xp,
                          DeviceBuffer<Float3>                  d_v,
                          const DeviceBuffer<Float3>            d_f,
                          const DeviceBuffer<float>             d_inverseMasses,
                          const float                           dt,
                          const bool                            doTemperatureScaling,
                          const int                             numTempScaleValues,
                          const DeviceBuffer<unsigned short>    d_tempScaleGroups,
                          const DeviceBuffer<float>             d_lambdas,
                          const ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
                          const Float3                          prVelocityScalingMatrixDiagonal,
                          const DeviceStream&                   deviceStream)
{
    // Checking the buffer types against the kernel argument types
    static_assert(sizeof(*d_inverseMasses) == sizeof(float), "Incompatible types");

    KernelLaunchConfig kernelLaunchConfig;

    kernelLaunchConfig.gridSize[0]      = divideRoundUp(numAtoms, c_threadsPerBlock);
    kernelLaunchConfig.blockSize[0]     = c_threadsPerBlock;
    kernelLaunchConfig.blockSize[1]     = 1;
    kernelLaunchConfig.blockSize[2]     = 1;
    kernelLaunchConfig.sharedMemorySize = 0;

    gmx::dispatchTemplatedFunction(
            [&](auto tempScalingType_, auto pressureScalingType_)
            {
                auto kernelPtr = leapFrogKernel<tempScalingType_, pressureScalingType_>;

                const auto kernelArgs = prepareGpuKernelArguments(kernelPtr,
                                                                  kernelLaunchConfig,
                                                                  &numAtoms,
                                                                  asFloat3Pointer(&d_x),
                                                                  asFloat3Pointer(&d_xp),
                                                                  asFloat3Pointer(&d_v),
                                                                  asFloat3Pointer(&d_f),
                                                                  &d_inverseMasses,
                                                                  &dt,
                                                                  &d_lambdas,
                                                                  &d_tempScaleGroups,
                                                                  &prVelocityScalingMatrixDiagonal);
                launchGpuKernel(
                        kernelPtr, kernelLaunchConfig, deviceStream, nullptr, "leapfrog_kernel", kernelArgs);
            },
            getTempScalingType(doTemperatureScaling, numTempScaleValues),
            parrinelloRahmanVelocityScaling);
}

void launchLeapFrogDriftOnlyKernel(const int        numAtoms,
                                   DeviceBuffer<Float3> d_x,
                                   DeviceBuffer<Float3> d_v,
                                   const float      dt,
                                   const DeviceStream& deviceStream)
{
    KernelLaunchConfig kernelLaunchConfig;

    kernelLaunchConfig.gridSize[0]      = divideRoundUp(numAtoms, c_threadsPerBlock);
    kernelLaunchConfig.blockSize[0]     = c_threadsPerBlock;
    kernelLaunchConfig.blockSize[1]     = 1;
    kernelLaunchConfig.blockSize[2]     = 1;
    kernelLaunchConfig.sharedMemorySize = 0;

    const auto kernelArgs = prepareGpuKernelArguments(leapFrogDriftOnlyKernel,
                                                      kernelLaunchConfig,
                                                      &numAtoms,
                                                      asFloat3Pointer(&d_x),
                                                      asFloat3Pointer(&d_v),
                                                      &dt);
    launchGpuKernel(leapFrogDriftOnlyKernel,
                    kernelLaunchConfig,
                    deviceStream,
                    nullptr,
                    "leapfrog_drift_only_kernel",
                    kernelArgs);
}

void launchExactRespaKickKernel(const int                  numAtoms,
                               DeviceBuffer<Float3>        d_x,
                               DeviceBuffer<Float3>        d_v,
                               const DeviceBuffer<Float3>  d_level0Force,
                               const DeviceBuffer<Float3>  d_level1Force,
                               const DeviceBuffer<Float3>  d_level2Force,
                               const DeviceBuffer<float>   d_inverseMasses,
                               const DeviceBuffer<int>     d_stateToNbnxm,
                               const int                   highestActiveLevel,
                               const float                 level0HalfDt,
                               const float                 level1HalfDt,
                               const float                 level2HalfDt,
                               const float                 driftDt,
                               const float                 velocityScaleFirst,
                               const float                 velocityScaleSecond,
                               const bool                  initialPhase,
                               const DeviceStream&         deviceStream)
{
    GMX_RELEASE_ASSERT(highestActiveLevel >= 0 && highestActiveLevel <= 2,
                       "Exact r-RESPA device kick supports 1-3 levels");
    GMX_RELEASE_ASSERT(d_level0Force != nullptr, "Exact r-RESPA level-0 device force is missing");
    GMX_RELEASE_ASSERT(highestActiveLevel < 1 || d_level1Force != nullptr,
                       "Exact r-RESPA level-1 device force is missing");
    GMX_RELEASE_ASSERT(highestActiveLevel < 2 || d_level2Force != nullptr,
                       "Exact r-RESPA level-2 device force is missing");

    KernelLaunchConfig config;
    config.gridSize[0]      = divideRoundUp(numAtoms, c_threadsPerBlock);
    config.blockSize[0]     = c_threadsPerBlock;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.sharedMemorySize = 0;

    const auto launch = [&](auto kernelPtr)
    {
        const auto kernelArgs = prepareGpuKernelArguments(kernelPtr,
                                                          config,
                                                          &numAtoms,
                                                          asFloat3Pointer(&d_x),
                                                          asFloat3Pointer(&d_v),
                                                          asFloat3Pointer(&d_level0Force),
                                                          asFloat3Pointer(&d_level1Force),
                                                          asFloat3Pointer(&d_level2Force),
                                                          &d_inverseMasses,
                                                          &d_stateToNbnxm,
                                                          &highestActiveLevel,
                                                          &level0HalfDt,
                                                          &level1HalfDt,
                                                          &level2HalfDt,
                                                          &driftDt,
                                                          &velocityScaleFirst,
                                                          &velocityScaleSecond);
        launchGpuKernel(kernelPtr,
                        config,
                        deviceStream,
                        nullptr,
                        initialPhase ? "exact_respa_kick_drift" : "exact_respa_kick",
                        kernelArgs);
    };

    if (initialPhase)
    {
        launch(exactRespaKickKernel<true>);
    }
    else
    {
        launch(exactRespaKickKernel<false>);
    }
}

void launchExactRespaKineticEnergyKernel(const int                  numAtoms,
                                         DeviceBuffer<Float3>        d_v,
                                         const DeviceBuffer<float>   d_masses,
                                         DeviceBuffer<float>         d_partialKineticEnergy,
                                         DeviceBuffer<float>         d_kineticEnergy,
                                         const DeviceStream&         deviceStream)
{
    GMX_RELEASE_ASSERT(d_v != nullptr, "Exact r-RESPA device velocities are missing");
    GMX_RELEASE_ASSERT(d_masses != nullptr, "Exact r-RESPA device masses are missing");
    GMX_RELEASE_ASSERT(d_partialKineticEnergy != nullptr,
                       "Exact r-RESPA kinetic partial buffer is missing");
    GMX_RELEASE_ASSERT(d_kineticEnergy != nullptr,
                       "Exact r-RESPA kinetic result buffer is missing");

    KernelLaunchConfig partialConfig;
    partialConfig.gridSize[0]      = c_exactRespaKineticReductionBlocks;
    partialConfig.blockSize[0]     = c_threadsPerBlock;
    partialConfig.blockSize[1]     = 1;
    partialConfig.blockSize[2]     = 1;
    partialConfig.sharedMemorySize = 0;
    const auto partialArgs = prepareGpuKernelArguments(exactRespaKineticEnergyPartialKernel,
                                                       partialConfig,
                                                       &numAtoms,
                                                       asFloat3Pointer(&d_v),
                                                       &d_masses,
                                                       &d_partialKineticEnergy);
    launchGpuKernel(exactRespaKineticEnergyPartialKernel,
                    partialConfig,
                    deviceStream,
                    nullptr,
                    "exact_respa_kinetic_partial",
                    partialArgs);

    KernelLaunchConfig finalizeConfig;
    finalizeConfig.gridSize[0]      = 1;
    finalizeConfig.blockSize[0]     = c_threadsPerBlock;
    finalizeConfig.blockSize[1]     = 1;
    finalizeConfig.blockSize[2]     = 1;
    finalizeConfig.sharedMemorySize = 0;
    const auto finalizeArgs = prepareGpuKernelArguments(exactRespaKineticEnergyFinalizeKernel,
                                                        finalizeConfig,
                                                        &d_partialKineticEnergy,
                                                        &d_kineticEnergy);
    launchGpuKernel(exactRespaKineticEnergyFinalizeKernel,
                    finalizeConfig,
                    deviceStream,
                    nullptr,
                    "exact_respa_kinetic_finalize",
                    finalizeArgs);
}

void launchExactRespaScaleVelocityKernel(const int                  numAtoms,
                                         DeviceBuffer<Float3>        d_v,
                                         const float                 velocityScale,
                                         const DeviceStream&         deviceStream)
{
    GMX_RELEASE_ASSERT(d_v != nullptr, "Exact r-RESPA device velocities are missing");

    KernelLaunchConfig config;
    config.gridSize[0]      = divideRoundUp(numAtoms, c_threadsPerBlock);
    config.blockSize[0]     = c_threadsPerBlock;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.sharedMemorySize = 0;
    const auto kernelArgs = prepareGpuKernelArguments(exactRespaScaleVelocityKernel,
                                                      config,
                                                      &numAtoms,
                                                      asFloat3Pointer(&d_v),
                                                      &velocityScale);
    launchGpuKernel(exactRespaScaleVelocityKernel,
                    config,
                    deviceStream,
                    nullptr,
                    "exact_respa_scale_velocity",
                    kernelArgs);
}

} // namespace gmx
