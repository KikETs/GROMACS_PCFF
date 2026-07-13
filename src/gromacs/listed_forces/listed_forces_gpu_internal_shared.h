/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2022- The GROMACS Authors
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
 * \brief Implements generic GPU bonded functionality
 *
 * \author Andrey Alekseenko <al42and@gmail.com>
 * \author Jon Vincent <jvincent@nvidia.com>
 * \author Magnus Lundborg <lundborg.magnus@gmail.com>
 * \author Berk Hess <hess@kth.se>
 * \author Szilárd Páll <pall.szilard@gmail.com>
 * \author Alan Gray <alang@nvidia.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_listed_forces
 */
#ifndef GMX_LISTED_FORCES_LISTED_FORCES_GPU_INTERNAL_SHARED_H
#define GMX_LISTED_FORCES_LISTED_FORCES_GPU_INTERNAL_SHARED_H

#include "config.h"

#include "gromacs/pbcutil/ishift.h"

#if GMX_GPU_CUDA
#    include "gromacs/gpu_utils/cuda_kernel_utils.cuh"
#    include "gromacs/pbcutil/pbc_aiuc_cuda.cuh"
#elif GMX_GPU_SYCL
#    include "gromacs/gpu_utils/sycl_kernel_utils.h"
#    include "gromacs/gpu_utils/vectype_ops_sycl.h"
#    include "gromacs/pbcutil/pbc_aiuc_sycl.h"
#elif GMX_GPU_HIP
#    include "gromacs/gpu_utils/hip_kernel_utils.h"
#    include "gromacs/gpu_utils/vectype_ops_hip.h"
#    include "gromacs/pbcutil/pbc_aiuc_hip.h"
#else
#    error Building GPU bonded for unsupported backend
#endif
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/math/units.h"

#include "listed_forces_gpu_impl.h"

#ifndef DOXYGEN

namespace gmx
{

static constexpr float c_deg2RadF = gmx::c_deg2Rad;
constexpr float        c_Pi       = M_PI;

GMX_DEVICE_FUNC_ATTRIBUTE static inline bool pcffClass2DebugModeEnabled(const int debugMode)
{
    return debugMode != static_cast<int>(PcffClass2DebugMode::None);
}

GMX_DEVICE_FUNC_ATTRIBUTE static inline void pcffClass2PhaseSinCosGpu(const float phase,
                                                                      float*      cosPhase,
                                                                      float*      sinPhase)
{
    if (phase == 0.0F)
    {
        *cosPhase = 1.0F;
        *sinPhase = 0.0F;
        return;
    }
    if (phase == c_Pi || phase == -c_Pi)
    {
        *cosPhase = -1.0F;
        *sinPhase = 0.0F;
        return;
    }

    *cosPhase = gmxDeviceCos(phase);
    *sinPhase = gmxDeviceSin(phase);
}

/* Some SYCL targets have troubles optimizing the dynamic array
 * member access despite the fact that all the loops are unrolled.
 *
 * See https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/
 * for a details on why dynamic access is problematic.
 *
 * This seems to affect:
 * - hipSYCL 0.9.4 + Clang 14-15 for AMD (but not NVIDIA),
 * - IntelLLVM 2023-02 for NVIDIA and Arc (but to a much lesser extent PVC).
 *
 * This wrapper avoid dynamic accesses into the array, replacing them
 * with a `switch` instead.
 *
 * Based on the optimization by AMD/StreamHPC for their HIP port.
 */
template<typename T>
struct FTypeArray
{
    static_assert(gmx::numFTypesOnGpu == 12,
                  "Please update the member initializer list and the switch below");
    constexpr FTypeArray(const T in[gmx::numFTypesOnGpu]) :
        data{ in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8], in[9], in[10], in[11] }
    {
    }
    GMX_DEVICE_FUNC_ATTRIBUTE constexpr T operator[](int idx) const
    {
        switch (idx)
        {
            case 0: return data[0];
            case 1: return data[1];
            case 2: return data[2];
            case 3: return data[3];
            case 4: return data[4];
            case 5: return data[5];
            case 6: return data[6];
            case 7: return data[7];
            case 8: return data[8];
            case 9: return data[9];
            case 10: return data[10];
            default: return data[11];
        }
    }
    T data[gmx::numFTypesOnGpu];
};

GMX_DEVICE_FUNC_ATTRIBUTE static inline float clampToRange(const float value, const float lower, const float upper)
{
    return (value < lower) ? lower : ((value > upper) ? upper : value);
}

GMX_DEVICE_FUNC_ATTRIBUTE static inline float maxFloat(const float a, const float b)
{
    return (a > b) ? a : b;
}

GMX_DEVICE_FUNC_ATTRIBUTE static inline float absFloat(const float value)
{
    return (value < 0.0F) ? -value : value;
}

GMX_DEVICE_FUNC_ATTRIBUTE static inline float gmxDeviceAsin(float input)
{
    input = clampToRange(input, -1.0F, 1.0F);
#if GMX_GPU_SYCL
    return sycl::asin(input);
#else
    return asinf(input);
#endif
}

GMX_DEVICE_FUNC_ATTRIBUTE static inline float invCosFromSinArgumentGpu(const float value, const float minCosSq)
{
    const float sinValue = clampToRange(value, -1.0F, 1.0F);
    const float cosSq    = maxFloat(1.0F - sinValue * sinValue, minCosSq);
    return gmxDeviceRSqrt(cosSq);
}

GMX_DEVICE_FUNC_ATTRIBUTE static inline float component(const DeviceFloat3& value, const int dim)
{
#if GMX_GPU_SYCL
    return value[dim];
#else
    switch (dim)
    {
        case 0: return value.x;
        case 1: return value.y;
        default: return value.z;
    }
#endif
}

GMX_DEVICE_FUNC_ATTRIBUTE static inline float& componentRef(DeviceFloat3& value, const int dim)
{
#if GMX_GPU_SYCL
    return value[dim];
#else
    switch (dim)
    {
        case 0: return value.x;
        case 1: return value.y;
        default: return value.z;
    }
#endif
}

GMX_DEVICE_FUNC_ATTRIBUTE static inline void pcffClass2CrossGpu(const float a[DIM],
                                                                const float b[DIM],
                                                                float       c[DIM])
{
    c[XX] = a[YY] * b[ZZ] - a[ZZ] * b[YY];
    c[YY] = a[ZZ] * b[XX] - a[XX] * b[ZZ];
    c[ZZ] = a[XX] * b[YY] - a[YY] * b[XX];
}

GMX_DEVICE_FUNC_ATTRIBUTE static inline float pcffClass2DotGpu(const float a[DIM], const float b[DIM])
{
    return a[XX] * b[XX] + a[YY] * b[YY] + a[ZZ] * b[ZZ];
}

/* Harmonic */
template<bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void harmonic_gpu(const float             kA,
                                                          const float             xA,
                                                          const float             x,
                                                          DevicePrivatePtr<float> V,
                                                          DevicePrivatePtr<float> F)
{
    constexpr float half = 0.5F;
    float           dx   = x - xA;
    float           dx2  = dx * dx;

    *F = -kA * dx;
    if constexpr (calcEner)
    {
        *V = half * kA * dx2;
    }
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void bonds_gpu(const int               i,
                                                       DevicePrivatePtr<float> vtot_loc,
                                                       const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                       const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                       const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                       DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                       DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                       const PbcAiuc&                pbcAiuc,
                                                       const int                     localId)
{
    const int type = gm_forceatoms[3 * i];
    const int ai   = gm_forceatoms[3 * i + 1];
    const int aj   = gm_forceatoms[3 * i + 2];

    /* dx = xi - xj, corrected for periodic boundary conditions. */
    DeviceFloat3 dx;
    int          ki = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dx);

    float dr2 = gmxDeviceNorm2(dx);
    float dr  = gmxDeviceSqrt(dr2);

    float vbond;
    float fbond;
    harmonic_gpu<calcEner>(
            gm_forceparams[type].harmonic.krA, gm_forceparams[type].harmonic.rA, dr, &vbond, &fbond);

    if constexpr (calcEner)
    {
        *vtot_loc += vbond;
    }

    if (dr2 != 0.0F)
    {
        fbond *= gmxDeviceRSqrt(dr2);

        DeviceFloat3 fij = fbond * dx;
        staggeredAtomicAddForce(gm_f, fij, ai, localId);
        staggeredAtomicAddForce(gm_f, -fij, aj, localId);
        if constexpr (calcVir)
        {
            if (ki != gmx::c_centralShiftIndex)
            {
                atomicFetchAddLocal(sm_fShiftLoc, ki, fij);
                atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, -fij);
            }
        }
    }
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void bond_class2_gpu(const int               i,
                                                             DevicePrivatePtr<float> vtot_loc,
                                                             const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                             const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                             const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                             DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                             DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                             const PbcAiuc&                pbcAiuc,
                                                             const int                     pcffClass2DebugMode,
                                                             const int                     localId)
{
    const int type = gm_forceatoms[3 * i];
    const int ai   = gm_forceatoms[3 * i + 1];
    const int aj   = gm_forceatoms[3 * i + 2];

    const auto& params = gm_forceparams[type].bond_class2;

    DeviceFloat3 dx;
    const int    ki  = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dx);
    const float  rsq = gmxDeviceNorm2(dx);
    const float  invR = (rsq > 0.0F) ? gmxDeviceRSqrt(rsq) : 0.0F;
    const float  r   = rsq * invR;
    const float  dr  = r - params.r0;
    const float  dr2 = dr * dr;
    const float  dr3 = dr2 * dr;
    float        de  = 0.0F;
    if (!pcffClass2DebugModeEnabled(pcffClass2DebugMode))
    {
        de = 2.0F * params.k2 * dr + 3.0F * params.k3 * dr2 + 4.0F * params.k4 * dr3;
    }
    else
    {
        switch (static_cast<PcffClass2DebugMode>(pcffClass2DebugMode))
        {
            case PcffClass2DebugMode::BondClass2K2Only:
                de = 2.0F * params.k2 * dr;
                break;
            case PcffClass2DebugMode::BondClass2K3Only:
                de = 3.0F * params.k3 * dr2;
                break;
            case PcffClass2DebugMode::BondClass2K4Only:
                de = 4.0F * params.k4 * dr3;
                break;
            default: break;
        }
    }
    const float fbond = -de * invR;

    if constexpr (calcEner)
    {
        float energyContribution = 0.0F;
        const float dr4          = dr3 * dr;
        if (!pcffClass2DebugModeEnabled(pcffClass2DebugMode))
        {
            energyContribution = params.k2 * dr2 + params.k3 * dr3 + params.k4 * dr4;
        }
        else
        {
            switch (static_cast<PcffClass2DebugMode>(pcffClass2DebugMode))
            {
                case PcffClass2DebugMode::BondClass2K2Only:
                    energyContribution = params.k2 * dr2;
                    break;
                case PcffClass2DebugMode::BondClass2K3Only:
                    energyContribution = params.k3 * dr3;
                    break;
                case PcffClass2DebugMode::BondClass2K4Only:
                    energyContribution = params.k4 * dr4;
                    break;
                default: break;
            }
        }
        *vtot_loc += energyContribution;
    }

    if (rsq > 0.0F)
    {
        const DeviceFloat3 fij = fbond * dx;
        staggeredAtomicAddForce(gm_f, fij, ai, localId);
        staggeredAtomicAddForce(gm_f, -fij, aj, localId);
        if constexpr (calcVir)
        {
            if (ki != gmx::c_centralShiftIndex)
            {
                atomicFetchAddLocal(sm_fShiftLoc, ki, fij);
                atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, -fij);
            }
        }
    }
}

template<bool returnShift>
GMX_DEVICE_FUNC_ATTRIBUTE static inline float bond_angle_gpu(const DeviceFloat4             xi,
                                                             const DeviceFloat4             xj,
                                                             const DeviceFloat4             xk,
                                                             const PbcAiuc&                 pbcAiuc,
                                                             DevicePrivatePtr<DeviceFloat3> r_ij,
                                                             DevicePrivatePtr<DeviceFloat3> r_kj,
                                                             DevicePrivatePtr<float>        costh,
                                                             DevicePrivatePtr<int>          t1,
                                                             DevicePrivatePtr<int>          t2)
{
    *t1 = pbcDxAiucGpu<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiucGpu<returnShift>(pbcAiuc, xk, xj, *r_kj);

    *costh = gmxDeviceCosAngle(*r_ij, *r_kj);
    // Return value is the angle between the bonds i-j and j-k
    return gmxDeviceAcos(*costh);
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void angles_gpu(const int               i,
                                                        DevicePrivatePtr<float> vtot_loc,
                                                        const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                        const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                        const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                        DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                        DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                        const PbcAiuc&                pbcAiuc,
                                                        const int                     localId)
{
    DeviceInt4 angleData = loadInt4(gm_forceatoms, i);
    const int  type      = angleData[0];
    const int  ai        = angleData[1];
    const int  aj        = angleData[2];
    const int  ak        = angleData[3];

    DeviceFloat3 r_ij;
    DeviceFloat3 r_kj;
    float        cos_theta;
    int          t1;
    int          t2;
    float        theta = bond_angle_gpu<calcVir>(
            gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj, &cos_theta, &t1, &t2);

    float va;
    float dVdt;
    harmonic_gpu<calcEner>(
            gm_forceparams[type].harmonic.krA, gm_forceparams[type].harmonic.rA * c_deg2RadF, theta, &va, &dVdt);

    if constexpr (calcEner)
    {
        *vtot_loc += va;
    }

    float cos_theta2 = cos_theta * cos_theta;
    if (cos_theta2 < 1.0F)
    {
        float st    = dVdt * gmxDeviceRSqrt(1.0F - cos_theta2);
        float sth   = st * cos_theta;
        float nrij2 = gmxDeviceNorm2(r_ij);
        float nrkj2 = gmxDeviceNorm2(r_kj);

        float nrij_1 = gmxDeviceRSqrt(nrij2);
        float nrkj_1 = gmxDeviceRSqrt(nrkj2);

        float cik = st * nrij_1 * nrkj_1;
        float cii = sth * nrij_1 * nrij_1;
        float ckk = sth * nrkj_1 * nrkj_1;

        DeviceFloat3 f_i = cii * r_ij - cik * r_kj;
        DeviceFloat3 f_k = ckk * r_kj - cik * r_ij;
        DeviceFloat3 f_j = -f_i - f_k;

        staggeredAtomicAddForce(gm_f, f_i, ai, localId);
        staggeredAtomicAddForce(gm_f, f_j, aj, localId);
        staggeredAtomicAddForce(gm_f, f_k, ak, localId);

        if constexpr (calcVir)
        {
            atomicFetchAddLocal(sm_fShiftLoc, t1, f_i);
            atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, f_j);
            atomicFetchAddLocal(sm_fShiftLoc, t2, f_k);
        }
    }
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void angle_class2_gpu(const int               i,
                                                              DevicePrivatePtr<float> vtot_loc,
                                                              const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                              const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                              const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                              DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                              DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                              const PbcAiuc&                pbcAiuc,
                                                              const int                     pcffClass2DebugMode,
                                                              const int                     localId)
{
    constexpr float c_small = 0.001F;

    DeviceInt4 angleData = loadInt4(gm_forceatoms, i);
    const int  type      = angleData[0];
    const int  ai        = angleData[1];
    const int  aj        = angleData[2];
    const int  ak        = angleData[3];

    const auto& params = gm_forceparams[type].angle_class2;

    DeviceFloat3 del1;
    DeviceFloat3 del2;
    const int    t1   = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], del1);
    const int    t2   = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ak], gm_xq[aj], del2);
    const float  rsq1 = gmxDeviceNorm2(del1);
    const float  rsq2 = gmxDeviceNorm2(del2);
    const float  invR1 = gmxDeviceRSqrt(rsq1);
    const float  invR2 = gmxDeviceRSqrt(rsq2);
    const float  invRsq1 = invR1 * invR1;
    const float  invRsq2 = invR2 * invR2;
    const float  r1   = rsq1 * invR1;
    const float  r2   = rsq2 * invR2;
    const float  invR12 = invR1 * invR2;

    float c = gmxDeviceInternalProd(del1, del2) * invR12;
    c       = clampToRange(c, -1.0F, 1.0F);

    const float invSinTheta = gmxDeviceRSqrt(maxFloat(1.0F - c * c, c_small * c_small));

    const float dtheta  = gmxDeviceAcos(c) - params.theta0;
    const float dtheta2 = dtheta * dtheta;

    const bool keepAll = !pcffClass2DebugModeEnabled(pcffClass2DebugMode);
    const bool keepAngleMain =
            keepAll || pcffClass2DebugMode == static_cast<int>(PcffClass2DebugMode::AngleClass2MainOnly);
    const bool keepBondBond =
            params.bb_k != 0.0F
            && (keepAll
                || pcffClass2DebugMode == static_cast<int>(PcffClass2DebugMode::AngleClass2BondBondOnly));
    const bool keepBondAngle1 =
            params.ba_k1 != 0.0F
            && (keepAll
                || pcffClass2DebugMode == static_cast<int>(PcffClass2DebugMode::AngleClass2BondAngle1Only));
    const bool keepBondAngle2 =
            params.ba_k2 != 0.0F
            && (keepAll
                || pcffClass2DebugMode == static_cast<int>(PcffClass2DebugMode::AngleClass2BondAngle2Only));

    DeviceFloat3 f_i = { 0.0F, 0.0F, 0.0F };
    DeviceFloat3 f_k = { 0.0F, 0.0F, 0.0F };

    if (keepAngleMain)
    {
        const float dtheta3 = dtheta2 * dtheta;
        const float deAngle =
                2.0F * params.k2 * dtheta + 3.0F * params.k3 * dtheta2 + 4.0F * params.k4 * dtheta3;
        const float a   = -deAngle * invSinTheta;
        const float a11 = a * c * invRsq1;
        const float a12 = -a * invR12;
        const float a22 = a * c * invRsq2;
        f_i += a11 * del1 + a12 * del2;
        f_k += a22 * del2 + a12 * del1;

        if constexpr (calcEner)
        {
            const float dtheta4 = dtheta3 * dtheta;
            *vtot_loc += params.k2 * dtheta2 + params.k3 * dtheta3 + params.k4 * dtheta4;
        }
    }

    if (keepBondBond)
    {
        const float dr1 = r1 - params.bb_r1;
        const float dr2 = r2 - params.bb_r2;
        const float tk1 = params.bb_k * dr1;
        const float tk2 = params.bb_k * dr2;
        f_i -= (tk2 * invR1) * del1;
        f_k -= (tk1 * invR2) * del2;

        if constexpr (calcEner)
        {
            *vtot_loc += params.bb_k * dr1 * dr2;
        }
    }

    if (keepBondAngle1 || keepBondAngle2)
    {
        const float dr1 = r1 - params.ba_r1;
        const float dr2 = r2 - params.ba_r2;

        const float aa1 = invSinTheta * dr1 * params.ba_k1;
        const float aa2 = invSinTheta * dr2 * params.ba_k2;

        float aa11 = aa1 * c * invRsq1;
        float aa12 = -aa1 * invR12;
        float aa21 = aa2 * c * invRsq1;
        float aa22 = -aa2 * invR12;

        const DeviceFloat3 v1 = aa11 * del1 + aa12 * del2;
        const DeviceFloat3 v2 = aa21 * del1 + aa22 * del2;

        aa11 = aa1 * c * invRsq2;
        aa21 = aa2 * c * invRsq2;

        const DeviceFloat3 v3 = aa11 * del2 + aa12 * del1;
        const DeviceFloat3 v4 = aa21 * del2 + aa22 * del1;

        const float b1 = params.ba_k1 * dtheta * invR1;
        const float b2 = params.ba_k2 * dtheta * invR2;

        if (keepBondAngle1)
        {
            f_i -= v1 + b1 * del1;
            f_k -= v3;

            if constexpr (calcEner)
            {
                *vtot_loc += params.ba_k1 * dr1 * dtheta;
            }
        }

        if (keepBondAngle2)
        {
            f_i -= v2;
            f_k -= v4 + b2 * del2;

            if constexpr (calcEner)
            {
                *vtot_loc += params.ba_k2 * dr2 * dtheta;
            }
        }
    }

    const DeviceFloat3 f_j = -f_i - f_k;

    staggeredAtomicAddForce(gm_f, f_i, ai, localId);
    staggeredAtomicAddForce(gm_f, f_j, aj, localId);
    staggeredAtomicAddForce(gm_f, f_k, ak, localId);

    if constexpr (calcVir)
    {
        atomicFetchAddLocal(sm_fShiftLoc, t1, f_i);
        atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, f_j);
        atomicFetchAddLocal(sm_fShiftLoc, t2, f_k);
    }
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void
urey_bradley_gpu(const int                                 i,
                 DevicePrivatePtr<float>                   vtot_loc,
                 const DeviceGlobalPtr<const t_iatom>      gm_forceatoms,
                 const DeviceGlobalPtr<const t_iparams>    gm_forceparams,
                 const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                 DeviceGlobalPtr<DeviceFloat3>             gm_f,
                 DeviceLocalPtr<DeviceFloat3>              sm_fShiftLoc,
                 const PbcAiuc&                            pbcAiuc,
                 const int                                 localId)
{
    DeviceInt4 ubData = loadInt4(gm_forceatoms, i);
    const int  type   = ubData[0];
    const int  ai     = ubData[1];
    const int  aj     = ubData[2];
    const int  ak     = ubData[3];

    const float th0A = gm_forceparams[type].u_b.thetaA * c_deg2RadF;
    const float kthA = gm_forceparams[type].u_b.kthetaA;
    const float r13A = gm_forceparams[type].u_b.r13A;
    const float kUBA = gm_forceparams[type].u_b.kUBA;

    DeviceFloat3 r_ij;
    DeviceFloat3 r_kj;
    float        cos_theta;
    int          t1;
    int          t2;
    float        theta = bond_angle_gpu<calcVir>(
            gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj, &cos_theta, &t1, &t2);

    float va;
    float dVdt;
    harmonic_gpu<calcEner>(kthA, th0A, theta, &va, &dVdt);

    if (calcEner)
    {
        *vtot_loc += va;
    }

    DeviceFloat3 r_ik;
    int          ki = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[ak], r_ik);

    float dr2 = gmxDeviceNorm2(r_ik);
    float dr  = dr2 * gmxDeviceRSqrt(dr2);

    float vbond;
    float fbond;
    harmonic_gpu<calcEner>(kUBA, r13A, dr, &vbond, &fbond);

    float cos_theta2 = cos_theta * cos_theta;

    DeviceFloat3 f_i = { 0.0F, 0.0F, 0.0F };
    DeviceFloat3 f_j = { 0.0F, 0.0F, 0.0F };
    DeviceFloat3 f_k = { 0.0F, 0.0F, 0.0F };

    if (cos_theta2 < 1.0F)
    {
        float st  = dVdt * gmxDeviceRSqrt(1.0F - cos_theta2);
        float sth = st * cos_theta;

        float nrkj2 = gmxDeviceNorm2(r_kj);
        float nrij2 = gmxDeviceNorm2(r_ij);

        float cik = st * gmxDeviceRSqrt(nrkj2 * nrij2);
        float cii = sth / nrij2;
        float ckk = sth / nrkj2;

        f_i = cii * r_ij - cik * r_kj;
        f_k = ckk * r_kj - cik * r_ij;
        f_j = -f_i - f_k;

        if constexpr (calcVir)
        {
            atomicFetchAddLocal(sm_fShiftLoc, t1, f_i);
            atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, f_j);
            atomicFetchAddLocal(sm_fShiftLoc, t2, f_k);
        }
    }

    /* Time for the bond calculations */
    if (dr2 != 0.0F)
    {
        if constexpr (calcEner)
        {
            *vtot_loc += vbond;
        }

        fbond *= gmxDeviceRSqrt(dr2);
        DeviceFloat3 fik = fbond * r_ik;
        f_i += fik;
        f_k -= fik;


        if constexpr (calcVir)
        {
            if (ki != gmx::c_centralShiftIndex)
            {
                atomicFetchAddLocal(sm_fShiftLoc, ki, fik);
                atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, -fik);
            }
        }
    }

    if ((cos_theta2 < 1.0F) || (dr2 != 0.0F))
    {
        staggeredAtomicAddForce(gm_f, f_i, ai, localId);
        staggeredAtomicAddForce(gm_f, f_k, ak, localId);
    }

    if (cos_theta2 < 1.0F)
    {
        staggeredAtomicAddForce(gm_f, f_j, aj, localId);
    }
}

template<bool returnShift, typename T>
GMX_DEVICE_FUNC_ATTRIBUTE static inline float dih_angle_gpu(const T                        xi,
                                                            const T                        xj,
                                                            const T                        xk,
                                                            const T                        xl,
                                                            const PbcAiuc&                 pbcAiuc,
                                                            DevicePrivatePtr<DeviceFloat3> r_ij,
                                                            DevicePrivatePtr<DeviceFloat3> r_kj,
                                                            DevicePrivatePtr<DeviceFloat3> r_kl,
                                                            DevicePrivatePtr<DeviceFloat3> m,
                                                            DevicePrivatePtr<DeviceFloat3> n,
                                                            DevicePrivatePtr<int>          t1,
                                                            DevicePrivatePtr<int>          t2,
                                                            DevicePrivatePtr<int>          t3)
{
    *t1 = pbcDxAiucGpu<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiucGpu<returnShift>(pbcAiuc, xk, xj, *r_kj);
    *t3 = pbcDxAiucGpu<returnShift>(pbcAiuc, xk, xl, *r_kl);

    *m         = gmxDeviceCrossProd(*r_ij, *r_kj);
    *n         = gmxDeviceCrossProd(*r_kj, *r_kl);
    float phi  = gmxDeviceAngle(*m, *n);
    float ipr  = gmxDeviceInternalProd(*r_ij, *n);
    float sign = (ipr < 0.0F) ? -1.0F : 1.0F;
    phi        = sign * phi;

    return phi;
}

template<bool returnShift, typename T>
GMX_DEVICE_FUNC_ATTRIBUTE static inline float dih_angle_gpu_sincos(const T        xi,
                                                                   const T        xj,
                                                                   const T        xk,
                                                                   const T        xl,
                                                                   const PbcAiuc& pbcAiuc,
                                                                   DevicePrivatePtr<DeviceFloat3> r_ij,
                                                                   DevicePrivatePtr<DeviceFloat3> r_kj,
                                                                   DevicePrivatePtr<DeviceFloat3> r_kl,
                                                                   DevicePrivatePtr<DeviceFloat3> m,
                                                                   DevicePrivatePtr<DeviceFloat3> n,
                                                                   DevicePrivatePtr<int>   t1,
                                                                   DevicePrivatePtr<int>   t2,
                                                                   DevicePrivatePtr<float> cosval)
{
    *t1 = pbcDxAiucGpu<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiucGpu<returnShift>(pbcAiuc, xk, xj, *r_kj);
    pbcDxAiucGpu<returnShift>(pbcAiuc, xk, xl, *r_kl);

    *m = gmxDeviceCrossProd(*r_ij, *r_kj);
    *n = gmxDeviceCrossProd(*r_kj, *r_kl);

    DeviceFloat3 w = gmxDeviceCrossProd(*m, *n);

    float wlen = gmxDeviceSqrt(gmxDeviceNorm2(w));
    float s    = gmxDeviceInternalProd(*m, *n);

    float mLenSq = gmxDeviceNorm2(*m);
    float nLenSq = gmxDeviceNorm2(*n);
    float mnInv  = gmxDeviceRSqrt(mLenSq * nLenSq);

    *cosval      = s * mnInv;
    float sinval = wlen * mnInv;

    float ipr  = gmxDeviceInternalProd(*r_ij, *n);
    float sign = (ipr < 0.0F) ? -1.0F : 1.0F;

    return sign * sinval;
}

GMX_DEVICE_FUNC_ATTRIBUTE
static inline void dopdihs_gpu(const float             cpA,
                               const float             phiA,
                               const int               mult,
                               const float             phi,
                               DevicePrivatePtr<float> v,
                               DevicePrivatePtr<float> f)
{
    float mdphi = mult * phi - phiA * c_deg2RadF;
    float sdphi = gmxDeviceSin(mdphi);
    *v          = cpA * (1.0F + gmxDeviceCos(mdphi));
    *f          = -cpA * mult * sdphi;
}

template<bool calcVir>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void do_dih_fup_gpu(const int                     i,
                                                            const int                     j,
                                                            const int                     k,
                                                            const int                     l,
                                                            const float                   ddphi,
                                                            const DeviceFloat3            r_ij,
                                                            const DeviceFloat3            r_kj,
                                                            const DeviceFloat3            r_kl,
                                                            const DeviceFloat3            m,
                                                            const DeviceFloat3            n,
                                                            DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                            DeviceLocalPtr<DeviceFloat3> sm_fShiftLoc,
                                                            const PbcAiuc& pbcAiuc,
                                                            const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                            const int t1,
                                                            const int t2,
                                                            const int localId)
{
    float iprm  = gmxDeviceNorm2(m);
    float iprn  = gmxDeviceNorm2(n);
    float nrkj2 = gmxDeviceNorm2(r_kj);
    float toler = nrkj2 * GMX_REAL_EPS;
    if ((iprm > toler) && (iprn > toler))
    {
        float        nrkj_1 = gmxDeviceRSqrt(nrkj2);
        float        nrkj_2 = nrkj_1 * nrkj_1;
        float        nrkj   = nrkj2 * nrkj_1;
        float        a      = -ddphi * nrkj / iprm;
        DeviceFloat3 f_i    = a * m;
        float        b      = ddphi * nrkj / iprn;
        DeviceFloat3 f_l    = b * n;
        float        p      = gmxDeviceInternalProd(r_ij, r_kj);
        p *= nrkj_2;
        float q = gmxDeviceInternalProd(r_kl, r_kj);
        q *= nrkj_2;
        DeviceFloat3 uvec = p * f_i;
        DeviceFloat3 vvec = q * f_l;
        DeviceFloat3 svec = uvec - vvec;
        DeviceFloat3 f_j  = f_i - svec;
        DeviceFloat3 f_k  = f_l + svec;

        staggeredAtomicAddForce(gm_f, f_i, i, localId);
        staggeredAtomicAddForce(gm_f, -f_j, j, localId);
        staggeredAtomicAddForce(gm_f, -f_k, k, localId);
        staggeredAtomicAddForce(gm_f, f_l, l, localId);

        if constexpr (calcVir)
        {
            DeviceFloat3 dx_jl;
            int          t3 = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[l], gm_xq[j], dx_jl);

            atomicFetchAddLocal(sm_fShiftLoc, t1, f_i);
            atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, -f_j);
            atomicFetchAddLocal(sm_fShiftLoc, t2, -f_k);
            atomicFetchAddLocal(sm_fShiftLoc, t3, f_l);
        }
    }
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void pdihs_gpu(const int               i,
                                                       DevicePrivatePtr<float> vtot_loc,
                                                       const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                       const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                       const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                       DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                       DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                       const PbcAiuc&                pbcAiuc,
                                                       const int                     localId)
{
    int type = gm_forceatoms[5 * i];
    int ai   = gm_forceatoms[5 * i + 1];
    int aj   = gm_forceatoms[5 * i + 2];
    int ak   = gm_forceatoms[5 * i + 3];
    int al   = gm_forceatoms[5 * i + 4];

    DeviceFloat3 r_ij;
    DeviceFloat3 r_kj;
    DeviceFloat3 r_kl;
    DeviceFloat3 m;
    DeviceFloat3 n;
    int          t1;
    int          t2;
    int          t3;
    float        phi = dih_angle_gpu<calcVir>(
            gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

    float vpd;
    float ddphi;
    dopdihs_gpu(gm_forceparams[type].pdihs.cpA,
                gm_forceparams[type].pdihs.phiA,
                gm_forceparams[type].pdihs.mult,
                phi,
                &vpd,
                &ddphi);

    if constexpr (calcEner)
    {
        *vtot_loc += vpd;
    }

    do_dih_fup_gpu<calcVir>(
            ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, localId);
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void rbdihs_gpu(const int               i,
                                                        DevicePrivatePtr<float> vtot_loc,
                                                        const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                        const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                        const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                        DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                        DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                        const PbcAiuc&                pbcAiuc,
                                                        const int                     localId)
{
    constexpr float c0 = 0.0F, c1 = 1.0F, c2 = 2.0F, c3 = 3.0F, c4 = 4.0F, c5 = 5.0F;

    {
        int type = gm_forceatoms[5 * i];
        int ai   = gm_forceatoms[5 * i + 1];
        int aj   = gm_forceatoms[5 * i + 2];
        int ak   = gm_forceatoms[5 * i + 3];
        int al   = gm_forceatoms[5 * i + 4];

        DeviceFloat3 r_ij;
        DeviceFloat3 r_kj;
        DeviceFloat3 r_kl;
        DeviceFloat3 m;
        DeviceFloat3 n;
        int          t1;
        int          t2;

        // Changing the sign of sin and cos to convert to polymer convention
        float       cos_phi;
        const float negative_sin_phi = dih_angle_gpu_sincos<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &cos_phi);
        cos_phi *= -1;

        float parm[NR_RBDIHS];
#    pragma unroll NR_RBDIHS
        for (int j = 0; j < NR_RBDIHS; j++)
        {
            parm[j] = gm_forceparams[type].rbdihs.rbcA[j];
        }
        /* Calculate cosine powers */
        /* Calculate the energy */
        /* Calculate the derivative */
        float v      = parm[0];
        float ddphi  = c0;
        float cosfac = c1;

        float rbp = parm[1];
        ddphi += rbp * cosfac;
        cosfac *= cos_phi;
        if constexpr (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[2];
        ddphi += c2 * rbp * cosfac;
        cosfac *= cos_phi;
        if constexpr (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[3];
        ddphi += c3 * rbp * cosfac;
        cosfac *= cos_phi;
        if constexpr (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[4];
        ddphi += c4 * rbp * cosfac;
        cosfac *= cos_phi;
        if constexpr (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[5];
        ddphi += c5 * rbp * cosfac;
        cosfac *= cos_phi;
        if constexpr (calcEner)
        {
            v += cosfac * rbp;
        }

        ddphi = ddphi * negative_sin_phi;

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, localId);
        if constexpr (calcEner)
        {
            *vtot_loc += v;
        }
    }
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void dihedral_class2_gpu(const int               i,
                                                                 DevicePrivatePtr<float> vtot_loc,
                                                                 const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                                 const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                                 const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                                 DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                                 DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                                 const PbcAiuc&                pbcAiuc,
                                                                 const int                     localId)
{
    constexpr float c_small = 1.0e-7F;

    const int type = gm_forceatoms[5 * i];
    const int ai   = gm_forceatoms[5 * i + 1];
    const int aj   = gm_forceatoms[5 * i + 2];
    const int ak   = gm_forceatoms[5 * i + 3];
    const int al   = gm_forceatoms[5 * i + 4];

    const auto& params = gm_forceparams[type].dihedral_class2;
    const bool  hasPrimaryTorsion2 = params.k2 != 0.0F;
    const bool  hasPrimaryTorsion3 = params.k3 != 0.0F;
    const bool  hasMiddleBondTorsion =
            params.mbt_f1 != 0.0F || params.mbt_f2 != 0.0F || params.mbt_f3 != 0.0F;
    const bool hasEndBondTorsion1 =
            params.ebt_f1_1 != 0.0F || params.ebt_f2_1 != 0.0F || params.ebt_f3_1 != 0.0F;
    const bool hasEndBondTorsion2 =
            params.ebt_f1_2 != 0.0F || params.ebt_f2_2 != 0.0F || params.ebt_f3_2 != 0.0F;
    const bool hasAngleTorsion1 =
            params.at_f1_1 != 0.0F || params.at_f2_1 != 0.0F || params.at_f3_1 != 0.0F;
    const bool hasAngleTorsion2 =
            params.at_f1_2 != 0.0F || params.at_f2_2 != 0.0F || params.at_f3_2 != 0.0F;
    const bool hasAngleAngleTorsion = params.aat_k != 0.0F;
    const bool needsBondTorsionDerivatives =
            hasMiddleBondTorsion || hasEndBondTorsion1 || hasEndBondTorsion2;
    const bool needsAngleTorsionDerivatives =
            hasAngleTorsion1 || hasAngleTorsion2 || hasAngleAngleTorsion;
    const bool needsPhiMultiples = hasPrimaryTorsion2 || hasPrimaryTorsion3
                                   || needsBondTorsionDerivatives
                                   || needsAngleTorsionDerivatives;

    DeviceFloat3 vb1;
    DeviceFloat3 vb2;
    DeviceFloat3 vb3;
    const int    t1 = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], vb1);
    const int    t2 = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ak], gm_xq[aj], vb2);
    pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[al], gm_xq[ak], vb3);

    const float vb1x = component(vb1, XX);
    const float vb1y = component(vb1, YY);
    const float vb1z = component(vb1, ZZ);
    const float vb2x = component(vb2, XX);
    const float vb2y = component(vb2, YY);
    const float vb2z = component(vb2, ZZ);
    const float vb3x = component(vb3, XX);
    const float vb3y = component(vb3, YY);
    const float vb3z = component(vb3, ZZ);
    const float vb2xm = -vb2x;
    const float vb2ym = -vb2y;
    const float vb2zm = -vb2z;

    const float r1mag2 = gmxDeviceInternalProd(vb1, vb1);
    const float r2mag2 = gmxDeviceInternalProd(vb2, vb2);
    const float r3mag2 = gmxDeviceInternalProd(vb3, vb3);
    const float rb1    = gmxDeviceRSqrt(r1mag2);
    const float rb2    = gmxDeviceRSqrt(r2mag2);
    const float rb3    = gmxDeviceRSqrt(r3mag2);
    const float sb1    = rb1 * rb1;
    const float sb2    = rb2 * rb2;
    const float sb3    = rb3 * rb3;
    const float r1     = r1mag2 * rb1;
    const float r2     = r2mag2 * rb2;
    const float r3     = r3mag2 * rb3;

    float c0      = gmxDeviceInternalProd(vb1, vb3) * rb1 * rb3;
    const float r12c1 = rb1 * rb2;
    const float r12c2 = rb2 * rb3;
    float costh12 = gmxDeviceInternalProd(vb1, vb2) * r12c1;
    float costh13 = c0;
    float costh23 = (vb2xm * vb3x + vb2ym * vb3y + vb2zm * vb3z) * r12c2;

    costh12 = clampToRange(costh12, -1.0F, 1.0F);
    costh13 = clampToRange(costh13, -1.0F, 1.0F);
    costh23 = clampToRange(costh23, -1.0F, 1.0F);
    c0      = costh13;

    float sin2 = 1.0F - costh12 * costh12;
    float sc1  = gmxDeviceRSqrt(maxFloat(sin2, c_small * c_small));

    sin2      = 1.0F - costh23 * costh23;
    float sc2 = gmxDeviceRSqrt(maxFloat(sin2, c_small * c_small));

    const float s1  = sc1 * sc1;
    const float s2  = sc2 * sc2;
    const float s12 = sc1 * sc2;
    float c = clampToRange((c0 + costh12 * costh23) * s12, -1.0F, 1.0F);

    const float cosphi = c;

    const float sinPhiSq = maxFloat(1.0F - c * c, 0.0F);
    const float absSinPhi = gmxDeviceSqrt(sinPhiSq);
    float       sinphi    = maxFloat(absSinPhi, c_small);
    float       sinphiForMultiples = absSinPhi;
    float       invSinPhi = gmxDeviceRSqrt(maxFloat(sinPhiSq, c_small * c_small));

    const float n123x      = vb1y * vb2z - vb1z * vb2y;
    const float n123y      = vb1z * vb2x - vb1x * vb2z;
    const float n123z      = vb1x * vb2y - vb1y * vb2x;
    const float n123DotVb3 = n123x * vb3x + n123y * vb3y + n123z * vb3z;
    if (n123DotVb3 > 0.0F)
    {
        sinphi = -sinphi;
        sinphiForMultiples = -sinphiForMultiples;
        invSinPhi = -invSinPhi;
    }

    const float a11 = -c * sb1 * s1;
    const float a22 = sb2 * (2.0F * costh13 * s12 - c * (s1 + s2));
    const float a33 = -c * sb3 * s2;
    const float a12 = r12c1 * (costh12 * c * s1 + costh23 * s12);
    const float a13 = rb1 * rb3 * s12;
    const float a23 = r12c2 * (-costh23 * c * s2 - costh12 * s12);

    const float sx1  = a11 * vb1x + a12 * vb2x + a13 * vb3x;
    const float sx2  = a12 * vb1x + a22 * vb2x + a23 * vb3x;
    const float sx12 = a13 * vb1x + a23 * vb2x + a33 * vb3x;
    const float sy1  = a11 * vb1y + a12 * vb2y + a13 * vb3y;
    const float sy2  = a12 * vb1y + a22 * vb2y + a23 * vb3y;
    const float sy12 = a13 * vb1y + a23 * vb2y + a33 * vb3y;
    const float sz1  = a11 * vb1z + a12 * vb2z + a13 * vb3z;
    const float sz2  = a12 * vb1z + a22 * vb2z + a23 * vb3z;
    const float sz12 = a13 * vb1z + a23 * vb2z + a33 * vb3z;

    float dcosphidr[4][DIM];
    float dphidr[4][DIM];
    dcosphidr[0][XX] = -sx1;
    dcosphidr[0][YY] = -sy1;
    dcosphidr[0][ZZ] = -sz1;
    dcosphidr[1][XX] = sx2 + sx1;
    dcosphidr[1][YY] = sy2 + sy1;
    dcosphidr[1][ZZ] = sz2 + sz1;
    dcosphidr[2][XX] = sx12 - sx2;
    dcosphidr[2][YY] = sy12 - sy2;
    dcosphidr[2][ZZ] = sz12 - sz2;
    dcosphidr[3][XX] = -sx12;
    dcosphidr[3][YY] = -sy12;
    dcosphidr[3][ZZ] = -sz12;

    for (int atom = 0; atom < 4; atom++)
    {
        for (int dim = 0; dim < DIM; dim++)
        {
            dphidr[atom][dim] = -dcosphidr[atom][dim] * invSinPhi;
        }
    }

    DeviceFloat3 fabcd[4] = {};

    const float cos2phi = needsPhiMultiples ? 2.0F * cosphi * cosphi - 1.0F : 0.0F;
    const float cos3phi = needsPhiMultiples ? cosphi * (4.0F * cosphi * cosphi - 3.0F) : 0.0F;
    const float sin2phi = needsPhiMultiples ? 2.0F * sinphiForMultiples * cosphi : 0.0F;
    const float sin3phi =
            needsPhiMultiples ? sinphiForMultiples * (3.0F - 4.0F * sinphiForMultiples * sinphiForMultiples) : 0.0F;
    float       dihedralEnergy = 0.0F;
    float       deDihedral     = 0.0F;
    if (params.k1 != 0.0F)
    {
        float cosPhi1;
        float sinPhi1;
        pcffClass2PhaseSinCosGpu(params.phi1, &cosPhi1, &sinPhi1);
        const float cosD1 = cosphi * cosPhi1 + sinphiForMultiples * sinPhi1;
        const float sinD1 = sinphiForMultiples * cosPhi1 - cosphi * sinPhi1;
        dihedralEnergy += params.k1 * (1.0F - cosD1);
        deDihedral += params.k1 * sinD1;
    }
    if (params.k2 != 0.0F)
    {
        float cosPhi2;
        float sinPhi2;
        pcffClass2PhaseSinCosGpu(params.phi2, &cosPhi2, &sinPhi2);
        const float cosD2 = cos2phi * cosPhi2 + sin2phi * sinPhi2;
        const float sinD2 = sin2phi * cosPhi2 - cos2phi * sinPhi2;
        dihedralEnergy += params.k2 * (1.0F - cosD2);
        deDihedral += 2.0F * params.k2 * sinD2;
    }
    if (params.k3 != 0.0F)
    {
        float cosPhi3;
        float sinPhi3;
        pcffClass2PhaseSinCosGpu(params.phi3, &cosPhi3, &sinPhi3);
        const float cosD3 = cos3phi * cosPhi3 + sin3phi * sinPhi3;
        const float sinD3 = sin3phi * cosPhi3 - cos3phi * sinPhi3;
        dihedralEnergy += params.k3 * (1.0F - cosD3);
        deDihedral += 3.0F * params.k3 * sinD3;
    }
    if constexpr (calcEner)
    {
        *vtot_loc += dihedralEnergy;
    }

    for (int atom = 0; atom < 4; atom++)
    {
        for (int dim = 0; dim < DIM; dim++)
        {
            componentRef(fabcd[atom], dim) = deDihedral * dphidr[atom][dim];
        }
    }

    float dbonddr[3][4][DIM] = {};
    if (needsBondTorsionDerivatives)
    {
        dbonddr[0][0][XX] = vb1x * rb1;
        dbonddr[0][0][YY] = vb1y * rb1;
        dbonddr[0][0][ZZ] = vb1z * rb1;
        dbonddr[0][1][XX] = -vb1x * rb1;
        dbonddr[0][1][YY] = -vb1y * rb1;
        dbonddr[0][1][ZZ] = -vb1z * rb1;

        dbonddr[1][1][XX] = vb2x * rb2;
        dbonddr[1][1][YY] = vb2y * rb2;
        dbonddr[1][1][ZZ] = vb2z * rb2;
        dbonddr[1][2][XX] = -vb2x * rb2;
        dbonddr[1][2][YY] = -vb2y * rb2;
        dbonddr[1][2][ZZ] = -vb2z * rb2;

        dbonddr[2][2][XX] = vb3x * rb3;
        dbonddr[2][2][YY] = vb3y * rb3;
        dbonddr[2][2][ZZ] = vb3z * rb3;
        dbonddr[2][3][XX] = -vb3x * rb3;
        dbonddr[2][3][YY] = -vb3y * rb3;
        dbonddr[2][3][ZZ] = -vb3z * rb3;
    }

    float dthetadr[2][4][DIM] = {};
    if (needsAngleTorsionDerivatives)
    {
        const float t1l = costh12 * sb1;
        const float t2l = costh23 * sb2;
        const float t3l = costh12 * sb2;
        const float t4l = costh23 * sb3;

        dthetadr[0][0][XX] = sc1 * (t1l * vb1x - vb2x * r12c1);
        dthetadr[0][0][YY] = sc1 * (t1l * vb1y - vb2y * r12c1);
        dthetadr[0][0][ZZ] = sc1 * (t1l * vb1z - vb2z * r12c1);
        dthetadr[0][1][XX] =
                sc1 * ((-t1l * vb1x) + (vb2x * r12c1) + (-t3l * vb2x) + (vb1x * r12c1));
        dthetadr[0][1][YY] =
                sc1 * ((-t1l * vb1y) + (vb2y * r12c1) + (-t3l * vb2y) + (vb1y * r12c1));
        dthetadr[0][1][ZZ] =
                sc1 * ((-t1l * vb1z) + (vb2z * r12c1) + (-t3l * vb2z) + (vb1z * r12c1));
        dthetadr[0][2][XX] = sc1 * (t3l * vb2x - vb1x * r12c1);
        dthetadr[0][2][YY] = sc1 * (t3l * vb2y - vb1y * r12c1);
        dthetadr[0][2][ZZ] = sc1 * (t3l * vb2z - vb1z * r12c1);

        dthetadr[1][1][XX] = sc2 * (t2l * vb2x + vb3x * r12c2);
        dthetadr[1][1][YY] = sc2 * (t2l * vb2y + vb3y * r12c2);
        dthetadr[1][1][ZZ] = sc2 * (t2l * vb2z + vb3z * r12c2);
        dthetadr[1][2][XX] =
                sc2 * ((-t2l * vb2x) - (vb3x * r12c2) + (t4l * vb3x) + (vb2x * r12c2));
        dthetadr[1][2][YY] =
                sc2 * ((-t2l * vb2y) - (vb3y * r12c2) + (t4l * vb3y) + (vb2y * r12c2));
        dthetadr[1][2][ZZ] =
                sc2 * ((-t2l * vb2z) - (vb3z * r12c2) + (t4l * vb3z) + (vb2z * r12c2));
        dthetadr[1][3][XX] = -sc2 * (t4l * vb3x + vb2x * r12c2);
        dthetadr[1][3][YY] = -sc2 * (t4l * vb3y + vb2y * r12c2);
        dthetadr[1][3][ZZ] = -sc2 * (t4l * vb3z + vb2z * r12c2);
    }

    const float theta12 = needsAngleTorsionDerivatives ? gmxDeviceAcos(costh12) : 0.0F;
    const float theta23 = needsAngleTorsionDerivatives ? gmxDeviceAcos(costh23) : 0.0F;
    float bt1    = 0.0F;
    float bt2    = 0.0F;
    float bt3    = 0.0F;
    float sumbte = 0.0F;
    float sumbtf = 0.0F;
    float db     = 0.0F;

    if (hasMiddleBondTorsion)
    {
        bt1    = params.mbt_f1 * cosphi;
        bt2    = params.mbt_f2 * cos2phi;
        bt3    = params.mbt_f3 * cos3phi;
        sumbte = bt1 + bt2 + bt3;
        db     = r2 - params.mbt_r0;
        if constexpr (calcEner)
        {
            *vtot_loc += db * sumbte;
        }

        bt1    = -params.mbt_f1 * sinphi;
        bt2    = -2.0F * params.mbt_f2 * sin2phi;
        bt3    = -3.0F * params.mbt_f3 * sin3phi;
        sumbtf = bt1 + bt2 + bt3;
        for (int atom = 0; atom < 4; atom++)
        {
            for (int dim = 0; dim < DIM; dim++)
            {
                componentRef(fabcd[atom], dim) +=
                        db * sumbtf * dphidr[atom][dim] + sumbte * dbonddr[1][atom][dim];
            }
        }
    }

    if (hasEndBondTorsion1)
    {
        bt1    = params.ebt_f1_1 * cosphi;
        bt2    = params.ebt_f2_1 * cos2phi;
        bt3    = params.ebt_f3_1 * cos3phi;
        sumbte = bt1 + bt2 + bt3;
        db     = r1 - params.ebt_r0_1;
        if constexpr (calcEner)
        {
            *vtot_loc += db * sumbte;
        }

        bt1    = params.ebt_f1_1 * sinphi;
        bt2    = 2.0F * params.ebt_f2_1 * sin2phi;
        bt3    = 3.0F * params.ebt_f3_1 * sin3phi;
        sumbtf = bt1 + bt2 + bt3;
        for (int atom = 0; atom < 4; atom++)
        {
            for (int dim = 0; dim < DIM; dim++)
            {
                componentRef(fabcd[atom], dim) -=
                        db * sumbtf * dphidr[atom][dim] + sumbte * dbonddr[0][atom][dim];
            }
        }
    }

    if (hasEndBondTorsion2)
    {
        bt1    = params.ebt_f1_2 * cosphi;
        bt2    = params.ebt_f2_2 * cos2phi;
        bt3    = params.ebt_f3_2 * cos3phi;
        sumbte = bt1 + bt2 + bt3;
        db     = r3 - params.ebt_r0_2;
        if constexpr (calcEner)
        {
            *vtot_loc += db * sumbte;
        }

        bt1    = -params.ebt_f1_2 * sinphi;
        bt2    = -2.0F * params.ebt_f2_2 * sin2phi;
        bt3    = -3.0F * params.ebt_f3_2 * sin3phi;
        sumbtf = bt1 + bt2 + bt3;
        for (int atom = 0; atom < 4; atom++)
        {
            for (int dim = 0; dim < DIM; dim++)
            {
                componentRef(fabcd[atom], dim) +=
                        db * sumbtf * dphidr[atom][dim] + sumbte * dbonddr[2][atom][dim];
            }
        }
    }

    if (hasAngleTorsion1)
    {
        float at1 = params.at_f1_1 * cosphi;
        float at2 = params.at_f2_1 * cos2phi;
        float at3 = params.at_f3_1 * cos3phi;
        sumbte    = at1 + at2 + at3;
        float da  = theta12 - params.at_theta0_1;
        if constexpr (calcEner)
        {
            *vtot_loc += da * sumbte;
        }

        bt1    = params.at_f1_1 * sinphi;
        bt2    = 2.0F * params.at_f2_1 * sin2phi;
        bt3    = 3.0F * params.at_f3_1 * sin3phi;
        sumbtf = bt1 + bt2 + bt3;
        for (int atom = 0; atom < 4; atom++)
        {
            for (int dim = 0; dim < DIM; dim++)
            {
                componentRef(fabcd[atom], dim) -=
                        da * sumbtf * dphidr[atom][dim] + sumbte * dthetadr[0][atom][dim];
            }
        }
    }

    if (hasAngleTorsion2)
    {
        float at1 = params.at_f1_2 * cosphi;
        float at2 = params.at_f2_2 * cos2phi;
        float at3 = params.at_f3_2 * cos3phi;
        sumbte    = at1 + at2 + at3;
        float da  = theta23 - params.at_theta0_2;
        if constexpr (calcEner)
        {
            *vtot_loc += da * sumbte;
        }

        bt1    = -params.at_f1_2 * sinphi;
        bt2    = -2.0F * params.at_f2_2 * sin2phi;
        bt3    = -3.0F * params.at_f3_2 * sin3phi;
        sumbtf = bt1 + bt2 + bt3;
        for (int atom = 0; atom < 4; atom++)
        {
            for (int dim = 0; dim < DIM; dim++)
            {
                componentRef(fabcd[atom], dim) +=
                        da * sumbtf * dphidr[atom][dim] + sumbte * dthetadr[1][atom][dim];
            }
        }
    }

    if (hasAngleAngleTorsion)
    {
        const float da1 = theta12 - params.aat_theta0_1;
        const float da2 = theta23 - params.aat_theta0_2;
        if constexpr (calcEner)
        {
            *vtot_loc += params.aat_k * da1 * da2 * cosphi;
        }
        for (int atom = 0; atom < 4; atom++)
        {
            for (int dim = 0; dim < DIM; dim++)
            {
                componentRef(fabcd[atom], dim) -=
                        params.aat_k
                        * (cosphi * (da2 * dthetadr[0][atom][dim] - da1 * dthetadr[1][atom][dim])
                           + sinphi * da1 * da2 * dphidr[atom][dim]);
            }
        }
    }

    if (absFloat(params.bb13t_k) > c_small)
    {
        const float dr1 = r1 - params.bb13t_r10;
        const float dr2 = r3 - params.bb13t_r30;
        const float tk1 = -params.bb13t_k * dr1 * rb3;
        const float tk2 = -params.bb13t_k * dr2 * rb1;

        if constexpr (calcEner)
        {
            *vtot_loc += params.bb13t_k * dr1 * dr2;
        }

        fabcd[0] = fabcd[0] + tk2 * vb1;
        fabcd[1] = fabcd[1] - tk2 * vb1;
        fabcd[2] = fabcd[2] - tk1 * vb3;
        fabcd[3] = fabcd[3] + tk1 * vb3;
    }

    staggeredAtomicAddForce(gm_f, fabcd[0], ai, localId);
    staggeredAtomicAddForce(gm_f, fabcd[1], aj, localId);
    staggeredAtomicAddForce(gm_f, fabcd[2], ak, localId);
    staggeredAtomicAddForce(gm_f, fabcd[3], al, localId);

    if constexpr (calcVir)
    {
        DeviceFloat3 h;
        const int    t3 = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[al], gm_xq[aj], h);
        atomicFetchAddLocal(sm_fShiftLoc, t1, fabcd[0]);
        atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, fabcd[1]);
        atomicFetchAddLocal(sm_fShiftLoc, t2, fabcd[2]);
        atomicFetchAddLocal(sm_fShiftLoc, t3, fabcd[3]);
    }
}

template<bool calcVir, bool calcEner, bool allowImproperChiTerm = true>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void improper_class2_gpu(const int               i,
                                                                 DevicePrivatePtr<float> vtot_loc,
                                                                 const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                                 const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                                 const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                                 DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                                 DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                                 const PbcAiuc&                pbcAiuc,
                                                                 const int                     localId)
{
    constexpr float c_small = 1.0e-7F;

    const int type = gm_forceatoms[5 * i];
    const int ai   = gm_forceatoms[5 * i + 1];
    const int aj   = gm_forceatoms[5 * i + 2];
    const int ak   = gm_forceatoms[5 * i + 3];
    const int al   = gm_forceatoms[5 * i + 4];

    const auto& params = gm_forceparams[type].improper_class2;
    const bool  hasChiTerm = allowImproperChiTerm && params.k0 != 0.0F;
    const bool  hasAngleAngleTerm1 = params.aa_k1 != 0.0F;
    const bool  hasAngleAngleTerm2 = params.aa_k2 != 0.0F;
    const bool  hasAngleAngleTerm3 = params.aa_k3 != 0.0F;
    const bool  hasAngleAngleTerm  = hasAngleAngleTerm1 || hasAngleAngleTerm2 || hasAngleAngleTerm3;
    if (!hasChiTerm && !hasAngleAngleTerm)
    {
        return;
    }

    DeviceFloat3 delrVec[3];
    const int    t1 = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], delrVec[0]);
    const int    t2 = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ak], gm_xq[aj], delrVec[1]);
    const int    t3 = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[al], gm_xq[aj], delrVec[2]);

    float delr[3][DIM] = {};
    for (int interaction = 0; interaction < 3; ++interaction)
    {
        for (int dim = 0; dim < DIM; ++dim)
        {
            delr[interaction][dim] = component(delrVec[interaction], dim);
        }
    }

    float fabcd[4][DIM] = {};

    float rinvmag[3]  = {};
    float rinvmag2[3] = {};
    float rmag2[3]    = {};
    float costheta[3] = {};
    float cossqtheta[3] = {};
    float sinsqtheta[3] = {};
    float invstheta[3] = {};
    if (hasChiTerm || hasAngleAngleTerm)
    {
        for (int interaction = 0; interaction < 3; ++interaction)
        {
            rmag2[interaction]    = pcffClass2DotGpu(delr[interaction], delr[interaction]);
            rinvmag[interaction]  = gmxDeviceRSqrt(rmag2[interaction]);
            rinvmag2[interaction] = rinvmag[interaction] * rinvmag[interaction];
        }

        costheta[0] = pcffClass2DotGpu(delr[0], delr[1]) * rinvmag[0] * rinvmag[1];
        costheta[1] = pcffClass2DotGpu(delr[1], delr[2]) * rinvmag[1] * rinvmag[2];
        costheta[2] = pcffClass2DotGpu(delr[0], delr[2]) * rinvmag[0] * rinvmag[2];

        for (int interaction = 0; interaction < 3; ++interaction)
        {
            costheta[interaction] =
                    clampToRange(costheta[interaction], -1.0F + c_small, 1.0F - c_small);
            cossqtheta[interaction] = costheta[interaction] * costheta[interaction];
            sinsqtheta[interaction] = maxFloat(1.0F - cossqtheta[interaction], c_small);
            invstheta[interaction]  = gmxDeviceRSqrt(sinsqtheta[interaction]);
        }
    }

    if constexpr (allowImproperChiTerm)
    {
        if (hasChiTerm)
        {
            float rABxrCB[DIM], rDBxrAB[DIM], rCBxrDB[DIM];
            float ddelr[3][4]         = {};
            float dr[3][4][DIM]       = {};
            float dinvr[3][4][DIM]    = {};
            float dthetadr[3][4][DIM] = {};
            float dinvsth[3][4][DIM]  = {};
            float dinv3r[4][DIM]      = {};
            float dinvs3r[3][4][DIM]  = {};
            float rCBxdrDB[DIM], drCBxrDB[DIM], rDBxdrAB[DIM], drDBxrAB[DIM];
            float drABxrCB[DIM], rABxdrCB[DIM];
            float dd[DIM];
            float fdot[3][4][DIM]   = {};
            float invs3r[3];
            float dtotalchi[4][DIM] = {};

            pcffClass2CrossGpu(delr[0], delr[1], rABxrCB);
            pcffClass2CrossGpu(delr[2], delr[0], rDBxrAB);
            pcffClass2CrossGpu(delr[1], delr[2], rCBxrDB);

            const float dotCBDBAB = pcffClass2DotGpu(rCBxrDB, delr[0]);
            const float dotDBABCB = pcffClass2DotGpu(rDBxrAB, delr[1]);
            const float dotABCBDB = pcffClass2DotGpu(rABxrCB, delr[2]);

            const float inv3r = rinvmag[0] * rinvmag[1] * rinvmag[2];
            invs3r[0]         = invstheta[1] * inv3r;
            invs3r[1]         = invstheta[2] * inv3r;
            invs3r[2]         = invstheta[0] * inv3r;

            const float sinChiABCD = dotCBDBAB * invs3r[0];
            const float sinChiCBDA = dotDBABCB * invs3r[1];
            const float sinChiDBAC = dotABCBDB * invs3r[2];
            const float chiABCD    = gmxDeviceAsin(sinChiABCD);
            const float chiCBDA    = gmxDeviceAsin(sinChiCBDA);
            const float chiDBAC    = gmxDeviceAsin(sinChiDBAC);
            const float deltachi = (chiABCD + chiCBDA + chiDBAC) / 3.0F - params.chi0;
            const float invCosChiABCD = invCosFromSinArgumentGpu(sinChiABCD, c_small);
            const float invCosChiCBDA = invCosFromSinArgumentGpu(sinChiCBDA, c_small);
            const float invCosChiDBAC = invCosFromSinArgumentGpu(sinChiDBAC, c_small);

            if constexpr (calcEner)
            {
                *vtot_loc += params.k0 * deltachi * deltachi;
            }

            ddelr[0][0] = 1.0F;
            ddelr[0][1] = -1.0F;
            ddelr[1][1] = -1.0F;
            ddelr[1][2] = 1.0F;
            ddelr[2][1] = -1.0F;
            ddelr[2][3] = 1.0F;

            for (int interaction = 0; interaction < 3; ++interaction)
            {
                for (int atom = 0; atom < 4; ++atom)
                {
                    for (int dim = 0; dim < DIM; ++dim)
                    {
                        dr[interaction][atom][dim] =
                                delr[interaction][dim] * ddelr[interaction][atom] * rinvmag[interaction];
                        dinvr[interaction][atom][dim] =
                                -dr[interaction][atom][dim] * rinvmag2[interaction];
                    }
                }
            }

            for (int atom = 0; atom < 4; ++atom)
            {
                for (int dim = 0; dim < DIM; ++dim)
                {
                    dinv3r[atom][dim] = rinvmag[1]
                                                * (rinvmag[2] * dinvr[0][atom][dim]
                                                   + rinvmag[0] * dinvr[2][atom][dim])
                                        + rinvmag[2] * rinvmag[0] * dinvr[1][atom][dim];
                }
            }

            float tt1 = costheta[0] * rinvmag2[0];
            float tt3 = costheta[0] * rinvmag2[1];
            float sc1 = invstheta[0];

        dthetadr[0][0][XX] = sc1 * (tt1 * delr[0][XX] - delr[1][XX] * rinvmag[0] * rinvmag[1]);
        dthetadr[0][0][YY] = sc1 * (tt1 * delr[0][YY] - delr[1][YY] * rinvmag[0] * rinvmag[1]);
        dthetadr[0][0][ZZ] = sc1 * (tt1 * delr[0][ZZ] - delr[1][ZZ] * rinvmag[0] * rinvmag[1]);
        dthetadr[0][1][XX] = -sc1 * (tt1 * delr[0][XX] - delr[1][XX] * rinvmag[0] * rinvmag[1]
                                     + tt3 * delr[1][XX] - delr[0][XX] * rinvmag[0] * rinvmag[1]);
        dthetadr[0][1][YY] = -sc1 * (tt1 * delr[0][YY] - delr[1][YY] * rinvmag[0] * rinvmag[1]
                                     + tt3 * delr[1][YY] - delr[0][YY] * rinvmag[0] * rinvmag[1]);
        dthetadr[0][1][ZZ] = -sc1 * (tt1 * delr[0][ZZ] - delr[1][ZZ] * rinvmag[0] * rinvmag[1]
                                     + tt3 * delr[1][ZZ] - delr[0][ZZ] * rinvmag[0] * rinvmag[1]);
        dthetadr[0][2][XX] = sc1 * (tt3 * delr[1][XX] - delr[0][XX] * rinvmag[0] * rinvmag[1]);
        dthetadr[0][2][YY] = sc1 * (tt3 * delr[1][YY] - delr[0][YY] * rinvmag[0] * rinvmag[1]);
        dthetadr[0][2][ZZ] = sc1 * (tt3 * delr[1][ZZ] - delr[0][ZZ] * rinvmag[0] * rinvmag[1]);

        tt1 = costheta[1] * rinvmag2[1];
        tt3 = costheta[1] * rinvmag2[2];
        sc1 = invstheta[1];

        dthetadr[1][2][XX] = sc1 * (tt1 * delr[1][XX] - delr[2][XX] * rinvmag[1] * rinvmag[2]);
        dthetadr[1][2][YY] = sc1 * (tt1 * delr[1][YY] - delr[2][YY] * rinvmag[1] * rinvmag[2]);
        dthetadr[1][2][ZZ] = sc1 * (tt1 * delr[1][ZZ] - delr[2][ZZ] * rinvmag[1] * rinvmag[2]);
        dthetadr[1][1][XX] = -sc1 * (tt1 * delr[1][XX] - delr[2][XX] * rinvmag[1] * rinvmag[2]
                                     + tt3 * delr[2][XX] - delr[1][XX] * rinvmag[2] * rinvmag[1]);
        dthetadr[1][1][YY] = -sc1 * (tt1 * delr[1][YY] - delr[2][YY] * rinvmag[1] * rinvmag[2]
                                     + tt3 * delr[2][YY] - delr[1][YY] * rinvmag[2] * rinvmag[1]);
        dthetadr[1][1][ZZ] = -sc1 * (tt1 * delr[1][ZZ] - delr[2][ZZ] * rinvmag[1] * rinvmag[2]
                                     + tt3 * delr[2][ZZ] - delr[1][ZZ] * rinvmag[2] * rinvmag[1]);
        dthetadr[1][3][XX] = sc1 * (tt3 * delr[2][XX] - delr[1][XX] * rinvmag[2] * rinvmag[1]);
        dthetadr[1][3][YY] = sc1 * (tt3 * delr[2][YY] - delr[1][YY] * rinvmag[2] * rinvmag[1]);
        dthetadr[1][3][ZZ] = sc1 * (tt3 * delr[2][ZZ] - delr[1][ZZ] * rinvmag[2] * rinvmag[1]);

        tt1 = costheta[2] * rinvmag2[0];
        tt3 = costheta[2] * rinvmag2[2];
        sc1 = invstheta[2];

        dthetadr[2][0][XX] = sc1 * (tt1 * delr[0][XX] - delr[2][XX] * rinvmag[0] * rinvmag[2]);
        dthetadr[2][0][YY] = sc1 * (tt1 * delr[0][YY] - delr[2][YY] * rinvmag[0] * rinvmag[2]);
        dthetadr[2][0][ZZ] = sc1 * (tt1 * delr[0][ZZ] - delr[2][ZZ] * rinvmag[0] * rinvmag[2]);
        dthetadr[2][1][XX] = -sc1 * (tt1 * delr[0][XX] - delr[2][XX] * rinvmag[0] * rinvmag[2]
                                     + tt3 * delr[2][XX] - delr[0][XX] * rinvmag[2] * rinvmag[0]);
        dthetadr[2][1][YY] = -sc1 * (tt1 * delr[0][YY] - delr[2][YY] * rinvmag[0] * rinvmag[2]
                                     + tt3 * delr[2][YY] - delr[0][YY] * rinvmag[2] * rinvmag[0]);
        dthetadr[2][1][ZZ] = -sc1 * (tt1 * delr[0][ZZ] - delr[2][ZZ] * rinvmag[0] * rinvmag[2]
                                     + tt3 * delr[2][ZZ] - delr[0][ZZ] * rinvmag[2] * rinvmag[0]);
        dthetadr[2][3][XX] = sc1 * (tt3 * delr[2][XX] - delr[0][XX] * rinvmag[2] * rinvmag[0]);
        dthetadr[2][3][YY] = sc1 * (tt3 * delr[2][YY] - delr[0][YY] * rinvmag[2] * rinvmag[0]);
        dthetadr[2][3][ZZ] = sc1 * (tt3 * delr[2][ZZ] - delr[0][ZZ] * rinvmag[2] * rinvmag[0]);

        for (int interaction = 0; interaction < 3; ++interaction)
        {
            const float cossin2 = -costheta[interaction] * invstheta[interaction] * invstheta[interaction];
            for (int atom = 0; atom < 4; ++atom)
            {
                for (int dim = 0; dim < DIM; ++dim)
                {
                    dinvsth[interaction][atom][dim] = cossin2 * dthetadr[interaction][atom][dim];
                }
            }
        }

        for (int interaction = 0; interaction < 3; ++interaction)
        {
            for (int atom = 0; atom < 4; ++atom)
            {
                for (int dim = 0; dim < DIM; ++dim)
                {
                    dinvs3r[interaction][atom][dim] =
                            invstheta[(interaction + 1) % 3] * dinv3r[atom][dim]
                            + inv3r * dinvsth[(interaction + 1) % 3][atom][dim];
                }
            }
        }

        float drAB[DIM][4][DIM] = {};
        float drCB[DIM][4][DIM] = {};
        float drDB[DIM][4][DIM] = {};
        for (int dim = 0; dim < DIM; ++dim)
        {
            drCB[dim][1][dim] = -1.0F;
            drAB[dim][1][dim] = -1.0F;
            drDB[dim][1][dim] = -1.0F;
            drDB[dim][3][dim] = 1.0F;
            drCB[dim][2][dim] = 1.0F;
            drAB[dim][0][dim] = 1.0F;
        }

        for (int dim = 0; dim < DIM; ++dim)
        {
            for (int atom = 0; atom < 4; ++atom)
            {
                pcffClass2CrossGpu(delr[1], drDB[dim][atom], rCBxdrDB);
                pcffClass2CrossGpu(drCB[dim][atom], delr[2], drCBxrDB);
                for (int axis = 0; axis < DIM; ++axis)
                {
                    dd[axis] = rCBxdrDB[axis] + drCBxrDB[axis];
                }
                fdot[0][atom][dim] =
                        pcffClass2DotGpu(dd, delr[0]) + pcffClass2DotGpu(rCBxrDB, drAB[dim][atom]);

                pcffClass2CrossGpu(delr[2], drAB[dim][atom], rDBxdrAB);
                pcffClass2CrossGpu(drDB[dim][atom], delr[0], drDBxrAB);
                for (int axis = 0; axis < DIM; ++axis)
                {
                    dd[axis] = rDBxdrAB[axis] + drDBxrAB[axis];
                }
                fdot[1][atom][dim] =
                        pcffClass2DotGpu(dd, delr[1]) + pcffClass2DotGpu(rDBxrAB, drCB[dim][atom]);

                pcffClass2CrossGpu(delr[0], drCB[dim][atom], rABxdrCB);
                pcffClass2CrossGpu(drAB[dim][atom], delr[1], drABxrCB);
                for (int axis = 0; axis < DIM; ++axis)
                {
                    dd[axis] = rABxdrCB[axis] + drABxrCB[axis];
                }
                fdot[2][atom][dim] =
                        pcffClass2DotGpu(dd, delr[2]) + pcffClass2DotGpu(rABxrCB, drDB[dim][atom]);
            }
        }

        for (int atom = 0; atom < 4; ++atom)
        {
            for (int dim = 0; dim < DIM; ++dim)
            {
                const float f0 =
                        (fdot[0][atom][dim] * invs3r[0] + dinvs3r[0][atom][dim] * dotCBDBAB) * invCosChiABCD;
                const float f1 =
                        (fdot[1][atom][dim] * invs3r[1] + dinvs3r[1][atom][dim] * dotDBABCB) * invCosChiCBDA;
                const float f2 =
                        (fdot[2][atom][dim] * invs3r[2] + dinvs3r[2][atom][dim] * dotABCBDB) * invCosChiDBAC;
                dtotalchi[atom][dim] = (f0 + f1 + f2) / 3.0F;
                fabcd[atom][dim] += -2.0F * params.k0 * deltachi * dtotalchi[atom][dim];
            }
        }
        }
    }

    if (hasAngleAngleTerm)
    {
        const bool needsThetaABC = hasAngleAngleTerm1 || hasAngleAngleTerm2;
        const bool needsThetaCBD = hasAngleAngleTerm1 || hasAngleAngleTerm3;
        const bool needsThetaABD = hasAngleAngleTerm2 || hasAngleAngleTerm3;

        const float delxAB = delr[0][XX];
        const float delyAB = delr[0][YY];
        const float delzAB = delr[0][ZZ];
        const float delxBC = delr[1][XX];
        const float delyBC = delr[1][YY];
        const float delzBC = delr[1][ZZ];
        const float delxBD = delr[2][XX];
        const float delyBD = delr[2][YY];
        const float delzBD = delr[2][ZZ];

        const float invRAB2 = rinvmag[0] * rinvmag[0];
        const float invRBC2 = rinvmag[1] * rinvmag[1];
        const float invRBD2 = rinvmag[2] * rinvmag[2];
        const float costhABC = costheta[0];
        const float costhABD = costheta[2];
        const float costhCBD = costheta[1];

        const float dthABC = needsThetaABC ? gmxDeviceAcos(costhABC) - params.aa_theta0_1 : 0.0F;
        const float dthABD = needsThetaABD ? gmxDeviceAcos(costhABD) - params.aa_theta0_2 : 0.0F;
        const float dthCBD = needsThetaCBD ? gmxDeviceAcos(costhCBD) - params.aa_theta0_3 : 0.0F;

        if constexpr (calcEner)
        {
            if (hasAngleAngleTerm1)
            {
                *vtot_loc += params.aa_k1 * dthABC * dthCBD;
            }
            if (hasAngleAngleTerm2)
            {
                *vtot_loc += params.aa_k2 * dthABC * dthABD;
            }
            if (hasAngleAngleTerm3)
            {
                *vtot_loc += params.aa_k3 * dthABD * dthCBD;
            }
        }

        float dthetadr[3][4][DIM] = {};

        if (needsThetaABC)
        {
            const float sc1 = invstheta[0];
            const float t1l = costhABC * invRAB2;
            const float t3l = costhABC * invRBC2;
            const float r12 = rinvmag[0] * rinvmag[1];

            dthetadr[0][0][XX] = sc1 * (t1l * delxAB - delxBC * r12);
            dthetadr[0][0][YY] = sc1 * (t1l * delyAB - delyBC * r12);
            dthetadr[0][0][ZZ] = sc1 * (t1l * delzAB - delzBC * r12);
            dthetadr[0][1][XX] = sc1 * (-t1l * delxAB + delxBC * r12 - t3l * delxBC + delxAB * r12);
            dthetadr[0][1][YY] = sc1 * (-t1l * delyAB + delyBC * r12 - t3l * delyBC + delyAB * r12);
            dthetadr[0][1][ZZ] = sc1 * (-t1l * delzAB + delzBC * r12 - t3l * delzBC + delzAB * r12);
            dthetadr[0][2][XX] = sc1 * (t3l * delxBC - delxAB * r12);
            dthetadr[0][2][YY] = sc1 * (t3l * delyBC - delyAB * r12);
            dthetadr[0][2][ZZ] = sc1 * (t3l * delzBC - delzAB * r12);
        }

        if (needsThetaCBD)
        {
            const float sc1 = invstheta[1];
            const float t1l = costhCBD * invRBC2;
            const float t3l = costhCBD * invRBD2;
            const float r12 = rinvmag[1] * rinvmag[2];

            dthetadr[1][2][XX] = sc1 * (t1l * delxBC - delxBD * r12);
            dthetadr[1][2][YY] = sc1 * (t1l * delyBC - delyBD * r12);
            dthetadr[1][2][ZZ] = sc1 * (t1l * delzBC - delzBD * r12);
            dthetadr[1][1][XX] = sc1 * (-t1l * delxBC + delxBD * r12 - t3l * delxBD + delxBC * r12);
            dthetadr[1][1][YY] = sc1 * (-t1l * delyBC + delyBD * r12 - t3l * delyBD + delyBC * r12);
            dthetadr[1][1][ZZ] = sc1 * (-t1l * delzBC + delzBD * r12 - t3l * delzBD + delzBC * r12);
            dthetadr[1][3][XX] = sc1 * (t3l * delxBD - delxBC * r12);
            dthetadr[1][3][YY] = sc1 * (t3l * delyBD - delyBC * r12);
            dthetadr[1][3][ZZ] = sc1 * (t3l * delzBD - delzBC * r12);
        }

        if (needsThetaABD)
        {
            const float sc1 = invstheta[2];
            const float t1l = costhABD * invRAB2;
            const float t3l = costhABD * invRBD2;
            const float r12 = rinvmag[0] * rinvmag[2];

            dthetadr[2][0][XX] = sc1 * (t1l * delxAB - delxBD * r12);
            dthetadr[2][0][YY] = sc1 * (t1l * delyAB - delyBD * r12);
            dthetadr[2][0][ZZ] = sc1 * (t1l * delzAB - delzBD * r12);
            dthetadr[2][1][XX] = sc1 * (-t1l * delxAB + delxBD * r12 - t3l * delxBD + delxAB * r12);
            dthetadr[2][1][YY] = sc1 * (-t1l * delyAB + delyBD * r12 - t3l * delyBD + delyAB * r12);
            dthetadr[2][1][ZZ] = sc1 * (-t1l * delzAB + delzBD * r12 - t3l * delzBD + delzAB * r12);
            dthetadr[2][3][XX] = sc1 * (t3l * delxBD - delxAB * r12);
            dthetadr[2][3][YY] = sc1 * (t3l * delyBD - delyAB * r12);
            dthetadr[2][3][ZZ] = sc1 * (t3l * delzBD - delzAB * r12);
        }

        for (int atom = 0; atom < 4; ++atom)
        {
            for (int dim = 0; dim < DIM; ++dim)
            {
                if (hasAngleAngleTerm1)
                {
                    fabcd[atom][dim] -=
                            params.aa_k1
                            * (dthABC * dthetadr[1][atom][dim] + dthCBD * dthetadr[0][atom][dim]);
                }
                if (hasAngleAngleTerm2)
                {
                    fabcd[atom][dim] -=
                            params.aa_k2
                            * (dthABC * dthetadr[2][atom][dim] + dthABD * dthetadr[0][atom][dim]);
                }
                if (hasAngleAngleTerm3)
                {
                    fabcd[atom][dim] -=
                            params.aa_k3
                            * (dthABD * dthetadr[1][atom][dim] + dthCBD * dthetadr[2][atom][dim]);
                }
            }
        }
    }

    const DeviceFloat3 f_i = { fabcd[0][XX], fabcd[0][YY], fabcd[0][ZZ] };
    const DeviceFloat3 f_j = { fabcd[1][XX], fabcd[1][YY], fabcd[1][ZZ] };
    const DeviceFloat3 f_k = { fabcd[2][XX], fabcd[2][YY], fabcd[2][ZZ] };
    const DeviceFloat3 f_l = { fabcd[3][XX], fabcd[3][YY], fabcd[3][ZZ] };

    staggeredAtomicAddForce(gm_f, f_i, ai, localId);
    staggeredAtomicAddForce(gm_f, f_j, aj, localId);
    staggeredAtomicAddForce(gm_f, f_k, ak, localId);
    staggeredAtomicAddForce(gm_f, f_l, al, localId);

    if constexpr (calcVir)
    {
        atomicFetchAddLocal(sm_fShiftLoc, t1, f_i);
        atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, f_j);
        atomicFetchAddLocal(sm_fShiftLoc, t2, f_k);
        atomicFetchAddLocal(sm_fShiftLoc, t3, f_l);
    }
}

//! Wrap angle from range [-3*pi; 3*pi) to [-pi; pi)
GMX_DEVICE_FUNC_ATTRIBUTE
static constexpr float wrapAngle(float a)
{
    constexpr float c_twoPi = 2.0F * c_Pi;
    if (a >= c_Pi)
    {
        return a - c_twoPi;
    }
    else if (a < -c_Pi)
    {
        return a + c_twoPi;
    }
    else
    {
        return a;
    }
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void idihs_gpu(const int               i,
                                                       DevicePrivatePtr<float> vtot_loc,
                                                       const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                       const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                       const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                       DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                       DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                       const PbcAiuc&                pbcAiuc,
                                                       const int                     localId)
{
    int type = gm_forceatoms[5 * i];
    int ai   = gm_forceatoms[5 * i + 1];
    int aj   = gm_forceatoms[5 * i + 2];
    int ak   = gm_forceatoms[5 * i + 3];
    int al   = gm_forceatoms[5 * i + 4];

    DeviceFloat3 r_ij;
    DeviceFloat3 r_kj;
    DeviceFloat3 r_kl;
    DeviceFloat3 m;
    DeviceFloat3 n;
    int          t1;
    int          t2;
    int          t3;
    float        phi = dih_angle_gpu<calcVir>(
            gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

    /* phi can jump if phi0 is close to Pi/-Pi, which will cause huge
     * force changes if we just apply a normal harmonic.
     * Instead, we first calculate phi-phi0 and take it modulo (-Pi,Pi).
     * This means we will never have the periodicity problem, unless
     * the dihedral is Pi away from phiO, which is very unlikely due to
     * the potential.
     */
    float kA = gm_forceparams[type].harmonic.krA;
    float pA = gm_forceparams[type].harmonic.rA;

    float phi0 = pA * c_deg2RadF;

    float dp = wrapAngle(phi - phi0);

    float ddphi = -kA * dp;

    do_dih_fup_gpu<calcVir>(
            ai, aj, ak, al, -ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, localId);

    if constexpr (calcEner)
    {
        *vtot_loc += -0.5F * ddphi * dp;
    }
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void pairs_gpu(const int i,
                                                       const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                       const DeviceGlobalPtr<const t_iparams> gm_iparams,
                                                       const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                       DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                       DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                       const PbcAiuc&                pbcAiuc,
                                                       const int                     repulsionPower,
                                                       const float                   scale_factor,
                                                       DevicePrivatePtr<float>       vtot_loc,
                                                       DevicePrivatePtr<float>       vtotElec_loc,
                                                       const int                     localId)
{
    int type = gm_forceatoms[3 * i];
    int ai   = gm_forceatoms[3 * i + 1];
    int aj   = gm_forceatoms[3 * i + 2];

    float qq  = gm_xq[ai][3] * gm_xq[aj][3];
    float c6  = gm_iparams[type].lj14.c6A;
    float c12 = gm_iparams[type].lj14.c12A;

    /* Do we need to apply full periodic boundary conditions? */
    DeviceFloat3 dr;
    int          fshift_index = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dr);

    float r2    = gmxDeviceNorm2(dr);
    float rinv  = gmxDeviceRSqrt(r2);
    float rinv2 = rinv * rinv;
    float rinv6 = rinv2 * rinv2 * rinv2;
    float rinvRep =
            (repulsionPower == 9) ? (rinv6 * rinv2 * rinv) : (rinv6 * rinv6);

    /* Calculate the Coulomb force * r */
    float velec = scale_factor * qq * rinv;

    /* Calculate the LJ force * r and add it to the Coulomb part */
    float fr = repulsionPower * c12 * rinvRep - 6.0F * c6 * rinv6 + velec;

    float        finvr = fr * rinv2;
    DeviceFloat3 f     = finvr * dr;

    /* Add the forces */
    staggeredAtomicAddForce(gm_f, f, ai, localId);
    staggeredAtomicAddForce(gm_f, -f, aj, localId);
    if constexpr (calcVir)
    {
        if (fshift_index != gmx::c_centralShiftIndex)
        {
            atomicFetchAddLocal(sm_fShiftLoc, fshift_index, f);
            atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, -f);
        }
    }

    // The elec and vdW contributions to the energy are separated only for the pairs
    // code, and combined later on.
    if constexpr (calcEner)
    {
        *vtot_loc += c12 * rinvRep - c6 * rinv6;
        *vtotElec_loc += velec;
    }
}

} // namespace gmx

#endif

#endif
