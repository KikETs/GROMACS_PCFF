/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2012- The GROMACS Authors
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
 * \brief
 * Declares the nbnxm pair interaction kernel function types and kind counts, also declares utility functions used in nbnxm_kernel.cpp.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#ifndef GMX_NBXNM_KERNEL_COMMON_H
#define GMX_NBXNM_KERNEL_COMMON_H

#include <cstdlib>
#include <cstring>

#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/utility/vectypes.h"
/* nbnxn_atomdata_t and nbnxn_pairlist_t could be forward declared, but that requires modifications in all SIMD kernel files */
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/real.h"

#include "nbnxm_enums.h"
#include "pairlist.h"

enum class CoulombInteractionType : int;
enum class VanDerWaalsType : int;
enum class InteractionModifiers : int;
enum class LongRangeVdW : int;

namespace gmx
{
enum class EwaldExclusionType : int;

/*! \brief Pair-interaction kernel type that also calculates energies.
 */
typedef void(NbnxmKernelFunc)(const NbnxnPairlistCpu&    nbl,
                              const nbnxn_atomdata_t&    nbat,
                              const interaction_const_t& ic,
                              const rvec*                shift_vec,
                              nbnxn_atomdata_output_t*   out);

//! \brief Lookup function for Coulomb kernel type
CoulombKernelType getCoulombKernelType(EwaldExclusionType     ewaldExclusionType,
                                       CoulombInteractionType coulombInteractionType,
                                       bool                   haveEqualCoulombVwdRadii,
                                       bool                   nbnxmIsDirectCoulombProvider);

/*! \brief Kinds of Van der Waals treatments in NBNxM SIMD kernels
 *
 * The \p LJCUT_COMB refers to the LJ combination rule for the short range.
 * The \p EWALDCOMB refers to the combination rule for the grid part.
 * \p vdwktNR is the number of VdW treatments for the SIMD kernels.
 * \p vdwktNR_ref is the number of VdW treatments for the C reference kernels.
 * These two numbers differ, because currently only the reference kernels
 * support LB combination rules for the LJ-Ewald grid part.
 */
enum
{
    vdwktLJCUT_COMBGEOM,
    vdwktLJCUT_COMBLB,
    vdwktLJCUT_COMBNONE,
    vdwktLJFORCESWITCH,
    vdwktLJPOTSWITCH,
    vdwktLJEWALDCOMBGEOM,
    vdwktLJEWALDCOMBLB,
    vdwktNR = vdwktLJEWALDCOMBLB,
    vdwktNR_ref
};

//! \brief Lookup function for Vdw kernel type
int getVdwKernelType(NbnxmKernelType      kernelType,
                     LJCombinationRule    ljCombinationRule,
                     VanDerWaalsType      vanDerWaalsType,
                     InteractionModifiers interactionModifiers,
                     LongRangeVdW         longRangeVdW);

/*! \brief Clears the shift forces.
 */
void clear_fshift(real* fshift);

/*! \brief Reduces the collected energy terms over the pair-lists/threads.
 */
void reduce_energies_over_lists(const nbnxn_atomdata_t* nbat, int nlist, real* Vvdw, real* Vc);

/*! \brief Returns whether the current CPU NBNXM launch should apply exact r-RESPA split weights. */
inline bool exactRespaCpuPairSplitLaunchActive(const interaction_const_t& ic)
{
    return ic.exactRespaCpuPairSplit.configured && ic.exactRespaCpuPairSplit.active
           && !ic.exactRespaCpuPairSplit.nativeMultiActive;
}

/*! \brief Returns whether the current CPU NBNXM launch writes multiple exact contributions. */
inline bool exactRespaCpuPairSplitNativeMultiLaunchActive(const interaction_const_t& ic)
{
    return ic.exactRespaCpuPairSplit.configured && ic.exactRespaCpuPairSplit.active
           && ic.exactRespaCpuPairSplit.nativeMultiActive
           && ic.exactRespaCpuPairSplit.nativeMultiContributionCount > 1;
}

/*! \brief Returns the current native multi-contribution count. */
inline int exactRespaCpuPairSplitNativeMultiContributionCount(const interaction_const_t& ic)
{
    GMX_RELEASE_ASSERT(exactRespaCpuPairSplitNativeMultiLaunchActive(ic),
                       "Native multi-contribution exact r-RESPA CPU launch must be active");
    return ic.exactRespaCpuPairSplit.nativeMultiContributionCount;
}

/*! \brief Returns one active native multi-contribution by launch-local index. */
inline MtsNonbondedRespaContribution exactRespaCpuPairSplitNativeMultiContribution(
        const interaction_const_t& ic, const int contributionIndex)
{
    GMX_RELEASE_ASSERT(exactRespaCpuPairSplitNativeMultiLaunchActive(ic),
                       "Native multi-contribution exact r-RESPA CPU launch must be active");
    GMX_RELEASE_ASSERT(contributionIndex >= 0
                               && contributionIndex < ic.exactRespaCpuPairSplit.nativeMultiContributionCount,
                       "Native multi-contribution exact r-RESPA CPU contribution index is out of range");
    return ic.exactRespaCpuPairSplit.nativeMultiContributions[contributionIndex];
}

/*! \brief Returns whether the native multi inner/middle dual-contribution fast path is enabled. */
inline bool exactRespaCpuPairSplitNativeMultiTwoContributionFastPathEnabled()
{
    static const bool enabled = []()
    {
        const char* env =
                std::getenv("GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_DISABLE_TWO_CONTRIB_FASTPATH");
        return env == nullptr || *env == '\0' || std::strcmp(env, "0") == 0;
    }();
    return enabled;
}

/*! \brief Returns whether one exact r-RESPA contribution owns the correction term. */
inline bool exactRespaCpuPairSplitContributionAddsCorrection(
        const MtsNonbondedRespaContribution contribution)
{
    return contribution == MtsNonbondedRespaContribution::Outer
           || contribution == MtsNonbondedRespaContribution::Full;
}

/*! \brief Cubic switch that matches the scalar exact r-RESPA split path. */
inline real exactRespaSwitchIn(const real r, const real off, const real on)
{
    if (on <= off)
    {
        return (r >= on ? 1.0_real : 0.0_real);
    }
    if (r <= off)
    {
        return 0.0_real;
    }
    if (r >= on)
    {
        return 1.0_real;
    }

    const real x = (r - off) / (on - off);
    return x * x * (3.0_real - 2.0_real * x);
}

/*! \brief Returns the exact r-RESPA direct-force weight for \p contribution at distance \p r. */
inline real exactRespaCpuPairSplitWeightForContribution(const interaction_const_t& ic,
                                                        const MtsNonbondedRespaContribution contribution,
                                                        const real r)
{
    if (!(exactRespaCpuPairSplitLaunchActive(ic) || exactRespaCpuPairSplitNativeMultiLaunchActive(ic)))
    {
        return 1.0_real;
    }

    GMX_RELEASE_ASSERT(ic.exactRespaCpuPairSplit.configured,
                       "Exact r-RESPA CPU split launch must have configured split metadata");
    switch (contribution)
    {
        case MtsNonbondedRespaContribution::Inner:
        {
            if (ic.exactRespaCpuPairSplit.hasMiddle)
            {
                return 1.0_real - exactRespaSwitchIn(
                                          r,
                                          ic.exactRespaCpuPairSplit.innerOff,
                                          ic.exactRespaCpuPairSplit.innerOn);
            }
            return 1.0_real - exactRespaSwitchIn(
                                      r,
                                      ic.exactRespaCpuPairSplit.outerOn,
                                      ic.exactRespaCpuPairSplit.outerOff);
        }
        case MtsNonbondedRespaContribution::Middle:
        {
            if (!ic.exactRespaCpuPairSplit.hasMiddle)
            {
                return 0.0_real;
            }
            const real switchIntoMiddle = exactRespaSwitchIn(
                    r, ic.exactRespaCpuPairSplit.innerOff, ic.exactRespaCpuPairSplit.innerOn);
            const real switchIntoOuter = exactRespaSwitchIn(
                    r, ic.exactRespaCpuPairSplit.outerOn, ic.exactRespaCpuPairSplit.outerOff);
            return switchIntoMiddle * (1.0_real - switchIntoOuter);
        }
        case MtsNonbondedRespaContribution::Outer:
            return exactRespaSwitchIn(
                    r, ic.exactRespaCpuPairSplit.outerOn, ic.exactRespaCpuPairSplit.outerOff);
        case MtsNonbondedRespaContribution::Full: return 1.0_real;
        case MtsNonbondedRespaContribution::Count:
            GMX_RELEASE_ASSERT(false, "Invalid exact r-RESPA CPU contribution");
            return 0.0_real;
    }

    GMX_RELEASE_ASSERT(false, "Unhandled exact r-RESPA CPU contribution");
    return 0.0_real;
}

/*! \brief Returns the exact r-RESPA direct-force weight for the current launch at distance \p r. */
inline real exactRespaCpuPairSplitWeight(const interaction_const_t& ic, const real r)
{
    return exactRespaCpuPairSplitWeightForContribution(ic, ic.exactRespaCpuPairSplit.contribution, r);
}

/*! \brief Returns whether the current exact r-RESPA launch owns the correction term. */
inline bool exactRespaCpuPairSplitAddsCorrection(const interaction_const_t& ic)
{
    return !exactRespaCpuPairSplitLaunchActive(ic)
           || exactRespaCpuPairSplitContributionAddsCorrection(ic.exactRespaCpuPairSplit.contribution);
}

} // namespace gmx

#endif
