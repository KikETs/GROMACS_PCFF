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
 * Declares CPU reference kernels
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_KERNELS_REFERENCE_KERNEL_REF_4X4_H
#define GMX_NBNXM_KERNELS_REFERENCE_KERNEL_REF_4X4_H

#include <array>
#include <cstdint>
#include <vector>

#include "gromacs/nbnxm/kernel_common.h"

namespace gmx
{

//! All the different CPU reference kernel functions.
//! \{
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJ_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJFsw_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJPsw_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJEwCombGeom_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJEwCombLB_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJ_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJFsw_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJPsw_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJEwCombGeom_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJEwCombLB_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJFsw_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJPsw_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombGeom_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombLB_F_ref;
#if GMX_USE_EXT_FMM
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJ_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJFsw_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJPsw_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJEwCombGeom_F_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJEwCombLB_F_ref;
#endif

NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJ_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJFsw_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJPsw_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJEwCombGeom_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJEwCombLB_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJ_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJFsw_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJPsw_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJEwCombGeom_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJEwCombLB_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJFsw_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJPsw_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombGeom_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombLB_VF_ref;
#if GMX_USE_EXT_FMM
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJ_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJFsw_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJPsw_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJEwCombGeom_VF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJEwCombLB_VF_ref;
#endif

NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJ_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJFsw_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJPsw_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJEwCombGeom_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecRF_VdwLJEwCombLB_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJ_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJFsw_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJPsw_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJEwCombGeom_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTab_VdwLJEwCombLB_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJFsw_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJPsw_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombGeom_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombLB_VgrpF_ref;
#if GMX_USE_EXT_FMM
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJ_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJFsw_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJPsw_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJEwCombGeom_VgrpF_ref;
NbnxmKernelFunc nbnxn_kernel_4x4_ElecNone_VdwLJEwCombLB_VgrpF_ref;
#endif

bool m2qPlain4x4EarliestRawTraceEnabled();
void resetM2qPlain4x4EarliestRawTrace();
void accumulateM2qPlain4x4EarliestRawTrace(real ljEnergy);
real readM2qPlain4x4EarliestRawTrace();
bool m2rPlain4x4AmplificationTraceEnabled();
void resetM2rPlain4x4AmplificationTrace();
void accumulateM2rPlain4x4KernelLocalAggregateTrace(real ljEnergy);
real readM2rPlain4x4KernelLocalAggregateTrace();
bool m2sPlain4x4InternalTraceEnabled();
void resetM2sPlain4x4InternalTrace();
void noteM2sPlain4x4FirstWriteTargetTotal(const real* values, int count);
real readM2sPlain4x4FirstWriteTargetTotal();
bool m2sPlain4x4FirstWriteCaptured();
bool m2uPlain4x4WriteOrdinalTraceEnabled();
void resetM2uPlain4x4WriteOrdinalTrace();
void noteM2uPlain4x4WriteTargetTotal(const real* values, int count);
std::vector<double> readM2uPlain4x4WriteOrdinalTotals();
bool m2vPlain4x4AlignedEventTraceEnabled();
void resetM2vPlain4x4AlignedEventTrace();
void noteM2vPlain4x4AlignedEvent(real ljEnergyDelta);
std::vector<double> readM2vPlain4x4AlignedEventTotals();
struct M2wPlain4x4AlignedEventRecord
{
    int    alignedEventOrdinal = 0;
    int    pairI               = 0;
    int    pairJ               = 0;
    int    typeI               = 0;
    int    typeJ               = 0;
    int    ciIndex             = 0;
    int    cjIndex             = 0;
    int    iIndex              = 0;
    int    jIndex              = 0;
    double runningTotalBefore  = 0.0;
    double runningTotalAfter   = 0.0;
    double rawLjTerm           = 0.0;
    double scalingFactor       = 0.0;
    double finalEventLj        = 0.0;
    double c6                  = 0.0;
    double c12                 = 0.0;
    double rsq                 = 0.0;
    double r                   = 0.0;
};
bool m2wPlain4x4AlignedEventTraceEnabled();
void resetM2wPlain4x4AlignedEventTrace();
void noteM2wPlain4x4AlignedEvent(int ai,
                                 int aj,
                                 int typeI,
                                 int typeJ,
                                 int ciIndex,
                                 int cjIndex,
                                 int iIndex,
                                 int jIndex,
                                 real c6,
                                 real c12,
                                 real rsq,
                                 real r,
                                 real rawLjTerm,
                                 real scalingFactor,
                                 real finalEventLj);
std::vector<M2wPlain4x4AlignedEventRecord> readM2wPlain4x4AlignedEventRecords();
std::vector<double>                        readM2wPlain4x4AlignedEventTotals();

struct M2xPlain4x4GeometryEventData
{
    int    pairI                = 0;
    int    pairJ                = 0;
    int    typeI                = 0;
    int    typeJ                = 0;
    int    ciIndex              = 0;
    int    cjIndex              = 0;
    int    iIndex               = 0;
    int    jIndex               = 0;
    int    shiftIndex           = 0;
    double coordISourceX        = 0.0;
    double coordISourceY        = 0.0;
    double coordISourceZ        = 0.0;
    double coordJSourceX        = 0.0;
    double coordJSourceY        = 0.0;
    double coordJSourceZ        = 0.0;
    double shiftX               = 0.0;
    double shiftY               = 0.0;
    double shiftZ               = 0.0;
    double coordIShiftedX       = 0.0;
    double coordIShiftedY       = 0.0;
    double coordIShiftedZ       = 0.0;
    double dx                   = 0.0;
    double dy                   = 0.0;
    double dz                   = 0.0;
    double rsq                  = 0.0;
    double r                    = 0.0;
    double rawLjTerm            = 0.0;
    double finalEventLj         = 0.0;
};
bool m2xPlain4x4GeometryTraceEnabled();
void resetM2xPlain4x4GeometryTrace();
void noteM2xPlain4x4GeometryEvent(const M2xPlain4x4GeometryEventData& data);
bool m2pPlain4x4CoulombFirstWriteTraceEnabled();
void noteM2pPlain4x4CoulombFirstWrite(real        targetBefore,
                                      real        writeValue,
                                      real        targetAfter,
                                      int         energyIndex,
                                      const char* codeLocation);
bool m2pPlain4x4CoulombProducerTraceEnabled();
void noteM2pPlain4x4CoulombProducer(int         pairI,
                                    int         pairJ,
                                    int         energyIndex,
                                    real        excludedMask,
                                    real        qq,
                                    real        interact,
                                    real        rinv,
                                    real        ewaldShift,
                                    int         tableIndex,
                                    real        frac,
                                    real        fexcl,
                                    real        vcorr,
                                    real        vcoul,
                                    real        vcoulUnmasked,
                                    const char* codeLocation);
bool m2pPlain4x4MultiStepCoulombPairTraceEnabled();
void noteM2pPlain4x4MultiStepCoulombPairContribution(int         pairI,
                                                     int         pairJ,
                                                     int         energyIndex,
                                                     int         shiftIndex,
                                                     real        coordIX,
                                                     real        coordIY,
                                                     real        coordIZ,
                                                     real        coordJX,
                                                     real        coordJY,
                                                     real        coordJZ,
                                                     real        shiftX,
                                                     real        shiftY,
                                                     real        shiftZ,
                                                     real        dx,
                                                     real        dy,
                                                     real        dz,
                                                     real        rsq,
                                                     int         clusterI,
                                                     int         clusterJ,
                                                     int         localI,
                                                     int         localJ,
                                                     real        qq,
                                                     real        interact,
                                                     real        rinv,
                                                     int         tableIndex,
                                                     real        frac,
                                                     real        fexcl,
                                                     real        vcorr,
                                                     real        vcoul,
                                                     const char* codeLocation);
bool m2pPlain4x4CoulombSelfTraceEnabled();
void noteM2pPlain4x4CoulombSelfContribution(int         atom,
                                            int         energyIndex,
                                            real        charge,
                                            real        selfEnergy,
                                            real        targetBefore,
                                            real        targetAfter,
                                            const char* codeLocation);
void resetM2pPlain4x4CoulombContractReplay();
bool m2pPlain4x4CoulombContractReplayEnabled();
void noteM2pPlain4x4CoulombContractReplayPairContribution(int energyIndex, real vcoul);
void noteM2pPlain4x4CoulombContractReplaySelfContribution(int energyIndex, real selfEnergy);
double readM2pPlain4x4CoulombContractReplayTotal();
double readM2pPlain4x4CoulombContractReplayPairOnlyTotal();
void   setM2pPlain4x4CurrentStep(int64_t step);
int64_t readM2pPlain4x4CurrentStep();
void resetM2pPlain4x4LjContractReplay();
bool m2pPlain4x4LjContractReplayEnabled();
void noteM2pPlain4x4LjContractReplayPairContribution(real vlj);
double readM2pPlain4x4LjContractReplayTotal();

struct M2pPlain4x4TracedForcePair
{
    std::array<std::array<double, DIM>, 2> atoms = { { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } } };
};

void                    resetM2pPlain4x4RealspaceForceSubcomponentTrace();
bool                    m2pPlain4x4RealspaceForceSubcomponentTraceEnabled();
void                    noteM2pPlain4x4RealspaceForceSubcomponents(int  ai,
                                                                   int  aj,
                                                                   real ljFx,
                                                                   real ljFy,
                                                                   real ljFz,
                                                                   real coulombSrFx,
                                                                   real coulombSrFy,
                                                                   real coulombSrFz,
                                                                   real exclusionCorrectionFx,
                                                                   real exclusionCorrectionFy,
                                                                   real exclusionCorrectionFz,
                                                                   real combinedFx,
                                                                   real combinedFy,
                                                                   real combinedFz);
M2pPlain4x4TracedForcePair readM2pPlain4x4LjSrForcePair();
M2pPlain4x4TracedForcePair readM2pPlain4x4CoulombSrForcePair();
M2pPlain4x4TracedForcePair readM2pPlain4x4ExclusionCorrectionForcePair();
M2pPlain4x4TracedForcePair readM2pPlain4x4CombinedRealspaceForcePair();

//! \}

#ifdef INCLUDE_KERNELFUNCTION_TABLES

/*! \brief Declare and define the kernel function pointer lookup tables.
 *
 * The minor index of the array goes over both the LJ combination rules,
 * which is only supported by plain cut-off, and the LJ switch/PME functions.
 * For the C reference kernels, unlike the SIMD kernels, there is not much
 * advantage in using combination rules, so we (re-)use the same kernel.
 */
//! \{
static NbnxmKernelFunc* const nbnxn_kernel_4x4_noener_ref[static_cast<int>(CoulombKernelType::Count)][vdwktNR_ref] = {
    { nbnxn_kernel_4x4_ElecRF_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJFsw_F_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJPsw_F_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJEwCombGeom_F_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJEwCombLB_F_ref },
    { nbnxn_kernel_4x4_ElecQSTab_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJFsw_F_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJPsw_F_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJEwCombGeom_F_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJEwCombLB_F_ref },
    { nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJFsw_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJPsw_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombGeom_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombLB_F_ref },
    { nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJFsw_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJPsw_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombGeom_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombLB_F_ref },
    { nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJFsw_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJPsw_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombGeom_F_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombLB_F_ref },
#    if GMX_USE_EXT_FMM
    { nbnxn_kernel_4x4_ElecNone_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJ_F_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJFsw_F_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJPsw_F_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJEwCombGeom_F_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJEwCombLB_F_ref }
#    else
    { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr }
#    endif
};

static NbnxmKernelFunc* const nbnxn_kernel_4x4_ener_ref[static_cast<int>(CoulombKernelType::Count)][vdwktNR_ref] = {
    { nbnxn_kernel_4x4_ElecRF_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJFsw_VF_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJPsw_VF_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJEwCombGeom_VF_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJEwCombLB_VF_ref },
    { nbnxn_kernel_4x4_ElecQSTab_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJFsw_VF_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJPsw_VF_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJEwCombGeom_VF_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJEwCombLB_VF_ref },
    { nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJFsw_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJPsw_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombGeom_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombLB_VF_ref },
    { nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJFsw_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJPsw_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombGeom_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombLB_VF_ref },
    { nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJFsw_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJPsw_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombGeom_VF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombLB_VF_ref },
#    if GMX_USE_EXT_FMM
    { nbnxn_kernel_4x4_ElecNone_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJ_VF_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJFsw_VF_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJPsw_VF_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJEwCombGeom_VF_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJEwCombLB_VF_ref }
#    else
    { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr }
#    endif
};

static NbnxmKernelFunc* const nbnxn_kernel_4x4_energrp_ref[static_cast<int>(CoulombKernelType::Count)][vdwktNR_ref] = {
    { nbnxn_kernel_4x4_ElecRF_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJFsw_VgrpF_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJPsw_VgrpF_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJEwCombGeom_VgrpF_ref,
      nbnxn_kernel_4x4_ElecRF_VdwLJEwCombLB_VgrpF_ref },
    { nbnxn_kernel_4x4_ElecQSTab_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJFsw_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJPsw_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJEwCombGeom_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTab_VdwLJEwCombLB_VgrpF_ref },
    { nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJFsw_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJPsw_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombGeom_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombLB_VgrpF_ref },
    { nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJFsw_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJPsw_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombGeom_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombLB_VgrpF_ref },
    { nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJFsw_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJPsw_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombGeom_VgrpF_ref,
      nbnxn_kernel_4x4_ElecQSTabTwinCut_VdwLJEwCombLB_VgrpF_ref },
#    if GMX_USE_EXT_FMM
    { nbnxn_kernel_4x4_ElecNone_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJ_VgrpF_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJFsw_VgrpF_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJPsw_VgrpF_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJEwCombGeom_VgrpF_ref,
      nbnxn_kernel_4x4_ElecNone_VdwLJEwCombLB_VgrpF_ref }
#    else
    { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr }
#    endif
};
//! \}

#endif /* INCLUDE_KERNELFUNCTION_TABLES */

} // namespace gmx

#endif
