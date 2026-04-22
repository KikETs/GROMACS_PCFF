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

#include "gmxpre.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "kernels_reference/kernel_gpu_ref.h"

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/mdlib/enerdata_utils.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/pairlist.h"
#include "gromacs/simd/simd.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/vectypes.h"

#include "kernel_common.h"
#include "nbnxm_enums.h"
#include "nbnxm_geometry.h"
#include "nbnxm_gpu.h"
#include "nbnxm_simd.h"
#include "pairlistset.h"
#include "pairlistsets.h"
#define INCLUDE_KERNELFUNCTION_TABLES
#include "kernels_reference/kernel_ref_1x1.h"
#include "kernels_reference/kernel_ref_4x4.h"
#if GMX_HAVE_NBNXM_SIMD_2XMM
#    include "kernels_simd_2xmm/kernels.h"
#endif
#if GMX_HAVE_NBNXM_SIMD_4XM
#    include "kernels_simd_4xm/kernels.h"
#endif
#undef INCLUDE_FUNCTION_TABLES
#include "simd_energy_accumulator.h"

namespace gmx
{

namespace
{

bool exactRespaCpuNbnxmKernelSupported(const NbnxmKernelType kernelType)
{
    switch (kernelType)
    {
        case NbnxmKernelType::Cpu4x4_PlainC:
        case NbnxmKernelType::Cpu4xN_Simd_4xN:
        case NbnxmKernelType::Cpu4xN_Simd_2xNN:
        case NbnxmKernelType::Cpu1x1_PlainC: return true;
        default: return false;
    }
}

void appendM2pTraceTextLine(const char* traceDirPath, const char* fileName, const std::string& line)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    std::filesystem::create_directories(traceDirPath);
    std::ofstream output(std::filesystem::path(traceDirPath) / fileName, std::ios::app);
    output << line << '\n';
}

void writeM2pTraceTextFile(const char* traceDirPath, const char* fileName, const std::string& contents)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    std::filesystem::create_directories(traceDirPath);
    std::ofstream output(std::filesystem::path(traceDirPath) / fileName, std::ios::trunc);
    output << contents;
}

const std::vector<int64_t>& multiStepCoulombTraceSteps()
{
    static const std::vector<int64_t> steps = []()
    {
        std::vector<int64_t> parsedSteps;
        const char*          value = std::getenv("GMX_PCFF_RESPA_TRACE_MULTI_STEP_COULOMB_STEPS");
        if (value == nullptr || *value == '\0')
        {
            return parsedSteps;
        }

        std::stringstream ss(value);
        std::string       item;
        while (std::getline(ss, item, ','))
        {
            if (!item.empty())
            {
                parsedSteps.push_back(std::stoll(item));
            }
        }
        return parsedSteps;
    }();
    return steps;
}

bool shouldTraceMultiStepCoulombStep(const int64_t step)
{
    const auto& traceSteps = multiStepCoulombTraceSteps();
    return std::find(traceSteps.begin(), traceSteps.end(), step) != traceSteps.end();
}

struct KernelEnergyReadTrace
{
    double firstReadTotal = 0.0;
    double finalTotal     = 0.0;
    bool   firstCaptured  = false;
};

double sumKernelStoredEnergyOutputs(const nbnxn_atomdata_t* nbat, int nlist, const bool vdwEnergy)
{
    const int nenergrp = nbat->params().numEnergyGroups;
    double    total    = 0.0;

    for (int nb = 0; nb < nlist; ++nb)
    {
        const auto& energies = vdwEnergy ? nbat->outputBuffer(nb).Vvdw : nbat->outputBuffer(nb).Vc;
        for (int i = 0; i < nenergrp; ++i)
        {
            for (int j = 0; j < nenergrp; ++j)
            {
                total += energies[i * nenergrp + j];
            }
        }
    }

    return total;
}

KernelEnergyReadTrace traceKernelEnergyOutputs(const nbnxn_atomdata_t* nbat, int nlist, const bool vdwEnergy)
{
    const int             nenergrp = nbat->params().numEnergyGroups;
    KernelEnergyReadTrace trace;
    double                total = 0.0;

    const auto noteRead = [&trace, &total](const real value)
    {
        total += value;
        if (!trace.firstCaptured)
        {
            trace.firstReadTotal = total;
            trace.firstCaptured  = true;
        }
    };

    for (int nb = 0; nb < nlist; ++nb)
    {
        const auto& energies = vdwEnergy ? nbat->outputBuffer(nb).Vvdw : nbat->outputBuffer(nb).Vc;
        for (int i = 0; i < nenergrp; ++i)
        {
            const int indDiagonal = i * nenergrp + i;
            noteRead(energies[indDiagonal]);
            for (int j = i + 1; j < nenergrp; ++j)
            {
                const int ind  = i * nenergrp + j;
                const int indr = j * nenergrp + i;
                noteRead(energies[ind]);
                noteRead(energies[indr]);
            }
        }
    }

    trace.finalTotal = total;
    return trace;
}

double sumKernelEnergyOutputs(const nbnxn_atomdata_t* nbat, int nlist, const bool vdwEnergy)
{
    return traceKernelEnergyOutputs(nbat, nlist, vdwEnergy).finalTotal;
}

} // namespace
enum class InteractionLocality : int;

CoulombKernelType getCoulombKernelType(const EwaldExclusionType     ewaldExclusionType,
                                       const CoulombInteractionType coulombInteractionType,
                                       const bool                   haveEqualCoulombVwdRadii,
                                       const bool                   nbnxmIsDirectCoulombProvider)
{
    if (usingRF(coulombInteractionType) || coulombInteractionType == CoulombInteractionType::Cut)
    {
        return CoulombKernelType::ReactionField;
    }
    else if (coulombInteractionType == CoulombInteractionType::Fmm)
    {
        if (nbnxmIsDirectCoulombProvider)
        {
            GMX_RELEASE_ASSERT(
                    false,
                    "FMM is not yet supported in GROMACS for short-range coulomb interactions");

            return CoulombKernelType::Fmm;
        }
        else
        {
            return CoulombKernelType::None;
        }
    }
    else
    {
        if (ewaldExclusionType == EwaldExclusionType::Table)
        {
            if (haveEqualCoulombVwdRadii)
            {
                return CoulombKernelType::Table;
            }
            else
            {
                return CoulombKernelType::TableTwin;
            }
        }
        else
        {
            if (haveEqualCoulombVwdRadii)
            {
                return CoulombKernelType::Ewald;
            }
            else
            {
                return CoulombKernelType::EwaldTwin;
            }
        }
    }
}

int getVdwKernelType(const NbnxmKernelType      kernelType,
                     const LJCombinationRule    ljCombinationRule,
                     const VanDerWaalsType      vanDerWaalsType,
                     const InteractionModifiers interactionModifiers,
                     const LongRangeVdW         longRangeVdW)
{
    if (vanDerWaalsType == VanDerWaalsType::Cut)
    {
        switch (interactionModifiers)
        {
            case InteractionModifiers::None:
            case InteractionModifiers::PotShift:
                switch (ljCombinationRule)
                {
                    case LJCombinationRule::Geometric: return vdwktLJCUT_COMBGEOM;
                    case LJCombinationRule::LorentzBerthelot: return vdwktLJCUT_COMBLB;
                    case LJCombinationRule::None: return vdwktLJCUT_COMBNONE;
                    default: GMX_THROW(gmx::InvalidInputError("Unknown combination rule"));
                }
            case InteractionModifiers::ForceSwitch: return vdwktLJFORCESWITCH;
            case InteractionModifiers::PotSwitch: return vdwktLJPOTSWITCH;
            default:
                std::string errorMsg =
                        gmx::formatString("Unsupported VdW interaction modifier %s (%d)",
                                          enumValueToString(interactionModifiers),
                                          static_cast<int>(interactionModifiers));
                GMX_THROW(gmx::InvalidInputError(errorMsg));
        }
    }
    else if (vanDerWaalsType == VanDerWaalsType::Pme)
    {
        if (longRangeVdW == LongRangeVdW::Geom)
        {
            return vdwktLJEWALDCOMBGEOM;
        }
        else
        {
            /* At setup we (should have) selected the C reference kernel */
            GMX_RELEASE_ASSERT(kernelTypeIsPlainC(kernelType),
                               "Only the C reference nbnxn SIMD kernel supports LJ-PME with LB "
                               "combination rules");
            return vdwktLJEWALDCOMBLB;
        }
    }
    else
    {
        std::string errorMsg = gmx::formatString("Unsupported VdW interaction type %s (%d)",
                                                 enumValueToString(vanDerWaalsType),
                                                 static_cast<int>(vanDerWaalsType));
        GMX_THROW(gmx::InvalidInputError(errorMsg));
    }
}

/*! \brief Dispatches the non-bonded N versus M atom cluster CPU kernels.
 *
 * OpenMP parallelization is performed within this function.
 * Energy reduction, but not force and shift force reduction, is performed
 * within this function.
 *
 * \param[in]     pairlistSet   Pairlists with local or non-local interactions to compute
 * \param[in]     kernelSetup   The non-bonded kernel setup
 * \param[in,out] nbat          The atomdata for the interactions
 * \param[in]     ic            Non-bonded interaction constants
 * \param[in]     shiftVectors  The PBC shift vectors
 * \param[in]     stepWork      Flags that tell what to compute
 * \param[in]     clearF        Enum that tells if to clear the force output buffer
 * \param[out]    vCoulomb      Output buffer for Coulomb energies
 * \param[out]    vVdw          Output buffer for Van der Waals energies
 * \param[in]     wcycle        Pointer to cycle counting data structure.
 */
static void nbnxn_kernel_cpu(const PairlistSet&             pairlistSet,
                             const NbnxmKernelSetup&        kernelSetup,
                             nbnxn_atomdata_t*              nbat,
                             const interaction_const_t&     ic,
                             gmx::ArrayRef<const gmx::RVec> shiftVectors,
                             const gmx::StepWorkload&       stepWork,
                             int                            clearF,
                             real*                          vCoulomb,
                             real*                          vVdw,
                             gmx_wallcycle*                 wcycle)
{
    const nbnxn_atomdata_t::Params& nbatParams = nbat->params();

    GMX_ASSERT(ic.vdw.type != VanDerWaalsType::Pme
                       || ((ic.vdw.pmeCombinationRule == LongRangeVdW::Geom
                            && nbatParams.ljCombinationRule == LJCombinationRule::Geometric)
                           || (ic.vdw.pmeCombinationRule == LongRangeVdW::LB
                               && nbatParams.ljCombinationRule == LJCombinationRule::LorentzBerthelot)),
               "nbat combination rule parameters should match those for LJ-PME");

    const int coulkt = static_cast<int>(getCoulombKernelType(kernelSetup.ewaldExclusionType,
                                                             ic.coulomb.type,
                                                             (ic.coulomb.cutoff == ic.vdw.cutoff),
                                                             ic.nbnxmIsDirectCoulombProvider));

    const int vdwkt = getVdwKernelType(kernelSetup.kernelType,
                                       nbatParams.ljCombinationRule,
                                       ic.vdw.type,
                                       ic.vdw.modifier,
                                       ic.vdw.pmeCombinationRule);

    const bool usingSimdKernel = kernelTypeIsSimd(kernelSetup.kernelType);
    const char* m2wTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2W_TRACE_DIR");
    const char* m2wCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2W_CASE_LABEL");
    const bool  dumpM2wTrace = (m2wTraceDirPath != nullptr && *m2wTraceDirPath != '\0');
    const char* m2xTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2X_TRACE_DIR");
    const char* gmx_unused m2xCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2X_CASE_LABEL");
    const bool  dumpM2xTrace = (m2xTraceDirPath != nullptr && *m2xTraceDirPath != '\0');
    const char* m2vTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2V_TRACE_DIR");
    const char* m2vCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2V_CASE_LABEL");
    const bool  dumpM2vTrace = (m2vTraceDirPath != nullptr && *m2vTraceDirPath != '\0');
    const char* m2uTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2U_TRACE_DIR");
    const char* m2uCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2U_CASE_LABEL");
    const bool  dumpM2uTrace = (m2uTraceDirPath != nullptr && *m2uTraceDirPath != '\0');
    const char* m2sTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2S_TRACE_DIR");
    const char* m2sCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2S_CASE_LABEL");
    const bool  dumpM2sTrace = (m2sTraceDirPath != nullptr && *m2sTraceDirPath != '\0');
    const char* m2rTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2R_TRACE_DIR");
    const char* m2rCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2R_CASE_LABEL");
    const bool  dumpM2rTrace = (m2rTraceDirPath != nullptr && *m2rTraceDirPath != '\0');
    const char* m2qTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2Q_TRACE_DIR");
    const char* m2qCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2Q_CASE_LABEL");
    const bool  dumpM2qTrace = (m2qTraceDirPath != nullptr && *m2qTraceDirPath != '\0');
    const char* m2pTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2P_TRACE_DIR");
    const char* m2pCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2P_CASE_LABEL");
    const bool  dumpM2pTrace = (m2pTraceDirPath != nullptr && *m2pTraceDirPath != '\0');
    if ((dumpM2wTrace || dumpM2xTrace || dumpM2vTrace || dumpM2uTrace || dumpM2qTrace || dumpM2rTrace
         || dumpM2sTrace || dumpM2pTrace)
        && kernelSetup.kernelType == NbnxmKernelType::Cpu4x4_PlainC)
    {
        resetM2wPlain4x4AlignedEventTrace();
        resetM2xPlain4x4GeometryTrace();
        resetM2vPlain4x4AlignedEventTrace();
        resetM2uPlain4x4WriteOrdinalTrace();
        resetM2qPlain4x4EarliestRawTrace();
        resetM2rPlain4x4AmplificationTrace();
        resetM2sPlain4x4InternalTrace();
        resetM2pPlain4x4CoulombContractReplay();
        resetM2pPlain4x4LjContractReplay();
        resetM2pPlain4x4RealspaceForceSubcomponentTrace();
    }

    gmx::ArrayRef<const NbnxnPairlistCpu> pairlists = pairlistSet.cpuLists();
    const int64_t                         currentTraceStep               = readM2pPlain4x4CurrentStep();
    const bool                            dumpMultiStepCoulombStateTrace =
            dumpM2pTrace && kernelSetup.kernelType == NbnxmKernelType::Cpu4x4_PlainC
            && shouldTraceMultiStepCoulombStep(currentTraceStep);
    const double plainNativeCoulBeforeStep =
            dumpMultiStepCoulombStateTrace ? sumKernelStoredEnergyOutputs(nbat, pairlists.ssize(), false) : 0.0;
    if (dumpMultiStepCoulombStateTrace)
    {
        static std::string clearedMultiStepTracePath;
        const std::string  tracePath =
                (std::filesystem::path(m2pTraceDirPath) / "multistep_coulomb_state_trace.txt").string();
        if (tracePath != clearedMultiStepTracePath)
        {
            writeM2pTraceTextFile(m2pTraceDirPath, "multistep_coulomb_state_trace.txt", "");
            clearedMultiStepTracePath = tracePath;
        }
    }

    const auto* shiftVecPointer          = as_rvec_array(shiftVectors.data());
    const bool  exactRespaNativeMultiActive =
            exactRespaCpuPairSplitNativeMultiLaunchActive(ic) && stepWork.computeForces;

    int gmx_unused nthreads = gmx_omp_nthreads_get(ModuleMultiThread::Nonbonded);
    wallcycle_sub_start(wcycle, WallCycleSubCounter::NonbondedClear);
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (gmx::Index nb = 0; nb < pairlists.ssize(); nb++)
    {
        // Presently, the kernels do not call C++ code that can throw,
        // so no need for a try/catch pair in this OpenMP region.
        nbnxn_atomdata_output_t& out = nbat->outputBuffer(nb);

        if (clearF == enbvClearFYes && exactRespaNativeMultiActive)
        {
            const int contributionCount = exactRespaCpuPairSplitNativeMultiContributionCount(ic);
            for (int contributionIndex = 0; contributionIndex < contributionCount; ++contributionIndex)
            {
                nbnxn_atomdata_output_t* nativeOut =
                        nbat->correspondingNativeMultiContributionOutputBuffer(contributionIndex, &out);
                std::fill(nativeOut->f.begin(), nativeOut->f.end(), 0.0_real);
                clear_fshift(nativeOut->fshift.data());
            }
        }
        else if (clearF == enbvClearFYes)
        {
            nbat->clearForceBuffer(nb);

            clear_fshift(out.fshift.data());
        }

        if (nb == 0)
        {
            wallcycle_sub_stop(wcycle, WallCycleSubCounter::NonbondedClear);
            wallcycle_sub_start(wcycle, WallCycleSubCounter::NonbondedKernel);
        }

        // TODO: Change to reference
        const NbnxnPairlistCpu& pairlist = pairlists[nb];

        if (!stepWork.computeEnergy)
        {
            /* Don't calculate energies */
            switch (kernelSetup.kernelType)
            {
                case NbnxmKernelType::Cpu4x4_PlainC:
                    nbnxn_kernel_4x4_noener_ref[coulkt][vdwkt](pairlist, *nbat, ic, shiftVecPointer, &out);
                    break;
#if GMX_HAVE_NBNXM_SIMD_2XMM
                case NbnxmKernelType::Cpu4xN_Simd_2xNN:
                    gmx::nbnxmKernelNoenerSimd2xmm[coulkt][vdwkt](
                            pairlist, *nbat, ic, shiftVecPointer, &out);
                    break;
#endif
#if GMX_HAVE_NBNXM_SIMD_4XM
                case NbnxmKernelType::Cpu4xN_Simd_4xN:
                    gmx::nbnxmKernelNoenerSimd4xm[coulkt][vdwkt](
                            pairlist, *nbat, ic, shiftVecPointer, &out);
                    break;
#endif
                case NbnxmKernelType::Cpu1x1_PlainC:
                    nbnxn_kernel_1x1_noener_ref[coulkt][vdwkt](pairlist, *nbat, ic, shiftVecPointer, &out);
                    break;
                default: GMX_RELEASE_ASSERT(false, "Unsupported kernel architecture");
            }
        }
        else if (out.Vvdw.size() == 1)
        {
            /* A single energy group (pair) */

            if (usingSimdKernel)
            {
                out.accumulatorSingleEnergies->clearEnergies();
            }
            else
            {
                out.Vvdw[0] = 0;
                out.Vc[0]   = 0;
            }

            switch (kernelSetup.kernelType)
            {
                case NbnxmKernelType::Cpu4x4_PlainC:
                    nbnxn_kernel_4x4_ener_ref[coulkt][vdwkt](pairlist, *nbat, ic, shiftVecPointer, &out);
                    break;
#if GMX_HAVE_NBNXM_SIMD_2XMM
                case NbnxmKernelType::Cpu4xN_Simd_2xNN:
                    gmx::nbnxmKernelEnerSimd2xmm[coulkt][vdwkt](pairlist, *nbat, ic, shiftVecPointer, &out);
                    break;
#endif
#if GMX_HAVE_NBNXM_SIMD_4XM
                case NbnxmKernelType::Cpu4xN_Simd_4xN:
                    gmx::nbnxmKernelEnerSimd4xm[coulkt][vdwkt](pairlist, *nbat, ic, shiftVecPointer, &out);
                    break;
#endif
                case NbnxmKernelType::Cpu1x1_PlainC:
                    nbnxn_kernel_1x1_ener_ref[coulkt][vdwkt](pairlist, *nbat, ic, shiftVecPointer, &out);
                    break;
                default: GMX_RELEASE_ASSERT(false, "Unsupported kernel architecture");
            }

            if (usingSimdKernel)
            {
                out.accumulatorSingleEnergies->getEnergies(out.Vc, out.Vvdw);
            }
        }
        else
        {
            /* Calculate energy group contributions */

            if (usingSimdKernel)
            {
                out.accumulatorGroupEnergies->clearEnergiesAndSetEnergyGroupsForJClusters(
                        *nbatParams.energyGroupsPerCluster);
            }
            else
            {
                std::fill(out.Vvdw.begin(), out.Vvdw.end(), 0.0_real);
                std::fill(out.Vc.begin(), out.Vc.end(), 0.0_real);
            }

            switch (kernelSetup.kernelType)
            {
                case NbnxmKernelType::Cpu4x4_PlainC:
                    nbnxn_kernel_4x4_energrp_ref[coulkt][vdwkt](pairlist, *nbat, ic, shiftVecPointer, &out);
                    break;
#if GMX_HAVE_NBNXM_SIMD_2XMM
                case NbnxmKernelType::Cpu4xN_Simd_2xNN:
                    gmx::nbnxmKernelEnergrpSimd2xmm[coulkt][vdwkt](
                            pairlist, *nbat, ic, shiftVecPointer, &out);
                    break;
#endif
#if GMX_HAVE_NBNXM_SIMD_4XM
                case NbnxmKernelType::Cpu4xN_Simd_4xN:
                    gmx::nbnxmKernelEnergrpSimd4xm[coulkt][vdwkt](
                            pairlist, *nbat, ic, shiftVecPointer, &out);
                    break;
#endif
                case NbnxmKernelType::Cpu1x1_PlainC:
                    nbnxn_kernel_1x1_energrp_ref[coulkt][vdwkt](pairlist, *nbat, ic, shiftVecPointer, &out);
                    break;
                default: GMX_RELEASE_ASSERT(false, "Unsupported kernel architecture");
            }

            if (usingSimdKernel)
            {
                out.accumulatorGroupEnergies->getEnergies(out.Vc, out.Vvdw);
            }
        }
    }
    wallcycle_sub_stop(wcycle, WallCycleSubCounter::NonbondedKernel);

    if (!stepWork.computeEnergy && dumpMultiStepCoulombStateTrace)
    {
        appendM2pTraceTextLine(
                m2pTraceDirPath,
                "multistep_coulomb_state_trace.txt",
                "side=PLAIN step=" + std::to_string(currentTraceStep)
                        + " code_location=src/gromacs/nbnxm/kerneldispatch.cpp:post_kernel_no_energy_step"
                        + " compute_energy=false"
                        + " plain_native_coul_total_before_step="
                        + formatString("%.15f", plainNativeCoulBeforeStep)
                        + " plain_native_coul_final=0.000000000000000"
                        + " plain_replay_coul_total_before_step=0.000000000000000"
                        + " plain_replay_coul_total_after_pairs=0.000000000000000"
                        + " plain_replay_coul_total_after_excluded=0.000000000000000"
                        + " plain_replay_coul_total_after_self=0.000000000000000"
                        + " plain_replay_coul_final=0.000000000000000");
    }

    if (stepWork.computeEnergy)
    {
        const char* ljSrTraceDirPath =
                dumpM2wTrace ? m2wTraceDirPath
                             : (dumpM2vTrace ? m2vTraceDirPath
                             : (dumpM2uTrace ? m2uTraceDirPath
                             : (dumpM2sTrace ? m2sTraceDirPath
                                             : (dumpM2rTrace ? m2rTraceDirPath
                                                             : (dumpM2qTrace ? m2qTraceDirPath
                                                                             : std::getenv(
                                                                                       "GMX_PCFF_RESPA_M2P_TRACE_DIR"))))));
        const char* ljSrCaseLabelEnv =
                dumpM2wTrace ? m2wCaseLabelEnv
                             : (dumpM2vTrace ? m2vCaseLabelEnv
                             : (dumpM2uTrace ? m2uCaseLabelEnv
                             : (dumpM2sTrace ? m2sCaseLabelEnv
                                             : (dumpM2rTrace ? m2rCaseLabelEnv
                                                             : (dumpM2qTrace ? m2qCaseLabelEnv
                                                                             : std::getenv(
                                                                                       "GMX_PCFF_RESPA_M2P_CASE_LABEL"))))));
        if ((dumpM2qTrace || dumpM2rTrace || dumpM2sTrace || dumpM2uTrace || dumpM2vTrace || dumpM2wTrace)
            && kernelSetup.kernelType == NbnxmKernelType::Cpu4x4_PlainC)
        {
            appendM2pTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_lj_sr_internal_trace.txt",
                    "stage=EARLIEST_RAW_STAGE code_location=src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:pre_Vvdw_accumulation case_label="
                            + std::string(
                                    (ljSrCaseLabelEnv != nullptr && *ljSrCaseLabelEnv != '\0') ? ljSrCaseLabelEnv :
                                                                                                  "unknown")
                            + " execution_path=plain_cpu4x4_ref_kernel_inner kernel_type="
                            + std::string(nbnxmKernelTypeToName(kernelSetup.kernelType))
                            + " using_simd_kernel=false trace_role=contract_matched_raw_lj_formation_aggregate lj_sr="
                            + formatString("%.15f", readM2qPlain4x4EarliestRawTrace()));
        }
        if ((dumpM2rTrace || dumpM2sTrace) && kernelSetup.kernelType == NbnxmKernelType::Cpu4x4_PlainC)
        {
            appendM2pTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_lj_sr_internal_trace.txt",
                    "stage=INTERMEDIATE_LOCAL_STAGE code_location=src/gromacs/nbnxm/kernels_reference/kernel_ref_outer.h:post_kernel_local_energy_buffer_before_dispatch_transfer case_label="
                            + std::string(
                                    (ljSrCaseLabelEnv != nullptr && *ljSrCaseLabelEnv != '\0') ? ljSrCaseLabelEnv :
                                                                                                  "unknown")
                            + " execution_path=plain_cpu4x4_ref_kernel_local_energy_buffer kernel_type="
                            + std::string(nbnxmKernelTypeToName(kernelSetup.kernelType))
                            + " using_simd_kernel=false trace_role=contract_matched_kernel_local_lj_aggregate lj_sr="
                            + formatString("%.15f", readM2rPlain4x4KernelLocalAggregateTrace()));
        }
        if (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0')
        {
            const double             rawPostWriteLjSrTotal = sumKernelStoredEnergyOutputs(nbat, pairlists.ssize(), true);
            const auto               rawLjReadTrace        = traceKernelEnergyOutputs(nbat, pairlists.ssize(), true);
            const auto               rawCoulReadTrace      = traceKernelEnergyOutputs(nbat, pairlists.ssize(), false);
            const double             plainPatchContractReplayTotal = readM2pPlain4x4CoulombContractReplayTotal();
            const double             rawLjSrTotal          = rawLjReadTrace.finalTotal;
            const double             rawCoulSrTotal        = rawCoulReadTrace.finalTotal;
            const std::string        caseLabel =
                    (ljSrCaseLabelEnv != nullptr && *ljSrCaseLabelEnv != '\0') ? ljSrCaseLabelEnv :
                    ((m2pCaseLabelEnv != nullptr && *m2pCaseLabelEnv != '\0') ? m2pCaseLabelEnv : "unknown");
            const std::string        kernelType = nbnxmKernelTypeToName(kernelSetup.kernelType);
            const std::string        simdLabel  = usingSimdKernel ? "true" : "false";
            if ((dumpM2vTrace || dumpM2wTrace) && kernelSetup.kernelType == NbnxmKernelType::Cpu4x4_PlainC)
            {
                const auto alignedEventTotals =
                        dumpM2wTrace ? readM2wPlain4x4AlignedEventTotals() : readM2vPlain4x4AlignedEventTotals();
                if (!alignedEventTotals.empty())
                {
                    for (std::size_t eventIndex = 0; eventIndex < alignedEventTotals.size(); ++eventIndex)
                    {
                        appendM2pTraceTextLine(
                                ljSrTraceDirPath,
                                "step0_lj_sr_internal_trace.txt",
                                "stage=ALIGNED_WRITE_EVENT_" + std::to_string(eventIndex + 1)
                                        + " code_location=src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:after_plain_pair_energy_event case_label="
                                        + caseLabel
                                        + " execution_path=plain_aligned_pair_energy_event kernel_type="
                                        + kernelType + " using_simd_kernel=" + simdLabel
                                        + " aligned_contract=running_total_after_admitted_pair_energy_event aligned_event_ordinal="
                                        + std::to_string(eventIndex + 1) + " lj_sr="
                                        + formatString("%.15f", alignedEventTotals[eventIndex]));
                    }
                    appendM2pTraceTextLine(
                            ljSrTraceDirPath,
                            "step0_lj_sr_internal_trace.txt",
                            "stage=ALIGNED_LAST_EVENT_BEFORE_RAW_POST_WRITE code_location=src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:after_plain_last_pair_energy_event case_label="
                                    + caseLabel
                                    + " execution_path=plain_aligned_pair_energy_after_last_event kernel_type="
                                    + kernelType + " using_simd_kernel=" + simdLabel
                                    + " aligned_contract=running_total_after_admitted_pair_energy_event aligned_event_ordinal="
                                    + std::to_string(alignedEventTotals.size()) + " lj_sr="
                                    + formatString("%.15f", alignedEventTotals.back()));
                }
                appendM2pTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_lj_sr_internal_trace.txt",
                        "stage=RAW_POST_WRITE_EQUIVALENT code_location=src/gromacs/nbnxm/kerneldispatch.cpp:plain_output_buffer_post_kernel case_label="
                                + caseLabel + " execution_path=plain_aligned_post_write_equivalent kernel_type="
                                + kernelType + " using_simd_kernel=" + simdLabel
                                + " trace_role=post_aligned_event_target_state lj_sr="
                                + formatString("%.15f", rawPostWriteLjSrTotal));
                if (rawLjReadTrace.firstCaptured)
                {
                    appendM2pTraceTextLine(
                            ljSrTraceDirPath,
                            "step0_lj_sr_internal_trace.txt",
                            "stage=RAW_FIRST_READ_OR_REDUCE code_location=src/gromacs/nbnxm/kerneldispatch.cpp:sumKernelEnergyOutputs_first_read case_label="
                                    + caseLabel
                                    + " execution_path=plain_sumKernelEnergyOutputs_first_read kernel_type="
                                    + kernelType + " using_simd_kernel=" + simdLabel
                                    + " trace_role=first_reducer_read_partial_total lj_sr="
                                    + formatString("%.15f", rawLjReadTrace.firstReadTotal));
                }
                appendM2pTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_lj_sr_internal_trace.txt",
                        "stage=RAW_POST_READ_OR_REDUCE code_location=src/gromacs/nbnxm/kerneldispatch.cpp:sumKernelEnergyOutputs_final_total case_label="
                                + caseLabel + " execution_path=plain_sumKernelEnergyOutputs_post_read kernel_type="
                                + kernelType + " using_simd_kernel=" + simdLabel
                                + " trace_role=post_reducer_total lj_sr="
                                + formatString("%.15f", rawLjReadTrace.finalTotal));
            }
            else if ((dumpM2sTrace || dumpM2uTrace) && kernelSetup.kernelType == NbnxmKernelType::Cpu4x4_PlainC)
            {
                const auto writeOrdinalTotals = readM2uPlain4x4WriteOrdinalTotals();
                appendM2pTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_lj_sr_internal_trace.txt",
                        "stage=RAW_PRE_TRANSFER code_location=src/gromacs/nbnxm/kerneldispatch.cpp:plain_pre_output_buffer_transfer case_label="
                                + caseLabel
                                + " execution_path=plain_cpu4x4_ref_kernel_pre_transfer kernel_type=" + kernelType
                                + " using_simd_kernel=" + simdLabel
                                + " trace_role=source_aggregate_before_first_output_buffer_write lj_sr="
                                + formatString("%.15f", readM2rPlain4x4KernelLocalAggregateTrace()));
                if (dumpM2uTrace && !writeOrdinalTotals.empty())
                {
                    appendM2pTraceTextLine(
                            ljSrTraceDirPath,
                            "step0_lj_sr_internal_trace.txt",
                            "stage=RAW_FIRST_WRITE code_location=src/gromacs/nbnxm/kernels_reference/kernel_ref_outer.h:after_outputBuffer_Vvdw_write_ordinal_1 case_label="
                                    + caseLabel
                                    + " execution_path=plain_outputBuffer_after_write_ordinal kernel_type="
                                    + kernelType + " using_simd_kernel=" + simdLabel
                                    + " target_container=outputBuffer.Vvdw output_buffer_count="
                                    + std::to_string(pairlists.ssize())
                                    + " trace_role=running_total_after_write_ordinal write_ordinal=1 lj_sr="
                                    + formatString("%.15f", writeOrdinalTotals.front()));
                    for (std::size_t ordinalIndex = 1; ordinalIndex < writeOrdinalTotals.size(); ++ordinalIndex)
                    {
                        appendM2pTraceTextLine(
                                ljSrTraceDirPath,
                                "step0_lj_sr_internal_trace.txt",
                                "stage=AFTER_WRITE_ORDINAL_" + std::to_string(ordinalIndex + 1)
                                        + " code_location=src/gromacs/nbnxm/kernels_reference/kernel_ref_outer.h:after_outputBuffer_Vvdw_write_ordinal case_label="
                                        + caseLabel
                                        + " execution_path=plain_outputBuffer_after_write_ordinal kernel_type="
                                        + kernelType + " using_simd_kernel=" + simdLabel
                                        + " target_container=outputBuffer.Vvdw output_buffer_count="
                                        + std::to_string(pairlists.ssize())
                                        + " trace_role=running_total_after_write_ordinal write_ordinal="
                                        + std::to_string(ordinalIndex + 1) + " lj_sr="
                                        + formatString("%.15f", writeOrdinalTotals[ordinalIndex]));
                    }
                    appendM2pTraceTextLine(
                            ljSrTraceDirPath,
                            "step0_lj_sr_internal_trace.txt",
                            "stage=AFTER_LAST_WRITE_BEFORE_RAW_POST_WRITE code_location=src/gromacs/nbnxm/kernels_reference/kernel_ref_outer.h:after_outputBuffer_Vvdw_last_write case_label="
                                    + caseLabel
                                    + " execution_path=plain_outputBuffer_after_last_write kernel_type="
                                    + kernelType + " using_simd_kernel=" + simdLabel
                                    + " target_container=outputBuffer.Vvdw output_buffer_count="
                                    + std::to_string(pairlists.ssize())
                                    + " trace_role=running_total_after_last_write_before_raw_post_write write_ordinal="
                                    + std::to_string(writeOrdinalTotals.size()) + " lj_sr="
                                    + formatString("%.15f", writeOrdinalTotals.back()));
                }
                else if (m2sPlain4x4FirstWriteCaptured())
                {
                    appendM2pTraceTextLine(
                            ljSrTraceDirPath,
                            "step0_lj_sr_internal_trace.txt",
                            "stage=RAW_FIRST_WRITE code_location=src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:first_output_buffer_mutation case_label="
                                    + caseLabel
                                    + " execution_path=plain_cpu4x4_ref_kernel_first_output_write kernel_type="
                                    + kernelType + " using_simd_kernel=" + simdLabel
                                    + " trace_role=first_output_buffer_write_target lj_sr="
                                    + formatString("%.15f", readM2sPlain4x4FirstWriteTargetTotal()));
                }
                appendM2pTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_lj_sr_internal_trace.txt",
                        "stage=RAW_POST_WRITE code_location=src/gromacs/nbnxm/kerneldispatch.cpp:plain_output_buffer_post_kernel case_label="
                                + caseLabel + " execution_path=plain_output_buffer_after_kernel_write kernel_type="
                                + kernelType + " using_simd_kernel=" + simdLabel
                                + " target_container=outputBuffer.Vvdw output_buffer_count="
                                + std::to_string(pairlists.ssize()) + " write_count="
                                + std::to_string(writeOrdinalTotals.size())
                                + " trace_role=post_write_target_state lj_sr="
                                + formatString("%.15f", rawPostWriteLjSrTotal));
                if (rawLjReadTrace.firstCaptured)
                {
                    appendM2pTraceTextLine(
                            ljSrTraceDirPath,
                            "step0_lj_sr_internal_trace.txt",
                            "stage=RAW_FIRST_READ_OR_REDUCE code_location=src/gromacs/nbnxm/kerneldispatch.cpp:sumKernelEnergyOutputs_first_read case_label="
                                    + caseLabel
                                    + " execution_path=plain_sumKernelEnergyOutputs_first_read kernel_type="
                                    + kernelType + " using_simd_kernel=" + simdLabel
                                    + " trace_role=first_reducer_read_partial_total lj_sr="
                                    + formatString("%.15f", rawLjReadTrace.firstReadTotal));
                }
                appendM2pTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_lj_sr_internal_trace.txt",
                        "stage=RAW_POST_READ_OR_REDUCE code_location=src/gromacs/nbnxm/kerneldispatch.cpp:sumKernelEnergyOutputs_final_total case_label="
                                + caseLabel + " execution_path=plain_sumKernelEnergyOutputs_post_read kernel_type="
                                + kernelType + " using_simd_kernel=" + simdLabel
                                + " trace_role=post_reducer_total lj_sr="
                                + formatString("%.15f", rawLjReadTrace.finalTotal));
            }
            appendM2pTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_lj_sr_internal_trace.txt",
                    "stage=RAW_SR_FORMATION code_location=src/gromacs/nbnxm/kerneldispatch.cpp:430 case_label="
                            + caseLabel + " execution_path=plain_nbnxm_cpu kernel_type=" + kernelType
                            + " using_simd_kernel=" + simdLabel
                            + " trace_role=thread_output_pre_reduce lj_sr="
                            + formatString("%.15f", rawLjSrTotal) + " coulomb_sr="
                            + formatString("%.15f", rawCoulSrTotal));
            appendM2pTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_coulomb_source_truth_trace.txt",
                    "side=PLAIN variable=rawCoulReadTrace.finalTotal role=plain_comparable_coulomb_source before=0.000000000000000 delta="
                            + formatString("%.15f", rawCoulSrTotal) + " after="
                            + formatString("%.15f", rawCoulSrTotal)
                            + " code_location=src/gromacs/nbnxm/kerneldispatch.cpp:thread_output_pre_reduce");
            const double plainPatchLjContractReplayTotal = readM2pPlain4x4LjContractReplayTotal();
            appendM2pTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_lj_source_truth_trace.txt",
                    "side=PLAIN variable=rawLjReadTrace.finalTotal role=plain_native_lj_sr_source before=0.000000000000000 delta="
                            + formatString("%.15f", rawLjSrTotal) + " after="
                            + formatString("%.15f", rawLjSrTotal)
                            + " code_location=src/gromacs/nbnxm/kerneldispatch.cpp:thread_output_pre_reduce");
            appendM2pTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_lj_source_truth_trace.txt",
                    "side=PLAIN variable=plainPatchLjContractReplay.finalTotal role=plain_shadow_patch_lj_contract_source before=0.000000000000000 delta="
                            + formatString("%.15f", plainPatchLjContractReplayTotal) + " after="
                            + formatString("%.15f", plainPatchLjContractReplayTotal)
                            + " code_location=src/gromacs/nbnxm/kerneldispatch.cpp:thread_output_pre_reduce");
            appendM2pTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_coulomb_source_truth_trace.txt",
                    "side=PLAIN variable=plainPatchContractReplay.finalTotal role=plain_shadow_patch_contract_source before=0.000000000000000 delta="
                            + formatString("%.15f", plainPatchContractReplayTotal) + " after="
                            + formatString("%.15f", plainPatchContractReplayTotal)
                            + " code_location=src/gromacs/nbnxm/kerneldispatch.cpp:thread_output_pre_reduce");
            if (dumpMultiStepCoulombStateTrace)
            {
                const double plainPatchContractReplayPairOnlyTotal =
                        readM2pPlain4x4CoulombContractReplayPairOnlyTotal();
                appendM2pTraceTextLine(
                        ljSrTraceDirPath,
                        "multistep_coulomb_state_trace.txt",
                        "side=PLAIN step=" + std::to_string(currentTraceStep)
                                + " code_location=src/gromacs/nbnxm/kerneldispatch.cpp:thread_output_pre_reduce"
                                + " compute_energy=true"
                                + " plain_native_coul_total_before_step="
                                + formatString("%.15f", plainNativeCoulBeforeStep)
                                + " plain_native_coul_final=" + formatString("%.15f", rawCoulSrTotal)
                                + " plain_replay_coul_total_before_step=0.000000000000000"
                                + " plain_replay_coul_total_after_pairs="
                                + formatString("%.15f", plainPatchContractReplayPairOnlyTotal)
                                + " plain_replay_coul_total_after_excluded="
                                + formatString("%.15f", plainPatchContractReplayPairOnlyTotal)
                                + " plain_replay_coul_total_after_self="
                                + formatString("%.15f", plainPatchContractReplayTotal)
                                + " plain_replay_coul_final="
                                + formatString("%.15f", plainPatchContractReplayTotal));
            }
            appendM2pTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_coulomb_sr_component_trace.txt",
                    "stage=PRE_SR_ACCUMULATION_COMPARABLE code_location=src/gromacs/nbnxm/kerneldispatch.cpp:thread_output_pre_reduce case_label="
                            + caseLabel + " execution_path=plain_nbnxm_cpu kernel_type=" + kernelType
                            + " using_simd_kernel=" + simdLabel + " plain_coulomb_sr="
                            + formatString("%.15f", rawCoulSrTotal));
        }

        reduce_energies_over_lists(nbat, pairlists.ssize(), vVdw, vCoulomb);
    }
}

static void accountFlops(t_nrnb*                    nrnb,
                         const PairlistSet&         pairlistSet,
                         const nonbonded_verlet_t&  nbv,
                         const interaction_const_t& ic,
                         const gmx::StepWorkload&   stepWork)
{
    const bool usingGpuKernels = nbv.useGpu();

    int enr_nbnxn_kernel_ljc = eNRNB;
    if (usingRF(ic.coulomb.type) || ic.coulomb.type == CoulombInteractionType::Cut)
    {
        enr_nbnxn_kernel_ljc = eNR_NBNXN_LJ_RF;
    }
    else if ((!usingGpuKernels && nbv.kernelSetup().ewaldExclusionType == EwaldExclusionType::Analytical)
             || (usingGpuKernels && gpu_is_kernel_ewald_analytical(nbv.gpuNbv())))
    {
        enr_nbnxn_kernel_ljc = eNR_NBNXN_LJ_EWALD;
    }
    else
    {
        enr_nbnxn_kernel_ljc = eNR_NBNXN_LJ_TAB;
    }
    int enr_nbnxn_kernel_lj = eNR_NBNXN_LJ;
    if (stepWork.computeEnergy)
    {
        /* In eNR_??? the nbnxn F+E kernels are always the F kernel + 1 */
        enr_nbnxn_kernel_ljc += 1;
        enr_nbnxn_kernel_lj += 1;
    }

    inc_nrnb(nrnb, enr_nbnxn_kernel_ljc, pairlistSet.natpair_ljq_);
    inc_nrnb(nrnb, enr_nbnxn_kernel_lj, pairlistSet.natpair_lj_);
    /* The Coulomb-only kernels are offset -eNR_NBNXN_LJ_RF+eNR_NBNXN_RF */
    inc_nrnb(nrnb, enr_nbnxn_kernel_ljc - eNR_NBNXN_LJ_RF + eNR_NBNXN_RF, pairlistSet.natpair_q_);

    if (ic.vdw.modifier == InteractionModifiers::ForceSwitch)
    {
        /* We add up the switch cost separately */
        inc_nrnb(nrnb,
                 eNR_NBNXN_ADD_LJ_FSW + (stepWork.computeEnergy ? 1 : 0),
                 pairlistSet.natpair_ljq_ + pairlistSet.natpair_lj_);
    }
    if (ic.vdw.modifier == InteractionModifiers::PotSwitch)
    {
        /* We add up the switch cost separately */
        inc_nrnb(nrnb,
                 eNR_NBNXN_ADD_LJ_PSW + (stepWork.computeEnergy ? 1 : 0),
                 pairlistSet.natpair_ljq_ + pairlistSet.natpair_lj_);
    }
    if (ic.vdw.type == VanDerWaalsType::Pme)
    {
        /* We add up the LJ Ewald cost separately */
        inc_nrnb(nrnb,
                 eNR_NBNXN_ADD_LJ_EWALD + (stepWork.computeEnergy ? 1 : 0),
                 pairlistSet.natpair_ljq_ + pairlistSet.natpair_lj_);
    }
}

class ExactRespaCpuLaunchGuard
{
public:
    ExactRespaCpuLaunchGuard(const interaction_const_t& ic, const StepWorkload& stepWork) : ic_(ic)
    {
        previousActive_       = ic_.exactRespaCpuPairSplit.active;
        previousNativeMultiActive_ = ic_.exactRespaCpuPairSplit.nativeMultiActive;
        previousContribution_ = ic_.exactRespaCpuPairSplit.contribution;
        previousNativeMultiContributionCount_ =
                ic_.exactRespaCpuPairSplit.nativeMultiContributionCount;
        previousNativeMultiContributions_ = ic_.exactRespaCpuPairSplit.nativeMultiContributions;

        const bool useSplitLaunch =
                stepWork.nonbondedRespaContribution != MtsNonbondedRespaContribution::Full;
        if (useSplitLaunch)
        {
            GMX_RELEASE_ASSERT(ic_.exactRespaCpuPairSplit.configured,
                               "Exact r-RESPA CPU NBNXM launches require configured split metadata");
            GMX_RELEASE_ASSERT(ic_.vdw.type == VanDerWaalsType::Cut
                                       && ic_.vdw.modifier == InteractionModifiers::None
                                       && usingPmeOrEwald(ic_.coulomb.type)
                                       && ic_.coulomb.modifier == InteractionModifiers::None,
                               "Exact r-RESPA CPU NBNXM launches support only cut-off LJ with PME/Ewald Coulomb and no real-space modifiers");
        }

        ic_.exactRespaCpuPairSplit.active       = useSplitLaunch;
        ic_.exactRespaCpuPairSplit.nativeMultiActive = false;
        ic_.exactRespaCpuPairSplit.contribution = stepWork.nonbondedRespaContribution;
        ic_.exactRespaCpuPairSplit.nativeMultiContributionCount = 0;
        ic_.exactRespaCpuPairSplit.nativeMultiContributions.fill(
                MtsNonbondedRespaContribution::Full);
    }

    ExactRespaCpuLaunchGuard(const interaction_const_t& ic,
                             gmx::ArrayRef<const MtsNonbondedRespaContribution> contributions) :
        ic_(ic)
    {
        previousActive_       = ic_.exactRespaCpuPairSplit.active;
        previousNativeMultiActive_ = ic_.exactRespaCpuPairSplit.nativeMultiActive;
        previousContribution_ = ic_.exactRespaCpuPairSplit.contribution;
        previousNativeMultiContributionCount_ =
                ic_.exactRespaCpuPairSplit.nativeMultiContributionCount;
        previousNativeMultiContributions_ = ic_.exactRespaCpuPairSplit.nativeMultiContributions;

        GMX_RELEASE_ASSERT(ic_.exactRespaCpuPairSplit.configured,
                           "Exact r-RESPA CPU native multi-contribution launches require configured split metadata");
        GMX_RELEASE_ASSERT(contributions.size() > 1,
                           "Exact r-RESPA CPU native multi-contribution launches require at least two contributions");
        GMX_RELEASE_ASSERT(
                contributions.size() <= interaction_const_t::c_maxExactRespaNativeMultiContributions,
                "Exact r-RESPA CPU native multi-contribution launch exceeds the compiled contribution bound");
        GMX_RELEASE_ASSERT(ic_.vdw.type == VanDerWaalsType::Cut
                                   && ic_.vdw.modifier == InteractionModifiers::None
                                   && usingPmeOrEwald(ic_.coulomb.type)
                                   && ic_.coulomb.modifier == InteractionModifiers::None,
                           "Exact r-RESPA CPU native multi-contribution launches support only cut-off LJ with PME/Ewald Coulomb and no real-space modifiers");

        ic_.exactRespaCpuPairSplit.active                     = true;
        ic_.exactRespaCpuPairSplit.nativeMultiActive          = true;
        ic_.exactRespaCpuPairSplit.contribution               = MtsNonbondedRespaContribution::Full;
        ic_.exactRespaCpuPairSplit.nativeMultiContributionCount = contributions.size();
        ic_.exactRespaCpuPairSplit.nativeMultiContributions.fill(
                MtsNonbondedRespaContribution::Full);
        for (int contributionIndex = 0; contributionIndex < contributions.ssize(); ++contributionIndex)
        {
            ic_.exactRespaCpuPairSplit.nativeMultiContributions[contributionIndex] =
                    contributions[contributionIndex];
        }
    }

    ~ExactRespaCpuLaunchGuard()
    {
        ic_.exactRespaCpuPairSplit.active       = previousActive_;
        ic_.exactRespaCpuPairSplit.nativeMultiActive = previousNativeMultiActive_;
        ic_.exactRespaCpuPairSplit.contribution = previousContribution_;
        ic_.exactRespaCpuPairSplit.nativeMultiContributionCount =
                previousNativeMultiContributionCount_;
        ic_.exactRespaCpuPairSplit.nativeMultiContributions = previousNativeMultiContributions_;
    }

private:
    const interaction_const_t&           ic_;
    bool                                 previousActive_ = false;
    bool                                 previousNativeMultiActive_ = false;
    MtsNonbondedRespaContribution        previousContribution_ =
            MtsNonbondedRespaContribution::Full;
    int                                  previousNativeMultiContributionCount_ = 0;
    std::array<MtsNonbondedRespaContribution,
               interaction_const_t::c_maxExactRespaNativeMultiContributions>
            previousNativeMultiContributions_ = { MtsNonbondedRespaContribution::Full,
                                                 MtsNonbondedRespaContribution::Full,
                                                 MtsNonbondedRespaContribution::Full };
};

void nonbonded_verlet_t::dispatchNonbondedKernel(gmx::InteractionLocality       iLocality,
                                                 const interaction_const_t&     ic,
                                                 const gmx::StepWorkload&       stepWork,
                                                 int                            clearF,
                                                 gmx::ArrayRef<const gmx::RVec> shiftvec,
                                                 gmx::ArrayRef<real> repulsionDispersionSR,
                                                 gmx::ArrayRef<real> CoulombSR,
                                                 t_nrnb*             nrnb) const
{
    const PairlistSet& pairlistSet = pairlistSets().pairlistSet(iLocality);
    const ExactRespaCpuLaunchGuard exactRespaCpuLaunchGuard(ic, stepWork);

    switch (kernelSetup().kernelType)
    {
        case NbnxmKernelType::Cpu4x4_PlainC:
        case NbnxmKernelType::Cpu4xN_Simd_4xN:
        case NbnxmKernelType::Cpu4xN_Simd_2xNN:
        case NbnxmKernelType::Cpu1x1_PlainC:
            nbnxn_kernel_cpu(pairlistSet,
                             kernelSetup(),
                             nbat_.get(),
                             ic,
                             shiftvec,
                             stepWork,
                             clearF,
                             CoulombSR.data(),
                             repulsionDispersionSR.data(),
                             wcycle_);
            break;

        case NbnxmKernelType::Gpu8x8x8:
            if (stepWork.nonbondedRespaContribution != MtsNonbondedRespaContribution::Full)
            {
                GMX_RELEASE_ASSERT(ic.vdw.type == VanDerWaalsType::Cut
                                           && ic.vdw.modifier == InteractionModifiers::None
                                           && usingPmeOrEwald(ic.coulomb.type)
                                           && ic.coulomb.modifier == InteractionModifiers::None,
                                   "GPU NBNXM exact LAMMPS-style r-RESPA launches currently "
                                   "support only cut-off LJ with PME/Ewald Coulomb and no real-space modifiers");
            }
            gpu_launch_kernel(gpuNbv_, stepWork, iLocality);
            break;

        case NbnxmKernelType::Cpu8x8x8_PlainC:
            GMX_RELEASE_ASSERT(
                    stepWork.nonbondedRespaContribution == MtsNonbondedRespaContribution::Full,
                    "GPU-reference NBNXM does not yet support exact LAMMPS-style r-RESPA per-contribution launches");
            nbnxn_kernel_gpu_ref(pairlistSet.gpuList(),
                                 nbat_.get(),
                                 ic,
                                 shiftvec,
                                 stepWork,
                                 clearF,
                                 nbat_->outputBuffer(0).f,
                                 nbat_->outputBuffer(0).fshift.data(),
                                 CoulombSR.data(),
                                 repulsionDispersionSR.data());
            break;

        default: GMX_RELEASE_ASSERT(false, "Invalid nonbonded kernel type passed!");
    }

    if (nrnb)
    {
        accountFlops(nrnb, pairlistSet, *this, ic, stepWork);
    }
}

void nonbonded_verlet_t::dispatchExactRespaCpuNativeMultiKernel(
        gmx::InteractionLocality                    iLocality,
        const interaction_const_t&                 ic,
        gmx::ArrayRef<const MtsNonbondedRespaContribution> contributions,
        const gmx::StepWorkload&                   stepWork,
        int                                        clearF,
        gmx::ArrayRef<const gmx::RVec>             shiftvec,
        gmx::ArrayRef<real>                        repulsionDispersionSR,
        gmx::ArrayRef<real>                        CoulombSR,
        t_nrnb*                                    nrnb) const
{
    GMX_RELEASE_ASSERT(exactRespaCpuNbnxmKernelSupported(kernelSetup().kernelType),
                       "Exact r-RESPA CPU native multi-contribution dispatch requires a CPU NBNXM kernel");
    GMX_RELEASE_ASSERT(!contributions.empty(),
                       "Exact r-RESPA CPU native multi-contribution dispatch requires at least one contribution");
    GMX_RELEASE_ASSERT(stepWork.computeForces,
                       "Exact r-RESPA CPU native multi-contribution dispatch requires force computation");
    GMX_RELEASE_ASSERT(!stepWork.computeEnergy
                               || (!repulsionDispersionSR.empty() && !CoulombSR.empty()),
                       "Exact r-RESPA CPU native multi-contribution dispatch requires real-space energy sinks on energy steps");
    nbat_->ensureNativeMultiContributionOutputBuffers(contributions.ssize());

    std::array<real, 1> dummyVdwEnergy = { 0.0_real };
    std::array<real, 1> dummyCoulEnergy = { 0.0_real };
    real*               coulombEnergy   =
            stepWork.computeEnergy ? CoulombSR.data() : dummyCoulEnergy.data();
    real* repulsionDispersionEnergy =
            stepWork.computeEnergy ? repulsionDispersionSR.data() : dummyVdwEnergy.data();
    const PairlistSet&  pairlistSet = pairlistSets().pairlistSet(iLocality);
    const ExactRespaCpuLaunchGuard exactRespaCpuLaunchGuard(ic, contributions);

    nbnxn_kernel_cpu(pairlistSet,
                     kernelSetup(),
                     nbat_.get(),
                     ic,
                     shiftvec,
                     stepWork,
                     clearF,
                     coulombEnergy,
                     repulsionDispersionEnergy,
                     wcycle_);

    if (nrnb)
    {
        accountFlops(nrnb, pairlistSet, *this, ic, stepWork);
    }
}

} // namespace gmx
