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

#include "kernel_ref_4x4.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <algorithm>

#include "../nbnxm_geometry.h"

#include "gromacs/math/functions.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/simd_energy_accumulator.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/vec.h"

namespace gmx
{

namespace
{

std::mutex m2qPlain4x4TraceMutex;
double     m2qPlain4x4EarliestRawLjTotal = 0.0;
bool       m2qPlain4x4TraceEnabled       = false;
std::mutex m2rPlain4x4TraceMutex;
double     m2rPlain4x4KernelLocalLjTotal = 0.0;
bool       m2rPlain4x4TraceEnabled       = false;
std::mutex m2sPlain4x4TraceMutex;
double     m2sPlain4x4FirstWriteLjTotal  = 0.0;
bool       m2sPlain4x4FirstWriteCapturedFlag = false;
bool       m2sPlain4x4TraceEnabled       = false;
std::mutex         m2uPlain4x4TraceMutex;
std::vector<double> m2uPlain4x4WriteOrdinalLjTotals;
bool               m2uPlain4x4TraceEnabled = false;
std::mutex         m2vPlain4x4TraceMutex;
std::vector<double> m2vPlain4x4AlignedEventLjTotals;
double             m2vPlain4x4AlignedEventLjRunningTotal = 0.0;
bool               m2vPlain4x4TraceEnabled = false;
std::mutex                        m2wPlain4x4TraceMutex;
std::vector<double>               m2wPlain4x4AlignedEventLjTotals;
std::vector<M2wPlain4x4AlignedEventRecord> m2wPlain4x4AlignedEventRecords;
double                            m2wPlain4x4AlignedEventLjRunningTotal = 0.0;
int                               m2wPlain4x4AlignedEventOrdinal        = 0;
bool                              m2wPlain4x4TraceEnabled               = false;
std::string                       m2wPlain4x4ClearedTracePath;
std::mutex                        m2xPlain4x4TraceMutex;
bool                              m2xPlain4x4TraceEnabled               = false;
int                               m2xPlain4x4AlignedEventOrdinal        = 0;
std::string                       m2xPlain4x4ClearedTracePath;
std::mutex                        m2pPlain4x4CoulombWriteTraceMutex;
int                               m2pPlain4x4CoulombWriteOrdinal        = 0;
std::string                       m2pPlain4x4CoulombWriteClearedTracePath;
std::mutex                        m2pPlain4x4CoulombProducerTraceMutex;
int                               m2pPlain4x4CoulombProducerOrdinal     = 0;
std::string                       m2pPlain4x4CoulombProducerClearedTracePath;
double                            m2pPlain4x4CoulombProducerPrefixSum   = 0.0;
std::mutex                        m2pPlain4x4MultiStepCoulombPairTraceMutex;
int                               m2pPlain4x4MultiStepCoulombPairOrdinal   = 0;
std::string                       m2pPlain4x4MultiStepCoulombPairClearedTracePath;
int64_t                           m2pPlain4x4MultiStepCoulombPairCurrentStep = -1;
std::vector<real>                 m2pPlain4x4MultiStepCoulombPairReplayTerms;
std::mutex                        m2pPlain4x4CoulombSelfTraceMutex;
int                               m2pPlain4x4CoulombSelfOrdinal         = 0;
std::string                       m2pPlain4x4CoulombSelfClearedTracePath;
double                            m2pPlain4x4CoulombSelfPrefixSum       = 0.0;
std::mutex                        m2pPlain4x4CoulombContractReplayMutex;
bool                              m2pPlain4x4CoulombContractReplayTraceEnabled = false;
std::vector<std::pair<int, real>> m2pPlain4x4CoulombContractReplayPairContributions;
std::vector<std::pair<int, real>> m2pPlain4x4CoulombContractReplaySelfContributions;
std::mutex                        m2pPlain4x4CurrentStepMutex;
int64_t                           m2pPlain4x4CurrentStep = -1;
std::mutex                        m2pPlain4x4LjContractReplayMutex;
bool                              m2pPlain4x4LjContractReplayTraceEnabled = false;
std::vector<real>                 m2pPlain4x4LjContractReplayPairContributions;
std::mutex                        m2pPlain4x4ExclusionEquivalenceTraceMutex;
std::mutex                        m2pPlain4x4RealspaceForceSubcomponentTraceMutex;
bool                              m2pPlain4x4RealspaceForceSubcomponentTraceEnabledFlag = false;
M2pPlain4x4TracedForcePair        m2pPlain4x4LjSrForcePair;
M2pPlain4x4TracedForcePair        m2pPlain4x4CoulombSrForcePair;
M2pPlain4x4TracedForcePair        m2pPlain4x4ExclusionCorrectionForcePair;
M2pPlain4x4TracedForcePair        m2pPlain4x4CombinedRealspaceForcePair;
std::mutex                        m2pPlain4x4PairTotalTraceMutex;
std::string                       m2pPlain4x4PairTotalClearedTracePath;
int64_t                           m2pPlain4x4PairTotalCurrentStep = -1;

const char* activeLjSrTraceDirPath()
{
    const char* traceDir = std::getenv("GMX_PCFF_RESPA_M2W_TRACE_DIR");
    if (traceDir != nullptr && *traceDir != '\0')
    {
        return traceDir;
    }
    traceDir = std::getenv("GMX_PCFF_RESPA_M2V_TRACE_DIR");
    if (traceDir != nullptr && *traceDir != '\0')
    {
        return traceDir;
    }
    traceDir = std::getenv("GMX_PCFF_RESPA_M2U_TRACE_DIR");
    if (traceDir != nullptr && *traceDir != '\0')
    {
        return traceDir;
    }
    traceDir = std::getenv("GMX_PCFF_RESPA_M2S_TRACE_DIR");
    if (traceDir != nullptr && *traceDir != '\0')
    {
        return traceDir;
    }
    traceDir = std::getenv("GMX_PCFF_RESPA_M2R_TRACE_DIR");
    if (traceDir != nullptr && *traceDir != '\0')
    {
        return traceDir;
    }
    traceDir = std::getenv("GMX_PCFF_RESPA_M2Q_TRACE_DIR");
    if (traceDir != nullptr && *traceDir != '\0')
    {
        return traceDir;
    }
    traceDir = std::getenv("GMX_PCFF_RESPA_M2P_TRACE_DIR");
    if (traceDir != nullptr && *traceDir != '\0')
    {
        return traceDir;
    }
    return nullptr;
}

bool activeLjSrOptionalTraceEnabled(const char* envVarName)
{
    const char* value = std::getenv(envVarName);
    return (value != nullptr && *value != '\0');
}

std::string m2pPlain4x4ExclusionEquivalenceTracePath()
{
    const char* traceDir = activeLjSrTraceDirPath();
    if (traceDir == nullptr || *traceDir == '\0')
    {
        return {};
    }
    return std::string(traceDir) + "/step0_exclusion_equivalence_pair_trace.txt";
}

std::string m2pPlain4x4Step2PairTotalTracePath()
{
    if (!activeLjSrOptionalTraceEnabled("GMX_PCFF_RESPA_TRACE_STEP1_SUBSET01_FORCEGROUP_AUDIT"))
    {
        return {};
    }
    const char* traceDir = activeLjSrTraceDirPath();
    if (traceDir == nullptr || *traceDir == '\0')
    {
        return {};
    }
    return std::string(traceDir) + "/step2_plain_pair_total_trace.txt";
}

void clearTracedForcePair(M2pPlain4x4TracedForcePair* pair)
{
    if (pair == nullptr)
    {
        return;
    }
    for (auto& atom : pair->atoms)
    {
        atom.fill(0.0);
    }
}

void addForceContributionToTrackedAtoms(M2pPlain4x4TracedForcePair* pair,
                                        int                         ai,
                                        int                         aj,
                                        real                        fx,
                                        real                        fy,
                                        real                        fz)
{
    if (pair == nullptr)
    {
        return;
    }

    const std::array<real, DIM> contribution = { fx, fy, fz };
    for (const auto [traceAtomIndex, atomIndex] : { std::pair<int, int>{ 0, 0 }, std::pair<int, int>{ 1, 5 } })
    {
        if (ai == atomIndex)
        {
            for (int dim = 0; dim < DIM; ++dim)
            {
                pair->atoms[traceAtomIndex][dim] += contribution[dim];
            }
        }
        if (aj == atomIndex)
        {
            for (int dim = 0; dim < DIM; ++dim)
            {
                pair->atoms[traceAtomIndex][dim] -= contribution[dim];
            }
        }
    }
}

std::string m2xPlain4x4GeometryTracePath()
{
    const char* traceDir = std::getenv("GMX_PCFF_RESPA_M2X_TRACE_DIR");
    if (traceDir == nullptr || *traceDir == '\0')
    {
        return {};
    }
    return std::string(traceDir) + "/step0_event_669_geometry_trace.txt";
}

void appendM2xPlain4x4GeometryStageLine(const char* stage,
                                        const M2xPlain4x4GeometryEventData& data,
                                        bool includeShiftedCoord,
                                        bool includeDx,
                                        bool includeRsq,
                                        bool includeR)
{
    const std::string tracePath = m2xPlain4x4GeometryTracePath();
    if (tracePath.empty())
    {
        return;
    }

    const char* caseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2X_CASE_LABEL");
    const char* caseLabel = (caseLabelEnv != nullptr && *caseLabelEnv != '\0') ? caseLabelEnv : "unknown";
    const char* codeLocation = "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:plain_event_669_geometry_trace";
    if (std::strcmp(stage, "GEOM_COORD_SOURCE") == 0)
    {
        codeLocation = "src/gromacs/nbnxm/kernels_reference/kernel_ref_outer.h:coord_source_before_shift";
    }
    else if (std::strcmp(stage, "GEOM_SHIFT_OR_PBC_APPLY") == 0)
    {
        codeLocation = "src/gromacs/nbnxm/kernels_reference/kernel_ref_outer.h:shiftvec_application_into_xi";
    }
    else if (std::strcmp(stage, "GEOM_DXDYDZ_CONSTRUCTION") == 0)
    {
        codeLocation = "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:dxdyzdz_construction";
    }
    else if (std::strcmp(stage, "GEOM_RSQ_FORMATION") == 0)
    {
        codeLocation = "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:rsq_formation_before_lj";
    }
    else if (std::strcmp(stage, "EVENT_669_LJ_INPUT") == 0)
    {
        codeLocation = "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:event_669_lj_input";
    }

    std::ostringstream line;
    line << std::fixed << std::setprecision(15);
    line << "stage=" << stage
         << " code_location=" << codeLocation
         << " case_label=" << caseLabel
         << " execution_path=plain_event_669_geometry_trace"
         << " aligned_contract=running_total_after_admitted_pair_energy_event"
         << " aligned_event_ordinal=669"
         << " pair_i=" << data.pairI
         << " pair_j=" << data.pairJ
         << " type_i=" << data.typeI
         << " type_j=" << data.typeJ
         << " event_ordering_key=" << data.pairI << "_" << data.pairJ
         << " ci_index=" << data.ciIndex
         << " cj_index=" << data.cjIndex
         << " i_index=" << data.iIndex
         << " j_index=" << data.jIndex
         << " shift_index=" << data.shiftIndex
         << " coord_i_x=" << data.coordISourceX
         << " coord_i_y=" << data.coordISourceY
         << " coord_i_z=" << data.coordISourceZ
         << " coord_j_x=" << data.coordJSourceX
         << " coord_j_y=" << data.coordJSourceY
         << " coord_j_z=" << data.coordJSourceZ
         << " shift_x=" << data.shiftX
         << " shift_y=" << data.shiftY
         << " shift_z=" << data.shiftZ;
    if (includeShiftedCoord)
    {
        line << " coord_i_shifted_x=" << data.coordIShiftedX
             << " coord_i_shifted_y=" << data.coordIShiftedY
             << " coord_i_shifted_z=" << data.coordIShiftedZ;
    }
    if (includeDx)
    {
        line << " dx=" << data.dx
             << " dy=" << data.dy
             << " dz=" << data.dz;
    }
    if (includeRsq)
    {
        line << " rsq=" << data.rsq;
    }
    if (includeR)
    {
        line << " r=" << data.r
             << " raw_lj_term=" << data.rawLjTerm
             << " final_event_lj_contribution=" << data.finalEventLj;
    }

    std::ofstream out(tracePath, std::ios::app);
    if (out)
    {
        out << line.str() << '\n';
    }
}

std::string m2wPlain4x4IdentityTracePath()
{
    const char* traceDir = std::getenv("GMX_PCFF_RESPA_M2W_TRACE_DIR");
    if (traceDir == nullptr || *traceDir == '\0')
    {
        return {};
    }
    return std::string(traceDir) + "/step0_aligned_event_identity_trace.txt";
}

void appendM2wPlain4x4IdentityTraceLine(const M2wPlain4x4AlignedEventRecord& record)
{
    const std::string tracePath = m2wPlain4x4IdentityTracePath();
    if (tracePath.empty())
    {
        return;
    }

    const char* caseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2W_CASE_LABEL");
    const char* caseLabel = (caseLabelEnv != nullptr && *caseLabelEnv != '\0') ? caseLabelEnv : "unknown";

    std::ostringstream line;
    line << std::fixed << std::setprecision(15);
    line << "stage=ALIGNED_WRITE_EVENT_" << record.alignedEventOrdinal
         << " code_location=src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:after_plain_pair_energy_event"
         << " case_label=" << caseLabel
         << " execution_path=plain_aligned_pair_energy_event"
         << " aligned_contract=running_total_after_admitted_pair_energy_event"
         << " aligned_event_ordinal=" << record.alignedEventOrdinal
         << " pair_i=" << record.pairI
         << " pair_j=" << record.pairJ
         << " type_i=" << record.typeI
         << " type_j=" << record.typeJ
         << " pair_ordinal=" << (record.alignedEventOrdinal - 1)
         << " event_ordering_key=" << record.pairI << "_" << record.pairJ
         << " ci_index=" << record.ciIndex
         << " cj_index=" << record.cjIndex
         << " i_index=" << record.iIndex
         << " j_index=" << record.jIndex
         << " running_total_before=" << record.runningTotalBefore
         << " raw_lj_term=" << record.rawLjTerm
         << " scaling_factor=" << record.scalingFactor
         << " final_event_lj_contribution=" << record.finalEventLj
         << " running_total_after=" << record.runningTotalAfter
         << " c6=" << record.c6
         << " c12=" << record.c12
         << " rsq=" << record.rsq
         << " r=" << record.r;

    std::ofstream out(tracePath, std::ios::app);
    if (out)
    {
        out << line.str() << '\n';
    }
}

std::string m2pPlain4x4CoulombFirstWriteTracePath()
{
    if (!activeLjSrOptionalTraceEnabled("GMX_PCFF_RESPA_TRACE_COULOMB_FIRST_WRITES"))
    {
        return {};
    }
    const char* traceDir = activeLjSrTraceDirPath();
    if (traceDir == nullptr || *traceDir == '\0')
    {
        return {};
    }
    return std::string(traceDir) + "/step0_excluded_coulomb_first_writes.txt";
}

std::string m2pPlain4x4CoulombProducerTracePath()
{
    const char* traceDir = activeLjSrTraceDirPath();
    if (traceDir == nullptr || *traceDir == '\0')
    {
        return {};
    }
    return std::string(traceDir) + "/step0_excluded_coulomb_detail_rows.txt";
}

std::string m2pPlain4x4CoulombPrefixTracePath()
{
    const char* traceDir = activeLjSrTraceDirPath();
    if (traceDir == nullptr || *traceDir == '\0')
    {
        return {};
    }
    return std::string(traceDir) + "/step0_excluded_coulomb_prefix_checkpoints.txt";
}

std::string m2pPlain4x4CoulombSelfDetailTracePath()
{
    const char* traceDir = activeLjSrTraceDirPath();
    if (traceDir == nullptr || *traceDir == '\0')
    {
        return {};
    }
    return std::string(traceDir) + "/step0_coulomb_self_detail_rows.txt";
}

std::string m2pPlain4x4CoulombSelfPrefixTracePath()
{
    const char* traceDir = activeLjSrTraceDirPath();
    if (traceDir == nullptr || *traceDir == '\0')
    {
        return {};
    }
    return std::string(traceDir) + "/step0_coulomb_self_prefix_checkpoints.txt";
}

std::string m2pPlain4x4MultiStepCoulombPairDetailTracePath()
{
    const char* traceDir = activeLjSrTraceDirPath();
    if (traceDir == nullptr || *traceDir == '\0')
    {
        return {};
    }
    return std::string(traceDir) + "/multistep_coulomb_pair_detail_rows.txt";
}

std::string m2pPlain4x4MultiStepCoulombPairPrefixTracePath()
{
    const char* traceDir = activeLjSrTraceDirPath();
    if (traceDir == nullptr || *traceDir == '\0')
    {
        return {};
    }
    return std::string(traceDir) + "/multistep_coulomb_pair_prefix_trace.txt";
}

std::vector<int> parseOrdinalList(const char* envName)
{
    const char* value = std::getenv(envName);
    if (value == nullptr || *value == '\0')
    {
        return {};
    }

    std::vector<int> result;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty())
        {
            result.push_back(std::stoi(item));
        }
    }
    return result;
}

const std::vector<int>& m2pPlain4x4CoulombPrefixCheckpoints()
{
    static const std::vector<int> checkpoints = parseOrdinalList("GMX_PCFF_RESPA_COULOMB_PREFIX_CHECKPOINTS");
    return checkpoints;
}

const std::vector<int>& m2pPlain4x4CoulombDetailOrdinals()
{
    static const std::vector<int> ordinals = parseOrdinalList("GMX_PCFF_RESPA_COULOMB_DETAIL_ORDINALS");
    return ordinals;
}

const std::vector<int>& m2pPlain4x4CoulombSelfPrefixCheckpoints()
{
    static const std::vector<int> checkpoints =
            parseOrdinalList("GMX_PCFF_RESPA_COULOMB_SELF_PREFIX_CHECKPOINTS");
    return checkpoints;
}

const std::vector<int>& m2pPlain4x4CoulombSelfDetailOrdinals()
{
    static const std::vector<int> ordinals =
            parseOrdinalList("GMX_PCFF_RESPA_COULOMB_SELF_DETAIL_ORDINALS");
    return ordinals;
}

const std::vector<int>& m2pPlain4x4MultiStepCoulombPairPrefixCheckpoints()
{
    static const std::vector<int> checkpoints =
            parseOrdinalList("GMX_PCFF_RESPA_MULTI_STEP_COULOMB_PAIR_PREFIX_CHECKPOINTS");
    return checkpoints;
}

const std::vector<int>& m2pPlain4x4MultiStepCoulombPairDetailOrdinals()
{
    static const std::vector<int> ordinals =
            parseOrdinalList("GMX_PCFF_RESPA_MULTI_STEP_COULOMB_PAIR_DETAIL_ORDINALS");
    return ordinals;
}

const std::vector<int64_t>& m2pPlain4x4MultiStepCoulombTraceSteps()
{
    static const std::vector<int64_t> steps = []()
    {
        std::vector<int64_t> result;
        const char*          value = std::getenv("GMX_PCFF_RESPA_TRACE_MULTI_STEP_COULOMB_STEPS");
        if (value == nullptr || *value == '\0')
        {
            return result;
        }

        std::stringstream ss(value);
        std::string       item;
        while (std::getline(ss, item, ','))
        {
            if (!item.empty())
            {
                result.push_back(std::stoll(item));
            }
        }
        return result;
    }();
    return steps;
}

bool m2pPlain4x4ShouldTraceMultiStepCoulombStep(int64_t step)
{
    const auto& steps = m2pPlain4x4MultiStepCoulombTraceSteps();
    return std::find(steps.begin(), steps.end(), step) != steps.end();
}

double sumMultiStepReplayTerms(const std::vector<real>& replayTerms)
{
    double total = 0.0;
    for (const real value : replayTerms)
    {
        total += value;
    }
    return total;
}

bool ordinalRequested(int ordinal, const std::vector<int>& ordinals)
{
    return std::find(ordinals.begin(), ordinals.end(), ordinal) != ordinals.end();
}

} // namespace

bool m2qPlain4x4EarliestRawTraceEnabled()
{
    return m2qPlain4x4TraceEnabled;
}

void resetM2qPlain4x4EarliestRawTrace()
{
    const char* traceDir = activeLjSrTraceDirPath();
    m2qPlain4x4TraceEnabled = (traceDir != nullptr && *traceDir != '\0');
    std::lock_guard<std::mutex> guard(m2qPlain4x4TraceMutex);
    m2qPlain4x4EarliestRawLjTotal = 0.0;
}

void accumulateM2qPlain4x4EarliestRawTrace(real ljEnergy)
{
    if (!m2qPlain4x4TraceEnabled)
    {
        return;
    }
    std::lock_guard<std::mutex> guard(m2qPlain4x4TraceMutex);
    m2qPlain4x4EarliestRawLjTotal += ljEnergy;
}

real readM2qPlain4x4EarliestRawTrace()
{
    std::lock_guard<std::mutex> guard(m2qPlain4x4TraceMutex);
    return m2qPlain4x4EarliestRawLjTotal;
}

bool m2rPlain4x4AmplificationTraceEnabled()
{
    return m2rPlain4x4TraceEnabled;
}

void resetM2rPlain4x4AmplificationTrace()
{
    const char* traceDir = activeLjSrTraceDirPath();
    m2rPlain4x4TraceEnabled = (traceDir != nullptr && *traceDir != '\0');
    std::lock_guard<std::mutex> guard(m2rPlain4x4TraceMutex);
    m2rPlain4x4KernelLocalLjTotal = 0.0;
}

void accumulateM2rPlain4x4KernelLocalAggregateTrace(real ljEnergy)
{
    if (!m2rPlain4x4TraceEnabled)
    {
        return;
    }
    std::lock_guard<std::mutex> guard(m2rPlain4x4TraceMutex);
    m2rPlain4x4KernelLocalLjTotal += ljEnergy;
}

real readM2rPlain4x4KernelLocalAggregateTrace()
{
    std::lock_guard<std::mutex> guard(m2rPlain4x4TraceMutex);
    return m2rPlain4x4KernelLocalLjTotal;
}

bool m2sPlain4x4InternalTraceEnabled()
{
    return m2sPlain4x4TraceEnabled;
}

void resetM2sPlain4x4InternalTrace()
{
    const char* traceDir = activeLjSrTraceDirPath();
    m2sPlain4x4TraceEnabled = (traceDir != nullptr && *traceDir != '\0');
    std::lock_guard<std::mutex> guard(m2sPlain4x4TraceMutex);
    m2sPlain4x4FirstWriteLjTotal  = 0.0;
    m2sPlain4x4FirstWriteCapturedFlag = false;
}

void noteM2sPlain4x4FirstWriteTargetTotal(const real* values, int count)
{
    if (!m2sPlain4x4TraceEnabled)
    {
        return;
    }
    std::lock_guard<std::mutex> guard(m2sPlain4x4TraceMutex);
    if (m2sPlain4x4FirstWriteCapturedFlag)
    {
        return;
    }
    double total = 0.0;
    for (int i = 0; i < count; ++i)
    {
        total += values[i];
    }
    m2sPlain4x4FirstWriteLjTotal  = total;
    m2sPlain4x4FirstWriteCapturedFlag = true;
}

real readM2sPlain4x4FirstWriteTargetTotal()
{
    std::lock_guard<std::mutex> guard(m2sPlain4x4TraceMutex);
    return m2sPlain4x4FirstWriteLjTotal;
}

bool m2sPlain4x4FirstWriteCaptured()
{
    std::lock_guard<std::mutex> guard(m2sPlain4x4TraceMutex);
    return m2sPlain4x4FirstWriteCapturedFlag;
}

bool m2uPlain4x4WriteOrdinalTraceEnabled()
{
    return m2uPlain4x4TraceEnabled;
}

void resetM2uPlain4x4WriteOrdinalTrace()
{
    const char* traceDir = std::getenv("GMX_PCFF_RESPA_M2U_TRACE_DIR");
    m2uPlain4x4TraceEnabled = (traceDir != nullptr && *traceDir != '\0');
    std::lock_guard<std::mutex> guard(m2uPlain4x4TraceMutex);
    m2uPlain4x4WriteOrdinalLjTotals.clear();
}

void noteM2uPlain4x4WriteTargetTotal(const real* values, int count)
{
    if (!m2uPlain4x4TraceEnabled)
    {
        return;
    }
    double total = 0.0;
    for (int i = 0; i < count; ++i)
    {
        total += values[i];
    }
    std::lock_guard<std::mutex> guard(m2uPlain4x4TraceMutex);
    m2uPlain4x4WriteOrdinalLjTotals.push_back(total);
}

std::vector<double> readM2uPlain4x4WriteOrdinalTotals()
{
    std::lock_guard<std::mutex> guard(m2uPlain4x4TraceMutex);
    return m2uPlain4x4WriteOrdinalLjTotals;
}

bool m2vPlain4x4AlignedEventTraceEnabled()
{
    return m2vPlain4x4TraceEnabled;
}

void resetM2vPlain4x4AlignedEventTrace()
{
    const char* traceDir = std::getenv("GMX_PCFF_RESPA_M2V_TRACE_DIR");
    m2vPlain4x4TraceEnabled = (traceDir != nullptr && *traceDir != '\0');
    std::lock_guard<std::mutex> guard(m2vPlain4x4TraceMutex);
    m2vPlain4x4AlignedEventLjTotals.clear();
    m2vPlain4x4AlignedEventLjRunningTotal = 0.0;
}

void noteM2vPlain4x4AlignedEvent(real ljEnergyDelta)
{
    if (!m2vPlain4x4TraceEnabled)
    {
        return;
    }
    std::lock_guard<std::mutex> guard(m2vPlain4x4TraceMutex);
    m2vPlain4x4AlignedEventLjRunningTotal += ljEnergyDelta;
    m2vPlain4x4AlignedEventLjTotals.push_back(m2vPlain4x4AlignedEventLjRunningTotal);
}

std::vector<double> readM2vPlain4x4AlignedEventTotals()
{
    std::lock_guard<std::mutex> guard(m2vPlain4x4TraceMutex);
    return m2vPlain4x4AlignedEventLjTotals;
}

bool m2wPlain4x4AlignedEventTraceEnabled()
{
    return m2wPlain4x4TraceEnabled;
}

bool m2xPlain4x4GeometryTraceEnabled()
{
    return m2xPlain4x4TraceEnabled;
}

void resetM2wPlain4x4AlignedEventTrace()
{
    const char* traceDir = std::getenv("GMX_PCFF_RESPA_M2W_TRACE_DIR");
    m2wPlain4x4TraceEnabled = (traceDir != nullptr && *traceDir != '\0');
    std::lock_guard<std::mutex> guard(m2wPlain4x4TraceMutex);
    m2wPlain4x4AlignedEventLjTotals.clear();
    m2wPlain4x4AlignedEventRecords.clear();
    m2wPlain4x4AlignedEventLjRunningTotal = 0.0;
    m2wPlain4x4AlignedEventOrdinal        = 0;
    const std::string tracePath = m2wPlain4x4IdentityTracePath();
    if (!tracePath.empty() && tracePath != m2wPlain4x4ClearedTracePath)
    {
        std::remove(tracePath.c_str());
        m2wPlain4x4ClearedTracePath = tracePath;
    }
}

void resetM2xPlain4x4GeometryTrace()
{
    const char* traceDir = std::getenv("GMX_PCFF_RESPA_M2X_TRACE_DIR");
    m2xPlain4x4TraceEnabled = (traceDir != nullptr && *traceDir != '\0');
    std::lock_guard<std::mutex> guard(m2xPlain4x4TraceMutex);
    m2xPlain4x4AlignedEventOrdinal = 0;
    const std::string tracePath = m2xPlain4x4GeometryTracePath();
    if (!tracePath.empty() && tracePath != m2xPlain4x4ClearedTracePath)
    {
        std::remove(tracePath.c_str());
        m2xPlain4x4ClearedTracePath = tracePath;
    }
}

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
                                 real finalEventLj)
{
    if (!m2wPlain4x4TraceEnabled)
    {
        return;
    }
    std::lock_guard<std::mutex> guard(m2wPlain4x4TraceMutex);
    const double runningBefore = m2wPlain4x4AlignedEventLjRunningTotal;
    m2wPlain4x4AlignedEventLjRunningTotal += finalEventLj;
    ++m2wPlain4x4AlignedEventOrdinal;
    m2wPlain4x4AlignedEventLjTotals.push_back(m2wPlain4x4AlignedEventLjRunningTotal);
    if (m2wPlain4x4AlignedEventOrdinal >= 668 && m2wPlain4x4AlignedEventOrdinal <= 670)
    {
        M2wPlain4x4AlignedEventRecord record;
        record.alignedEventOrdinal = m2wPlain4x4AlignedEventOrdinal;
        record.pairI               = ai;
        record.pairJ               = aj;
        record.typeI               = typeI;
        record.typeJ               = typeJ;
        record.ciIndex             = ciIndex;
        record.cjIndex             = cjIndex;
        record.iIndex              = iIndex;
        record.jIndex              = jIndex;
        record.runningTotalBefore  = runningBefore;
        record.runningTotalAfter   = m2wPlain4x4AlignedEventLjRunningTotal;
        record.rawLjTerm           = rawLjTerm;
        record.scalingFactor       = scalingFactor;
        record.finalEventLj        = finalEventLj;
        record.c6                  = c6;
        record.c12                 = c12;
        record.rsq                 = rsq;
        record.r                   = r;
        m2wPlain4x4AlignedEventRecords.push_back(record);
        appendM2wPlain4x4IdentityTraceLine(record);
    }
}

void noteM2xPlain4x4GeometryEvent(const M2xPlain4x4GeometryEventData& data)
{
    if (!m2xPlain4x4TraceEnabled)
    {
        return;
    }

    std::lock_guard<std::mutex> guard(m2xPlain4x4TraceMutex);
    ++m2xPlain4x4AlignedEventOrdinal;
    if (m2xPlain4x4AlignedEventOrdinal != 669)
    {
        return;
    }

    appendM2xPlain4x4GeometryStageLine("GEOM_COORD_SOURCE", data, false, false, false, false);
    appendM2xPlain4x4GeometryStageLine("GEOM_SHIFT_OR_PBC_APPLY", data, true, false, false, false);
    appendM2xPlain4x4GeometryStageLine("GEOM_DXDYDZ_CONSTRUCTION", data, true, true, false, false);
    appendM2xPlain4x4GeometryStageLine("GEOM_RSQ_FORMATION", data, true, true, true, false);
    appendM2xPlain4x4GeometryStageLine("EVENT_669_LJ_INPUT", data, true, true, true, true);
}

bool m2pPlain4x4CoulombFirstWriteTraceEnabled()
{
    return !m2pPlain4x4CoulombFirstWriteTracePath().empty();
}

void noteM2pPlain4x4CoulombFirstWrite(real        targetBefore,
                                      real        writeValue,
                                      real        targetAfter,
                                      int         energyIndex,
                                      const char* codeLocation)
{
    const std::string tracePath = m2pPlain4x4CoulombFirstWriteTracePath();
    if (tracePath.empty())
    {
        return;
    }

    std::lock_guard<std::mutex> guard(m2pPlain4x4CoulombWriteTraceMutex);
    if (m2pPlain4x4CoulombWriteClearedTracePath != tracePath)
    {
        std::ofstream clearOut(tracePath, std::ios::trunc);
        m2pPlain4x4CoulombWriteClearedTracePath = tracePath;
        m2pPlain4x4CoulombWriteOrdinal          = 0;
    }
    if (m2pPlain4x4CoulombWriteOrdinal >= 5)
    {
        return;
    }

    ++m2pPlain4x4CoulombWriteOrdinal;

    std::ofstream out(tracePath, std::ios::app);
    if (!out)
    {
        return;
    }

    out << std::fixed << std::setprecision(15)
        << "side=PLAIN"
        << " write_ordinal=" << m2pPlain4x4CoulombWriteOrdinal
        << " code_location=" << codeLocation
        << " energyIndex=" << energyIndex
        << " target_before=" << targetBefore
        << " write_value=" << writeValue
        << " target_after=" << targetAfter
        << '\n';
}

bool m2pPlain4x4CoulombProducerTraceEnabled()
{
    return !m2pPlain4x4CoulombProducerTracePath().empty()
           && (!m2pPlain4x4CoulombPrefixCheckpoints().empty()
               || !m2pPlain4x4CoulombDetailOrdinals().empty());
}

bool m2pPlain4x4CoulombSelfTraceEnabled()
{
    return !m2pPlain4x4CoulombSelfDetailTracePath().empty()
           && (!m2pPlain4x4CoulombSelfPrefixCheckpoints().empty()
               || !m2pPlain4x4CoulombSelfDetailOrdinals().empty());
}

bool m2pPlain4x4MultiStepCoulombPairTraceEnabled()
{
    const int64_t currentStep = readM2pPlain4x4CurrentStep();
    return m2pPlain4x4ShouldTraceMultiStepCoulombStep(currentStep)
           && !m2pPlain4x4MultiStepCoulombPairDetailTracePath().empty()
           && (!m2pPlain4x4MultiStepCoulombPairPrefixCheckpoints().empty()
               || !m2pPlain4x4MultiStepCoulombPairDetailOrdinals().empty());
}

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
                                    const char* codeLocation)
{
    const std::string detailTracePath = m2pPlain4x4CoulombProducerTracePath();
    const std::string prefixTracePath = m2pPlain4x4CoulombPrefixTracePath();
    if (detailTracePath.empty() || prefixTracePath.empty())
    {
        return;
    }

    std::lock_guard<std::mutex> guard(m2pPlain4x4CoulombProducerTraceMutex);
    if (m2pPlain4x4CoulombProducerClearedTracePath != detailTracePath)
    {
        std::ofstream clearDetail(detailTracePath, std::ios::trunc);
        std::ofstream clearPrefix(prefixTracePath, std::ios::trunc);
        m2pPlain4x4CoulombProducerClearedTracePath = detailTracePath;
        m2pPlain4x4CoulombProducerOrdinal          = 0;
        m2pPlain4x4CoulombProducerPrefixSum        = 0.0;
    }

    const double targetBefore = m2pPlain4x4CoulombProducerPrefixSum;
    ++m2pPlain4x4CoulombProducerOrdinal;
    m2pPlain4x4CoulombProducerPrefixSum += vcoul;
    const double targetAfter = m2pPlain4x4CoulombProducerPrefixSum;

    const auto& prefixCheckpoints = m2pPlain4x4CoulombPrefixCheckpoints();
    if (ordinalRequested(m2pPlain4x4CoulombProducerOrdinal, prefixCheckpoints))
    {
        std::ofstream prefixOut(prefixTracePath, std::ios::app);
        if (prefixOut)
        {
            prefixOut << std::fixed << std::setprecision(15)
                      << "side=PLAIN"
                      << " producer_count=" << m2pPlain4x4CoulombProducerOrdinal
                      << " cumulative_coulomb_prefix_sum=" << m2pPlain4x4CoulombProducerPrefixSum
                      << '\n';
        }
    }

    const auto& detailOrdinals = m2pPlain4x4CoulombDetailOrdinals();
    if (!ordinalRequested(m2pPlain4x4CoulombProducerOrdinal, detailOrdinals))
    {
        return;
    }

    std::ofstream out(detailTracePath, std::ios::app);
    if (!out)
    {
        return;
    }

    out << std::fixed << std::setprecision(15)
        << "side=PLAIN"
        << " producer_ordinal=" << m2pPlain4x4CoulombProducerOrdinal
        << " code_location=" << codeLocation
        << " pair_i=" << pairI
        << " pair_j=" << pairJ
        << " energyIndex=" << energyIndex
        << " target_before=" << targetBefore
        << " target_after=" << targetAfter
        << " excludedMask=" << excludedMask
        << " qq=" << qq
        << " interact=" << interact
        << " rinv=" << rinv
        << " ewald_shift=" << ewaldShift
        << " table_index=" << tableIndex
        << " frac=" << frac
        << " fexcl=" << fexcl
        << " vcorr=" << vcorr
        << " vcoul_unmasked=" << vcoulUnmasked
        << " vcoul=" << vcoul
        << '\n';
}

void noteM2pPlain4x4CoulombSelfContribution(int         atom,
                                            int         energyIndex,
                                            real        charge,
                                            real        selfEnergy,
                                            real        targetBefore,
                                            real        targetAfter,
                                            const char* codeLocation)
{
    const std::string detailTracePath = m2pPlain4x4CoulombSelfDetailTracePath();
    const std::string prefixTracePath = m2pPlain4x4CoulombSelfPrefixTracePath();
    if (detailTracePath.empty() || prefixTracePath.empty())
    {
        return;
    }

    std::lock_guard<std::mutex> guard(m2pPlain4x4CoulombSelfTraceMutex);
    if (m2pPlain4x4CoulombSelfClearedTracePath != detailTracePath)
    {
        std::ofstream clearDetail(detailTracePath, std::ios::trunc);
        std::ofstream clearPrefix(prefixTracePath, std::ios::trunc);
        m2pPlain4x4CoulombSelfClearedTracePath = detailTracePath;
        m2pPlain4x4CoulombSelfOrdinal          = 0;
        m2pPlain4x4CoulombSelfPrefixSum        = 0.0;
    }

    const double prefixBefore = m2pPlain4x4CoulombSelfPrefixSum;
    ++m2pPlain4x4CoulombSelfOrdinal;
    m2pPlain4x4CoulombSelfPrefixSum += selfEnergy;
    const double prefixAfter = m2pPlain4x4CoulombSelfPrefixSum;

    const auto& prefixCheckpoints = m2pPlain4x4CoulombSelfPrefixCheckpoints();
    if (ordinalRequested(m2pPlain4x4CoulombSelfOrdinal, prefixCheckpoints))
    {
        std::ofstream prefixOut(prefixTracePath, std::ios::app);
        if (prefixOut)
        {
            prefixOut << std::fixed << std::setprecision(15)
                      << "side=PLAIN"
                      << " atom_ordinal=" << m2pPlain4x4CoulombSelfOrdinal
                      << " cumulative_self_coulomb_prefix_sum=" << m2pPlain4x4CoulombSelfPrefixSum
                      << '\n';
        }
    }

    const auto& detailOrdinals = m2pPlain4x4CoulombSelfDetailOrdinals();
    if (!ordinalRequested(m2pPlain4x4CoulombSelfOrdinal, detailOrdinals))
    {
        return;
    }

    std::ofstream out(detailTracePath, std::ios::app);
    if (!out)
    {
        return;
    }

    out << std::fixed << std::setprecision(15)
        << "side=PLAIN"
        << " atom_ordinal=" << m2pPlain4x4CoulombSelfOrdinal
        << " atom=" << atom
        << " energyIndex=" << energyIndex
        << " charge=" << charge
        << " selfEnergy=" << selfEnergy
        << " prefix_before=" << prefixBefore
        << " prefix_after=" << prefixAfter
        << " target_before=" << targetBefore
        << " target_after=" << targetAfter
        << " code_location=" << codeLocation
        << '\n';
}

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
                                                     const char* codeLocation)
{
    const int64_t currentStep = readM2pPlain4x4CurrentStep();
    if (!m2pPlain4x4ShouldTraceMultiStepCoulombStep(currentStep))
    {
        return;
    }

    const std::string detailTracePath = m2pPlain4x4MultiStepCoulombPairDetailTracePath();
    const std::string prefixTracePath = m2pPlain4x4MultiStepCoulombPairPrefixTracePath();
    if (detailTracePath.empty() || prefixTracePath.empty())
    {
        return;
    }

    const auto& prefixCheckpoints = m2pPlain4x4MultiStepCoulombPairPrefixCheckpoints();
    const auto& detailOrdinals    = m2pPlain4x4MultiStepCoulombPairDetailOrdinals();
    if (prefixCheckpoints.empty() && detailOrdinals.empty())
    {
        return;
    }

    std::lock_guard<std::mutex> guard(m2pPlain4x4MultiStepCoulombPairTraceMutex);
    if (m2pPlain4x4MultiStepCoulombPairClearedTracePath != detailTracePath)
    {
        std::ofstream clearDetail(detailTracePath, std::ios::trunc);
        std::ofstream clearPrefix(prefixTracePath, std::ios::trunc);
        m2pPlain4x4MultiStepCoulombPairClearedTracePath = detailTracePath;
        m2pPlain4x4MultiStepCoulombPairCurrentStep      = -1;
        m2pPlain4x4MultiStepCoulombPairOrdinal          = 0;
        m2pPlain4x4MultiStepCoulombPairReplayTerms.clear();
    }
    if (m2pPlain4x4MultiStepCoulombPairCurrentStep != currentStep)
    {
        m2pPlain4x4MultiStepCoulombPairCurrentStep = currentStep;
        m2pPlain4x4MultiStepCoulombPairOrdinal     = 0;
        m2pPlain4x4MultiStepCoulombPairReplayTerms.clear();
    }

    const double cumulativeBefore = sumMultiStepReplayTerms(m2pPlain4x4MultiStepCoulombPairReplayTerms);
    if (energyIndex >= 0)
    {
        if (static_cast<int>(m2pPlain4x4MultiStepCoulombPairReplayTerms.size()) <= energyIndex)
        {
            m2pPlain4x4MultiStepCoulombPairReplayTerms.resize(energyIndex + 1, 0.0_real);
        }
        m2pPlain4x4MultiStepCoulombPairReplayTerms[energyIndex] += vcoul;
    }
    ++m2pPlain4x4MultiStepCoulombPairOrdinal;
    const double cumulativeAfter = sumMultiStepReplayTerms(m2pPlain4x4MultiStepCoulombPairReplayTerms);

    if (ordinalRequested(m2pPlain4x4MultiStepCoulombPairOrdinal, prefixCheckpoints))
    {
        std::ofstream prefixOut(prefixTracePath, std::ios::app);
        if (prefixOut)
        {
            prefixOut << std::fixed << std::setprecision(15)
                      << "side=PLAIN"
                      << " step=" << currentStep
                      << " pair_ordinal=" << m2pPlain4x4MultiStepCoulombPairOrdinal
                      << " cumulative_coulomb_prefix_sum=" << cumulativeAfter
                      << '\n';
        }
    }

    if (!ordinalRequested(m2pPlain4x4MultiStepCoulombPairOrdinal, detailOrdinals))
    {
        return;
    }

    std::ofstream out(detailTracePath, std::ios::app);
    if (!out)
    {
        return;
    }

    out << std::fixed << std::setprecision(15)
        << "side=PLAIN"
        << " step=" << currentStep
        << " pair_ordinal=" << m2pPlain4x4MultiStepCoulombPairOrdinal
        << " pair_i=" << pairI
        << " pair_j=" << pairJ
        << " energyIndex=" << energyIndex
        << " shiftIndex=" << shiftIndex
        << " coord_i_x=" << coordIX
        << " coord_i_y=" << coordIY
        << " coord_i_z=" << coordIZ
        << " coord_j_x=" << coordJX
        << " coord_j_y=" << coordJY
        << " coord_j_z=" << coordJZ
        << " shift_x=" << shiftX
        << " shift_y=" << shiftY
        << " shift_z=" << shiftZ
        << " dx=" << dx
        << " dy=" << dy
        << " dz=" << dz
        << " rsq=" << rsq
        << " qq=" << qq
        << " rinv=" << rinv
        << " table_index=" << tableIndex
        << " frac=" << frac
        << " fexcl=" << fexcl
        << " vcorr=" << vcorr
        << " interact=" << interact
        << " pair_contribution=" << vcoul
        << " cumulative_before=" << cumulativeBefore
        << " cumulative_after=" << cumulativeAfter
        << " cluster_i=" << clusterI
        << " cluster_j=" << clusterJ
        << " local_i=" << localI
        << " local_j=" << localJ
        << " code_location=" << codeLocation
        << '\n';
}

std::vector<M2wPlain4x4AlignedEventRecord> readM2wPlain4x4AlignedEventRecords()
{
    std::lock_guard<std::mutex> guard(m2wPlain4x4TraceMutex);
    return m2wPlain4x4AlignedEventRecords;
}

std::vector<double> readM2wPlain4x4AlignedEventTotals()
{
    std::lock_guard<std::mutex> guard(m2wPlain4x4TraceMutex);
    return m2wPlain4x4AlignedEventLjTotals;
}

void resetM2pPlain4x4CoulombContractReplay()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4CoulombContractReplayMutex);
    m2pPlain4x4CoulombContractReplayTraceEnabled = (activeLjSrTraceDirPath() != nullptr);
    m2pPlain4x4CoulombContractReplayPairContributions.clear();
    m2pPlain4x4CoulombContractReplaySelfContributions.clear();
}

bool m2pPlain4x4CoulombContractReplayEnabled()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4CoulombContractReplayMutex);
    return m2pPlain4x4CoulombContractReplayTraceEnabled;
}

void noteM2pPlain4x4CoulombContractReplayPairContribution(int energyIndex, real vcoul)
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4CoulombContractReplayMutex);
    if (!m2pPlain4x4CoulombContractReplayTraceEnabled || vcoul == 0.0)
    {
        return;
    }
    m2pPlain4x4CoulombContractReplayPairContributions.emplace_back(energyIndex, vcoul);
}

void noteM2pPlain4x4CoulombContractReplaySelfContribution(int energyIndex, real selfEnergy)
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4CoulombContractReplayMutex);
    if (!m2pPlain4x4CoulombContractReplayTraceEnabled || selfEnergy == 0.0)
    {
        return;
    }
    m2pPlain4x4CoulombContractReplaySelfContributions.emplace_back(energyIndex, selfEnergy);
}

double readM2pPlain4x4CoulombContractReplayTotal()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4CoulombContractReplayMutex);
    if (!m2pPlain4x4CoulombContractReplayTraceEnabled)
    {
        return 0.0;
    }

    int maxEnergyIndex = -1;
    for (const auto& contribution : m2pPlain4x4CoulombContractReplayPairContributions)
    {
        maxEnergyIndex = std::max(maxEnergyIndex, contribution.first);
    }
    for (const auto& contribution : m2pPlain4x4CoulombContractReplaySelfContributions)
    {
        maxEnergyIndex = std::max(maxEnergyIndex, contribution.first);
    }
    if (maxEnergyIndex < 0)
    {
        return 0.0;
    }

    std::vector<real> replayTerms(maxEnergyIndex + 1, 0.0_real);
    for (const auto& contribution : m2pPlain4x4CoulombContractReplayPairContributions)
    {
        replayTerms[contribution.first] += contribution.second;
    }
    for (const auto& contribution : m2pPlain4x4CoulombContractReplaySelfContributions)
    {
        replayTerms[contribution.first] += contribution.second;
    }

    double total = 0.0;
    for (const real value : replayTerms)
    {
        total += value;
    }
    return total;
}

double readM2pPlain4x4CoulombContractReplayPairOnlyTotal()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4CoulombContractReplayMutex);
    if (!m2pPlain4x4CoulombContractReplayTraceEnabled)
    {
        return 0.0;
    }

    int maxEnergyIndex = -1;
    for (const auto& contribution : m2pPlain4x4CoulombContractReplayPairContributions)
    {
        maxEnergyIndex = std::max(maxEnergyIndex, contribution.first);
    }
    if (maxEnergyIndex < 0)
    {
        return 0.0;
    }

    std::vector<real> replayTerms(maxEnergyIndex + 1, 0.0_real);
    for (const auto& contribution : m2pPlain4x4CoulombContractReplayPairContributions)
    {
        replayTerms[contribution.first] += contribution.second;
    }

    double total = 0.0;
    for (const real value : replayTerms)
    {
        total += value;
    }
    return total;
}

void setM2pPlain4x4CurrentStep(int64_t step)
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4CurrentStepMutex);
    m2pPlain4x4CurrentStep = step;
}

int64_t readM2pPlain4x4CurrentStep()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4CurrentStepMutex);
    return m2pPlain4x4CurrentStep;
}

void resetM2pPlain4x4LjContractReplay()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4LjContractReplayMutex);
    m2pPlain4x4LjContractReplayTraceEnabled = (activeLjSrTraceDirPath() != nullptr);
    m2pPlain4x4LjContractReplayPairContributions.clear();
}

bool m2pPlain4x4LjContractReplayEnabled()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4LjContractReplayMutex);
    return m2pPlain4x4LjContractReplayTraceEnabled;
}

void noteM2pPlain4x4LjContractReplayPairContribution(real vlj)
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4LjContractReplayMutex);
    if (!m2pPlain4x4LjContractReplayTraceEnabled || vlj == 0.0)
    {
        return;
    }
    m2pPlain4x4LjContractReplayPairContributions.push_back(vlj);
}

double readM2pPlain4x4LjContractReplayTotal()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4LjContractReplayMutex);
    if (!m2pPlain4x4LjContractReplayTraceEnabled)
    {
        return 0.0;
    }

    double total = 0.0;
    for (const real value : m2pPlain4x4LjContractReplayPairContributions)
    {
        total += static_cast<double>(value);
    }
    return total;
}

bool m2pPlain4x4ExclusionEquivalenceTraceEnabled()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4ExclusionEquivalenceTraceMutex);
    return activeLjSrOptionalTraceEnabled("GMX_PCFF_RESPA_TRACE_EXCLUSION_EQUIVALENCE")
           && readM2pPlain4x4CurrentStep() == 0 && !m2pPlain4x4ExclusionEquivalenceTracePath().empty();
}

void noteM2pPlain4x4ExclusionEquivalencePair(int         ai,
                                             int         aj,
                                             real        interact,
                                             real        excludedMask,
                                             real        skipmask,
                                             real        qq,
                                             int         tableIndex,
                                             real        frac,
                                             real        fexcl,
                                             real        vcorr,
                                             real        correctionScalarUnmasked,
                                             real        correctionScalarEffective,
                                             real        correctionForceUnmaskedFx,
                                             real        correctionForceUnmaskedFy,
                                             real        correctionForceUnmaskedFz,
                                             real        correctionForceEffectiveFx,
                                             real        correctionForceEffectiveFy,
                                             real        correctionForceEffectiveFz,
                                             real        combinedForceFx,
                                             real        combinedForceFy,
                                             real        combinedForceFz,
                                             const char* sinkTarget,
                                             bool        sinkWriteExecuted,
                                             const char* codeLocation)
{
    if (!(ai == 0 || ai == 5 || aj == 0 || aj == 5))
    {
        return;
    }

    std::lock_guard<std::mutex> guard(m2pPlain4x4ExclusionEquivalenceTraceMutex);
    if (!activeLjSrOptionalTraceEnabled("GMX_PCFF_RESPA_TRACE_EXCLUSION_EQUIVALENCE")
        || readM2pPlain4x4CurrentStep() != 0)
    {
        return;
    }

    const std::string tracePath = m2pPlain4x4ExclusionEquivalenceTracePath();
    if (tracePath.empty())
    {
        return;
    }

    std::ostringstream line;
    line << std::fixed << std::setprecision(15);
    line << "side=PLAIN"
         << " step=0"
         << " ai=" << ai
         << " aj=" << aj
         << " pair_key=" << ai << "_" << aj
         << " source_path=plain_reference_kernel"
         << " membership_source=kernel_pair_entry"
         << " list_kind=kernel_pair_entry"
         << " interact=" << interact
         << " excluded_mask=" << excludedMask
         << " skipmask=" << skipmask
         << " qq=" << qq
         << " table_index=" << tableIndex
         << " frac=" << frac
         << " fexcl=" << fexcl
         << " vcorr=" << vcorr
         << " correction_force_scalar_unmasked=" << correctionScalarUnmasked
         << " correction_force_scalar_effective=" << correctionScalarEffective
         << " correction_force_unmasked_fx=" << correctionForceUnmaskedFx
         << " correction_force_unmasked_fy=" << correctionForceUnmaskedFy
         << " correction_force_unmasked_fz=" << correctionForceUnmaskedFz
         << " correction_force_effective_fx=" << correctionForceEffectiveFx
         << " correction_force_effective_fy=" << correctionForceEffectiveFy
         << " correction_force_effective_fz=" << correctionForceEffectiveFz
         << " combined_force_fx=" << combinedForceFx
         << " combined_force_fy=" << combinedForceFy
         << " combined_force_fz=" << combinedForceFz
         << " treated_as_exclusion_correction_producer_unmasked="
         << (correctionScalarUnmasked != 0.0 ? "true" : "false")
         << " treated_as_exclusion_correction_producer_effective="
         << (correctionScalarEffective != 0.0 ? "true" : "false")
         << " sink_target=" << (sinkTarget != nullptr ? sinkTarget : "unknown")
         << " sink_write_executed=" << (sinkWriteExecuted ? "true" : "false")
         << " code_location=" << (codeLocation != nullptr ? codeLocation : "unknown");

    std::ofstream out(tracePath, std::ios::app);
    if (out)
    {
        out << line.str() << '\n';
    }
}

void noteM2pPlain4x4Step2PairTotal(int         ai,
                                   int         aj,
                                   real        r,
                                   real        rawLjScalar,
                                   real        bareCoulombScalar,
                                   real        correctionScalar,
                                   real        ljFx,
                                   real        ljFy,
                                   real        ljFz,
                                   real        coulombFx,
                                   real        coulombFy,
                                   real        coulombFz,
                                   real        correctionFx,
                                   real        correctionFy,
                                   real        correctionFz,
                                   real        totalFx,
                                   real        totalFy,
                                   real        totalFz,
                                   real        qq,
                                   real        rinv,
                                   int         tableIndex,
                                   real        frac,
                                   real        fexcl,
                                   real        vcorr,
                                   const char* codeLocation)
{
    if (!(ai == 0 || ai == 5 || aj == 0 || aj == 5))
    {
        return;
    }

    const int64_t currentStep = readM2pPlain4x4CurrentStep();
    if (currentStep != 2)
    {
        return;
    }

    const std::string tracePath = m2pPlain4x4Step2PairTotalTracePath();
    if (tracePath.empty())
    {
        return;
    }

    std::lock_guard<std::mutex> guard(m2pPlain4x4PairTotalTraceMutex);
    if (m2pPlain4x4PairTotalClearedTracePath != tracePath || m2pPlain4x4PairTotalCurrentStep != currentStep)
    {
        std::ofstream clearTrace(tracePath, std::ios::trunc);
        m2pPlain4x4PairTotalClearedTracePath = tracePath;
        m2pPlain4x4PairTotalCurrentStep      = currentStep;
    }

    const auto appendForFocusAtom = [&](const int atomFocus)
    {
        if (atomFocus != ai && atomFocus != aj)
        {
            return;
        }

        const real sign = (atomFocus == ai) ? 1.0_real : -1.0_real;

        std::ostringstream line;
        line << std::fixed << std::setprecision(15);
        line << "side=PLAIN"
             << " step=" << currentStep
             << " pair_i=" << ai
             << " pair_j=" << aj
             << " atom_focus=" << atomFocus
             << " comparison_group=" << currentStep << "_" << ai << "_" << aj << "_" << atomFocus
             << " r=" << r
             << " plain_rawLjScalar=" << rawLjScalar
             << " plain_bareCoulombScalar=" << bareCoulombScalar
             << " plain_correctionScalar=" << correctionScalar
             << " qq=" << qq
             << " rinv=" << rinv
             << " table_index=" << tableIndex
             << " frac=" << frac
             << " fexcl=" << fexcl
             << " vcorr=" << vcorr
             << " plain_lj_force_x=" << sign * ljFx
             << " plain_lj_force_y=" << sign * ljFy
             << " plain_lj_force_z=" << sign * ljFz
             << " plain_coulomb_force_x=" << sign * coulombFx
             << " plain_coulomb_force_y=" << sign * coulombFy
             << " plain_coulomb_force_z=" << sign * coulombFz
             << " plain_correction_force_x=" << sign * correctionFx
             << " plain_correction_force_y=" << sign * correctionFy
             << " plain_correction_force_z=" << sign * correctionFz
             << " plain_total_force_x=" << sign * totalFx
             << " plain_total_force_y=" << sign * totalFy
             << " plain_total_force_z=" << sign * totalFz
             << " code_location=" << (codeLocation != nullptr ? codeLocation : "unknown");

        std::ofstream out(tracePath, std::ios::app);
        if (out)
        {
            out << line.str() << '\n';
        }
    };

    appendForFocusAtom(0);
    appendForFocusAtom(5);
}

void resetM2pPlain4x4RealspaceForceSubcomponentTrace()
{
    int64_t currentStep = -1;
    {
        std::lock_guard<std::mutex> stepGuard(m2pPlain4x4CurrentStepMutex);
        currentStep = m2pPlain4x4CurrentStep;
    }

    std::lock_guard<std::mutex> guard(m2pPlain4x4RealspaceForceSubcomponentTraceMutex);
    m2pPlain4x4RealspaceForceSubcomponentTraceEnabledFlag =
            (activeLjSrOptionalTraceEnabled("GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS")
             && currentStep == 0)
            || (activeLjSrOptionalTraceEnabled("GMX_PCFF_RESPA_TRACE_STEP1_SUBSET01_FORCEGROUP_AUDIT")
                && currentStep == 2);
    clearTracedForcePair(&m2pPlain4x4LjSrForcePair);
    clearTracedForcePair(&m2pPlain4x4CoulombSrForcePair);
    clearTracedForcePair(&m2pPlain4x4ExclusionCorrectionForcePair);
    clearTracedForcePair(&m2pPlain4x4CombinedRealspaceForcePair);
}

bool m2pPlain4x4RealspaceForceSubcomponentTraceEnabled()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4RealspaceForceSubcomponentTraceMutex);
    return m2pPlain4x4RealspaceForceSubcomponentTraceEnabledFlag;
}

void noteM2pPlain4x4RealspaceForceSubcomponents(int  ai,
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
                                                real combinedFz)
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4RealspaceForceSubcomponentTraceMutex);
    if (!m2pPlain4x4RealspaceForceSubcomponentTraceEnabledFlag)
    {
        return;
    }

    addForceContributionToTrackedAtoms(&m2pPlain4x4LjSrForcePair, ai, aj, ljFx, ljFy, ljFz);
    addForceContributionToTrackedAtoms(
            &m2pPlain4x4CoulombSrForcePair, ai, aj, coulombSrFx, coulombSrFy, coulombSrFz);
    addForceContributionToTrackedAtoms(&m2pPlain4x4ExclusionCorrectionForcePair,
                                       ai,
                                       aj,
                                       exclusionCorrectionFx,
                                       exclusionCorrectionFy,
                                       exclusionCorrectionFz);
    addForceContributionToTrackedAtoms(
            &m2pPlain4x4CombinedRealspaceForcePair, ai, aj, combinedFx, combinedFy, combinedFz);
}

M2pPlain4x4TracedForcePair readM2pPlain4x4LjSrForcePair()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4RealspaceForceSubcomponentTraceMutex);
    return m2pPlain4x4LjSrForcePair;
}

M2pPlain4x4TracedForcePair readM2pPlain4x4CoulombSrForcePair()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4RealspaceForceSubcomponentTraceMutex);
    return m2pPlain4x4CoulombSrForcePair;
}

M2pPlain4x4TracedForcePair readM2pPlain4x4ExclusionCorrectionForcePair()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4RealspaceForceSubcomponentTraceMutex);
    return m2pPlain4x4ExclusionCorrectionForcePair;
}

M2pPlain4x4TracedForcePair readM2pPlain4x4CombinedRealspaceForcePair()
{
    std::lock_guard<std::mutex> guard(m2pPlain4x4RealspaceForceSubcomponentTraceMutex);
    return m2pPlain4x4CombinedRealspaceForcePair;
}

#define UNROLLI 4
#define UNROLLJ 4

// No vectorization of the j-loop possible for the 4x4 kernel
#define VECTORIZE_JLOOP 0

static_assert(UNROLLI == sc_iClusterSize(NbnxmKernelType::Cpu4x4_PlainC),
              "Unroll size should match that of of the kernel type");

#define GMX_PCFF_RESPA_M2Q_PLAIN_RAW_TRACE_ENABLED
#define GMX_PCFF_RESPA_M2R_PLAIN_TRACE_ENABLED
#define GMX_PCFF_RESPA_M2S_PLAIN_TRACE_ENABLED
#define GMX_PCFF_RESPA_M2U_PLAIN_TRACE_ENABLED
#define GMX_PCFF_RESPA_M2V_PLAIN_TRACE_ENABLED
#define GMX_PCFF_RESPA_M2W_PLAIN_TRACE_ENABLED
#define GMX_PCFF_RESPA_M2X_PLAIN_TRACE_ENABLED

/* Analytical reaction-field kernels */
#define CALC_COUL_RF
#define LJ_CUT
#include "kernel_ref_includes.h"
#undef LJ_CUT
#define LJ_FORCE_SWITCH
#include "kernel_ref_includes.h"
#undef LJ_FORCE_SWITCH
#define LJ_POT_SWITCH
#include "kernel_ref_includes.h"
#undef LJ_POT_SWITCH
#define LJ_EWALD
#define LJ_CUT
#define LJ_EWALD_COMB_GEOM
#include "kernel_ref_includes.h"
#undef LJ_EWALD_COMB_GEOM
#define LJ_EWALD_COMB_LB
#include "kernel_ref_includes.h"
#undef LJ_EWALD_COMB_LB
#undef LJ_CUT
#undef LJ_EWALD
#undef CALC_COUL_RF


/* Tabulated exclusion interaction electrostatics kernels */
#define CALC_COUL_TAB
#define LJ_CUT
#include "kernel_ref_includes.h"
#undef LJ_CUT
#define LJ_FORCE_SWITCH
#include "kernel_ref_includes.h"
#undef LJ_FORCE_SWITCH
#define LJ_POT_SWITCH
#include "kernel_ref_includes.h"
#undef LJ_POT_SWITCH
#define LJ_EWALD
#define LJ_CUT
#define LJ_EWALD_COMB_GEOM
#include "kernel_ref_includes.h"
#undef LJ_EWALD_COMB_GEOM
#define LJ_EWALD_COMB_LB
#include "kernel_ref_includes.h"
#undef LJ_EWALD_COMB_LB
#undef LJ_CUT
#undef LJ_EWALD
/* Twin-range cut-off kernels */
#define VDW_CUTOFF_CHECK
#define LJ_CUT
#include "kernel_ref_includes.h"
#undef LJ_CUT
#define LJ_FORCE_SWITCH
#include "kernel_ref_includes.h"
#undef LJ_FORCE_SWITCH
#define LJ_POT_SWITCH
#include "kernel_ref_includes.h"
#undef LJ_POT_SWITCH
#define LJ_EWALD
#define LJ_CUT
#define LJ_EWALD_COMB_GEOM
#include "kernel_ref_includes.h"
#undef LJ_EWALD_COMB_GEOM
#define LJ_EWALD_COMB_LB
#include "kernel_ref_includes.h"
#undef LJ_EWALD_COMB_LB
#undef LJ_CUT
#undef LJ_EWALD
#undef VDW_CUTOFF_CHECK
#undef CALC_COUL_TAB

#if GMX_USE_EXT_FMM

#    define CALC_COUL_NONE
#    define VDW_CUTOFF_CHECK
#    define LJ_CUT
#    include "kernel_ref_includes.h"
#    undef LJ_CUT
#    define LJ_FORCE_SWITCH
#    include "kernel_ref_includes.h"
#    undef LJ_FORCE_SWITCH
#    define LJ_POT_SWITCH
#    include "kernel_ref_includes.h"
#    undef LJ_POT_SWITCH
#    define LJ_EWALD
#    define LJ_CUT
#    define LJ_EWALD_COMB_GEOM
#    include "kernel_ref_includes.h"
#    undef LJ_EWALD_COMB_GEOM
#    define LJ_EWALD_COMB_LB
#    include "kernel_ref_includes.h"
#    undef LJ_EWALD_COMB_LB
#    undef LJ_CUT
#    undef LJ_EWALD
#    undef VDW_CUTOFF_CHECK
#    undef CALC_COUL_NONE

#endif

#undef GMX_PCFF_RESPA_M2Q_PLAIN_RAW_TRACE_ENABLED
#undef GMX_PCFF_RESPA_M2R_PLAIN_TRACE_ENABLED
#undef GMX_PCFF_RESPA_M2S_PLAIN_TRACE_ENABLED
#undef GMX_PCFF_RESPA_M2U_PLAIN_TRACE_ENABLED
#undef GMX_PCFF_RESPA_M2V_PLAIN_TRACE_ENABLED
#undef GMX_PCFF_RESPA_M2W_PLAIN_TRACE_ENABLED
#undef GMX_PCFF_RESPA_M2X_PLAIN_TRACE_ENABLED

} // namespace gmx
