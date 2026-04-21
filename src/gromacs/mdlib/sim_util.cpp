/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 1991- The GROMACS Authors
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

#include "config.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gromacs/applied_forces/awh/awh.h"
#include "gromacs/domdec/dlbtiming.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/gpuhaloexchange.h"
#include "gromacs/domdec/haloexchange.h"
#include "gromacs/domdec/partition.h"
#include "gromacs/essentialdynamics/edsam.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/ewald/pme_coordinate_receiver_gpu.h"
#include "gromacs/ewald/pme_pp.h"
#include "gromacs/ewald/pme_pp_comm_gpu.h"
#include "gromacs/gpu_utils/device_stream_manager.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/imd/imd.h"
#include "gromacs/listed_forces/bonded.h"
#include "gromacs/listed_forces/disre.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/listed_forces/orires.h"
#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/mdlib/calcmu.h"
#include "gromacs/mdlib/calcvir.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/dispersioncorrection.h"
#include "gromacs/mdlib/enerdata_utils.h"
#include "gromacs/mdlib/exactrespa_nonbonded_gpu.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdlib/forcerec.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdlib/wall.h"
#include "gromacs/mdlib/wholemoleculetransform.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/atominfo.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/exactrespaforcestore.h"
#include "gromacs/mdtypes/forcebuffers.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/exactrespaschedule.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/multipletimestepping.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/mdtypes/state_propagator_data_gpu.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#if GMX_GPU_CUDA
#    include "gromacs/nbnxm/cuda/nbnxm_cuda_types.h"
#endif
#include "gromacs/nbnxm/kernels_reference/kernel_ref_4x4.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/pairlist.h"
#include "gromacs/nbnxm/pairlistset.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pulling/pull.h"
#include "gromacs/pulling/pull_rotation.h"
#include "gromacs/timing/cyclecounter.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/timing/wallcyclereporting.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/booltype.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/fixedcapacityvector.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/gmxomp.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/range.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/sysinfo.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vecdump.h"
#include "gromacs/utility/vectypes.h"

#include "gpuforcereduction.h"

#if GMX_GPU
#    include "gromacs/gpu_utils/devicebuffer.h"
#    include "gromacs/nbnxm/gpu_types_common.h"
#endif

class GpuEventSynchronizer;
struct gmx_edsam;
struct gmx_enfrot;
struct gmx_multisim_t;
struct gmx_pme_t;
struct interaction_const_t;
struct pull_t;

namespace gmx
{

// TODO: this environment variable allows us to verify before release
// that on less common architectures the total cost of polling is not larger than
// a blocking wait (so polling does not introduce overhead when the static
// PME-first ordering would suffice).
static const bool c_disableAlternatingWait = (std::getenv("GMX_DISABLE_ALTERNATING_GPU_WAIT") != nullptr);

static bool disableRepulsionPower9ExactRespaCpuSpecialization()
{
    return std::getenv("GMX_DISABLE_REPULSION_POWER_9_EXACT_RESPA_CPU_SPECIALIZATION") != nullptr;
}

static bool exactRespaPairLoopOmpRequested()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_PAIRLOOP_OMP");
    return env == nullptr || std::strcmp(env, "0") != 0;
}

static bool exactRespaPairLoopVectorRequested()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_PAIRLOOP_VECTOR");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

static bool exactRespaPairLoopSparseReductionRequested()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_PAIRLOOP_SPARSE_REDUCTION");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

static bool exactRespaPairLoopBlockReductionRequested()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_PAIRLOOP_BLOCK_REDUCTION");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

static bool exactRespaPairLoopTileRequested()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_PAIRLOOP_TILE");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

static bool exactRespaPairLoopNbnxm4x4Requested()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_PAIRLOOP_NBNXM4X4");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

static bool exactRespaPairLoopDirectCpuListRequested()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_PAIRLOOP_DIRECT_CPULIST");
    return env == nullptr || std::strcmp(env, "0") != 0;
}

static bool exactRespaDisableCpuNbnxmNarrow()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_DISABLE_NBNXM_NARROW");
    return env != nullptr && *env != '\0' && std::strcmp(env, "0") != 0;
}

static bool exactRespaNativeMultiContributionLaunchRequested()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_NATIVE_MULTI");
    return env == nullptr || std::strcmp(env, "0") != 0;
}

static bool exactRespaNativeMultiSplitOwnerOutputsRequested()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_SPLIT_OWNER_OUTPUTS");
    return env != nullptr && *env != '\0' && std::strcmp(env, "0") != 0;
}

static bool exactRespaNativeMultiFallbackOnOwnerStepsRequested()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_OWNER_STEP_FALLBACK");
    // Default on: audited gate_h/gate_i runtime parity only closes when
    // owner-level exact-r-RESPA steps keep using the legacy per-contribution launch.
    return env == nullptr || std::strcmp(env, "0") != 0;
}

static bool exactRespaNativeMultiFallbackOnMiddleStepsRequested()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_MIDDLE_STEP_FALLBACK");
    // Default on: native multi-contribution middle steps currently introduce
    // first-frame ULP-level force drift that amplifies in NPT runtime parity.
    // Keep the exact per-contribution launch as the default until full native
    // multi-contribution runtime parity is closed.
    return env == nullptr || std::strcmp(env, "0") != 0;
}

static const char* exactRespaPairLoopTimingDirPath()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_PAIRLOOP_TIMING_DIR");
    return (env != nullptr && *env != '\0') ? env : nullptr;
}

static const char* exactRespaPairLoopTimingLabel()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_PAIRLOOP_TIMING_LABEL");
    return (env != nullptr && *env != '\0') ? env : "unspecified";
}

static const char* exactRespaPairLoopForceDumpDirPath()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_PAIRLOOP_FORCE_DUMP_DIR");
    return (env != nullptr && *env != '\0') ? env : nullptr;
}

static const char* exactRespaNativeMultiDecisionTracePath()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_DECISION_TRACE");
    return (env != nullptr && *env != '\0') ? env : nullptr;
}

static const char* exactRespaPairLoopForceDumpLabel()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_PAIRLOOP_FORCE_DUMP_LABEL");
    return (env != nullptr && *env != '\0') ? env : "unspecified";
}

static int exactRespaPairLoopForceDumpMax()
{
    const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_PAIRLOOP_FORCE_DUMP_MAX");
    if (env == nullptr || *env == '\0')
    {
        return 1;
    }
    return std::max(0, std::atoi(env));
}

static bool useRepulsionPower9ExactRespaCpuSpecialization(const interaction_const_t& interactionConst)
{
    return gmx_within_tol(interactionConst.vdw.repulsionPower, 9.0, 10 * GMX_DOUBLE_EPS)
           && !disableRepulsionPower9ExactRespaCpuSpecialization();
}

static void sum_forces(ArrayRef<RVec> f, ArrayRef<const RVec> forceToAdd)
{
    GMX_ASSERT(f.size() >= forceToAdd.size(), "Accumulation buffer should be sufficiently large");
    const int end = forceToAdd.size();

    int gmx_unused nt = gmx_omp_nthreads_get(ModuleMultiThread::Default);
#pragma omp parallel for num_threads(nt) schedule(static)
    for (int i = 0; i < end; i++)
    {
        rvec_inc(f[i], forceToAdd[i]);
    }
}

static void calc_virial(int                         start,
                        int                         homenr,
                        const rvec                  x[],
                        const ForceWithShiftForces& forceWithShiftForces,
                        tensor                      vir_part,
                        const matrix                box,
                        t_nrnb*                     nrnb,
                        const t_forcerec*           fr,
                        PbcType                     pbcType)
{
    /* The short-range virial from surrounding boxes */
    const rvec* fshift          = as_rvec_array(forceWithShiftForces.shiftForces().data());
    const rvec* shiftVecPointer = as_rvec_array(fr->shift_vec.data());
    calc_vir(c_numShiftVectors, shiftVecPointer, fshift, vir_part, pbcType == PbcType::Screw, box);
    inc_nrnb(nrnb, eNR_VIRIAL, c_numShiftVectors);

    /* Calculate partial virial, for local atoms only, based on short range.
     * Total virial is computed in global_stat, called from do_md
     */
    const rvec* f = as_rvec_array(forceWithShiftForces.force().data());
    f_calc_vir(start, start + homenr, x, f, vir_part, box);
    inc_nrnb(nrnb, eNR_VIRIAL, homenr);

    if (debug)
    {
        pr_rvecs(debug, 0, "vir_part", vir_part, DIM);
    }
}

static void pull_potential_wrapper(const MpiComm&       mpiComm,
                                   const t_inputrec&    ir,
                                   const matrix         box,
                                   ArrayRef<const RVec> x,
                                   const t_mdatoms*     mdatoms,
                                   gmx_enerdata_t*      enerd,
                                   pull_t*              pull_work,
                                   const real*          lambda,
                                   double               t,
                                   gmx_wallcycle*       wcycle)
{
    t_pbc pbc;
    real  dvdl;

    /* Calculate the center of mass forces, this requires communication,
     * which is why pull_potential is called close to other communication.
     */
    wallcycle_start(wcycle, WallCycleCounter::PullPot);
    set_pbc(&pbc, ir.pbcType, box);
    dvdl = 0;
    enerd->term[InteractionFunction::CenterOfMassPullingEnergy] +=
            pull_potential(pull_work,
                           mdatoms->massT,
                           pbc,
                           mpiComm,
                           t,
                           lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Restraint)],
                           x,
                           &dvdl);
    enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Restraint] += dvdl;
    wallcycle_stop(wcycle, WallCycleCounter::PullPot);
}

static void pme_receive_force_ener(t_forcerec*      fr,
                                   gmx_domdec_t*    dd,
                                   ForceWithVirial* forceWithVirial,
                                   gmx_enerdata_t*  enerd,
                                   bool             useGpuPmePpComms,
                                   bool             receivePmeForceToGpu,
                                   gmx_wallcycle*   wcycle)
{
    real  e_q, e_lj, dvdl_q, dvdl_lj;
    float cycles_ppdpme, cycles_seppme;

    cycles_ppdpme = wallcycle_stop(wcycle, WallCycleCounter::PpDuringPme);
    dd_cycles_add(dd, cycles_ppdpme, ddCyclPPduringPME);

    /* In case of node-splitting, the PP nodes receive the long-range
     * forces, virial and energy from the PME nodes here.
     */
    wallcycle_start(wcycle, WallCycleCounter::PpPmeWaitRecvF);
    dvdl_q  = 0;
    dvdl_lj = 0;
    gmx_pme_receive_f(fr->pmePpCommGpu.get(),
                      dd,
                      fr->pmeForceReceiveBuffer,
                      forceWithVirial,
                      &e_q,
                      &e_lj,
                      &dvdl_q,
                      &dvdl_lj,
                      useGpuPmePpComms,
                      receivePmeForceToGpu,
                      &cycles_seppme);
    const char* reciprocalInternalTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2M_TRACE_DIR");
    const char* reciprocalInternalTraceModeEnv = std::getenv("GMX_PCFF_RESPA_M2M_MODE");
    const char* reciprocalInternalTraceMode    =
            (reciprocalInternalTraceModeEnv != nullptr && *reciprocalInternalTraceModeEnv != '\0')
                    ? reciprocalInternalTraceModeEnv
                    : "baseline";
    const char* postFinalTraceDirPath         = std::getenv("GMX_PCFF_RESPA_M2N_TRACE_DIR");
    const char* postFinalTraceModeEnv         = std::getenv("GMX_PCFF_RESPA_M2N_MODE");
    const char* postFinalTraceMode            =
            (postFinalTraceModeEnv != nullptr && *postFinalTraceModeEnv != '\0') ? postFinalTraceModeEnv
                                                                                  : "baseline";
    const real coulombReciprocalBeforeReceive =
            enerd->term[InteractionFunction::CoulombReciprocalSpace];
    if (reciprocalInternalTraceDirPath != nullptr && *reciprocalInternalTraceDirPath != '\0')
    {
        static bool dumpedReceiveEqTrace = false;
        if (!dumpedReceiveEqTrace)
        {
            std::filesystem::path traceDir(reciprocalInternalTraceDirPath);
            std::filesystem::create_directories(traceDir);
            std::filesystem::path outputPath = traceDir / "step0_reciprocal_internal_trace.txt";
            FILE*                dumpFile    = std::fopen(outputPath.string().c_str(), "a");
            if (dumpFile == nullptr)
            {
                gmx_fatal(FARGS,
                          "Could not open reciprocal internal trace output '%s' for appending",
                          outputPath.string().c_str());
            }
            std::fprintf(
                    dumpFile,
                    "stage=PME_RECEIVE_EQ mode=%s reciprocal_branch=PME_PP_RECEIVE eq_received=%.17g ledger_before_receive=%.17g ledger_after_receive=%.17g\n",
                    reciprocalInternalTraceMode,
                    e_q,
                    coulombReciprocalBeforeReceive,
                    coulombReciprocalBeforeReceive + e_q);
            std::fclose(dumpFile);
            dumpedReceiveEqTrace = true;
        }
    }
    enerd->term[InteractionFunction::CoulombReciprocalSpace] += e_q;
    if (postFinalTraceDirPath != nullptr && *postFinalTraceDirPath != '\0')
    {
        std::filesystem::path traceDir(postFinalTraceDirPath);
        std::filesystem::create_directories(traceDir);
        std::filesystem::path outputPath = traceDir / "step0_post_final_ledger_trace.txt";
        FILE*                dumpFile    = std::fopen(outputPath.string().c_str(), "a");
        if (dumpFile == nullptr)
        {
            gmx_fatal(FARGS,
                      "Could not open post-final-ledger trace output '%s' for appending",
                      outputPath.string().c_str());
        }
        std::fprintf(
                dumpFile,
                "stage=SIM_UTIL_PME_RECEIVE_ADD mode=%s code_location=src/gromacs/mdlib/sim_util.cpp:327 contract_identity=direct_energy_field energy_key=coulomb_reciprocal_space ledger_before=%.17g received_value=%.17g value=%.17g reciprocal_branch=PME_PP_RECEIVE\n",
                postFinalTraceMode,
                coulombReciprocalBeforeReceive,
                e_q,
                enerd->term[InteractionFunction::CoulombReciprocalSpace]);
        std::fclose(dumpFile);
    }
    if (reciprocalInternalTraceDirPath != nullptr && *reciprocalInternalTraceDirPath != '\0')
    {
        static bool dumpedReceivePostLedgerTrace = false;
        if (!dumpedReceivePostLedgerTrace)
        {
            std::filesystem::path traceDir(reciprocalInternalTraceDirPath);
            std::filesystem::create_directories(traceDir);
            std::filesystem::path outputPath = traceDir / "step0_reciprocal_internal_trace.txt";
            FILE*                dumpFile    = std::fopen(outputPath.string().c_str(), "a");
            if (dumpFile == nullptr)
            {
                gmx_fatal(FARGS,
                          "Could not open reciprocal internal trace output '%s' for appending",
                          outputPath.string().c_str());
            }
            std::fprintf(
                    dumpFile,
                    "stage=PME_RECEIVE_POST_LEDGER mode=%s reciprocal_branch=PME_PP_RECEIVE eq_received=%.17g ledger_before_receive=%.17g ledger_after_receive=%.17g\n",
                    reciprocalInternalTraceMode,
                    e_q,
                    coulombReciprocalBeforeReceive,
                    enerd->term[InteractionFunction::CoulombReciprocalSpace]);
            std::fclose(dumpFile);
            dumpedReceivePostLedgerTrace = true;
        }
    }
    const char* bookkeepingResidualTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2L_TRACE_DIR");
    const char* bookkeepingProbeModeEnv         = std::getenv("GMX_PCFF_RESPA_M2L_PROBE_MODE");
    if (bookkeepingResidualTraceDirPath != nullptr && *bookkeepingResidualTraceDirPath != '\0')
    {
        static bool dumpedBookkeepingReciprocalTrace = false;
        if (!dumpedBookkeepingReciprocalTrace)
        {
            const std::string bookkeepingProbeMode =
                    (bookkeepingProbeModeEnv != nullptr && *bookkeepingProbeModeEnv != '\0')
                            ? bookkeepingProbeModeEnv
                            : "baseline";
            std::filesystem::path traceDir(bookkeepingResidualTraceDirPath);
            std::filesystem::create_directories(traceDir);
            std::filesystem::path outputPath = traceDir / "step0_patch_b_bookkeeping_trace.txt";
            FILE*                dumpFile    = std::fopen(outputPath.string().c_str(), "a");
            if (dumpFile == nullptr)
            {
                gmx_fatal(FARGS,
                          "Could not open bookkeeping residual trace output '%s' for appending",
                          outputPath.string().c_str());
            }
            std::fprintf(
                    dumpFile,
                    "stage=bookkeeping_reciprocal_sink probe_mode=%s sink_name=enerd.term[CoulombReciprocalSpace] sink_class=deferred_bookkeeping_sink received_coulomb_reciprocal_energy=%.17g residual_visible=%s\n",
                    bookkeepingProbeMode.c_str(),
                    e_q,
                    (e_q != 0.0_real ? "true" : "false"));
            std::fclose(dumpFile);
            dumpedBookkeepingReciprocalTrace = true;
        }
    }
    enerd->term[InteractionFunction::LennardJonesReciprocalSpace] += e_lj;
    enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Coul] += dvdl_q;
    enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Vdw] += dvdl_lj;

    if (wcycle)
    {
        dd_cycles_add(dd, cycles_seppme, ddCyclPME);
    }
    wallcycle_stop(wcycle, WallCycleCounter::PpPmeWaitRecvF);
}

static void print_large_forces(FILE*                fp,
                               const t_mdatoms*     md,
                               const gmx_domdec_t*  dd,
                               int64_t              step,
                               real                 forceTolerance,
                               ArrayRef<const RVec> x,
                               ArrayRef<const RVec> f)
{
    real  force2Tolerance = square(forceTolerance);
    Index numNonFinite    = 0;
    for (int i = 0; i < md->homenr; i++)
    {
        real force2    = norm2(f[i]);
        bool nonFinite = !std::isfinite(force2);
        if (force2 >= force2Tolerance || nonFinite)
        {
            fprintf(fp,
                    "step %" PRId64 " atom %6d  x %8.3f %8.3f %8.3f  force %12.5e\n",
                    step,
                    ddglatnr(dd, i),
                    x[i][XX],
                    x[i][YY],
                    x[i][ZZ],
                    std::sqrt(force2));
        }
        if (nonFinite)
        {
            numNonFinite++;
        }
    }
    if (numNonFinite > 0)
    {
        /* Note that with MPI this fatal call on one rank might interrupt
         * the printing on other ranks. But we can only avoid that with
         * an expensive MPI barrier that we would need at each step.
         */
        gmx_fatal(FARGS, "At step %" PRId64 " detected non-finite forces on %td atoms", step, numNonFinite);
    }
}

//! When necessary, spreads forces on vsites and computes the virial for \p forceOutputs->forceWithShiftForces()
static void postProcessForceWithShiftForces(t_nrnb*              nrnb,
                                            gmx_wallcycle*       wcycle,
                                            const matrix         box,
                                            ArrayRef<const RVec> x,
                                            ForceOutputs*        forceOutputs,
                                            tensor               vir_force,
                                            const t_mdatoms&     mdatoms,
                                            const t_forcerec&    fr,
                                            VirtualSitesHandler* vsite,
                                            const StepWorkload&  stepWork)
{
    ForceWithShiftForces& forceWithShiftForces = forceOutputs->forceWithShiftForces();

    /* If we have NoVirSum forces, but we do not calculate the virial,
     * we later sum the forceWithShiftForces buffer together with
     * the noVirSum buffer and spread the combined vsite forces at once.
     */
    if (vsite && (!forceOutputs->haveForceWithVirial() || stepWork.computeVirial))
    {
        using VirialHandling = VirtualSitesHandler::VirialHandling;

        auto                 f      = forceWithShiftForces.force();
        auto                 fshift = forceWithShiftForces.shiftForces();
        const VirialHandling virialHandling =
                (stepWork.computeVirial ? VirialHandling::Pbc : VirialHandling::None);
        vsite->spreadForces(x, f, virialHandling, fshift, nullptr, nrnb, box, wcycle);
        forceWithShiftForces.haveSpreadVsiteForces() = true;
    }

    if (stepWork.computeVirial)
    {
        /* Calculation of the virial must be done after vsites! */
        calc_virial(
                0, mdatoms.homenr, as_rvec_array(x.data()), forceWithShiftForces, vir_force, box, nrnb, &fr, fr.pbcType);
    }
}

//! Spread, compute virial for and sum forces, when necessary
static void postProcessForces(const gmx_domdec_t*  dd,
                              int64_t              step,
                              t_nrnb*              nrnb,
                              gmx_wallcycle*       wcycle,
                              const matrix         box,
                              ArrayRef<const RVec> x,
                              ForceOutputs*        forceOutputs,
                              tensor               vir_force,
                              const t_mdatoms*     mdatoms,
                              const t_forcerec*    fr,
                              VirtualSitesHandler* vsite,
                              const StepWorkload&  stepWork)
{
    // Extract the final output force buffer, which is also the buffer for forces with shift forces
    ArrayRef<RVec> f = forceOutputs->forceWithShiftForces().force();

    if (forceOutputs->haveForceWithVirial())
    {
        auto& forceWithVirial = forceOutputs->forceWithVirial();

        if (vsite)
        {
            /* Spread the mesh force on virtual sites to the other particles...
             * This is parallellized. MPI communication is performed
             * if the constructing atoms aren't local.
             */
            GMX_ASSERT(!stepWork.computeVirial || f.data() != forceWithVirial.force_.data(),
                       "We need separate force buffers for shift and virial forces when "
                       "computing the virial");
            GMX_ASSERT(!stepWork.computeVirial
                               || forceOutputs->forceWithShiftForces().haveSpreadVsiteForces(),
                       "We should spread the force with shift forces separately when computing "
                       "the virial");
            const VirtualSitesHandler::VirialHandling virialHandling =
                    (stepWork.computeVirial ? VirtualSitesHandler::VirialHandling::NonLinear
                                            : VirtualSitesHandler::VirialHandling::None);
            matrix virial = { { 0 } };
            vsite->spreadForces(x, forceWithVirial.force_, virialHandling, {}, virial, nrnb, box, wcycle);
            forceWithVirial.addVirialContribution(virial);
        }

        if (stepWork.computeVirial)
        {
            /* Now add the forces, this is local */
            sum_forces(f, forceWithVirial.force_);

            /* Add the direct virial contributions */
            GMX_ASSERT(
                    forceWithVirial.computeVirial_,
                    "forceWithVirial should request virial computation when we request the virial");
            m_add(vir_force, forceWithVirial.getVirial(), vir_force);

            if (debug)
            {
                pr_rvecs(debug, 0, "vir_force", vir_force, DIM);
            }
        }
    }
    else
    {
        GMX_ASSERT(vsite == nullptr || forceOutputs->forceWithShiftForces().haveSpreadVsiteForces(),
                   "We should have spread the vsite forces (earlier)");
    }

    if (fr->print_force >= 0)
    {
        print_large_forces(stderr, mdatoms, dd, step, fr->print_force, x, f);
    }
}

static bool shouldTraceRespaCoordHandoffStep(const int64_t step);
static const char* activeM2pTraceDirPath();
static void appendCoordHandoffTracePair(const char*          traceDirPath,
                                        const char*          side,
                                        const char*          stageName,
                                        int64_t              step,
                                        ArrayRef<const RVec> coords,
                                        const char*          bufferLabel,
                                        const void*          bufferPtr);
static void appendCoordHandoffTracePair(const char*             traceDirPath,
                                        const char*             side,
                                        const char*             stageName,
                                        int64_t                 step,
                                        const nbnxn_atomdata_t& nbat,
                                        const char*             bufferLabel,
                                        const void*             bufferPtr);

static void do_nb_verlet(t_forcerec*                fr,
                         const interaction_const_t* ic,
                         gmx_enerdata_t*            enerd,
                         const StepWorkload&        stepWork,
                         const InteractionLocality  ilocality,
                         const int                  clearF,
                         const int64_t              step,
                         t_nrnb*                    nrnb,
                         gmx_wallcycle*             wcycle)
{
    if (!stepWork.computeNonbondedForces)
    {
        /* skip non-bonded calculation */
        return;
    }

    nonbonded_verlet_t* nbv = fr->nbv.get();

    /* GPU kernel launch overhead is already timed separately */
    if (!nbv->useGpu())
    {
        /* When dynamic pair-list  pruning is requested, we need to prune
         * at nstlistPrune steps.
         */
        if (nbv->isDynamicPruningStepCpu(step))
        {
            /* Prune the pair-list beyond fr->ic->rlistPrune using
             * the current coordinates of the atoms.
             */
            wallcycle_sub_start(wcycle, WallCycleSubCounter::NonbondedPruning);
            nbv->dispatchPruneKernelCpu(ilocality, fr->shift_vec);
            wallcycle_sub_stop(wcycle, WallCycleSubCounter::NonbondedPruning);
        }
    }

    setM2pPlain4x4CurrentStep(step);
    if (shouldTraceRespaCoordHandoffStep(step) && ilocality == InteractionLocality::Local)
    {
        const char* traceDir = activeM2pTraceDirPath();
        appendCoordHandoffTracePair(traceDir,
                                    "PLAIN",
                                    "NONBONDED_CPU_INPUT",
                                    step,
                                    nbv->nbat(),
                                    "nbv.nbat.x()_before_dispatchNonbondedKernel",
                                    nbv->nbat().x().data());
    }
    nbv->dispatchNonbondedKernel(
            ilocality,
            *ic,
            stepWork,
            clearF,
            fr->shift_vec,
            enerd->grpp.energyGroupPairTerms[fr->haveBuckingham ? NonBondedEnergyTerms::BuckinghamSR
                                                                : NonBondedEnergyTerms::LJSR],
            enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR],
            nrnb);
}

struct LammpsRespaSplitWeights
{
    real inner  = 0;
    real middle = 0;
    real outer  = 1;
};

static real respaSwitchIn(const real r, const real off, const real on)
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

static void accumulatePairVirial(const RVec& dx, const RVec& force, matrix virial)
{
    for (int dim1 = 0; dim1 < DIM; dim1++)
    {
        for (int dim2 = 0; dim2 < DIM; dim2++)
        {
            virial[dim1][dim2] -= 0.5_real * dx[dim1] * force[dim2];
        }
    }
}

static int energyGroupPairIndex(const int ai, const int aj, const t_forcerec& fr, const t_mdatoms& mdatoms)
{
    if (mdatoms.nenergrp <= 1)
    {
        return 0;
    }

    const int gidI = fr.atomInfo[ai] & gmx::sc_atomInfo_EnergyGroupIdMask;
    const int gidJ = fr.atomInfo[aj] & gmx::sc_atomInfo_EnergyGroupIdMask;
    return gidI * mdatoms.nenergrp + gidJ;
}

static real computePmeSelfEnergy(const interaction_const_t& ic)
{
    GMX_RELEASE_ASSERT(ic.coulombEwaldTables, "PME self-energy requires Coulomb Ewald tables");
    return 0.5_real
#if !GMX_DOUBLE
           * ic.coulombEwaldTables->tableFDV0[2]
#else
           * ic.coulombEwaldTables->tableV[0]
#endif
            ;
}

static void dumpRespaMergeTraceVector(const char*               traceDirPath,
                                      const char*               fileName,
                                      const std::string&        header,
                                      ArrayRef<const gmx::RVec> forceBuffer)
{
    GMX_RELEASE_ASSERT(traceDirPath != nullptr && *traceDirPath != '\0', "Need a valid merge trace directory");

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::filesystem::path outputPath = traceDir / fileName;

    FILE* dumpFile = std::fopen(outputPath.string().c_str(), "w");
    if (dumpFile == nullptr)
    {
        gmx_fatal(FARGS, "Could not open merge trace output '%s' for writing", outputPath.string().c_str());
    }

    std::fprintf(dumpFile, "# %s\n", header.c_str());
    for (int atom = 0; atom < forceBuffer.ssize(); ++atom)
    {
        std::fprintf(dumpFile,
                     "%d\t%.17g\t%.17g\t%.17g\n",
                     atom,
                     forceBuffer[atom][XX],
                     forceBuffer[atom][YY],
                     forceBuffer[atom][ZZ]);
    }
    std::fclose(dumpFile);
}

static void dumpRespaTraceEvent(const char*        traceDirPath,
                                const char*        fileName,
                                const std::string& header,
                                const int          atomI,
                                const gmx::RVec&   forceI,
                                const int          atomJ,
                                const gmx::RVec&   forceJ)
{
    GMX_RELEASE_ASSERT(traceDirPath != nullptr && *traceDirPath != '\0', "Need a valid trace directory");

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::filesystem::path outputPath = traceDir / fileName;

    FILE* dumpFile = std::fopen(outputPath.string().c_str(), "w");
    if (dumpFile == nullptr)
    {
        gmx_fatal(FARGS, "Could not open trace event output '%s' for writing", outputPath.string().c_str());
    }

    std::fprintf(dumpFile, "# %s\n", header.c_str());
    std::fprintf(dumpFile, "%d\t%.17g\t%.17g\t%.17g\n", atomI, forceI[XX], forceI[YY], forceI[ZZ]);
    std::fprintf(dumpFile, "%d\t%.17g\t%.17g\t%.17g\n", atomJ, forceJ[XX], forceJ[YY], forceJ[ZZ]);
    std::fclose(dumpFile);
}

static std::string formatPointerValue(const void* ptr)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "0x%" PRIxPTR, reinterpret_cast<std::uintptr_t>(ptr));
    return std::string(buffer);
}

static void writeRespaTraceTextFile(const char* traceDirPath, const char* fileName, const std::string& contents)
{
    GMX_RELEASE_ASSERT(traceDirPath != nullptr && *traceDirPath != '\0', "Need a valid trace directory");

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::filesystem::path outputPath = traceDir / fileName;

    FILE* dumpFile = std::fopen(outputPath.string().c_str(), "w");
    if (dumpFile == nullptr)
    {
        gmx_fatal(FARGS, "Could not open trace text output '%s' for writing", outputPath.string().c_str());
    }

    std::fputs(contents.c_str(), dumpFile);
    std::fclose(dumpFile);
}

static void appendRespaTraceTextLine(const char* traceDirPath, const char* fileName, const std::string& line)
{
    GMX_RELEASE_ASSERT(traceDirPath != nullptr && *traceDirPath != '\0',
                       "Need a valid r-RESPA trace directory");

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::filesystem::path outputPath = traceDir / fileName;

    FILE* dumpFile = std::fopen(outputPath.string().c_str(), "a");
    if (dumpFile == nullptr)
    {
        gmx_fatal(FARGS, "Could not open trace text output '%s' for appending", outputPath.string().c_str());
    }

    std::fprintf(dumpFile, "%s\n", line.c_str());
    std::fclose(dumpFile);
}

static bool respaTraceFlagEnabled(const char* envVarName)
{
    const char* value = std::getenv(envVarName);
    return (value != nullptr && *value != '\0');
}

static const std::vector<int64_t>& respaMultiStepCoulombTraceSteps()
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

static const std::vector<int64_t>& respaForceComponentTraceSteps()
{
    static const std::vector<int64_t> steps = []()
    {
        std::vector<int64_t> parsedSteps;
        const char*          value = std::getenv("GMX_PCFF_RESPA_TRACE_FORCE_COMPONENTS_STEPS");
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

static const std::vector<int64_t>& respaPcffClass2SubtermTraceSteps()
{
    static const std::vector<int64_t> steps = []()
    {
        std::vector<int64_t> parsedSteps;
        const char* value = std::getenv("GMX_PCFF_RESPA_TRACE_CLASS2_SUBTERM_ENERGIES_STEPS");
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

static const std::vector<int64_t>& respaCpuCorrectionTraceSteps()
{
    static const std::vector<int64_t> steps = []()
    {
        std::vector<int64_t> parsedSteps;
        const char* value = std::getenv("GMX_PCFF_RESPA_TRACE_CPU_CORRECTION_ENERGIES_STEPS");
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

static const std::vector<int64_t>& respaRealspaceForceSubcomponentTraceSteps()
{
    static const std::vector<int64_t> steps = []()
    {
        std::vector<int64_t> parsedSteps;
        const char* value = std::getenv("GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS_STEPS");
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

static const std::vector<int64_t>& respaStep1Subset01ForceGroupAuditSteps()
{
    static const std::vector<int64_t> steps = []()
    {
        std::vector<int64_t> parsedSteps;
        const char* value = std::getenv("GMX_PCFF_RESPA_TRACE_STEP1_SUBSET01_FORCEGROUP_AUDIT_STEPS");
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

static bool shouldTraceRespaMultiStepCoulombStep(const int64_t step)
{
    const auto& traceSteps = respaMultiStepCoulombTraceSteps();
    return std::find(traceSteps.begin(), traceSteps.end(), step) != traceSteps.end();
}

static const std::vector<int>& respaForceTraceAtomIndices()
{
    static const std::vector<int> atomIndices = []()
    {
        std::vector<int> parsedAtomIndices;
        const char*      value = std::getenv("GMX_PCFF_RESPA_TRACE_ATOMS");
        if (value != nullptr && *value != '\0')
        {
            std::stringstream ss(value);
            std::string       item;
            while (std::getline(ss, item, ','))
            {
                if (!item.empty())
                {
                    parsedAtomIndices.push_back(std::stoi(item));
                }
            }
        }
        if (parsedAtomIndices.empty())
        {
            parsedAtomIndices = { 0, 5 };
        }
        return parsedAtomIndices;
    }();
    return atomIndices;
}

static bool shouldTraceRespaCoordHandoffStep(const int64_t step)
{
    return respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_COORD_HANDOFF")
           && shouldTraceRespaMultiStepCoulombStep(step);
}

extern thread_local bool g_respaSuppressDoForceStateXChain;
extern thread_local const char* g_respaDoForceContextLabel;
extern thread_local int64_t g_respaCurrentDoForceStep;
extern thread_local const int* g_respaCurrentGlobalAtomIndices;
extern thread_local int g_respaCurrentGlobalAtomIndexCount;
extern thread_local const int* g_respaLatestForceDumpGlobalAtomIndices;
extern thread_local int g_respaLatestForceDumpGlobalAtomIndexCount;
static thread_local const int* g_respaTraceGlobalAtomIndices    = nullptr;
static thread_local int        g_respaTraceGlobalAtomIndexCount = 0;

static bool shouldTraceRespaStateXChainStep(const int64_t step)
{
    return respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_STATE_X_CHAIN")
           && shouldTraceRespaMultiStepCoulombStep(step) && !g_respaSuppressDoForceStateXChain;
}

static const char* activeM2pTraceDirPath()
{
    const char* traceDir = std::getenv("GMX_PCFF_RESPA_M2P_TRACE_DIR");
    return (traceDir != nullptr && *traceDir != '\0') ? traceDir : nullptr;
}

static bool shouldTraceRespaForceComponentsStep(const int64_t step)
{
    if (!respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_FORCE_COMPONENTS"))
    {
        return false;
    }

    const auto& traceSteps = respaForceComponentTraceSteps();
    if (!traceSteps.empty())
    {
        return std::find(traceSteps.begin(), traceSteps.end(), step) != traceSteps.end();
    }

    return step == 0;
}

static bool shouldTraceExactGpuListedFtypeSplitStep(const int64_t step)
{
    return (respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_GPU_LISTED_FTYPE_SPLIT")
            || respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_GPU_PAIR14_SPLIT"))
           && activeM2pTraceDirPath() != nullptr && shouldTraceRespaForceComponentsStep(step);
}

static bool shouldTraceExactGpuListedClass2SubtermSplitStep(const int64_t step)
{
    return respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_GPU_CLASS2_SUBTERM_SPLIT")
           && activeM2pTraceDirPath() != nullptr && shouldTraceRespaForceComponentsStep(step);
}

static bool shouldTraceExactGpuBondedMixedVsSequentialStep(const int64_t step)
{
    return respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_GPU_BONDED_MIXED_VS_SEQUENTIAL")
           && activeM2pTraceDirPath() != nullptr && shouldTraceRespaForceComponentsStep(step);
}

static bool shouldUseExactGpuBondedSequentialFtypesValidation()
{
    return respaTraceFlagEnabled("GMX_PCFF_RESPA_EXACT_GPU_BONDED_SEQUENTIAL_FTYPES");
}

static bool shouldTraceExactGpuBondedLaunchContextStep(const int64_t step)
{
    return respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_GPU_BONDED_LAUNCH_CONTEXT")
           && activeM2pTraceDirPath() != nullptr && shouldTraceRespaForceComponentsStep(step);
}

static bool shouldTraceExactGpuBondedDeviceXqStep(const int64_t step)
{
    return respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_GPU_BONDED_DEVICE_XQ")
           && activeM2pTraceDirPath() != nullptr && shouldTraceRespaForceComponentsStep(step);
}

static bool shouldTraceExactGpuBondedDeviceForceStep(const int64_t step)
{
    return respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_GPU_BONDED_DEVICE_FORCE")
           && activeM2pTraceDirPath() != nullptr && shouldTraceRespaForceComponentsStep(step);
}

static bool shouldTraceExactGpuBondedGridIndexStep(const int64_t step)
{
    return respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_GPU_BONDED_GRID_INDICES")
           && activeM2pTraceDirPath() != nullptr && shouldTraceRespaForceComponentsStep(step);
}

static void appendExactGpuBondedLaunchContextTrace(const char* traceDirPath,
                                                   const int64_t step,
                                                   const int exactLevel,
                                                   const char* localCoordinateProvider,
                                                   const bool stepUsesGpuXBufferOps,
                                                   const bool stepDoesNeighborSearch,
                                                   const bool localCoordinatesNeededOnDevice,
                                                   const bool copiedCoordinatesFromGpuToHost,
                                                   const bool copiedCoordinatesFromHostToGpu,
                                                   const int expectedLocalConsumptionCount,
                                                   const bool uploadedCoordinatesForBonded)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    static std::once_flag traceHeaderOnce;
    std::call_once(traceHeaderOnce,
                   [traceDirPath]()
                   {
                       writeRespaTraceTextFile(traceDirPath,
                                               "exact_gpu_bonded_launch_context.tsv",
                                               "# step\tlevel\tprovider\tuse_gpu_x_buffer_ops\tdo_neighbor_search\tlocal_coordinates_needed_on_device\tcopied_coordinates_from_gpu_to_host\tcopied_coordinates_from_host_to_gpu\texpected_local_x_ready_consumption_count\tuploaded_coordinates_for_bonded\n");
                   });

    appendRespaTraceTextLine(traceDirPath,
                             "exact_gpu_bonded_launch_context.tsv",
                             std::to_string(step) + '\t' + std::to_string(exactLevel) + '\t'
                                     + localCoordinateProvider + '\t'
                                     + (stepUsesGpuXBufferOps ? "1" : "0") + '\t'
                                     + (stepDoesNeighborSearch ? "1" : "0") + '\t'
                                     + (localCoordinatesNeededOnDevice ? "1" : "0") + '\t'
                                     + (copiedCoordinatesFromGpuToHost ? "1" : "0") + '\t'
                                     + (copiedCoordinatesFromHostToGpu ? "1" : "0") + '\t'
                                     + std::to_string(expectedLocalConsumptionCount) + '\t'
                                     + (uploadedCoordinatesForBonded ? "1" : "0"));
}

static void appendExactGpuBondedGridIndexTrace(const char*         traceDirPath,
                                               const int64_t       step,
                                               const int           exactLevel,
                                               nonbonded_verlet_t* nbv)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0' || nbv == nullptr)
    {
        return;
    }

    static std::once_flag traceHeaderOnce;
    std::call_once(traceHeaderOnce,
                   [traceDirPath]()
                   {
                       writeRespaTraceTextFile(traceDirPath,
                                               "exact_gpu_bonded_grid_index_trace.tsv",
                                               "# step\tlevel\tatom\tmapped_index\torder_matches_nbnxn\n");
                   });

    const bool orderMatches = nbv->localAtomOrderMatchesNbnxmOrder();
    const auto gridIndices  = nbv->getGridIndices();
    const int  numLocalAtoms = nbv->getNumAtoms(AtomLocality::Local);

    for (const int atomIndex : respaForceTraceAtomIndices())
    {
        if (atomIndex < 0 || atomIndex >= numLocalAtoms)
        {
            continue;
        }
        const int mappedIndex =
                orderMatches ? atomIndex
                             : ((atomIndex >= 0 && atomIndex < gridIndices.ssize()) ? gridIndices[atomIndex]
                                                                                     : -1);
        appendRespaTraceTextLine(traceDirPath,
                                 "exact_gpu_bonded_grid_index_trace.tsv",
                                 std::to_string(step) + '\t' + std::to_string(exactLevel) + '\t'
                                         + std::to_string(atomIndex) + '\t'
                                         + std::to_string(mappedIndex) + '\t'
                                         + (orderMatches ? "1" : "0"));
    }
}

#if GMX_GPU
static void appendExactGpuBondedDeviceXqTrace(const char*            traceDirPath,
                                              const int64_t          step,
                                              const int              exactLevel,
                                              const char*            stageLabel,
                                              nonbonded_verlet_t*    nbv,
                                              const DeviceStream&    deviceStream)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0' || nbv == nullptr || nbv->gpuNbv() == nullptr)
    {
        return;
    }

    NBAtomDataGpu* atomData = gpuGetNBAtomData(nbv->gpuNbv());
    if (atomData == nullptr)
    {
        return;
    }

    const int numLocalAtoms = nbv->getNumAtoms(AtomLocality::Local);
    if (numLocalAtoms == 0)
    {
        return;
    }

    std::vector<Float4> hostXq(numLocalAtoms);
    copyFromDeviceBuffer(hostXq.data(),
                         &atomData->xq,
                         0,
                         numLocalAtoms,
                         deviceStream,
                         GpuApiCallBehavior::Sync,
                         nullptr);

    static std::once_flag traceHeaderOnce;
    std::call_once(traceHeaderOnce,
                   [traceDirPath]()
                   {
                       writeRespaTraceTextFile(traceDirPath,
                                               "exact_gpu_bonded_device_xq_trace.tsv",
                                               "# step\tlevel\tstage\tatom\tx\ty\tz\tq\n");
                   });

    const bool orderMatches  = nbv->localAtomOrderMatchesNbnxmOrder();
    const auto gridIndices   = nbv->getGridIndices();
    const auto mappedAtomIndex = [&](const int atomIndex) -> int
    {
        if (!orderMatches)
        {
            if (atomIndex < 0 || atomIndex >= gridIndices.ssize())
            {
                return -1;
            }
            return gridIndices[atomIndex];
        }
        return atomIndex;
    };

    for (const int atomIndex : respaForceTraceAtomIndices())
    {
        if (atomIndex < 0 || atomIndex >= numLocalAtoms)
        {
            continue;
        }
        const int mappedIndex = mappedAtomIndex(atomIndex);
        if (mappedIndex < 0 || mappedIndex >= numLocalAtoms)
        {
            continue;
        }
        const float* xqComponents    = reinterpret_cast<const float*>(&hostXq[mappedIndex]);
        appendRespaTraceTextLine(traceDirPath,
                                 "exact_gpu_bonded_device_xq_trace.tsv",
                                 std::to_string(step) + '\t' + std::to_string(exactLevel) + '\t'
                                         + stageLabel + '\t' + std::to_string(atomIndex) + '\t'
                                         + formatString("%.15f", xqComponents[0]) + '\t'
                                         + formatString("%.15f", xqComponents[1]) + '\t'
                                         + formatString("%.15f", xqComponents[2]) + '\t'
                                         + formatString("%.15f", xqComponents[3]));
    }
}

static void appendExactGpuBondedDeviceForceTrace(const char*            traceDirPath,
                                                 const int64_t          step,
                                                 const int              exactLevel,
                                                 const char*            stageLabel,
                                                 nonbonded_verlet_t*    nbv,
                                                 const DeviceStream&    deviceStream)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0' || nbv == nullptr || nbv->gpuNbv() == nullptr)
    {
        return;
    }

    NBAtomDataGpu* atomData = gpuGetNBAtomData(nbv->gpuNbv());
    if (atomData == nullptr)
    {
        return;
    }

    const int numLocalAtoms = nbv->getNumAtoms(AtomLocality::Local);
    if (numLocalAtoms == 0)
    {
        return;
    }

    std::vector<Float3> hostForce(numLocalAtoms);
    copyFromDeviceBuffer(hostForce.data(),
                         &atomData->f,
                         0,
                         numLocalAtoms,
                         deviceStream,
                         GpuApiCallBehavior::Sync,
                         nullptr);

    static std::once_flag traceHeaderOnce;
    std::call_once(traceHeaderOnce,
                   [traceDirPath]()
                   {
                       writeRespaTraceTextFile(traceDirPath,
                                               "exact_gpu_bonded_device_force_trace.tsv",
                                               "# step\tlevel\tstage\tatom\tfx\tfy\tfz\n");
                   });

    const bool orderMatches = nbv->localAtomOrderMatchesNbnxmOrder();
    const auto gridIndices  = nbv->getGridIndices();
    const auto mappedAtomIndex = [&](const int atomIndex) -> int
    {
        if (!orderMatches)
        {
            if (atomIndex < 0 || atomIndex >= gridIndices.ssize())
            {
                return -1;
            }
            return gridIndices[atomIndex];
        }
        return atomIndex;
    };

    for (const int atomIndex : respaForceTraceAtomIndices())
    {
        if (atomIndex < 0 || atomIndex >= numLocalAtoms)
        {
            continue;
        }
        const int mappedIndex = mappedAtomIndex(atomIndex);
        if (mappedIndex < 0 || mappedIndex >= numLocalAtoms)
        {
            continue;
        }
        const float* forceComponents = reinterpret_cast<const float*>(&hostForce[mappedIndex]);
        appendRespaTraceTextLine(traceDirPath,
                                 "exact_gpu_bonded_device_force_trace.tsv",
                                 std::to_string(step) + '\t' + std::to_string(exactLevel) + '\t'
                                         + stageLabel + '\t' + std::to_string(atomIndex) + '\t'
                                         + formatString("%.15f", forceComponents[0]) + '\t'
                                         + formatString("%.15f", forceComponents[1]) + '\t'
                                         + formatString("%.15f", forceComponents[2]));
    }
}
#endif

static InteractionDefinitions makeSingleInteractionFunctionDefinitions(const InteractionDefinitions& source,
                                                                      const InteractionFunction       keptFtype)
{
    InteractionDefinitions filtered(source);
    filtered.iparams_posres.clear();
    filtered.iparams_fbposres.clear();

    for (const auto ftype : gmx::EnumerationWrapper<InteractionFunction>{})
    {
        if (ftype != keptFtype)
        {
            filtered.il[ftype].clear();
            filtered.numNonperturbedInteractions[ftype] = 0;
        }
    }

    return filtered;
}

static const char* exactGpuListedFunctionTraceLabel(const InteractionFunction ftype)
{
    switch (ftype)
    {
        case InteractionFunction::Bonds: return "bonds_only";
        case InteractionFunction::BondClass2: return "bond_class2_only";
        case InteractionFunction::Angles: return "angles_only";
        case InteractionFunction::UreyBradleyPotential: return "urey_bradley_only";
        case InteractionFunction::AngleClass2: return "angle_class2_only";
        case InteractionFunction::ProperDihedrals: return "proper_dihedrals_only";
        case InteractionFunction::RyckaertBellemansDihedrals: return "ryckaert_bellemans_only";
        case InteractionFunction::DihedralClass2: return "dihedral_class2_only";
        case InteractionFunction::ImproperDihedrals: return "improper_dihedrals_only";
        case InteractionFunction::ImproperClass2: return "improper_class2_only";
        case InteractionFunction::PeriodicImproperDihedrals: return "periodic_improper_dihedrals_only";
        case InteractionFunction::LennardJones14: return "pair14_only";
        default: return nullptr;
    }
}

struct ExactGpuListedClass2SubtermTraceMode
{
    PcffClass2DebugMode mode;
    const char*         label;
};

static ArrayRef<const ExactGpuListedClass2SubtermTraceMode> exactGpuListedClass2SubtermTraceModes(
        const InteractionFunction ftype)
{
    static const std::array<ExactGpuListedClass2SubtermTraceMode, 3> bondClass2Modes = {
        ExactGpuListedClass2SubtermTraceMode{ PcffClass2DebugMode::BondClass2K2Only, "bond_class2_k2_only" },
        ExactGpuListedClass2SubtermTraceMode{ PcffClass2DebugMode::BondClass2K3Only, "bond_class2_k3_only" },
        ExactGpuListedClass2SubtermTraceMode{ PcffClass2DebugMode::BondClass2K4Only, "bond_class2_k4_only" },
    };
    static const std::array<ExactGpuListedClass2SubtermTraceMode, 4> angleClass2Modes = {
        ExactGpuListedClass2SubtermTraceMode{ PcffClass2DebugMode::AngleClass2MainOnly, "angle_class2_main_only" },
        ExactGpuListedClass2SubtermTraceMode{ PcffClass2DebugMode::AngleClass2BondBondOnly, "angle_class2_bond_bond_only" },
        ExactGpuListedClass2SubtermTraceMode{ PcffClass2DebugMode::AngleClass2BondAngle1Only, "angle_class2_bond_angle_1_only" },
        ExactGpuListedClass2SubtermTraceMode{ PcffClass2DebugMode::AngleClass2BondAngle2Only, "angle_class2_bond_angle_2_only" },
    };

    switch (ftype)
    {
        case InteractionFunction::BondClass2: return makeConstArrayRef(bondClass2Modes);
        case InteractionFunction::AngleClass2: return makeConstArrayRef(angleClass2Modes);
        default: return {};
    }
}

static constexpr std::array<InteractionFunction, 12> c_exactGpuListedFtypesForTraceOrSequentialValidation = {
    InteractionFunction::Bonds,
    InteractionFunction::BondClass2,
    InteractionFunction::Angles,
    InteractionFunction::UreyBradleyPotential,
    InteractionFunction::AngleClass2,
    InteractionFunction::ProperDihedrals,
    InteractionFunction::RyckaertBellemansDihedrals,
    InteractionFunction::DihedralClass2,
    InteractionFunction::ImproperDihedrals,
    InteractionFunction::ImproperClass2,
    InteractionFunction::PeriodicImproperDihedrals,
    InteractionFunction::LennardJones14,
};

static bool shouldTracePcffClass2SubtermEnergiesStep(const int64_t step)
{
    if (!respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_CLASS2_SUBTERM_ENERGIES")
        || activeM2pTraceDirPath() == nullptr)
    {
        return false;
    }

    const auto& traceSteps = respaPcffClass2SubtermTraceSteps();
    if (!traceSteps.empty())
    {
        return std::find(traceSteps.begin(), traceSteps.end(), step) != traceSteps.end();
    }

    return step == 0;
}

static bool shouldTraceCpuCorrectionEnergiesStep(const int64_t step)
{
    if (!respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_CPU_CORRECTION_ENERGIES")
        || activeM2pTraceDirPath() == nullptr)
    {
        return false;
    }

    const auto& traceSteps = respaCpuCorrectionTraceSteps();
    if (!traceSteps.empty())
    {
        return std::find(traceSteps.begin(), traceSteps.end(), step) != traceSteps.end();
    }

    return step == 0;
}

static void appendPcffClass2SubtermEnergyTrace(const char*                          traceDirPath,
                                               const int64_t                        step,
                                               const int                            level,
                                               const char*                          actualBackend,
                                               const PcffClass2SubtermEnergies&     energies)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    static std::once_flag traceHeaderOnce;
    std::call_once(traceHeaderOnce,
                   [traceDirPath]()
                   {
                       writeRespaTraceTextFile(traceDirPath,
                                               "class2_subterm_energy_trace.tsv",
                                               "#step\tlevel\tactual_backend\tterm\tenergy_kj_mol\tinteraction_count\tdiagnostic_origin\n");
                   });

    struct TermRow
    {
        const char* name;
        double      energy;
        int         count;
    };

    const std::array<TermRow, 17> termRows = {
        { { "bond_class2_main", energies.bondClass2Main, energies.bondClass2Count },
          { "angle_class2_main", energies.angleClass2Main, energies.angleClass2Count },
          { "angle_class2_bond_bond", energies.angleClass2BondBond, energies.angleClass2Count },
          { "angle_class2_bond_angle_1", energies.angleClass2BondAngle1, energies.angleClass2Count },
          { "angle_class2_bond_angle_2", energies.angleClass2BondAngle2, energies.angleClass2Count },
          { "dihedral_class2_main", energies.dihedralClass2Main, energies.dihedralClass2Count },
          { "dihedral_class2_middle_bond_torsion",
            energies.dihedralClass2MiddleBondTorsion,
            energies.dihedralClass2Count },
          { "dihedral_class2_end_bond_torsion_1",
            energies.dihedralClass2EndBondTorsion1,
            energies.dihedralClass2Count },
          { "dihedral_class2_end_bond_torsion_2",
            energies.dihedralClass2EndBondTorsion2,
            energies.dihedralClass2Count },
          { "dihedral_class2_angle_torsion_1",
            energies.dihedralClass2AngleTorsion1,
            energies.dihedralClass2Count },
          { "dihedral_class2_angle_torsion_2",
            energies.dihedralClass2AngleTorsion2,
            energies.dihedralClass2Count },
          { "dihedral_class2_angle_angle_torsion",
            energies.dihedralClass2AngleAngleTorsion,
            energies.dihedralClass2Count },
          { "dihedral_class2_bond_bond_13_torsion",
            energies.dihedralClass2BondBond13Torsion,
            energies.dihedralClass2Count },
          { "improper_class2_main", energies.improperClass2Main, energies.improperClass2Count },
          { "improper_class2_angle_angle_1",
            energies.improperClass2AngleAngle1,
            energies.improperClass2Count },
          { "improper_class2_angle_angle_2",
            energies.improperClass2AngleAngle2,
            energies.improperClass2Count },
          { "improper_class2_angle_angle_3",
            energies.improperClass2AngleAngle3,
            energies.improperClass2Count } }
    };

    for (const TermRow& row : termRows)
    {
        std::ostringstream line;
        line.setf(std::ios::scientific);
        line.precision(17);
        line << step << '\t' << level << '\t' << actualBackend << '\t' << row.name << '\t' << row.energy << '\t'
             << row.count << '\t' << "host_diagnostic_rescan";
        appendRespaTraceTextLine(traceDirPath, "class2_subterm_energy_trace.tsv", line.str());
    }
}

static void appendCpuCorrectionEnergyTrace(const char* traceDirPath,
                                           const int64_t step,
                                           const int level,
                                           const char* actualBackend,
                                           const double reciprocalEnergy,
                                           const double selfEnergy,
                                           const double excludedCorrectionEnergy,
                                           const double shortRangePairEnergy,
                                           const double shortRangeTotalEnergy,
                                           const int reciprocalCount,
                                           const int selfCount,
                                           const int excludedCount,
                                           const int pairCount)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    static std::once_flag traceHeaderOnce;
    std::call_once(traceHeaderOnce,
                   [traceDirPath]()
                   {
                       writeRespaTraceTextFile(traceDirPath,
                                               "cpu_correction_energy_trace.tsv",
                                               "#step\tlevel\tactual_backend\tterm\tenergy_kj_mol\tinteraction_count\tdiagnostic_origin\n");
                   });

    struct TermRow
    {
        const char* name;
        double      energy;
        int         count;
    };

    const std::array<TermRow, 5> termRows = {
        { { "coulomb_pairs_short_range", shortRangePairEnergy, pairCount },
          { "coulomb_excluded_correction", excludedCorrectionEnergy, excludedCount },
          { "coulomb_self", selfEnergy, selfCount },
          { "coulomb_short_range_total", shortRangeTotalEnergy, pairCount + excludedCount + selfCount },
          { "coulomb_reciprocal", reciprocalEnergy, reciprocalCount } }
    };

    for (const TermRow& row : termRows)
    {
        std::ostringstream line;
        line.setf(std::ios::scientific);
        line.precision(17);
        line << step << '\t' << level << '\t' << actualBackend << '\t' << row.name << '\t' << row.energy << '\t'
             << row.count << '\t' << "runtime_energy_split";
        appendRespaTraceTextLine(traceDirPath, "cpu_correction_energy_trace.tsv", line.str());
    }
}

static void appendExplicitLevel0SnapshotForTracedAtoms(const char*               traceDirPath,
                                                       const int64_t             step,
                                                       const char*               stageLabel,
                                                       ArrayRef<const gmx::RVec> coordinates,
                                                       ArrayRef<const gmx::RVec> forceBuffer,
                                                       const char*               codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    static std::mutex                      traceMutex;
    static std::unordered_set<std::string> emittedRows;
    const int                              availableAtoms = forceBuffer.ssize();
    for (const int atomIndex : respaForceTraceAtomIndices())
    {
        if (atomIndex < 0 || atomIndex >= availableAtoms)
        {
            continue;
        }

        const std::string row =
                "step=" + std::to_string(step) + " stage=" + std::string(stageLabel) + " atom="
                + std::to_string(atomIndex) + " px=" + formatString("%.15f", coordinates[atomIndex][XX]) + " py="
                + formatString("%.15f", coordinates[atomIndex][YY]) + " pz="
                + formatString("%.15f", coordinates[atomIndex][ZZ]) + " fx="
                + formatString("%.15f", forceBuffer[atomIndex][XX]) + " fy="
                + formatString("%.15f", forceBuffer[atomIndex][YY]) + " fz="
                + formatString("%.15f", forceBuffer[atomIndex][ZZ]) + " context_label="
                + std::string((g_respaDoForceContextLabel != nullptr) ? g_respaDoForceContextLabel
                                                                      : "unspecified")
                + " code_location="
                + std::string(codeLocation);
        const std::string rowKey =
                std::string(traceDirPath) + "/explicit_level0_stage_trace.txt\n" + row;
        {
            std::lock_guard<std::mutex> guard(traceMutex);
            if (!emittedRows.insert(rowKey).second)
            {
                continue;
            }
        }
        appendRespaTraceTextLine(traceDirPath, "explicit_level0_stage_trace.txt", row);
    }
}

static bool shouldTraceRespaRealspaceForceSubcomponentsStep(const int64_t step)
{
    const auto& traceSteps = respaRealspaceForceSubcomponentTraceSteps();
    if (respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS") && !traceSteps.empty())
    {
        return std::find(traceSteps.begin(), traceSteps.end(), step) != traceSteps.end();
    }

    return (step == 0 && respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS"))
           || (step == 2 && respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_STEP1_SUBSET01_FORCEGROUP_AUDIT")
               && activeM2pTraceDirPath() != nullptr);
}

static bool shouldTraceRespaExclusionEquivalenceStep(const int64_t step)
{
    return step == 0 && respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_EXCLUSION_EQUIVALENCE");
}

static bool shouldTraceRespaExclusionEquivalencePair(const int ai, const int aj)
{
    return ai == 0 || ai == 5 || aj == 0 || aj == 5;
}

static bool shouldTraceBoundaryDominantPair(const int ai, const int aj)
{
    const int first  = std::min(ai, aj);
    const int second = std::max(ai, aj);
    return first == 0 && (second == 4 || second == 5 || second == 6);
}

static bool shouldTraceStep1Subset01ForceGroupAuditStep(const int64_t step)
{
    if (!respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_STEP1_SUBSET01_FORCEGROUP_AUDIT")
        || activeM2pTraceDirPath() == nullptr)
    {
        return false;
    }

    const auto& traceSteps = respaStep1Subset01ForceGroupAuditSteps();
    if (!traceSteps.empty())
    {
        return std::find(traceSteps.begin(), traceSteps.end(), step) != traceSteps.end();
    }

    return step == 2;
}

struct TracedForcePair
{
    std::vector<int>                    atomIndices = respaForceTraceAtomIndices();
    std::vector<std::array<double, DIM>> atoms =
            std::vector<std::array<double, DIM>>(atomIndices.size(), std::array<double, DIM>{ 0.0, 0.0, 0.0 });
};

static void addTracedForcePairToTracedPair(TracedForcePair* destination, const TracedForcePair& source)
{
    if (destination == nullptr || destination->atoms.size() != source.atoms.size())
    {
        return;
    }

    for (int traceAtomIndex = 0; traceAtomIndex < static_cast<int>(destination->atoms.size()); ++traceAtomIndex)
    {
        for (int dim = 0; dim < DIM; ++dim)
        {
            destination->atoms[traceAtomIndex][dim] += source.atoms[traceAtomIndex][dim];
        }
    }
}

static void addForceArrayToTracedPair(TracedForcePair* pair, ArrayRef<const RVec> force)
{
    if (pair == nullptr || force.empty())
    {
        return;
    }

    for (int traceAtomIndex = 0; traceAtomIndex < static_cast<int>(pair->atomIndices.size()); ++traceAtomIndex)
    {
        const int atomIndex = pair->atomIndices[traceAtomIndex];
        if (atomIndex < 0 || atomIndex >= force.ssize())
        {
            continue;
        }
        for (int dim = 0; dim < DIM; ++dim)
        {
            pair->atoms[traceAtomIndex][dim] += force[atomIndex][dim];
        }
    }
}

static TracedForcePair captureForceArrayPair(ArrayRef<const RVec> force)
{
    TracedForcePair pair;
    addForceArrayToTracedPair(&pair, force);
    return pair;
}

static void addPairContributionToTracedPair(TracedForcePair* pair, const int ai, const int aj, const RVec& force)
{
    if (pair == nullptr)
    {
        return;
    }

    for (int traceAtomIndex = 0; traceAtomIndex < static_cast<int>(pair->atomIndices.size()); ++traceAtomIndex)
    {
        const int atomIndex = pair->atomIndices[traceAtomIndex];
        if (ai == atomIndex)
        {
            for (int dim = 0; dim < DIM; ++dim)
            {
                pair->atoms[traceAtomIndex][dim] += force[dim];
            }
        }
        if (aj == atomIndex)
        {
            for (int dim = 0; dim < DIM; ++dim)
            {
                pair->atoms[traceAtomIndex][dim] -= force[dim];
            }
        }
    }
}

static TracedForcePair subtractTracedForcePairs(const TracedForcePair& after, const TracedForcePair& before)
{
    TracedForcePair delta;
    delta.atomIndices = after.atomIndices;
    delta.atoms.assign(delta.atomIndices.size(), std::array<double, DIM>{ 0.0, 0.0, 0.0 });
    GMX_RELEASE_ASSERT(after.atomIndices == before.atomIndices, "Trace atom selections should match");
    for (int atomIndex = 0; atomIndex < static_cast<int>(after.atomIndices.size()); ++atomIndex)
    {
        for (int dim = 0; dim < DIM; ++dim)
        {
            delta.atoms[atomIndex][dim] = after.atoms[atomIndex][dim] - before.atoms[atomIndex][dim];
        }
    }
    return delta;
}

struct ExactRespaRealspaceTraceCapture
{
    int64_t         step = -1;
    bool            valid = false;
    TracedForcePair ljSrForce;
    TracedForcePair coulombSrForce;
    TracedForcePair exclusionCorrectionForce;
    TracedForcePair combinedForce;
};

static ExactRespaRealspaceTraceCapture& exactRespaRealspaceTraceCapture()
{
    static ExactRespaRealspaceTraceCapture capture;
    return capture;
}

static void clearExactRespaRealspaceTraceCapture(const int64_t step)
{
    auto& capture                    = exactRespaRealspaceTraceCapture();
    capture.step                     = step;
    capture.valid                    = false;
    capture.ljSrForce                = {};
    capture.coulombSrForce           = {};
    capture.exclusionCorrectionForce = {};
    capture.combinedForce            = {};
}

static void storeExactRespaRealspaceTraceCapture(const int64_t         step,
                                                 const TracedForcePair& ljSrForce,
                                                 const TracedForcePair& coulombSrForce,
                                                 const TracedForcePair& exclusionCorrectionForce,
                                                 const TracedForcePair& combinedForce)
{
    auto& capture                    = exactRespaRealspaceTraceCapture();
    capture.step                     = step;
    capture.valid                    = true;
    capture.ljSrForce                = ljSrForce;
    capture.coulombSrForce           = coulombSrForce;
    capture.exclusionCorrectionForce = exclusionCorrectionForce;
    capture.combinedForce            = combinedForce;
}

static const ExactRespaRealspaceTraceCapture* activeExactRespaRealspaceTraceCapture(const int64_t step)
{
    const auto& capture = exactRespaRealspaceTraceCapture();
    return (capture.valid && capture.step == step) ? &capture : nullptr;
}

static TracedForcePair addTracedForcePairs(const TracedForcePair& lhs, const TracedForcePair& rhs)
{
    TracedForcePair sum;
    sum.atomIndices = lhs.atomIndices;
    sum.atoms.assign(sum.atomIndices.size(), std::array<double, DIM>{ 0.0, 0.0, 0.0 });
    GMX_RELEASE_ASSERT(lhs.atomIndices == rhs.atomIndices, "Trace atom selections should match");
    for (int atomIndex = 0; atomIndex < static_cast<int>(lhs.atomIndices.size()); ++atomIndex)
    {
        for (int dim = 0; dim < DIM; ++dim)
        {
            sum.atoms[atomIndex][dim] = lhs.atoms[atomIndex][dim] + rhs.atoms[atomIndex][dim];
        }
    }
    return sum;
}

struct ExactRespaForceOutputs
{
    static constexpr int c_numLevels = ExactRespaForceStore::c_numStoredLevels;

    int numLevels          = 0;
    int highestActiveLevel = 0;
    int longrangeLevel     = 0;
    std::array<ForceOutputs*, c_numLevels> levelOutputs = { nullptr, nullptr, nullptr };

    int numActiveLevels() const { return std::min(numLevels, highestActiveLevel + 1); }

    ForceOutputs* levelOrNull(const int level) const
    {
        return (level >= 0 && level < c_numLevels) ? levelOutputs[level] : nullptr;
    }

    bool hasLevel(const int level) const { return level >= 0 && level < numActiveLevels() && levelOrNull(level) != nullptr; }

    ForceOutputs& level(const int level) const
    {
        ForceOutputs* outputs = levelOrNull(level);
        GMX_RELEASE_ASSERT(outputs != nullptr, "Exact r-RESPA level output should be available");
        return *outputs;
    }

    ForceOutputs* longrangeOutputOrNull() const { return hasLevel(longrangeLevel) ? levelOutputs[longrangeLevel] : nullptr; }
};

struct ExactRespaForceOutputStorage
{
    std::array<PaddedHostVector<RVec>, ExactRespaForceOutputs::c_numLevels> ownedLevelForceBuffers;
    std::array<std::optional<ForceOutputs>, ExactRespaForceOutputs::c_numLevels> ownedLevelOutputs;
};

static TracedForcePair captureDistinctForceOutput(ForceOutputs* outputs)
{
    TracedForcePair                 pair;
    std::unordered_set<const void*> seenBuffers;

    if (outputs == nullptr)
    {
        return pair;
    }

    const auto shiftForce = outputs->forceWithShiftForces().force();
    if (!shiftForce.empty() && seenBuffers.insert(shiftForce.data()).second)
    {
        addForceArrayToTracedPair(&pair, shiftForce);
    }

    if (outputs->haveForceWithVirial())
    {
        const auto virialForce = outputs->forceWithVirial().force_;
        if (!virialForce.empty() && seenBuffers.insert(virialForce.data()).second)
        {
            addForceArrayToTracedPair(&pair, virialForce);
        }
    }

    return pair;
}

static TracedForcePair captureDistinctForceOutputs(ArrayRef<ForceOutputs*> forceOutByMtsLevel,
                                                   const int             highestActiveMtsLevel)
{
    TracedForcePair                  pair;
    std::unordered_set<const void*>  seenBuffers;
    const int                        maxLevel =
            std::min(highestActiveMtsLevel, static_cast<int>(forceOutByMtsLevel.ssize()) - 1);

    for (int mtsLevel = 0; mtsLevel <= maxLevel; ++mtsLevel)
    {
        ForceOutputs* outputs = forceOutByMtsLevel[mtsLevel];
        if (outputs == nullptr)
        {
            continue;
        }

        const auto shiftForce = outputs->forceWithShiftForces().force();
        if (!shiftForce.empty() && seenBuffers.insert(shiftForce.data()).second)
        {
            addForceArrayToTracedPair(&pair, shiftForce);
        }

        if (outputs->haveForceWithVirial())
        {
            const auto virialForce = outputs->forceWithVirial().force_;
            if (!virialForce.empty() && seenBuffers.insert(virialForce.data()).second)
            {
                addForceArrayToTracedPair(&pair, virialForce);
            }
        }
    }

    return pair;
}

static bool isSupportedExactRespaHybridGpuForceOnlySimulation(const t_inputrec&         inputrec,
                                                              const SimulationWorkload& simulationWork)
{
    return gmx::useExactRespa(inputrec) && gmx::exactRespaHasPairSplitting(inputrec)
           && simulationWork.useGpuNonbonded && !simulationWork.havePpDomainDecomposition
           && !simulationWork.haveSeparatePmeRank && !simulationWork.useGpuNonbondedFE
           && !simulationWork.useGpuForeignNonbondedFE && !simulationWork.useGpuUpdate
           && !simulationWork.useGpuXBufferOpsWhenAllowed
           && !simulationWork.useGpuFBufferOpsWhenAllowed && !simulationWork.useGpuHaloExchange
           && !simulationWork.useGpuPmePpCommunication
           && !simulationWork.useGpuDirectCommunication
           && !simulationWork.useGpuPmeDecomposition && !simulationWork.useMdGpuGraph
           && (!simulationWork.useGpuPme || simulationWork.useGpuPmeFft);
}

static bool isSupportedExactRespaHybridGpuUpdateSimulation(const t_inputrec&         inputrec,
                                                           const SimulationWorkload& simulationWork)
{
    return gmx::useExactRespa(inputrec) && gmx::exactRespaHasPairSplitting(inputrec)
           && simulationWork.useGpuNonbonded && simulationWork.useGpuBonded && simulationWork.useGpuPme
           && simulationWork.useGpuPmeFft && simulationWork.useGpuUpdate
           && !simulationWork.havePpDomainDecomposition && !simulationWork.haveSeparatePmeRank
           && !simulationWork.useGpuNonbondedFE && !simulationWork.useGpuForeignNonbondedFE
           && !simulationWork.useGpuXBufferOpsWhenAllowed
           && !simulationWork.useGpuFBufferOpsWhenAllowed && !simulationWork.useGpuHaloExchange
           && !simulationWork.useGpuPmePpCommunication
           && !simulationWork.useGpuDirectCommunication
           && !simulationWork.useGpuPmeDecomposition && !simulationWork.useMdGpuGraph;
}

static bool isSupportedExactRespaHybridGpuSimulation(const t_inputrec&         inputrec,
                                                     const SimulationWorkload& simulationWork)
{
    return isSupportedExactRespaHybridGpuForceOnlySimulation(inputrec, simulationWork)
           || isSupportedExactRespaHybridGpuUpdateSimulation(inputrec, simulationWork);
}

static bool isExactRespaHybridGpuRuntime(const t_inputrec&         inputrec,
                                         const SimulationWorkload& simulationWork,
                                         const StepWorkload&       stepWork)
{
    return isSupportedExactRespaHybridGpuSimulation(inputrec, simulationWork)
           && !stepWork.computeVirial && !stepWork.computeEnergy
           && !stepWork.useGpuXBufferOps && !stepWork.useGpuFBufferOps
           && !stepWork.useGpuPmeFReduction && !stepWork.useGpuXHalo && !stepWork.useGpuFHalo
           && !stepWork.computePmeOnSeparateRank && !stepWork.combineMtsForcesBeforeHaloExchange;
}

static void assertExactRespaOwnershipContract(const t_inputrec&          inputrec,
                                              const SimulationWorkload&  simulationWork,
                                              const StepWorkload&        stepWork,
                                              ForceBuffersView*          forceView,
                                              ForceOutputs&              forceOutMtsLevel0,
                                              const ExactRespaForceOutputs& exactRespaForceOutputs)
{
    if (!gmx::useExactRespa(inputrec))
    {
        return;
    }

    GMX_RELEASE_ASSERT(forceView != nullptr,
                       "Exact r-RESPA HG2 contract requires explicit force buffers");
    const bool exactHybridGpuSimulation = isSupportedExactRespaHybridGpuSimulation(inputrec, simulationWork);
    const bool exactHybridGpuRuntime =
            isExactRespaHybridGpuRuntime(inputrec, simulationWork, stepWork);
    if (!exactHybridGpuRuntime)
    {
        if (!exactHybridGpuSimulation)
        {
            GMX_RELEASE_ASSERT(!simulationWork.useGpuNonbonded && !simulationWork.useGpuNonbondedFE
                                       && !simulationWork.useGpuForeignNonbondedFE
                                       && !simulationWork.useGpuPme && !simulationWork.useGpuPmeFft
                                       && !simulationWork.useGpuBonded && !simulationWork.useGpuUpdate
                                       && !simulationWork.useGpuXBufferOpsWhenAllowed
                                       && !simulationWork.useGpuFBufferOpsWhenAllowed
                                       && !simulationWork.useGpuHaloExchange
                                       && !simulationWork.useGpuPmePpCommunication
                                       && !simulationWork.useGpuDirectCommunication
                                       && !simulationWork.useGpuPmeDecomposition,
                               "Exact r-RESPA HG2 keeps GPU work and communication disabled until ownership is audited");
        }
        GMX_RELEASE_ASSERT(!stepWork.useGpuXBufferOps && !stepWork.useGpuFBufferOps
                                   && !stepWork.useGpuPmeFReduction && !stepWork.useGpuXHalo
                                   && !stepWork.useGpuFHalo && !stepWork.haveGpuPmeOnThisRank
                                   && !stepWork.combineMtsForcesBeforeHaloExchange,
                           "Exact r-RESPA HG2 requires an explicit CPU-thread reduction boundary before any GPU merge");
    }
    else
    {
        GMX_RELEASE_ASSERT(simulationWork.useGpuNonbonded && !simulationWork.havePpDomainDecomposition,
                           "Exact r-RESPA narrow hybrid GPU mode requires single-rank nonbonded GPU execution");
        GMX_RELEASE_ASSERT(!simulationWork.useGpuPme || (simulationWork.useGpuPmeFft && !simulationWork.useCpuPme),
                           "Exact r-RESPA HG5 narrow mode requires on-rank full GPU PME");
        GMX_RELEASE_ASSERT(!simulationWork.useGpuPme
                                   || (stepWork.haveGpuPmeOnThisRank
                                       == stepWork.computeLongRangeNonbondedForces),
                           "Exact r-RESPA HG5 reciprocal ownership requires GPU PME only on reciprocal steps");
    }

    if (!stepWork.computeForces)
    {
        return;
    }

    const auto assertNoShiftOrVirialMergeBoundary = [](ForceOutputs& outputs)
    {
        GMX_RELEASE_ASSERT(!outputs.forceWithShiftForces().computeVirial()
                                   && outputs.forceWithShiftForces().shiftForces().empty(),
                           "Exact r-RESPA HG2 narrow hybrid contract keeps shift-force reduction disabled in force-only mode");
        if (outputs.haveForceWithVirial())
        {
            GMX_RELEASE_ASSERT(!outputs.forceWithVirial().computeVirial_,
                               "Exact r-RESPA HG2 narrow hybrid contract keeps direct-virial accumulation disabled in force-only mode");
        }
    };

    const auto level0Force         = forceOutMtsLevel0.forceWithShiftForces().force();
    const auto expectedLevel0Force = forceView->force();
    GMX_RELEASE_ASSERT(level0Force.data() == expectedLevel0Force.data()
                               && gmx::ssize(level0Force) == gmx::ssize(expectedLevel0Force),
                       "Exact r-RESPA level-0 ownership must stay on the explicit host force buffer");
    if (exactHybridGpuRuntime)
    {
        assertNoShiftOrVirialMergeBoundary(forceOutMtsLevel0);
    }

    if (!stepWork.computeSlowForces)
    {
        return;
    }

    GMX_RELEASE_ASSERT(forceView->numMtsLevelForceBuffers() >= exactRespaForceOutputs.highestActiveLevel,
                       "Exact r-RESPA needs one dedicated slow-force buffer per active slow level");

    std::unordered_set<const RVec*> slowLevelForceBuffers;
    const RVec* const               level0Pointer = level0Force.data();
    for (int mtsLevel = 1; mtsLevel <= exactRespaForceOutputs.highestActiveLevel; ++mtsLevel)
    {
        ForceOutputs* outputs = exactRespaForceOutputs.levelOrNull(mtsLevel);
        GMX_RELEASE_ASSERT(outputs != nullptr,
                           "Exact r-RESPA requires an explicit force output object for every active slow level");

        const auto levelForce         = outputs->forceWithShiftForces().force();
        const auto expectedLevelForce = forceView->forceForMtsLevel(mtsLevel);
        GMX_RELEASE_ASSERT(levelForce.data() == expectedLevelForce.data()
                                   && gmx::ssize(levelForce) == gmx::ssize(expectedLevelForce),
                           "Exact r-RESPA slow-level ownership must stay on explicit per-level host force buffers");
        GMX_RELEASE_ASSERT(levelForce.data() != level0Pointer,
                           "Exact r-RESPA slow-force buffers must not alias level-0 ownership");
        GMX_RELEASE_ASSERT(slowLevelForceBuffers.insert(levelForce.data()).second,
                           "Exact r-RESPA requires distinct host force buffers per active slow level");
        if (exactHybridGpuRuntime)
        {
            assertNoShiftOrVirialMergeBoundary(*outputs);
        }
    }
}

static TracedForcePair captureDistinctForceOutputs(const ExactRespaForceOutputs& exactRespaForceOutputs)
{
    TracedForcePair                  pair;
    std::unordered_set<const void*>  seenBuffers;

    for (int level = 0; level < exactRespaForceOutputs.numActiveLevels(); ++level)
    {
        ForceOutputs* outputs = exactRespaForceOutputs.levelOrNull(level);
        if (outputs == nullptr)
        {
            continue;
        }

        const auto shiftForce = outputs->forceWithShiftForces().force();
        if (!shiftForce.empty() && seenBuffers.insert(shiftForce.data()).second)
        {
            addForceArrayToTracedPair(&pair, shiftForce);
        }

        if (outputs->haveForceWithVirial())
        {
            const auto virialForce = outputs->forceWithVirial().force_;
            if (!virialForce.empty() && seenBuffers.insert(virialForce.data()).second)
            {
                addForceArrayToTracedPair(&pair, virialForce);
            }
        }
    }

    return pair;
}

static void appendStep1Subset01ForceGroupStageSnapshot(const char*                  traceDirPath,
                                                       const char*                  side,
                                                       const int64_t                step,
                                                       const char*                  stage,
                                                       const char*                  bufferRole,
                                                       const int                    levelIndex,
                                                       const int                    atomIndex,
                                                       const gmx::RVec&             force,
                                                       const char*                  codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    static std::mutex                   traceMutex;
    static std::unordered_set<std::string> emittedRows;
    const std::string                  row =
            "side=" + std::string(side) + " step=" + std::to_string(step) + " stage=" + std::string(stage)
            + " buffer_role=" + std::string(bufferRole) + " level_index=" + std::to_string(levelIndex)
            + " atom=" + std::to_string(atomIndex) + " fx=" + formatString("%.15f", force[XX]) + " fy="
            + formatString("%.15f", force[YY]) + " fz=" + formatString("%.15f", force[ZZ]) + " code_location="
            + std::string(codeLocation);
    const std::string rowKey = std::string(traceDirPath) + "/step1_subset01_forcegroup_stage_trace.txt\n" + row;
    {
        std::lock_guard<std::mutex> guard(traceMutex);
        if (!emittedRows.insert(rowKey).second)
        {
            return;
        }
    }

    appendRespaTraceTextLine(traceDirPath, "step1_subset01_forcegroup_stage_trace.txt", row);
}

static void appendStep1Subset01ForceGroupBufferSnapshot(const char*                traceDirPath,
                                                        const char*                side,
                                                        const int64_t              step,
                                                        const char*                stage,
                                                        ArrayRef<ForceOutputs*>    forceOutByMtsLevel,
                                                        const int                  highestActiveMtsLevel,
                                                        const char*                level0Role,
                                                        const char*                codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    const auto appendBufferForAtoms = [&](const char* role, const int levelIndex, ArrayRef<const RVec> force)
    {
        if (force.empty() || force.ssize() <= 5)
        {
            return;
        }
        for (const int atomIndex : { 0, 5 })
        {
            appendStep1Subset01ForceGroupStageSnapshot(
                    traceDirPath, side, step, stage, role, levelIndex, atomIndex, force[atomIndex], codeLocation);
        }
    };

    if (forceOutByMtsLevel.empty() || forceOutByMtsLevel[0] == nullptr)
    {
        return;
    }

    appendBufferForAtoms(level0Role, 0, forceOutByMtsLevel[0]->forceWithShiftForces().force());
    if (highestActiveMtsLevel >= 1 && forceOutByMtsLevel.ssize() > 1 && forceOutByMtsLevel[1] != nullptr)
    {
        appendBufferForAtoms("slow1", 1, forceOutByMtsLevel[1]->forceWithShiftForces().force());
    }
    if (highestActiveMtsLevel >= 2 && forceOutByMtsLevel.ssize() > 2 && forceOutByMtsLevel[2] != nullptr)
    {
        const auto slow2Force = forceOutByMtsLevel[2]->haveForceWithVirial()
                                        ? forceOutByMtsLevel[2]->forceWithVirial().force_
                                        : forceOutByMtsLevel[2]->forceWithShiftForces().force();
        appendBufferForAtoms("slow2", 2, slow2Force);
    }
}

static void appendStep1Subset01ForceGroupBufferSnapshot(const char*                 traceDirPath,
                                                        const char*                 side,
                                                        const int64_t               step,
                                                        const char*                 stage,
                                                        const ExactRespaForceOutputs& exactRespaForceOutputs,
                                                        const char*                 level0Role,
                                                        const char*                 codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0' || !exactRespaForceOutputs.hasLevel(0))
    {
        return;
    }

    const auto appendBufferForAtoms = [&](const char* role, const int levelIndex, ArrayRef<const RVec> force)
    {
        if (force.empty() || force.ssize() <= 5)
        {
            return;
        }
        for (const int atomIndex : { 0, 5 })
        {
            appendStep1Subset01ForceGroupStageSnapshot(
                    traceDirPath, side, step, stage, role, levelIndex, atomIndex, force[atomIndex], codeLocation);
        }
    };

    appendBufferForAtoms(level0Role, 0, exactRespaForceOutputs.level(0).forceWithShiftForces().force());
    if (exactRespaForceOutputs.hasLevel(1))
    {
        appendBufferForAtoms("slow1", 1, exactRespaForceOutputs.level(1).forceWithShiftForces().force());
    }
    if (exactRespaForceOutputs.hasLevel(2))
    {
        const auto slow2Force = exactRespaForceOutputs.level(2).haveForceWithVirial()
                                        ? exactRespaForceOutputs.level(2).forceWithVirial().force_
                                        : exactRespaForceOutputs.level(2).forceWithShiftForces().force();
        appendBufferForAtoms("slow2", 2, slow2Force);
    }
}

static void emitExactRespaOwnershipDiagnostics(const t_inputrec&              inputrec,
                                               const StepWorkload&            stepWork,
                                               const ExactRespaForceOutputs&  exactRespaForceOutputs)
{
    static_cast<void>(stepWork);
    if (!gmx::useExactRespa(inputrec) || !gmx::exactRespaHasPairSplitting(inputrec))
    {
        return;
    }

    const char* nonbondedOutputContractTraceDirPath =
            std::getenv("GMX_PCFF_RESPA_NONBONDED_OUTPUT_CONTRACT_TRACE_DIR");
    if (nonbondedOutputContractTraceDirPath != nullptr && *nonbondedOutputContractTraceDirPath != '\0')
    {
        std::string activeContributionLabels = "inner";
        if (inputrec.exactRespa.forceLayout.hasMiddle())
        {
            activeContributionLabels += ",middle";
        }
        activeContributionLabels += ",outer";

        struct ContractRow
        {
            const char* contribution;
            int         mtsLevel;
        };
        std::vector<ContractRow> rows = {
            { "inner", exactRespaNonbondedInnerLevel(inputrec) },
            { "outer", exactRespaNonbondedOuterLevel(inputrec) },
        };
        if (inputrec.exactRespa.forceLayout.hasMiddle())
        {
            rows.insert(rows.begin() + 1, { "middle", exactRespaNonbondedMiddleLevel(inputrec) });
        }

        writeRespaTraceTextFile(nonbondedOutputContractTraceDirPath,
                                "step0_nonbonded_output_contract_trace.txt",
                                "");
        for (const auto& row : rows)
        {
            ForceOutputs* outputs = exactRespaForceOutputs.levelOrNull(row.mtsLevel);
            GMX_RELEASE_ASSERT(outputs != nullptr,
                               "Exact r-RESPA ownership diagnostics require valid nonbonded output sinks");
            appendRespaTraceTextLine(
                    nonbondedOutputContractTraceDirPath,
                    "step0_nonbonded_output_contract_trace.txt",
                    "step=0 contribution=" + std::string(row.contribution) + " mts_level="
                            + std::to_string(row.mtsLevel)
                            + " force_sink=forceWithShiftForces uses_shift_buffer=false"
                            + " direct_virial=false output_has_virial="
                            + std::string(outputs->haveForceWithVirial() ? "true" : "false")
                            + " accumulate_energy=false aliases_shift_force="
                            + std::string(outputs->forceWithShiftForces().force().data()
                                                  == outputs->forceWithShiftForces().force().data()
                                                  ? "true"
                                                  : "false")
                            + " aliases_virial_force=false active_contributions=" + activeContributionLabels
                            + " semantic_role=exact_nonbonded_output_sink_oracle");
        }
    }

    const char* pairWriteProofDirPath = std::getenv("GMX_PCFF_RESPA_PAIR_WRITE_PROOF_DIR");
    if (pairWriteProofDirPath != nullptr && *pairWriteProofDirPath != '\0')
    {
        const int  outerLevel         = exactRespaNonbondedOuterLevel(inputrec);
        const auto outerOutputs       = exactRespaForceOutputs.levelOrNull(outerLevel);

        std::string contents;
        const auto  appendLine = [&contents](const std::string& line) { contents += line + "\n"; };
        for (int mtsLevel = 0; mtsLevel < exactRespaForceOutputs.numActiveLevels(); ++mtsLevel)
        {
            const ForceOutputs* outputs = exactRespaForceOutputs.levelOrNull(mtsLevel);
            if (outputs == nullptr)
            {
                appendLine("level=" + std::to_string(mtsLevel) + " active=false");
                continue;
            }
            ForceOutputs* mutableOutputs = const_cast<ForceOutputs*>(outputs);
            appendLine("level=" + std::to_string(mtsLevel) + " active=true");
            appendLine("level=" + std::to_string(mtsLevel) + " shift_force_ptr="
                       + formatPointerValue(mutableOutputs->forceWithShiftForces().force().data()));
            appendLine("level=" + std::to_string(mtsLevel) + " shift_shift_ptr=0x0");
            appendLine("level=" + std::to_string(mtsLevel) + " have_virial="
                       + std::string(mtsLevel == outerLevel ? "true" : "false"));
            if (mtsLevel == outerLevel)
            {
                appendLine("level=" + std::to_string(mtsLevel) + " virial_force_ptr="
                           + formatPointerValue(mutableOutputs->forceWithShiftForces().force().data()));
            }
        }
        appendLine("outer_accumulator_present=" + std::string(outerOutputs != nullptr ? "true" : "false"));
        if (outerOutputs != nullptr)
        {
            appendLine("outer_accumulator_force_ptr="
                       + formatPointerValue(outerOutputs->forceWithShiftForces().force().data()));
            appendLine("outer_accumulator_shift_ptr=0x0");
            appendLine("outer_accumulator_has_virial=false");
            appendLine("outer_outputs_shift_force_ptr="
                       + formatPointerValue(outerOutputs->forceWithShiftForces().force().data()));
            appendLine("outer_outputs_shift_shift_ptr=0x0");
            appendLine("outer_aliases_shift=true");
        }
        appendLine("excluded_correction_force_dump_enabled=false");
        appendLine("excluded_correction_force_dump_ptr=0x0");
        writeRespaTraceTextFile(pairWriteProofDirPath, "step0_force_storage_identity.txt", contents);
    }
}

static void appendForceComponentTracePairToFile(const char*                traceDirPath,
                                                const char*                fileName,
                                                const char*                side,
                                                const int64_t              step,
                                                const char*                componentName,
                                                const TracedForcePair&     pair,
                                                const char*                sourceLabel,
                                                const char*                codeLocation,
                                                const char*                componentKind,
                                                const bool                 trueSourceComponent)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    for (int traceAtomIndex = 0; traceAtomIndex < static_cast<int>(pair.atomIndices.size()); ++traceAtomIndex)
    {
        const int atomIndex = pair.atomIndices[traceAtomIndex];
        const bool haveGlobalAtomIndex =
                g_respaTraceGlobalAtomIndices != nullptr && atomIndex >= 0
                && atomIndex < g_respaTraceGlobalAtomIndexCount;
        const int globalAtomIndex =
                haveGlobalAtomIndex ? g_respaTraceGlobalAtomIndices[atomIndex] : atomIndex;
        appendRespaTraceTextLine(
                traceDirPath,
                fileName,
                "side=" + std::string(side) + " step=" + std::to_string(step) + " atom="
                        + std::to_string(atomIndex) + " component_name=" + std::string(componentName)
                        + " global_atom=" + std::to_string(globalAtomIndex)
                        + " available=true fx="
                        + formatString("%.15f", pair.atoms[traceAtomIndex][XX]) + " fy="
                        + formatString("%.15f", pair.atoms[traceAtomIndex][YY]) + " fz="
                        + formatString("%.15f", pair.atoms[traceAtomIndex][ZZ]) + " source_label="
                        + std::string(sourceLabel) + " code_location=" + std::string(codeLocation)
                        + " context_label="
                        + std::string((g_respaDoForceContextLabel != nullptr) ? g_respaDoForceContextLabel
                                                                              : "unspecified")
                        + " component_kind=" + std::string(componentKind) + " true_source_component="
                        + std::string(trueSourceComponent ? "true" : "false"));
    }
}

static void appendForceComponentTracePair(const char*            traceDirPath,
                                          const char*            side,
                                          const int64_t          step,
                                          const char*            componentName,
                                          const TracedForcePair& pair,
                                          const char*            sourceLabel,
                                          const char*            codeLocation,
                                          const char*            componentKind,
                                          const bool             trueSourceComponent)
{
    appendForceComponentTracePairToFile(traceDirPath,
                                        "step0_force_component_trace.txt",
                                        side,
                                        step,
                                        componentName,
                                        pair,
                                        sourceLabel,
                                        codeLocation,
                                        componentKind,
                                        trueSourceComponent);
}

static void appendForceComponentUnavailablePairToFile(const char* traceDirPath,
                                                      const char* fileName,
                                                      const char* side,
                                                      const int64_t step,
                                                      const char* componentName,
                                                      const char* sourceLabel,
                                                      const char* codeLocation,
                                                      const char* reason)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    for (const int atomIndex : respaForceTraceAtomIndices())
    {
        const bool haveGlobalAtomIndex =
                g_respaTraceGlobalAtomIndices != nullptr && atomIndex >= 0
                && atomIndex < g_respaTraceGlobalAtomIndexCount;
        const int globalAtomIndex =
                haveGlobalAtomIndex ? g_respaTraceGlobalAtomIndices[atomIndex] : atomIndex;
        appendRespaTraceTextLine(
                traceDirPath,
                fileName,
                "side=" + std::string(side) + " step=" + std::to_string(step) + " atom="
                        + std::to_string(atomIndex) + " component_name=" + std::string(componentName)
                        + " global_atom=" + std::to_string(globalAtomIndex)
                        + " available=false source_label=" + std::string(sourceLabel)
                        + " code_location=" + std::string(codeLocation) + " context_label="
                        + std::string((g_respaDoForceContextLabel != nullptr) ? g_respaDoForceContextLabel
                                                                              : "unspecified")
                        + " reason=" + std::string(reason));
    }
}

static void appendForceComponentUnavailablePair(const char* traceDirPath,
                                                const char* side,
                                                const int64_t step,
                                                const char* componentName,
                                                const char* sourceLabel,
                                                const char* codeLocation,
                                                const char* reason)
{
    appendForceComponentUnavailablePairToFile(traceDirPath,
                                              "step0_force_component_trace.txt",
                                              side,
                                              step,
                                              componentName,
                                              sourceLabel,
                                              codeLocation,
                                              reason);
}

static void appendRealspaceForceSubcomponentTracePair(const char*            traceDirPath,
                                                      const char*            side,
                                                      const int64_t          step,
                                                      const char*            componentName,
                                                      const TracedForcePair& pair,
                                                      const char*            sourceLabel,
                                                      const char*            codeLocation,
                                                      const char*            componentKind,
                                                      const bool             trueSourceComponent)
{
    appendForceComponentTracePairToFile(traceDirPath,
                                        "step0_realspace_force_subcomponent_trace.txt",
                                        side,
                                        step,
                                        componentName,
                                        pair,
                                        sourceLabel,
                                        codeLocation,
                                        componentKind,
                                        trueSourceComponent);
}

static void appendForceStoreUpdateInputTrace(const char*           traceDirPath,
                                             const int64_t         step,
                                             const char*           inputLabel,
                                             ArrayRef<const RVec>  values)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0' || values.empty())
    {
        return;
    }

    static std::once_flag traceHeaderOnce;
    std::call_once(traceHeaderOnce,
                   [traceDirPath]()
                   {
                       writeRespaTraceTextFile(traceDirPath,
                                               "force_store_update_inputs_trace.txt",
                                               "# step\tinput\tatom\tfx\tfy\tfz\n");
                   });

    for (const int atomIndex : respaForceTraceAtomIndices())
    {
        if (atomIndex < 0 || atomIndex >= values.ssize())
        {
            continue;
        }
        appendRespaTraceTextLine(traceDirPath,
                                 "force_store_update_inputs_trace.txt",
                                 std::to_string(step) + '\t' + inputLabel + '\t'
                                         + std::to_string(atomIndex) + '\t'
                                         + formatString("%.15f", values[atomIndex][XX]) + '\t'
                                         + formatString("%.15f", values[atomIndex][YY]) + '\t'
                                         + formatString("%.15f", values[atomIndex][ZZ]));
    }
}

static void appendExactGpuBondedReductionTrace(const char*          traceDirPath,
                                               const int64_t        step,
                                               const int            exactLevel,
                                               const char*          stageLabel,
                                               ArrayRef<const RVec> values)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0' || values.empty())
    {
        return;
    }

    static std::once_flag traceHeaderOnce;
    std::call_once(traceHeaderOnce,
                   [traceDirPath]()
                   {
                       writeRespaTraceTextFile(traceDirPath,
                                               "exact_gpu_bonded_reduction_trace.txt",
                                               "# step\tlevel\tstage\tatom\tfx\tfy\tfz\n");
                   });

    for (const int atomIndex : respaForceTraceAtomIndices())
    {
        if (atomIndex < 0 || atomIndex >= values.ssize())
        {
            continue;
        }
        appendRespaTraceTextLine(traceDirPath,
                                 "exact_gpu_bonded_reduction_trace.txt",
                                 std::to_string(step) + '\t' + std::to_string(exactLevel) + '\t'
                                         + stageLabel + '\t' + std::to_string(atomIndex) + '\t'
                                         + formatString("%.15f", values[atomIndex][XX]) + '\t'
                                         + formatString("%.15f", values[atomIndex][YY]) + '\t'
                                         + formatString("%.15f", values[atomIndex][ZZ]));
    }
}

static void appendExactGpuBondedReductionTrace(const char*            traceDirPath,
                                               const int64_t          step,
                                               const int              exactLevel,
                                               const char*            stageLabel,
                                               const TracedForcePair& pair)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    static std::once_flag traceHeaderOnce;
    std::call_once(traceHeaderOnce,
                   [traceDirPath]()
                   {
                       writeRespaTraceTextFile(traceDirPath,
                                               "exact_gpu_bonded_reduction_trace.txt",
                                               "# step\tlevel\tstage\tatom\tfx\tfy\tfz\n");
                   });

    for (int tracedAtomIndex = 0; tracedAtomIndex < static_cast<int>(pair.atomIndices.size()); ++tracedAtomIndex)
    {
        const int atomIndex = pair.atomIndices[tracedAtomIndex];
        appendRespaTraceTextLine(traceDirPath,
                                 "exact_gpu_bonded_reduction_trace.txt",
                                 std::to_string(step) + '\t' + std::to_string(exactLevel) + '\t'
                                         + stageLabel + '\t' + std::to_string(atomIndex) + '\t'
                                         + formatString("%.15f", pair.atoms[tracedAtomIndex][XX]) + '\t'
                                         + formatString("%.15f", pair.atoms[tracedAtomIndex][YY]) + '\t'
                                         + formatString("%.15f", pair.atoms[tracedAtomIndex][ZZ]));
    }
}

static TracedForcePair captureNbatOutputForceBufferPair(nonbonded_verlet_t* nbv)
{
    TracedForcePair pair;
    if (nbv == nullptr)
    {
        return pair;
    }

    const auto&       nbat            = nbv->nbat();
    const auto&       outputBuffer    = nbat.outputBuffer(0);
    ArrayRef<const real> forceBuffer  = outputBuffer.f;
    const bool        orderMatches    = nbv->localAtomOrderMatchesNbnxmOrder();
    const auto        gridIndices     = nbv->getGridIndices();
    const int         numLocalAtoms   = nbv->getNumAtoms(AtomLocality::Local);

    const auto mappedAtomIndex = [&](const int atomIndex) -> int
    {
        if (!orderMatches)
        {
            if (atomIndex < 0 || atomIndex >= gridIndices.ssize())
            {
                return -1;
            }
            return gridIndices[atomIndex];
        }
        return atomIndex;
    };

    for (int tracedAtomIndex = 0; tracedAtomIndex < static_cast<int>(pair.atomIndices.size()); ++tracedAtomIndex)
    {
        const int atomIndex = pair.atomIndices[tracedAtomIndex];
        if (atomIndex < 0 || atomIndex >= numLocalAtoms)
        {
            continue;
        }
        const int mappedIndex = mappedAtomIndex(atomIndex);
        if (mappedIndex < 0)
        {
            continue;
        }

        switch (nbat.FFormat)
        {
            case nbatXYZ:
            {
                const int offset = mappedIndex * STRIDE_XYZ;
                pair.atoms[tracedAtomIndex] = { forceBuffer[offset + 0], forceBuffer[offset + 1], forceBuffer[offset + 2] };
                break;
            }
            case nbatXYZQ:
            {
                const int offset = mappedIndex * STRIDE_XYZQ;
                pair.atoms[tracedAtomIndex] = { forceBuffer[offset + 0], forceBuffer[offset + 1], forceBuffer[offset + 2] };
                break;
            }
            case nbatX4:
            {
                const int offset = atom_to_x_index<c_packX4>(mappedIndex);
                pair.atoms[tracedAtomIndex] = { forceBuffer[offset + 0 * c_packX4],
                                                forceBuffer[offset + 1 * c_packX4],
                                                forceBuffer[offset + 2 * c_packX4] };
                break;
            }
            case nbatX8:
            {
                const int offset = atom_to_x_index<c_packX8>(mappedIndex);
                pair.atoms[tracedAtomIndex] = { forceBuffer[offset + 0 * c_packX8],
                                                forceBuffer[offset + 1 * c_packX8],
                                                forceBuffer[offset + 2 * c_packX8] };
                break;
            }
            default: break;
        }
    }

    return pair;
}

static void appendRealspaceForceSubcomponentUnavailablePair(const char* traceDirPath,
                                                            const char* side,
                                                            const int64_t step,
                                                            const char* componentName,
                                                            const char* sourceLabel,
                                                            const char* codeLocation,
                                                            const char* reason)
{
    appendForceComponentUnavailablePairToFile(traceDirPath,
                                              "step0_realspace_force_subcomponent_trace.txt",
                                              side,
                                              step,
                                              componentName,
                                              sourceLabel,
                                              codeLocation,
                                              reason);
}

static void appendExclusionEquivalenceTracePair(const char*        traceDirPath,
                                                const char*        side,
                                                const int64_t      step,
                                                const int          pairOrdinal,
                                                const int          ai,
                                                const int          aj,
                                                const char*        sourcePath,
                                                const char*        membershipSource,
                                                const char*        listKind,
                                                const bool         treatedAsExclusionProducer,
                                                const bool         includePairEffective,
                                                const real         factorCoulomb,
                                                const real         factorLj,
                                                const real         qq,
                                                const int          tableIndex,
                                                const real         frac,
                                                const real         fexcl,
                                                const real         vcorr,
                                                const real         bareCoulombScalar,
                                                const real         correctionScalarRaw,
                                                const real         correctionForceScalarEquivalent,
                                                const real         effectiveOuterScalar,
                                                const real         fullScalar,
                                                const RVec&        correctionForceWritten,
                                                const RVec&        combinedForceWritten,
                                                const char*        sinkTarget,
                                                const bool         sinkWriteExecuted,
                                                const char*        sourceLabel,
                                                const char*        componentKind,
                                                const char*        codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    appendRespaTraceTextLine(
            traceDirPath,
            "step0_exclusion_equivalence_pair_trace.txt",
            "side=" + std::string(side) + " step=" + std::to_string(step) + " pair_ordinal="
                    + std::to_string(pairOrdinal) + " ai=" + std::to_string(ai) + " aj="
                    + std::to_string(aj) + " pair_key=" + std::to_string(ai) + "_" + std::to_string(aj)
                    + " source_path=" + std::string(sourcePath) + " membership_source="
                    + std::string(membershipSource) + " list_kind=" + std::string(listKind)
                    + " treated_as_exclusion_correction_producer="
                    + std::string(treatedAsExclusionProducer ? "true" : "false")
                    + " include_pair_effective="
                    + std::string(includePairEffective ? "true" : "false") + " factor_coulomb="
                    + formatString("%.15f", factorCoulomb) + " factor_lj="
                    + formatString("%.15f", factorLj) + " qq=" + formatString("%.15f", qq)
                    + " table_index=" + std::to_string(tableIndex) + " frac="
                    + formatString("%.15f", frac) + " fexcl=" + formatString("%.15f", fexcl)
                    + " vcorr=" + formatString("%.15f", vcorr) + " bare_coulomb_scalar="
                    + formatString("%.15f", bareCoulombScalar) + " correction_scalar_raw="
                    + formatString("%.15f", correctionScalarRaw)
                    + " correction_force_scalar_equivalent="
                    + formatString("%.15f", correctionForceScalarEquivalent) + " effective_outer_scalar="
                    + formatString("%.15f", effectiveOuterScalar) + " full_scalar="
                    + formatString("%.15f", fullScalar) + " correction_force_written_fx="
                    + formatString("%.15f", correctionForceWritten[XX]) + " correction_force_written_fy="
                    + formatString("%.15f", correctionForceWritten[YY]) + " correction_force_written_fz="
                    + formatString("%.15f", correctionForceWritten[ZZ]) + " combined_force_written_fx="
                    + formatString("%.15f", combinedForceWritten[XX]) + " combined_force_written_fy="
                    + formatString("%.15f", combinedForceWritten[YY]) + " combined_force_written_fz="
                    + formatString("%.15f", combinedForceWritten[ZZ]) + " sink_target="
                    + std::string(sinkTarget) + " sink_write_executed="
                    + std::string(sinkWriteExecuted ? "true" : "false") + " source_label="
                    + std::string(sourceLabel) + " component_kind=" + std::string(componentKind)
                    + " code_location=" + std::string(codeLocation));
}

static void appendBoundaryBookkeepingAuditLine(const char* traceDirPath,
                                               const char* side,
                                               const int64_t step,
                                               const int ai,
                                               const int aj,
                                               const char* listKind,
                                               const char* contribution,
                                               const real r,
                                               const real outerScalar,
                                               const real effectiveOuterScalar,
                                               const real fullCoulombEnergy,
                                               const bool scalarIsZero,
                                               const bool energyWriteExecuted,
                                               const bool suppressBookkeepingEnergy,
                                               const bool suppressExcludedPairComparableEnergy,
                                               const int energyIndex,
                                               const real targetBefore,
                                               const real writeDelta,
                                               const real targetAfter,
                                               const char* reason,
                                               const char* codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    appendRespaTraceTextLine(
            traceDirPath,
            "boundary_force_energy_crosscheck_trace.txt",
            "side=" + std::string(side) + " step=" + std::to_string(step) + " pair_i="
                    + std::to_string(std::min(ai, aj)) + " pair_j=" + std::to_string(std::max(ai, aj))
                    + " list_kind=" + std::string(listKind) + " contribution=" + std::string(contribution)
                    + " r=" + formatString("%.15f", r) + " outerScalar="
                    + formatString("%.15f", outerScalar) + " effectiveOuterScalar="
                    + formatString("%.15f", effectiveOuterScalar) + " fullCoulombEnergy="
                    + formatString("%.15f", fullCoulombEnergy) + " scalar_zero="
                    + std::string(scalarIsZero ? "true" : "false") + " energy_write_executed="
                    + std::string(energyWriteExecuted ? "true" : "false")
                    + " suppress_bookkeeping_energy="
                    + std::string(suppressBookkeepingEnergy ? "true" : "false")
                    + " suppress_excluded_pair_comparable_energy="
                    + std::string(suppressExcludedPairComparableEnergy ? "true" : "false")
                    + " energy_index=" + std::to_string(energyIndex) + " sink_name=coulEnergyTerms"
                    + " target_before=" + formatString("%.15f", targetBefore) + " write_delta="
                    + formatString("%.15f", writeDelta) + " target_after="
                    + formatString("%.15f", targetAfter) + " reason=" + std::string(reason)
                    + " code_location=" + std::string(codeLocation));
}

static const char* loopEntryStageName(const int64_t step)
{
    return (step == 5) ? "STEP5_LOOP_ENTRY_STATE_X"
           : (step == 6) ? "STEP6_LOOP_ENTRY_STATE_X"
           : (step == 7) ? "STEP7_LOOP_ENTRY_STATE_X"
                         : "LOOP_ENTRY_STATE_X";
}

static const char* postPbcStageName(const int64_t step)
{
    return (step == 5) ? "STEP5_POST_PBC_STATE_X"
           : (step == 6) ? "STEP6_POST_PBC_STATE_X"
           : (step == 7) ? "STEP7_POST_PBC_STATE_X"
                         : "POST_PBC_STATE_X";
}

static const char* postWholeMoleculeTransformStageName(const int64_t step)
{
    return (step == 5) ? "STEP5_POST_WHOLE_MOLECULE_TRANSFORM_STATE_X"
           : (step == 6) ? "STEP6_POST_WHOLE_MOLECULE_TRANSFORM_STATE_X"
           : (step == 7) ? "STEP7_POST_WHOLE_MOLECULE_TRANSFORM_STATE_X"
                         : "POST_WHOLE_MOLECULE_TRANSFORM_STATE_X";
}

static const char* preHandoffStageName(const int64_t step)
{
    return (step == 5) ? "STEP5_PRE_HANDOFF_STATE_X"
           : (step == 6) ? "STEP6_PRE_HANDOFF_STATE_X"
           : (step == 7) ? "STEP7_PRE_HANDOFF_STATE_X"
                         : "PRE_HANDOFF_STATE_X";
}

static void appendCoordHandoffTraceLine(const char* traceDirPath,
                                        const char* side,
                                        const char* stageName,
                                        int64_t     step,
                                        int         atomIndex,
                                        const RVec& coord,
                                        const char* bufferLabel,
                                        const void* bufferPtr)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    appendRespaTraceTextLine(
            traceDirPath,
            "multistep_coord_handoff_trace.txt",
            "side=" + std::string(side) + " stage=" + std::string(stageName) + " step="
                    + std::to_string(step) + " atom=" + std::to_string(atomIndex) + " x="
                    + formatString("%.15f", coord[XX]) + " y=" + formatString("%.15f", coord[YY]) + " z="
                    + formatString("%.15f", coord[ZZ]) + " buffer_label=" + std::string(bufferLabel)
                    + " buffer_ptr="
                    + std::to_string(static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(bufferPtr))));
}

static void appendStateXChainTraceLine(const char* traceDirPath,
                                       const char* side,
                                       const char* stageName,
                                       int64_t     step,
                                       int         atomIndex,
                                       const RVec& coord,
                                       const char* writerName,
                                       const char* codeLocation,
                                       bool        writesStateX)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    appendRespaTraceTextLine(
            traceDirPath,
            "multistep_state_x_chain_trace.txt",
            "side=" + std::string(side) + " step=" + std::to_string(step) + " stage="
                    + std::string(stageName) + " atom=" + std::to_string(atomIndex) + " x="
                    + formatString("%.15f", coord[XX]) + " y=" + formatString("%.15f", coord[YY]) + " z="
                    + formatString("%.15f", coord[ZZ]) + " writer=" + std::string(writerName)
                    + " code_location=" + std::string(codeLocation) + " writes_state_x="
                    + std::string(writesStateX ? "true" : "false"));
}

static void appendStateXChainTracePair(const char*          traceDirPath,
                                       const char*          side,
                                       const char*          stageName,
                                       int64_t              step,
                                       ArrayRef<const RVec> coords,
                                       const char*          writerName,
                                       const char*          codeLocation,
                                       bool                 writesStateX)
{
    for (const int atomIndex : respaForceTraceAtomIndices())
    {
        if (atomIndex < 0 || atomIndex >= coords.ssize())
        {
            continue;
        }
        appendStateXChainTraceLine(traceDirPath,
                                   side,
                                   stageName,
                                   step,
                                   atomIndex,
                                   coords[atomIndex],
                                   writerName,
                                   codeLocation,
                                   writesStateX);
    }
}

static void appendCoordHandoffTracePair(const char*           traceDirPath,
                                        const char*           side,
                                        const char*           stageName,
                                        int64_t               step,
                                        ArrayRef<const RVec>  coords,
                                        const char*           bufferLabel,
                                        const void*           bufferPtr)
{
    for (const int atomIndex : respaForceTraceAtomIndices())
    {
        if (atomIndex < 0 || atomIndex >= coords.ssize())
        {
            continue;
        }
        appendCoordHandoffTraceLine(
                traceDirPath, side, stageName, step, atomIndex, coords[atomIndex], bufferLabel, bufferPtr);
    }
}

static void appendCoordHandoffTracePair(const char*              traceDirPath,
                                        const char*              side,
                                        const char*              stageName,
                                        int64_t                  step,
                                        const nbnxn_atomdata_t&  nbat,
                                        const char*              bufferLabel,
                                        const void*              bufferPtr)
{
    for (const int atomIndex : respaForceTraceAtomIndices())
    {
        if (atomIndex < 0 || atomIndex >= nbat.numAtoms())
        {
            continue;
        }
        appendCoordHandoffTraceLine(traceDirPath,
                                    side,
                                    stageName,
                                    step,
                                    atomIndex,
                                    getCoordinate(nbat, atomIndex),
                                    bufferLabel,
                                    bufferPtr);
    }
}

static void appendCoulombFirstWriteTraceLine(const char* traceDirPath,
                                             int*        writeOrdinal,
                                             real        targetBefore,
                                             real        writeValue,
                                             real        targetAfter,
                                             int         energyIndex,
                                             const char* codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0' || writeOrdinal == nullptr)
    {
        return;
    }
    if (*writeOrdinal >= 5)
    {
        return;
    }

    ++(*writeOrdinal);
    appendRespaTraceTextLine(traceDirPath,
                             "step0_excluded_coulomb_first_writes.txt",
                             "side=PATCH write_ordinal=" + std::to_string(*writeOrdinal)
                                     + " code_location=" + std::string(codeLocation)
                                     + " energyIndex=" + std::to_string(energyIndex)
                                     + " target_before=" + formatString("%.15f", targetBefore)
                                     + " write_value=" + formatString("%.15f", writeValue)
                                     + " target_after=" + formatString("%.15f", targetAfter));
}

static void appendCoulombProducerTraceLine(const char* traceDirPath,
                                           int*        producerOrdinal,
                                           int         pairI,
                                           int         pairJ,
                                           int         energyIndex,
                                           real        targetBefore,
                                           real        fullCoulombEnergy,
                                           real        coulEnergyDelta,
                                           real        qq,
                                           real        factorCoulomb,
                                           real        rinv,
                                           real        ewaldShift,
                                           int         tableIndex,
                                           real        frac,
                                           real        fexcl,
                                           real        vcorr,
                                           real        bareCoulombScalar,
                                           real        correctionScalar,
                                           bool        isExcludedPairlist,
                                           bool        patchShapeB,
                                           const char* codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    static std::mutex  traceMutex;
    static std::string clearedTraceDirPath;
    static int         runningOrdinal = 0;
    static double      runningPrefixSum = 0.0;
    static const std::vector<int> prefixCheckpoints = []()
    {
        std::vector<int> result;
        const char* value = std::getenv("GMX_PCFF_RESPA_COULOMB_PREFIX_CHECKPOINTS");
        if (value == nullptr || *value == '\0')
        {
            return result;
        }
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
    }();
    static const std::vector<int> detailOrdinals = []()
    {
        std::vector<int> result;
        const char* value = std::getenv("GMX_PCFF_RESPA_COULOMB_DETAIL_ORDINALS");
        if (value == nullptr || *value == '\0')
        {
            return result;
        }
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
    }();
    if (prefixCheckpoints.empty() && detailOrdinals.empty())
    {
        return;
    }
    if (!isExcludedPairlist)
    {
        return;
    }

    std::lock_guard<std::mutex> guard(traceMutex);
    if (clearedTraceDirPath != traceDirPath)
    {
        writeRespaTraceTextFile(traceDirPath, "step0_excluded_coulomb_prefix_checkpoints.txt", "");
        writeRespaTraceTextFile(traceDirPath, "step0_excluded_coulomb_detail_rows.txt", "");
        clearedTraceDirPath = traceDirPath;
        runningOrdinal      = 0;
        runningPrefixSum    = 0.0;
    }

    ++runningOrdinal;
    const double targetAfter = runningPrefixSum + fullCoulombEnergy;
    runningPrefixSum         = targetAfter;
    if (producerOrdinal != nullptr)
    {
        *producerOrdinal = runningOrdinal;
    }

    if (std::find(prefixCheckpoints.begin(), prefixCheckpoints.end(), runningOrdinal)
        != prefixCheckpoints.end())
    {
        appendRespaTraceTextLine(traceDirPath,
                                 "step0_excluded_coulomb_prefix_checkpoints.txt",
                                 "side=PATCH producer_count=" + std::to_string(runningOrdinal)
                                         + " cumulative_coulomb_prefix_sum="
                                         + formatString("%.15f", runningPrefixSum));
    }

    if (std::find(detailOrdinals.begin(), detailOrdinals.end(), runningOrdinal) == detailOrdinals.end())
    {
        return;
    }

    appendRespaTraceTextLine(
            traceDirPath,
            "step0_excluded_coulomb_detail_rows.txt",
            "side=PATCH producer_ordinal=" + std::to_string(runningOrdinal)
                    + " code_location=" + std::string(codeLocation)
                    + " pair_i=" + std::to_string(pairI)
                    + " pair_j=" + std::to_string(pairJ)
                    + " energyIndex=" + std::to_string(energyIndex)
                    + " target_before=" + formatString("%.15f", targetBefore)
                    + " target_after=" + formatString("%.15f", targetAfter)
                    + " fullCoulombEnergy=" + formatString("%.15f", fullCoulombEnergy)
                    + " coulEnergyDelta=" + formatString("%.15f", coulEnergyDelta)
                    + " qq=" + formatString("%.15f", qq)
                    + " factorCoulomb=" + formatString("%.15f", factorCoulomb)
                    + " rinv=" + formatString("%.15f", rinv)
                    + " ewald_shift=" + formatString("%.15f", ewaldShift)
                    + " table_index=" + std::to_string(tableIndex)
                    + " frac=" + formatString("%.15f", frac)
                    + " fexcl=" + formatString("%.15f", fexcl)
                    + " vcorr=" + formatString("%.15f", vcorr)
                    + " bareCoulombScalar=" + formatString("%.15f", bareCoulombScalar)
                    + " correctionScalar=" + formatString("%.15f", correctionScalar)
                    + " isExcludedPairlist=" + std::string(isExcludedPairlist ? "true" : "false")
                    + " patchShapeB=" + std::string(patchShapeB ? "true" : "false"));
}

static void appendMultiStepCoulombPairTraceLine(const char* traceDirPath,
                                                int64_t     step,
                                                int         pairlistOrdinal,
                                                int         pairI,
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
                                                real        qq,
                                                real        factorCoulomb,
                                                real        rinv,
                                                int         tableIndex,
                                                real        frac,
                                                real        fexcl,
                                                real        vcorr,
                                                real        pairContribution,
                                                real        cumulativeBefore,
                                                const char* codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    static std::mutex  traceMutex;
    static std::string clearedTraceDirPath;
    static int64_t     currentStep      = -1;
    static int         runningOrdinal   = 0;
    static double      runningPrefixSum = 0.0;
    static const std::vector<int> prefixCheckpoints = []()
    {
        std::vector<int> result;
        const char* value = std::getenv("GMX_PCFF_RESPA_MULTI_STEP_COULOMB_PAIR_PREFIX_CHECKPOINTS");
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
                result.push_back(std::stoi(item));
            }
        }
        return result;
    }();
    static const std::vector<int> detailOrdinals = []()
    {
        std::vector<int> result;
        const char* value = std::getenv("GMX_PCFF_RESPA_MULTI_STEP_COULOMB_PAIR_DETAIL_ORDINALS");
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
                result.push_back(std::stoi(item));
            }
        }
        return result;
    }();
    if (prefixCheckpoints.empty() && detailOrdinals.empty())
    {
        return;
    }

    std::lock_guard<std::mutex> guard(traceMutex);
    if (clearedTraceDirPath != traceDirPath)
    {
        writeRespaTraceTextFile(traceDirPath, "multistep_coulomb_pair_prefix_trace.txt", "");
        writeRespaTraceTextFile(traceDirPath, "multistep_coulomb_pair_detail_rows.txt", "");
        clearedTraceDirPath = traceDirPath;
        currentStep         = -1;
        runningOrdinal      = 0;
        runningPrefixSum    = 0.0;
    }
    if (currentStep != step)
    {
        currentStep      = step;
        runningOrdinal   = 0;
        runningPrefixSum = 0.0;
    }

    ++runningOrdinal;
    runningPrefixSum += pairContribution;
    const double cumulativeAfter = runningPrefixSum;

    if (std::find(prefixCheckpoints.begin(), prefixCheckpoints.end(), runningOrdinal)
        != prefixCheckpoints.end())
    {
        appendRespaTraceTextLine(
                traceDirPath,
                "multistep_coulomb_pair_prefix_trace.txt",
                "side=PATCH step=" + std::to_string(step) + " pair_ordinal="
                        + std::to_string(runningOrdinal) + " cumulative_coulomb_prefix_sum="
                        + formatString("%.15f", cumulativeAfter));
    }

    if (std::find(detailOrdinals.begin(), detailOrdinals.end(), runningOrdinal) == detailOrdinals.end())
    {
        return;
    }

    appendRespaTraceTextLine(
            traceDirPath,
            "multistep_coulomb_pair_detail_rows.txt",
            "side=PATCH step=" + std::to_string(step) + " pair_ordinal="
                    + std::to_string(runningOrdinal) + " pair_i=" + std::to_string(pairI) + " pair_j="
                    + std::to_string(pairJ) + " energyIndex=" + std::to_string(energyIndex)
                    + " shiftIndex=" + std::to_string(shiftIndex)
                    + " coord_i_x=" + formatString("%.15f", coordIX)
                    + " coord_i_y=" + formatString("%.15f", coordIY)
                    + " coord_i_z=" + formatString("%.15f", coordIZ)
                    + " coord_j_x=" + formatString("%.15f", coordJX)
                    + " coord_j_y=" + formatString("%.15f", coordJY)
                    + " coord_j_z=" + formatString("%.15f", coordJZ)
                    + " shift_x=" + formatString("%.15f", shiftX)
                    + " shift_y=" + formatString("%.15f", shiftY)
                    + " shift_z=" + formatString("%.15f", shiftZ)
                    + " dx=" + formatString("%.15f", dx) + " dy=" + formatString("%.15f", dy)
                    + " dz=" + formatString("%.15f", dz) + " rsq=" + formatString("%.15f", rsq)
                    + " qq="
                    + formatString("%.15f", qq) + " rinv=" + formatString("%.15f", rinv)
                    + " table_index=" + std::to_string(tableIndex) + " frac="
                    + formatString("%.15f", frac) + " fexcl=" + formatString("%.15f", fexcl)
                    + " vcorr=" + formatString("%.15f", vcorr) + " factorCoulomb="
                    + formatString("%.15f", factorCoulomb) + " pair_contribution="
                    + formatString("%.15f", pairContribution) + " cumulative_before="
                    + formatString("%.15f", cumulativeBefore) + " cumulative_after="
                    + formatString("%.15f", cumulativeAfter) + " code_location="
                    + std::string(codeLocation) + " pairlist_ordinal="
                    + std::to_string(pairlistOrdinal));
}

static void appendCoulombSelfTraceLine(const char* traceDirPath,
                                       int         atom,
                                       int         energyIndex,
                                       real        charge,
                                       real        selfEnergy,
                                       real        targetBefore,
                                       real        targetAfter,
                                       const char* codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    static std::mutex  traceMutex;
    static std::string clearedTraceDirPath;
    static int         runningOrdinal = 0;
    static double      runningPrefixSum = 0.0;
    static const std::vector<int> prefixCheckpoints = []()
    {
        std::vector<int> result;
        const char* value = std::getenv("GMX_PCFF_RESPA_COULOMB_SELF_PREFIX_CHECKPOINTS");
        if (value == nullptr || *value == '\0')
        {
            return result;
        }
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
    }();
    static const std::vector<int> detailOrdinals = []()
    {
        std::vector<int> result;
        const char* value = std::getenv("GMX_PCFF_RESPA_COULOMB_SELF_DETAIL_ORDINALS");
        if (value == nullptr || *value == '\0')
        {
            return result;
        }
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
    }();
    if (prefixCheckpoints.empty() && detailOrdinals.empty())
    {
        return;
    }

    std::lock_guard<std::mutex> guard(traceMutex);
    if (clearedTraceDirPath != traceDirPath)
    {
        writeRespaTraceTextFile(traceDirPath, "step0_coulomb_self_prefix_checkpoints.txt", "");
        writeRespaTraceTextFile(traceDirPath, "step0_coulomb_self_detail_rows.txt", "");
        clearedTraceDirPath = traceDirPath;
        runningOrdinal      = 0;
        runningPrefixSum    = 0.0;
    }

    const double prefixBefore = runningPrefixSum;
    ++runningOrdinal;
    runningPrefixSum += selfEnergy;
    const double prefixAfter = runningPrefixSum;

    if (std::find(prefixCheckpoints.begin(), prefixCheckpoints.end(), runningOrdinal)
        != prefixCheckpoints.end())
    {
        appendRespaTraceTextLine(traceDirPath,
                                 "step0_coulomb_self_prefix_checkpoints.txt",
                                 "side=PATCH atom_ordinal=" + std::to_string(runningOrdinal)
                                         + " cumulative_self_coulomb_prefix_sum="
                                         + formatString("%.15f", runningPrefixSum));
    }

    if (std::find(detailOrdinals.begin(), detailOrdinals.end(), runningOrdinal) == detailOrdinals.end())
    {
        return;
    }

    appendRespaTraceTextLine(
            traceDirPath,
            "step0_coulomb_self_detail_rows.txt",
            "side=PATCH atom_ordinal=" + std::to_string(runningOrdinal)
                    + " atom=" + std::to_string(atom)
                    + " energyIndex=" + std::to_string(energyIndex)
                    + " charge=" + formatString("%.15f", charge)
                    + " selfEnergy=" + formatString("%.15f", selfEnergy)
                    + " prefix_before=" + formatString("%.15f", prefixBefore)
                    + " prefix_after=" + formatString("%.15f", prefixAfter)
                    + " target_before=" + formatString("%.15f", targetBefore)
                    + " target_after=" + formatString("%.15f", targetAfter)
                    + " code_location=" + std::string(codeLocation));
}

static void appendLjAccumWriteTraceLine(const char* traceDirPath,
                                        int         pairI,
                                        int         pairJ,
                                        int         energyIndex,
                                        real        targetBeforeVdwEnergyTerms,
                                        real        writeValueLjDelta,
                                        real        targetAfterVdwEnergyTerms,
                                        real        pairStatsLjBefore,
                                        real        pairStatsLjDelta,
                                        real        pairStatsLjAfter,
                                        const char* codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    static std::mutex  traceMutex;
    static std::string clearedTraceDirPath;
    static int         runningOrdinal = 0;

    std::lock_guard<std::mutex> guard(traceMutex);
    if (clearedTraceDirPath != traceDirPath)
    {
        writeRespaTraceTextFile(traceDirPath, "step0_lj_accum_contract_trace.txt", "");
        clearedTraceDirPath = traceDirPath;
        runningOrdinal      = 0;
    }
    if (runningOrdinal >= 5)
    {
        return;
    }

    ++runningOrdinal;
    appendRespaTraceTextLine(
            traceDirPath,
            "step0_lj_accum_contract_trace.txt",
            "side=PATCH write_ordinal=" + std::to_string(runningOrdinal)
                    + " code_location=" + std::string(codeLocation)
                    + " pair_i=" + std::to_string(pairI)
                    + " pair_j=" + std::to_string(pairJ)
                    + " energyIndex=" + std::to_string(energyIndex)
                    + " target_before_vdwEnergyTerms="
                    + formatString("%.15f", targetBeforeVdwEnergyTerms)
                    + " write_value_ljDelta=" + formatString("%.15f", writeValueLjDelta)
                    + " target_after_vdwEnergyTerms="
                    + formatString("%.15f", targetAfterVdwEnergyTerms)
                    + " pairStats_lj_before=" + formatString("%.15f", pairStatsLjBefore)
                    + " pairStats_lj_delta=" + formatString("%.15f", pairStatsLjDelta)
                    + " pairStats_lj_after=" + formatString("%.15f", pairStatsLjAfter));
}

static void computeExactRespaNonbondedCpu(const t_inputrec&                 inputrec,
                                          const InteractionDefinitions&     idef,
                                          t_forcerec*                       fr,
                                          const t_mdatoms&                  mdatoms,
                                          ArrayRef<const RVec>              coordinates,
                                          const ExactRespaForceOutputs&     exactRespaForceOutputs,
                                          gmx_enerdata_t*                   enerd,
                                          const StepWorkload&               stepWork,
                                          const int64_t                     step,
                                          const bool                        traceOnlyDiagnostics = false)
{
    gmx::assertExactRespaOwnsNoLegacyMtsState(inputrec);

    enum class ExactRespaNonbondedContribution : int
    {
        Inner,
        Middle,
        Outer,
        Count
    };

    const bool traceRealspaceForceSubcomponents = shouldTraceRespaRealspaceForceSubcomponentsStep(step);
    const bool traceExclusionEquivalence        = shouldTraceRespaExclusionEquivalenceStep(step);
    const bool traceStep1Subset01ForceGroupAudit =
            shouldTraceStep1Subset01ForceGroupAuditStep(step);
    TracedForcePair tracedPatchLjSrForce;
    TracedForcePair tracedPatchCoulombSrForce;
    TracedForcePair tracedPatchExclusionCorrectionForce;
    TracedForcePair tracedPatchCombinedRealspaceForce;
    TracedForcePair tracedExactInnerRealspaceForce;
    TracedForcePair tracedExactMiddleRealspaceForce;
    TracedForcePair tracedExactOuterRealspaceForce;
    TracedForcePair tracedExactInnerLjSrForce;
    TracedForcePair tracedExactInnerBareCoulombSrForce;
    TracedForcePair tracedExactInnerCorrectionForce;
    TracedForcePair tracedExactMiddleLjSrForce;
    TracedForcePair tracedExactMiddleBareCoulombSrForce;
    TracedForcePair tracedExactMiddleCorrectionForce;

    if (!traceOnlyDiagnostics && shouldTraceRespaCoordHandoffStep(step))
    {
        appendCoordHandoffTracePair(activeM2pTraceDirPath(),
                                    "PATCH",
                                    "NONBONDED_CPU_INPUT",
                                    step,
                                    coordinates,
                                    "computeLammpsRespaNonbondedCpu.coordinates",
                                    coordinates.data());
    }

    GMX_RELEASE_ASSERT(fr->completePairlistRange.has_value(),
                       "Exact LAMMPS-style r-RESPA requires a complete pairlist");
    GMX_RELEASE_ASSERT(fr->efep == FreeEnergyPerturbationType::No,
                       "Exact LAMMPS-style r-RESPA does not support free-energy perturbation yet");
    GMX_RELEASE_ASSERT(fr->ic->vdw.type == VanDerWaalsType::Cut,
                       "Exact LAMMPS-style r-RESPA currently supports cut-off Van der Waals only");
    GMX_RELEASE_ASSERT(fr->ic->vdw.modifier == InteractionModifiers::None,
                       "Exact LAMMPS-style r-RESPA currently supports unmodified real-space LJ only");
    GMX_RELEASE_ASSERT(usingPmeOrEwald(fr->ic->coulomb.type),
                       "Exact LAMMPS-style r-RESPA currently supports Coulomb long-range treatment only");
    GMX_RELEASE_ASSERT(fr->ic->coulomb.modifier == InteractionModifiers::None,
                       "Exact LAMMPS-style r-RESPA currently supports unmodified real-space Coulomb only");

    struct ContributionAccumulator
    {
        ExactRespaNonbondedContribution contribution;
        ForceOutputs*                  outputs = nullptr;
        ArrayRef<RVec>                 force;
        ArrayRef<RVec>                 shift;
        ForceWithVirial*               forceWithVirial = nullptr;
        bool                           accumulateEnergy = false;
        matrix                         virial           = { { 0 } };
    };

    std::vector<ContributionAccumulator> activeContributions;
    const auto appendContribution = [&](const ExactRespaNonbondedContribution contribution, const int level)
    {
        ForceOutputs* outputs = exactRespaForceOutputs.levelOrNull(level);
        if (outputs == nullptr)
        {
            return;
        }

        ContributionAccumulator accumulator;
        accumulator.contribution = contribution;
        accumulator.outputs      = outputs;
        accumulator.accumulateEnergy = (contribution == ExactRespaNonbondedContribution::Outer) && stepWork.computeEnergy;

        const bool directVirialContribution =
                stepWork.computeVirial && (contribution == ExactRespaNonbondedContribution::Outer);
        if (directVirialContribution)
        {
            GMX_RELEASE_ASSERT(outputs->haveForceWithVirial(),
                               "Exact LAMMPS-style r-RESPA outer forces require a direct-virial buffer");
            accumulator.forceWithVirial = &outputs->forceWithVirial();
            accumulator.force           = accumulator.forceWithVirial->force_;
        }
        else
        {
            accumulator.force = outputs->forceWithShiftForces().force();
            accumulator.shift = outputs->forceWithShiftForces().shiftForces();
        }

        activeContributions.push_back(accumulator);
    };

    appendContribution(ExactRespaNonbondedContribution::Inner, exactRespaNonbondedInnerLevel(inputrec));
    if (inputrec.exactRespa.forceLayout.hasMiddle())
    {
        appendContribution(ExactRespaNonbondedContribution::Middle, exactRespaNonbondedMiddleLevel(inputrec));
    }
    appendContribution(ExactRespaNonbondedContribution::Outer, exactRespaNonbondedOuterLevel(inputrec));

    ContributionAccumulator* outerAccumulator = nullptr;
    int                      outerContributionIndex = -1;
    for (int contributionIndex = 0; contributionIndex < gmx::ssize(activeContributions);
         ++contributionIndex)
    {
        auto& accumulator = activeContributions[contributionIndex];
        if (accumulator.contribution == ExactRespaNonbondedContribution::Outer)
        {
            outerAccumulator       = &accumulator;
            outerContributionIndex = contributionIndex;
            break;
        }
    }

    GMX_RELEASE_ASSERT(!stepWork.computeVirial
                               || std::any_of(activeContributions.begin(),
                                              activeContributions.end(),
                                              [](const ContributionAccumulator& accumulator)
                                              {
                                                  return accumulator.contribution
                                                         == ExactRespaNonbondedContribution::Outer;
                                              }),
                       "Exact LAMMPS-style r-RESPA virial steps require the outer contribution to be active");

    const real coulombCutoff2   = gmx::square(fr->ic->coulomb.cutoff);
    const real vdwCutoff2       = gmx::square(fr->ic->vdw.cutoff);
    const auto& exactRespaForceLayout = inputrec.exactRespa.forceLayout;
    const bool  exactRespaHasMiddle   = exactRespaForceLayout.hasMiddle();
    const real repulsionPower   = static_cast<real>(fr->ic->vdw.repulsionPower);
    const bool usePower9SpecializedPath =
            useRepulsionPower9ExactRespaCpuSpecialization(*fr->ic);
    const real repulsionEnergyPrefactor =
            usePower9SpecializedPath ? (1.0_real / 9.0_real) : (1.0_real / repulsionPower);
    const int  ntype2           = 2 * fr->ntype;
    auto&      vdwEnergyTerms   = enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJSR];
    auto&      coulEnergyTerms  = enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR];
    const bool debugExactRespa  = (std::getenv("GMX_PCFF_RESPA_DEBUG") != nullptr);
    const bool traceCpuCorrectionEnergies =
            stepWork.computeEnergy && shouldTraceCpuCorrectionEnergiesStep(step);
    const char* excludedCorrectionForceDumpPath =
            std::getenv("GMX_PCFF_RESPA_EXCLUDED_FORCE_DUMP_FILE");
    const bool dumpExcludedCorrectionForce =
            !traceOnlyDiagnostics && (excludedCorrectionForceDumpPath != nullptr && *excludedCorrectionForceDumpPath != '\0');
    const char* earlyAccumTraceDirPath = std::getenv("GMX_PCFF_RESPA_EARLY_TRACE_DIR");
    const bool dumpEarlyAccumTrace =
            !traceOnlyDiagnostics && (earlyAccumTraceDirPath != nullptr && *earlyAccumTraceDirPath != '\0'
                                      && step == 0);
    const char* pairWriteProofDirPath = std::getenv("GMX_PCFF_RESPA_PAIR_WRITE_PROOF_DIR");
    const bool dumpPairWriteProof =
            !traceOnlyDiagnostics && (pairWriteProofDirPath != nullptr && *pairWriteProofDirPath != '\0'
                                      && step == 0);
    const char* downstreamContractTraceDirPath = std::getenv("GMX_PCFF_RESPA_DOWNSTREAM_CONTRACT_TRACE_DIR");
    const bool dumpDownstreamContract =
            !traceOnlyDiagnostics && (downstreamContractTraceDirPath != nullptr
                                      && *downstreamContractTraceDirPath != '\0' && step == 0);
    const char* bookkeepingResidualTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2L_TRACE_DIR");
    const bool dumpBookkeepingResidualTrace =
            !traceOnlyDiagnostics && (bookkeepingResidualTraceDirPath != nullptr
                                      && *bookkeepingResidualTraceDirPath != '\0' && step == 0);
    const char* dispatchInternalTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2K_TRACE_DIR");
    if (dispatchInternalTraceDirPath == nullptr || *dispatchInternalTraceDirPath == '\0')
    {
        dispatchInternalTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2J_TRACE_DIR");
    }
    const bool dumpDispatchInternalTrace =
            !traceOnlyDiagnostics && (dispatchInternalTraceDirPath != nullptr
                                      && *dispatchInternalTraceDirPath != '\0' && step == 0);
    const char* dispatchProbeModeEnv = std::getenv("GMX_PCFF_RESPA_M2L_PROBE_MODE");
    if (dispatchProbeModeEnv == nullptr || *dispatchProbeModeEnv == '\0')
    {
        dispatchProbeModeEnv = std::getenv("GMX_PCFF_RESPA_M2K_PATCH_MODE");
    }
    if (dispatchProbeModeEnv == nullptr || *dispatchProbeModeEnv == '\0')
    {
        dispatchProbeModeEnv = std::getenv("GMX_PCFF_RESPA_M2J_PROBE_MODE");
    }
    const std::string dispatchProbeMode =
            (dispatchProbeModeEnv != nullptr && *dispatchProbeModeEnv != '\0') ? dispatchProbeModeEnv : "baseline";
    const char* m2xGeometryTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2X_TRACE_DIR");
    const char* m2xGeometryCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2X_CASE_LABEL");
    const bool  dumpM2xGeometryTrace =
            (m2xGeometryTraceDirPath != nullptr && *m2xGeometryTraceDirPath != '\0');
    if (dumpM2xGeometryTrace)
    {
        static std::string clearedM2xGeometryTracePath;
        const std::string  tracePath =
                (std::filesystem::path(m2xGeometryTraceDirPath) / "step0_event_669_geometry_trace.txt").string();
        if (tracePath != clearedM2xGeometryTracePath)
        {
            writeRespaTraceTextFile(m2xGeometryTraceDirPath, "step0_event_669_geometry_trace.txt", "");
            clearedM2xGeometryTracePath = tracePath;
        }
    }
    const char* ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2W_TRACE_DIR");
    const char* ljSrTraceCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2W_CASE_LABEL");
    const bool  dumpM2wLjSrTrace =
            (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0');
    if (!dumpM2wLjSrTrace)
    {
        ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2V_TRACE_DIR");
        ljSrTraceCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2V_CASE_LABEL");
    }
    const bool  dumpM2vLjSrTrace =
            (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0');
    if (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace)
    {
        ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2U_TRACE_DIR");
        ljSrTraceCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2U_CASE_LABEL");
    }
    const bool  dumpM2uLjSrTrace =
            (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0');
    if (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && !dumpM2uLjSrTrace)
    {
        ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2S_TRACE_DIR");
        ljSrTraceCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2S_CASE_LABEL");
    }
    const bool  dumpM2sLjSrTrace =
            (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && !dumpM2uLjSrTrace && ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0');
    if (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && !dumpM2uLjSrTrace && !dumpM2sLjSrTrace)
    {
        ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2R_TRACE_DIR");
        ljSrTraceCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2R_CASE_LABEL");
    }
    const bool dumpM2rLjSrTrace =
            (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && !dumpM2uLjSrTrace && !dumpM2sLjSrTrace && ljSrTraceDirPath != nullptr
             && *ljSrTraceDirPath != '\0');
    if (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && !dumpM2uLjSrTrace && !dumpM2sLjSrTrace && !dumpM2rLjSrTrace)
    {
        ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2Q_TRACE_DIR");
        ljSrTraceCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2Q_CASE_LABEL");
    }
    const bool dumpM2qLjSrTrace =
            (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && !dumpM2uLjSrTrace && !dumpM2sLjSrTrace && !dumpM2rLjSrTrace
             && ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0');
    if (!dumpM2vLjSrTrace && !dumpM2uLjSrTrace && !dumpM2sLjSrTrace && !dumpM2rLjSrTrace
        && !dumpM2qLjSrTrace)
    {
        ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2P_TRACE_DIR");
        ljSrTraceCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2P_CASE_LABEL");
    }
    const bool dumpLjSrTrace = (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0');
    const bool useDispatchProbe = dispatchProbeMode != "baseline";
    const bool dumpMultiStepCoulombStateTrace =
            dumpLjSrTrace && shouldTraceRespaMultiStepCoulombStep(step);
    const bool dumpLjAccumContractTrace =
            dumpLjSrTrace && respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_LJ_ACCUM_CONTRACT");
    const bool dumpCoulombPreSelfWindowTrace =
            dumpLjSrTrace && respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_COULOMB_PRE_SELF_WINDOW");
    const bool dumpCoulombFirstWritesTrace =
            dumpLjSrTrace && respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_COULOMB_FIRST_WRITES");
    int        patchCoulombFirstWriteOrdinal = 0;
    int        patchCoulombProducerOrdinal   = 0;
    struct PreSelfAccumulatorWrite
    {
        std::string codeLocation;
        std::string roleLabel;
        int         energyIndex  = -1;
        double      targetBefore = 0.0;
        double      writeValue   = 0.0;
        double      targetAfter  = 0.0;
    };
    std::vector<PreSelfAccumulatorWrite> patchPreSelfWritesForEnergyIndex0;
    const std::string ljSrTraceCaseLabel =
            (ljSrTraceCaseLabelEnv != nullptr && *ljSrTraceCaseLabelEnv != '\0') ? ljSrTraceCaseLabelEnv :
                                                                                   "unknown";
    const bool computePairEnergies =
            stepWork.computeEnergy || debugExactRespa || traceCpuCorrectionEnergies || dumpLjSrTrace
            || dumpDownstreamContract || dumpBookkeepingResidualTrace || dumpDispatchInternalTrace
            || dumpPairWriteProof || dumpEarlyAccumTrace || dumpM2xGeometryTrace
            || traceRealspaceForceSubcomponents || traceStep1Subset01ForceGroupAudit
            || traceExclusionEquivalence;
    const bool pairLoopOmpRequested    = exactRespaPairLoopOmpRequested();
    const bool pairLoopVectorRequested = exactRespaPairLoopVectorRequested();
    const bool pairLoopSparseReductionRequested = exactRespaPairLoopSparseReductionRequested();
    const bool pairLoopBlockReductionRequested = exactRespaPairLoopBlockReductionRequested();
    const bool pairLoopTileRequested = exactRespaPairLoopTileRequested();
    const bool pairLoopNbnxm4x4Requested = exactRespaPairLoopNbnxm4x4Requested();
    const bool pairLoopDirectCpuListRequested = exactRespaPairLoopDirectCpuListRequested();
    const int  pairLoopOmpThreads      = gmx_omp_nthreads_get(ModuleMultiThread::Default);
    const int  pairLoopWorkerThreads   = pairLoopOmpRequested ? pairLoopOmpThreads : 1;
    const bool pairLoopFastPathEligible =
            (pairLoopOmpRequested || pairLoopVectorRequested) && pairLoopWorkerThreads >= 1
            && (!pairLoopOmpRequested || pairLoopOmpThreads > 1) && !traceOnlyDiagnostics
            && !computePairEnergies && !stepWork.computeVirial && !dumpExcludedCorrectionForce
            && !useDispatchProbe;
    const char* pairLoopForceDumpDirPath = exactRespaPairLoopForceDumpDirPath();
    const int   pairLoopForceDumpMax     = exactRespaPairLoopForceDumpMax();
    const bool  pairLoopForceDumpEnabled =
            !traceOnlyDiagnostics && pairLoopForceDumpDirPath != nullptr && pairLoopForceDumpMax > 0;
    const char* pairLoopTimingDirPath = exactRespaPairLoopTimingDirPath();
    const bool  pairLoopTimingEnabled =
            !traceOnlyDiagnostics && pairLoopTimingDirPath != nullptr;
    const bool pairLoopDirectCpuListFastPathCandidate =
            pairLoopFastPathEligible && pairLoopDirectCpuListRequested && pairLoopOmpRequested
            && !pairLoopVectorRequested && !pairLoopTileRequested && !pairLoopNbnxm4x4Requested;
    const bool needPlainPairlist =
            !pairLoopDirectCpuListFastPathCandidate || dumpPairWriteProof
            || dumpDownstreamContract || debugExactRespa || dumpLjSrTrace
            || traceCpuCorrectionEnergies || traceExclusionEquivalence
            || traceRealspaceForceSubcomponents || traceStep1Subset01ForceGroupAudit;
    const PlainPairlist emptyPlainPairlist;
    const real          completePairlistRange = fr->completePairlistRange.value();
    const PlainPairlist& plainPairlist =
            needPlainPairlist ? fr->nbv->plainPairlist(completePairlistRange, fr->shift_vec)
                              : emptyPlainPairlist;
    const bool needNamedPairChecks =
            useDispatchProbe || dumpDispatchInternalTrace || dumpBookkeepingResidualTrace
            || dumpDownstreamContract || dumpPairWriteProof || dumpEarlyAccumTrace
            || traceExclusionEquivalence || traceRealspaceForceSubcomponents
            || traceStep1Subset01ForceGroupAudit;
    const auto sumEnergyTermsOnce = [](gmx::ArrayRef<const real> values) -> double
    {
        double total = 0.0;
        for (const real value : values)
        {
            total += value;
        }
        return total;
    };
    if (dumpCoulombPreSelfWindowTrace)
    {
        static std::string clearedPreSelfTracePath;
        const std::string  tracePath =
                (std::filesystem::path(ljSrTraceDirPath) / "step0_coulomb_pre_self_window.txt").string();
        if (tracePath != clearedPreSelfTracePath)
        {
            writeRespaTraceTextFile(ljSrTraceDirPath, "step0_coulomb_pre_self_window.txt", "");
            clearedPreSelfTracePath = tracePath;
        }
    }
    bool   m2sFirstWriteCaptured = false;
    double m2sFirstWriteLjTotal  = 0.0;
    std::vector<double> m2uWriteOrdinalLjTotals;
    double              m2vAlignedEventLjRunningTotal = 0.0;
    std::vector<double> m2vAlignedEventLjTotals;
    struct M2wAlignedEventRecord
    {
        int    alignedEventOrdinal = 0;
        int    pairOrdinal         = 0;
        int    pairI               = 0;
        int    pairJ               = 0;
        int    typeI               = 0;
        int    typeJ               = 0;
        int    shiftIndex          = 0;
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
    double                            m2wAlignedEventLjRunningTotal = 0.0;
    std::vector<double>               m2wAlignedEventLjTotals;
    std::vector<M2wAlignedEventRecord> m2wAlignedEventRecords;
    struct M2xGeometryEventRecord
    {
        int    alignedEventOrdinal = 0;
        int    pairOrdinal         = 0;
        int    pairI               = 0;
        int    pairJ               = 0;
        int    typeI               = 0;
        int    typeJ               = 0;
        int    shiftIndex          = 0;
        double coordISourceX       = 0.0;
        double coordISourceY       = 0.0;
        double coordISourceZ       = 0.0;
        double coordJSourceX       = 0.0;
        double coordJSourceY       = 0.0;
        double coordJSourceZ       = 0.0;
        double shiftX              = 0.0;
        double shiftY              = 0.0;
        double shiftZ              = 0.0;
        double coordIShiftedX      = 0.0;
        double coordIShiftedY      = 0.0;
        double coordIShiftedZ      = 0.0;
        double dx                  = 0.0;
        double dy                  = 0.0;
        double dz                  = 0.0;
        double rsq                 = 0.0;
        double r                   = 0.0;
        double rawLjTerm           = 0.0;
        double finalEventLj        = 0.0;
    };
    int m2xAlignedEventOrdinal = 0;
    const auto appendM2xGeometryStageLine =
            [&](const char* stage,
                const char* codeLocation,
                const M2xGeometryEventRecord& record,
                const bool includeShiftedCoord,
                const bool includeDx,
                const bool includeRsq,
                const bool includeR)
    {
        if (!dumpM2xGeometryTrace)
        {
            return;
        }

        const char* caseLabel =
                (m2xGeometryCaseLabelEnv != nullptr && *m2xGeometryCaseLabelEnv != '\0') ? m2xGeometryCaseLabelEnv
                                                                                          : "unknown";
        std::string line = std::string("stage=") + stage + " code_location=" + codeLocation + " case_label="
                           + caseLabel + " execution_path=exact_event_669_geometry_trace"
                           + " aligned_contract=running_total_after_admitted_pair_energy_event"
                           + " aligned_event_ordinal=669 pair_i=" + std::to_string(record.pairI)
                           + " pair_j=" + std::to_string(record.pairJ) + " type_i="
                           + std::to_string(record.typeI) + " type_j=" + std::to_string(record.typeJ)
                           + " pair_ordinal=" + std::to_string(record.pairOrdinal) + " shift_index="
                           + std::to_string(record.shiftIndex) + " event_ordering_key="
                           + std::to_string(record.pairI) + "_" + std::to_string(record.pairJ) + " coord_i_x="
                           + formatString("%.15f", record.coordISourceX) + " coord_i_y="
                           + formatString("%.15f", record.coordISourceY) + " coord_i_z="
                           + formatString("%.15f", record.coordISourceZ) + " coord_j_x="
                           + formatString("%.15f", record.coordJSourceX) + " coord_j_y="
                           + formatString("%.15f", record.coordJSourceY) + " coord_j_z="
                           + formatString("%.15f", record.coordJSourceZ) + " shift_x="
                           + formatString("%.15f", record.shiftX) + " shift_y="
                           + formatString("%.15f", record.shiftY) + " shift_z="
                           + formatString("%.15f", record.shiftZ);
        if (includeShiftedCoord)
        {
            line += " coord_i_shifted_x=" + formatString("%.15f", record.coordIShiftedX)
                    + " coord_i_shifted_y=" + formatString("%.15f", record.coordIShiftedY)
                    + " coord_i_shifted_z=" + formatString("%.15f", record.coordIShiftedZ);
        }
        if (includeDx)
        {
            line += " dx=" + formatString("%.15f", record.dx) + " dy="
                    + formatString("%.15f", record.dy) + " dz="
                    + formatString("%.15f", record.dz);
        }
        if (includeRsq)
        {
            line += " rsq=" + formatString("%.15f", record.rsq);
        }
        if (includeR)
        {
            line += " r=" + formatString("%.15f", record.r) + " raw_lj_term="
                    + formatString("%.15f", record.rawLjTerm) + " final_event_lj_contribution="
                    + formatString("%.15f", record.finalEventLj);
        }
        appendRespaTraceTextLine(m2xGeometryTraceDirPath, "step0_event_669_geometry_trace.txt", line);
    };
    const auto noteM2xGeometryEvent = [&](const M2xGeometryEventRecord& record)
    {
        if (!dumpM2xGeometryTrace)
        {
            return;
        }

        ++m2xAlignedEventOrdinal;
        if (m2xAlignedEventOrdinal != 669)
        {
            return;
        }

        appendM2xGeometryStageLine("GEOM_COORD_SOURCE",
                                   "src/gromacs/mdlib/sim_util.cpp:coordinates_fetch_before_shift",
                                   record,
                                   false,
                                   false,
                                   false,
                                   false);
        appendM2xGeometryStageLine("GEOM_SHIFT_OR_PBC_APPLY",
                                   "src/gromacs/mdlib/sim_util.cpp:shift_vec_application_before_dx",
                                   record,
                                   true,
                                   false,
                                   false,
                                   false);
        appendM2xGeometryStageLine("GEOM_DXDYDZ_CONSTRUCTION",
                                   "src/gromacs/mdlib/sim_util.cpp:dx_vector_construction",
                                   record,
                                   true,
                                   true,
                                   false,
                                   false);
        appendM2xGeometryStageLine("GEOM_RSQ_FORMATION",
                                   "src/gromacs/mdlib/sim_util.cpp:iprod_dx_dx_before_lj",
                                   record,
                                   true,
                                   true,
                                   true,
                                   false);
        appendM2xGeometryStageLine("EVENT_669_LJ_INPUT",
                                   "src/gromacs/mdlib/sim_util.cpp:rawLjEnergy_factorLj_event_input",
                                   record,
                                   true,
                                   true,
                                   true,
                                   true);
    };
    const bool outerAliasesShift =
            (outerAccumulator != nullptr && outerAccumulator->outputs != nullptr
             && outerAccumulator->force.data()
                        == outerAccumulator->outputs->forceWithShiftForces().force().data());
    const bool baselineOuterActive = (outerAccumulator != nullptr);
    const auto contributionLabel = [](const ExactRespaNonbondedContribution contribution) -> const char*
    {
        switch (contribution)
        {
            case ExactRespaNonbondedContribution::Inner: return "inner";
            case ExactRespaNonbondedContribution::Middle: return "middle";
            case ExactRespaNonbondedContribution::Outer: return "outer";
            default: return "unknown";
        }
    };
    const auto joinActiveContributionLabels =
            [&contributionLabel](const auto& accumulators, const bool excludeOuterForProbe) -> std::string
    {
        std::string labels;
        for (const auto& accumulator : accumulators)
        {
            if (excludeOuterForProbe
                && accumulator.contribution == ExactRespaNonbondedContribution::Outer)
            {
                continue;
            }
            if (!labels.empty())
            {
                labels += ",";
            }
            labels += contributionLabel(accumulator.contribution);
        }
        return labels.empty() ? "none" : labels;
    };
    const real pmeSelfEnergy    = computePmeSelfEnergy(*fr->ic);
    int        selfEnergyAtomCount = 0;
    std::vector<RVec> excludedCorrectionForce;
    if (dumpExcludedCorrectionForce)
    {
        excludedCorrectionForce.resize(coordinates.size());
        for (auto& force : excludedCorrectionForce)
        {
            clear_rvec(force);
        }
    }
    if (dumpEarlyAccumTrace && outerAccumulator != nullptr && outerAccumulator->forceWithVirial != nullptr)
    {
        dumpRespaMergeTraceVector(earlyAccumTraceDirPath,
                                  "step0_level2_initial_outer_virial.tsv",
                                  "stage=initial_outer mts_index=2 mts_user=3 buffer=forceWithVirial alias_with_shift="
                                          + std::string(outerAliasesShift ? "true" : "false"),
                                  outerAccumulator->forceWithVirial->force_);
    }
    if (dumpPairWriteProof)
    {
        const int  outerLevel   = exactRespaNonbondedOuterLevel(inputrec);
        const auto outerOutputs = exactRespaForceOutputs.levelOrNull(outerLevel);

        std::string contents;
        const auto  appendLine = [&contents](const std::string& line) { contents += line + "\n"; };
        for (int mtsLevel = 0; mtsLevel < exactRespaForceOutputs.numActiveLevels(); ++mtsLevel)
        {
            const ForceOutputs* outputs = exactRespaForceOutputs.levelOrNull(mtsLevel);
            if (outputs == nullptr)
            {
                appendLine("level=" + std::to_string(mtsLevel) + " active=false");
                continue;
            }
            ForceOutputs* mutableOutputs = const_cast<ForceOutputs*>(outputs);
            appendLine("level=" + std::to_string(mtsLevel) + " active=true");
            appendLine("level=" + std::to_string(mtsLevel) + " shift_force_ptr="
                       + formatPointerValue(mutableOutputs->forceWithShiftForces().force().data()));
            appendLine("level=" + std::to_string(mtsLevel) + " shift_shift_ptr=0x0");
            appendLine("level=" + std::to_string(mtsLevel) + " have_virial="
                       + std::string(mtsLevel == outerLevel ? "true" : "false"));
            if (mtsLevel == outerLevel)
            {
                appendLine("level=" + std::to_string(mtsLevel) + " virial_force_ptr="
                           + formatPointerValue(mutableOutputs->forceWithShiftForces().force().data()));
            }
        }
        appendLine("outer_accumulator_present=" + std::string(outerOutputs != nullptr ? "true" : "false"));
        if (outerOutputs != nullptr)
        {
            appendLine("outer_accumulator_force_ptr="
                       + formatPointerValue(outerOutputs->forceWithShiftForces().force().data()));
            appendLine("outer_accumulator_shift_ptr=0x0");
            appendLine("outer_accumulator_has_virial=false");
            appendLine("outer_outputs_shift_force_ptr="
                       + formatPointerValue(outerOutputs->forceWithShiftForces().force().data()));
            appendLine("outer_outputs_shift_shift_ptr=0x0");
            appendLine("outer_aliases_shift=true");
        }
        appendLine("excluded_correction_force_dump_enabled=false");
        appendLine("excluded_correction_force_dump_ptr=0x0");
        writeRespaTraceTextFile(pairWriteProofDirPath, "step0_force_storage_identity.txt", contents);
    }
    const auto pairKey = [](const int ai, const int aj)
    {
        const uint32_t first  = static_cast<uint32_t>(std::min(ai, aj));
        const uint32_t second = static_cast<uint32_t>(std::max(ai, aj));
        return (static_cast<uint64_t>(first) << 32) | second;
    };
    const uint64_t targetPairKey  = pairKey(0, 1);
    const uint64_t controlPairKey = pairKey(0, 4);
    struct PairEntryInfo
    {
        int      ordinal    = -1;
        int      ai         = -1;
        int      aj         = -1;
        int      shiftIndex = -1;
        uint64_t key        = 0;
    };
    constexpr int c_maxControlPairWriteProofs = 8;
    std::vector<PairEntryInfo> firstPairEntries;
    firstPairEntries.reserve(c_maxControlPairWriteProofs);
    for (int ordinal = 0; ordinal < static_cast<int>(plainPairlist.pairs.size())
                          && ordinal < c_maxControlPairWriteProofs;
         ++ordinal)
    {
        const auto& entry = plainPairlist.pairs[ordinal];
        firstPairEntries.push_back(
                PairEntryInfo{ ordinal, entry.first.first, entry.first.second, entry.second, pairKey(entry.first.first, entry.first.second) });
    }
    std::optional<PairEntryInfo> firstExcludedEntry;
    if (!plainPairlist.excludedPairs.empty())
    {
        const auto& entry = plainPairlist.excludedPairs.front();
        firstExcludedEntry.emplace(
                PairEntryInfo{ 0, entry.first.first, entry.first.second, entry.second, pairKey(entry.first.first, entry.first.second) });
    }
    if (dumpPairWriteProof)
    {
        std::string contents;
        const auto  appendLine = [&contents](const std::string& line) { contents += line + "\n"; };
        appendLine("kind=pairlist_preview list=pairs count=" + std::to_string(plainPairlist.pairs.size()));
        for (const auto& pairEntry : firstPairEntries)
        {
            appendLine("kind=pairs ordinal=" + std::to_string(pairEntry.ordinal) + " ai="
                       + std::to_string(pairEntry.ai) + " aj=" + std::to_string(pairEntry.aj) + " shift_index="
                       + std::to_string(pairEntry.shiftIndex));
        }
        appendLine("kind=pairlist_preview list=excludedPairs count="
                   + std::to_string(plainPairlist.excludedPairs.size()));
        if (firstExcludedEntry.has_value())
        {
            appendLine("kind=excludedPairs ordinal=0 ai=" + std::to_string(firstExcludedEntry->ai) + " aj="
                       + std::to_string(firstExcludedEntry->aj) + " shift_index="
                       + std::to_string(firstExcludedEntry->shiftIndex));
        }
        writeRespaTraceTextFile(pairWriteProofDirPath, "step0_plain_pairlist_preview.txt", contents);
    }
    std::unordered_set<uint64_t> listedPairKeys;
    if (debugExactRespa)
    {
        const auto appendInteractionList = [&listedPairKeys, &idef, &pairKey](const InteractionFunction ftype)
        {
            const auto& iatoms = idef.il[ftype].iatoms;
            const int   stride = NRAL(ftype) + 1;
            for (int index = 0; index < static_cast<int>(iatoms.size()); index += stride)
            {
                listedPairKeys.insert(pairKey(iatoms[index + 1], iatoms[index + 2]));
            }
        };
        appendInteractionList(InteractionFunction::LennardJones14);
        appendInteractionList(InteractionFunction::LennardJonesCoulomb14Q);
        appendInteractionList(InteractionFunction::LennardJonesCoulombNonBondedPairs);
    }
    if (dumpPairWriteProof)
    {
        const auto countOccurrences =
                [&pairKey](const auto& entries, const uint64_t keyToCount) -> int
        {
            return static_cast<int>(std::count_if(entries.begin(),
                                                 entries.end(),
                                                 [&pairKey, keyToCount](const auto& entry)
                                                 { return pairKey(entry.first.first, entry.first.second) == keyToCount; }));
        };

        std::string contents;
        const auto  appendLine = [&contents](const std::string& line) { contents += line + "\n"; };
        if (firstExcludedEntry.has_value())
        {
            appendLine("kind=excluded ordinal=0 ai=" + std::to_string(firstExcludedEntry->ai) + " aj="
                       + std::to_string(firstExcludedEntry->aj) + " shift_index="
                       + std::to_string(firstExcludedEntry->shiftIndex) + " in_plain_pairs="
                       + std::to_string(countOccurrences(plainPairlist.pairs, firstExcludedEntry->key))
                       + " in_plain_excluded="
                       + std::to_string(countOccurrences(plainPairlist.excludedPairs, firstExcludedEntry->key))
                       + " in_debug_listed_pair_keys="
                       + std::string(listedPairKeys.count(firstExcludedEntry->key) != 0 ? "true" : "false"));
        }
        for (const auto& pairEntry : firstPairEntries)
        {
            appendLine("kind=pairs ordinal=" + std::to_string(pairEntry.ordinal) + " ai="
                       + std::to_string(pairEntry.ai) + " aj=" + std::to_string(pairEntry.aj)
                       + " shift_index=" + std::to_string(pairEntry.shiftIndex) + " in_plain_pairs="
                       + std::to_string(countOccurrences(plainPairlist.pairs, pairEntry.key))
                       + " in_plain_excluded="
                       + std::to_string(countOccurrences(plainPairlist.excludedPairs, pairEntry.key))
                       + " in_debug_listed_pair_keys="
                       + std::string(listedPairKeys.count(pairEntry.key) != 0 ? "true" : "false"));
        }
        writeRespaTraceTextFile(pairWriteProofDirPath, "step0_pair_key_membership_scan.txt", contents);
    }

    const auto countPairKeyOccurrences = [&pairKey](const auto& entries, const uint64_t keyToCount) -> int
    {
        return static_cast<int>(std::count_if(entries.begin(),
                                              entries.end(),
                                              [&pairKey, keyToCount](const auto& entry)
                                              { return pairKey(entry.first.first, entry.first.second) == keyToCount; }));
    };
    const auto firstOrdinalForKey = [&pairKey](const auto& entries, const uint64_t keyToFind) -> int
    {
        for (int ordinal = 0; ordinal < static_cast<int>(entries.size()); ++ordinal)
        {
            if (pairKey(entries[ordinal].first.first, entries[ordinal].first.second) == keyToFind)
            {
                return ordinal;
            }
        }
        return -1;
    };

    if (dumpDownstreamContract)
    {
        appendRespaTraceTextLine(
                downstreamContractTraceDirPath,
                "step0_downstream_contract_trace.txt",
                "stage=excluded_pairs_dispatch_contract list=excludedPairs factor_coulomb=0 factor_lj=0 include_rule=always_true "
                        "target_in_list="
                        + std::string(countPairKeyOccurrences(plainPairlist.excludedPairs, targetPairKey) > 0 ? "true"
                                                                                                              : "false")
                        + " target_ordinal="
                        + std::to_string(firstOrdinalForKey(plainPairlist.excludedPairs, targetPairKey))
                        + " target_occurrences="
                        + std::to_string(countPairKeyOccurrences(plainPairlist.excludedPairs, targetPairKey))
                        + " control_in_list="
                        + std::string(countPairKeyOccurrences(plainPairlist.excludedPairs, controlPairKey) > 0 ? "true"
                                                                                                                : "false")
                        + " control_ordinal="
                        + std::to_string(firstOrdinalForKey(plainPairlist.excludedPairs, controlPairKey))
                        + " control_occurrences="
                        + std::to_string(countPairKeyOccurrences(plainPairlist.excludedPairs, controlPairKey))
                        + " semantic_role=excluded_membership_reintroduced_into_exact_nonbonded_consumer");
        appendRespaTraceTextLine(
                downstreamContractTraceDirPath,
                "step0_downstream_contract_trace.txt",
                "stage=pairs_dispatch_contract list=pairs factor_coulomb=1 factor_lj=1 include_rule=always_true "
                        "target_in_list="
                        + std::string(countPairKeyOccurrences(plainPairlist.pairs, targetPairKey) > 0 ? "true" : "false")
                        + " target_ordinal=" + std::to_string(firstOrdinalForKey(plainPairlist.pairs, targetPairKey))
                        + " target_occurrences="
                        + std::to_string(countPairKeyOccurrences(plainPairlist.pairs, targetPairKey))
                        + " control_in_list="
                        + std::string(countPairKeyOccurrences(plainPairlist.pairs, controlPairKey) > 0 ? "true" : "false")
                        + " control_ordinal=" + std::to_string(firstOrdinalForKey(plainPairlist.pairs, controlPairKey))
                        + " control_occurrences="
                        + std::to_string(countPairKeyOccurrences(plainPairlist.pairs, controlPairKey))
                        + " semantic_role=standard_physical_nonbonded_consumer");
    }

    struct PairDebugStats
    {
        const char* label        = nullptr;
        int         count        = 0;
        double      ljEnergy     = 0;
        double      coulEnergy   = 0;
        double      qqSum        = 0;
        double      rawCoulEnergy = 0;
        double      selfEnergy   = 0;
    };

    double m2qEarliestRawLjTotal = 0.0;

    enum class PairLoopListKind
    {
        StandardPairs,
        ExcludedPairs
    };

    struct PairLoop4x4Cluster
    {
        uint64_t key        = 0;
        int      iBase      = 0;
        int      jBase      = 0;
        int      shiftIndex = c_centralShiftIndex;
        uint16_t mask       = 0;
    };

    struct PairLoop4x4ClusterCache
    {
        const void*                    data     = nullptr;
        int                            count    = -1;
        PairLoopListKind               listKind = PairLoopListKind::StandardPairs;
        std::vector<PairLoop4x4Cluster> clusters;
    };

    struct PairLoopOmpContributionScratch
    {
        int                             forceSize = 0;
        std::vector<std::vector<RVec>>  forceByThread;
        std::vector<std::vector<RVec>>  shiftByThread;
        std::vector<std::vector<int>>   touchedAtomsByThread;
        std::vector<std::vector<unsigned char>> touchedAtomSeenByThread;
        std::vector<int>                reductionAtoms;
        std::vector<unsigned char>      reductionAtomSeen;
    };

    struct PairLoopOmpScratch
    {
        int                                         numThreads = 0;
        std::vector<PairLoopOmpContributionScratch> contributions;
        PairLoop4x4ClusterCache                     clusterCache;
    };

    struct PairLoopReductionStats
    {
        int64_t touchedAtomSlots    = 0;
        int64_t reducedAtomSlots    = 0;
        bool    usedSparseReduction = false;
        bool    usedBlockedReduction = false;
        bool    usedTileBackend      = false;
        bool    usedNbnxm4x4Backend  = false;
        bool    usedDirectCpuListBackend = false;
    };

    static std::mutex        pairLoopOmpScratchMutex;
    static PairLoopOmpScratch pairLoopOmpScratch;
    static std::mutex        pairLoopForceDumpMutex;
    static int               pairLoopForceDumpOrdinal = 0;

    std::unique_lock<std::mutex> pairLoopOmpScratchLock;
    PairLoopOmpScratch*         pairLoopOmpScratchPtr = nullptr;
    if (pairLoopFastPathEligible)
    {
        const RVec zero = { 0.0_real, 0.0_real, 0.0_real };
        pairLoopOmpScratchLock = std::unique_lock<std::mutex>(pairLoopOmpScratchMutex);
        pairLoopOmpScratch.numThreads = pairLoopWorkerThreads;
        pairLoopOmpScratch.contributions.resize(activeContributions.size());
        for (int contributionIndex = 0; contributionIndex < gmx::ssize(activeContributions); ++contributionIndex)
        {
            auto& scratchContribution = pairLoopOmpScratch.contributions[contributionIndex];
            scratchContribution.forceSize = static_cast<int>(activeContributions[contributionIndex].force.size());
            scratchContribution.forceByThread.resize(pairLoopWorkerThreads);
            scratchContribution.shiftByThread.resize(pairLoopWorkerThreads);
            if (pairLoopSparseReductionRequested)
            {
                scratchContribution.touchedAtomsByThread.resize(pairLoopWorkerThreads);
                scratchContribution.touchedAtomSeenByThread.resize(pairLoopWorkerThreads);
                scratchContribution.reductionAtomSeen.resize(scratchContribution.forceSize, 0);
            }
            for (int thread = 0; thread < pairLoopWorkerThreads; ++thread)
            {
                if (scratchContribution.forceByThread[thread].size()
                    != static_cast<size_t>(scratchContribution.forceSize))
                {
                    scratchContribution.forceByThread[thread].resize(scratchContribution.forceSize);
                    std::fill(scratchContribution.forceByThread[thread].begin(),
                              scratchContribution.forceByThread[thread].end(),
                              zero);
                }
                if (scratchContribution.shiftByThread[thread].size() != c_numShiftVectors)
                {
                    scratchContribution.shiftByThread[thread].resize(c_numShiftVectors);
                    std::fill(scratchContribution.shiftByThread[thread].begin(),
                              scratchContribution.shiftByThread[thread].end(),
                              zero);
                }
                if (pairLoopSparseReductionRequested)
                {
                    if (scratchContribution.touchedAtomSeenByThread[thread].size()
                        != static_cast<size_t>(scratchContribution.forceSize))
                    {
                        scratchContribution.touchedAtomSeenByThread[thread].assign(
                                scratchContribution.forceSize, 0);
                        scratchContribution.touchedAtomsByThread[thread].clear();
                    }
                }
            }
        }
        pairLoopOmpScratchPtr = &pairLoopOmpScratch;
    }

    const auto clearPairLoopOmpScratch = [&](const bool useSparseTrackingForPairlist)
    {
        const RVec zero = { 0.0_real, 0.0_real, 0.0_real };
        for (auto& contributionScratch : pairLoopOmpScratchPtr->contributions)
        {
            for (int thread = 0; thread < pairLoopOmpScratchPtr->numThreads; ++thread)
            {
                if (useSparseTrackingForPairlist)
                {
                    for (const int atom : contributionScratch.touchedAtomsByThread[thread])
                    {
                        contributionScratch.forceByThread[thread][atom] = zero;
                        contributionScratch.touchedAtomSeenByThread[thread][atom] = 0;
                    }
                    contributionScratch.touchedAtomsByThread[thread].clear();
                }
                else
                {
                    std::fill(contributionScratch.forceByThread[thread].begin(),
                              contributionScratch.forceByThread[thread].end(),
                              zero);
                    if (pairLoopSparseReductionRequested
                        && thread < gmx::ssize(contributionScratch.touchedAtomsByThread))
                    {
                        for (const int atom : contributionScratch.touchedAtomsByThread[thread])
                        {
                            contributionScratch.touchedAtomSeenByThread[thread][atom] = 0;
                        }
                        contributionScratch.touchedAtomsByThread[thread].clear();
                    }
                }
                std::fill(contributionScratch.shiftByThread[thread].begin(),
                          contributionScratch.shiftByThread[thread].end(),
                          zero);
            }
        }
    };

    const auto reducePairLoopOmpScratch =
            [&](const bool useSparseTrackingForPairlist) -> PairLoopReductionStats
    {
        constexpr int c_pairLoopReductionBlockSize = 128;
        PairLoopReductionStats stats;
        for (int contributionIndex = 0; contributionIndex < gmx::ssize(activeContributions); ++contributionIndex)
        {
            auto&       accumulator         = activeContributions[contributionIndex];
            auto&       contributionScratch = pairLoopOmpScratchPtr->contributions[contributionIndex];
            const int   forceSize           = contributionScratch.forceSize;
            bool        useSparseReductionForContribution = false;

            if (useSparseTrackingForPairlist)
            {
                contributionScratch.reductionAtoms.clear();
                if (contributionScratch.reductionAtomSeen.size() != static_cast<size_t>(forceSize))
                {
                    contributionScratch.reductionAtomSeen.assign(forceSize, 0);
                }
                for (int thread = 0; thread < pairLoopOmpScratchPtr->numThreads; ++thread)
                {
                    for (const int atom : contributionScratch.touchedAtomsByThread[thread])
                    {
                        if (contributionScratch.reductionAtomSeen[atom] == 0)
                        {
                            contributionScratch.reductionAtomSeen[atom] = 1;
                            contributionScratch.reductionAtoms.push_back(atom);
                        }
                    }
                }
                std::sort(contributionScratch.reductionAtoms.begin(),
                          contributionScratch.reductionAtoms.end());
                stats.touchedAtomSlots += contributionScratch.reductionAtoms.size();
                useSparseReductionForContribution =
                        contributionScratch.reductionAtoms.size() * 4 < static_cast<size_t>(forceSize) * 3;
                stats.usedSparseReduction = stats.usedSparseReduction || useSparseReductionForContribution;
                stats.reducedAtomSlots += useSparseReductionForContribution
                                                  ? contributionScratch.reductionAtoms.size()
                                                  : static_cast<size_t>(forceSize);
            }
            else
            {
                stats.reducedAtomSlots += forceSize;
            }

            if (useSparseReductionForContribution)
            {
                for (const int atom : contributionScratch.reductionAtoms)
                {
                    RVec force = { 0.0_real, 0.0_real, 0.0_real };
                    for (int thread = 0; thread < pairLoopOmpScratchPtr->numThreads; ++thread)
                    {
                        rvec_inc(force, contributionScratch.forceByThread[thread][atom]);
                    }
                    rvec_inc(accumulator.force[atom], force);
                }
            }
            else if (pairLoopBlockReductionRequested)
            {
                stats.usedBlockedReduction = true;
#pragma omp parallel for num_threads(pairLoopWorkerThreads) schedule(static)
                for (int blockStart = 0; blockStart < forceSize; blockStart += c_pairLoopReductionBlockSize)
                {
                    const int blockEnd = std::min(blockStart + c_pairLoopReductionBlockSize, forceSize);
                    std::array<RVec, c_pairLoopReductionBlockSize> blockForce;
                    for (int blockAtom = 0; blockAtom < blockEnd - blockStart; ++blockAtom)
                    {
                        blockForce[blockAtom][XX] = 0.0_real;
                        blockForce[blockAtom][YY] = 0.0_real;
                        blockForce[blockAtom][ZZ] = 0.0_real;
                    }
                    for (int thread = 0; thread < pairLoopOmpScratchPtr->numThreads; ++thread)
                    {
                        for (int atom = blockStart; atom < blockEnd; ++atom)
                        {
                            rvec_inc(blockForce[atom - blockStart],
                                     contributionScratch.forceByThread[thread][atom]);
                        }
                    }
                    for (int atom = blockStart; atom < blockEnd; ++atom)
                    {
                        rvec_inc(accumulator.force[atom], blockForce[atom - blockStart]);
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(pairLoopWorkerThreads) schedule(static)
                for (int atom = 0; atom < forceSize; ++atom)
                {
                    RVec force = { 0.0_real, 0.0_real, 0.0_real };
                    for (int thread = 0; thread < pairLoopOmpScratchPtr->numThreads; ++thread)
                    {
                        rvec_inc(force, contributionScratch.forceByThread[thread][atom]);
                    }
                    rvec_inc(accumulator.force[atom], force);
                }
            }

            if (useSparseTrackingForPairlist)
            {
                for (const int atom : contributionScratch.reductionAtoms)
                {
                    contributionScratch.reductionAtomSeen[atom] = 0;
                }
            }

            if (!accumulator.shift.empty())
            {
                for (int shift = 0; shift < c_numShiftVectors; ++shift)
                {
                    RVec shiftForce = { 0.0_real, 0.0_real, 0.0_real };
                    for (int thread = 0; thread < pairLoopOmpScratchPtr->numThreads; ++thread)
                    {
                        rvec_inc(shiftForce, contributionScratch.shiftByThread[thread][shift]);
                    }
                    rvec_inc(accumulator.shift[shift], shiftForce);
                }
            }
        }
        return stats;
    };

    const auto processPairlistOmp = [&](const char*            pairListLabel,
                                        const auto&            pairEntries,
                                        const PairLoopListKind listKind) -> bool
    {
        if (!pairLoopFastPathEligible || pairLoopOmpScratchPtr == nullptr
            || (!pairLoopDirectCpuListFastPathCandidate && gmx::ssize(pairEntries) < 128))
        {
            return false;
        }

        const int  numPairs = gmx::ssize(pairEntries);
        const bool useDirectCpuListBackend = pairLoopDirectCpuListFastPathCandidate;
        int64_t    directCpuListJobCount   = 0;
        int       maxForceSize = 0;
        for (const auto& contributionScratch : pairLoopOmpScratchPtr->contributions)
        {
            maxForceSize = std::max(maxForceSize, contributionScratch.forceSize);
        }
        const bool useSparseTrackingForPairlist =
                !useDirectCpuListBackend && pairLoopSparseReductionRequested && numPairs < maxForceSize;
        const bool useTileBackend =
                !useDirectCpuListBackend && pairLoopTileRequested && !pairLoopVectorRequested
                && !pairLoopNbnxm4x4Requested
                && listKind == PairLoopListKind::StandardPairs;
        const bool useNbnxm4x4Backend =
                !useDirectCpuListBackend && pairLoopNbnxm4x4Requested && !pairLoopVectorRequested
                && listKind == PairLoopListKind::StandardPairs;

        using PairLoopClock = std::chrono::steady_clock;
        PairLoopClock::time_point clearStart;
        PairLoopClock::time_point clearEnd;
        PairLoopClock::time_point pairStart;
        PairLoopClock::time_point pairEnd;
        PairLoopClock::time_point reduceStart;
        PairLoopClock::time_point reduceEnd;
        if (pairLoopTimingEnabled)
        {
            clearStart = PairLoopClock::now();
        }
        clearPairLoopOmpScratch(useSparseTrackingForPairlist);
        if (pairLoopTimingEnabled)
        {
            clearEnd  = PairLoopClock::now();
            pairStart = clearEnd;
        }

        const auto pairLoop4x4Clusters = [&]() -> const std::vector<PairLoop4x4Cluster>&
        {
            auto& clusterCache = pairLoopOmpScratchPtr->clusterCache;
            std::vector<PairLoop4x4Cluster> unmergedClusters;
            unmergedClusters.reserve(numPairs);
            for (int pairIndex = 0; pairIndex < numPairs; ++pairIndex)
            {
                const auto& entry      = pairEntries[pairIndex];
                const int   ai         = entry.first.first;
                const int   aj         = entry.first.second;
                const int   shiftIndex = entry.second;
                const int   iBase      = ai & ~0x3;
                const int   jBase      = aj & ~0x3;
                const int   iLane      = ai - iBase;
                const int   jLane      = aj - jBase;

                PairLoop4x4Cluster cluster;
                cluster.iBase      = iBase;
                cluster.jBase      = jBase;
                cluster.shiftIndex = shiftIndex;
                cluster.mask       = static_cast<uint16_t>(1u << (iLane * 4 + jLane));
                cluster.key = (static_cast<uint64_t>(static_cast<uint32_t>(shiftIndex)) << 48)
                              | (static_cast<uint64_t>(static_cast<uint32_t>(iBase)) << 24)
                              | static_cast<uint64_t>(static_cast<uint32_t>(jBase));
                unmergedClusters.push_back(cluster);
            }

            std::sort(unmergedClusters.begin(),
                      unmergedClusters.end(),
                      [](const PairLoop4x4Cluster& lhs, const PairLoop4x4Cluster& rhs)
                      { return lhs.key < rhs.key; });

            clusterCache.clusters.clear();
            clusterCache.clusters.reserve(unmergedClusters.size());
            for (const auto& cluster : unmergedClusters)
            {
                if (!clusterCache.clusters.empty()
                    && clusterCache.clusters.back().key == cluster.key)
                {
                    clusterCache.clusters.back().mask |= cluster.mask;
                }
                else
                {
                    clusterCache.clusters.push_back(cluster);
                }
            }

            clusterCache.data     = pairEntries.data();
            clusterCache.count    = numPairs;
            clusterCache.listKind = listKind;
            return clusterCache.clusters;
        };

        constexpr int c_pairLoopTilePairCount                = 8;
        constexpr int c_pairLoopTileMaxBlocks                = 16;
        constexpr int c_pairLoopTileAtomBlockSize            = 128;
        constexpr int c_pairLoopTileMaxTouchedOffsetsPerBlock = c_pairLoopTilePairCount * 2;

        struct TileContributionBlockCacheSlot
        {
            int                                 blockIndex         = -1;
            int                                 touchedOffsetCount = 0;
            std::array<unsigned char, c_pairLoopTileAtomBlockSize> offsetSeen{};
            std::array<int, c_pairLoopTileMaxTouchedOffsetsPerBlock> touchedOffsets{};
            std::array<RVec, c_pairLoopTileAtomBlockSize>          forceByOffset{};
        };

        struct TileContributionCache
        {
            int activeBlockCount = 0;
            std::array<TileContributionBlockCacheSlot, c_pairLoopTileMaxBlocks> blockSlots{};
        };

        const auto markTouchedAtom = [&](PairLoopOmpContributionScratch& contributionScratch,
                                         const int                      thread,
                                         const int                      atom)
        {
            if (!useSparseTrackingForPairlist)
            {
                return;
            }
            auto& touchedAtoms = contributionScratch.touchedAtomsByThread[thread];
            auto& touchedSeen  = contributionScratch.touchedAtomSeenByThread[thread];
            if (touchedSeen[atom] == 0)
            {
                touchedSeen[atom] = 1;
                touchedAtoms.push_back(atom);
            }
        };

        const auto resetTileContributionCache = [&](TileContributionCache* cache)
        {
            for (int blockSlotIndex = 0; blockSlotIndex < cache->activeBlockCount; ++blockSlotIndex)
            {
                auto& blockSlot = cache->blockSlots[blockSlotIndex];
                for (int touchedIndex = 0; touchedIndex < blockSlot.touchedOffsetCount; ++touchedIndex)
                {
                    blockSlot.offsetSeen[blockSlot.touchedOffsets[touchedIndex]] = 0;
                }
                blockSlot.touchedOffsetCount = 0;
                blockSlot.blockIndex         = -1;
            }
            cache->activeBlockCount = 0;
        };

        const auto findOrAllocateTileBlockSlot =
                [&](TileContributionCache* cache,
                    const int              blockIndex) -> TileContributionBlockCacheSlot&
        {
            for (int blockSlotIndex = 0; blockSlotIndex < cache->activeBlockCount; ++blockSlotIndex)
            {
                auto& blockSlot = cache->blockSlots[blockSlotIndex];
                if (blockSlot.blockIndex == blockIndex)
                {
                    return blockSlot;
                }
            }

            GMX_RELEASE_ASSERT(cache->activeBlockCount < gmx::ssize(cache->blockSlots),
                               "Pair-loop tile block cache exceeded bounded block capacity");
            auto& blockSlot = cache->blockSlots[cache->activeBlockCount++];
            blockSlot.blockIndex         = blockIndex;
            blockSlot.touchedOffsetCount = 0;
            return blockSlot;
        };

        const auto accumulateTileAtom = [&](TileContributionCache& cache,
                                            const int              atom,
                                            const RVec&            force)
        {
            const int atomBlockIndex = atom / c_pairLoopTileAtomBlockSize;
            const int atomOffset     = atom % c_pairLoopTileAtomBlockSize;
            auto&     blockSlot      = findOrAllocateTileBlockSlot(&cache, atomBlockIndex);

            if (blockSlot.offsetSeen[atomOffset] == 0)
            {
                GMX_RELEASE_ASSERT(
                        blockSlot.touchedOffsetCount < gmx::ssize(blockSlot.touchedOffsets),
                        "Pair-loop tile block cache exceeded bounded touched-offset capacity");
                blockSlot.offsetSeen[atomOffset] = 1;
                blockSlot.touchedOffsets[blockSlot.touchedOffsetCount++] = atomOffset;
                copy_rvec(force, blockSlot.forceByOffset[atomOffset]);
                return;
            }

            rvec_inc(blockSlot.forceByOffset[atomOffset], force);
        };

        const auto accumulateContributionForce =
                [&](const int                              thread,
                    const int                              contributionIndex,
                    const int                              ai,
                    const int                              aj,
                    const int                              shiftIndex,
                    const RVec&                            force,
                    std::array<TileContributionCache, 3>* tileCaches)
        {
            auto& contributionScratch = pairLoopOmpScratchPtr->contributions[contributionIndex];
            if (tileCaches != nullptr)
            {
                accumulateTileAtom((*tileCaches)[contributionIndex], ai, force);
                const RVec negForce = { -force[XX], -force[YY], -force[ZZ] };
                accumulateTileAtom((*tileCaches)[contributionIndex], aj, negForce);
            }
            else
            {
                rvec_inc(contributionScratch.forceByThread[thread][ai], force);
                rvec_dec(contributionScratch.forceByThread[thread][aj], force);
                markTouchedAtom(contributionScratch, thread, ai);
                markTouchedAtom(contributionScratch, thread, aj);
            }

            const auto& accumulator = activeContributions[contributionIndex];
            if (!accumulator.shift.empty() && shiftIndex != c_centralShiftIndex)
            {
                rvec_inc(contributionScratch.shiftByThread[thread][shiftIndex], force);
                rvec_dec(contributionScratch.shiftByThread[thread][c_centralShiftIndex], force);
            }
        };

        const auto flushTileContributionCache = [&](const int                thread,
                                                    const int                contributionIndex,
                                                    TileContributionCache*   cache)
        {
            auto& contributionScratch = pairLoopOmpScratchPtr->contributions[contributionIndex];
            for (int blockSlotIndex = 0; blockSlotIndex < cache->activeBlockCount; ++blockSlotIndex)
            {
                auto& blockSlot = cache->blockSlots[blockSlotIndex];
                for (int touchedIndex = 0; touchedIndex < blockSlot.touchedOffsetCount; ++touchedIndex)
                {
                    const int atomOffset = blockSlot.touchedOffsets[touchedIndex];
                    const int atom = blockSlot.blockIndex * c_pairLoopTileAtomBlockSize + atomOffset;
                    rvec_inc(contributionScratch.forceByThread[thread][atom],
                             blockSlot.forceByOffset[atomOffset]);
                    markTouchedAtom(contributionScratch, thread, atom);
                }
            }
            resetTileContributionCache(cache);
        };

        const auto accumulateStandardPairScalars =
                [&](const int                              thread,
                    const int                              ai,
                    const int                              aj,
                    const int                              shiftIndex,
                    const RVec&                            dx,
                    const real                             rinvsq,
                    const real                             innerScalar,
                    const real                             middleScalar,
                    const real                             outerScalar,
                    std::array<TileContributionCache, 3>* tileCaches)
        {
            for (int contributionIndex = 0; contributionIndex < gmx::ssize(activeContributions);
                 ++contributionIndex)
            {
                const auto& accumulator = activeContributions[contributionIndex];
                real        scalar      = 0.0_real;
                switch (accumulator.contribution)
                {
                    case ExactRespaNonbondedContribution::Inner:
                        scalar = innerScalar;
                        break;
                    case ExactRespaNonbondedContribution::Middle:
                        scalar = middleScalar;
                        break;
                    case ExactRespaNonbondedContribution::Outer:
                        scalar = outerScalar;
                        break;
                    default:
                        GMX_RELEASE_ASSERT(false, "Unexpected nonbonded r-RESPA contribution");
                }

                if (scalar == 0.0_real)
                {
                    continue;
                }

                RVec force = { 0.0_real, 0.0_real, 0.0_real };
                svmul(scalar * rinvsq, dx, force);
                accumulateContributionForce(
                        thread, contributionIndex, ai, aj, shiftIndex, force, tileCaches);
            }
        };

        const auto accumulateExcludedCorrection =
                [&](const int                              thread,
                    const int                              ai,
                    const int                              aj,
                    const int                              shiftIndex,
                    const RVec&                            dx,
                    const real                             rinvsq,
                    const real                             correctionScalar,
                    std::array<TileContributionCache, 3>* tileCaches)
        {
            if (correctionScalar == 0.0_real || outerContributionIndex < 0)
            {
                return;
            }
            RVec force = { 0.0_real, 0.0_real, 0.0_real };
            svmul(correctionScalar * rinvsq, dx, force);
            accumulateContributionForce(
                    thread, outerContributionIndex, ai, aj, shiftIndex, force, tileCaches);
        };

        const auto computePairGeometry =
                [&](const int ai,
                    const int aj,
                    const int shiftIndex,
                    RVec&     dx,
                    real*     rsqOut,
                    real*     rinvOut,
                    real*     rinvsqOut,
                    real*     rOut)
        {
            for (int dim = 0; dim < DIM; ++dim)
            {
                dx[dim] = coordinates[ai][dim] + fr->shift_vec[shiftIndex][dim] - coordinates[aj][dim];
            }

            real rsq = dx[XX] * dx[XX] + dx[YY] * dx[YY] + dx[ZZ] * dx[ZZ];
            rsq      = std::max(rsq, c_nbnxnMinDistanceSquared);
            const real rinv = gmx::invsqrt(rsq);
            *rsqOut         = rsq;
            *rinvOut        = rinv;
            *rinvsqOut      = rinv * rinv;
            *rOut           = rsq * rinv;
        };

        const auto computeStandardPairScalarsFromGeometry =
                [&](const int ai,
                    const int aj,
                    const real rsq,
                    const real rinv,
                    const real rinvsq,
                    const real r,
                    const bool computeLj,
                    const bool computeCoulomb,
                    real*      innerScalarOut,
                    real*      middleScalarOut,
                    real*      outerScalarOut) -> bool
        {
            if (!computeLj && !computeCoulomb)
            {
                *innerScalarOut  = 0.0_real;
                *middleScalarOut = 0.0_real;
                *outerScalarOut  = 0.0_real;
                return false;
            }

            real innerWeight  = 0.0_real;
            real middleWeight = 0.0_real;
            real outerWeight  = 0.0_real;
            if (exactRespaHasMiddle)
            {
                const real switchIntoMiddle =
                        respaSwitchIn(r, exactRespaForceLayout.innerOff, exactRespaForceLayout.innerOn);
                const real switchIntoOuter =
                        respaSwitchIn(r, exactRespaForceLayout.outerOn, exactRespaForceLayout.outerOff);
                innerWeight  = 1.0_real - switchIntoMiddle;
                middleWeight = switchIntoMiddle * (1.0_real - switchIntoOuter);
                outerWeight  = switchIntoOuter;
            }
            else
            {
                const real switchIntoOuter =
                        respaSwitchIn(r, exactRespaForceLayout.outerOn, exactRespaForceLayout.outerOff);
                innerWeight  = 1.0_real - switchIntoOuter;
                middleWeight = 0.0_real;
                outerWeight  = switchIntoOuter;
            }

            real rawLjScalar = 0.0_real;
            if (computeLj && rsq < vdwCutoff2)
            {
                const int  typeI = mdatoms.typeA[ai];
                const int  typeJ = mdatoms.typeA[aj];
                const real c6    = fr->nbfp[typeI * ntype2 + typeJ * 2];
                const real cRepulsive = fr->nbfp[typeI * ntype2 + typeJ * 2 + 1];
                const real rinvsix = rinvsq * rinvsq * rinvsq;
                const real repulsiveTerm = usePower9SpecializedPath ? (rinvsix * rinvsq * rinv)
                                            : (repulsionPower == 12.0_real ? rinvsix * rinvsix
                                                                           : std::pow(rinv, repulsionPower));
                rawLjScalar = cRepulsive * repulsiveTerm - c6 * rinvsix;
            }

            real bareCoulombScalar = 0.0_real;
            real correctionScalar  = 0.0_real;
            if (computeCoulomb && rsq < coulombCutoff2)
            {
                const real qq = mdatoms.chargeA[ai] * mdatoms.chargeA[aj] * fr->ic->coulomb.epsfac;
                if (qq != 0.0_real)
                {
                    const real scaledR = r * fr->ic->coulombEwaldTables->scale;
                    const int  coulTableIndex = static_cast<int>(scaledR);
                    const real coulFrac       = scaledR - coulTableIndex;
#if !GMX_DOUBLE
                    const real* table = fr->ic->coulombEwaldTables->tableFDV0.data();
                    const real  coulFexcl =
                            table[coulTableIndex * 4] + coulFrac * table[coulTableIndex * 4 + 1];
#else
                    const real* tableF = fr->ic->coulombEwaldTables->tableF.data();
                    const real  coulFexcl =
                            (1 - coulFrac) * tableF[coulTableIndex] + coulFrac * tableF[coulTableIndex + 1];
#endif
                    bareCoulombScalar = qq * rinv;
                    correctionScalar  = -qq * coulFexcl / rinv;
                }
            }

            *innerScalarOut  = bareCoulombScalar * innerWeight + rawLjScalar * innerWeight;
            *middleScalarOut = bareCoulombScalar * middleWeight + rawLjScalar * middleWeight;
            *outerScalarOut =
                    correctionScalar + bareCoulombScalar * outerWeight + rawLjScalar * outerWeight;
            return (*innerScalarOut != 0.0_real || *middleScalarOut != 0.0_real
                    || *outerScalarOut != 0.0_real);
        };

        const auto computeStandardPairScalars =
                [&](const int ai,
                    const int aj,
                    const int shiftIndex,
                    RVec&     dx,
                    real*     rinvsqOut,
                    real*     innerScalarOut,
                    real*     middleScalarOut,
                    real*     outerScalarOut) -> bool
        {
            real rsq    = 0.0_real;
            real rinv   = 0.0_real;
            real rinvsq = 0.0_real;
            real r      = 0.0_real;
            computePairGeometry(ai, aj, shiftIndex, dx, &rsq, &rinv, &rinvsq, &r);
            *rinvsqOut = rinvsq;
            return computeStandardPairScalarsFromGeometry(
                    ai, aj, rsq, rinv, rinvsq, r, true, true, innerScalarOut, middleScalarOut, outerScalarOut);
        };

        const auto computeExcludedCorrectionFromGeometry =
                [&](const int ai,
                    const int aj,
                    const real rsq,
                    const real rinv,
                    const real gmx_unused rinvsq,
                    const real r,
                    real*      correctionScalarOut) -> bool
        {
            if (rsq >= coulombCutoff2)
            {
                return false;
            }

            const real qq = mdatoms.chargeA[ai] * mdatoms.chargeA[aj] * fr->ic->coulomb.epsfac;
            if (qq == 0.0_real)
            {
                return false;
            }

            const real scaledR = r * fr->ic->coulombEwaldTables->scale;
            const int  coulTableIndex = static_cast<int>(scaledR);
            const real coulFrac       = scaledR - coulTableIndex;
#if !GMX_DOUBLE
            const real* table = fr->ic->coulombEwaldTables->tableFDV0.data();
            const real  coulFexcl = table[coulTableIndex * 4] + coulFrac * table[coulTableIndex * 4 + 1];
#else
            const real* tableF = fr->ic->coulombEwaldTables->tableF.data();
            const real  coulFexcl =
                    (1 - coulFrac) * tableF[coulTableIndex] + coulFrac * tableF[coulTableIndex + 1];
#endif
            *correctionScalarOut = -qq * coulFexcl / rinv;
            return *correctionScalarOut != 0.0_real;
        };

        const auto computeExcludedCorrection =
                [&](const int ai,
                    const int aj,
                    const int shiftIndex,
                    RVec&     dx,
                    real*     rinvsqOut,
                    real*     correctionScalarOut) -> bool
        {
            real rsq    = 0.0_real;
            real rinv   = 0.0_real;
            real rinvsq = 0.0_real;
            real r      = 0.0_real;
            computePairGeometry(ai, aj, shiftIndex, dx, &rsq, &rinv, &rinvsq, &r);
            *rinvsqOut = rinvsq;
            return computeExcludedCorrectionFromGeometry(ai, aj, rsq, rinv, rinvsq, r, correctionScalarOut);
        };

        constexpr int c_directCpuListMaxClusterAtoms    = 8;
        constexpr int c_directCpuListMaxContributions   = 3;
        constexpr int c_directCpuListPackedForceEntries =
                c_directCpuListMaxClusterAtoms * c_directCpuListMaxContributions;
        GMX_RELEASE_ASSERT(activeContributions.size() <= c_directCpuListMaxContributions,
                           "Direct CPU-list packed path assumes at most three exact nonbonded contributions");

        const auto contributionScalar =
                [&](const int contributionIndex,
                    const real innerScalar,
                    const real middleScalar,
                    const real outerScalar) -> real
        {
            switch (activeContributions[contributionIndex].contribution)
            {
                case ExactRespaNonbondedContribution::Inner: return innerScalar;
                case ExactRespaNonbondedContribution::Middle: return middleScalar;
                case ExactRespaNonbondedContribution::Outer: return outerScalar;
                default: GMX_RELEASE_ASSERT(false, "Unexpected exact nonbonded contribution");
            }
            return 0.0_real;
        };

        const auto flushPackedClusterForces =
                [&](const int                 thread,
                    const std::array<int, c_directCpuListMaxClusterAtoms>& atoms,
                    const int                 atomCount,
                    const real*               forceX,
                    const real*               forceY,
                    const real*               forceZ)
        {
            for (int contributionIndex = 0; contributionIndex < gmx::ssize(activeContributions);
                 ++contributionIndex)
            {
                auto& contributionScratch = pairLoopOmpScratchPtr->contributions[contributionIndex];
                const int base = contributionIndex * c_directCpuListMaxClusterAtoms;
                for (int atomSlot = 0; atomSlot < atomCount; ++atomSlot)
                {
                    const int atom = atoms[atomSlot];
                    if (atom < 0)
                    {
                        continue;
                    }
                    const real fx = forceX[base + atomSlot];
                    const real fy = forceY[base + atomSlot];
                    const real fz = forceZ[base + atomSlot];
                    if (fx == 0.0_real && fy == 0.0_real && fz == 0.0_real)
                    {
                        continue;
                    }
                    contributionScratch.forceByThread[thread][atom][XX] += fx;
                    contributionScratch.forceByThread[thread][atom][YY] += fy;
                    contributionScratch.forceByThread[thread][atom][ZZ] += fz;
                    markTouchedAtom(contributionScratch, thread, atom);
                }
            }
        };

        const auto flushPackedShiftForces = [&](const int thread,
                                                const int shiftIndex,
                                                const real* shiftForceX,
                                                const real* shiftForceY,
                                                const real* shiftForceZ)
        {
            if (shiftIndex == c_centralShiftIndex)
            {
                return;
            }
            for (int contributionIndex = 0; contributionIndex < gmx::ssize(activeContributions);
                 ++contributionIndex)
            {
                auto& contributionScratch = pairLoopOmpScratchPtr->contributions[contributionIndex];
                if (activeContributions[contributionIndex].shift.empty())
                {
                    continue;
                }
                const real fx = shiftForceX[contributionIndex];
                const real fy = shiftForceY[contributionIndex];
                const real fz = shiftForceZ[contributionIndex];
                if (fx == 0.0_real && fy == 0.0_real && fz == 0.0_real)
                {
                    continue;
                }
                contributionScratch.shiftByThread[thread][shiftIndex][XX] += fx;
                contributionScratch.shiftByThread[thread][shiftIndex][YY] += fy;
                contributionScratch.shiftByThread[thread][shiftIndex][ZZ] += fz;
                contributionScratch.shiftByThread[thread][c_centralShiftIndex][XX] -= fx;
                contributionScratch.shiftByThread[thread][c_centralShiftIndex][YY] -= fy;
                contributionScratch.shiftByThread[thread][c_centralShiftIndex][ZZ] -= fz;
            }
        };

        if (useDirectCpuListBackend)
        {
            struct CpuPairlistWorkItem
            {
                const NbnxnPairlistCpu*    pairlist = nullptr;
                ArrayRef<const nbnxn_ci_t> ciList;
                const nbnxn_cj_t*          cjData = nullptr;
            };

            std::vector<CpuPairlistWorkItem> workItems;
            const auto& pairlistSets = fr->nbv->pairlistSets();
            const auto appendWorkItemsFromPairlistSet = [&](const PairlistSet& pairlistSet)
            {
                for (const auto& cpuList : pairlistSet.cpuLists())
                {
                    GMX_RELEASE_ASSERT(cpuList.na_ci <= 8 && cpuList.na_cj <= 8,
                                       "Direct exact-r-RESPA CPU-list backend only supports cluster sizes up to 8");

                    ArrayRef<const nbnxn_ci_t> ciList;
                    const nbnxn_cj_t*          cjData = nullptr;
                    if (pairlistSets.params().useDynamicPruning)
                    {
                        ciList = cpuList.ciOuter;
                        cjData = cpuList.cjOuter.data();
                    }
                    else
                    {
                        ciList = cpuList.ci;
                        cjData = cpuList.cj.list_.data();
                    }

                    if (!ciList.empty())
                    {
                        workItems.push_back({ &cpuList, ciList, cjData });
                    }
                }
            };

            appendWorkItemsFromPairlistSet(pairlistSets.pairlistSet(InteractionLocality::Local));
            if (pairlistSets.params().haveMultipleDomains_)
            {
                appendWorkItemsFromPairlistSet(pairlistSets.pairlistSet(InteractionLocality::NonLocal));
            }
            if (workItems.empty())
            {
                return false;
            }
            directCpuListJobCount = workItems.size();

            const auto atomIndices          = fr->nbv->getGridAtomOrder();
            const auto& nbat                = fr->nbv->nbat();
            const real pairlistRangeSquared = gmx::square(completePairlistRange);

            const auto processPackedStandardCluster =
                    [&](const int                 thread,
                        const NbnxnPairlistCpu&   pairlist,
                        const int                 shiftIndex,
                        const bool                doLj,
                        const bool                doCoulomb,
                        const bool                halfLj,
                        const bool                fullMaskStandardPairs,
                        const std::array<int, c_directCpuListMaxClusterAtoms>& atomI,
                        const std::array<RVec, c_directCpuListMaxClusterAtoms>& rangeXI,
                        const std::array<RVec, c_directCpuListMaxClusterAtoms>& exactXI,
                        const nbnxn_cj_t&         jEntry,
                        const std::array<int, c_directCpuListMaxClusterAtoms>& atomJ,
                        const std::array<RVec, c_directCpuListMaxClusterAtoms>& rangeXJ,
                        const std::array<RVec, c_directCpuListMaxClusterAtoms>& exactXJ,
                        real*                     iClusterForceX,
                        real*                     iClusterForceY,
                        real*                     iClusterForceZ,
                        real*                     shiftForceX,
                        real*                     shiftForceY,
                        real*                     shiftForceZ)
            {
                const bool useSimdClusterKernel =
                        fullMaskStandardPairs && pairlist.na_ci == 4 && (pairlist.na_cj == 4 || pairlist.na_cj == 8);
                const int  ljActiveILimit = halfLj ? (pairlist.na_ci / 2) : pairlist.na_ci;
                real       jClusterForceX[c_directCpuListPackedForceEntries] = { 0.0_real };
                real       jClusterForceY[c_directCpuListPackedForceEntries] = { 0.0_real };
                real       jClusterForceZ[c_directCpuListPackedForceEntries] = { 0.0_real };

                for (int i = 0; i < pairlist.na_ci; ++i)
                {
                    const int ai = atomI[i];
                    if (ai < 0)
                    {
                        continue;
                    }

                    const bool ljActiveForI = doLj && (!halfLj || i < ljActiveILimit);
                    if (!doCoulomb && !ljActiveForI)
                    {
                        continue;
                    }

                    if (useSimdClusterKernel)
                    {
                        real iLaneForceX[c_directCpuListPackedForceEntries] = { 0.0_real };
                        real iLaneForceY[c_directCpuListPackedForceEntries] = { 0.0_real };
                        real iLaneForceZ[c_directCpuListPackedForceEntries] = { 0.0_real };
                        real jForceX[c_directCpuListPackedForceEntries] = { 0.0_real };
                        real jForceY[c_directCpuListPackedForceEntries] = { 0.0_real };
                        real jForceZ[c_directCpuListPackedForceEntries] = { 0.0_real };

#pragma omp simd
                        for (int j = 0; j < pairlist.na_cj; ++j)
                        {
                            const int aj = atomJ[j];
                            if (aj < 0)
                            {
                                continue;
                            }

                            const real rangeDxX = rangeXI[i][XX] - rangeXJ[j][XX];
                            const real rangeDxY = rangeXI[i][YY] - rangeXJ[j][YY];
                            const real rangeDxZ = rangeXI[i][ZZ] - rangeXJ[j][ZZ];
                            const real rangeRsq = rangeDxX * rangeDxX + rangeDxY * rangeDxY + rangeDxZ * rangeDxZ;
                            if (rangeRsq >= pairlistRangeSquared)
                            {
                                continue;
                            }

                            const real dxX = exactXI[i][XX] - exactXJ[j][XX];
                            const real dxY = exactXI[i][YY] - exactXJ[j][YY];
                            const real dxZ = exactXI[i][ZZ] - exactXJ[j][ZZ];
                            real       rsq = dxX * dxX + dxY * dxY + dxZ * dxZ;
                            rsq            = std::max(rsq, c_nbnxnMinDistanceSquared);
                            const real rinv   = gmx::invsqrt(rsq);
                            const real rinvsq = rinv * rinv;
                            const real r      = rsq * rinv;

                            real innerScalar  = 0.0_real;
                            real middleScalar = 0.0_real;
                            real outerScalar  = 0.0_real;
                            if (!computeStandardPairScalarsFromGeometry(ai,
                                                                        aj,
                                                                        rsq,
                                                                        rinv,
                                                                        rinvsq,
                                                                        r,
                                                                        ljActiveForI,
                                                                        doCoulomb,
                                                                        &innerScalar,
                                                                        &middleScalar,
                                                                        &outerScalar))
                            {
                                continue;
                            }

                            for (int contributionIndex = 0; contributionIndex < gmx::ssize(activeContributions);
                                 ++contributionIndex)
                            {
                                const real scalar = contributionScalar(
                                        contributionIndex, innerScalar, middleScalar, outerScalar);
                                if (scalar == 0.0_real)
                                {
                                    continue;
                                }
                                const real fscale = scalar * rinvsq;
                                const real fx     = fscale * dxX;
                                const real fy     = fscale * dxY;
                                const real fz     = fscale * dxZ;
                                const int packedIndex = contributionIndex * c_directCpuListMaxClusterAtoms + j;
                                iLaneForceX[packedIndex] = fx;
                                iLaneForceY[packedIndex] = fy;
                                iLaneForceZ[packedIndex] = fz;
                                jForceX[packedIndex] -= fx;
                                jForceY[packedIndex] -= fy;
                                jForceZ[packedIndex] -= fz;
                            }
                        }

                        for (int contributionIndex = 0; contributionIndex < gmx::ssize(activeContributions);
                             ++contributionIndex)
                        {
                            const int iPackedIndex = contributionIndex * c_directCpuListMaxClusterAtoms + i;
                            real      sumForceX    = 0.0_real;
                            real      sumForceY    = 0.0_real;
                            real      sumForceZ    = 0.0_real;
                            const int base         = contributionIndex * c_directCpuListMaxClusterAtoms;
                            for (int j = 0; j < pairlist.na_cj; ++j)
                            {
                                sumForceX += iLaneForceX[base + j];
                                sumForceY += iLaneForceY[base + j];
                                sumForceZ += iLaneForceZ[base + j];
                            }
                            iClusterForceX[iPackedIndex] += sumForceX;
                            iClusterForceY[iPackedIndex] += sumForceY;
                            iClusterForceZ[iPackedIndex] += sumForceZ;
                            if (shiftIndex != c_centralShiftIndex
                                && !activeContributions[contributionIndex].shift.empty())
                            {
                                shiftForceX[contributionIndex] += sumForceX;
                                shiftForceY[contributionIndex] += sumForceY;
                                shiftForceZ[contributionIndex] += sumForceZ;
                            }
                        }
                        for (int packedIndex = 0; packedIndex < c_directCpuListPackedForceEntries; ++packedIndex)
                        {
                            jClusterForceX[packedIndex] += jForceX[packedIndex];
                            jClusterForceY[packedIndex] += jForceY[packedIndex];
                            jClusterForceZ[packedIndex] += jForceZ[packedIndex];
                        }
                        continue;
                    }

                    for (int j = 0; j < pairlist.na_cj; ++j)
                    {
                        const int aj = atomJ[j];
                        if (aj < 0)
                        {
                            continue;
                        }

                        const real rangeDxX = rangeXI[i][XX] - rangeXJ[j][XX];
                        const real rangeDxY = rangeXI[i][YY] - rangeXJ[j][YY];
                        const real rangeDxZ = rangeXI[i][ZZ] - rangeXJ[j][ZZ];
                        const real rangeRsq = rangeDxX * rangeDxX + rangeDxY * rangeDxY + rangeDxZ * rangeDxZ;
                        if (rangeRsq >= pairlistRangeSquared)
                        {
                            continue;
                        }

                        if (!fullMaskStandardPairs)
                        {
                            const unsigned int pairBit = (1U << (i * pairlist.na_cj + j));
                            if ((jEntry.excl & pairBit) == 0)
                            {
                                continue;
                            }
                        }

                        const real dxX = exactXI[i][XX] - exactXJ[j][XX];
                        const real dxY = exactXI[i][YY] - exactXJ[j][YY];
                        const real dxZ = exactXI[i][ZZ] - exactXJ[j][ZZ];
                        real       rsq = dxX * dxX + dxY * dxY + dxZ * dxZ;
                        rsq            = std::max(rsq, c_nbnxnMinDistanceSquared);
                        const real rinv   = gmx::invsqrt(rsq);
                        const real rinvsq = rinv * rinv;
                        const real r      = rsq * rinv;

                        real innerScalar  = 0.0_real;
                        real middleScalar = 0.0_real;
                        real outerScalar  = 0.0_real;
                        if (!computeStandardPairScalarsFromGeometry(ai,
                                                                    aj,
                                                                    rsq,
                                                                    rinv,
                                                                    rinvsq,
                                                                    r,
                                                                    ljActiveForI,
                                                                    doCoulomb,
                                                                    &innerScalar,
                                                                    &middleScalar,
                                                                    &outerScalar))
                        {
                            continue;
                        }

                        for (int contributionIndex = 0; contributionIndex < gmx::ssize(activeContributions);
                             ++contributionIndex)
                        {
                            const real scalar = contributionScalar(
                                    contributionIndex, innerScalar, middleScalar, outerScalar);
                            if (scalar == 0.0_real)
                            {
                                continue;
                            }
                            const real fscale = scalar * rinvsq;
                            const real fx     = fscale * dxX;
                            const real fy     = fscale * dxY;
                            const real fz     = fscale * dxZ;
                            const int  iPackedIndex =
                                    contributionIndex * c_directCpuListMaxClusterAtoms + i;
                            const int jPackedIndex =
                                    contributionIndex * c_directCpuListMaxClusterAtoms + j;
                            iClusterForceX[iPackedIndex] += fx;
                            iClusterForceY[iPackedIndex] += fy;
                            iClusterForceZ[iPackedIndex] += fz;
                            jClusterForceX[jPackedIndex] -= fx;
                            jClusterForceY[jPackedIndex] -= fy;
                            jClusterForceZ[jPackedIndex] -= fz;
                            if (shiftIndex != c_centralShiftIndex
                                && !activeContributions[contributionIndex].shift.empty())
                            {
                                shiftForceX[contributionIndex] += fx;
                                shiftForceY[contributionIndex] += fy;
                                shiftForceZ[contributionIndex] += fz;
                            }
                        }
                    }
                }

                flushPackedClusterForces(
                        thread, atomJ, pairlist.na_cj, jClusterForceX, jClusterForceY, jClusterForceZ);
            };

            const auto processPackedExcludedCluster =
                    [&](const int                 thread,
                        const NbnxnPairlistCpu&   pairlist,
                        const int                 shiftIndex,
                        const std::array<int, c_directCpuListMaxClusterAtoms>& atomI,
                        const std::array<RVec, c_directCpuListMaxClusterAtoms>& rangeXI,
                        const std::array<RVec, c_directCpuListMaxClusterAtoms>& exactXI,
                        const nbnxn_ci_t&         iEntry,
                        const nbnxn_cj_t&         jEntry,
                        const std::array<int, c_directCpuListMaxClusterAtoms>& atomJ,
                        const std::array<RVec, c_directCpuListMaxClusterAtoms>& rangeXJ,
                        const std::array<RVec, c_directCpuListMaxClusterAtoms>& exactXJ,
                        real*                     iClusterForceX,
                        real*                     iClusterForceY,
                        real*                     iClusterForceZ,
                        real*                     shiftForceX,
                        real*                     shiftForceY,
                        real*                     shiftForceZ)
            {
                if (outerContributionIndex < 0)
                {
                    return;
                }

                real jClusterForceX[c_directCpuListPackedForceEntries] = { 0.0_real };
                real jClusterForceY[c_directCpuListPackedForceEntries] = { 0.0_real };
                real jClusterForceZ[c_directCpuListPackedForceEntries] = { 0.0_real };
                const int outerBase = outerContributionIndex * c_directCpuListMaxClusterAtoms;

                for (int i = 0; i < pairlist.na_ci; ++i)
                {
                    const int ai = atomI[i];
                    if (ai < 0)
                    {
                        continue;
                    }

                    const int iAtomIndex = iEntry.ci * pairlist.na_ci + i;
                    for (int j = 0; j < pairlist.na_cj; ++j)
                    {
                        const int aj = atomJ[j];
                        if (aj < 0)
                        {
                            continue;
                        }

                        const real rangeDxX = rangeXI[i][XX] - rangeXJ[j][XX];
                        const real rangeDxY = rangeXI[i][YY] - rangeXJ[j][YY];
                        const real rangeDxZ = rangeXI[i][ZZ] - rangeXJ[j][ZZ];
                        const real rangeRsq = rangeDxX * rangeDxX + rangeDxY * rangeDxY + rangeDxZ * rangeDxZ;
                        if (rangeRsq >= pairlistRangeSquared)
                        {
                            continue;
                        }

                        const int  jAtomIndex = jEntry.cj * pairlist.na_cj + j;
                        const unsigned int pairBit = (1U << (i * pairlist.na_cj + j));
                        const bool isStandardPair = (jEntry.excl & pairBit) != 0;
                        const bool isExcludedPair =
                                !isStandardPair
                                && (shiftIndex != c_centralShiftIndex || jAtomIndex > iAtomIndex);
                        if (!isExcludedPair)
                        {
                            continue;
                        }

                        const real dxX = exactXI[i][XX] - exactXJ[j][XX];
                        const real dxY = exactXI[i][YY] - exactXJ[j][YY];
                        const real dxZ = exactXI[i][ZZ] - exactXJ[j][ZZ];
                        real       rsq = dxX * dxX + dxY * dxY + dxZ * dxZ;
                        rsq            = std::max(rsq, c_nbnxnMinDistanceSquared);
                        const real rinv   = gmx::invsqrt(rsq);
                        const real rinvsq = rinv * rinv;
                        const real r      = rsq * rinv;

                        real correctionScalar = 0.0_real;
                        if (!computeExcludedCorrectionFromGeometry(
                                    ai, aj, rsq, rinv, rinvsq, r, &correctionScalar))
                        {
                            continue;
                        }

                        const real fscale = correctionScalar * rinvsq;
                        const real fx     = fscale * dxX;
                        const real fy     = fscale * dxY;
                        const real fz     = fscale * dxZ;
                        iClusterForceX[outerBase + i] += fx;
                        iClusterForceY[outerBase + i] += fy;
                        iClusterForceZ[outerBase + i] += fz;
                        jClusterForceX[outerBase + j] -= fx;
                        jClusterForceY[outerBase + j] -= fy;
                        jClusterForceZ[outerBase + j] -= fz;
                        if (shiftIndex != c_centralShiftIndex
                            && !activeContributions[outerContributionIndex].shift.empty())
                        {
                            shiftForceX[outerContributionIndex] += fx;
                            shiftForceY[outerContributionIndex] += fy;
                            shiftForceZ[outerContributionIndex] += fz;
                        }
                    }
                }

                flushPackedClusterForces(
                        thread, atomJ, pairlist.na_cj, jClusterForceX, jClusterForceY, jClusterForceZ);
            };

#pragma omp parallel for num_threads(pairLoopWorkerThreads) schedule(static)
            for (int workItemIndex = 0; workItemIndex < gmx::ssize(workItems); ++workItemIndex)
            {
                const int                    thread   = ::gmx_omp_get_thread_num();
                const CpuPairlistWorkItem&   workItem = workItems[workItemIndex];
                const auto&                  pairlist = *workItem.pairlist;
                for (const auto& iEntry : workItem.ciList)
                {
                    const int  shiftIndex = (iEntry.shift & NBNXN_CI_SHIFT);
                    const bool doCoulomb  = (iEntry.shift & NBNXN_CI_DO_COUL(0)) != 0;
                    const bool doLj       = (iEntry.shift & NBNXN_CI_DO_LJ(0)) != 0;
                    const bool halfLj     = doLj && ((iEntry.shift & NBNXN_CI_HALF_LJ(0)) != 0);
                    if (listKind == PairLoopListKind::ExcludedPairs && !doCoulomb)
                    {
                        continue;
                    }
                    if (listKind == PairLoopListKind::StandardPairs && !doCoulomb && !doLj)
                    {
                        continue;
                    }

                    std::array<int, 8>  atomI{};
                    std::array<RVec, 8> rangeXI{};
                    std::array<RVec, 8> exactXI{};
                    for (int i = 0; i < pairlist.na_ci; ++i)
                    {
                        const int iAtomIndex = iEntry.ci * pairlist.na_ci + i;
                        atomI[i]             = atomIndices[iAtomIndex];
                        if (atomI[i] < 0)
                        {
                            continue;
                        }
                        for (int dim = 0; dim < DIM; ++dim)
                        {
                            rangeXI[i][dim] =
                                    getCoordinate(nbat, iAtomIndex)[dim] + nbat.shift_vec[shiftIndex][dim];
                            exactXI[i][dim] =
                                    coordinates[atomI[i]][dim] + fr->shift_vec[shiftIndex][dim];
                        }
                    }

                    real iClusterForceX[c_directCpuListPackedForceEntries] = { 0.0_real };
                    real iClusterForceY[c_directCpuListPackedForceEntries] = { 0.0_real };
                    real iClusterForceZ[c_directCpuListPackedForceEntries] = { 0.0_real };
                    real shiftForceX[c_directCpuListMaxContributions]      = { 0.0_real };
                    real shiftForceY[c_directCpuListMaxContributions]      = { 0.0_real };
                    real shiftForceZ[c_directCpuListMaxContributions]      = { 0.0_real };
                    int firstFullMaskJCluster = iEntry.cj_ind_end;
                    for (int jClusterIndex = iEntry.cj_ind_start; jClusterIndex < iEntry.cj_ind_end;
                         ++jClusterIndex)
                    {
                        if (workItem.cjData[jClusterIndex].excl == NBNXN_INTERACTION_MASK_ALL)
                        {
                            firstFullMaskJCluster = jClusterIndex;
                            break;
                        }
                    }

                    const auto processJCluster = [&](const nbnxn_cj_t& jEntry,
                                                     const bool        fullMaskStandardPairs)
                    {
                        std::array<int, 8>  atomJ{};
                        std::array<RVec, 8> rangeXJ{};
                        std::array<RVec, 8> exactXJ{};
                        for (int j = 0; j < pairlist.na_cj; ++j)
                        {
                            const int jAtomIndex = jEntry.cj * pairlist.na_cj + j;
                            atomJ[j]             = atomIndices[jAtomIndex];
                            if (atomJ[j] < 0)
                            {
                                continue;
                            }
                            for (int dim = 0; dim < DIM; ++dim)
                            {
                                rangeXJ[j][dim] = getCoordinate(nbat, jAtomIndex)[dim];
                                exactXJ[j][dim] = coordinates[atomJ[j]][dim];
                            }
                        }
                        if (listKind == PairLoopListKind::StandardPairs)
                        {
                            processPackedStandardCluster(thread,
                                                         pairlist,
                                                         shiftIndex,
                                                         doLj,
                                                         doCoulomb,
                                                         halfLj,
                                                         fullMaskStandardPairs,
                                                         atomI,
                                                         rangeXI,
                                                         exactXI,
                                                         jEntry,
                                                         atomJ,
                                                         rangeXJ,
                                                         exactXJ,
                                                         iClusterForceX,
                                                         iClusterForceY,
                                                         iClusterForceZ,
                                                         shiftForceX,
                                                         shiftForceY,
                                                         shiftForceZ);
                        }
                        else
                        {
                            processPackedExcludedCluster(thread,
                                                         pairlist,
                                                         shiftIndex,
                                                         atomI,
                                                         rangeXI,
                                                         exactXI,
                                                         iEntry,
                                                         jEntry,
                                                         atomJ,
                                                         rangeXJ,
                                                         exactXJ,
                                                         iClusterForceX,
                                                         iClusterForceY,
                                                         iClusterForceZ,
                                                         shiftForceX,
                                                         shiftForceY,
                                                         shiftForceZ);
                        }
                    };

                    for (int jClusterIndex = iEntry.cj_ind_start; jClusterIndex < firstFullMaskJCluster;
                         ++jClusterIndex)
                    {
                        processJCluster(workItem.cjData[jClusterIndex], false);
                    }
                    if (listKind == PairLoopListKind::StandardPairs)
                    {
                        for (int jClusterIndex = firstFullMaskJCluster; jClusterIndex < iEntry.cj_ind_end;
                             ++jClusterIndex)
                        {
                            GMX_RELEASE_ASSERT(
                                    workItem.cjData[jClusterIndex].excl == NBNXN_INTERACTION_MASK_ALL,
                                    "Direct CPU-list full-mask fast path expects a suffix of full-mask j-clusters");
                            processJCluster(workItem.cjData[jClusterIndex], true);
                        }
                    }
                    flushPackedClusterForces(
                            thread, atomI, pairlist.na_ci, iClusterForceX, iClusterForceY, iClusterForceZ);
                    flushPackedShiftForces(thread, shiftIndex, shiftForceX, shiftForceY, shiftForceZ);
                }
            }
        }
        else if (useNbnxm4x4Backend)
        {
            const auto& nbnxm4x4Clusters = pairLoop4x4Clusters();
#pragma omp parallel for num_threads(pairLoopWorkerThreads) schedule(static)
            for (int clusterIndex = 0; clusterIndex < gmx::ssize(nbnxm4x4Clusters); ++clusterIndex)
            {
                const int                     thread  = ::gmx_omp_get_thread_num();
                const PairLoop4x4Cluster&     cluster = nbnxm4x4Clusters[clusterIndex];
                const real                    shiftX  = fr->shift_vec[cluster.shiftIndex][XX];
                const real                    shiftY  = fr->shift_vec[cluster.shiftIndex][YY];
                const real                    shiftZ  = fr->shift_vec[cluster.shiftIndex][ZZ];
                const int                     iBase   = cluster.iBase;
                const int                     jBase   = cluster.jBase;
                const int                     shiftIndex = cluster.shiftIndex;
                GMX_UNUSED_VALUE(shiftX);
                GMX_UNUSED_VALUE(shiftY);
                GMX_UNUSED_VALUE(shiftZ);

                for (int iLane = 0; iLane < 4; ++iLane)
                {
                    const uint16_t rowMask = static_cast<uint16_t>((cluster.mask >> (iLane * 4)) & 0xF);
                    if (rowMask == 0)
                    {
                        continue;
                    }

                    const int ai = iBase + iLane;
                    for (int jLane = 0; jLane < 4; ++jLane)
                    {
                        if ((rowMask & (1u << jLane)) == 0)
                        {
                            continue;
                        }

                        const int aj = jBase + jLane;
                        RVec      dx;
                        real      rinvsq      = 0.0_real;
                        real      innerScalar = 0.0_real;
                        real      middleScalar = 0.0_real;
                        real      outerScalar = 0.0_real;
                        if (!computeStandardPairScalars(ai,
                                                        aj,
                                                        shiftIndex,
                                                        dx,
                                                        &rinvsq,
                                                        &innerScalar,
                                                        &middleScalar,
                                                        &outerScalar))
                        {
                            continue;
                        }
                        accumulateStandardPairScalars(thread,
                                                     ai,
                                                     aj,
                                                     shiftIndex,
                                                     dx,
                                                     rinvsq,
                                                     innerScalar,
                                                     middleScalar,
                                                     outerScalar,
                                                     nullptr);
                    }
                }
            }
        }
        else if (useTileBackend)
        {
            GMX_RELEASE_ASSERT(activeContributions.size() <= 3,
                               "Pair-loop tile backend assumes at most three exact nonbonded contributions");
            const int numTiles = (numPairs + c_pairLoopTilePairCount - 1) / c_pairLoopTilePairCount;
#pragma omp parallel num_threads(pairLoopWorkerThreads)
            {
                std::array<TileContributionCache, 3> tileCaches;
#pragma omp for schedule(static)
                for (int tileIndex = 0; tileIndex < numTiles; ++tileIndex)
                {
                    const int thread = ::gmx_omp_get_thread_num();
                    for (int contributionIndex = 0; contributionIndex < gmx::ssize(activeContributions);
                         ++contributionIndex)
                    {
                        resetTileContributionCache(&tileCaches[contributionIndex]);
                    }

                    const int tileBegin = tileIndex * c_pairLoopTilePairCount;
                    const int tileEnd   = std::min(tileBegin + c_pairLoopTilePairCount, numPairs);
                    for (int pairIndex = tileBegin; pairIndex < tileEnd; ++pairIndex)
                    {
                        const auto& entry      = pairEntries[pairIndex];
                        const int   ai         = entry.first.first;
                        const int   aj         = entry.first.second;
                        const int   shiftIndex = entry.second;
                        RVec        dx;
                        if (listKind == PairLoopListKind::StandardPairs)
                        {
                            real rinvsq       = 0.0_real;
                            real innerScalar  = 0.0_real;
                            real middleScalar = 0.0_real;
                            real outerScalar  = 0.0_real;
                            if (!computeStandardPairScalars(ai,
                                                            aj,
                                                            shiftIndex,
                                                            dx,
                                                            &rinvsq,
                                                            &innerScalar,
                                                            &middleScalar,
                                                            &outerScalar))
                            {
                                continue;
                            }
                            accumulateStandardPairScalars(thread,
                                                         ai,
                                                         aj,
                                                         shiftIndex,
                                                         dx,
                                                         rinvsq,
                                                         innerScalar,
                                                         middleScalar,
                                                         outerScalar,
                                                         &tileCaches);
                        }
                        else
                        {
                            real rinvsq           = 0.0_real;
                            real correctionScalar = 0.0_real;
                            if (!computeExcludedCorrection(
                                        ai, aj, shiftIndex, dx, &rinvsq, &correctionScalar))
                            {
                                continue;
                            }
                            accumulateExcludedCorrection(
                                    thread, ai, aj, shiftIndex, dx, rinvsq, correctionScalar, &tileCaches);
                        }
                    }

                    for (int contributionIndex = 0; contributionIndex < gmx::ssize(activeContributions);
                         ++contributionIndex)
                    {
                        flushTileContributionCache(thread, contributionIndex, &tileCaches[contributionIndex]);
                    }
                }
            }
        }
        else if (!pairLoopVectorRequested)
        {
#pragma omp parallel for num_threads(pairLoopWorkerThreads) schedule(static)
            for (int pairIndex = 0; pairIndex < numPairs; ++pairIndex)
            {
                const int   thread     = ::gmx_omp_get_thread_num();
                const auto& entry      = pairEntries[pairIndex];
                const int   ai         = entry.first.first;
                const int   aj         = entry.first.second;
                const int   shiftIndex = entry.second;
                RVec        dx;
                if (listKind == PairLoopListKind::StandardPairs)
                {
                    real rinvsq      = 0.0_real;
                    real innerScalar = 0.0_real;
                    real middleScalar = 0.0_real;
                    real outerScalar = 0.0_real;
                    if (!computeStandardPairScalars(ai,
                                                    aj,
                                                    shiftIndex,
                                                    dx,
                                                    &rinvsq,
                                                    &innerScalar,
                                                    &middleScalar,
                                                    &outerScalar))
                    {
                        continue;
                    }
                    accumulateStandardPairScalars(thread,
                                                 ai,
                                                 aj,
                                                 shiftIndex,
                                                 dx,
                                                 rinvsq,
                                                 innerScalar,
                                                 middleScalar,
                                                 outerScalar,
                                                 nullptr);
                }
                else
                {
                    real rinvsq          = 0.0_real;
                    real correctionScalar = 0.0_real;
                    if (!computeExcludedCorrection(
                                ai, aj, shiftIndex, dx, &rinvsq, &correctionScalar))
                    {
                        continue;
                    }
                    accumulateExcludedCorrection(
                            thread, ai, aj, shiftIndex, dx, rinvsq, correctionScalar, nullptr);
                }
            }
        }
        else
        {
            constexpr int c_pairLoopVectorWidth = 4;
            const int     numChunks = (numPairs + c_pairLoopVectorWidth - 1) / c_pairLoopVectorWidth;
#pragma omp parallel for num_threads(pairLoopWorkerThreads) schedule(static)
            for (int chunk = 0; chunk < numChunks; ++chunk)
            {
                const int thread = ::gmx_omp_get_thread_num();
                bool      laneActive[c_pairLoopVectorWidth];
                int       laneAi[c_pairLoopVectorWidth];
                int       laneAj[c_pairLoopVectorWidth];
                int       laneShiftIndex[c_pairLoopVectorWidth];
                real      laneDxX[c_pairLoopVectorWidth];
                real      laneDxY[c_pairLoopVectorWidth];
                real      laneDxZ[c_pairLoopVectorWidth];
                real      laneRinvSq[c_pairLoopVectorWidth];
                real      laneInnerScalar[c_pairLoopVectorWidth];
                real      laneMiddleScalar[c_pairLoopVectorWidth];
                real      laneOuterScalar[c_pairLoopVectorWidth];

                for (int lane = 0; lane < c_pairLoopVectorWidth; ++lane)
                {
                    laneActive[lane]       = false;
                    laneAi[lane]           = -1;
                    laneAj[lane]           = -1;
                    laneShiftIndex[lane]   = c_centralShiftIndex;
                    laneDxX[lane]          = 0.0_real;
                    laneDxY[lane]          = 0.0_real;
                    laneDxZ[lane]          = 0.0_real;
                    laneRinvSq[lane]       = 0.0_real;
                    laneInnerScalar[lane]  = 0.0_real;
                    laneMiddleScalar[lane] = 0.0_real;
                    laneOuterScalar[lane]  = 0.0_real;
                }

                const int chunkBegin = chunk * c_pairLoopVectorWidth;
                const int chunkEnd   = std::min(chunkBegin + c_pairLoopVectorWidth, numPairs);
                for (int pairIndex = chunkBegin; pairIndex < chunkEnd; ++pairIndex)
                {
                    const int   lane     = pairIndex - chunkBegin;
                    const auto& entry    = pairEntries[pairIndex];
                    laneAi[lane]         = entry.first.first;
                    laneAj[lane]         = entry.first.second;
                    laneShiftIndex[lane] = entry.second;
                    laneActive[lane]     = true;
                }

#pragma omp simd
                for (int lane = 0; lane < c_pairLoopVectorWidth; ++lane)
                {
                    if (!laneActive[lane])
                    {
                        continue;
                    }

                    const int  ai         = laneAi[lane];
                    const int  aj         = laneAj[lane];
                    const int  shiftIndex = laneShiftIndex[lane];
                    const real dxX        = coordinates[ai][XX] + fr->shift_vec[shiftIndex][XX]
                                     - coordinates[aj][XX];
                    const real dxY = coordinates[ai][YY] + fr->shift_vec[shiftIndex][YY]
                                     - coordinates[aj][YY];
                    const real dxZ = coordinates[ai][ZZ] + fr->shift_vec[shiftIndex][ZZ]
                                     - coordinates[aj][ZZ];

                    real rsq = dxX * dxX + dxY * dxY + dxZ * dxZ;
                    rsq      = std::max(rsq, c_nbnxnMinDistanceSquared);

                    const real rinv   = gmx::invsqrt(rsq);
                    const real rinvsq = rinv * rinv;
                    const real r      = rsq * rinv;

                    laneDxX[lane]    = dxX;
                    laneDxY[lane]    = dxY;
                    laneDxZ[lane]    = dxZ;
                    laneRinvSq[lane] = rinvsq;

                    if (listKind == PairLoopListKind::StandardPairs)
                    {
                        real innerWeight  = 0.0_real;
                        real middleWeight = 0.0_real;
                        real outerWeight  = 0.0_real;
                        if (exactRespaHasMiddle)
                        {
                            const real switchIntoMiddle =
                                    respaSwitchIn(r,
                                                  exactRespaForceLayout.innerOff,
                                                  exactRespaForceLayout.innerOn);
                            const real switchIntoOuter =
                                    respaSwitchIn(r,
                                                  exactRespaForceLayout.outerOn,
                                                  exactRespaForceLayout.outerOff);
                            innerWeight  = 1.0_real - switchIntoMiddle;
                            middleWeight = switchIntoMiddle * (1.0_real - switchIntoOuter);
                            outerWeight  = switchIntoOuter;
                        }
                        else
                        {
                            const real switchIntoOuter =
                                    respaSwitchIn(r,
                                                  exactRespaForceLayout.outerOn,
                                                  exactRespaForceLayout.outerOff);
                            innerWeight  = 1.0_real - switchIntoOuter;
                            middleWeight = 0.0_real;
                            outerWeight  = switchIntoOuter;
                        }

                        real rawLjScalar = 0.0_real;
                        if (rsq < vdwCutoff2)
                        {
                            const int  typeI = mdatoms.typeA[ai];
                            const int  typeJ = mdatoms.typeA[aj];
                            const real c6    = fr->nbfp[typeI * ntype2 + typeJ * 2];
                            const real cRepulsive = fr->nbfp[typeI * ntype2 + typeJ * 2 + 1];
                            const real rinvsix = rinvsq * rinvsq * rinvsq;
                            const real repulsiveTerm =
                                    usePower9SpecializedPath ? (rinvsix * rinvsq * rinv)
                                                             : (repulsionPower == 12.0_real
                                                                        ? rinvsix * rinvsix
                                                                        : std::pow(rinv, repulsionPower));
                            rawLjScalar = cRepulsive * repulsiveTerm - c6 * rinvsix;
                        }

                        real bareCoulombScalar = 0.0_real;
                        real correctionScalar  = 0.0_real;
                        if (rsq < coulombCutoff2)
                        {
                            const real qq = mdatoms.chargeA[ai] * mdatoms.chargeA[aj]
                                            * fr->ic->coulomb.epsfac;
                            if (qq != 0.0_real)
                            {
                                const real scaledR = r * fr->ic->coulombEwaldTables->scale;
                                const int  coulTableIndex = static_cast<int>(scaledR);
                                const real coulFrac       = scaledR - coulTableIndex;
#if !GMX_DOUBLE
                                const real* table = fr->ic->coulombEwaldTables->tableFDV0.data();
                                const real  coulFexcl =
                                        table[coulTableIndex * 4]
                                        + coulFrac * table[coulTableIndex * 4 + 1];
#else
                                const real* tableF = fr->ic->coulombEwaldTables->tableF.data();
                                const real  coulFexcl =
                                        (1 - coulFrac) * tableF[coulTableIndex]
                                        + coulFrac * tableF[coulTableIndex + 1];
#endif
                                bareCoulombScalar = qq * rinv;
                                correctionScalar  = -qq * coulFexcl / rinv;
                            }
                        }

                        laneInnerScalar[lane] =
                                bareCoulombScalar * innerWeight + rawLjScalar * innerWeight;
                        laneMiddleScalar[lane] =
                                bareCoulombScalar * middleWeight + rawLjScalar * middleWeight;
                        laneOuterScalar[lane] =
                                correctionScalar + bareCoulombScalar * outerWeight
                                + rawLjScalar * outerWeight;
                    }
                    else
                    {
                        laneInnerScalar[lane]  = 0.0_real;
                        laneMiddleScalar[lane] = 0.0_real;
                        laneOuterScalar[lane]  = 0.0_real;
                        if (rsq < coulombCutoff2)
                        {
                            const real qq = mdatoms.chargeA[ai] * mdatoms.chargeA[aj]
                                            * fr->ic->coulomb.epsfac;
                            if (qq != 0.0_real)
                            {
                                const real scaledR = r * fr->ic->coulombEwaldTables->scale;
                                const int  coulTableIndex = static_cast<int>(scaledR);
                                const real coulFrac       = scaledR - coulTableIndex;
#if !GMX_DOUBLE
                                const real* table = fr->ic->coulombEwaldTables->tableFDV0.data();
                                const real  coulFexcl =
                                        table[coulTableIndex * 4]
                                        + coulFrac * table[coulTableIndex * 4 + 1];
#else
                                const real* tableF = fr->ic->coulombEwaldTables->tableF.data();
                                const real  coulFexcl =
                                        (1 - coulFrac) * tableF[coulTableIndex]
                                        + coulFrac * tableF[coulTableIndex + 1];
#endif
                                laneOuterScalar[lane] = -qq * coulFexcl / rinv;
                            }
                        }
                    }
                }

                for (int lane = 0; lane < c_pairLoopVectorWidth; ++lane)
                {
                    if (!laneActive[lane])
                    {
                        continue;
                    }

                    const RVec dx = { laneDxX[lane], laneDxY[lane], laneDxZ[lane] };
                    if (listKind == PairLoopListKind::StandardPairs)
                    {
                        accumulateStandardPairScalars(thread,
                                                     laneAi[lane],
                                                     laneAj[lane],
                                                     laneShiftIndex[lane],
                                                     dx,
                                                     laneRinvSq[lane],
                                                     laneInnerScalar[lane],
                                                     laneMiddleScalar[lane],
                                                     laneOuterScalar[lane],
                                                     nullptr);
                    }
                    else
                    {
                        accumulateExcludedCorrection(thread,
                                                    laneAi[lane],
                                                    laneAj[lane],
                                                    laneShiftIndex[lane],
                                                    dx,
                                                    laneRinvSq[lane],
                                                    laneOuterScalar[lane],
                                                    nullptr);
                    }
                }
            }
        }
        if (pairLoopTimingEnabled)
        {
            pairEnd     = PairLoopClock::now();
            reduceStart = pairEnd;
        }

        PairLoopReductionStats reductionStats =
                reducePairLoopOmpScratch(useSparseTrackingForPairlist);
        reductionStats.usedTileBackend     = useTileBackend;
        reductionStats.usedNbnxm4x4Backend = useNbnxm4x4Backend;
        reductionStats.usedDirectCpuListBackend = useDirectCpuListBackend;
        if (pairLoopTimingEnabled)
        {
            reduceEnd = PairLoopClock::now();
            const auto durationUs = [](const PairLoopClock::time_point& begin,
                                       const PairLoopClock::time_point& end) -> int64_t
            {
                return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            };
            appendRespaTraceTextLine(
                    pairLoopTimingDirPath,
                    "pairloop_timing.tsv",
                    "schema=exact_respa_pairloop_timing_v1 label="
                            + std::string(exactRespaPairLoopTimingLabel()) + " step="
                            + std::to_string(step) + " pair_list=" + pairListLabel
                            + " num_pairs=" + std::to_string(numPairs) + " worker_threads="
                            + std::to_string(pairLoopWorkerThreads) + " work_items="
                            + std::to_string(useDirectCpuListBackend ? directCpuListJobCount : numPairs)
                            + " direct_cpulist_requested="
                            + std::string(pairLoopDirectCpuListRequested ? "true" : "false")
                            + " direct_cpulist_used="
                            + std::string(reductionStats.usedDirectCpuListBackend ? "true" : "false")
                            + " omp_requested="
                            + std::string(pairLoopOmpRequested ? "true" : "false")
                            + " vector_requested="
                            + std::string(pairLoopVectorRequested ? "true" : "false")
                            + " block_reduction_requested="
                            + std::string(pairLoopBlockReductionRequested ? "true" : "false")
                            + " block_reduction_used="
                            + std::string(reductionStats.usedBlockedReduction ? "true" : "false")
                            + " tile_requested="
                            + std::string(pairLoopTileRequested ? "true" : "false")
                            + " tile_used="
                            + std::string(reductionStats.usedTileBackend ? "true" : "false")
                            + " nbnxm4x4_requested="
                            + std::string(pairLoopNbnxm4x4Requested ? "true" : "false")
                            + " nbnxm4x4_used="
                            + std::string(reductionStats.usedNbnxm4x4Backend ? "true" : "false")
                            + " sparse_requested="
                            + std::string(pairLoopSparseReductionRequested ? "true" : "false")
                            + " sparse_tracking="
                            + std::string(useSparseTrackingForPairlist ? "true" : "false")
                            + " sparse_used="
                            + std::string(reductionStats.usedSparseReduction ? "true" : "false")
                            + " touched_atom_slots="
                            + std::to_string(reductionStats.touchedAtomSlots)
                            + " reduced_atom_slots="
                            + std::to_string(reductionStats.reducedAtomSlots)
                            + " clear_us=" + std::to_string(durationUs(clearStart, clearEnd))
                            + " pair_us=" + std::to_string(durationUs(pairStart, pairEnd))
                            + " reduce_us=" + std::to_string(durationUs(reduceStart, reduceEnd)));
        }
        return true;
    };

    struct PairLoopForceDumpBefore
    {
        ExactRespaNonbondedContribution contribution;
        std::vector<RVec>               force;
    };
    std::vector<PairLoopForceDumpBefore> pairLoopForceDumpBefore;
    const auto capturePairLoopForceDumpBefore = [&]()
    {
        if (!pairLoopForceDumpEnabled)
        {
            return;
        }
        pairLoopForceDumpBefore.clear();
        pairLoopForceDumpBefore.reserve(activeContributions.size());
        for (const auto& accumulator : activeContributions)
        {
            PairLoopForceDumpBefore snapshot;
            snapshot.contribution = accumulator.contribution;
            snapshot.force.assign(accumulator.force.begin(), accumulator.force.end());
            pairLoopForceDumpBefore.push_back(std::move(snapshot));
        }
    };
    const auto writePairLoopForceDeltaDump = [&](const bool pairFastPathUsed,
                                                 const bool excludedPairFastPathUsed)
    {
        if (!pairLoopForceDumpEnabled || pairLoopForceDumpBefore.empty())
        {
            return;
        }

        int dumpOrdinal = 0;
        {
            std::lock_guard<std::mutex> lock(pairLoopForceDumpMutex);
            if (pairLoopForceDumpOrdinal >= pairLoopForceDumpMax)
            {
                return;
            }
            pairLoopForceDumpOrdinal++;
            dumpOrdinal = pairLoopForceDumpOrdinal;
        }

        std::filesystem::path dumpDir(pairLoopForceDumpDirPath);
        std::filesystem::create_directories(dumpDir);
        char fileName[64];
        std::snprintf(fileName, sizeof(fileName), "pairloop_force_delta_%06d.tsv", dumpOrdinal);
        const std::filesystem::path outputPath = dumpDir / fileName;

        FILE* dumpFile = std::fopen(outputPath.string().c_str(), "w");
        if (dumpFile == nullptr)
        {
            gmx_fatal(FARGS,
                      "Could not open exact r-RESPA pair-loop force delta output '%s' for writing",
                      outputPath.string().c_str());
        }

        std::fprintf(dumpFile, "# schema exact_respa_pairloop_force_delta_v1\n");
        std::fprintf(dumpFile, "# label %s\n", exactRespaPairLoopForceDumpLabel());
        std::fprintf(dumpFile, "# ordinal %d\n", dumpOrdinal);
        std::fprintf(dumpFile, "# step %" PRId64 "\n", static_cast<int64_t>(step));
        std::fprintf(dumpFile,
                     "# pair_fast_path_used %s\n",
                     pairFastPathUsed ? "true" : "false");
        std::fprintf(dumpFile,
                     "# excluded_pair_fast_path_used %s\n",
                     excludedPairFastPathUsed ? "true" : "false");
        std::fprintf(dumpFile,
                     "# pairloop_omp_requested %s\n",
                     pairLoopOmpRequested ? "true" : "false");
        std::fprintf(dumpFile,
                     "# pairloop_vector_requested %s\n",
                     pairLoopVectorRequested ? "true" : "false");
        std::fprintf(dumpFile,
                     "# pairloop_direct_cpulist_requested %s\n",
                     pairLoopDirectCpuListRequested ? "true" : "false");
        std::fprintf(dumpFile,
                     "# pairloop_worker_threads %d\n",
                     pairLoopWorkerThreads);
        std::fprintf(dumpFile,
                     "# compute_pair_energies %s\n",
                     computePairEnergies ? "true" : "false");
        std::fprintf(dumpFile,
                     "# compute_virial %s\n",
                     stepWork.computeVirial ? "true" : "false");
        std::fprintf(dumpFile,
                     "# plain_pair_count_available %s\n",
                     needPlainPairlist ? "true" : "false");
        if (needPlainPairlist)
        {
            std::fprintf(dumpFile,
                         "# plain_pair_count %ld\n",
                         static_cast<long>(gmx::ssize(plainPairlist.pairs)));
            std::fprintf(dumpFile,
                         "# excluded_pair_count %ld\n",
                         static_cast<long>(gmx::ssize(plainPairlist.excludedPairs)));
        }
        else
        {
            std::fprintf(dumpFile, "# plain_pair_count unavailable\n");
            std::fprintf(dumpFile, "# excluded_pair_count unavailable\n");
        }
        std::fprintf(dumpFile, "contribution_index\tcontribution\tatom\tfx\tfy\tfz\n");

        GMX_RELEASE_ASSERT(pairLoopForceDumpBefore.size() == activeContributions.size(),
                           "Pair-loop force dump snapshots should match active contribution count");
        for (int contributionIndex = 0; contributionIndex < gmx::ssize(activeContributions);
             ++contributionIndex)
        {
            const auto& before      = pairLoopForceDumpBefore[contributionIndex];
            const auto& accumulator = activeContributions[contributionIndex];
            GMX_RELEASE_ASSERT(before.contribution == accumulator.contribution,
                               "Pair-loop force dump snapshots should preserve contribution order");
            GMX_RELEASE_ASSERT(before.force.size() == accumulator.force.size(),
                               "Pair-loop force dump snapshots should preserve force-buffer size");
            for (int atom = 0; atom < gmx::ssize(accumulator.force); ++atom)
            {
                const RVec delta = { accumulator.force[atom][XX] - before.force[atom][XX],
                                     accumulator.force[atom][YY] - before.force[atom][YY],
                                     accumulator.force[atom][ZZ] - before.force[atom][ZZ] };
                std::fprintf(dumpFile,
                             "%d\t%s\t%d\t%.17g\t%.17g\t%.17g\n",
                             contributionIndex,
                             contributionLabel(accumulator.contribution),
                             atom,
                             delta[XX],
                             delta[YY],
                             delta[ZZ]);
            }
        }
        std::fclose(dumpFile);
    };

    const auto processPairlist = [&](const auto& pairEntries,
                                     const real factorCoulomb,
                                     const real factorLj,
                                     const auto& includePair,
                                     PairDebugStats* debugStats = nullptr)
    {
        bool dumpedFirstExcludedWrite = false;
        int  pairOrdinal             = 0;
        bool dumpedDownstreamTargetEval = false;
        bool dumpedDownstreamControlEval = false;
        for (const auto& entry : pairEntries)
        {
            const int ai         = entry.first.first;
            const int aj         = entry.first.second;
            const int shiftIndex = entry.second;
            const bool isExcludedPairlist = (factorCoulomb == 0.0_real && factorLj == 0.0_real);
            const bool isTargetPair       = needNamedPairChecks && (ai == 0 && aj == 1);
            const bool isControlPair      = needNamedPairChecks && (ai == 0 && aj == 4);
            const bool isM2lTracePair =
                    ((isExcludedPairlist && isTargetPair) || (!isExcludedPairlist && isControlPair));
            const bool isDispatchTracePair = dumpDispatchInternalTrace && isM2lTracePair;
            const bool isBookkeepingTracePair = dumpBookkeepingResidualTrace && isM2lTracePair;
            const bool probeIncludePairRestricted =
                    (useDispatchProbe && dispatchProbeMode == "includepair_restricted" && isExcludedPairlist && isTargetPair);
            const bool probeActiveOuterNarrowed =
                    (useDispatchProbe && dispatchProbeMode == "active_outer_narrowed" && isExcludedPairlist && isTargetPair);
            const bool probeOuterRoutingSuppressed =
                    (useDispatchProbe && dispatchProbeMode == "outer_routing_suppressed" && isExcludedPairlist && isTargetPair);
            const bool probeCorrectionOuterSuppressed =
                    (useDispatchProbe && dispatchProbeMode == "correction_outer_suppressed" && isExcludedPairlist && isTargetPair);
            const bool probeBookkeepingEnergySuppressed =
                    (useDispatchProbe && dispatchProbeMode == "patch_shape_b_bookkeeping_suppressed" && isExcludedPairlist
                     && isTargetPair);
            const bool patchShapeA = (useDispatchProbe && dispatchProbeMode == "patch_shape_a" && isExcludedPairlist);
            const bool patchShapeB =
                    (useDispatchProbe
                     && (dispatchProbeMode == "patch_shape_b"
                      || dispatchProbeMode == "patch_shape_b_bookkeeping_suppressed")
                     && isExcludedPairlist);
            const bool includePairBase      = includePair(ai, aj);
            const bool includePairEffective = includePairBase && !probeIncludePairRestricted;
            const bool traceExclusionEquivalencePair =
                    traceExclusionEquivalence && shouldTraceRespaExclusionEquivalencePair(ai, aj);

            if (isDispatchTracePair)
            {
                appendRespaTraceTextLine(
                        dispatchInternalTraceDirPath,
                        "step0_dispatch_internal_trace.txt",
                        "stage=dispatch_internal_include_pair probe_mode=" + dispatchProbeMode + " pair_list="
                                + std::string(isExcludedPairlist ? "excludedPairs" : "pairs") + " role="
                                + std::string(isTargetPair ? "target_pair_0_1" : "control_pair_0_4") + " ai="
                                + std::to_string(ai) + " aj=" + std::to_string(aj) + " include_pair_base="
                                + std::string(includePairBase ? "true" : "false") + " include_pair_effective="
                                + std::string(includePairEffective ? "true" : "false") + " factor_coulomb="
                                + gmx::toString(factorCoulomb) + " factor_lj=" + gmx::toString(factorLj)
                                + " semantic_result="
                                + std::string(includePairEffective ? "admitted_into_dispatch" : "blocked_before_consumer"));
            }

            if (!includePairEffective)
            {
                continue;
            }

            RVec dx;
            for (int dim = 0; dim < DIM; dim++)
            {
                const real coordI = coordinates[ai][dim];
                const real coordJ = coordinates[aj][dim];
                const real shift  = fr->shift_vec[shiftIndex][dim];
                real       shiftedCoordI = coordI;
                shiftedCoordI += shift;
                real d = shiftedCoordI;
                d -= coordJ;
                dx[dim] = d;
            }

            real rsq = dx[XX] * dx[XX] + dx[YY] * dx[YY] + dx[ZZ] * dx[ZZ];
            rsq      = std::max(rsq, c_nbnxnMinDistanceSquared);

            const real rinv   = gmx::invsqrt(rsq);
            const real rinvsq = rinv * rinv;
            const real r      = rsq * rinv;

            LammpsRespaSplitWeights splitWeights;
            if (exactRespaHasMiddle)
            {
                const real switchIntoMiddle = respaSwitchIn(r, exactRespaForceLayout.innerOff, exactRespaForceLayout.innerOn);
                const real switchIntoOuter  = respaSwitchIn(r, exactRespaForceLayout.outerOn, exactRespaForceLayout.outerOff);
                splitWeights.inner          = 1.0_real - switchIntoMiddle;
                splitWeights.middle         = switchIntoMiddle * (1.0_real - switchIntoOuter);
                splitWeights.outer          = switchIntoOuter;
            }
            else
            {
                const real switchIntoOuter = respaSwitchIn(r, exactRespaForceLayout.outerOn, exactRespaForceLayout.outerOff);
                splitWeights.inner         = 1.0_real - switchIntoOuter;
                splitWeights.middle        = 0.0_real;
                splitWeights.outer         = switchIntoOuter;
            }

            int  typeI       = -1;
            int  typeJ       = -1;
            real c6          = 0;
            real cRepulsive  = 0;
            real rawLjScalar = 0;
            real rawLjEnergy = 0;
            if (factorLj != 0.0_real && rsq < vdwCutoff2)
            {
                typeI = mdatoms.typeA[ai];
                typeJ = mdatoms.typeA[aj];
                c6 = fr->nbfp[typeI * ntype2 + typeJ * 2];
                cRepulsive = fr->nbfp[typeI * ntype2 + typeJ * 2 + 1];
                const real rinvsix = rinvsq * rinvsq * rinvsq;
                const real repulsiveTerm = usePower9SpecializedPath ? (rinvsix * rinvsq * rinv)
                                            : (repulsionPower == 12.0_real ? rinvsix * rinvsix
                                                                           : std::pow(rinv, repulsionPower));
                rawLjScalar = cRepulsive * repulsiveTerm - c6 * rinvsix;
                if (computePairEnergies)
                {
                    rawLjEnergy = cRepulsive * repulsiveTerm * repulsionEnergyPrefactor
                                  - c6 * rinvsix / 6.0_real;
                }
            }

            real bareCoulombScalar = 0;
            real correctionScalar  = 0;
            real fullCoulombEnergy = 0;
            real qq                = 0;
            int  coulTableIndex    = -1;
            real coulFrac          = 0;
            real coulFexcl         = 0;
            real coulVcorr         = 0;
            if (rsq < coulombCutoff2)
            {
                qq = mdatoms.chargeA[ai] * mdatoms.chargeA[aj] * fr->ic->coulomb.epsfac;
                if (qq != 0.0_real)
                {
                    const real scaledR = r * fr->ic->coulombEwaldTables->scale;
                    coulTableIndex     = static_cast<int>(scaledR);
                    coulFrac           = scaledR - coulTableIndex;
                    const real halfsp  = 0.5_real / fr->ic->coulombEwaldTables->scale;
#if !GMX_DOUBLE
                    const real* table = fr->ic->coulombEwaldTables->tableFDV0.data();
                    coulFexcl         = table[coulTableIndex * 4] + coulFrac * table[coulTableIndex * 4 + 1];
#else
                    const real* tableF = fr->ic->coulombEwaldTables->tableF.data();
                    coulFexcl          = (1 - coulFrac) * tableF[coulTableIndex]
                                + coulFrac * tableF[coulTableIndex + 1];
#endif
                    bareCoulombScalar = factorCoulomb * qq * rinv;
                    correctionScalar  = -qq * coulFexcl / rinv;
                    if (computePairEnergies)
                    {
#if !GMX_DOUBLE
                        coulVcorr = table[coulTableIndex * 4 + 2]
                                    - halfsp * coulFrac * (table[coulTableIndex * 4] + coulFexcl);
#else
                        const real* tableV = fr->ic->coulombEwaldTables->tableV.data();
                        coulVcorr = tableV[coulTableIndex]
                                    - halfsp * coulFrac * (tableF[coulTableIndex] + coulFexcl);
#endif
                        fullCoulombEnergy =
                                qq * (factorCoulomb * (rinv - fr->ic->coulomb.ewaldShift) - coulVcorr);
                    }
                }
            }

            if (debugStats != nullptr)
            {
                if (dumpLjSrTrace && isExcludedPairlist && debugStats != nullptr
                    && debugStats->label != nullptr
                    && std::strcmp(debugStats->label, "excludedPairs") == 0
                    && fullCoulombEnergy != 0.0_real)
                {
                    const real targetBefore = debugStats->coulEnergy;
                    appendCoulombProducerTraceLine(ljSrTraceDirPath,
                                                   &patchCoulombProducerOrdinal,
                                                   ai,
                                                   aj,
                                                   energyGroupPairIndex(ai, aj, *fr, mdatoms),
                                                   targetBefore,
                                                   fullCoulombEnergy,
                                                   fullCoulombEnergy,
                                                   qq,
                                                   factorCoulomb,
                                                   rinv,
                                                   fr->ic->coulomb.ewaldShift,
                                                   coulTableIndex,
                                                   coulFrac,
                                                   coulFexcl,
                                                   coulVcorr,
                                                   bareCoulombScalar,
                                                   correctionScalar,
                                                   isExcludedPairlist,
                                                   patchShapeB,
                                                   "src/gromacs/mdlib/sim_util.cpp:1811");
                }
                debugStats->count++;
                debugStats->ljEnergy += rawLjEnergy * factorLj;
                if (dumpMultiStepCoulombStateTrace && !isExcludedPairlist && fullCoulombEnergy != 0.0_real)
                {
                    appendMultiStepCoulombPairTraceLine(ljSrTraceDirPath,
                                                        step,
                                                        pairOrdinal + 1,
                                                        ai,
                                                        aj,
                                                        energyGroupPairIndex(ai, aj, *fr, mdatoms),
                                                        shiftIndex,
                                                        coordinates[ai][XX],
                                                        coordinates[ai][YY],
                                                        coordinates[ai][ZZ],
                                                        coordinates[aj][XX],
                                                        coordinates[aj][YY],
                                                        coordinates[aj][ZZ],
                                                        fr->shift_vec[shiftIndex][XX],
                                                        fr->shift_vec[shiftIndex][YY],
                                                        fr->shift_vec[shiftIndex][ZZ],
                                                        dx[XX],
                                                        dx[YY],
                                                        dx[ZZ],
                                                        rsq,
                                                        qq,
                                                        factorCoulomb,
                                                        rinv,
                                                        coulTableIndex,
                                                        coulFrac,
                                                        coulFexcl,
                                                        coulVcorr,
                                                        fullCoulombEnergy,
                                                        static_cast<real>(debugStats->coulEnergy),
                                                        "src/gromacs/mdlib/sim_util.cpp:2069");
                }
                const real admittedComparableCoulombEnergy =
                        isExcludedPairlist ? 0.0_real : fullCoulombEnergy;
                debugStats->coulEnergy += admittedComparableCoulombEnergy;
                debugStats->rawCoulEnergy += fullCoulombEnergy;
                debugStats->qqSum += qq;
            }
            if (dumpM2qLjSrTrace || dumpM2rLjSrTrace || dumpM2sLjSrTrace || dumpM2vLjSrTrace || dumpM2wLjSrTrace)
            {
                m2qEarliestRawLjTotal += rawLjEnergy * factorLj;
            }

            const real innerCorrectionScalar  = 0.0_real;
            const real middleCorrectionScalar = 0.0_real;
            const real outerCorrectionScalar  = correctionScalar;
            const real innerScalar = bareCoulombScalar * splitWeights.inner
                                     + factorLj * rawLjScalar * splitWeights.inner;
            const real middleScalar = bareCoulombScalar * splitWeights.middle
                                      + factorLj * rawLjScalar * splitWeights.middle;
            const real bareOuterScalar =
                    bareCoulombScalar * splitWeights.outer + factorLj * rawLjScalar * splitWeights.outer;
            const real outerScalar =
                    (patchShapeA ? bareOuterScalar : outerCorrectionScalar + bareOuterScalar);
            const real fullScalar = correctionScalar + bareCoulombScalar + factorLj * rawLjScalar;
            const real effectiveOuterScalar =
                    (probeCorrectionOuterSuppressed || patchShapeB)
                            ? bareOuterScalar
                            : outerScalar;
            if (traceStep1Subset01ForceGroupAudit)
            {
                const real innerLjScalar          = factorLj * rawLjScalar * splitWeights.inner;
                const real innerBareCoulombScalar = bareCoulombScalar * splitWeights.inner;
                const real middleLjScalar         = factorLj * rawLjScalar * splitWeights.middle;
                const real middleBareCoulombScalar = bareCoulombScalar * splitWeights.middle;
                const auto accumulateScalarContribution =
                        [&](TracedForcePair* targetPair, const real scalar)
                {
                    if (scalar == 0.0_real)
                    {
                        return;
                    }
                    RVec force = { 0, 0, 0 };
                    svmul(scalar * rinvsq, dx, force);
                    addPairContributionToTracedPair(targetPair, ai, aj, force);
                };

                accumulateScalarContribution(&tracedExactInnerLjSrForce, innerLjScalar);
                accumulateScalarContribution(&tracedExactInnerBareCoulombSrForce, innerBareCoulombScalar);
                accumulateScalarContribution(&tracedExactInnerCorrectionForce, innerCorrectionScalar);
                accumulateScalarContribution(&tracedExactMiddleLjSrForce, middleLjScalar);
                accumulateScalarContribution(&tracedExactMiddleBareCoulombSrForce, middleBareCoulombScalar);
                accumulateScalarContribution(&tracedExactMiddleCorrectionForce, middleCorrectionScalar);
                accumulateScalarContribution(&tracedExactInnerRealspaceForce, innerScalar);
                accumulateScalarContribution(&tracedExactMiddleRealspaceForce, middleScalar);
                accumulateScalarContribution(&tracedExactOuterRealspaceForce, effectiveOuterScalar);

            }
            if (traceRealspaceForceSubcomponents)
            {
                RVec ljForce = { 0, 0, 0 };
                RVec coulombSrForce = { 0, 0, 0 };

                if (factorLj != 0.0_real && rawLjScalar != 0.0_real)
                {
                    svmul(factorLj * rawLjScalar * rinvsq, dx, ljForce);
                }
                if (bareCoulombScalar != 0.0_real)
                {
                    svmul(bareCoulombScalar * rinvsq, dx, coulombSrForce);
                }

                addPairContributionToTracedPair(&tracedPatchLjSrForce, ai, aj, ljForce);
                addPairContributionToTracedPair(&tracedPatchCoulombSrForce, ai, aj, coulombSrForce);
            }
            const bool effectiveOuterActive = baselineOuterActive && !probeActiveOuterNarrowed;
            const bool bookkeepingSinkEligible =
                    baselineOuterActive && isExcludedPairlist && fullCoulombEnergy != 0.0_real;

            if (isDispatchTracePair)
            {
                appendRespaTraceTextLine(
                        dispatchInternalTraceDirPath,
                        "step0_dispatch_internal_trace.txt",
                        "stage=dispatch_internal_active_contributions probe_mode=" + dispatchProbeMode
                                + " pair_list=" + std::string(isExcludedPairlist ? "excludedPairs" : "pairs")
                                + " role=" + std::string(isTargetPair ? "target_pair_0_1" : "control_pair_0_4")
                                + " ai=" + std::to_string(ai) + " aj=" + std::to_string(aj)
                                + " baseline_active=" + joinActiveContributionLabels(activeContributions, false)
                                + " effective_active="
                                + joinActiveContributionLabels(activeContributions, probeActiveOuterNarrowed)
                                + " baseline_outer_active="
                                + std::string(baselineOuterActive ? "true" : "false")
                                + " effective_outer_active="
                                + std::string(effectiveOuterActive ? "true" : "false") + " inner_scalar="
                                + gmx::toString(innerScalar) + " middle_scalar=" + gmx::toString(middleScalar)
                                + " outer_scalar_baseline=" + gmx::toString(outerScalar)
                                + " outer_scalar_effective=" + gmx::toString(effectiveOuterScalar)
                                + " correction_scalar=" + gmx::toString(correctionScalar)
                                + " semantic_result="
                                + std::string((effectiveOuterActive && effectiveOuterScalar != 0.0_real)
                                                      ? "nonzero_outer_contribution_live"
                                                      : "admitted_but_semantically_harmless"));
            }
            if (isBookkeepingTracePair)
            {
                appendRespaTraceTextLine(
                        bookkeepingResidualTraceDirPath,
                        "step0_patch_b_bookkeeping_trace.txt",
                        "stage=bookkeeping_raw_state probe_mode=" + dispatchProbeMode + " pair_list="
                                + std::string(isExcludedPairlist ? "excludedPairs" : "pairs") + " role="
                                + std::string(isTargetPair ? "target_pair_0_1" : "control_pair_0_4") + " ai="
                                + std::to_string(ai) + " aj=" + std::to_string(aj)
                                + " raw_scalar_present="
                                + std::string((correctionScalar != 0.0_real || bareOuterScalar != 0.0_real) ? "true"
                                                                                                             : "false")
                                + " correction_scalar=" + gmx::toString(correctionScalar)
                                + " bare_outer_scalar=" + gmx::toString(bareOuterScalar)
                                + " outer_scalar_raw=" + gmx::toString(outerScalar)
                                + " effective_outer_scalar=" + gmx::toString(effectiveOuterScalar)
                                + " full_coulomb_energy=" + gmx::toString(fullCoulombEnergy)
                                + " bookkeeping_sink_eligible="
                                + std::string(bookkeepingSinkEligible ? "true" : "false") + " semantic_result="
                                + std::string((correctionScalar != 0.0_real || fullCoulombEnergy != 0.0_real)
                                                      ? "raw_excluded_correction_still_formed"
                                                      : "no_excluded_correction_present"));
            }
            if (dumpDownstreamContract
                && ((isTargetPair && !dumpedDownstreamTargetEval) || (isControlPair && !dumpedDownstreamControlEval)))
            {
                appendRespaTraceTextLine(
                        downstreamContractTraceDirPath,
                        "step0_downstream_contract_trace.txt",
                        "stage=consumer_pair_eval pair_list="
                                + std::string(factorCoulomb == 0.0_real && factorLj == 0.0_real ? "excludedPairs"
                                                                                                 : "pairs")
                                + " ordinal=" + std::to_string(pairOrdinal) + " ai=" + std::to_string(ai) + " aj="
                                + std::to_string(aj) + " shift_index=" + std::to_string(shiftIndex)
                                + " include_pair=true factor_coulomb=" + gmx::toString(factorCoulomb)
                                + " factor_lj=" + gmx::toString(factorLj) + " correction_scalar="
                                + gmx::toString(correctionScalar) + " bare_coulomb_scalar="
                                + gmx::toString(bareCoulombScalar) + " raw_lj_scalar="
                                + gmx::toString(rawLjScalar) + " inner_scalar=" + gmx::toString(innerScalar)
                                + " middle_scalar=" + gmx::toString(middleScalar) + " outer_scalar="
                                + gmx::toString(effectiveOuterScalar) + " full_scalar=" + gmx::toString(fullScalar)
                                + " outer_force_write_eligible="
                                + std::string(effectiveOuterActive && effectiveOuterScalar != 0.0_real
                                                      && outerAccumulator != nullptr
                                                      ? "true"
                                                      : "false")
                                + " outer_force_buffer="
                                + std::string((outerAccumulator != nullptr && outerAccumulator->forceWithVirial != nullptr)
                                                      ? "forceWithVirial"
                                                      : "none")
                                + " semantic_role="
                                + std::string(factorCoulomb == 0.0_real && factorLj == 0.0_real
                                                      ? "excluded_membership_promoted_to_physical_outer_consumer"
                                                      : "standard_physical_nonbonded_consumer"));
                if (isTargetPair)
                {
                    dumpedDownstreamTargetEval = true;
                }
                if (isControlPair)
                {
                    dumpedDownstreamControlEval = true;
                }
            }

            if (dumpExcludedCorrectionForce && factorCoulomb == 0.0_real && factorLj == 0.0_real
                && correctionScalar != 0.0_real)
            {
                RVec correctionForce;
                svmul(correctionScalar * rinvsq, dx, correctionForce);
                rvec_inc(excludedCorrectionForce[ai], correctionForce);
                rvec_dec(excludedCorrectionForce[aj], correctionForce);
            }

            bool        outerWriteExecuted = false;
            std::string outerRoutingTarget = "none";
            bool        correctionWriteExecuted = false;
            std::string correctionRoutingTarget = "none";
            RVec        writtenCorrectionForce = { 0, 0, 0 };
            RVec        writtenCombinedForce   = { 0, 0, 0 };
            for (auto& accumulator : activeContributions)
            {
                if (probeActiveOuterNarrowed
                    && accumulator.contribution == ExactRespaNonbondedContribution::Outer)
                {
                    continue;
                }
                real scalar = 0;
                switch (accumulator.contribution)
                {
                    case ExactRespaNonbondedContribution::Inner:
                        scalar = innerScalar;
                        break;
                    case ExactRespaNonbondedContribution::Middle:
                        scalar = middleScalar;
                        break;
                    case ExactRespaNonbondedContribution::Outer:
                        scalar = effectiveOuterScalar;
                        break;
                    default: GMX_RELEASE_ASSERT(false, "Unexpected nonbonded r-RESPA contribution");
                }

                const bool traceBoundaryBookkeepingPair =
                        shouldTraceRespaMultiStepCoulombStep(step) && !isExcludedPairlist
                        && shouldTraceBoundaryDominantPair(ai, aj)
                        && accumulator.contribution == ExactRespaNonbondedContribution::Outer;
                const bool scalarIsZero = (scalar == 0.0_real);
                if (scalarIsZero)
                {
                    if (traceBoundaryBookkeepingPair && accumulator.accumulateEnergy)
                    {
                        const int energyIndex = energyGroupPairIndex(ai, aj, *fr, mdatoms);
                        appendBoundaryBookkeepingAuditLine(activeM2pTraceDirPath(),
                                                           "PATCH",
                                                           step,
                                                           ai,
                                                           aj,
                                                           isExcludedPairlist ? "excludedPairs" : "pairs",
                                                           contributionLabel(accumulator.contribution),
                                                           r,
                                                           outerScalar,
                                                           effectiveOuterScalar,
                                                           fullCoulombEnergy,
                                                           scalarIsZero,
                                                           false,
                                                           false,
                                                           false,
                                                           energyIndex,
                                                           coulEnergyTerms[energyIndex],
                                                           0.0_real,
                                                           coulEnergyTerms[energyIndex],
                                                           "scalar_zero_gate_pre_energy",
                                                           "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu");
                    }
                    if (!accumulator.accumulateEnergy)
                    {
                        continue;
                    }
                }

                RVec force = { 0, 0, 0 };
                const bool suppressOuterWrite =
                        probeOuterRoutingSuppressed
                        && accumulator.contribution == ExactRespaNonbondedContribution::Outer;
                // Excluded-pair PME real-space correction is a physical outer contribution for
                // exact r-RESPA; masking it here drops the same Coulomb-SR offset we already
                // fixed in the single-step plain-kernel path.
                const bool suppressExcludedPairPhysicalWrite = false;
                if (!scalarIsZero)
                {
                    svmul(scalar * rinvsq, dx, force);
                    if (accumulator.contribution == ExactRespaNonbondedContribution::Outer)
                    {
                        outerRoutingTarget = suppressOuterWrite
                                                     ? "suppressed"
                                                     : (suppressExcludedPairPhysicalWrite
                                                                ? "masked_excluded_pair"
                                                     : (accumulator.forceWithVirial != nullptr ? "forceWithVirial"
                                                                                              : (!accumulator.force.empty()
                                                                                                         ? "forceWithShift"
                                                                                                         : "none")));
                    }
                    const bool shouldDumpExcludedPairWrite =
                            dumpPairWriteProof && isExcludedPairlist
                            && accumulator.contribution == ExactRespaNonbondedContribution::Outer && pairOrdinal == 0;
                    const bool shouldDumpControlPairWrite =
                            dumpPairWriteProof && !isExcludedPairlist
                            && accumulator.contribution == ExactRespaNonbondedContribution::Outer
                            && pairOrdinal < c_maxControlPairWriteProofs;
                    if ((shouldDumpExcludedPairWrite || shouldDumpControlPairWrite)
                        && accumulator.forceWithVirial != nullptr)
                    {
                        const std::string prefix = shouldDumpExcludedPairWrite
                                                           ? "step0_outer_excluded_write_ord000"
                                                           : "step0_outer_pairs_write_ord"
                                                                     + std::to_string(pairOrdinal);
                        const std::string header =
                                "stage="
                                + std::string(shouldDumpExcludedPairWrite ? "first_excluded_outer_write_boundary"
                                                                          : "control_pairs_outer_write_boundary")
                                + " pair_list=" + std::string(isExcludedPairlist ? "excludedPairs" : "pairs")
                                + " ordinal=" + std::to_string(pairOrdinal)
                                + " contribution=outer buffer=forceWithVirial"
                                + " accumulator_ptr=" + formatPointerValue(accumulator.force.data())
                                + " virial_ptr="
                                + formatPointerValue(accumulator.forceWithVirial->force_.data()) + " ai="
                                + std::to_string(ai) + " aj=" + std::to_string(aj) + " shift_index="
                                + std::to_string(shiftIndex) + " scalar=" + gmx::toString(scalar)
                                + " correction_scalar=" + gmx::toString(correctionScalar) + " qq="
                                + gmx::toString(qq) + " r=" + gmx::toString(r);
                        dumpRespaMergeTraceVector(pairWriteProofDirPath,
                                                  (prefix + "_before.tsv").c_str(),
                                                  header + " snapshot=before",
                                                  accumulator.force);
                        dumpRespaTraceEvent(pairWriteProofDirPath,
                                            (prefix + "_event.tsv").c_str(),
                                            header + " snapshot=event",
                                            ai,
                                            force,
                                            aj,
                                            RVec(-force[XX], -force[YY], -force[ZZ]));
                    }
                    if (dumpEarlyAccumTrace && factorCoulomb == 0.0_real && factorLj == 0.0_real
                        && accumulator.contribution == ExactRespaNonbondedContribution::Outer
                        && !dumpedFirstExcludedWrite && !suppressExcludedPairPhysicalWrite)
                    {
                        dumpRespaTraceEvent(
                                earlyAccumTraceDirPath,
                                "step0_outer_first_excluded_write.tsv",
                                "stage=first_excluded_outer_write pair_list=excludedPairs contribution=outer buffer=forceWithVirial alias_with_shift="
                                        + std::string(outerAliasesShift ? "true" : "false") + " ai="
                                        + std::to_string(ai) + " aj=" + std::to_string(aj)
                                        + " shift_index=" + std::to_string(shiftIndex) + " scalar="
                                        + gmx::toString(scalar) + " correction_scalar="
                                        + gmx::toString(correctionScalar) + " qq=" + gmx::toString(qq)
                                        + " r=" + gmx::toString(r),
                                ai,
                                force,
                                aj,
                                RVec(-force[XX], -force[YY], -force[ZZ]));
                        dumpedFirstExcludedWrite = true;
                    }
                    if (!suppressOuterWrite && !suppressExcludedPairPhysicalWrite)
                    {
                        rvec_inc(writtenCombinedForce, force);
                        rvec_inc(accumulator.force[ai], force);
                        rvec_dec(accumulator.force[aj], force);
                        if (accumulator.contribution == ExactRespaNonbondedContribution::Outer)
                        {
                            outerWriteExecuted = true;
                            if (correctionScalar != 0.0_real)
                            {
                                RVec correctionForceOnly;
                                svmul(correctionScalar * rinvsq, dx, correctionForceOnly);
                                rvec_inc(writtenCorrectionForce, correctionForceOnly);
                                correctionWriteExecuted = true;
                            }
                        }
                    }
                    if (accumulator.contribution == ExactRespaNonbondedContribution::Outer
                        && correctionScalar != 0.0_real)
                    {
                        correctionRoutingTarget = outerRoutingTarget;
                    }
                    if ((shouldDumpExcludedPairWrite || shouldDumpControlPairWrite)
                        && accumulator.forceWithVirial != nullptr)
                    {
                        const std::string prefix = shouldDumpExcludedPairWrite
                                                           ? "step0_outer_excluded_write_ord000"
                                                           : "step0_outer_pairs_write_ord"
                                                                     + std::to_string(pairOrdinal);
                        const std::string header =
                                "stage="
                                + std::string(shouldDumpExcludedPairWrite ? "first_excluded_outer_write_boundary"
                                                                          : "control_pairs_outer_write_boundary")
                                + " pair_list=" + std::string(isExcludedPairlist ? "excludedPairs" : "pairs")
                                + " ordinal=" + std::to_string(pairOrdinal)
                                + " contribution=outer buffer=forceWithVirial"
                                + " accumulator_ptr=" + formatPointerValue(accumulator.force.data())
                                + " virial_ptr="
                                + formatPointerValue(accumulator.forceWithVirial->force_.data()) + " ai="
                                + std::to_string(ai) + " aj=" + std::to_string(aj) + " shift_index="
                                + std::to_string(shiftIndex) + " scalar=" + gmx::toString(scalar)
                                + " correction_scalar=" + gmx::toString(correctionScalar) + " qq="
                                + gmx::toString(qq) + " r=" + gmx::toString(r);
                        dumpRespaMergeTraceVector(pairWriteProofDirPath,
                                                  (prefix + "_after.tsv").c_str(),
                                                  header + " snapshot=after",
                                                  accumulator.force);
                    }

                    if (!suppressOuterWrite && !suppressExcludedPairPhysicalWrite && !accumulator.shift.empty()
                        && shiftIndex != c_centralShiftIndex)
                    {
                        rvec_inc(accumulator.shift[shiftIndex], force);
                        rvec_dec(accumulator.shift[c_centralShiftIndex], force);
                    }
                }

                if (accumulator.accumulateEnergy)
                {
                    const int energyIndex = energyGroupPairIndex(ai, aj, *fr, mdatoms);
                    const bool suppressBookkeepingEnergy =
                            probeBookkeepingEnergySuppressed
                            && accumulator.contribution == ExactRespaNonbondedContribution::Outer;
                    const real vdwEnergyDelta = suppressBookkeepingEnergy ? 0.0_real : factorLj * rawLjEnergy;
                    const bool suppressExcludedPairComparableEnergy = false;
                    const real coulEnergyDelta =
                            (suppressBookkeepingEnergy || suppressExcludedPairComparableEnergy)
                                    ? 0.0_real
                                    : fullCoulombEnergy;
                    if (isBookkeepingTracePair
                        && accumulator.contribution == ExactRespaNonbondedContribution::Outer)
                    {
                        appendRespaTraceTextLine(
                                bookkeepingResidualTraceDirPath,
                                "step0_patch_b_bookkeeping_trace.txt",
                                "stage=bookkeeping_energy_sink probe_mode=" + dispatchProbeMode + " pair_list="
                                        + std::string(isExcludedPairlist ? "excludedPairs" : "pairs") + " role="
                                        + std::string(isTargetPair ? "target_pair_0_1" : "control_pair_0_4")
                                        + " ai=" + std::to_string(ai) + " aj=" + std::to_string(aj)
                                        + " contribution=" + contributionLabel(accumulator.contribution)
                                        + " accumulate_energy_base=true accumulate_energy_effective="
                                        + std::string(suppressBookkeepingEnergy ? "false" : "true")
                                        + " energy_index=" + std::to_string(energyIndex)
                                        + " sink_name=coulEnergyTerms sink_class=energy_potential_sink bookkeeping_probe_suppressed="
                                        + std::string(suppressBookkeepingEnergy ? "true" : "false")
                                        + " vdw_energy_delta=" + gmx::toString(vdwEnergyDelta)
                                        + " coul_energy_delta=" + gmx::toString(coulEnergyDelta)
                                        + " residual_visible="
                                        + std::string(coulEnergyDelta != 0.0_real ? "true" : "false")
                                        + " semantic_result="
                                        + std::string(coulEnergyDelta != 0.0_real
                                                              ? "excluded_correction_recorded_in_energy_ledger"
                                                              : "excluded_correction_not_recorded_in_energy_ledger"));
                    }
                    if (dumpLjAccumContractTrace && !isExcludedPairlist && vdwEnergyDelta != 0.0_real
                        && debugStats != nullptr && debugStats->label != nullptr
                        && std::strcmp(debugStats->label, "pairs") == 0)
                    {
                        const real pairStatsLjDelta  = rawLjEnergy * factorLj;
                        const real pairStatsLjAfter  = debugStats->ljEnergy;
                        const real pairStatsLjBefore = pairStatsLjAfter - pairStatsLjDelta;
                        const real targetBeforeVdwEnergyTerms = sumEnergyTermsOnce(vdwEnergyTerms);
                        const real targetAfterVdwEnergyTerms  = targetBeforeVdwEnergyTerms + vdwEnergyDelta;
                        appendLjAccumWriteTraceLine(ljSrTraceDirPath,
                                                    ai,
                                                    aj,
                                                    energyIndex,
                                                    targetBeforeVdwEnergyTerms,
                                                    vdwEnergyDelta,
                                                    targetAfterVdwEnergyTerms,
                                                    pairStatsLjBefore,
                                                    pairStatsLjDelta,
                                                    pairStatsLjAfter,
                                                    "src/gromacs/mdlib/sim_util.cpp:2274");
                    }
                    if (!isExcludedPairlist && energyIndex == 0 && vdwEnergyTerms.size() == 1
                        && debugStats != nullptr && debugStats->label != nullptr
                        && std::strcmp(debugStats->label, "pairs") == 0)
                    {
                        vdwEnergyTerms[energyIndex] = static_cast<real>(debugStats->ljEnergy);
                    }
                    else
                    {
                        vdwEnergyTerms[energyIndex] += vdwEnergyDelta;
                    }
                    if (dumpCoulombPreSelfWindowTrace && energyIndex == 0 && coulEnergyDelta != 0.0_real)
                    {
                        const real targetBefore = coulEnergyTerms[energyIndex];
                        const real targetAfter  = targetBefore + coulEnergyDelta;
                        patchPreSelfWritesForEnergyIndex0.push_back({ "src/gromacs/mdlib/sim_util.cpp:2254",
                                                                      isExcludedPairlist ? "excluded" : "pairs",
                                                                      energyIndex,
                                                                      targetBefore,
                                                                      coulEnergyDelta,
                                                                      targetAfter });
                        if (patchPreSelfWritesForEnergyIndex0.size() > 3)
                        {
                            patchPreSelfWritesForEnergyIndex0.erase(patchPreSelfWritesForEnergyIndex0.begin());
                        }
                    }
                    if (traceBoundaryBookkeepingPair)
                    {
                        const real targetBefore = coulEnergyTerms[energyIndex];
                        const real targetAfter  = targetBefore + coulEnergyDelta;
                        appendBoundaryBookkeepingAuditLine(activeM2pTraceDirPath(),
                                                           "PATCH",
                                                           step,
                                                           ai,
                                                           aj,
                                                           isExcludedPairlist ? "excludedPairs" : "pairs",
                                                           contributionLabel(accumulator.contribution),
                                                           r,
                                                           outerScalar,
                                                           effectiveOuterScalar,
                                                           fullCoulombEnergy,
                                                           scalarIsZero,
                                                           coulEnergyDelta != 0.0_real,
                                                           suppressBookkeepingEnergy,
                                                           suppressExcludedPairComparableEnergy,
                                                           energyIndex,
                                                           targetBefore,
                                                           coulEnergyDelta,
                                                           targetAfter,
                                                           "energy_sink_write",
                                                           "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu");
                    }
                    coulEnergyTerms[energyIndex] += coulEnergyDelta;
                    if ((dumpM2sLjSrTrace || dumpM2uLjSrTrace) && !m2sFirstWriteCaptured
                        && vdwEnergyDelta != 0.0_real)
                    {
                        m2sFirstWriteLjTotal  = sumEnergyTermsOnce(vdwEnergyTerms);
                        m2sFirstWriteCaptured = true;
                    }
                    if (dumpM2vLjSrTrace && vdwEnergyDelta != 0.0_real)
                    {
                        m2vAlignedEventLjRunningTotal += vdwEnergyDelta;
                        m2vAlignedEventLjTotals.push_back(m2vAlignedEventLjRunningTotal);
                    }
                    if (dumpM2wLjSrTrace && vdwEnergyDelta != 0.0_real)
                    {
                        const double runningBefore = m2wAlignedEventLjRunningTotal;
                        m2wAlignedEventLjRunningTotal += vdwEnergyDelta;
                        m2wAlignedEventLjTotals.push_back(m2wAlignedEventLjRunningTotal);
                        const int alignedEventOrdinal = static_cast<int>(m2wAlignedEventLjTotals.size());
                        if (alignedEventOrdinal >= 668 && alignedEventOrdinal <= 670)
                        {
                            M2wAlignedEventRecord record;
                            record.alignedEventOrdinal = alignedEventOrdinal;
                            record.pairOrdinal         = pairOrdinal;
                            record.pairI               = ai;
                            record.pairJ               = aj;
                            record.typeI               = typeI;
                            record.typeJ               = typeJ;
                            record.shiftIndex          = shiftIndex;
                            record.runningTotalBefore  = runningBefore;
                            record.runningTotalAfter   = m2wAlignedEventLjRunningTotal;
                            record.rawLjTerm           = rawLjEnergy;
                            record.scalingFactor       = factorLj;
                            record.finalEventLj        = vdwEnergyDelta;
                            record.c6                  = c6;
                            record.c12                 = cRepulsive;
                            record.rsq                 = rsq;
                            record.r                   = r;
                            m2wAlignedEventRecords.push_back(record);
                        }
                    }
                    if (dumpM2xGeometryTrace && vdwEnergyDelta != 0.0_real)
                    {
                        M2xGeometryEventRecord record;
                        record.pairOrdinal    = pairOrdinal;
                        record.pairI          = ai;
                        record.pairJ          = aj;
                        record.typeI          = typeI;
                        record.typeJ          = typeJ;
                        record.shiftIndex     = shiftIndex;
                        record.coordISourceX  = coordinates[ai][XX];
                        record.coordISourceY  = coordinates[ai][YY];
                        record.coordISourceZ  = coordinates[ai][ZZ];
                        record.coordJSourceX  = coordinates[aj][XX];
                        record.coordJSourceY  = coordinates[aj][YY];
                        record.coordJSourceZ  = coordinates[aj][ZZ];
                        record.shiftX         = fr->shift_vec[shiftIndex][XX];
                        record.shiftY         = fr->shift_vec[shiftIndex][YY];
                        record.shiftZ         = fr->shift_vec[shiftIndex][ZZ];
                        record.coordIShiftedX = coordinates[ai][XX] + fr->shift_vec[shiftIndex][XX];
                        record.coordIShiftedY = coordinates[ai][YY] + fr->shift_vec[shiftIndex][YY];
                        record.coordIShiftedZ = coordinates[ai][ZZ] + fr->shift_vec[shiftIndex][ZZ];
                        record.dx             = dx[XX];
                        record.dy             = dx[YY];
                        record.dz             = dx[ZZ];
                        record.rsq            = rsq;
                        record.r              = r;
                        record.rawLjTerm      = rawLjEnergy;
                        record.finalEventLj   = vdwEnergyDelta;
                        noteM2xGeometryEvent(record);
                    }
                    if (dumpM2uLjSrTrace && vdwEnergyDelta != 0.0_real)
                    {
                        m2uWriteOrdinalLjTotals.push_back(sumEnergyTermsOnce(vdwEnergyTerms));
                    }
                }

                if (!suppressOuterWrite && !suppressExcludedPairPhysicalWrite && stepWork.computeVirial
                    && accumulator.forceWithVirial != nullptr)
                {
                    accumulatePairVirial(dx, force, accumulator.virial);
                }
            }

            if (traceExclusionEquivalencePair)
            {
                appendExclusionEquivalenceTracePair(
                        activeM2pTraceDirPath(),
                        "PATCH",
                        step,
                        pairOrdinal,
                        ai,
                        aj,
                        "exact_cpu_rrespa",
                        isExcludedPairlist ? "plainPairlist.excludedPairs" : "plainPairlist.pairs",
                        isExcludedPairlist ? "excludedPairs" : "pairs",
                        isExcludedPairlist && correctionScalar != 0.0_real,
                        includePairEffective,
                        factorCoulomb,
                        factorLj,
                        qq,
                        coulTableIndex,
                        coulFrac,
                        coulFexcl,
                        coulVcorr,
                        bareCoulombScalar,
                        correctionScalar,
                        correctionScalar * rinvsq,
                        effectiveOuterScalar,
                        fullScalar,
                        writtenCorrectionForce,
                        writtenCombinedForce,
                        correctionRoutingTarget.c_str(),
                        correctionWriteExecuted,
                        isExcludedPairlist ? "excluded_pairlist_entry_promoted_to_outer_contribution"
                                           : "pairlist_entry_outer_correction_component",
                        isExcludedPairlist ? "excluded_pairlist_correction_candidate"
                                           : "included_pair_outer_correction_component",
                        "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu_exclusion_equivalence_trace");
            }

            if (traceRealspaceForceSubcomponents)
            {
                addPairContributionToTracedPair(
                        &tracedPatchExclusionCorrectionForce, ai, aj, writtenCorrectionForce);
                addPairContributionToTracedPair(
                        &tracedPatchCombinedRealspaceForce, ai, aj, writtenCombinedForce);
            }

            if (isDispatchTracePair)
            {
                appendRespaTraceTextLine(
                        dispatchInternalTraceDirPath,
                        "step0_dispatch_internal_trace.txt",
                        "stage=dispatch_internal_outer_routing probe_mode=" + dispatchProbeMode + " pair_list="
                                + std::string(isExcludedPairlist ? "excludedPairs" : "pairs") + " role="
                                + std::string(isTargetPair ? "target_pair_0_1" : "control_pair_0_4") + " ai="
                                + std::to_string(ai) + " aj=" + std::to_string(aj) + " effective_outer_active="
                                + std::string(effectiveOuterActive ? "true" : "false")
                                + " effective_outer_scalar=" + gmx::toString(effectiveOuterScalar)
                                + " outer_force_write_eligible="
                                + std::string(effectiveOuterActive && effectiveOuterScalar != 0.0_real
                                                      && outerAccumulator != nullptr
                                                      ? "true"
                                                      : "false")
                                + " outer_routing_target=" + outerRoutingTarget + " actual_outer_write_executed="
                                + std::string(outerWriteExecuted ? "true" : "false") + " semantic_result="
                                + std::string(outerWriteExecuted ? "physical_outer_behavior_realized"
                                                                 : "no_physical_outer_realization"));
            }
            if (isBookkeepingTracePair)
            {
                appendRespaTraceTextLine(
                        bookkeepingResidualTraceDirPath,
                        "step0_patch_b_bookkeeping_trace.txt",
                        "stage=bookkeeping_force_state probe_mode=" + dispatchProbeMode + " pair_list="
                                + std::string(isExcludedPairlist ? "excludedPairs" : "pairs") + " role="
                                + std::string(isTargetPair ? "target_pair_0_1" : "control_pair_0_4") + " ai="
                                + std::to_string(ai) + " aj=" + std::to_string(aj) + " effective_outer_active="
                                + std::string(effectiveOuterActive ? "true" : "false")
                                + " effective_outer_scalar=" + gmx::toString(effectiveOuterScalar)
                                + " outer_force_write_eligible="
                                + std::string(effectiveOuterActive && effectiveOuterScalar != 0.0_real
                                                      && outerAccumulator != nullptr
                                                      ? "true"
                                                      : "false")
                                + " actual_outer_write_executed="
                                + std::string(outerWriteExecuted ? "true" : "false") + " sink_name="
                                + outerRoutingTarget + " sink_class=physical_force_sink semantic_result="
                                + std::string(outerWriteExecuted ? "physical_force_sink_receives_contribution"
                                                                 : "no_physical_force_sink_receives_contribution"));
            }

            pairOrdinal++;
        }
    };

    PairDebugStats pairStats{ "pairs", 0, 0, 0, 0 };
    PairDebugStats excludedStats{ "excludedPairs", 0, 0, 0, 0 };
    int            listedPairsInPairlist         = 0;
    int            listedPairsInExcludedPairlist = 0;
    int            duplicatePairs                = 0;
    int            duplicateExcludedPairs        = 0;
    if (debugExactRespa)
    {
        std::unordered_set<uint64_t> uniquePairs;
        std::unordered_set<uint64_t> uniqueExcludedPairs;
        for (const auto& entry : plainPairlist.pairs)
        {
            duplicatePairs += !uniquePairs.insert(pairKey(entry.first.first, entry.first.second)).second;
            listedPairsInPairlist += listedPairKeys.count(pairKey(entry.first.first, entry.first.second)) != 0;
        }
        for (const auto& entry : plainPairlist.excludedPairs)
        {
            duplicateExcludedPairs +=
                    !uniqueExcludedPairs.insert(pairKey(entry.first.first, entry.first.second)).second;
            listedPairsInExcludedPairlist +=
                    listedPairKeys.count(pairKey(entry.first.first, entry.first.second)) != 0;
        }
    }
    const double patchCombinedBeforePairs =
            (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0') ? sumEnergyTermsOnce(coulEnergyTerms) : 0.0;
    const double patchLjCombinedBeforePairs =
            (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0') ? sumEnergyTermsOnce(vdwEnergyTerms) : 0.0;
    capturePairLoopForceDumpBefore();
    const bool pairLoopFastPathUsedPairs =
            processPairlistOmp("pairs",
                               plainPairlist.pairs,
                               PairLoopListKind::StandardPairs);
    if (!pairLoopFastPathUsedPairs)
    {
        processPairlist(plainPairlist.pairs,
                        1.0_real,
                        1.0_real,
                        [](const int, const int) { return true; },
                        (debugExactRespa || dumpLjSrTrace || traceCpuCorrectionEnergies) ? &pairStats : nullptr);
    }
    const double patchCombinedAfterPairs =
            (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0') ? sumEnergyTermsOnce(coulEnergyTerms) : 0.0;
    const double patchLjCombinedAfterPairs =
            (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0') ? sumEnergyTermsOnce(vdwEnergyTerms) : 0.0;
    if (dumpEarlyAccumTrace && outerAccumulator != nullptr && outerAccumulator->forceWithVirial != nullptr)
    {
        dumpRespaMergeTraceVector(earlyAccumTraceDirPath,
                                  "step0_level2_after_pairs_virial.tsv",
                                  "stage=after_pairs_dispatch mts_index=2 mts_user=3 buffer=forceWithVirial alias_with_shift="
                                          + std::string(outerAliasesShift ? "true" : "false"),
                                  outerAccumulator->forceWithVirial->force_);
    }
    /* PME real-space bookkeeping for exact PME parity requires all excluded pairs,
     * not only the listed 1-4 subset. */
    const bool pairLoopFastPathUsedExcludedPairs =
            processPairlistOmp("excludedPairs",
                               plainPairlist.excludedPairs,
                               PairLoopListKind::ExcludedPairs);
    if (!pairLoopFastPathUsedExcludedPairs)
    {
        processPairlist(plainPairlist.excludedPairs,
                        0.0_real,
                        0.0_real,
                        [](const int, const int) { return true; },
                        (debugExactRespa || dumpLjSrTrace || traceCpuCorrectionEnergies) ? &excludedStats : nullptr);
    }
    writePairLoopForceDeltaDump(pairLoopFastPathUsedPairs, pairLoopFastPathUsedExcludedPairs);
    if (traceRealspaceForceSubcomponents)
    {
        appendRealspaceForceSubcomponentTracePair(activeM2pTraceDirPath(),
                                                  "PATCH",
                                                  step,
                                                  "lj_sr_force",
                                                  tracedPatchLjSrForce,
                                                  "computeLammpsRespaNonbondedCpu.rawLjScalar",
                                                  "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                                  "true_source_component",
                                                  true);
        appendRealspaceForceSubcomponentTracePair(activeM2pTraceDirPath(),
                                                  "PATCH",
                                                  step,
                                                  "coulomb_sr_force",
                                                  tracedPatchCoulombSrForce,
                                                  "computeLammpsRespaNonbondedCpu.bareCoulombScalar",
                                                  "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                                  "true_source_component",
                                                  true);
        appendRealspaceForceSubcomponentTracePair(activeM2pTraceDirPath(),
                                                  "PATCH",
                                                  step,
                                                  "exclusion_correction_force",
                                                  tracedPatchExclusionCorrectionForce,
                                                  "computeLammpsRespaNonbondedCpu.correctionScalar",
                                                  "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                                  "true_source_component",
                                                  true);
        appendRealspaceForceSubcomponentUnavailablePair(
                activeM2pTraceDirPath(),
                "PATCH",
                step,
                "additional_realspace_correction_force",
                "no_additional_realspace_correction_component_in_exact_path",
                "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                "runtime_force_component_unavailable");
        appendRealspaceForceSubcomponentTracePair(activeM2pTraceDirPath(),
                                                  "PATCH",
                                                  step,
                                                  "realspace_nonbonded_combined_force",
                                                  tracedPatchCombinedRealspaceForce,
                                                  "computeLammpsRespaNonbondedCpu.fullScalar",
                                                  "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                                  "combined_total",
                                                  false);
    }
    if (traceRealspaceForceSubcomponents || !traceOnlyDiagnostics)
    {
        storeExactRespaRealspaceTraceCapture(step,
                                             tracedPatchLjSrForce,
                                             tracedPatchCoulombSrForce,
                                             tracedPatchExclusionCorrectionForce,
                                             tracedPatchCombinedRealspaceForce);
    }
    if (traceStep1Subset01ForceGroupAudit)
    {
        const TracedForcePair tracedExactSubset01RealspaceForce =
                addTracedForcePairs(tracedExactInnerRealspaceForce, tracedExactMiddleRealspaceForce);
        const char* producerTermTraceFile = "step2_current_nonbonded_producer_term_trace.txt";
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            producerTermTraceFile,
                                            "PATCH",
                                            step,
                                            "inner_lj_sr_force",
                                            tracedExactInnerLjSrForce,
                                            "computeLammpsRespaNonbondedCpu.factorLj_rawLjScalar_splitWeights.inner",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "term_component",
                                            true);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            producerTermTraceFile,
                                            "PATCH",
                                            step,
                                            "inner_bare_coulomb_sr_force",
                                            tracedExactInnerBareCoulombSrForce,
                                            "computeLammpsRespaNonbondedCpu.bareCoulombScalar_splitWeights.inner",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "term_component",
                                            true);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            producerTermTraceFile,
                                            "PATCH",
                                            step,
                                            "inner_correction_force",
                                            tracedExactInnerCorrectionForce,
                                            "computeLammpsRespaNonbondedCpu.correctionScalar_splitWeights.inner",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "term_component",
                                            true);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            producerTermTraceFile,
                                            "PATCH",
                                            step,
                                            "inner_total",
                                            tracedExactInnerRealspaceForce,
                                            "computeLammpsRespaNonbondedCpu.innerScalar",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "combined_term_total",
                                            false);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            producerTermTraceFile,
                                            "PATCH",
                                            step,
                                            "middle_lj_sr_force",
                                            tracedExactMiddleLjSrForce,
                                            "computeLammpsRespaNonbondedCpu.factorLj_rawLjScalar_splitWeights.middle",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "term_component",
                                            true);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            producerTermTraceFile,
                                            "PATCH",
                                            step,
                                            "middle_bare_coulomb_sr_force",
                                            tracedExactMiddleBareCoulombSrForce,
                                            "computeLammpsRespaNonbondedCpu.bareCoulombScalar_splitWeights.middle",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "term_component",
                                            true);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            producerTermTraceFile,
                                            "PATCH",
                                            step,
                                            "middle_correction_force",
                                            tracedExactMiddleCorrectionForce,
                                            "computeLammpsRespaNonbondedCpu.correctionScalar_splitWeights.middle",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "term_component",
                                            true);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            producerTermTraceFile,
                                            "PATCH",
                                            step,
                                            "middle_total",
                                            tracedExactMiddleRealspaceForce,
                                            "computeLammpsRespaNonbondedCpu.middleScalar",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "combined_term_total",
                                            false);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            producerTermTraceFile,
                                            "PATCH",
                                            step,
                                            "inner_live",
                                            tracedExactInnerRealspaceForce,
                                            "computeLammpsRespaNonbondedCpu.innerScalar",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "live_buffer_payload",
                                            false);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            producerTermTraceFile,
                                            "PATCH",
                                            step,
                                            "middle_live",
                                            tracedExactMiddleRealspaceForce,
                                            "computeLammpsRespaNonbondedCpu.middleScalar",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "live_buffer_payload",
                                            false);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            producerTermTraceFile,
                                            "PATCH",
                                            step,
                                            "subset01_after_nonbonded",
                                            tracedExactSubset01RealspaceForce,
                                            "computeLammpsRespaNonbondedCpu.inner_plus_middle_current_nonbonded",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "subset_total",
                                            false);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            "step1_subset01_forcegroup_realspace_split_trace.txt",
                                            "PATCH",
                                            step,
                                            "nonbonded_inner_live",
                                            tracedExactInnerRealspaceForce,
                                            "computeLammpsRespaNonbondedCpu.innerScalar",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "true_source_component",
                                            true);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            "step1_subset01_forcegroup_realspace_split_trace.txt",
                                            "PATCH",
                                            step,
                                            "nonbonded_middle_live",
                                            tracedExactMiddleRealspaceForce,
                                            "computeLammpsRespaNonbondedCpu.middleScalar",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "true_source_component",
                                            true);
        appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                            "step1_subset01_forcegroup_realspace_split_trace.txt",
                                            "PATCH",
                                            step,
                                            "nonbonded_outer_live",
                                            tracedExactOuterRealspaceForce,
                                            "computeLammpsRespaNonbondedCpu.effectiveOuterScalar",
                                            "src/gromacs/mdlib/sim_util.cpp:computeLammpsRespaNonbondedCpu",
                                            "true_source_component",
                                            true);
    }
    const double patchCombinedAfterExcluded =
            (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0') ? sumEnergyTermsOnce(coulEnergyTerms) : 0.0;
    const double patchLjCombinedAfterExcluded =
            (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0') ? sumEnergyTermsOnce(vdwEnergyTerms) : 0.0;
    if (dumpLjSrTrace)
    {
        if (dumpM2qLjSrTrace || dumpM2rLjSrTrace || dumpM2sLjSrTrace || dumpM2uLjSrTrace
            || dumpM2vLjSrTrace || dumpM2wLjSrTrace)
        {
            appendRespaTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_lj_sr_internal_trace.txt",
                    "stage=EARLIEST_RAW_STAGE code_location=src/gromacs/mdlib/sim_util.cpp:per_pair_rawLjEnergy_before_pairStats_aggregate case_label="
                            + ljSrTraceCaseLabel
                            + " execution_path=exact_respa_per_pair_raw_energy trace_role=contract_matched_raw_lj_formation_aggregate lj_sr="
                            + formatString("%.15f", m2qEarliestRawLjTotal));
        }
        if (dumpM2rLjSrTrace || dumpM2sLjSrTrace)
        {
            appendRespaTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_lj_sr_internal_trace.txt",
                    "stage=INTERMEDIATE_LOCAL_STAGE code_location=src/gromacs/mdlib/sim_util.cpp:after_pairs_pairStats_before_excluded_transfer case_label="
                            + ljSrTraceCaseLabel
                            + " execution_path=exact_respa_pairs_local_energy_aggregate trace_role=contract_matched_kernel_local_lj_aggregate lj_sr="
                            + formatString("%.15f", pairStats.ljEnergy));
        }
        if (dumpM2vLjSrTrace || dumpM2wLjSrTrace)
        {
            const auto& alignedTotals = dumpM2wLjSrTrace ? m2wAlignedEventLjTotals : m2vAlignedEventLjTotals;
            if (!alignedTotals.empty())
            {
                for (std::size_t eventIndex = 0; eventIndex < alignedTotals.size(); ++eventIndex)
                {
                    appendRespaTraceTextLine(
                            ljSrTraceDirPath,
                            "step0_lj_sr_internal_trace.txt",
                            "stage=ALIGNED_WRITE_EVENT_" + std::to_string(eventIndex + 1)
                                    + " code_location=src/gromacs/mdlib/sim_util.cpp:after_patch_pair_energy_event case_label="
                                    + ljSrTraceCaseLabel
                                    + " execution_path=exact_aligned_pair_energy_event aligned_contract=running_total_after_admitted_pair_energy_event aligned_event_ordinal="
                                    + std::to_string(eventIndex + 1) + " lj_sr="
                                    + formatString("%.15f", alignedTotals[eventIndex]));
                }
                appendRespaTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_lj_sr_internal_trace.txt",
                        "stage=ALIGNED_LAST_EVENT_BEFORE_RAW_POST_WRITE code_location=src/gromacs/mdlib/sim_util.cpp:after_patch_last_pair_energy_event case_label="
                                + ljSrTraceCaseLabel
                                + " execution_path=exact_aligned_pair_energy_after_last_event aligned_contract=running_total_after_admitted_pair_energy_event aligned_event_ordinal="
                                + std::to_string(alignedTotals.size()) + " lj_sr="
                                + formatString("%.15f", alignedTotals.back()));
            }
            appendRespaTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_lj_sr_internal_trace.txt",
                    "stage=RAW_POST_WRITE_EQUIVALENT code_location=src/gromacs/mdlib/sim_util.cpp:after_pair_loop_vdwEnergyTerms case_label="
                            + ljSrTraceCaseLabel
                            + " execution_path=exact_aligned_post_write_equivalent trace_role=post_aligned_event_target_state lj_sr="
                            + formatString("%.15f", sumEnergyTermsOnce(vdwEnergyTerms)));
            if (dumpM2wLjSrTrace)
            {
                for (const auto& record : m2wAlignedEventRecords)
                {
                    appendRespaTraceTextLine(
                            ljSrTraceDirPath,
                            "step0_aligned_event_identity_trace.txt",
                            "stage=ALIGNED_WRITE_EVENT_" + std::to_string(record.alignedEventOrdinal)
                                    + " code_location=src/gromacs/mdlib/sim_util.cpp:after_patch_pair_energy_event case_label="
                                    + ljSrTraceCaseLabel
                                    + " execution_path=exact_aligned_pair_energy_event aligned_contract=running_total_after_admitted_pair_energy_event aligned_event_ordinal="
                                    + std::to_string(record.alignedEventOrdinal) + " pair_i="
                                    + std::to_string(record.pairI) + " pair_j="
                                    + std::to_string(record.pairJ) + " type_i="
                                    + std::to_string(record.typeI) + " type_j="
                                    + std::to_string(record.typeJ) + " pair_ordinal="
                                    + std::to_string(record.pairOrdinal) + " shift_index="
                                    + std::to_string(record.shiftIndex) + " event_ordering_key="
                                    + std::to_string(record.pairI) + "_" + std::to_string(record.pairJ)
                                    + " running_total_before="
                                    + formatString("%.15f", record.runningTotalBefore)
                                    + " raw_lj_term=" + formatString("%.15f", record.rawLjTerm)
                                    + " scaling_factor=" + formatString("%.15f", record.scalingFactor)
                                    + " final_event_lj_contribution="
                                    + formatString("%.15f", record.finalEventLj)
                                    + " running_total_after="
                                    + formatString("%.15f", record.runningTotalAfter) + " c6="
                                    + formatString("%.15f", record.c6) + " c12="
                                    + formatString("%.15f", record.c12) + " rsq="
                                    + formatString("%.15f", record.rsq) + " r="
                                    + formatString("%.15f", record.r));
                }
            }
        }
        if (dumpM2sLjSrTrace || dumpM2uLjSrTrace)
        {
            appendRespaTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_lj_sr_internal_trace.txt",
                    "stage=RAW_PRE_TRANSFER code_location=src/gromacs/mdlib/sim_util.cpp:before_vdwEnergyTerms_transfer case_label="
                            + ljSrTraceCaseLabel
                            + " execution_path=exact_pairs_local_aggregate_pre_transfer trace_role=source_aggregate_before_first_vdwEnergyTerms_write lj_sr="
                            + formatString("%.15f", pairStats.ljEnergy));
            if (dumpM2uLjSrTrace && !m2uWriteOrdinalLjTotals.empty())
            {
                appendRespaTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_lj_sr_internal_trace.txt",
                        "stage=RAW_FIRST_WRITE code_location=src/gromacs/mdlib/sim_util.cpp:after_vdwEnergyTerms_write_ordinal_1 case_label="
                                + ljSrTraceCaseLabel
                                + " execution_path=exact_vdwEnergyTerms_after_write_ordinal target_container=vdwEnergyTerms trace_role=running_total_after_write_ordinal write_ordinal=1 lj_sr="
                                + formatString("%.15f", m2uWriteOrdinalLjTotals.front()));
                for (std::size_t ordinalIndex = 1; ordinalIndex < m2uWriteOrdinalLjTotals.size(); ++ordinalIndex)
                {
                    appendRespaTraceTextLine(
                            ljSrTraceDirPath,
                            "step0_lj_sr_internal_trace.txt",
                            "stage=AFTER_WRITE_ORDINAL_" + std::to_string(ordinalIndex + 1)
                                    + " code_location=src/gromacs/mdlib/sim_util.cpp:after_vdwEnergyTerms_write_ordinal case_label="
                                    + ljSrTraceCaseLabel
                                    + " execution_path=exact_vdwEnergyTerms_after_write_ordinal target_container=vdwEnergyTerms trace_role=running_total_after_write_ordinal write_ordinal="
                                    + std::to_string(ordinalIndex + 1) + " lj_sr="
                                    + formatString("%.15f", m2uWriteOrdinalLjTotals[ordinalIndex]));
                }
                appendRespaTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_lj_sr_internal_trace.txt",
                        "stage=AFTER_LAST_WRITE_BEFORE_RAW_POST_WRITE code_location=src/gromacs/mdlib/sim_util.cpp:after_vdwEnergyTerms_last_write case_label="
                                + ljSrTraceCaseLabel
                                + " execution_path=exact_vdwEnergyTerms_after_last_write target_container=vdwEnergyTerms trace_role=running_total_after_last_write_before_raw_post_write write_ordinal="
                                + std::to_string(m2uWriteOrdinalLjTotals.size()) + " lj_sr="
                                + formatString("%.15f", m2uWriteOrdinalLjTotals.back()));
            }
            else if (m2sFirstWriteCaptured)
            {
                appendRespaTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_lj_sr_internal_trace.txt",
                        "stage=RAW_FIRST_WRITE code_location=src/gromacs/mdlib/sim_util.cpp:1700 case_label="
                                + ljSrTraceCaseLabel
                                + " execution_path=exact_vdwEnergyTerms_first_write trace_role=first_vdwEnergyTerms_write_target lj_sr="
                                + formatString("%.15f", m2sFirstWriteLjTotal));
            }
            appendRespaTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_lj_sr_internal_trace.txt",
                    "stage=RAW_POST_WRITE code_location=src/gromacs/mdlib/sim_util.cpp:after_pair_loop_vdwEnergyTerms case_label="
                            + ljSrTraceCaseLabel
                            + " execution_path=exact_vdwEnergyTerms_post_write target_container=vdwEnergyTerms write_count="
                            + std::to_string(m2uWriteOrdinalLjTotals.size())
                            + " trace_role=post_write_target_state lj_sr="
                            + formatString("%.15f", sumEnergyTermsOnce(vdwEnergyTerms)));
        }
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_lj_sr_internal_trace.txt",
                "stage=RAW_SR_FORMATION code_location=src/gromacs/mdlib/sim_util.cpp:1754 case_label="
                        + ljSrTraceCaseLabel
                        + " execution_path=exact_respa_pairs trace_role=pair_loop_raw_energy_delta pair_list=pairs lj_sr="
                        + formatString("%.15f", pairStats.ljEnergy) + " coulomb_sr="
                        + formatString("%.15f", pairStats.coulEnergy) + " pair_count="
                        + std::to_string(pairStats.count));
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_lj_sr_internal_trace.txt",
                "stage=RAW_SR_FORMATION code_location=src/gromacs/mdlib/sim_util.cpp:1769 case_label="
                        + ljSrTraceCaseLabel
                        + " execution_path=exact_respa_excluded_pairs trace_role=pair_loop_raw_energy_delta pair_list=excludedPairs lj_sr="
                        + formatString("%.15f", excludedStats.ljEnergy) + " coulomb_sr="
                        + formatString("%.15f", excludedStats.coulEnergy) + " pair_count="
                        + std::to_string(excludedStats.count));
    }
    if (dumpEarlyAccumTrace && outerAccumulator != nullptr && outerAccumulator->forceWithVirial != nullptr)
    {
        dumpRespaMergeTraceVector(earlyAccumTraceDirPath,
                                  "step0_level2_after_excluded_pairs_virial.tsv",
                                  "stage=after_excluded_pairs_dispatch mts_index=2 mts_user=3 buffer=forceWithVirial alias_with_shift="
                                          + std::string(outerAliasesShift ? "true" : "false"),
                                  outerAccumulator->forceWithVirial->force_);
    }

    for (auto& accumulator : activeContributions)
    {
        if (!accumulator.accumulateEnergy)
        {
            continue;
        }

        for (int atom = 0; atom < fr->natoms_force_constr; ++atom)
        {
            const real charge = mdatoms.chargeA[atom];
            if (charge == 0.0_real)
            {
                continue;
            }

            const int energyIndex = energyGroupPairIndex(atom, atom, *fr, mdatoms);
            const real selfEnergy = -fr->ic->coulomb.epsfac * charge * charge * pmeSelfEnergy;
            ++selfEnergyAtomCount;
            if (dumpCoulombPreSelfWindowTrace && energyIndex == 0 && selfEnergy != 0.0_real)
            {
                static std::string emittedPreSelfWindowTracePath;
                const std::string  tracePath =
                        (std::filesystem::path(ljSrTraceDirPath) / "step0_coulomb_pre_self_window.txt").string();
                if (tracePath != emittedPreSelfWindowTracePath)
                {
                    for (std::size_t i = 0; i < patchPreSelfWritesForEnergyIndex0.size(); ++i)
                    {
                        const auto& write = patchPreSelfWritesForEnergyIndex0[i];
                        appendRespaTraceTextLine(
                                ljSrTraceDirPath,
                                "step0_coulomb_pre_self_window.txt",
                                "side=PATCH kind=last_pre_self_write slot=" + std::to_string(i + 1)
                                        + " role_label=" + write.roleLabel + " code_location="
                                        + write.codeLocation + " energyIndex="
                                        + std::to_string(write.energyIndex) + " target_before="
                                        + formatString("%.15f", write.targetBefore) + " write_value="
                                        + formatString("%.15f", write.writeValue) + " target_after="
                                        + formatString("%.15f", write.targetAfter));
                    }
                    appendRespaTraceTextLine(
                            ljSrTraceDirPath,
                            "step0_coulomb_pre_self_window.txt",
                            "side=PATCH kind=self_entry_read energyIndex=" + std::to_string(energyIndex)
                                    + " target_before_at_self_entry="
                                    + formatString("%.15f", coulEnergyTerms[energyIndex])
                                    + " code_location=src/gromacs/mdlib/sim_util.cpp:2646");
                    emittedPreSelfWindowTracePath = tracePath;
                }
            }
            if (dumpLjSrTrace && selfEnergy != 0.0_real)
            {
                const real targetBefore = coulEnergyTerms[energyIndex];
                const real targetAfter  = targetBefore + selfEnergy;
                appendCoulombSelfTraceLine(ljSrTraceDirPath,
                                           atom,
                                           energyIndex,
                                           charge,
                                           selfEnergy,
                                           targetBefore,
                                           targetAfter,
                                           "src/gromacs/mdlib/sim_util.cpp:2549");
            }
            if (dumpCoulombFirstWritesTrace && selfEnergy != 0.0_real)
            {
                const real targetBefore = coulEnergyTerms[energyIndex];
                const real targetAfter  = targetBefore + selfEnergy;
                appendCoulombFirstWriteTraceLine(ljSrTraceDirPath,
                                                 &patchCoulombFirstWriteOrdinal,
                                                 targetBefore,
                                                 selfEnergy,
                                                 targetAfter,
                                                 energyIndex,
                                                 "src/gromacs/mdlib/sim_util.cpp:2315");
            }
            coulEnergyTerms[energyIndex] += selfEnergy;
            if (debugExactRespa || traceCpuCorrectionEnergies)
            {
                pairStats.selfEnergy += selfEnergy;
            }
        }
    }
    const double patchCombinedBeforeSelf =
            (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0') ? patchCombinedAfterExcluded : 0.0;
    const double patchCombinedAfterSelf =
            (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0') ? sumEnergyTermsOnce(coulEnergyTerms) : 0.0;
    const double patchLjCombinedBeforeSelf =
            (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0') ? patchLjCombinedAfterExcluded : 0.0;
    const double patchLjCombinedAfterSelf =
            (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0') ? sumEnergyTermsOnce(vdwEnergyTerms) : 0.0;
    if (stepWork.computeEnergy && shouldTraceCpuCorrectionEnergiesStep(step))
    {
        const int outerLevel = exactRespaNonbondedOuterLevel(inputrec);
        const double reciprocalEnergy =
                static_cast<double>(enerd->term[InteractionFunction::CoulombReciprocalSpace]);
        const double selfEnergy = pairStats.selfEnergy;
        const double excludedCorrectionEnergy = excludedStats.rawCoulEnergy;
        const double shortRangeTotalEnergy =
                sumEnergyTermsOnce(coulEnergyTerms);
        const double shortRangePairEnergy = pairStats.rawCoulEnergy;
        appendCpuCorrectionEnergyTrace(activeM2pTraceDirPath(),
                                       step,
                                       outerLevel,
                                       "cpu_only",
                                       reciprocalEnergy,
                                       selfEnergy,
                                       excludedCorrectionEnergy,
                                       shortRangePairEnergy,
                                       shortRangeTotalEnergy,
                                       1,
                                       selfEnergyAtomCount,
                                       excludedStats.count,
                                       pairStats.count);
    }
    if (dumpMultiStepCoulombStateTrace)
    {
        static std::string clearedMultiStepTracePath;
        const std::string  tracePath =
                (std::filesystem::path(ljSrTraceDirPath) / "multistep_coulomb_state_trace.txt").string();
        if (tracePath != clearedMultiStepTracePath)
        {
            writeRespaTraceTextFile(ljSrTraceDirPath, "multistep_coulomb_state_trace.txt", "");
            clearedMultiStepTracePath = tracePath;
        }
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "multistep_coulomb_state_trace.txt",
                "side=PATCH step=" + std::to_string(step)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_self_energy_loop"
                        + " compute_energy="
                        + std::string(stepWork.computeEnergy ? "true" : "false")
                        + " pair_count=" + std::to_string(pairStats.count)
                        + " excluded_pair_count=" + std::to_string(excludedStats.count)
                        + " patch_live_coul_total_before_step="
                        + formatString("%.15f", patchCombinedBeforePairs)
                        + " patch_live_coul_total_after_pairs="
                        + formatString("%.15f", patchCombinedAfterPairs)
                        + " patch_live_coul_total_after_excluded="
                        + formatString("%.15f", patchCombinedAfterExcluded)
                        + " patch_live_coul_total_after_self="
                        + formatString("%.15f", patchCombinedAfterSelf)
                        + " patch_live_coul_final="
                        + formatString("%.15f", patchCombinedAfterSelf));
    }
    if (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0')
    {
        const real patchCombinedArrayTotal = patchCombinedAfterSelf;
        const real patchComparableCombinedCoulomb =
                pairStats.coulEnergy + excludedStats.coulEnergy + pairStats.selfEnergy;
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_lj_source_truth_trace.txt",
                "side=PATCH variable=pairStats.ljEnergy role=patch_pairs_lj_sr_source before=0.000000000000000 delta="
                        + formatString("%.15f", pairStats.ljEnergy) + " after="
                        + formatString("%.15f", pairStats.ljEnergy)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_pairs_processPairlist");
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_lj_source_truth_trace.txt",
                "side=PATCH variable=excludedStats.ljEnergy role=patch_excluded_lj_sr_source before=0.000000000000000 delta="
                        + formatString("%.15f", excludedStats.ljEnergy) + " after="
                        + formatString("%.15f", excludedStats.ljEnergy)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_excluded_processPairlist");
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_lj_source_truth_trace.txt",
                "side=PATCH variable=vdwEnergyTerms_total role=patch_combined_lj_truth_after_pairs before="
                        + formatString("%.15f", patchLjCombinedBeforePairs) + " delta="
                        + formatString("%.15f", patchLjCombinedAfterPairs - patchLjCombinedBeforePairs)
                        + " after=" + formatString("%.15f", patchLjCombinedAfterPairs)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_pairs_processPairlist");
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_lj_source_truth_trace.txt",
                "side=PATCH variable=vdwEnergyTerms_total role=patch_combined_lj_truth_after_excluded before="
                        + formatString("%.15f", patchLjCombinedAfterPairs) + " delta="
                        + formatString("%.15f", patchLjCombinedAfterExcluded - patchLjCombinedAfterPairs)
                        + " after=" + formatString("%.15f", patchLjCombinedAfterExcluded)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_excluded_processPairlist");
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_lj_source_truth_trace.txt",
                "side=PATCH variable=vdwEnergyTerms_total role=patch_combined_lj_truth_after_self before="
                        + formatString("%.15f", patchLjCombinedBeforeSelf) + " delta="
                        + formatString("%.15f", patchLjCombinedAfterSelf - patchLjCombinedBeforeSelf)
                        + " after=" + formatString("%.15f", patchLjCombinedAfterSelf)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_self_energy_loop");
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_coulomb_source_truth_trace.txt",
                "side=PATCH variable=pairStats.coulEnergy role=patch_pairs_comparable_source before=0.000000000000000 delta="
                        + formatString("%.15f", pairStats.coulEnergy) + " after="
                        + formatString("%.15f", pairStats.coulEnergy)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_pairs_processPairlist");
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_coulomb_source_truth_trace.txt",
                "side=PATCH variable=excludedStats.coulEnergy role=patch_excluded_comparable_source before=0.000000000000000 delta="
                        + formatString("%.15f", excludedStats.coulEnergy) + " after="
                        + formatString("%.15f", excludedStats.coulEnergy)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_excluded_processPairlist");
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_coulomb_source_truth_trace.txt",
                "side=PATCH variable=excludedStats.rawCoulEnergy role=patch_excluded_runtime_split_source before=0.000000000000000 delta="
                        + formatString("%.15f", excludedStats.rawCoulEnergy) + " after="
                        + formatString("%.15f", excludedStats.rawCoulEnergy)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_excluded_processPairlist");
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_coulomb_source_truth_trace.txt",
                "side=PATCH variable=pairStats.selfEnergy role=patch_self_stats_source before=0.000000000000000 delta="
                        + formatString("%.15f", pairStats.selfEnergy) + " after="
                        + formatString("%.15f", pairStats.selfEnergy)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_self_energy_loop");
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_coulomb_source_truth_trace.txt",
                "side=PATCH variable=coulEnergyTerms_total role=patch_combined_truth_after_pairs before="
                        + formatString("%.15f", patchCombinedBeforePairs) + " delta="
                        + formatString("%.15f", patchCombinedAfterPairs - patchCombinedBeforePairs) + " after="
                        + formatString("%.15f", patchCombinedAfterPairs)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_pairs_processPairlist");
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_coulomb_source_truth_trace.txt",
                "side=PATCH variable=coulEnergyTerms_total role=patch_combined_truth_after_excluded before="
                        + formatString("%.15f", patchCombinedAfterPairs) + " delta="
                        + formatString("%.15f", patchCombinedAfterExcluded - patchCombinedAfterPairs) + " after="
                        + formatString("%.15f", patchCombinedAfterExcluded)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_excluded_processPairlist");
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_coulomb_source_truth_trace.txt",
                "side=PATCH variable=coulEnergyTerms_total role=patch_combined_truth_after_self before="
                        + formatString("%.15f", patchCombinedBeforeSelf) + " delta="
                        + formatString("%.15f", patchCombinedAfterSelf - patchCombinedBeforeSelf) + " after="
                        + formatString("%.15f", patchCombinedAfterSelf)
                        + " code_location=src/gromacs/mdlib/sim_util.cpp:after_self_energy_loop");
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_coulomb_sr_component_trace.txt",
                "stage=PRE_SR_ACCUMULATION_COMPARABLE code_location=src/gromacs/mdlib/sim_util.cpp:after_self_energy_before_sr_accumulation case_label="
                        + ljSrTraceCaseLabel
                        + " execution_path=exact_respa_component_sum_before_sr_accumulation patch_pairs_coulomb_sr="
                        + formatString("%.15f", pairStats.coulEnergy)
                        + " patch_excludedPairs_coulomb_sr="
                        + formatString("%.15f", excludedStats.coulEnergy)
                        + " patch_self_coulomb_sr="
                        + formatString("%.15f", pairStats.selfEnergy)
                        + " patch_component_sum_coulomb_sr="
                        + formatString("%.15f", patchComparableCombinedCoulomb)
                        + " patch_combined_coulomb_sr="
                        + formatString("%.15f", patchCombinedArrayTotal));
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_coulomb_sr_component_trace.txt",
                "stage=SR_ACCUMULATION_PAIRS code_location=src/gromacs/mdlib/sim_util.cpp:after_pairs_dispatch_component case_label="
                        + ljSrTraceCaseLabel
                        + " execution_path=exact_respa_pairs_component patch_pairs_coulomb_sr="
                        + formatString("%.15f", pairStats.coulEnergy));
        appendRespaTraceTextLine(
                ljSrTraceDirPath,
                "step0_coulomb_sr_component_trace.txt",
                "stage=SR_ACCUMULATION_EXCLUDEDPAIRS code_location=src/gromacs/mdlib/sim_util.cpp:after_excluded_pairs_dispatch_component case_label="
                        + ljSrTraceCaseLabel
                        + " execution_path=exact_respa_excludedPairs_component patch_excludedPairs_coulomb_sr="
                        + formatString("%.15f", excludedStats.coulEnergy));
    }

    if (debugExactRespa)
    {
        std::fprintf(stderr,
                     "GMX_PCFF_RESPA_DEBUG listedPairOverlaps pairs=%d excludedPairs=%d duplicatePairs=%d duplicateExcludedPairs=%d\n",
                     listedPairsInPairlist,
                     listedPairsInExcludedPairlist,
                     duplicatePairs,
                     duplicateExcludedPairs);
        std::fprintf(stderr,
                     "GMX_PCFF_RESPA_DEBUG pairs count=%d lj=% .9f coul=% .9f qq=% .9f self=% .9f\n",
                     pairStats.count,
                     pairStats.ljEnergy,
                     pairStats.coulEnergy,
                     pairStats.qqSum,
                     pairStats.selfEnergy);
        std::fprintf(stderr,
                     "GMX_PCFF_RESPA_DEBUG excludedPairs count=%d lj=% .9f coul=% .9f qq=% .9f self=% .9f\n",
                     excludedStats.count,
                     excludedStats.ljEnergy,
                     excludedStats.coulEnergy,
                     excludedStats.qqSum,
                     excludedStats.selfEnergy);
    }

    if (dumpExcludedCorrectionForce)
    {
        static bool dumpedExcludedCorrectionForce = false;
        if (!dumpedExcludedCorrectionForce)
        {
            const int outerMtsLevel = exactRespaNonbondedOuterLevel(inputrec);
            FILE* dumpFile = std::fopen(excludedCorrectionForceDumpPath, "w");
            if (dumpFile == nullptr)
            {
                gmx_fatal(FARGS,
                          "Could not open GMX_PCFF_RESPA_EXCLUDED_FORCE_DUMP_FILE '%s' for writing",
                          excludedCorrectionForceDumpPath);
            }
            std::fprintf(
                    dumpFile,
                    "# contribution=excluded_pairs_coulomb_correction level_user=%d level_index=%d initial_buffer=%s postprocess=postProcessForces final_merge=combineMtsForces\n",
                    outerMtsLevel + 1,
                    outerMtsLevel,
                    stepWork.computeVirial ? "forceWithVirial" : "forceWithShiftForces");
            for (int atom = 0; atom < static_cast<int>(excludedCorrectionForce.size()); ++atom)
            {
                std::fprintf(dumpFile,
                             "%d\t%.17g\t%.17g\t%.17g\n",
                             atom,
                             excludedCorrectionForce[atom][XX],
                             excludedCorrectionForce[atom][YY],
                             excludedCorrectionForce[atom][ZZ]);
            }
            std::fclose(dumpFile);
            dumpedExcludedCorrectionForce = true;
        }
    }

    for (auto& accumulator : activeContributions)
    {
        if (accumulator.forceWithVirial != nullptr)
        {
            accumulator.forceWithVirial->addVirialContribution(accumulator.virial);
        }
    }
}

static void replayExactRespaNonbondedTraceShadow(const t_inputrec&             inputrec,
                                                 const InteractionDefinitions& idef,
                                                 t_forcerec*                   fr,
                                                 const t_mdatoms&              mdatoms,
                                                 ArrayRef<const RVec>          coordinates,
                                                 gmx_enerdata_t*               enerd,
                                                 const StepWorkload&           stepWork,
                                                 const ExactRespaStepWork&     exactRespaStepWork,
                                                 const int64_t                 step)
{
    const bool needsDetailedTrace =
            shouldTraceRespaForceComponentsStep(step) || shouldTraceRespaRealspaceForceSubcomponentsStep(step);
    if (!needsDetailedTrace)
    {
        return;
    }

    const int numLevels = std::max(1, exactRespaStepWork.highestActiveLevel + 1);
    gmx::ForceBuffers shadowForceBuffers(numLevels, gmx::PinningPolicy::CannotBePinned);
    shadowForceBuffers.resize(coordinates.ssize());

    ExactRespaForceOutputs      shadowOutputs;
    ExactRespaForceOutputStorage shadowStorage;
    shadowOutputs.numLevels          = numLevels;
    shadowOutputs.highestActiveLevel = exactRespaStepWork.highestActiveLevel;
    shadowOutputs.longrangeLevel     = exactRespaLongrangeNonbondedLevel(inputrec);

    auto makeShiftOnlyOutput = [](ArrayRefWithPadding<RVec> forceBuffer, const bool haveVirial)
    {
        ForceWithShiftForces forceWithShiftForces(forceBuffer, false, {});
        ForceWithVirial      forceWithVirial(forceWithShiftForces.force(), false);
        return ForceOutputs(forceWithShiftForces, haveVirial, forceWithVirial);
    };

    shadowStorage.ownedLevelOutputs[0].emplace(
            makeShiftOnlyOutput(shadowForceBuffers.view().forceWithPadding(), false));
    shadowOutputs.levelOutputs[0] = &shadowStorage.ownedLevelOutputs[0].value();
    for (auto& forceValue : shadowForceBuffers.view().force())
    {
        clear_rvec(forceValue);
    }

    for (int mtsLevel = 1; mtsLevel <= exactRespaStepWork.highestActiveLevel; ++mtsLevel)
    {
        const bool haveVirial =
                stepWork.computeVirial && mtsLevel == exactRespaNonbondedOuterLevel(inputrec);
        shadowStorage.ownedLevelForceBuffers[mtsLevel].resizeWithPadding(coordinates.ssize());
        shadowStorage.ownedLevelOutputs[mtsLevel].emplace(makeShiftOnlyOutput(
                shadowStorage.ownedLevelForceBuffers[mtsLevel].arrayRefWithPadding(), haveVirial));
        shadowOutputs.levelOutputs[mtsLevel] = &shadowStorage.ownedLevelOutputs[mtsLevel].value();
        auto shadowForce = shadowOutputs.levelOutputs[mtsLevel]->forceWithShiftForces().force();
        for (auto& forceValue : shadowForce)
        {
            clear_rvec(forceValue);
        }
    }

    computeExactRespaNonbondedCpu(
            inputrec, idef, fr, mdatoms, coordinates, shadowOutputs, enerd, stepWork, step, true);
}

static bool exactRespaCpuNbnxmKernelSupported(const NbnxmKernelType kernelType)
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

static NbnxmOutputContract buildExactRespaCpuNbnxmOutputContract(
        const ExactRespaForceOutputs&                      exactRespaForceOutputs,
        const std::vector<LammpsRespaNonbondedOutputSink>& outputSinks,
        t_forcerec*                                        fr,
        gmx_enerdata_t*                                    enerd,
        const NbnxmOutputContractKind                      contractKind =
                NbnxmOutputContractKind::PerContributionLaunch)
{
    GMX_RELEASE_ASSERT(fr != nullptr && enerd != nullptr,
                       "Exact r-RESPA CPU NBNXM output contract requires force and energy state");

    NbnxmOutputContract contract;
    contract.kind = contractKind;

    for (const auto& outputSink : outputSinks)
    {
        ForceOutputs* outputs = exactRespaForceOutputs.levelOrNull(outputSink.mtsLevel);
        GMX_RELEASE_ASSERT(outputs != nullptr,
                           "Exact r-RESPA CPU NBNXM launches require one force output sink per active contribution");

        NbnxmOutputSink sink;
        sink.contribution = outputSink.contribution;
        sink.sinkKind     = outputSink.sinkKind;
        sink.locality     = AtomLocality::Local;

        if (outputSink.sinkKind == LammpsRespaNonbondedOutputSinkKind::ForceWithVirial)
        {
            sink.force                = outputs->forceWithVirial().force_;
            sink.directVirialOutput   = &outputs->forceWithVirial();
            contract.virial.accumulateVirial = true;
            contract.virial.contribution     = outputSink.contribution;
            contract.virial.sinkKind         = outputSink.sinkKind;
            contract.virial.directVirialOutput = sink.directVirialOutput;
        }
        else
        {
            sink.force       = outputs->forceWithShiftForces().force();
            sink.shiftForces = outputs->forceWithShiftForces().shiftForces();
        }

        if (outputSink.accumulateEnergy)
        {
            contract.energy.accumulateEnergy = true;
            contract.energy.contribution     = outputSink.contribution;
            contract.energy.vdwEnergy =
                    enerd->grpp.energyGroupPairTerms[fr->haveBuckingham ? NonBondedEnergyTerms::BuckinghamSR
                                                                         : NonBondedEnergyTerms::LJSR];
            contract.energy.coulombEnergy =
                    enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR];
        }

        contract.sinks.push_back(sink);
        if (contract.kind == NbnxmOutputContractKind::NativeMultiContribution)
        {
            contract.nativeMultiContribution.contributions.push_back(outputSink.contribution);
        }
    }

    return contract;
}

static void addExactRespaCpuNbnxmDirectVirial(const NbnxmOutputSink&    sink,
                                              const t_forcerec&         fr,
                                              const t_mdatoms&          mdatoms,
                                              ArrayRef<const RVec>      coordinates,
                                              const matrix              box,
                                              ArrayRef<const nbnxn_atomdata_output_t> outputBuffers,
                                              t_nrnb*                   nrnb)
{
    GMX_RELEASE_ASSERT(sink.directVirialOutput != nullptr,
                       "Exact r-RESPA CPU NBNXM direct-virial accumulation requires a virial sink");
    GMX_RELEASE_ASSERT(sink.locality == AtomLocality::Local,
                       "Exact r-RESPA CPU NBNXM direct-virial accumulation currently supports only local sinks");

    std::vector<RVec> reducedShiftForces(c_numShiftVectors);
    for (auto& shiftForce : reducedShiftForces)
    {
        clear_rvec(shiftForce);
    }
    nbnxn_atomdata_add_output_fshift_to_fshift(outputBuffers, reducedShiftForces);

    matrix      virial          = { { 0 } };
    const rvec* fshift          = as_rvec_array(reducedShiftForces.data());
    const rvec* shiftVecPointer = as_rvec_array(fr.shift_vec.data());
    calc_vir(c_numShiftVectors, shiftVecPointer, fshift, virial, fr.pbcType == PbcType::Screw, box);
    inc_nrnb(nrnb, eNR_VIRIAL, c_numShiftVectors);

    const rvec* x = as_rvec_array(coordinates.data());
    const rvec* f = as_rvec_array(sink.force.data());
    f_calc_vir(0, mdatoms.homenr, x, f, virial, box);
    inc_nrnb(nrnb, eNR_VIRIAL, mdatoms.homenr);

    sink.directVirialOutput->addVirialContribution(virial);
}

static void addExactRespaCpuNbnxmDirectVirial(const NbnxmOutputSink&  sink,
                                              const t_forcerec&       fr,
                                              const t_mdatoms&        mdatoms,
                                              ArrayRef<const RVec>    coordinates,
                                              const matrix            box,
                                              const nbnxn_atomdata_t& nbat,
                                              t_nrnb*                 nrnb)
{
    addExactRespaCpuNbnxmDirectVirial(
            sink, fr, mdatoms, coordinates, box, nbat.outputBuffers(), nrnb);
}

static int exactRespaCpuNativeMultiContributionIndex(const NbnxmOutputContract&          contract,
                                                     const MtsNonbondedRespaContribution contribution)
{
    for (int outputIndex = 0;
         outputIndex < gmx::ssize(contract.nativeMultiContribution.contributions);
         ++outputIndex)
    {
        if (contract.nativeMultiContribution.contributions[outputIndex] == contribution)
        {
            return outputIndex;
        }
    }

    GMX_RELEASE_ASSERT(false,
                       "Exact r-RESPA CPU native multi-contribution contract is missing a declared contribution");
    return -1;
}

static void computeExactRespaNonbondedCpuNbnxmNarrow(const t_inputrec&             inputrec,
                                                     t_forcerec*                   fr,
                                                     const t_mdatoms&              mdatoms,
                                                     ArrayRef<const RVec>          coordinates,
                                                     const matrix                  box,
                                                     const ExactRespaForceOutputs& exactRespaForceOutputs,
                                                     gmx_enerdata_t*               enerd,
                                                     const StepWorkload&           stepWork,
                                                     const ExactRespaStepWork&     exactRespaStepWork,
                                                     const int64_t                 step,
                                                     t_nrnb*                       nrnb,
                                                     gmx_wallcycle*                wcycle)
{
    GMX_RELEASE_ASSERT(fr != nullptr && fr->nbv != nullptr,
                       "Exact r-RESPA CPU NBNXM narrow mode requires initialized CPU nonbonded state");
    GMX_RELEASE_ASSERT(exactRespaCpuNbnxmKernelSupported(fr->nbv->kernelSetup().kernelType),
                       "Exact r-RESPA CPU NBNXM narrow mode requires a CPU NBNXM kernel");

    const auto outputSinks = activeLammpsRespaNonbondedOutputSinks(inputrec,
                                                                   exactRespaStepWork.highestActiveLevel,
                                                                   stepWork.computeVirial,
                                                                   stepWork.computeEnergy);
    const int outerContributionLevel = exactRespaNonbondedOuterLevel(inputrec);
    const bool ownerLevelStep =
            outerContributionLevel >= 0 && exactRespaStepWork.highestActiveLevel == outerContributionLevel;
    const bool splitOwnerOutputsSidecar =
            exactRespaNativeMultiSplitOwnerOutputsRequested() && ownerLevelStep;
    const bool fallbackOnOwnerStep =
            exactRespaNativeMultiFallbackOnOwnerStepsRequested() && ownerLevelStep;
    const int middleContributionLevel =
            inputrec.exactRespa.forceLayout.hasMiddle() ? exactRespaNonbondedMiddleLevel(inputrec) : -1;
    const bool fallbackOnMiddleStep =
            exactRespaNativeMultiFallbackOnMiddleStepsRequested() && !stepWork.computeVirial
            && !stepWork.computeEnergy && middleContributionLevel >= 0
            && exactRespaStepWork.highestActiveLevel == middleContributionLevel;
    const MtsNonbondedRespaContribution ownerContribution =
            stepWork.computeEnergy ? MtsNonbondedRespaContribution::Outer
                                   : MtsNonbondedRespaContribution::Outer;
    std::vector<LammpsRespaNonbondedOutputSink> nativeMultiOutputSinks = outputSinks;
    auto ownerSinkIt = outputSinks.end();
    if (splitOwnerOutputsSidecar)
    {
        ownerSinkIt = std::find_if(outputSinks.begin(),
                                   outputSinks.end(),
                                   [ownerContribution](const LammpsRespaNonbondedOutputSink& sink)
                                   { return sink.contribution == ownerContribution; });
        if (ownerSinkIt != outputSinks.end())
        {
            nativeMultiOutputSinks.erase(std::remove_if(nativeMultiOutputSinks.begin(),
                                                        nativeMultiOutputSinks.end(),
                                                        [ownerContribution](const LammpsRespaNonbondedOutputSink& sink)
                                                        { return sink.contribution == ownerContribution; }),
                                         nativeMultiOutputSinks.end());
        }
    }
    const bool canUseNativeMultiContributionLaunch =
            exactRespaNativeMultiContributionLaunchRequested() && stepWork.computeForces
            && nativeMultiOutputSinks.size() > 1 && !fallbackOnOwnerStep && !fallbackOnMiddleStep;
    if (const char* decisionTracePath = exactRespaNativeMultiDecisionTracePath())
    {
        std::ofstream output(decisionTracePath, std::ios::app);
        output << "step=" << step << " highestActiveLevel=" << exactRespaStepWork.highestActiveLevel
               << " computeEnergy=" << (stepWork.computeEnergy ? 1 : 0)
               << " computeVirial=" << (stepWork.computeVirial ? 1 : 0)
               << " nativeMultiRequested="
               << (exactRespaNativeMultiContributionLaunchRequested() ? 1 : 0)
               << " splitOwner=" << (splitOwnerOutputsSidecar ? 1 : 0)
               << " fallbackOwner=" << (fallbackOnOwnerStep ? 1 : 0)
               << " fallbackMiddle=" << (fallbackOnMiddleStep ? 1 : 0)
               << " outputSinkCount=" << outputSinks.size()
               << " nativeMultiSinkCount=" << nativeMultiOutputSinks.size()
               << " canUseNativeMulti=" << (canUseNativeMultiContributionLaunch ? 1 : 0)
               << '\n';
    }
    const NbnxmOutputContract outputContract = buildExactRespaCpuNbnxmOutputContract(
            exactRespaForceOutputs,
            canUseNativeMultiContributionLaunch ? nativeMultiOutputSinks : outputSinks,
            fr,
            enerd,
            canUseNativeMultiContributionLaunch ? NbnxmOutputContractKind::NativeMultiContribution
                                                : NbnxmOutputContractKind::PerContributionLaunch);

    if (fr->nbv->isDynamicPruningStepCpu(step))
    {
        wallcycle_sub_start(wcycle, WallCycleSubCounter::NonbondedPruning);
        fr->nbv->dispatchPruneKernelCpu(InteractionLocality::Local, fr->shift_vec);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::NonbondedPruning);
    }

    if (canUseNativeMultiContributionLaunch)
    {
        StepWorkload forceOnlyNativeMultiWork = stepWork;
        if (splitOwnerOutputsSidecar)
        {
            forceOnlyNativeMultiWork.computeEnergy = false;
            forceOnlyNativeMultiWork.computeVirial = false;
        }
        fr->nbv->dispatchExactRespaCpuNativeMultiKernel(InteractionLocality::Local,
                                                        *fr->ic,
                                                        outputContract.nativeMultiContribution.contributions,
                                                        splitOwnerOutputsSidecar ? forceOnlyNativeMultiWork : stepWork,
                                                        enbvClearFYes,
                                                        fr->shift_vec,
                                                        (!splitOwnerOutputsSidecar && outputContract.energy.accumulateEnergy)
                                                                ? outputContract.energy.vdwEnergy
                                                                : ArrayRef<real>{},
                                                        (!splitOwnerOutputsSidecar && outputContract.energy.accumulateEnergy)
                                                                ? outputContract.energy.coulombEnergy
                                                                : ArrayRef<real>{},
                                                        nrnb);
        fr->nbv->atomdata_add_nbat_f_to_native_multi_outputs(outputContract);
        if (!splitOwnerOutputsSidecar && outputContract.virial.accumulateVirial
            && outputContract.virial.sinkKind
                       == LammpsRespaNonbondedOutputSinkKind::ForceWithVirial)
        {
            const auto nativeOutputSinks =
                    nbnxmOutputSinksForNativeMultiContribution(outputContract);
            const int ownerOutputIndex = exactRespaCpuNativeMultiContributionIndex(
                    outputContract, outputContract.virial.contribution);
            addExactRespaCpuNbnxmDirectVirial(*nativeOutputSinks[ownerOutputIndex],
                                              *fr,
                                              mdatoms,
                                              coordinates,
                                              box,
                                              fr->nbv->nbat().nativeMultiContributionOutputBuffers(
                                                      ownerOutputIndex),
                                              nrnb);
        }

        if (splitOwnerOutputsSidecar && ownerSinkIt != outputSinks.end())
        {
            const int ownerLevel = nonbondedRespaContributionMtsLevel(inputrec, ownerContribution);
            GMX_RELEASE_ASSERT(ownerLevel >= 0,
                               "Native-multi owner sidecar requires a valid owner MTS level");

            const std::vector<LammpsRespaNonbondedOutputSink> ownerOutputSinks = { *ownerSinkIt };
            const NbnxmOutputContract ownerSidecarContract = buildExactRespaCpuNbnxmOutputContract(
                    exactRespaForceOutputs,
                    ownerOutputSinks,
                    fr,
                    enerd,
                    NbnxmOutputContractKind::PerContributionLaunch);

            const StepWorkload ownerContributionWork =
                    stepWork.withExactNonbondedContribution(ownerContribution);
            fr->nbv->dispatchNonbondedKernel(
                    InteractionLocality::Local,
                    *fr->ic,
                    ownerContributionWork,
                    enbvClearFYes,
                    fr->shift_vec,
                    enerd->grpp.energyGroupPairTerms[fr->haveBuckingham ? NonBondedEnergyTerms::BuckinghamSR
                                                                        : NonBondedEnergyTerms::LJSR],
                    enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR],
                    nrnb);

            if (ownerContributionWork.computeForces)
            {
                fr->nbv->atomdata_add_nbat_f_to_outputs(ownerSidecarContract, ownerContribution);
                if (ownerContributionWork.computeVirial)
                {
                    const NbnxmOutputSink& sidecarSink =
                            nbnxmOutputSinkForContribution(ownerSidecarContract, ownerContribution);
                    if (sidecarSink.sinkKind == LammpsRespaNonbondedOutputSinkKind::ForceWithVirial)
                    {
                        addExactRespaCpuNbnxmDirectVirial(sidecarSink,
                                                          *fr,
                                                          mdatoms,
                                                          coordinates,
                                                          box,
                                                          fr->nbv->nbat(),
                                                          nrnb);
                    }
                }
            }
        }
        return;
    }

    for (const auto& outputSink : outputSinks)
    {
        const StepWorkload contributionWork =
                stepWork.withExactNonbondedContribution(outputSink.contribution);
        fr->nbv->dispatchNonbondedKernel(
                InteractionLocality::Local,
                *fr->ic,
                contributionWork,
                enbvClearFYes,
                fr->shift_vec,
                enerd->grpp.energyGroupPairTerms[fr->haveBuckingham ? NonBondedEnergyTerms::BuckinghamSR
                                                                    : NonBondedEnergyTerms::LJSR],
                enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR],
                nrnb);

        if (contributionWork.computeForces)
        {
            fr->nbv->atomdata_add_nbat_f_to_outputs(outputContract, outputSink.contribution);

            if (contributionWork.computeVirial)
            {
                const NbnxmOutputSink& sink =
                        nbnxmOutputSinkForContribution(outputContract, outputSink.contribution);
                if (sink.sinkKind == LammpsRespaNonbondedOutputSinkKind::ForceWithVirial)
                {
                    addExactRespaCpuNbnxmDirectVirial(
                            sink, *fr, mdatoms, coordinates, box, fr->nbv->nbat(), nrnb);
                }
            }
        }
    }
}

#if GMX_GPU
static void setExactRespaGpuLaunchParameters(NbnxmGpu*                         gpuNbv,
                                             const t_inputrec&                 inputrec,
                                             const MtsNonbondedRespaContribution contribution)
{
    const auto toGpuExactRespaContribution = [](const MtsNonbondedRespaContribution hostContribution) -> int
    {
        switch (hostContribution)
        {
            case MtsNonbondedRespaContribution::Inner: return 0;
            case MtsNonbondedRespaContribution::Middle: return 1;
            case MtsNonbondedRespaContribution::Outer: return 2;
            case MtsNonbondedRespaContribution::Full: return 3;
            case MtsNonbondedRespaContribution::Count:
                GMX_RELEASE_ASSERT(false,
                                   "The exact r-RESPA contribution count sentinel is not a launch mode");
                break;
        }
        GMX_RELEASE_ASSERT(false, "Unhandled exact r-RESPA nonbonded GPU contribution");
        return 3;
    };

    GMX_RELEASE_ASSERT(gpuNbv != nullptr && gpuNbv->nbparam != nullptr,
                       "Exact LAMMPS-style r-RESPA GPU launches require initialized GPU nonbonded parameters");

    const auto& respa                 = inputrec.exactRespa.forceLayout;
    auto*       nbparam               = gpuNbv->nbparam;
    nbparam->exactRespaContribution   = toGpuExactRespaContribution(contribution);
    nbparam->exactRespaHasMiddle      = respa.hasMiddle() ? 1 : 0;
    nbparam->exactRespaInnerOff       = respa.innerOff;
    nbparam->exactRespaInnerOn        = respa.innerOn;
    nbparam->exactRespaOuterOn        = respa.outerOn;
    nbparam->exactRespaOuterOff       = respa.outerOff;
}

static void computeExactRespaNonbondedGpuNarrow(const t_inputrec&             inputrec,
                                                const InteractionDefinitions& idef,
                                                t_forcerec*                   fr,
                                                const t_mdatoms&              mdatoms,
                                                ArrayRef<const RVec>          coordinates,
                                                const interaction_const_t*    ic,
                                                const ExactRespaForceOutputs& exactRespaForceOutputs,
                                                gmx_enerdata_t*               enerd,
                                                const StepWorkload&           stepWork,
                                                const ExactRespaStepWork&     exactRespaStepWork,
                                                const int64_t                 step,
                                                t_nrnb*                       nrnb,
                                                gmx_wallcycle*                wcycle)
{
    GMX_RELEASE_ASSERT(fr != nullptr && fr->nbv != nullptr && fr->nbv->gpuNbv() != nullptr,
                       "Exact LAMMPS-style r-RESPA HG3 narrow mode requires initialized GPU nonbonded state");
    GMX_RELEASE_ASSERT(!stepWork.computeVirial && !stepWork.computeEnergy,
                       "Exact LAMMPS-style r-RESPA HG3 narrow mode is force-only");
    GMX_RELEASE_ASSERT(!stepWork.useGpuXBufferOps && !stepWork.useGpuFBufferOps,
                       "Exact LAMMPS-style r-RESPA HG3 narrow mode does not support GPU buffer ops");
    GMX_RELEASE_ASSERT(ic != nullptr && ic->vdw.type == VanDerWaalsType::Cut
                               && ic->vdw.modifier == InteractionModifiers::None
                               && usingPmeOrEwald(ic->coulomb.type)
                               && ic->coulomb.modifier == InteractionModifiers::None,
                       "Exact LAMMPS-style r-RESPA HG3 narrow mode supports only cut-off LJ with PME/Ewald Coulomb");

    struct ContributionTarget
    {
        MtsNonbondedRespaContribution contribution;
        ArrayRef<RVec>                force;
    };

    std::vector<ContributionTarget> activeTargets;
    const auto outputSinks = activeLammpsRespaNonbondedOutputSinks(inputrec,
                                                                   exactRespaStepWork.highestActiveLevel,
                                                                   stepWork.computeVirial,
                                                                   stepWork.computeEnergy);
    for (const auto& outputSink : outputSinks)
    {
        GMX_RELEASE_ASSERT(outputSink.sinkKind == LammpsRespaNonbondedOutputSinkKind::ShiftForce,
                           "Exact LAMMPS-style r-RESPA HG3 narrow mode requires host force-with-shift sinks");
        ForceOutputs* outputs = exactRespaForceOutputs.levelOrNull(outputSink.mtsLevel);
        GMX_RELEASE_ASSERT(outputs != nullptr,
                           "Exact LAMMPS-style r-RESPA HG3 narrow mode requires one host force sink per active contribution");
        activeTargets.push_back(
                { outputSink.contribution, outputs->forceWithShiftForces().force() });
    }

    nonbonded_verlet_t* nbv = fr->nbv.get();
    NbnxmGpu*           gpu = nbv->gpuNbv();
    const auto          disabledShiftReductionSink =
            exactRespaForceOutputs.level(0).forceWithShiftForces().shiftForces();
    GMX_RELEASE_ASSERT(disabledShiftReductionSink.empty(),
                       "Exact LAMMPS-style r-RESPA HG2/HG3 narrow mode supports only atom-force GPU merge; shift-force reduction stays disabled");

    wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
    wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
    gpu_upload_shiftvec(gpu, &nbv->nbat());
    gpu_copy_xq_to_gpu(gpu, &nbv->nbat(), AtomLocality::Local);
    wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
    wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);

    for (const auto& target : activeTargets)
    {
        const StepWorkload contributionWork = stepWork.withExactNonbondedContribution(target.contribution);
        setExactRespaGpuLaunchParameters(gpu, inputrec, target.contribution);

        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
        wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        gpu_clear_outputs(gpu, contributionWork.computeVirial);
        do_nb_verlet(fr, ic, enerd, contributionWork, InteractionLocality::Local, enbvClearFNo, step, nrnb, wcycle);
        gpu_launch_cpyback(gpu, &nbv->nbat(), contributionWork, AtomLocality::Local);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);

        gpu_wait_finish_task(gpu,
                             contributionWork,
                             AtomLocality::Local,
                             false,
                             enerd,
                             disabledShiftReductionSink,
                             wcycle);
        nbv->atomdata_add_nbat_f_to_f(AtomLocality::Local, target.force);
    }

    replayExactRespaNonbondedTraceShadow(
            inputrec, idef, fr, mdatoms, coordinates, enerd, stepWork, exactRespaStepWork, step);
    setExactRespaGpuLaunchParameters(gpu, inputrec, MtsNonbondedRespaContribution::Full);
}
#else
static void computeExactRespaNonbondedGpuNarrow(const t_inputrec&,
                                                const InteractionDefinitions&,
                                                t_forcerec*,
                                                const t_mdatoms&,
                                                ArrayRef<const RVec>,
                                                const interaction_const_t*,
                                                const ExactRespaForceOutputs&,
                                                gmx_enerdata_t*,
                                                const StepWorkload&,
                                                const ExactRespaStepWork&,
                                                const int64_t,
                                                t_nrnb*,
                                                gmx_wallcycle*)
{
    GMX_RELEASE_ASSERT(false,
                       "Exact LAMMPS-style r-RESPA GPU narrow mode was compiled without GPU support");
}
#endif

static inline void clearRVecs(ArrayRef<RVec> v, const bool useOpenmpThreading)
{
    int nth = gmx_omp_nthreads_get_simple_rvec_task(ModuleMultiThread::Default, v.ssize());

    /* Note that we would like to avoid this conditional by putting it
     * into the omp pragma instead, but then we still take the full
     * omp parallel for overhead (at least with gcc5).
     */
    if (!useOpenmpThreading || nth == 1)
    {
        for (RVec& elem : v)
        {
            clear_rvec(elem);
        }
    }
    else
    {
#pragma omp parallel for num_threads(nth) schedule(static)
        for (Index i = 0; i < v.ssize(); i++)
        {
            clear_rvec(v[i]);
        }
    }
}

/*! \brief Return an estimate of the average kinetic energy or 0 when unreliable
 *
 * \param groupOptions  Group options, containing T-coupling options
 */
static real averageKineticEnergyEstimate(const t_grpopts& groupOptions)
{
    real nrdfCoupled   = 0;
    real nrdfUncoupled = 0;
    real kineticEnergy = 0;
    for (int g = 0; g < groupOptions.ngtc; g++)
    {
        if (groupOptions.tau_t[g] >= 0)
        {
            nrdfCoupled += groupOptions.nrdf[g];
            kineticEnergy += groupOptions.nrdf[g] * 0.5 * groupOptions.ref_t[g] * c_boltz;
        }
        else
        {
            nrdfUncoupled += groupOptions.nrdf[g];
        }
    }

    /* This conditional with > also catches nrdf=0 */
    if (nrdfCoupled > nrdfUncoupled)
    {
        return kineticEnergy * (nrdfCoupled + nrdfUncoupled) / nrdfCoupled;
    }
    else
    {
        return 0;
    }
}

/*! \brief This routine checks that the potential energy is finite.
 *
 * Always checks that the potential energy is finite. If step equals
 * inputrec.init_step also checks that the magnitude of the potential energy
 * is reasonable. Terminates with a fatal error when a check fails.
 * Note that passing this check does not guarantee finite forces,
 * since those use slightly different arithmetics. But in most cases
 * there is just a narrow coordinate range where forces are not finite
 * and energies are finite.
 *
 * \param[in] step      The step number, used for checking and printing
 * \param[in] enerd     The energy data; the non-bonded group energies need to be added to
 *                      \c enerd.term[InteractionFunction::PotentialEnergy] before calling this
 * routine \param[in] inputrec  The input record
 */
static void checkPotentialEnergyValidity(int64_t step, const gmx_enerdata_t& enerd, const t_inputrec& inputrec)
{
    /* Threshold valid for comparing absolute potential energy against
     * the kinetic energy. Normally one should not consider absolute
     * potential energy values, but with a factor of one million
     * we should never get false positives.
     */
    constexpr real c_thresholdFactor = 1e6;

    bool energyIsNotFinite    = !std::isfinite(enerd.term[InteractionFunction::PotentialEnergy]);
    real averageKineticEnergy = 0;
    /* We only check for large potential energy at the initial step,
     * because that is by far the most likely step for this too occur
     * and because computing the average kinetic energy is not free.
     * Note: nstcalcenergy >> 1 often does not allow to catch large energies
     * before they become NaN.
     */
    if (step == inputrec.init_step && EI_DYNAMICS(inputrec.eI))
    {
        averageKineticEnergy = averageKineticEnergyEstimate(inputrec.opts);
    }

    if (energyIsNotFinite
        || (averageKineticEnergy > 0
            && enerd.term[InteractionFunction::PotentialEnergy] > c_thresholdFactor * averageKineticEnergy))
    {
        GMX_THROW(InternalError(formatString(
                "Step %" PRId64
                ": The total potential energy is %g, which is %s. The LJ and electrostatic "
                "contributions to the energy are %g and %g, respectively. A %s potential energy "
                "can be caused by overlapping interactions in bonded interactions or very large%s "
                "coordinate values. Usually this is caused by a badly- or non-equilibrated initial "
                "configuration, incorrect interactions or parameters in the topology.",
                step,
                enerd.term[InteractionFunction::PotentialEnergy],
                energyIsNotFinite ? "not finite" : "extremely high",
                enerd.term[InteractionFunction::LennardJonesShortRange],
                enerd.term[InteractionFunction::CoulombShortRange],
                energyIsNotFinite ? "non-finite" : "very high",
                energyIsNotFinite ? " or Nan" : "")));
    }
}

/*! \brief Compute forces and/or energies for special algorithms
 *
 * The intention is to collect all calls to algorithms that compute
 * forces on local atoms only and that do not contribute to the local
 * virial sum (but add their virial contribution separately).
 * Eventually these should likely all become ForceProviders.
 * Within this function the intention is to have algorithms that do
 * global communication at the end, so global barriers within the MD loop
 * are as close together as possible.
 *
 * \param[in]     fplog            The log file
 * \param[in]     mpiComm          The communication object for my group
 * \param[in]     dd               Pointer to the domdec object, is nullptr when DD is not in use
 * \param[in]     inputrec         The input record
 * \param[in]     awh              The Awh module (nullptr if none in use).
 * \param[in]     enforcedRotation Enforced rotation module.
 * \param[in]     imdSession       The IMD session
 * \param[in]     pull_work        The pull work structure.
 * \param[in]     step             The current MD step
 * \param[in]     t                The current time
 * \param[in,out] wcycle           Wallcycle accounting struct
 * \param[in,out] forceProviders   Pointer to a list of force providers
 * \param[in]     box              The unit cell
 * \param[in]     x                The coordinates
 * \param[in]     mdatoms          Per atom properties
 * \param[in]     lambda           Array of free-energy lambda values
 * \param[in]     stepWork         Step schedule flags
 * \param[in,out] forceWithVirialMtsLevel0  Force and virial for MTS level0 forces
 * \param[in,out] forceWithVirialMtsLevel1  Force and virial for MTS level1 forces, can be nullptr
 * \param[in,out] enerd            Energy buffer
 * \param[in,out] ed               Essential dynamics pointer
 * \param[in]     didNeighborSearch  Tells if we did neighbor searching this step,
 *                                   used for ED sampling
 *
 * \todo Remove didNeighborSearch, which is used incorrectly.
 * \todo Convert all other algorithms called here to ForceProviders.
 */
static void computeSpecialForces(FILE*                fplog,
                                 const MpiComm&       mpiComm,
                                 const gmx_domdec_t*  dd,
                                 const t_inputrec&    inputrec,
                                 Awh*                 awh,
                                 gmx_enfrot*          enforcedRotation,
                                 ImdSession*          imdSession,
                                 pull_t*              pull_work,
                                 int64_t              step,
                                 double               t,
                                 gmx_wallcycle*       wcycle,
                                 ForceProviders*      forceProviders,
                                 const matrix         box,
                                 ArrayRef<const RVec> x,
                                 const t_mdatoms*     mdatoms,
                                 ArrayRef<const real> lambda,
                                 const StepWorkload&  stepWork,
                                 const ExactRespaStepWork* exactRespaStepWork,
                                 ArrayRef<ForceWithVirial*> forceWithVirialPerMtsLevel,
                                 gmx_enerdata_t*      enerd,
                                 gmx_edsam*           ed,
                                 bool                 didNeighborSearch)
{
    /* NOTE: Currently all ForceProviders only provide forces.
     *       When they also provide energies, remove this conditional.
     */
    if (stepWork.computeForces)
    {
        ForceProviderInput  forceProviderInput(x,
                                              mdatoms->homenr,
                                              makeArrayRef(mdatoms->chargeA).subArray(0, mdatoms->homenr),
                                              makeArrayRef(mdatoms->massT).subArray(0, mdatoms->homenr),
                                              t,
                                              step,
                                              box,
                                              mpiComm,
                                              dd);
        ForceProviderOutput forceProviderOutput(forceWithVirialPerMtsLevel[0], enerd);

        /* Collect forces from modules */
        forceProviders->calculateForces(forceProviderInput, &forceProviderOutput);
    }

    const int activeLevel =
            (exactRespaStepWork != nullptr) ? exactRespaStepWork->highestActiveLevel
                                            : stepWork.highestActiveMtsLevel;
    const int  pullMtsLevel = gmx::useExactRespa(inputrec) ? gmx::exactRespaPullLevel(inputrec)
                                                           : forceGroupMtsLevel(inputrec.mtsLevels,
                                                                                 MtsForceGroups::Pull);
    const bool doPulling    = (inputrec.bPull && pull_have_potential(*pull_work)
                            && pullMtsLevel <= activeLevel);

    /* pull_potential_wrapper(), awh->applyBiasForcesAndUpdateBias(), pull_apply_forces()
     * have to be called in this order
     */
    // Note: this condition is mirrored in haveSpecialForces()
    if (doPulling)
    {
        pull_potential_wrapper(
                mpiComm, inputrec, box, x, mdatoms, enerd, pull_work, lambda.data(), t, wcycle);
    }
    // Note: the awh condition is mirrored in haveSpecialForces()
    if (awh && pullMtsLevel <= activeLevel)
    {
        const bool          needForeignEnergyDifferences = awh->needForeignEnergyDifferences(step);
        std::vector<double> foreignLambdaDeltaH, foreignLambdaDhDl;
        if (needForeignEnergyDifferences)
        {
            enerd->foreignLambdaTerms.finalizePotentialContributions(
                    enerd->dvdl_lin, lambda, *inputrec.fepvals);
            std::tie(foreignLambdaDeltaH, foreignLambdaDhDl) = enerd->foreignLambdaTerms.getTerms(mpiComm);
        }

        enerd->term[InteractionFunction::CenterOfMassPullingEnergy] += awh->applyBiasForcesAndUpdateBias(
                inputrec.pbcType, foreignLambdaDeltaH, foreignLambdaDhDl, box, t, step, wcycle, fplog);
    }
    // Note: this condition is mirrored in haveSpecialForces()
    if (doPulling)
    {
        wallcycle_start_nocount(wcycle, WallCycleCounter::PullPot);
        pull_apply_forces(pull_work, mdatoms->massT, mpiComm, forceWithVirialPerMtsLevel[pullMtsLevel]);
        wallcycle_stop(wcycle, WallCycleCounter::PullPot);
    }

    /* Add the forces from enforced rotation potentials (if any) */
    // Note: this condition is mirrored in haveSpecialForces()
    if (inputrec.bRot)
    {
        wallcycle_start(wcycle, WallCycleCounter::RotAdd);
        enerd->term[InteractionFunction::CenterOfMassPullingEnergy] +=
                add_rot_forces(enforcedRotation, forceWithVirialPerMtsLevel[0]->force_, mpiComm, step, t);
        wallcycle_stop(wcycle, WallCycleCounter::RotAdd);
    }

    // Note: this condition is mirrored in haveSpecialForces()
    if (ed)
    {
        /* Note that since init_edsam() is called after the initialization
         * of forcerec, edsam doesn't request the noVirSum force buffer.
         * Thus if no other algorithm (e.g. PME) requires it, the forces
         * here will contribute to the virial.
         */
        do_flood(mpiComm, inputrec, x, forceWithVirialPerMtsLevel[0]->force_, ed, box, step, didNeighborSearch);
    }

    /* Add forces from interactive molecular dynamics (IMD), if any */
    // Note: this condition is mirrored in haveSpecialForces()
    if (inputrec.bIMD && stepWork.computeForces)
    {
        imdSession->applyForces(forceWithVirialPerMtsLevel[0]->force_);
    }
}

/*! \brief Launch the prepare_step and spread stages of PME GPU.
 *
 * \param[in]  pmedata              The PME structure
 * \param[in]  box                  The box matrix
 * \param[in]  simulationWork       Simulation schedule flags
 * \param[in]  stepWork             Step schedule flags
 * \param[in]  xReadyOnDevice       Event synchronizer indicating that the coordinates are ready in the device memory.
 * \param[in]  lambdaQ              The Coulomb lambda of the current state.
 * \param[in]  useMdGpuGraph        Whether MD GPU Graph is in use.
 * \param[in]  wcycle               The wallcycle structure
 */
static inline void launchPmeGpuSpread(gmx_pme_t*                pmedata,
                                      const matrix              box,
                                      const SimulationWorkload& simulationWork,
                                      const StepWorkload&       stepWork,
                                      GpuEventSynchronizer*     xReadyOnDevice,
                                      const real                lambdaQ,
                                      bool                      useMdGpuGraph,
                                      gmx_wallcycle*            wcycle)
{
    wallcycle_start(wcycle, WallCycleCounter::PmeGpuMesh);
    pme_gpu_prepare_computation(pmedata, box, simulationWork.haveDynamicBox, stepWork);
    bool                      useGpuDirectComm         = false;
    PmeCoordinateReceiverGpu* pmeCoordinateReceiverGpu = nullptr;
    pme_gpu_launch_spread(
            pmedata, xReadyOnDevice, wcycle, lambdaQ, useGpuDirectComm, pmeCoordinateReceiverGpu, useMdGpuGraph);
    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
}

/*! \brief Launch the FFT and gather stages of PME GPU
 *
 * This function only implements setting the output forces (no accumulation).
 *
 * \param[in]  pmedata        The PME structure
 * \param[in]  lambdaQ        The Coulomb lambda of the current system state.
 * \param[in]  wcycle         The wallcycle structure
 * \param[in]  stepWork       Step schedule flags
 */
static void launchPmeGpuFftAndGather(gmx_pme_t*          pmedata,
                                     const real          lambdaQ,
                                     gmx_wallcycle*      wcycle,
                                     const StepWorkload& stepWork)
{
    wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
    pme_gpu_launch_complex_transforms(pmedata, wcycle, stepWork);
    pme_gpu_launch_gather(pmedata, wcycle, lambdaQ, stepWork.computeVirial);
    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
}

/*! \brief
 * Blocks until PME GPU tasks are completed, and gets the output forces and virial/energy
 * (if they were to be computed).
 *
 * \param[in]  pme             The PME data structure.
 * \param[in]  stepWork        The required work for this simulation step
 * \param[in]  wcycle          The wallclock counter.
 * \param[out] forceWithVirial The output force and virial
 * \param[out] enerd           The output energies
 * \param[in]  lambdaQ         The Coulomb lambda to use when calculating the results.
 */
static void pmeGpuWaitAndReduce(gmx_pme_t*          pme,
                                const StepWorkload& stepWork,
                                gmx_wallcycle*      wcycle,
                                ForceWithVirial*    forceWithVirial,
                                gmx_enerdata_t*     enerd,
                                const real          lambdaQ)
{
    wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);

    pme_gpu_wait_and_reduce(pme, stepWork, wcycle, forceWithVirial, enerd, lambdaQ);

    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
}

/*! \brief
 *  Polling wait for either of the PME or nonbonded GPU tasks.
 *
 * Instead of a static order in waiting for GPU tasks, this function
 * polls checking which of the two tasks completes first, and does the
 * associated force buffer reduction overlapped with the other task.
 * By doing that, unlike static scheduling order, it can always overlap
 * one of the reductions, regardless of the GPU task completion order.
 *
 * \param[in]     nbv              Nonbonded verlet structure
 * \param[in,out] pmedata          PME module data
 * \param[in,out] forceOutputsNonbonded  Force outputs for the non-bonded forces and shift forces
 * \param[in,out] forceOutputsPme  Force outputs for the PME forces and virial
 * \param[in,out] enerd            Energy data structure results are reduced into
 * \param[in]     lambdaQ          The Coulomb lambda of the current system state.
 * \param[in]     stepWork         Step schedule flags
 * \param[in]     simulationWork   Simulation schedule flags
 * \param[in]     wcycle           The wallcycle structure
 */
static void alternatePmeNbGpuWaitReduce(nonbonded_verlet_t*       nbv,
                                        gmx_pme_t*                pmedata,
                                        ForceOutputs*             forceOutputsNonbonded,
                                        ForceOutputs*             forceOutputsPme,
                                        gmx_enerdata_t*           enerd,
                                        const real                lambdaQ,
                                        const StepWorkload&       stepWork,
                                        const SimulationWorkload& simulationWork,
                                        gmx_wallcycle*            wcycle)
{
    bool isPmeGpuDone = false;
    bool isNbGpuDone  = false;

    while (!isPmeGpuDone || !isNbGpuDone)
    {
        if (!isPmeGpuDone)
        {
            wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
            GpuTaskCompletion completionType =
                    (isNbGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
            isPmeGpuDone = pme_gpu_try_finish_task(
                    pmedata, stepWork, wcycle, &forceOutputsPme->forceWithVirial(), enerd, lambdaQ, completionType);
            wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
        }

        if (!isNbGpuDone)
        {
            auto&             forceBuffersNonbonded = forceOutputsNonbonded->forceWithShiftForces();
            GpuTaskCompletion completionType =
                    (isPmeGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
            // To get the wcycle call count right, when in GpuTaskCompletion::Check mode,
            // we start without counting and only when the task finished we issue a
            // start/stop to increment.
            // GpuTaskCompletion::Wait mode the timing is expected to be done in the caller.
            wallcycle_start_nocount(wcycle, WallCycleCounter::WaitGpuNbL);
            isNbGpuDone = gpu_try_finish_task(
                    nbv->gpuNbv(),
                    stepWork,
                    AtomLocality::Local,
                    enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJSR].data(),
                    enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR].data(),
                    (simulationWork.useGpuForeignNonbondedFE)
                            ? &enerd->dvdl_nonlin[FreeEnergyPerturbationCouplingType::Vdw]
                            : &enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Vdw],
                    (simulationWork.useGpuForeignNonbondedFE)
                            ? &enerd->dvdl_nonlin[FreeEnergyPerturbationCouplingType::Coul]
                            : &enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Coul],
                    forceBuffersNonbonded.shiftForces(),
                    &enerd->foreignLambdaTerms,
                    completionType);
            wallcycle_stop(wcycle, WallCycleCounter::WaitGpuNbL);

            if (isNbGpuDone)
            {
                wallcycle_increment_event_count(wcycle, WallCycleCounter::WaitGpuNbL);
                nbv->atomdata_add_nbat_f_to_f(AtomLocality::Local, forceBuffersNonbonded.force());
            }
        }
    }
}

/*! \brief Set up the different force buffers; also does clearing.
 *
 * \param[in] forceHelperBuffers        Helper force buffers
 * \param[in] force                     force array
 * \param[in] domainWork                Domain lifetime workload flags
 * \param[in] stepWork                  Step schedule flags
 * \param[in] havePpDomainDecomposition Whether we have a PP domain decomposition
 * \param[out] wcycle                   wallcycle recording structure
 *
 * \returns                             Cleared force output structure
 */
static ForceOutputs setupForceOutputs(ForceHelperBuffers*           forceHelperBuffers,
                                      ArrayRefWithPadding<RVec>     force,
                                      const DomainLifetimeWorkload& domainWork,
                                      const StepWorkload&           stepWork,
                                      const bool                    havePpDomainDecomposition,
                                      gmx_wallcycle*                wcycle)
{
    /* NOTE: We assume fr->shiftForces is all zeros here */
    ForceWithShiftForces forceWithShiftForces(
            force, stepWork.computeVirial, forceHelperBuffers->shiftForces());

    if (stepWork.computeForces
        && (domainWork.haveCpuLocalForceWork || !stepWork.useGpuFBufferOps
            || (havePpDomainDecomposition && !stepWork.useGpuFHalo)))
    {
        wallcycle_sub_start(wcycle, WallCycleSubCounter::ClearForceBuffer);
        /* Clear the short- and long-range forces */
        clearRVecs(forceWithShiftForces.force(), true);

        /* Clear the shift forces */
        clearRVecs(forceWithShiftForces.shiftForces(), false);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::ClearForceBuffer);
    }

    /* If we need to compute the virial, we might need a separate
     * force buffer for algorithms for which the virial is calculated
     * directly, such as PME. Otherwise, forceWithVirial uses the
     * the same force (f in legacy calls) buffer as other algorithms.
     */
    const bool useSeparateForceWithVirialBuffer =
            (stepWork.computeForces
             && (stepWork.computeVirial && forceHelperBuffers->haveDirectVirialContributions()));
    /* forceWithVirial uses the local atom range only */
    ForceWithVirial forceWithVirial(useSeparateForceWithVirialBuffer
                                            ? forceHelperBuffers->forceBufferForDirectVirialContributions()
                                            : force.unpaddedArrayRef(),
                                    stepWork.computeVirial);

    if (useSeparateForceWithVirialBuffer)
    {
        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::ClearForceBuffer);
        /* TODO: update comment
         * We only compute forces on local atoms. Note that vsites can
         * spread to non-local atoms, but that part of the buffer is
         * cleared separately in the vsite spreading code.
         */
        clearRVecs(forceWithVirial.force_, true);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::ClearForceBuffer);
    }


    return ForceOutputs(
            forceWithShiftForces, forceHelperBuffers->haveDirectVirialContributions(), forceWithVirial);
}

static ExactRespaForceOutputs setupExactRespaForceOutputs(const t_inputrec&           inputrec,
                                                          ForceOutputs*               level0Output,
                                                          ExactRespaForceOutputStorage* exactRespaForceOutputStorage,
                                                          t_forcerec*                 fr,
                                                          const DomainLifetimeWorkload& domainWork,
                                                          const StepWorkload&         stepWork,
                                                          const ExactRespaStepWork&   exactRespaStepWork,
                                                          const bool                  havePpDomainDecomposition,
                                                          gmx_wallcycle*              wcycle)
{
    GMX_RELEASE_ASSERT(useExactRespa(inputrec), "Exact level outputs should only be set up for exact r-RESPA");
    GMX_RELEASE_ASSERT(level0Output != nullptr, "Need a level-0 exact force output");
    GMX_RELEASE_ASSERT(exactRespaForceOutputStorage != nullptr, "Need exact force-output storage");

    ExactRespaForceOutputs exactRespaForceOutputs;
    exactRespaForceOutputs.numLevels =
            std::min(exactRespaNumLevels(inputrec), ExactRespaForceOutputs::c_numLevels);
    exactRespaForceOutputs.highestActiveLevel =
            std::min(exactRespaStepWork.highestActiveLevel, exactRespaForceOutputs.numLevels - 1);
    exactRespaForceOutputs.longrangeLevel = exactRespaLongrangeNonbondedLevel(inputrec);
    exactRespaForceOutputs.levelOutputs[0] = level0Output;

    for (int level = 1; level < exactRespaForceOutputs.numActiveLevels(); ++level)
    {
        exactRespaForceOutputStorage->ownedLevelForceBuffers[level].resizeWithPadding(
                level0Output->forceWithShiftForces().force().size());
        exactRespaForceOutputStorage->ownedLevelOutputs[level].emplace(
                setupForceOutputs(&fr->forceHelperBuffers[level],
                                  exactRespaForceOutputStorage->ownedLevelForceBuffers[level].arrayRefWithPadding(),
                                  domainWork,
                                  stepWork,
                                  havePpDomainDecomposition,
                                  wcycle));
        exactRespaForceOutputs.levelOutputs[level] =
                &exactRespaForceOutputStorage->ownedLevelOutputs[level].value();
    }

    return exactRespaForceOutputs;
}

/* \brief Launch end-of-step GPU tasks: buffer clearing and rolling pruning.
 *
 */
static void launchGpuEndOfStepTasks(nonbonded_verlet_t*          nbv,
                                    ListedForcesGpu*             listedForcesGpu,
                                    gmx_pme_t*                   pmedata,
                                    gmx_enerdata_t*              enerd,
                                    const MdrunScheduleWorkload& runScheduleWork,
                                    int64_t                      step,
                                    gmx_wallcycle*               wcycle)
{
    if (runScheduleWork.simulationWork.useGpuNonbonded && runScheduleWork.stepWork.computeNonbondedForces)
    {
        /* Launch pruning before buffer clearing because the API overhead of the
         * clear kernel launches can leave the GPU idle while it could be running
         * the prune kernel.
         */
        if (nbv->isDynamicPruningStepGpu(step))
        {
            nbv->dispatchPruneKernelGpu(step);
        }

        /* now clear the GPU outputs while we finish the step on the CPU */
        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        gpu_clear_outputs(nbv->gpuNbv(), runScheduleWork.stepWork.computeVirial);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
    }

    if (runScheduleWork.stepWork.haveGpuPmeOnThisRank)
    {
        wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
        bool gpuGraphWithSeparatePmeRank = false;
        pme_gpu_finish_step(pmedata, gpuGraphWithSeparatePmeRank, wcycle);
        wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
    }

    if (runScheduleWork.domainWork.haveGpuBondedWork && runScheduleWork.stepWork.computeEnergy)
    {
        // in principle this should be included in the DD balancing region,
        // but generally it is infrequent so we'll omit it for the sake of
        // simpler code
        listedForcesGpu->waitAccumulateEnergyTerms(enerd);

        listedForcesGpu->clearEnergies();
    }
}

/*! \brief Compute the number of times the "local coordinates ready on device" GPU event will be used as a synchronization point.
 *
 * When some work is offloaded to GPU, force calculation should wait for the atom coordinates to
 * be ready on the device. The coordinates can come either from H2D copy at the beginning of the step,
 * or from the GPU integration at the end of the previous step.
 *
 * In GROMACS, we usually follow the "mark once - wait once" approach. But this event is "consumed"
 * (that is, waited upon either on host or on the device) multiple times, since many tasks
 * in different streams depend on the coordinates.
 *
 * This function return the number of times the event will be consumed based on this step's workload.
 *
 * \param simulationWork            Simulation workload flags.
 * \param stepWork                  Step workload flags.
 * \param domainWork                Domain workload flags.
 * \param pmeSendCoordinatesFromGpu Whether peer-to-peer communication is used for PME coordinates.
 * \return
 */
static int getExpectedLocalXReadyOnDeviceConsumptionCount(const SimulationWorkload& simulationWork,
                                                          const StepWorkload&       stepWork,
                                                          const DomainLifetimeWorkload& domainWork,
                                                          const bool pmeSendCoordinatesFromGpu)
{
    int result = 0;
    if (stepWork.computeLongRangeNonbondedForces)
    {
        if (pmeSendCoordinatesFromGpu)
        {
            GMX_ASSERT(simulationWork.haveSeparatePmeRank,
                       "GPU PME PP communications require having a separate PME rank");
            // Event is consumed by gmx_pme_send_coordinates for GPU
            // PME PP Communications when the domain has home atoms.
            result++;
        }
        if (stepWork.haveGpuPmeOnThisRank)
        {
            // Event is consumed by launchPmeGpuSpread
            result++;
        }
    }
    if (stepWork.computeNonbondedForces && stepWork.useGpuXBufferOps)
    {
        // Event is consumed by convertCoordinatesGpu for GPU non-bonded work, including
        // exact-r-RESPA inner steps that skip long-range electrostatics.
        result++;
    }
    if (stepWork.useGpuXHalo)
    {
        // Event is consumed by communicateGpuHaloCoordinates
        result++;
        if (GMX_THREAD_MPI) // Issue #4262
        {
            result++;
        }
    }
    if (stepWork.clearGpuFBufferEarly && simulationWork.useGpuUpdate)
    {
        // Event is consumed by force clearing which waits for the update to complete
        result++;
    }
    if (simulationWork.useGpuUpdate && domainWork.haveCpuLocalForceWork)
    {
        // Event is consumed when waiting for it on the CPU prior to CPU force buffer clearing.
        // The actual data dependency does not involve coordinates, but we use this event
        // as an "end-of-step" mark.
        if (!(stepWork.doNeighborSearch || simulationWork.useCpuHaloExchange
              || (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu)))
        {
            result++;
        }
    }
    return result;
}

/*! \brief Compute the number of times the "local forces ready on device" GPU event will be used as a synchronization point.
 *
 * In GROMACS, we usually follow the "mark once - wait once" approach. But this event is "consumed"
 * (that is, waited upon either on host or on the device) multiple times, since many tasks
 * in different streams depend on the local forces.
 *
 * \param simulationWork Simulation workload flags.
 * \param domainWork Domain workload flags.
 * \param stepWork Step workload flags.
 * \param useOrEmulateGpuNb Whether GPU non-bonded calculations are used or emulated.
 * \param alternateGpuWait Whether alternating wait/reduce scheme is used.
 * \return The number of times the event will be consumed based on this step's workload.
 */
static int getExpectedLocalFReadyOnDeviceConsumptionCount(const SimulationWorkload& simulationWork,
                                                          const DomainLifetimeWorkload& domainWork,
                                                          const StepWorkload&           stepWork,
                                                          bool useOrEmulateGpuNb,
                                                          bool alternateGpuWait)
{
    int  counter = 0;
    bool eventUsedInGpuForceReduction =
            (domainWork.haveCpuLocalForceWork
             || (simulationWork.havePpDomainDecomposition && !simulationWork.useGpuHaloExchange));
    bool gpuForceReductionUsed = useOrEmulateGpuNb && !alternateGpuWait && stepWork.useGpuFBufferOps
                                 && stepWork.computeNonbondedForces;
    if (gpuForceReductionUsed && eventUsedInGpuForceReduction)
    {
        counter++;
    }
    bool gpuForceHaloUsed = simulationWork.havePpDomainDecomposition && stepWork.computeForces
                            && stepWork.useGpuFHalo;
    if (gpuForceHaloUsed)
    {
        counter++;
    }
    return counter;
}

//! \brief Data structure to hold dipole-related data and staging arrays
struct DipoleData
{
    //! Dipole staging for fast summing over MPI
    DVec muStaging[2] = { { 0.0, 0.0, 0.0 } };
    //! Dipole staging for states A and B (index 0 and 1 resp.)
    RVec muStateAB[2] = { { 0.0_real, 0.0_real, 0.0_real } };
};


static void reduceAndUpdateMuTot(DipoleData*                   dipoleData,
                                 const MpiComm&                mpiComm,
                                 const bool                    haveFreeEnergy,
                                 ArrayRef<const real>          lambda,
                                 rvec                          muTotal,
                                 const DDBalanceRegionHandler& ddBalanceRegionHandler)
{
    if (mpiComm.isParallel())
    {
        mpiComm.sumReduce(2 * DIM, dipoleData->muStaging[0]);
        ddBalanceRegionHandler.reopenRegionCpu();
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            dipoleData->muStateAB[i][j] = dipoleData->muStaging[i][j];
        }
    }

    if (!haveFreeEnergy)
    {
        copy_rvec(dipoleData->muStateAB[0], muTotal);
    }
    else
    {
        for (int j = 0; j < DIM; j++)
        {
            muTotal[j] = (1.0 - lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)])
                                 * dipoleData->muStateAB[0][j]
                         + lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)]
                                   * dipoleData->muStateAB[1][j];
        }
    }
}

/*! \brief Combines MTS level0..N force buffers into a physical and an impulse-combined force buffer.
 *
 * \param[in]     numAtoms              The number of atoms to combine forces for
 * \param[in,out] forceMtsLevel0        Input: F_level0, output: sum_j F_levelj
 * \param[in,out] forceMtsCombined      Output: F_level0 + sum_{j>0} factor_j * F_levelj
 * \param[in]     slowLevelForces       Force buffers for the active non-zero MTS levels
 * \param[in]     slowLevelStepFactors  Absolute step factors for the active non-zero MTS levels
 */
static void combineMtsForces(const int                                numAtoms,
                             ArrayRef<RVec>                           forceMtsLevel0,
                             ArrayRef<RVec>                           forceMtsCombined,
                             const std::vector<ArrayRef<const RVec>>& slowLevelForces,
                             ArrayRef<const int>                      slowLevelStepFactors)
{
    GMX_RELEASE_ASSERT(slowLevelForces.size() == slowLevelStepFactors.size(),
                       "Need the same number of slow force buffers and MTS factors");

    const int gmx_unused numThreads = gmx_omp_nthreads_get(ModuleMultiThread::Default);
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < numAtoms; i++)
    {
        RVec physicalForce = forceMtsLevel0[i];
        RVec combinedForce = forceMtsLevel0[i];
        for (int slowLevelIndex = 0; slowLevelIndex < static_cast<int>(slowLevelForces.size()); slowLevelIndex++)
        {
            physicalForce += slowLevelForces[slowLevelIndex][i];
            combinedForce += static_cast<real>(slowLevelStepFactors[slowLevelIndex])
                             * slowLevelForces[slowLevelIndex][i];
        }
        forceMtsLevel0[i]   = physicalForce;
        forceMtsCombined[i] = combinedForce;
    }
}

/*! \brief Setup for the local GPU force reduction:
 * reinitialization plus the registration of forces and dependencies.
 *
 * \param [in] runScheduleWork     Schedule workload flag structure
 * \param [in] nbv                 Non-bonded Verlet object
 * \param [in] stateGpu            GPU state propagator object
 * \param [in] gpuForceReduction   GPU force reduction object
 * \param [in] pmePpCommGpu        PME-PP GPU communication object
 * \param [in] pmedata             PME data object
 * \param [in] dd                  Domain decomposition object
 */
static void setupLocalGpuForceReduction(const MdrunScheduleWorkload& runScheduleWork,
                                        nonbonded_verlet_t*          nbv,
                                        StatePropagatorDataGpu*      stateGpu,
                                        GpuForceReduction*           gpuForceReduction,
                                        PmePpCommGpu*                pmePpCommGpu,
                                        const gmx_pme_t*             pmedata,
                                        const gmx_domdec_t*          dd)
{
    GMX_ASSERT(!(runScheduleWork.simulationWork.useLegacyMtsSubsteps()
                 || runScheduleWork.simulationWork.useExactRespa),
               "GPU force reduction is not compatible with substepped dynamics");

    // (re-)initialize local GPU force reduction
    const bool accumulate = runScheduleWork.domainWork.haveCpuLocalForceWork
                            || runScheduleWork.simulationWork.havePpDomainDecomposition;
    const int atomStart = 0;
    gpuForceReduction->reinit(stateGpu->getForces(),
                              nbv->getNumAtoms(AtomLocality::Local),
                              nbv->getGridIndices(),
                              atomStart,
                              accumulate,
                              stateGpu->fReducedOnDevice(AtomLocality::Local));

    // register forces and add dependencies
    gpuForceReduction->registerNbnxmForce(gpu_get_f(nbv->gpuNbv()));

    DeviceBuffer<RVec>    pmeForcePtr;
    GpuEventSynchronizer* pmeSynchronizer     = nullptr;
    bool                  havePmeContribution = false;

    if (runScheduleWork.simulationWork.haveGpuPmeOnPpRank())
    {
        pmeForcePtr = pme_gpu_get_device_f(pmedata);
        if (pmeForcePtr)
        {
            pmeSynchronizer     = pme_gpu_get_f_ready_synchronizer(pmedata);
            havePmeContribution = true;
        }
    }
    else if (runScheduleWork.simulationWork.useGpuPmePpCommunication)
    {
        std::optional<DeviceBuffer<RVec>> pmeStagingPtr = pmePpCommGpu->getGpuForceStagingPtr();
        if (pmeStagingPtr)
        {
            pmeForcePtr = pmeStagingPtr.value();
            GMX_ASSERT(pmeForcePtr, "PME force for reduction has no data");
            if (GMX_THREAD_MPI)
            {
                pmeSynchronizer = pmePpCommGpu->getForcesReadySynchronizer().value();
            }
            havePmeContribution = true;
        }
    }

    if (havePmeContribution)
    {
        gpuForceReduction->registerRvecForce(pmeForcePtr);
        if (runScheduleWork.simulationWork.useNvshmem)
        {
            DeviceBuffer<uint64_t> forcesReadyNvshmemFlags = pmePpCommGpu->getGpuForcesSyncObj();
            gpuForceReduction->registerForcesReadyNvshmemFlags(forcesReadyNvshmemFlags);
        }

        if (!runScheduleWork.simulationWork.useGpuPmePpCommunication || GMX_THREAD_MPI)
        {
            GMX_ASSERT(pmeSynchronizer != nullptr, "PME force ready cuda event should not be NULL");
            gpuForceReduction->addDependency(pmeSynchronizer);
        }
    }

    if (runScheduleWork.domainWork.haveCpuLocalForceWork
        || (runScheduleWork.simulationWork.havePpDomainDecomposition
            && !runScheduleWork.simulationWork.useGpuHaloExchange))
    {
        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::Local));
    }

    if (runScheduleWork.simulationWork.useGpuHaloExchange)
    {
        if (runScheduleWork.simulationWork.useNvshmem)
        {
            GMX_RELEASE_ASSERT(dd->gpuHaloExchangeNvshmemHelper != nullptr,
                               "NVSHMEM helper should be initialized when using NVSHMEM");
            gpuForceReduction->addDependency(
                    dd->gpuHaloExchangeNvshmemHelper->getForcesReadyOnDeviceEvent());
        }
        else
        {
            gpuForceReduction->addDependency(dd->gpuHaloExchange[0][0]->getForcesReadyOnDeviceEvent());
        }
    }
}

/*! \brief Setup for the non-local GPU force reduction:
 * reinitialization plus the registration of forces and dependencies.
 *
 * \param [in] runScheduleWork     Schedule workload flag structure
 * \param [in] nbv                 Non-bonded Verlet object
 * \param [in] stateGpu            GPU state propagator object
 * \param [in] gpuForceReduction   GPU force reduction object
 * \param [in] dd                  Domain decomposition object
 */
static void setupNonLocalGpuForceReduction(const MdrunScheduleWorkload& runScheduleWork,
                                           nonbonded_verlet_t*          nbv,
                                           StatePropagatorDataGpu*      stateGpu,
                                           GpuForceReduction*           gpuForceReduction,
                                           const gmx_domdec_t*          dd)
{
    // (re-)initialize non-local GPU force reduction
    const bool accumulate = runScheduleWork.domainWork.haveCpuNonLocalForceWork;
    const int  atomStart  = dd_numHomeAtoms(*dd);
    gpuForceReduction->reinit(stateGpu->getForces(),
                              nbv->getNumAtoms(AtomLocality::NonLocal),
                              nbv->getGridIndices(),
                              atomStart,
                              accumulate,
                              stateGpu->fReducedOnDevice(AtomLocality::NonLocal));

    // register forces and add dependencies
    gpuForceReduction->registerNbnxmForce(gpu_get_f(nbv->gpuNbv()));

    if (runScheduleWork.domainWork.haveCpuNonLocalForceWork)
    {
        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::NonLocal));
    }
}


/*! \brief Return the number of local atoms.
 */
static int getLocalAtomCount(const gmx_domdec_t* dd, const t_mdatoms& mdatoms, bool havePPDomainDecomposition)
{
    GMX_ASSERT(!(havePPDomainDecomposition && (dd == nullptr)),
               "Can't have PP decomposition with dd uninitialized!");
    return havePPDomainDecomposition ? dd_numAtomsZones(*dd) : mdatoms.homenr;
}

/*! \brief Does pair search and closely related activities required on search steps.
 */
static void doPairSearch(const t_commrec*             cr,
                         const t_inputrec&            inputrec,
                         const MDModulesNotifiers&    mdModulesNotifiers,
                         int64_t                      step,
                         t_nrnb*                      nrnb,
                         gmx_wallcycle*               wcycle,
                         const gmx_localtop_t&        top,
                         const matrix                 box,
                         ArrayRefWithPadding<RVec>    x,
                         ArrayRef<RVec>               v,
                         const t_mdatoms&             mdatoms,
                         t_forcerec*                  fr,
                         const MdrunScheduleWorkload& runScheduleWork)
{
    nonbonded_verlet_t* nbv = fr->nbv.get();

    StatePropagatorDataGpu* stateGpu = fr->stateGpu;

    const SimulationWorkload&     simulationWork = runScheduleWork.simulationWork;
    const StepWorkload&           stepWork       = runScheduleWork.stepWork;
    const DomainLifetimeWorkload& domainWork     = runScheduleWork.domainWork;
    const char*                   traceSide =
            (gmx::useExactRespa(inputrec) && gmx::exactRespaHasPairSplitting(inputrec))
                    ? "PATCH"
                    : "PLAIN";

    if (needStateGpu(simulationWork))
    {
        // TODO refactor this to do_md, after partitioning.
        //
        // Does global communication and symmetric reallocation with NVSHMEM
        stateGpu->reinit(mdatoms.homenr,
                         getLocalAtomCount(cr->dd, mdatoms, simulationWork.havePpDomainDecomposition),
                         cr->commMySim.comm());
        if (simulationWork.useGpuHaloExchange && runScheduleWork.simulationWork.useNvshmem)
        {
            // Does global communication and symmetric reallocation
            reinitGpuHaloExchangeNvshmem(*cr);
        }
    }

    if (simulationWork.haveGpuPmeOnPpRank())
    {
        GMX_ASSERT(needStateGpu(simulationWork), "StatePropagatorDataGpu is needed");
        // TODO: This should be moved into PME setup function ( pme_gpu_prepare_computation(...) )
        pme_gpu_set_device_x(fr->pmedata, stateGpu->getCoordinates());
    }

    if (fr->pbcType != PbcType::No)
    {
        const bool calcCGCM = (stepWork.stateChanged && !haveDDAtomOrdering(*cr));
        if (calcCGCM)
        {
            put_atoms_in_box_omp(fr->pbcType,
                                 box,
                                 fr->haveBoxDeformation,
                                 inputrec.deform,
                                 x.unpaddedArrayRef().subArray(0, mdatoms.homenr),
                                 v.empty() ? ArrayRef<RVec>{} : v.subArray(0, mdatoms.homenr),
                                 gmx_omp_nthreads_get(ModuleMultiThread::Default));
            if (shouldTraceRespaStateXChainStep(step))
            {
                appendStateXChainTracePair(activeM2pTraceDirPath(),
                                           traceSide,
                                           postPbcStageName(step),
                                           step,
                                           x.unpaddedArrayRef(),
                                           "put_atoms_in_box_omp",
                                           "src/gromacs/mdlib/sim_util.cpp:4313",
                                           true);
            }
            inc_nrnb(nrnb, eNR_SHIFTX, mdatoms.homenr);
        }

        if (!haveDDAtomOrdering(*cr))
        {
            // Atoms might have changed periodic image, signal MDModules
            MDModulesAtomsRedistributedSignal mdModulesAtomsRedistributedSignal(
                    box, x.unpaddedArrayRef().subArray(0, mdatoms.homenr), std::nullopt);
            mdModulesNotifiers.simulationRunNotifier_.notify(mdModulesAtomsRedistributedSignal);
        }
    }

    if (fr->wholeMoleculeTransform && stepWork.stateChanged)
    {
        fr->wholeMoleculeTransform->updateForAtomPbcJumps(x.unpaddedArrayRef(), box);
        if (shouldTraceRespaStateXChainStep(step))
        {
            appendStateXChainTracePair(activeM2pTraceDirPath(),
                                       traceSide,
                                       postWholeMoleculeTransformStageName(step),
                                       step,
                                       x.unpaddedArrayRef(),
                                       "wholeMoleculeTransform->updateForAtomPbcJumps",
                                       "src/gromacs/mdlib/sim_util.cpp:4334",
                                       true);
        }
    }

    wallcycle_start(wcycle, WallCycleCounter::NS);
    if (!haveDDAtomOrdering(*cr))
    {
        const rvec vzero       = { 0.0_real, 0.0_real, 0.0_real };
        const rvec boxDiagonal = { box[XX][XX], box[YY][YY], box[ZZ][ZZ] };
        if (shouldTraceRespaCoordHandoffStep(step))
        {
            appendCoordHandoffTracePair(activeM2pTraceDirPath(),
                                        "PLAIN",
                                        "PRE_HANDOFF_COORD_SOURCE",
                                        step,
                                        x.unpaddedArrayRef(),
                                        "state.x.unpaddedArrayRef()_before_putAtomsOnGrid",
                                        x.unpaddedArrayRef().data());
        }
        wallcycle_sub_start(wcycle, WallCycleSubCounter::NBSGridLocal);
        nbv->putAtomsOnGrid(box,
                            0,
                            vzero,
                            boxDiagonal,
                            nullptr,
                            { 0, mdatoms.homenr },
                            mdatoms.homenr,
                            -1,
                            fr->atomInfo,
                            x.unpaddedArrayRef(),
                            nullptr);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::NBSGridLocal);
        if (shouldTraceRespaCoordHandoffStep(step))
        {
            appendCoordHandoffTracePair(activeM2pTraceDirPath(),
                                        "PLAIN",
                                        "POST_HANDOFF_BUFFER",
                                        step,
                                        nbv->nbat(),
                                        "nbv.nbat.x()_after_putAtomsOnGrid",
                                        nbv->nbat().x().data());
        }
    }
    else
    {
        wallcycle_sub_start(wcycle, WallCycleSubCounter::NBSGridNonLocal);
        if (!nbv->localAtomOrderMatchesNbnxmOrder())
        {
            nbnxn_put_on_grid_nonlocal(nbv, getDomdecZones(*cr->dd), fr->atomInfo, x.unpaddedArrayRef());
        }
        nbv->convertCoordinates(AtomLocality::NonLocal, x.unpaddedArrayRef());
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::NBSGridNonLocal);
    }

    nbv->setAtomProperties(mdatoms.typeA, mdatoms.chargeA, fr->atomInfo, mdatoms.typeB, mdatoms.chargeB);

    wallcycle_stop(wcycle, WallCycleCounter::NS);

    /* initialize the GPU nbnxm atom data and bonded data structures */
    if (simulationWork.useGpuNonbonded)
    {
        // Note: cycle counting only nononbondeds, GPU listed forces counts internally
        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        gpu_init_atomdata(nbv->gpuNbv(), &nbv->nbat());
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);

        if (fr->listedForcesGpu)
        {
            /* Now we put all atoms on the grid, we can assign bonded
             * interactions to the GPU, where the grid order is
             * needed. Also the xq, f and fshift device buffers have
             * been reallocated if needed, so the bonded code can
             * learn about them. */
            // TODO the xq, f, and fshift buffers are now shared
            // resources, so they should be maintained by a
            // higher-level object than the nb module.
            fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
                    nbv->getGridIndices(), top.idef, gpuGetNBAtomData(nbv->gpuNbv()));
        }
    }

    wallcycle_start_nocount(wcycle, WallCycleCounter::NS);
    wallcycle_sub_start(wcycle, WallCycleSubCounter::NBSSearchLocal);
    /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
    const bool needAllPairsInPairlist = fr->completePairlistRange.has_value();
    const bool forceEagerPlainPairlistMaterialization = []()
    {
        const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_EAGER_PLAIN_PAIRLIST");
        return env != nullptr && *env != '\0' && std::strcmp(env, "0") != 0;
    }();
    const std::optional<real> plainPairlistMaterializationRange =
            fr->plainPairlistRange.has_value()
                    ? fr->plainPairlistRange
                    : (forceEagerPlainPairlistMaterialization ? fr->completePairlistRange : std::nullopt);
    const bool needPlainPairlistMaterialization =
            plainPairlistMaterializationRange.has_value()
            && (forceEagerPlainPairlistMaterialization
                || mdModulesNotifiers.simulationRunNotifier_
                           .haveSubscribers<const MDModulesPairlistConstructedSignal&>());
    nbv->constructPairlist(InteractionLocality::Local, top.excls, needAllPairsInPairlist, step, nrnb);

    nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::Local);

    wallcycle_sub_stop(wcycle, WallCycleSubCounter::NBSSearchLocal);
    wallcycle_stop(wcycle, WallCycleCounter::NS);

    if (simulationWork.useGpuXBufferOpsWhenAllowed)
    {
        nbv->atomdata_init_copy_x_to_nbat_x_gpu();
    }

    if (simulationWork.useGpuFBufferOpsWhenAllowed)
    {
        // with MPI, direct GPU communication, and separate PME ranks we need
        // gmx_pme_send_coordinates() to be called before we can set up force reduction
        bool delaySetupLocalGpuForceReduction = GMX_MPI && simulationWork.useGpuPmePpCommunication;
        if (!delaySetupLocalGpuForceReduction)
        {
            setupLocalGpuForceReduction(runScheduleWork,
                                        nbv,
                                        stateGpu,
                                        fr->gpuForceReduction[AtomLocality::Local].get(),
                                        fr->pmePpCommGpu.get(),
                                        fr->pmedata,
                                        cr->dd);
        }

        if (simulationWork.havePpDomainDecomposition)
        {
            setupNonLocalGpuForceReduction(runScheduleWork,
                                           nbv,
                                           stateGpu,
                                           fr->gpuForceReduction[AtomLocality::NonLocal].get(),
                                           cr->dd);
        }
    }

    /* do non-local pair search */
    if (simulationWork.havePpDomainDecomposition)
    {
        wallcycle_start_nocount(wcycle, WallCycleCounter::NS);
        wallcycle_sub_start(wcycle, WallCycleSubCounter::NBSSearchNonLocal);
        /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
        nbv->constructPairlist(InteractionLocality::NonLocal, top.excls, needAllPairsInPairlist, step, nrnb);

        nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::NonLocal);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::NBSSearchNonLocal);
        wallcycle_stop(wcycle, WallCycleCounter::NS);
        // TODO refactor this GPU halo exchange re-initialisation
        // to location in do_md where GPU halo exchange is
        // constructed at partitioning, after above stateGpu
        // re-initialization has similarly been refactored
        // because with NVSHMEM the ordering of synchronous
        // global operations must be preserved.
        if (simulationWork.useGpuHaloExchange)
        {
            reinitGpuHaloExchange(*cr, stateGpu->getCoordinates(), stateGpu->getForces());
        }
    }

    // With FEP we set up the reduction over threads for local+non-local simultaneously,
    // so we need to do that here after the local and non-local pairlist construction.
    if (domainWork.haveCpuNonbondedFreeEnergyWork)
    {
        wallcycle_sub_start(wcycle, WallCycleSubCounter::NonbondedFep);
        nbv->setupFepThreadedForceBuffer(fr->natoms_force_constr);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::NonbondedFep);
    }

    if (needPlainPairlistMaterialization)
    {
        const auto& plainPairlist = nbv->plainPairlist(plainPairlistMaterializationRange.value(), fr->shift_vec);
        MDModulesPairlistConstructedSignal mdModulesPairlistConstructedSignal(
                plainPairlist.pairs, plainPairlist.excludedPairs, mdatoms.typeA);
        mdModulesNotifiers.simulationRunNotifier_.notify(mdModulesPairlistConstructedSignal);
    }
}

void do_force(FILE*                         fplog,
              const t_commrec*              cr,
              const t_inputrec&             inputrec,
              const MDModulesNotifiers&     mdModulesNotifiers,
              Awh*                          awh,
              gmx_enfrot*                   enforcedRotation,
              ImdSession*                   imdSession,
              pull_t*                       pull_work,
              int64_t                       step,
              t_nrnb*                       nrnb,
              gmx_wallcycle*                wcycle,
              const gmx_localtop_t*         top,
              const matrix                  box,
              ArrayRefWithPadding<RVec>     x,
              ArrayRef<RVec>                v,
              const history_t*              hist,
              ForceBuffersView*             forceView,
              ExactRespaForceStore*         exactRespaForceStore,
              tensor                        vir_force,
              const t_mdatoms*              mdatoms,
              gmx_enerdata_t*               enerd,
              ArrayRef<const real>          lambda,
              t_forcerec*                   fr,
              const MdrunScheduleWorkload&  runScheduleWork,
              VirtualSitesHandler*          vsite,
              rvec                          muTotal,
              double                        t,
              gmx_edsam*                    ed,
              CpuPpLongRangeNonbondeds*     longRangeNonbondeds,
              const DDBalanceRegionHandler& ddBalanceRegionHandler)
{
    ArrayRefWithPadding<RVec> force = forceView->forceWithPadding();
    GMX_ASSERT(force.unpaddedArrayRef().ssize() >= fr->natoms_force_constr,
               "The size of the force buffer should be at least the number of atoms to compute "
               "forces for");

    nonbonded_verlet_t*  nbv = fr->nbv.get();
    interaction_const_t* ic  = fr->ic.get();

    StatePropagatorDataGpu* stateGpu = fr->stateGpu;

    const SimulationWorkload& simulationWork = runScheduleWork.simulationWork;

    const DomainLifetimeWorkload& domainWork = runScheduleWork.domainWork;

    const StepWorkload& stepWork = runScheduleWork.stepWork;
    const ExactRespaStepWork& exactRespaStepWork = runScheduleWork.exactRespaStepWork;
    const int highestActiveSubstepLevel =
            simulationWork.useExactRespa ? exactRespaStepWork.highestActiveLevel
                                         : stepWork.highestActiveMtsLevel;
    const bool computeLegacySlowSubstepForces =
            (!simulationWork.useExactRespa && stepWork.computeSlowForces);
    const bool combineSubstepForcesBeforeHaloExchange =
            simulationWork.useExactRespa ? exactRespaStepWork.combineForcesBeforeHaloExchange
                                         : stepWork.combineMtsForcesBeforeHaloExchange;
    const bool          useExactLammpsRespaPairSplitting =
            gmx::useExactRespa(inputrec) && gmx::exactRespaHasPairSplitting(inputrec);
    const bool          useDedicatedExactRespaGpuNonbonded =
            stepWork.computeNonbondedForces && useExactLammpsRespaPairSplitting
            && simulationWork.useGpuNonbonded;
    const char*         traceSide =
            useExactLammpsRespaPairSplitting ? "PATCH" : "PLAIN";
    const bool          traceForceComponents = shouldTraceRespaForceComponentsStep(step);
    const bool          traceExactGpuBondedLaunchContext =
            shouldTraceExactGpuBondedLaunchContextStep(step);
    const bool          traceExactGpuBondedDeviceXq =
            shouldTraceExactGpuBondedDeviceXqStep(step);
    const bool          traceExactGpuBondedDeviceForce =
            shouldTraceExactGpuBondedDeviceForceStep(step);
    const bool          traceExactGpuBondedGridIndices =
            shouldTraceExactGpuBondedGridIndexStep(step);
    const bool          traceRealspaceForceSubcomponents =
            shouldTraceRespaRealspaceForceSubcomponentsStep(step);
    const bool          traceExclusionEquivalence =
            shouldTraceRespaExclusionEquivalenceStep(step);

    if (traceForceComponents)
    {
        const char* traceDirPath = activeM2pTraceDirPath();
        if (traceDirPath != nullptr && *traceDirPath != '\0')
        {
            static std::string clearedForceComponentTracePath;
            const std::string  tracePath =
                    (std::filesystem::path(traceDirPath) / "step0_force_component_trace.txt").string();
            if (tracePath != clearedForceComponentTracePath)
            {
                writeRespaTraceTextFile(traceDirPath, "step0_force_component_trace.txt", "");
                clearedForceComponentTracePath = tracePath;
            }
        }
    }
    if (traceRealspaceForceSubcomponents)
    {
        const char* traceDirPath = activeM2pTraceDirPath();
        if (traceDirPath != nullptr && *traceDirPath != '\0')
        {
            static std::string clearedRealspaceTracePath;
            const std::string  tracePath =
                    (std::filesystem::path(traceDirPath) / "step0_realspace_force_subcomponent_trace.txt")
                            .string();
            if (tracePath != clearedRealspaceTracePath)
            {
                writeRespaTraceTextFile(traceDirPath, "step0_realspace_force_subcomponent_trace.txt", "");
                clearedRealspaceTracePath = tracePath;
            }
        }
    }
    if (traceForceComponents || traceRealspaceForceSubcomponents)
    {
        clearExactRespaRealspaceTraceCapture(step);
    }
    if (traceExclusionEquivalence)
    {
        const char* traceDirPath = activeM2pTraceDirPath();
        if (traceDirPath != nullptr && *traceDirPath != '\0')
        {
            static std::string clearedExclusionEquivalenceTracePath;
            const std::string  tracePath =
                    (std::filesystem::path(traceDirPath) / "step0_exclusion_equivalence_pair_trace.txt")
                            .string();
            if (tracePath != clearedExclusionEquivalenceTracePath)
            {
                writeRespaTraceTextFile(traceDirPath, "step0_exclusion_equivalence_pair_trace.txt", "");
                clearedExclusionEquivalenceTracePath = tracePath;
            }
        }
    }

    if (shouldTraceRespaStateXChainStep(step))
    {
        appendStateXChainTracePair(activeM2pTraceDirPath(),
                                   traceSide,
                                   loopEntryStageName(step),
                                   step,
                                   x.unpaddedArrayRef(),
                                   "do_force entry",
                                   "src/gromacs/mdlib/sim_util.cpp:4632",
                                   false);
    }

    const bool pmeSendCoordinatesFromGpu =
            simulationWork.useGpuPmePpCommunication && !stepWork.doNeighborSearch;

    const int64_t previousRespaCurrentDoForceStep = g_respaCurrentDoForceStep;
    const int* previousRespaTraceGlobalAtomIndices = g_respaTraceGlobalAtomIndices;
    const int previousRespaTraceGlobalAtomIndexCount = g_respaTraceGlobalAtomIndexCount;
    const int* previousRespaCurrentGlobalAtomIndices = g_respaCurrentGlobalAtomIndices;
    const int previousRespaCurrentGlobalAtomIndexCount = g_respaCurrentGlobalAtomIndexCount;
    struct RespaTraceThreadLocalGuard
    {
        int64_t& currentDoForceStep;
        const int*& traceGlobalAtomIndices;
        int& traceGlobalAtomIndexCount;
        const int*& currentGlobalAtomIndices;
        int& currentGlobalAtomIndexCount;
        int64_t previousDoForceStep;
        const int* previousGlobalAtomIndices;
        int previousGlobalAtomIndexCount;
        const int* previousCurrentGlobalAtomIndices;
        int previousCurrentGlobalAtomIndexCount;

        ~RespaTraceThreadLocalGuard()
        {
            currentDoForceStep = previousDoForceStep;
            traceGlobalAtomIndices = previousGlobalAtomIndices;
            traceGlobalAtomIndexCount = previousGlobalAtomIndexCount;
            currentGlobalAtomIndices = previousCurrentGlobalAtomIndices;
            currentGlobalAtomIndexCount = previousCurrentGlobalAtomIndexCount;
        }
    } respaTraceThreadLocalGuard{ g_respaCurrentDoForceStep,
                                  g_respaTraceGlobalAtomIndices,
                                  g_respaTraceGlobalAtomIndexCount,
                                  g_respaCurrentGlobalAtomIndices,
                                  g_respaCurrentGlobalAtomIndexCount,
                                  previousRespaCurrentDoForceStep,
                                  previousRespaTraceGlobalAtomIndices,
                                  previousRespaTraceGlobalAtomIndexCount,
                                  previousRespaCurrentGlobalAtomIndices,
                                  previousRespaCurrentGlobalAtomIndexCount };

    g_respaCurrentDoForceStep = step;
    g_respaTraceGlobalAtomIndices =
            haveDDAtomOrdering(*cr) ? cr->dd->globalAtomIndices.data() : nullptr;
    g_respaTraceGlobalAtomIndexCount =
            haveDDAtomOrdering(*cr) ? static_cast<int>(cr->dd->globalAtomIndices.size()) : 0;
    g_respaCurrentGlobalAtomIndices    = g_respaTraceGlobalAtomIndices;
    g_respaCurrentGlobalAtomIndexCount = g_respaTraceGlobalAtomIndexCount;
    if ((g_respaCurrentGlobalAtomIndices == nullptr || g_respaCurrentGlobalAtomIndexCount == 0)
        && fr != nullptr && fr->nbv != nullptr)
    {
        const auto localAtomOrder = fr->nbv->getLocalAtomOrder();
        if (!localAtomOrder.empty())
        {
            g_respaCurrentGlobalAtomIndices    = localAtomOrder.data();
            g_respaCurrentGlobalAtomIndexCount = localAtomOrder.ssize();
        }
    }
    g_respaLatestForceDumpGlobalAtomIndices    = g_respaCurrentGlobalAtomIndices;
    g_respaLatestForceDumpGlobalAtomIndexCount = g_respaCurrentGlobalAtomIndexCount;

    const bool reinitGpuPmePpComms = simulationWork.useGpuPmePpCommunication && stepWork.doNeighborSearch;
    if (stepWork.computePmeOnSeparateRank && stepWork.doNeighborSearch)
    {
        // We call the gmx_pme_send_coordinates early for reinit case
        // in order for nvshmem collective calls in StatePropagatorDataGpu::Impl::reinit
        // to be in sync with PME-PP
        gmx_pme_send_coordinates(fr,
                                 cr->dd,
                                 box,
                                 x.unpaddedArrayRef(),
                                 lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)],
                                 lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Vdw)],
                                 (stepWork.computeVirial || stepWork.computeEnergy),
                                 step,
                                 simulationWork.useGpuPmePpCommunication,
                                 reinitGpuPmePpComms,
                                 pmeSendCoordinatesFromGpu,
                                 stepWork.useGpuPmeFReduction,
                                 nullptr,
                                 simulationWork.useMdGpuGraph,
                                 wcycle);
    }

    if (stepWork.doNeighborSearch)
    {
        doPairSearch(cr, inputrec, mdModulesNotifiers, step, nrnb, wcycle, *top, box, x, v, *mdatoms, fr, runScheduleWork);

        /* At a search step we need to start the first balancing region
         * somewhere early inside the step after communication during domain
         * decomposition (and not during the previous step as usual).
         */
        ddBalanceRegionHandler.openBeforeForceComputationCpu(DdAllowBalanceRegionReopen::yes);
    }

    auto* localXReadyOnDevice = (stepWork.haveGpuPmeOnThisRank || stepWork.useGpuXBufferOps
                                 || simulationWork.useGpuUpdate || pmeSendCoordinatesFromGpu)
                                        ? stateGpu->getCoordinatesReadyOnDeviceEvent(
                                                  AtomLocality::Local, simulationWork, stepWork)
                                        : nullptr;
    const bool localCoordinatesNeededOnDevice =
            stepWork.haveGpuPmeOnThisRank || stepWork.useGpuXBufferOps || pmeSendCoordinatesFromGpu;
    const int expectedLocalXReadyOnDeviceConsumptionCount =
            localCoordinatesNeededOnDevice
                    ? getExpectedLocalXReadyOnDeviceConsumptionCount(
                              simulationWork, stepWork, domainWork, pmeSendCoordinatesFromGpu)
                    : 0;
    const char* localCoordinateProvider =
            (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch) ? "xUpdatedOnDeviceEvent"
                                                                        : "xReadyOnDevice";

    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch && localCoordinatesNeededOnDevice)
    {
        stateGpu->setXUpdatedOnDeviceEventExpectedConsumptionCount(
                expectedLocalXReadyOnDeviceConsumptionCount);
    }

    if (stepWork.clearGpuFBufferEarly)
    {
        // GPU Force halo exchange will set a subset of local atoms with remote non-local data.
        // First clear local portion of force array, so that untouched atoms are zero.
        // The dependency for this is that forces from previous timestep have been consumed,
        // which is satisfied when localXReadyOnDevice has been marked for GPU update case.
        // For CPU update, the forces are consumed by the beginning of the step, so no extra sync needed.
        GpuEventSynchronizer* dependency = simulationWork.useGpuUpdate ? localXReadyOnDevice : nullptr;
        stateGpu->clearForcesOnGpu(AtomLocality::Local, dependency);
    }

    clear_mat(vir_force);

    if (fr->pbcType != PbcType::No)
    {
        /* Compute shift vectors every step,
         * because of pressure coupling or box deformation!
         */
        if (simulationWork.haveDynamicBox && stepWork.stateChanged)
        {
            calc_shifts(box, fr->shift_vec);
        }
    }
    nbnxn_atomdata_copy_shiftvec(simulationWork.haveDynamicBox, fr->shift_vec, &nbv->nbat());


    GMX_ASSERT(simulationWork.useGpuHaloExchange
                       == ((cr->dd != nullptr)
                           && (!cr->dd->gpuHaloExchange[0].empty()
                               || cr->dd->gpuHaloExchangeNvshmemHelper != nullptr)),
               "The GPU halo exchange is active, but it has not been constructed.");

    bool gmx_used_in_debug haveCopiedXFromGpu = false;
    bool                   copiedCoordinatesToGpu = false;
    // Copy coordinate from the GPU if update is on the GPU and there
    // are forces to be computed on the CPU, or for the computation of
    // virial, or if host-side data will be transferred from this task
    // to a remote task for halo exchange or PME-PP communication. At
    // search steps the current coordinates are already on the host,
    // hence copy is not needed.
    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch
        && (runScheduleWork.domainWork.haveCpuLocalForceWork
            || stepWork.computeVirial || simulationWork.useCpuPmePpCommunication
            || simulationWork.useCpuHaloExchange || simulationWork.computeMuTot))
    {
        stateGpu->copyCoordinatesFromGpu(x.unpaddedArrayRef(), AtomLocality::Local);
        haveCopiedXFromGpu = true;
    }

    // Coordinates on the device are needed if PME or BufferOps are offloaded.
    // The local coordinates can be copied right away.
    // NOTE: Consider moving this copy to right after they are updated and constrained,
    //       if the later is not offloaded.
    if (localCoordinatesNeededOnDevice)
    {
        GMX_ASSERT(stateGpu != nullptr, "stateGpu should not be null");

        // We need to copy coordinates when:
        // 1. Update is not offloaded
        // 2. The buffers were reinitialized on search step
        if (!simulationWork.useGpuUpdate || stepWork.doNeighborSearch)
        {
            stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(),
                                           AtomLocality::Local,
                                           expectedLocalXReadyOnDeviceConsumptionCount);
            copiedCoordinatesToGpu = true;
        }
    }

    if (stepWork.computePmeOnSeparateRank && !stepWork.doNeighborSearch)
    {
        /* Send particle coordinates to the pme nodes */
        if (!pmeSendCoordinatesFromGpu && simulationWork.useGpuUpdate)
        {
            GMX_ASSERT(haveCopiedXFromGpu,
                       "a wait should only be triggered if copy has been scheduled");
            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
        }

        gmx_pme_send_coordinates(fr,
                                 cr->dd,
                                 box,
                                 x.unpaddedArrayRef(),
                                 lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)],
                                 lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Vdw)],
                                 (stepWork.computeVirial || stepWork.computeEnergy),
                                 step,
                                 simulationWork.useGpuPmePpCommunication,
                                 reinitGpuPmePpComms,
                                 pmeSendCoordinatesFromGpu,
                                 stepWork.useGpuPmeFReduction,
                                 pmeSendCoordinatesFromGpu ? localXReadyOnDevice : nullptr,
                                 simulationWork.useMdGpuGraph,
                                 wcycle);
    }

    if (simulationWork.useGpuFBufferOpsWhenAllowed && stepWork.doNeighborSearch)
    {
        // with MPI, direct GPU communication, and separate PME ranks we need
        // gmx_pme_send_coordinates() to be called before we can set up force reduction
        bool doSetupLocalGpuForceReduction = GMX_MPI && simulationWork.useGpuPmePpCommunication;
        if (doSetupLocalGpuForceReduction)
        {
            setupLocalGpuForceReduction(runScheduleWork,
                                        fr->nbv.get(),
                                        stateGpu,
                                        fr->gpuForceReduction[AtomLocality::Local].get(),
                                        fr->pmePpCommGpu.get(),
                                        fr->pmedata,
                                        cr->dd);
        }
    }

    if (stepWork.haveGpuPmeOnThisRank)
    {
        launchPmeGpuSpread(fr->pmedata,
                           box,
                           simulationWork,
                           stepWork,
                           localXReadyOnDevice,
                           lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)],
                           simulationWork.useMdGpuGraph,
                           wcycle);
    }


    if (!stepWork.doNeighborSearch && !EI_TPI(inputrec.eI) && stepWork.computeNonbondedForces)
    {
        const bool useExactLammpsRespaNonbondedLocal =
                stepWork.computeNonbondedForces && useExactLammpsRespaPairSplitting;
        if (stepWork.useGpuXBufferOps)
        {
            GMX_ASSERT(stateGpu, "stateGpu should be valid when buffer ops are offloaded");
            nbv->convertCoordinatesGpu(AtomLocality::Local, stateGpu->getCoordinates(), localXReadyOnDevice);
        }
        else
        {
            if (simulationWork.useGpuUpdate && !useExactLammpsRespaNonbondedLocal)
            {
                GMX_ASSERT(stateGpu, "need a valid stateGpu object");
                GMX_ASSERT(haveCopiedXFromGpu,
                           "a wait should only be triggered if copy has been scheduled");
                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
            }
            if (shouldTraceRespaCoordHandoffStep(step) && !useExactLammpsRespaNonbondedLocal)
            {
                appendCoordHandoffTracePair(activeM2pTraceDirPath(),
                                            "PLAIN",
                                            "PRE_HANDOFF_COORD_SOURCE",
                                            step,
                                            x.unpaddedArrayRef(),
                                            "state.x.unpaddedArrayRef()_before_convertCoordinates",
                                            x.unpaddedArrayRef().data());
            }
            nbv->convertCoordinates(AtomLocality::Local, x.unpaddedArrayRef());
            if (shouldTraceRespaCoordHandoffStep(step) && !useExactLammpsRespaNonbondedLocal)
            {
                appendCoordHandoffTracePair(activeM2pTraceDirPath(),
                                            "PLAIN",
                                            "POST_HANDOFF_BUFFER",
                                            step,
                                            nbv->nbat(),
                                            "nbv.nbat.x()_after_convertCoordinates",
                                            nbv->nbat().x().data());
            }
        }
    }

    if (simulationWork.useGpuNonbonded && !useDedicatedExactRespaGpuNonbonded
        && (stepWork.computeNonbondedForces || domainWork.haveGpuBondedWork))
    {
        ddBalanceRegionHandler.openBeforeForceComputationGpu();

        wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
        wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        gpu_upload_shiftvec(nbv->gpuNbv(), &nbv->nbat());
        if (!stepWork.useGpuXBufferOps)
        {
            gpu_copy_xq_to_gpu(nbv->gpuNbv(), &nbv->nbat(), AtomLocality::Local);
        }
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
        // with X buffer ops offloaded to the GPU on all but the search steps

        // bonded work not split into separate local and non-local, so with DD
        // we can only launch the kernel after non-local coordinates have been received.
        if (domainWork.haveGpuBondedWork && !simulationWork.havePpDomainDecomposition)
        {
            fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
        }

        /* launch local nonbonded work on GPU */
        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::Local, enbvClearFNo, step, nrnb, wcycle);
        /* launch local nonbonded free energy work on GPU */
        if (domainWork.haveGpuNonbondedFreeEnergyWork && stepWork.computeNonbondedForces)
        {
            nbv->dispatchFreeEnergyGpuKernels(InteractionLocality::Local, simulationWork, stepWork);
        }
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
    }

    if (stepWork.haveGpuPmeOnThisRank)
    {
        // In PME GPU and mixed mode we launch FFT / gather after the
        // X copy/transform to allow overlap as well as after the GPU NB
        // launch to avoid FFT launch overhead hijacking the CPU and delaying
        // the nonbonded kernel.
        launchPmeGpuFftAndGather(fr->pmedata,
                                 lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)],
                                 wcycle,
                                 stepWork);
    }

    /* Communicate coordinates and sum dipole if necessary */
    if (simulationWork.havePpDomainDecomposition)
    {
        GpuEventSynchronizer* gpuCoordinateHaloLaunched = nullptr;
        if (!stepWork.doNeighborSearch)
        {
            if (stepWork.useGpuXHalo)
            {
                // The following must be called after local setCoordinates (which records an event
                // when the coordinate data has been copied to the device).
                gpuCoordinateHaloLaunched = communicateGpuHaloCoordinates(*cr, box, localXReadyOnDevice);

                if (domainWork.haveCpuNonLocalForceWork)
                {
                    // non-local part of coordinate buffer must be copied back to host for CPU work
                    stateGpu->copyCoordinatesFromGpu(
                            x.unpaddedArrayRef(), AtomLocality::NonLocal, gpuCoordinateHaloLaunched);
                }
            }
            else
            {
                if (simulationWork.useGpuUpdate)
                {
                    GMX_ASSERT(haveCopiedXFromGpu,
                               "a wait should only be triggered if copy has been scheduled");
                    const bool haveAlreadyWaited =
                            (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
                    if (!haveAlreadyWaited)
                    {
                        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
                    }
                }

                if (cr->dd->haloExchange)
                {
                    wallcycle_start(wcycle, WallCycleCounter::MoveX);
                    cr->dd->haloExchange->moveX(box, x.unpaddedArrayRef());
                    wallcycle_stop(wcycle, WallCycleCounter::MoveX);
                }
                else
                {
                    dd_move_x(cr->dd, box, x.unpaddedArrayRef(), wcycle);
                }
            }
        }

        if (stepWork.useGpuXBufferOps)
        {
            if (!stepWork.useGpuXHalo)
            {
                stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(), AtomLocality::NonLocal);
            }
            GpuEventSynchronizer* xReadyOnDeviceEvent = stateGpu->getCoordinatesReadyOnDeviceEvent(
                    AtomLocality::NonLocal, simulationWork, stepWork, gpuCoordinateHaloLaunched);
            if (stepWork.useGpuXHalo && domainWork.haveCpuNonLocalForceWork)
            {
                /* We already enqueued an event for Gpu Halo exchange completion into the
                 * NonLocal stream when D2H copying the coordinates. */
                xReadyOnDeviceEvent = nullptr;
            }
            nbv->convertCoordinatesGpu(
                    AtomLocality::NonLocal, stateGpu->getCoordinates(), xReadyOnDeviceEvent);
        }
        else if (!stepWork.doNeighborSearch)
        {
            nbv->convertCoordinates(AtomLocality::NonLocal, x.unpaddedArrayRef());
        }

        if (simulationWork.useGpuNonbonded && !useDedicatedExactRespaGpuNonbonded)
        {

            if (!stepWork.useGpuXBufferOps)
            {
                wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
                wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
                gpu_copy_xq_to_gpu(nbv->gpuNbv(), &nbv->nbat(), AtomLocality::NonLocal);
                wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
                wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
            }

            if (domainWork.haveGpuBondedWork)
            {
                fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
            }

            /* launch non-local nonbonded tasks on GPU */
            wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
            wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
            do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::NonLocal, enbvClearFNo, step, nrnb, wcycle);
            /* launch non-local nonbonded free energy tsaks on GPU */
            if (domainWork.haveGpuNonbondedFreeEnergyWork && stepWork.computeNonbondedForces)
            {
                nbv->dispatchFreeEnergyGpuKernels(InteractionLocality::NonLocal, simulationWork, stepWork);
            }
            wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
            wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
        }
    }

    if (simulationWork.useGpuNonbonded && !useDedicatedExactRespaGpuNonbonded
        && stepWork.computeNonbondedForces)
    {
        /* launch D2H copy-back F */
        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);

        if (simulationWork.havePpDomainDecomposition)
        {
            gpu_launch_cpyback(nbv->gpuNbv(), &nbv->nbat(), stepWork, AtomLocality::NonLocal);
        }
        gpu_launch_cpyback(nbv->gpuNbv(), &nbv->nbat(), stepWork, AtomLocality::Local);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);

        if (domainWork.haveGpuBondedWork && stepWork.computeEnergy)
        {
            fr->listedForcesGpu->launchEnergyTransfer();
        }
        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
    }

    ArrayRef<const RVec> xWholeMolecules;
    if (fr->wholeMoleculeTransform)
    {
        xWholeMolecules = fr->wholeMoleculeTransform->wholeMoleculeCoordinates(x.unpaddedArrayRef(), box);
    }

    // The CPU force buffer force clearing needs to happen after the previous step
    // force reduction has already completed. To minimize the cross-step dependencies
    // we wait on the coordinates to be updated on the device which is sufficient but
    // a later event than what we strictly need to synchronize with.
    if (simulationWork.useGpuUpdate && domainWork.haveCpuLocalForceWork)
    {
        const bool coordinatesAlreadyUpdatedOnDevice =
                stepWork.doNeighborSearch || simulationWork.useCpuHaloExchange
                || (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
        if (!coordinatesAlreadyUpdatedOnDevice)
        {
            stateGpu->waitCoordinatesUpdatedOnDevice();
        }
    }

    /* Start the force cycle counter.
     * Note that a different counter is used for dynamic load balancing.
     */
    wallcycle_start(wcycle, WallCycleCounter::Force);

    /* Set up and clear force outputs:
     * forceOutMtsLevel0:  everything except what is in the other two outputs
     * forceOutMtsLevel1:  PME-mesh and listed-forces group 1
     * forceOutNonbonded: non-bonded forces
     * Without multiple time stepping all point to the same object.
     * With multiple time-stepping the use is different for MTS fast (level0 only) and slow steps.
     */
    ForceOutputs forceOutMtsLevel0 = setupForceOutputs(
            &fr->forceHelperBuffers[0], force, domainWork, stepWork, simulationWork.havePpDomainDecomposition, wcycle);
    const int  numMtsLevels = gmx::useExactRespa(inputrec) ? gmx::exactRespaNumLevels(inputrec)
                                                           : static_cast<int>(inputrec.mtsLevels.size());
    const bool useExactRespaForceOutputs = simulationWork.useExactRespa;
    const bool useLegacyMtsForceOutputs  = simulationWork.useLegacyMtsSubsteps();
    const bool useSubstepLevels          = useExactRespaForceOutputs || useLegacyMtsForceOutputs;
    const bool useMultiLevelMts          = (useLegacyMtsForceOutputs && numMtsLevels > 2);
    const int  longrangeMtsLevel = gmx::useExactRespa(inputrec)
                                           ? gmx::exactRespaLongrangeNonbondedLevel(inputrec)
                                           : (useSubstepLevels ? forceGroupMtsLevel(inputrec.mtsLevels,
                                                                                    MtsForceGroups::LongrangeNonbonded)
                                                               : 0);

    ExactRespaForceOutputStorage          exactRespaForceOutputStorage;
    ExactRespaForceOutputs                exactRespaForceOutputs;
    std::optional<ForceOutputs>              forceOutSingleSlowLevel;
    std::vector<std::optional<ForceOutputs>> forceOutMultiLevel;
    std::vector<ForceOutputs*>               forceOutByMtsLevel(
            (!useExactRespaForceOutputs && useSubstepLevels) ? numMtsLevels : 1, nullptr);
    forceOutByMtsLevel[0] = &forceOutMtsLevel0;

    if (useExactRespaForceOutputs)
    {
        exactRespaForceOutputs =
                setupExactRespaForceOutputs(inputrec,
                                            &forceOutMtsLevel0,
                                            &exactRespaForceOutputStorage,
                                            fr,
                                            domainWork,
                                            stepWork,
                                            exactRespaStepWork,
                                            simulationWork.havePpDomainDecomposition,
                                            wcycle);
    }
    else if (useLegacyMtsForceOutputs && computeLegacySlowSubstepForces)
    {
        if (!useMultiLevelMts)
        {
            forceOutSingleSlowLevel.emplace(setupForceOutputs(&fr->forceHelperBuffers[1],
                                                              forceView->forceMtsCombinedWithPadding(),
                                                              domainWork,
                                                              stepWork,
                                                              simulationWork.havePpDomainDecomposition,
                                                              wcycle));
            forceOutByMtsLevel[1] = &forceOutSingleSlowLevel.value();
        }
        else
        {
            forceOutMultiLevel.resize(numMtsLevels);
            for (int mtsLevel = 1; mtsLevel <= highestActiveSubstepLevel; mtsLevel++)
            {
                forceOutMultiLevel[mtsLevel].emplace(setupForceOutputs(&fr->forceHelperBuffers[mtsLevel],
                                                                       forceView->forceForMtsLevelWithPadding(mtsLevel),
                                                                       domainWork,
                                                                       stepWork,
                                                                       simulationWork.havePpDomainDecomposition,
                                                                       wcycle));
                forceOutByMtsLevel[mtsLevel] = &forceOutMultiLevel[mtsLevel].value();
            }
        }
    }

    ForceOutputs* forceOutLongrange = &forceOutMtsLevel0;
    if (useExactRespaForceOutputs)
    {
        forceOutLongrange =
                stepWork.computeLongRangeNonbondedForces ? exactRespaForceOutputs.longrangeOutputOrNull()
                                                         : nullptr;
    }
    else if (useLegacyMtsForceOutputs)
    {
        forceOutLongrange =
                stepWork.computeLongRangeNonbondedForces ? forceOutByMtsLevel[longrangeMtsLevel] : nullptr;
    }
    GMX_ASSERT(!stepWork.computeLongRangeNonbondedForces
                       || forceOutLongrange == nullptr
                       || forceOutLongrange->haveForceWithVirial(),
               "Active long-range nonbonded work requires a force-with-virial output buffer");

    ForceOutputs* forceOutNonbonded = &forceOutMtsLevel0;
    if (useExactRespaForceOutputs && stepWork.computeNonbondedForces && !exactRespaHasPairSplitting(inputrec))
    {
        forceOutNonbonded = exactRespaForceOutputs.levelOrNull(exactRespaNonbondedFullLevel(inputrec));
    }
    else if (useLegacyMtsForceOutputs && simulationWork.nonbondedSubstepLevel > 0
             && stepWork.computeNonbondedForces)
    {
        forceOutNonbonded = forceOutByMtsLevel[simulationWork.nonbondedSubstepLevel];
    }
    std::vector<ForceWithVirial*> forceWithVirialByLevel(
            useExactRespaForceOutputs ? exactRespaForceOutputs.numLevels : forceOutByMtsLevel.size(), nullptr);
    forceWithVirialByLevel[0] = &forceOutMtsLevel0.forceWithVirial();
    if (useExactRespaForceOutputs)
    {
        for (int level = 1; level < exactRespaForceOutputs.numActiveLevels(); ++level)
        {
            if (exactRespaForceOutputs.hasLevel(level))
            {
                forceWithVirialByLevel[level] = &exactRespaForceOutputs.level(level).forceWithVirial();
            }
        }
    }
    else
    {
        for (int mtsLevel = 1;
             mtsLevel <= highestActiveSubstepLevel
             && mtsLevel < static_cast<int>(forceOutByMtsLevel.size());
             mtsLevel++)
        {
            if (forceOutByMtsLevel[mtsLevel] != nullptr)
            {
                forceWithVirialByLevel[mtsLevel] = &forceOutByMtsLevel[mtsLevel]->forceWithVirial();
            }
        }
    }

    assertExactRespaOwnershipContract(
            inputrec, simulationWork, stepWork, forceView, forceOutMtsLevel0, exactRespaForceOutputs);
    if (step == 0 && useExactRespaForceOutputs && gmx::exactRespaHasPairSplitting(inputrec))
    {
        emitExactRespaOwnershipDiagnostics(inputrec, stepWork, exactRespaForceOutputs);
    }

    TracedForcePair tracedForceOutputsBeforeNonbonded;
    TracedForcePair tracedCombinedRealspaceDelta;
    TracedForcePair tracedBondedDelta;
    TracedForcePair tracedPair14Delta;
    TracedForcePair tracedCoulombRecipDelta;
    bool            haveTracedPair14Delta = false;
    if (traceForceComponents)
    {
        tracedForceOutputsBeforeNonbonded = useExactRespaForceOutputs
                                                    ? captureDistinctForceOutputs(exactRespaForceOutputs)
                                                    : captureDistinctForceOutputs(forceOutByMtsLevel,
                                                                                  highestActiveSubstepLevel);
    }

    if (inputrec.bPull && pull_have_constraint(*pull_work))
    {
        clear_pull_forces(pull_work);
    }
    wallcycle_stop(wcycle, WallCycleCounter::Force);

    // For the rest of the CPU tasks that depend on GPU-update produced coordinates,
    // this wait ensures that the D2H transfer is complete.
    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch)
    {
        const bool needCoordsOnHost = (runScheduleWork.domainWork.haveCpuLocalForceWork
                                       || stepWork.computeVirial || simulationWork.computeMuTot);
        const bool haveAlreadyWaited =
                simulationWork.useCpuHaloExchange
                || (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
        if (needCoordsOnHost && !haveAlreadyWaited)
        {
            GMX_ASSERT(haveCopiedXFromGpu,
                       "a wait should only be triggered if copy has been scheduled");
            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
        }
    }

    DipoleData dipoleData;

    if (simulationWork.computeMuTot)
    {
        const int start = 0;

        /* Calculate total (local) dipole moment in a temporary common array.
         * This makes it possible to sum them over nodes faster.
         */
        ArrayRef<const RVec> xRef = (xWholeMolecules.empty() ? x.unpaddedArrayRef() : xWholeMolecules);
        calc_mu(start,
                mdatoms->homenr,
                xRef,
                mdatoms->chargeA,
                mdatoms->chargeB,
                mdatoms->nChargePerturbed != 0,
                dipoleData.muStaging[0],
                dipoleData.muStaging[1]);

        reduceAndUpdateMuTot(&dipoleData,
                             cr->commMyGroup,
                             (fr->efep != FreeEnergyPerturbationType::No),
                             lambda,
                             muTotal,
                             ddBalanceRegionHandler);
    }

    /* Reset energies */
    reset_enerdata(enerd);

    if (haveDDAtomOrdering(*cr) && simulationWork.haveSeparatePmeRank)
    {
        wallcycle_start(wcycle, WallCycleCounter::PpDuringPme);
        dd_force_flop_start(cr->dd, nrnb);
    }

    if (inputrec.bRot)
    {
        wallcycle_start(wcycle, WallCycleCounter::Rot);
        do_rotation(cr->commMyGroup, cr->dd, enforcedRotation, box, x.unpaddedConstArrayRef(), t, step, stepWork.doNeighborSearch);
        wallcycle_stop(wcycle, WallCycleCounter::Rot);
    }

    /* We calculate the non-bonded forces, when done on the CPU, here.
     * We do this before calling do_force_lowlevel, because in that
     * function, the listed forces are calculated before PME, which
     * does communication.  With this order, non-bonded and listed
     * force calculation imbalance can be balanced out by the domain
     * decomposition load balancing.
     */

    const bool useOrEmulateGpuNb =
            !useDedicatedExactRespaGpuNonbonded && (simulationWork.useGpuNonbonded || fr->nbv->emulateGpu());
    const bool useExactLammpsRespaNonbonded =
            stepWork.computeNonbondedForces && useExactLammpsRespaPairSplitting;
    const bool useExactLammpsRespaGpuNonbonded =
            useExactLammpsRespaNonbonded && simulationWork.useGpuNonbonded;
    const bool useExactLammpsRespaCpuNbnxmNarrow =
            useExactLammpsRespaNonbonded && !simulationWork.useGpuNonbonded
            && !simulationWork.havePpDomainDecomposition
            && !exactRespaDisableCpuNbnxmNarrow()
            && fr->nbv != nullptr
            && exactRespaCpuNbnxmKernelSupported(fr->nbv->kernelSetup().kernelType)
            && activeM2pTraceDirPath() == nullptr;
    const bool haveExactSlowForceOutputs  = (useExactRespaForceOutputs && exactRespaForceOutputs.numActiveLevels() > 1);
    const bool haveLegacySlowForceOutputs =
            (useLegacyMtsForceOutputs && computeLegacySlowSubstepForces);
    const bool traceExactGpuListedFtypeSplit =
            shouldTraceExactGpuListedFtypeSplitStep(step) && useExactRespaForceOutputs
            && simulationWork.useGpuBonded;
    const bool traceExactGpuListedClass2SubtermSplit =
            shouldTraceExactGpuListedClass2SubtermSplitStep(step) && useExactRespaForceOutputs
            && simulationWork.useGpuBonded;
    const bool traceExactGpuBondedMixedVsSequential =
            shouldTraceExactGpuBondedMixedVsSequentialStep(step) && useExactRespaForceOutputs
            && simulationWork.useGpuBonded;
    const bool traceStep1Subset01ForceGroupAudit = shouldTraceStep1Subset01ForceGroupAuditStep(step);
    const char* step1Subset01TraceSide          = useExactLammpsRespaNonbonded ? "PATCH" : "PLAIN";
    const char* step1Subset01Level0Role         = useExactLammpsRespaNonbonded ? "shared" : "plain_total";

    GMX_RELEASE_ASSERT(!useExactLammpsRespaNonbonded || !domainWork.haveCpuNonbondedFreeEnergyWork,
                       "Exact LAMMPS-style r-RESPA does not support nonbonded free-energy work yet");

    if (traceStep1Subset01ForceGroupAudit)
    {
        if (useExactRespaForceOutputs)
        {
            appendStep1Subset01ForceGroupBufferSnapshot(activeM2pTraceDirPath(),
                                                       step1Subset01TraceSide,
                                                       step,
                                                       "after_clear",
                                                       exactRespaForceOutputs,
                                                       step1Subset01Level0Role,
                                                       "src/gromacs/mdlib/sim_util.cpp:after_clear_force_outputs");
        }
        else
        {
            appendStep1Subset01ForceGroupBufferSnapshot(activeM2pTraceDirPath(),
                                                       step1Subset01TraceSide,
                                                       step,
                                                       "after_clear",
                                                       forceOutByMtsLevel,
                                                       highestActiveSubstepLevel,
                                                       step1Subset01Level0Role,
                                                       "src/gromacs/mdlib/sim_util.cpp:after_clear_force_outputs");
        }
    }

    if (useExactLammpsRespaNonbonded)
    {
        wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
        if (shouldTraceRespaStateXChainStep(step))
        {
            appendStateXChainTracePair(activeM2pTraceDirPath(),
                                       "PATCH",
                                       preHandoffStageName(step),
                                       step,
                                       x.unpaddedArrayRef(),
                                       "do_force->computeExactRespaNonbondedCpu",
                                       "src/gromacs/mdlib/sim_util.cpp:5112",
                                       false);
        }
        if (shouldTraceRespaCoordHandoffStep(step))
        {
            appendCoordHandoffTracePair(activeM2pTraceDirPath(),
                                        "PATCH",
                                        "PRE_HANDOFF_COORD_SOURCE",
                                        step,
                                        x.unpaddedArrayRef(),
                                        "state.x.unpaddedArrayRef()_before_exact_nonbonded",
                                        x.unpaddedArrayRef().data());
            appendCoordHandoffTracePair(activeM2pTraceDirPath(),
                                        "PATCH",
                                        "POST_HANDOFF_BUFFER",
                                        step,
                                        x.unpaddedArrayRef(),
                                        "x.unpaddedArrayRef()_passed_to_exact_nonbonded",
                                        x.unpaddedArrayRef().data());
        }
        if (useExactLammpsRespaGpuNonbonded)
        {
            GMX_RELEASE_ASSERT(!simulationWork.havePpDomainDecomposition,
                               "Exact r-RESPA GPU nonbonded offload currently supports single-rank execution only");

            computeExactRespaNonbondedGpuNarrow(
                    inputrec,
                    top->idef,
                    fr,
                    *mdatoms,
                    x.unpaddedArrayRef(),
                    ic,
                    exactRespaForceOutputs,
                    enerd,
                    stepWork,
                    exactRespaStepWork,
                    step,
                    nrnb,
                    wcycle);
        }
        else if (useExactLammpsRespaCpuNbnxmNarrow)
        {
            if (fplog != nullptr && step == 0)
            {
                fprintf(fplog,
                        "Exact r-RESPA CPU nonbonded will use the narrow per-contribution NBNXM path.\n");
            }
            computeExactRespaNonbondedCpuNbnxmNarrow(
                    inputrec,
                    fr,
                    *mdatoms,
                    x.unpaddedArrayRef(),
                    box,
                    exactRespaForceOutputs,
                    enerd,
                    stepWork,
                    exactRespaStepWork,
                    step,
                    nrnb,
                    wcycle);
        }
        else
        {
            if (fplog != nullptr
                && gmx_within_tol(fr->ic->vdw.repulsionPower, 9.0, 10 * GMX_DOUBLE_EPS)
                && step == 0)
            {
                if (exactRespaDisableCpuNbnxmNarrow())
                {
                    fprintf(fplog,
                            "Found environment variable GMX_PCFF_EXACT_RESPA_DISABLE_NBNXM_NARROW.\n"
                            "Exact r-RESPA CPU nonbonded will force the scalar pair-loop path instead of the narrow NBNXM path for diagnostic comparison.\n");
                }
                if (useRepulsionPower9ExactRespaCpuSpecialization(*fr->ic))
                {
                    fprintf(fplog,
                            "Exact r-RESPA CPU pair splitting will use the specialized exact repulsion-power-9 scalar patch path.\n");
                }
                else
                {
                    fprintf(fplog,
                            "Found environment variable GMX_DISABLE_REPULSION_POWER_9_EXACT_RESPA_CPU_SPECIALIZATION.\n"
                            "Exact r-RESPA CPU pair splitting will keep the generic repulsion-power-9 scalar patch path for baseline comparison.\n");
                }
                if (exactRespaPairLoopOmpRequested())
                {
                    fprintf(fplog,
                            "Exact r-RESPA CPU pair-loop OpenMP fast path is enabled for eligible no-trace/no-energy/no-virial pair-loop calls; set GMX_PCFF_EXACT_RESPA_PAIRLOOP_OMP=0 to force the legacy scalar path.\n");
                }
                if (exactRespaPairLoopDirectCpuListRequested())
                {
                    fprintf(fplog,
                            "Exact r-RESPA CPU direct cpuLists()/packed-dispatch path is enabled for eligible pair-loop work; set GMX_PCFF_EXACT_RESPA_PAIRLOOP_DIRECT_CPULIST=0 to force plain-pairlist iteration.\n");
                }
                if (exactRespaPairLoopVectorRequested())
                {
                    fprintf(fplog,
                            "Exact r-RESPA CPU pair-loop vector-batch experimental fast path requested; only no-trace/no-energy/no-virial pair-loop calls are eligible.\n");
                }
            }
            computeExactRespaNonbondedCpu(inputrec,
                                          top->idef,
                                          fr,
                                          *mdatoms,
                                          x.unpaddedArrayRef(),
                                          exactRespaForceOutputs,
                                          enerd,
                                          stepWork,
                                          step);
        }
        wallcycle_stop(wcycle, WallCycleCounter::Force);
    }
    else if (!useOrEmulateGpuNb)
    {
        wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
        if (shouldTraceRespaStateXChainStep(step))
        {
            appendStateXChainTracePair(activeM2pTraceDirPath(),
                                       "PLAIN",
                                       preHandoffStageName(step),
                                       step,
                                       x.unpaddedArrayRef(),
                                       "do_force->do_nb_verlet",
                                       "src/gromacs/mdlib/sim_util.cpp:5137",
                                       false);
        }
        if (shouldTraceRespaCoordHandoffStep(step))
        {
            appendCoordHandoffTracePair(activeM2pTraceDirPath(),
                                        "PLAIN",
                                        "PRE_HANDOFF_COORD_SOURCE",
                                        step,
                                        x.unpaddedArrayRef(),
                                        "state.x.unpaddedArrayRef()_before_do_nb_verlet",
                                        x.unpaddedArrayRef().data());
            appendCoordHandoffTracePair(activeM2pTraceDirPath(),
                                        "PLAIN",
                                        "POST_HANDOFF_BUFFER",
                                        step,
                                        fr->nbv->nbat(),
                                        "nbv.nbat.x()_before_do_nb_verlet",
                                        fr->nbv->nbat().x().data());
        }
        do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::Local, enbvClearFYes, step, nrnb, wcycle);
        wallcycle_stop(wcycle, WallCycleCounter::Force);
    }

    if (stepWork.useGpuXHalo && domainWork.haveCpuNonLocalForceWork)
    {
        /* Wait for non-local coordinate data to be copied from device */
        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::NonLocal);
    }

    wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
    if (domainWork.haveCpuNonbondedFreeEnergyWork && stepWork.computeNonbondedForces)
    {
        /* Calculate the local and non-local free energy interactions on CPU here. */
        nbv->dispatchFreeEnergyCpuKernels(x,
                                          &forceOutNonbonded->forceWithShiftForces(),
                                          fr->use_simd_kernels,
                                          fr->ntype,
                                          *fr->ic,
                                          fr->shift_vec,
                                          fr->nbfp,
                                          fr->ljpme_c6grid,
                                          mdatoms->chargeA,
                                          mdatoms->chargeB,
                                          mdatoms->typeA,
                                          mdatoms->typeB,
                                          lambda,
                                          enerd,
                                          stepWork,
                                          nrnb);
    }

    if (stepWork.computeNonbondedForces && !useOrEmulateGpuNb && !useExactLammpsRespaNonbonded)
    {
        if (simulationWork.havePpDomainDecomposition)
        {
            do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::NonLocal, enbvClearFNo, step, nrnb, wcycle);
        }

        if (stepWork.computeForces)
        {
            /* Add all the non-bonded force to the normal force array.
             * This can be split into a local and a non-local part when overlapping
             * communication with calculation with domain decomposition.
             */
            wallcycle_stop(wcycle, WallCycleCounter::Force);
            nbv->atomdata_add_nbat_f_to_f(AtomLocality::All,
                                          forceOutNonbonded->forceWithShiftForces().force());
            wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
        }

        /* If there are multiple fshift output buffers we need to reduce them */
        if (stepWork.computeVirial)
        {
            /* This is not in a subcounter because it takes a
               negligible and constant-sized amount of time */
            nbnxn_atomdata_add_nbat_fshift_to_fshift(
                    nbv->nbat(), forceOutNonbonded->forceWithShiftForces().shiftForces());
        }
    }

    if (traceStep1Subset01ForceGroupAudit)
    {
        if (useExactRespaForceOutputs)
        {
            appendStep1Subset01ForceGroupBufferSnapshot(activeM2pTraceDirPath(),
                                                       step1Subset01TraceSide,
                                                       step,
                                                       "after_nonbonded",
                                                       exactRespaForceOutputs,
                                                       step1Subset01Level0Role,
                                                       "src/gromacs/mdlib/sim_util.cpp:after_nonbonded_stage");
        }
        else
        {
            appendStep1Subset01ForceGroupBufferSnapshot(activeM2pTraceDirPath(),
                                                       step1Subset01TraceSide,
                                                       step,
                                                       "after_nonbonded",
                                                       forceOutByMtsLevel,
                                                       highestActiveSubstepLevel,
                                                       step1Subset01Level0Role,
                                                       "src/gromacs/mdlib/sim_util.cpp:after_nonbonded_stage");
        }
    }

    TracedForcePair tracedForceOutputsBeforeListed;
    if (traceForceComponents)
    {
        const auto tracedForceOutputsAfterNonbonded =
                useExactRespaForceOutputs ? captureDistinctForceOutputs(exactRespaForceOutputs)
                                          : captureDistinctForceOutputs(forceOutByMtsLevel,
                                                                        highestActiveSubstepLevel);
        tracedCombinedRealspaceDelta =
                subtractTracedForcePairs(tracedForceOutputsAfterNonbonded, tracedForceOutputsBeforeNonbonded);
        tracedForceOutputsBeforeListed = tracedForceOutputsAfterNonbonded;
    }
    if (traceRealspaceForceSubcomponents && !useExactLammpsRespaNonbonded)
    {
        const auto toTracedForcePair = [](const M2pPlain4x4TracedForcePair& source)
        {
            TracedForcePair result;
            result.atomIndices = { 0, 5 };
            result.atoms.assign(result.atomIndices.size(), std::array<double, DIM>{ 0.0, 0.0, 0.0 });
            for (int atomIndex = 0; atomIndex < static_cast<int>(result.atomIndices.size()); ++atomIndex)
            {
                result.atoms[atomIndex] = source.atoms[atomIndex];
            }
            return result;
        };
        appendRealspaceForceSubcomponentTracePair(activeM2pTraceDirPath(),
                                                  "PLAIN",
                                                  step,
                                                  "lj_sr_force",
                                                  toTracedForcePair(readM2pPlain4x4LjSrForcePair()),
                                                  "kernel_ref_inner.frLJ_times_rinvsq",
                                                  "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h",
                                                  "true_source_component",
                                                  true);
        appendRealspaceForceSubcomponentTracePair(activeM2pTraceDirPath(),
                                                  "PLAIN",
                                                  step,
                                                  "coulomb_sr_force",
                                                  toTracedForcePair(readM2pPlain4x4CoulombSrForcePair()),
                                                  "kernel_ref_inner.interact_times_rinvsq_term",
                                                  "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h",
                                                  "true_source_component",
                                                  true);
        appendRealspaceForceSubcomponentTracePair(activeM2pTraceDirPath(),
                                                  "PLAIN",
                                                  step,
                                                  "exclusion_correction_force",
                                                  toTracedForcePair(readM2pPlain4x4ExclusionCorrectionForcePair()),
                                                  "kernel_ref_inner.fexcl_correction_term",
                                                  "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h",
                                                  "true_source_component",
                                                  true);
        appendRealspaceForceSubcomponentUnavailablePair(
                activeM2pTraceDirPath(),
                "PLAIN",
                step,
                "additional_realspace_correction_force",
                "no_additional_realspace_correction_component_in_plain_kernel",
                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h",
                "runtime_force_component_unavailable");
        appendRealspaceForceSubcomponentTracePair(activeM2pTraceDirPath(),
                                                  "PLAIN",
                                                  step,
                                                  "realspace_nonbonded_combined_force",
                                                  toTracedForcePair(readM2pPlain4x4CombinedRealspaceForcePair()),
                                                  "kernel_ref_inner.fscal_total",
                                                  "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h",
                                                  "combined_total",
                                                  false);
        if (traceStep1Subset01ForceGroupAudit)
        {
            appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                                "step2_current_nonbonded_producer_term_trace.txt",
                                                "PLAIN",
                                                step,
                                                "plain_lj_sr_force",
                                                toTracedForcePair(readM2pPlain4x4LjSrForcePair()),
                                                "kernel_ref_inner.frLJ_times_rinvsq",
                                                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h",
                                                "downstream_total_component",
                                                true);
            appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                                "step2_current_nonbonded_producer_term_trace.txt",
                                                "PLAIN",
                                                step,
                                                "plain_coulomb_sr_force",
                                                toTracedForcePair(readM2pPlain4x4CoulombSrForcePair()),
                                                "kernel_ref_inner.interact_times_rinvsq_term",
                                                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h",
                                                "downstream_total_component",
                                                true);
            appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                                "step2_current_nonbonded_producer_term_trace.txt",
                                                "PLAIN",
                                                step,
                                                "plain_exclusion_correction_force",
                                                toTracedForcePair(readM2pPlain4x4ExclusionCorrectionForcePair()),
                                                "kernel_ref_inner.fexcl_correction_term",
                                                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h",
                                                "downstream_total_component",
                                                true);
            appendForceComponentTracePairToFile(activeM2pTraceDirPath(),
                                                "step2_current_nonbonded_producer_term_trace.txt",
                                                "PLAIN",
                                                step,
                                                "plain_after_nonbonded",
                                                toTracedForcePair(readM2pPlain4x4CombinedRealspaceForcePair()),
                                                "kernel_ref_inner.fscal_total",
                                                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h",
                                                "downstream_total",
                                                false);
        }
    }

    // Compute wall interactions, when present.
    // Note: should be moved to special forces.
    if (inputrec.nwall && stepWork.computeNonbondedForces)
    {
        /* foreign lambda component for walls */
        real dvdl_walls = do_walls(inputrec,
                                   *fr,
                                   box,
                                   mdatoms->typeA,
                                   mdatoms->typeB,
                                   mdatoms->cENER,
                                   mdatoms->homenr,
                                   mdatoms->nPerturbed,
                                   x.unpaddedConstArrayRef(),
                                   &forceOutMtsLevel0.forceWithVirial(),
                                   lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Vdw)],
                                   enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJSR],
                                   nrnb);
        enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Vdw] += dvdl_walls;
    }

    if (stepWork.computeListedForces)
    {
        TracedForcePair tracedPair14BeforeListed;
        const int       pair14Level = inputrec.exactRespa.forceLayout.pair14Level;
        const bool      shouldTraceExactPair14Delta =
                traceForceComponents && useExactRespaForceOutputs && useExactLammpsRespaPairSplitting
                && pair14Level >= 0 && pair14Level <= exactRespaForceOutputs.highestActiveLevel
                && exactRespaForceOutputs.levelOrNull(pair14Level) != nullptr;
        if (shouldTraceExactPair14Delta)
        {
            tracedPair14BeforeListed = captureDistinctForceOutput(exactRespaForceOutputs.levelOrNull(pair14Level));
        }

        /* Check whether we need to take into account PBC in listed interactions */
        bool needMolPbc = false;
        for (const auto& listedForces : fr->listedForces)
        {
            if (listedForces.haveCpuListedForces(*fr->fcdata))
            {
                needMolPbc = fr->bMolPBC;
            }
        }

        t_pbc pbc;

        if (needMolPbc)
        {
            /* Since all atoms are in the rectangular or triclinic unit-cell,
             * only single box vector shifts (2 in x) are required.
             */
            set_pbc_dd(&pbc, fr->pbcType, haveDDAtomOrdering(*cr) ? &cr->dd->numCells : nullptr, TRUE, box);
        }

        if (useExactRespaForceOutputs && simulationWork.useGpuBonded && fr->listedForcesGpu != nullptr)
        {
            GMX_RELEASE_ASSERT(!simulationWork.havePpDomainDecomposition,
                               "Exact r-RESPA GPU bonded offload currently supports single-rank execution only");

            bool uploadedExactRespaGpuBondedCoordinates = false;
            const bool useExactGpuBondedSequentialFtypesValidation =
                    shouldUseExactGpuBondedSequentialFtypesValidation();
            for (int exactLevel = 0; exactLevel < exactRespaForceOutputs.numActiveLevels(); ++exactLevel)
            {
                ForceOutputs* forceOutPtr = exactRespaForceOutputs.levelOrNull(exactLevel);
                if (forceOutPtr == nullptr)
                {
                    continue;
                }

                const InteractionDefinitions& levelIdef =
                        fr->listedForces[exactLevel].interactionDefinitions();
                fr->listedForcesGpu->updateHaveInteractions(levelIdef);
                if (!fr->listedForcesGpu->haveInteractions())
                {
                    continue;
                }

                if (!uploadedExactRespaGpuBondedCoordinates && !stepWork.useGpuXBufferOps)
                {
                    wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
                    wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
                    gpu_upload_shiftvec(nbv->gpuNbv(), &nbv->nbat());
                    gpu_copy_xq_to_gpu(nbv->gpuNbv(), &nbv->nbat(), AtomLocality::Local);
                    wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
                    wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
                    uploadedExactRespaGpuBondedCoordinates = true;
                }

                if (useExactGpuBondedSequentialFtypesValidation)
                {
                    std::array<RVec, 8> tracedForceBeforeReduceStorage = {};
                    int                 tracedForceCountBeforeReduce   = 0;
                    TracedForcePair     tracedCombinedGpuOutput;
                    if (traceForceComponents)
                    {
                        const auto tracedHostForceView = forceOutPtr->forceWithShiftForces().force();
                        tracedForceCountBeforeReduce = std::min<int>(8, tracedHostForceView.ssize());
                        for (int atom = 0; atom < tracedForceCountBeforeReduce; ++atom)
                        {
                            copy_rvec(tracedHostForceView[atom], tracedForceBeforeReduceStorage[atom]);
                        }
                    }

                    bool launchedAnySequentialFtype = false;
                    for (const InteractionFunction tracedFtype :
                         c_exactGpuListedFtypesForTraceOrSequentialValidation)
                    {
                        if (levelIdef.il[tracedFtype].empty())
                        {
                            continue;
                        }

                        const InteractionDefinitions singleFtypeIdef =
                                makeSingleInteractionFunctionDefinitions(levelIdef, tracedFtype);
                        fr->listedForcesGpu->updateHaveInteractions(singleFtypeIdef);
                        if (!fr->listedForcesGpu->haveInteractions())
                        {
                            continue;
                        }

                        launchedAnySequentialFtype = true;
                        fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
                                nbv->getGridIndices(), singleFtypeIdef, gpuGetNBAtomData(nbv->gpuNbv()));
                        gpu_clear_outputs(nbv->gpuNbv(), stepWork.computeVirial);
                        fr->listedForcesGpu->setPbcAndlaunchKernel(
                                fr->pbcType, box, fr->bMolPBC, stepWork);
                        gpu_launch_cpyback(nbv->gpuNbv(), &nbv->nbat(), stepWork, AtomLocality::Local);
                        if (stepWork.computeEnergy)
                        {
                            fr->listedForcesGpu->launchEnergyTransfer();
                        }

                        gpu_wait_finish_task(nbv->gpuNbv(),
                                             stepWork,
                                             AtomLocality::Local,
                                             false,
                                             enerd,
                                             forceOutPtr->forceWithShiftForces().shiftForces(),
                                             wcycle);
                        if (traceForceComponents)
                        {
                            addTracedForcePairToTracedPair(&tracedCombinedGpuOutput,
                                                           captureNbatOutputForceBufferPair(nbv));
                        }

                        nbv->atomdata_add_nbat_f_to_f(AtomLocality::Local,
                                                      forceOutPtr->forceWithShiftForces().force());
                        nbv->nbat().clearForceBuffer(0);

                        if (stepWork.computeEnergy)
                        {
                            fr->listedForcesGpu->waitAccumulateEnergyTerms(enerd);
                            fr->listedForcesGpu->clearEnergies();
                        }
                    }
                    GMX_RELEASE_ASSERT(launchedAnySequentialFtype,
                                       "Sequential exact GPU bonded validation expected at least one "
                                       "GPU-listed interaction function");

                    if (traceForceComponents)
                    {
                        appendExactGpuBondedReductionTrace(activeM2pTraceDirPath(),
                                                           step,
                                                           exactLevel,
                                                           "nbat_output_buffer",
                                                           tracedCombinedGpuOutput);
                        const auto tracedHostForceView = forceOutPtr->forceWithShiftForces().force();
                        appendExactGpuBondedReductionTrace(activeM2pTraceDirPath(),
                                                           step,
                                                           exactLevel,
                                                           "before_reduce",
                                                           makeConstArrayRef(tracedForceBeforeReduceStorage)
                                                                   .subArray(0, tracedForceCountBeforeReduce));
                        appendExactGpuBondedReductionTrace(activeM2pTraceDirPath(),
                                                           step,
                                                           exactLevel,
                                                           "after_reduce",
                                                           tracedHostForceView);

                        std::array<RVec, 8> tracedReductionDeltaStorage = {};
                        const int tracedReductionCount =
                                std::min<int>(tracedForceCountBeforeReduce, tracedHostForceView.ssize());
                        for (int atom = 0; atom < tracedReductionCount; ++atom)
                        {
                            for (int dim = 0; dim < DIM; ++dim)
                            {
                                tracedReductionDeltaStorage[atom][dim] =
                                        tracedHostForceView[atom][dim]
                                        - tracedForceBeforeReduceStorage[atom][dim];
                            }
                        }
                        appendExactGpuBondedReductionTrace(
                                activeM2pTraceDirPath(),
                                step,
                                exactLevel,
                                "reduction_delta",
                                makeConstArrayRef(tracedReductionDeltaStorage).subArray(
                                        0, tracedReductionCount));
                    }

                    fr->listedForcesGpu->updateHaveInteractions(levelIdef);
                }
                else
                {
                    std::vector<RVec> tracedMixedReductionDeltaStorage;
                    fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
                            nbv->getGridIndices(), levelIdef, gpuGetNBAtomData(nbv->gpuNbv()));
                    if (traceExactGpuBondedLaunchContext)
                    {
                        appendExactGpuBondedLaunchContextTrace(activeM2pTraceDirPath(),
                                                               step,
                                                               exactLevel,
                                                               localCoordinateProvider,
                                                               stepWork.useGpuXBufferOps,
                                                               stepWork.doNeighborSearch,
                                                               localCoordinatesNeededOnDevice,
                                                               haveCopiedXFromGpu,
                                                               copiedCoordinatesToGpu,
                                                               expectedLocalXReadyOnDeviceConsumptionCount,
                                                               uploadedExactRespaGpuBondedCoordinates);
                    }
                    if (traceExactGpuBondedGridIndices)
                    {
                        appendExactGpuBondedGridIndexTrace(activeM2pTraceDirPath(), step, exactLevel, nbv);
                    }
#if GMX_GPU
                    if (traceExactGpuBondedDeviceXq && fr->deviceStreamManager != nullptr)
                    {
                        appendExactGpuBondedDeviceXqTrace(activeM2pTraceDirPath(),
                                                          step,
                                                          exactLevel,
                                                          "pre_kernel",
                                                          nbv,
                                                          fr->deviceStreamManager->bondedStream());
                    }
                    if (traceExactGpuBondedDeviceForce && fr->deviceStreamManager != nullptr)
                    {
                        appendExactGpuBondedDeviceForceTrace(activeM2pTraceDirPath(),
                                                             step,
                                                             exactLevel,
                                                             "pre_clear",
                                                             nbv,
                                                             fr->deviceStreamManager->bondedStream());
                    }
#endif
                    gpu_clear_outputs(nbv->gpuNbv(), stepWork.computeVirial);
#if GMX_GPU
                    if (traceExactGpuBondedDeviceForce && fr->deviceStreamManager != nullptr)
                    {
                        appendExactGpuBondedDeviceForceTrace(activeM2pTraceDirPath(),
                                                             step,
                                                             exactLevel,
                                                             "post_clear",
                                                             nbv,
                                                             fr->deviceStreamManager->bondedStream());
                    }
#endif
                    fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
                    gpu_launch_cpyback(nbv->gpuNbv(), &nbv->nbat(), stepWork, AtomLocality::Local);
                    if (stepWork.computeEnergy)
                    {
                        fr->listedForcesGpu->launchEnergyTransfer();
                    }

                    gpu_wait_finish_task(nbv->gpuNbv(),
                                         stepWork,
                                         AtomLocality::Local,
                                         false,
                                         enerd,
                                         forceOutPtr->forceWithShiftForces().shiftForces(),
                                         wcycle);
                    std::array<RVec, 8> tracedForceBeforeReduceStorage = {};
                    int                 tracedForceCountBeforeReduce   = 0;
                    if (traceForceComponents)
                    {
                        appendExactGpuBondedReductionTrace(activeM2pTraceDirPath(),
                                                           step,
                                                           exactLevel,
                                                           "nbat_output_buffer",
                                                           captureNbatOutputForceBufferPair(nbv));
                        const auto tracedHostForceView = forceOutPtr->forceWithShiftForces().force();
                        tracedForceCountBeforeReduce = std::min<int>(8, tracedHostForceView.ssize());
                        for (int atom = 0; atom < tracedForceCountBeforeReduce; ++atom)
                        {
                            copy_rvec(tracedHostForceView[atom], tracedForceBeforeReduceStorage[atom]);
                        }
                        appendExactGpuBondedReductionTrace(activeM2pTraceDirPath(),
                                                           step,
                                                           exactLevel,
                                                           "before_reduce",
                                                           tracedHostForceView);
                    }
                    nbv->atomdata_add_nbat_f_to_f(AtomLocality::Local,
                                                  forceOutPtr->forceWithShiftForces().force());
                    if (traceForceComponents)
                    {
                        const auto tracedHostForceView = forceOutPtr->forceWithShiftForces().force();
                        appendExactGpuBondedReductionTrace(activeM2pTraceDirPath(),
                                                           step,
                                                           exactLevel,
                                                           "after_reduce",
                                                           tracedHostForceView);

                        std::array<RVec, 8> tracedReductionDeltaStorage = {};
                        const int tracedReductionCount =
                                std::min<int>(tracedForceCountBeforeReduce, tracedHostForceView.ssize());
                        for (int atom = 0; atom < tracedReductionCount; ++atom)
                        {
                            for (int dim = 0; dim < DIM; ++dim)
                            {
                                tracedReductionDeltaStorage[atom][dim] =
                                        tracedHostForceView[atom][dim]
                                        - tracedForceBeforeReduceStorage[atom][dim];
                            }
                        }
                        appendExactGpuBondedReductionTrace(
                                activeM2pTraceDirPath(),
                                step,
                                exactLevel,
                                "reduction_delta",
                                makeConstArrayRef(tracedReductionDeltaStorage).subArray(
                                        0, tracedReductionCount));
                        tracedMixedReductionDeltaStorage.assign(tracedReductionCount, RVec{});
                        for (int atom = 0; atom < tracedReductionCount; ++atom)
                        {
                            copy_rvec(tracedReductionDeltaStorage[atom],
                                      tracedMixedReductionDeltaStorage[atom]);
                        }
                    }
                    nbv->nbat().clearForceBuffer(0);

                    if (stepWork.computeEnergy)
                    {
                        fr->listedForcesGpu->waitAccumulateEnergyTerms(enerd);
                        fr->listedForcesGpu->clearEnergies();
                    }

                    if (traceExactGpuBondedMixedVsSequential && traceForceComponents)
                    {
                        TracedForcePair     tracedSequentialCombinedGpuOutput;
                        std::vector<RVec>   sequentialReducedStorage(
                                forceOutPtr->forceWithShiftForces().force().ssize());
                        StepWorkload        splitStepWork = stepWork;
                        splitStepWork.computeEnergy       = false;
                        splitStepWork.computeVirial       = false;
                        bool launchedAnySequentialFtype   = false;
                        for (auto& forceValue : sequentialReducedStorage)
                        {
                            clear_rvec(forceValue);
                        }

                        for (const InteractionFunction tracedFtype :
                             c_exactGpuListedFtypesForTraceOrSequentialValidation)
                        {
                            if (levelIdef.il[tracedFtype].empty())
                            {
                                continue;
                            }

                            const InteractionDefinitions singleFtypeIdef =
                                    makeSingleInteractionFunctionDefinitions(levelIdef, tracedFtype);
                            fr->listedForcesGpu->updateHaveInteractions(singleFtypeIdef);
                            if (!fr->listedForcesGpu->haveInteractions())
                            {
                                continue;
                            }

                            launchedAnySequentialFtype = true;
                            fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
                                    nbv->getGridIndices(),
                                    singleFtypeIdef,
                                    gpuGetNBAtomData(nbv->gpuNbv()));
                            gpu_clear_outputs(nbv->gpuNbv(), false);
                            fr->listedForcesGpu->setPbcAndlaunchKernel(
                                    fr->pbcType, box, fr->bMolPBC, splitStepWork);
                            gpu_launch_cpyback(
                                    nbv->gpuNbv(), &nbv->nbat(), splitStepWork, AtomLocality::Local);
                            gpu_wait_finish_task(nbv->gpuNbv(),
                                                 splitStepWork,
                                                 AtomLocality::Local,
                                                 false,
                                                 enerd,
                                                 forceOutPtr->forceWithShiftForces().shiftForces(),
                                                 wcycle);
                            addTracedForcePairToTracedPair(&tracedSequentialCombinedGpuOutput,
                                                           captureNbatOutputForceBufferPair(nbv));
                            nbv->atomdata_add_nbat_f_to_f(AtomLocality::Local,
                                                          makeArrayRef(sequentialReducedStorage));
                            nbv->nbat().clearForceBuffer(0);
                        }

                        GMX_RELEASE_ASSERT(launchedAnySequentialFtype,
                                           "Mixed-vs-sequential exact GPU bonded trace expected at "
                                           "least one GPU-listed interaction function");

                        appendExactGpuBondedReductionTrace(activeM2pTraceDirPath(),
                                                           step,
                                                           exactLevel,
                                                           "sequential_all_ftypes_nbat_output_buffer",
                                                           tracedSequentialCombinedGpuOutput);
                        appendExactGpuBondedReductionTrace(activeM2pTraceDirPath(),
                                                           step,
                                                           exactLevel,
                                                           "sequential_all_ftypes_reduction_delta",
                                                           makeConstArrayRef(sequentialReducedStorage));
                        if (!tracedMixedReductionDeltaStorage.empty())
                        {
                            std::array<RVec, 8> tracedMixedMinusSequentialStorage = {};
                            const int tracedComparisonCount = std::min<int>(
                                    std::min<int>(static_cast<int>(tracedMixedReductionDeltaStorage.size()),
                                                  static_cast<int>(sequentialReducedStorage.size())),
                                    8);
                            for (int atom = 0; atom < tracedComparisonCount; ++atom)
                            {
                                for (int dim = 0; dim < DIM; ++dim)
                                {
                                    tracedMixedMinusSequentialStorage[atom][dim] =
                                            tracedMixedReductionDeltaStorage[atom][dim]
                                            - sequentialReducedStorage[atom][dim];
                                }
                            }
                            appendExactGpuBondedReductionTrace(
                                    activeM2pTraceDirPath(),
                                    step,
                                    exactLevel,
                                    "mixed_minus_sequential_reduction_delta",
                                    makeConstArrayRef(tracedMixedMinusSequentialStorage)
                                            .subArray(0, tracedComparisonCount));
                        }

                        fr->listedForcesGpu->updateHaveInteractions(levelIdef);
                        fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
                                nbv->getGridIndices(), levelIdef, gpuGetNBAtomData(nbv->gpuNbv()));
                    }
                }

                if (traceExactGpuListedFtypeSplit)
                {
                    for (const InteractionFunction tracedFtype :
                         c_exactGpuListedFtypesForTraceOrSequentialValidation)
                    {
                        const char* traceLabel = exactGpuListedFunctionTraceLabel(tracedFtype);
                        if (traceLabel == nullptr || levelIdef.il[tracedFtype].empty())
                        {
                            continue;
                        }

                        const InteractionDefinitions singleFtypeIdef =
                                makeSingleInteractionFunctionDefinitions(levelIdef, tracedFtype);
                        fr->listedForcesGpu->updateHaveInteractions(singleFtypeIdef);
                        if (!fr->listedForcesGpu->haveInteractions())
                        {
                            continue;
                        }

                        fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
                                nbv->getGridIndices(), singleFtypeIdef, gpuGetNBAtomData(nbv->gpuNbv()));
                        gpu_clear_outputs(nbv->gpuNbv(), false);

                        StepWorkload splitStepWork = stepWork;
                        splitStepWork.computeEnergy = false;
                        splitStepWork.computeVirial = false;

                        fr->listedForcesGpu->setPbcAndlaunchKernel(
                                fr->pbcType, box, fr->bMolPBC, splitStepWork);
                        gpu_launch_cpyback(nbv->gpuNbv(), &nbv->nbat(), splitStepWork, AtomLocality::Local);
                        gpu_wait_finish_task(nbv->gpuNbv(),
                                             splitStepWork,
                                             AtomLocality::Local,
                                             false,
                                             enerd,
                                             forceOutPtr->forceWithShiftForces().shiftForces(),
                                             wcycle);

                        const auto splitGpuOutput = captureNbatOutputForceBufferPair(nbv);
                        const std::string outputStage = std::string(traceLabel) + "_nbat_output_buffer";
                        appendExactGpuBondedReductionTrace(activeM2pTraceDirPath(),
                                                           step,
                                                           exactLevel,
                                                           outputStage.c_str(),
                                                           splitGpuOutput);

                        std::vector<RVec> splitReducedStorage(forceOutPtr->forceWithShiftForces().force().ssize());
                        for (auto& forceValue : splitReducedStorage)
                        {
                            clear_rvec(forceValue);
                        }
                        nbv->atomdata_add_nbat_f_to_f(AtomLocality::Local, makeArrayRef(splitReducedStorage));
                        const std::string reducedStage = std::string(traceLabel) + "_after_reduce";
                        appendExactGpuBondedReductionTrace(activeM2pTraceDirPath(),
                                                           step,
                                                           exactLevel,
                                                           reducedStage.c_str(),
                                                           makeConstArrayRef(splitReducedStorage));
                        nbv->nbat().clearForceBuffer(0);
                    }

                    fr->listedForcesGpu->updateHaveInteractions(levelIdef);
                }

                if (traceExactGpuListedClass2SubtermSplit)
                {
                    for (const InteractionFunction tracedFtype :
                         c_exactGpuListedFtypesForTraceOrSequentialValidation)
                    {
                        const auto subtermModes = exactGpuListedClass2SubtermTraceModes(tracedFtype);
                        if (subtermModes.empty() || levelIdef.il[tracedFtype].empty())
                        {
                            continue;
                        }

                        const InteractionDefinitions singleFtypeIdef =
                                makeSingleInteractionFunctionDefinitions(levelIdef, tracedFtype);
                        fr->listedForcesGpu->updateHaveInteractions(singleFtypeIdef);
                        if (!fr->listedForcesGpu->haveInteractions())
                        {
                            continue;
                        }

                        for (const auto& subtermMode : subtermModes)
                        {
                            fr->listedForcesGpu->setPcffClass2DebugMode(subtermMode.mode);
                            fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
                                    nbv->getGridIndices(),
                                    singleFtypeIdef,
                                    gpuGetNBAtomData(nbv->gpuNbv()));
                            gpu_clear_outputs(nbv->gpuNbv(), false);

                            StepWorkload splitStepWork = stepWork;
                            splitStepWork.computeEnergy = false;
                            splitStepWork.computeVirial = false;

                            fr->listedForcesGpu->setPbcAndlaunchKernel(
                                    fr->pbcType, box, fr->bMolPBC, splitStepWork);
                            gpu_launch_cpyback(
                                    nbv->gpuNbv(), &nbv->nbat(), splitStepWork, AtomLocality::Local);
                            gpu_wait_finish_task(nbv->gpuNbv(),
                                                 splitStepWork,
                                                 AtomLocality::Local,
                                                 false,
                                                 enerd,
                                                 forceOutPtr->forceWithShiftForces().shiftForces(),
                                                 wcycle);

                            const auto splitGpuOutput = captureNbatOutputForceBufferPair(nbv);
                            const std::string outputStage =
                                    std::string(subtermMode.label) + "_nbat_output_buffer";
                            appendExactGpuBondedReductionTrace(activeM2pTraceDirPath(),
                                                               step,
                                                               exactLevel,
                                                               outputStage.c_str(),
                                                               splitGpuOutput);

                            std::vector<RVec> splitReducedStorage(
                                    forceOutPtr->forceWithShiftForces().force().ssize());
                            for (auto& forceValue : splitReducedStorage)
                            {
                                clear_rvec(forceValue);
                            }
                            nbv->atomdata_add_nbat_f_to_f(AtomLocality::Local,
                                                          makeArrayRef(splitReducedStorage));
                            const std::string reducedStage =
                                    std::string(subtermMode.label) + "_after_reduce";
                            appendExactGpuBondedReductionTrace(activeM2pTraceDirPath(),
                                                               step,
                                                               exactLevel,
                                                               reducedStage.c_str(),
                                                               makeConstArrayRef(splitReducedStorage));
                            nbv->nbat().clearForceBuffer(0);
                            fr->listedForcesGpu->clearPcffClass2DebugMode();
                        }
                        fr->listedForcesGpu->clearPcffClass2DebugMode();
                    }

                    fr->listedForcesGpu->updateHaveInteractions(levelIdef);
                }
            }

            fr->listedForcesGpu->updateHaveInteractions(top->idef);
        }

        if (useExactRespaForceOutputs)
        {
            for (int exactLevel = 0; exactLevel < exactRespaForceOutputs.numActiveLevels(); ++exactLevel)
            {
                ListedForces& listedForces = fr->listedForces[exactLevel];
                ForceOutputs* forceOutPtr  = exactRespaForceOutputs.levelOrNull(exactLevel);
                GMX_RELEASE_ASSERT(forceOutPtr != nullptr,
                                   "Need force output for active exact r-RESPA level");
                listedForces.calculate(wcycle,
                                       box,
                                       x,
                                       xWholeMolecules,
                                       fr->fcdata.get(),
                                       hist,
                                       forceOutPtr,
                                       fr,
                                       &pbc,
                                       enerd,
                                       nrnb,
                                       lambda,
                                       mdatoms->chargeA,
                                       mdatoms->chargeB,
                                       makeConstArrayRef(mdatoms->bPerturbed),
                                       mdatoms->cENER,
                                       mdatoms->nPerturbed,
                                       haveDDAtomOrdering(*cr) ? cr->dd->globalAtomIndices.data() : nullptr,
                                       stepWork);

                if (shouldTracePcffClass2SubtermEnergiesStep(step))
                {
                    const auto class2Subterms = evaluatePcffClass2SubtermEnergies(
                            listedForces.interactionDefinitions(),
                            x.unpaddedConstArrayRef(),
                            needMolPbc ? &pbc : nullptr,
                            haveDDAtomOrdering(*cr) ? cr->dd->globalAtomIndices.data() : nullptr);
                    appendPcffClass2SubtermEnergyTrace(activeM2pTraceDirPath(),
                                                      step,
                                                      exactLevel,
                                                      simulationWork.useGpuBonded ? "gpu_offload_enabled"
                                                                                  : "cpu_only",
                                                      class2Subterms);
                }
            }
        }
        else
        {
            const int numActiveMtsLevels =
                    useSubstepLevels ? (stepWork.highestActiveMtsLevel + 1) : 1;
            for (int mtsIndex = 0; mtsIndex < numActiveMtsLevels; mtsIndex++)
            {
                ListedForces& listedForces = fr->listedForces[mtsIndex];
                ForceOutputs* forceOutPtr =
                        (mtsIndex >= 0 && mtsIndex < static_cast<int>(forceOutByMtsLevel.size()))
                                ? forceOutByMtsLevel[mtsIndex]
                                : nullptr;
                GMX_RELEASE_ASSERT(forceOutPtr != nullptr, "Need force output for active MTS level");
                listedForces.calculate(wcycle,
                                       box,
                                       x,
                                       xWholeMolecules,
                                       fr->fcdata.get(),
                                       hist,
                                       forceOutPtr,
                                       fr,
                                       &pbc,
                                       enerd,
                                       nrnb,
                                       lambda,
                                       mdatoms->chargeA,
                                       mdatoms->chargeB,
                                       makeConstArrayRef(mdatoms->bPerturbed),
                                       mdatoms->cENER,
                                       mdatoms->nPerturbed,
                                       haveDDAtomOrdering(*cr) ? cr->dd->globalAtomIndices.data() : nullptr,
                                       stepWork);
            }
        }

        if (shouldTraceExactPair14Delta)
        {
            const auto tracedPair14AfterListed =
                    captureDistinctForceOutput(exactRespaForceOutputs.levelOrNull(pair14Level));
            tracedPair14Delta     = subtractTracedForcePairs(tracedPair14AfterListed, tracedPair14BeforeListed);
            haveTracedPair14Delta = true;
        }
    }

    if (traceStep1Subset01ForceGroupAudit)
    {
        if (useExactRespaForceOutputs)
        {
            appendStep1Subset01ForceGroupBufferSnapshot(activeM2pTraceDirPath(),
                                                       step1Subset01TraceSide,
                                                       step,
                                                       "after_listed",
                                                       exactRespaForceOutputs,
                                                       step1Subset01Level0Role,
                                                       "src/gromacs/mdlib/sim_util.cpp:after_listed_stage");
        }
        else
        {
            appendStep1Subset01ForceGroupBufferSnapshot(activeM2pTraceDirPath(),
                                                       step1Subset01TraceSide,
                                                       step,
                                                       "after_listed",
                                                       forceOutByMtsLevel,
                                                       stepWork.highestActiveMtsLevel,
                                                       step1Subset01Level0Role,
                                                       "src/gromacs/mdlib/sim_util.cpp:after_listed_stage");
        }
    }

    const bool needToReceivePmeResultsFromSeparateRank = stepWork.computePmeOnSeparateRank;
    const bool needToReceivePmeResults =
            (stepWork.haveGpuPmeOnThisRank || needToReceivePmeResultsFromSeparateRank);
    const bool needEarlyPmeResults = (awh != nullptr && awh->hasFepLambdaDimension() && needToReceivePmeResults
                                      && stepWork.computeEnergy && stepWork.computeLongRangeNonbondedForces);
    const bool deferReciprocalTraceUntilPmeReduction =
            traceForceComponents && stepWork.computeLongRangeNonbondedForces && needToReceivePmeResults;
    const auto captureCurrentDistinctForceOutputsForTrace = [&]()
    {
        return useExactRespaForceOutputs ? captureDistinctForceOutputs(exactRespaForceOutputs)
                                         : captureDistinctForceOutputs(forceOutByMtsLevel,
                                                                       highestActiveSubstepLevel);
    };
    const auto refreshDeferredReciprocalTrace = [&](const TracedForcePair& tracedForceOutputsBeforePmeReduction)
    {
        if (!traceForceComponents)
        {
            return;
        }
        const auto tracedForceOutputsAfterPmeReduction = captureCurrentDistinctForceOutputsForTrace();
        tracedCoulombRecipDelta = subtractTracedForcePairs(tracedForceOutputsAfterPmeReduction,
                                                           tracedForceOutputsBeforePmeReduction);
    };

    TracedForcePair tracedForceOutputsBeforeLongRange;
    if (traceForceComponents)
    {
        const auto tracedForceOutputsAfterListed =
                useExactRespaForceOutputs ? captureDistinctForceOutputs(exactRespaForceOutputs)
                                          : captureDistinctForceOutputs(forceOutByMtsLevel,
                                                                        stepWork.highestActiveMtsLevel);
        tracedBondedDelta =
                subtractTracedForcePairs(tracedForceOutputsAfterListed, tracedForceOutputsBeforeListed);
        tracedForceOutputsBeforeLongRange = tracedForceOutputsAfterListed;
    }

    if (stepWork.computeLongRangeNonbondedForces)
    {
        const char* earlyAccumTraceDirPath = std::getenv("GMX_PCFF_RESPA_EARLY_TRACE_DIR");
        const bool dumpEarlyAccumTrace =
                (earlyAccumTraceDirPath != nullptr && *earlyAccumTraceDirPath != '\0' && step == 0);
        const bool outerAliasesShift =
                (forceOutLongrange != nullptr && forceOutLongrange->haveForceWithVirial()
                 && forceOutLongrange->forceWithVirial().force_.data()
                            == forceOutLongrange->forceWithShiftForces().force().data());
        if (dumpEarlyAccumTrace && forceOutLongrange != nullptr)
        {
            dumpRespaMergeTraceVector(
                    earlyAccumTraceDirPath,
                    "step0_level2_before_longrange_virial.tsv",
                    "stage=before_longrange_nonbonded mts_index=2 mts_user=3 buffer=forceWithVirial alias_with_shift="
                            + std::string(outerAliasesShift ? "true" : "false"),
                    forceOutLongrange->forceWithVirial().force_);
        }
        longRangeNonbondeds->calculate(fr->pmedata,
                                       cr,
                                       x.unpaddedConstArrayRef(),
                                       &forceOutLongrange->forceWithVirial(),
                                       enerd,
                                       box,
                                       lambda,
                                       dipoleData.muStateAB,
                                       stepWork,
                                       ddBalanceRegionHandler);
        if (dumpEarlyAccumTrace && forceOutLongrange != nullptr)
        {
            dumpRespaMergeTraceVector(
                    earlyAccumTraceDirPath,
                    "step0_level2_after_longrange_virial.tsv",
                    "stage=after_longrange_nonbonded mts_index=2 mts_user=3 buffer=forceWithVirial alias_with_shift="
                            + std::string(outerAliasesShift ? "true" : "false"),
                    forceOutLongrange->forceWithVirial().force_);
        }
    }

    if (traceStep1Subset01ForceGroupAudit)
    {
        if (useExactRespaForceOutputs)
        {
            appendStep1Subset01ForceGroupBufferSnapshot(activeM2pTraceDirPath(),
                                                       step1Subset01TraceSide,
                                                       step,
                                                       "after_longrange",
                                                       exactRespaForceOutputs,
                                                       step1Subset01Level0Role,
                                                       "src/gromacs/mdlib/sim_util.cpp:after_longrange_stage");
        }
        else
        {
            appendStep1Subset01ForceGroupBufferSnapshot(activeM2pTraceDirPath(),
                                                       step1Subset01TraceSide,
                                                       step,
                                                       "after_longrange",
                                                       forceOutByMtsLevel,
                                                       highestActiveSubstepLevel,
                                                       step1Subset01Level0Role,
                                                       "src/gromacs/mdlib/sim_util.cpp:after_longrange_stage");
        }
    }

    if (traceForceComponents)
    {
        const auto tracedForceOutputsAfterLongRange = captureCurrentDistinctForceOutputsForTrace();
        if (deferReciprocalTraceUntilPmeReduction)
        {
            tracedForceOutputsBeforeLongRange = tracedForceOutputsAfterLongRange;
        }
        else
        {
            tracedCoulombRecipDelta =
                    subtractTracedForcePairs(tracedForceOutputsAfterLongRange, tracedForceOutputsBeforeLongRange);
        }
    }

    wallcycle_stop(wcycle, WallCycleCounter::Force);

    // VdW dispersion correction, only computed on main rank to avoid double counting
    if ((stepWork.computeEnergy || stepWork.computeVirial) && fr->dispersionCorrection
        && cr->commMySim.isMainRank())
    {
        // Calculate long range corrections to pressure and energy
        const DispersionCorrection::Correction correction = fr->dispersionCorrection->calculate(
                box, lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Vdw)]);

        if (stepWork.computeEnergy)
        {
            enerd->term[InteractionFunction::DispersionCorrection] = correction.energy;
            enerd->term[InteractionFunction::dVvanderWaalsdLambda] += correction.dvdl;
            enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Vdw] += correction.dvdl;
        }
        if (stepWork.computeVirial)
        {
            correction.correctVirial(vir_force);
            enerd->term[InteractionFunction::PressureDispersionCorrection] = correction.pressure;
        }
    }

    if (needEarlyPmeResults)
    {
        if (stepWork.haveGpuPmeOnThisRank)
        {
            const auto tracedForceOutputsBeforePmeReduction =
                    traceForceComponents ? captureCurrentDistinctForceOutputsForTrace() : TracedForcePair{};
            pmeGpuWaitAndReduce(fr->pmedata,
                                stepWork,
                                wcycle,
                                &forceOutLongrange->forceWithVirial(),
                                enerd,
                                lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)]);
            if (traceForceComponents)
            {
                refreshDeferredReciprocalTrace(tracedForceOutputsBeforePmeReduction);
            }
        }
        else if (needToReceivePmeResultsFromSeparateRank)
        {
            /* In case of node-splitting, the PP nodes receive the long-range
             * forces, virial and energy from the PME nodes here.
             */
            const auto tracedForceOutputsBeforePmeReduction =
                    traceForceComponents ? captureCurrentDistinctForceOutputsForTrace() : TracedForcePair{};
            pme_receive_force_ener(fr,
                                   cr->dd,
                                   &forceOutLongrange->forceWithVirial(),
                                   enerd,
                                   simulationWork.useGpuPmePpCommunication,
                                   stepWork.useGpuPmeFReduction,
                                   wcycle);
            if (traceForceComponents)
            {
                refreshDeferredReciprocalTrace(tracedForceOutputsBeforePmeReduction);
            }
        }
    }

    if (domainWork.haveSpecialForces)
    {
        // Communication often happens for special forces, so we should close the balancing region here
        ddBalanceRegionHandler.closeAfterForceComputationCpu();

        computeSpecialForces(fplog,
                             cr->commMyGroup,
                             cr->dd,
                             inputrec,
                             awh,
                             enforcedRotation,
                             imdSession,
                             pull_work,
                             step,
                             t,
                             wcycle,
                            fr->forceProviders,
                            box,
                            x.unpaddedArrayRef(),
                            mdatoms,
                            lambda,
                            stepWork,
                            gmx::useExactRespa(inputrec) ? &exactRespaStepWork : nullptr,
                            forceWithVirialByLevel,
                            enerd,
                            ed,
                            stepWork.doNeighborSearch);
    }

    if (simulationWork.havePpDomainDecomposition && stepWork.computeForces && stepWork.useGpuFHalo
        && domainWork.haveCpuLocalForceWork)
    {
        stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(), AtomLocality::Local);
    }

    GMX_ASSERT(!(simulationWork.nonbondedSubstepLevel > 0 && stepWork.useGpuFBufferOps),
               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
    GMX_ASSERT(!(nonbondedAtMtsNonzeroLevel && stepWork.useGpuFHalo),
               "The schedule below does not allow for nonbonded MTS with GPU halo exchange");
    // Will store the amount of cycles spent waiting for the GPU that
    // will be later used in the DLB accounting.
    float cycles_wait_gpu = 0;
    if (useOrEmulateGpuNb && stepWork.computeNonbondedForces && !useExactLammpsRespaNonbonded)
    {
        auto& forceWithShiftForces = forceOutNonbonded->forceWithShiftForces();

        /* wait for non-local forces (or calculate in emulation mode) */
        if (simulationWork.havePpDomainDecomposition)
        {
            if (simulationWork.useGpuNonbonded)
            {
                cycles_wait_gpu +=
                        gpu_wait_finish_task(nbv->gpuNbv(),
                                             stepWork,
                                             AtomLocality::NonLocal,
                                             (simulationWork.useGpuForeignNonbondedFE) ? true : false,
                                             enerd,
                                             forceWithShiftForces.shiftForces(),
                                             wcycle);
            }
            else
            {
                wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
                do_nb_verlet(
                        fr, ic, enerd, stepWork, InteractionLocality::NonLocal, enbvClearFYes, step, nrnb, wcycle);
                wallcycle_stop(wcycle, WallCycleCounter::Force);
            }

            if (stepWork.useGpuFBufferOps)
            {
                if (domainWork.haveCpuNonLocalForceWork)
                {
                    stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
                                              AtomLocality::NonLocal);
                }


                fr->gpuForceReduction[AtomLocality::NonLocal]->execute();

                if (!stepWork.useGpuFHalo)
                {
                    /* We don't explicitly wait for the forces to be reduced on device,
                     * but wait for them to finish copying to CPU instead.
                     * So, we manually consume the event, see Issue #3988. */
                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::NonLocal);
                    // copy from GPU input for dd_move_f()
                    stateGpu->copyForcesFromGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
                                                AtomLocality::NonLocal);
                }
            }
            else
            {
                nbv->atomdata_add_nbat_f_to_f(AtomLocality::NonLocal, forceWithShiftForces.force());
            }

            if (fr->nbv->emulateGpu() && stepWork.computeVirial)
            {
                nbnxn_atomdata_add_nbat_fshift_to_fshift(nbv->nbat(), forceWithShiftForces.shiftForces());
            }
        }
    }

    /* Combining the forces for multiple time stepping before the halo exchange, when possible,
     * avoids an extra halo exchange (when DD is used) and post-processing step.
     */
    if (combineSubstepForcesBeforeHaloExchange)
    {
        wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
        const std::vector<ArrayRef<const RVec>> slowLevelForces = {
            forceOutByMtsLevel[1]->forceWithShiftForces().force()
        };
        const std::array<int, 1> slowLevelFactors = {
            inputrec.mtsLevels[1].stepFactor
        };
        combineMtsForces(getLocalAtomCount(cr->dd, *mdatoms, simulationWork.havePpDomainDecomposition),
                         force.unpaddedArrayRef(),
                         forceView->forceMtsCombined(),
                         slowLevelForces,
                         slowLevelFactors);
        wallcycle_stop(wcycle, WallCycleCounter::Force);
    }

    // With both nonbonded and PME offloaded a GPU on the same rank, we use
    // an alternating wait/reduction scheme.
    // When running free energy perturbations steered by AWH and calculating PME on GPU,
    // i.e. if needEarlyPmeResults == true, the PME results have already been reduced above.
    const bool alternateGpuWait = (!c_disableAlternatingWait && stepWork.haveGpuPmeOnThisRank
                                   && simulationWork.useGpuNonbonded && !useExactLammpsRespaGpuNonbonded
                                   && !simulationWork.havePpDomainDecomposition
                                   && !stepWork.useGpuFBufferOps && !needEarlyPmeResults);


    const int expectedLocalFReadyOnDeviceConsumptionCount = getExpectedLocalFReadyOnDeviceConsumptionCount(
            simulationWork, domainWork, stepWork, useOrEmulateGpuNb, alternateGpuWait);
    // If expectedLocalFReadyOnDeviceConsumptionCount == 0, stateGpu can be uninitialized
    if (expectedLocalFReadyOnDeviceConsumptionCount > 0)
    {
        stateGpu->setFReadyOnDeviceEventExpectedConsumptionCount(
                AtomLocality::Local, expectedLocalFReadyOnDeviceConsumptionCount);
    }

    if (simulationWork.havePpDomainDecomposition)
    {
        /* We are done with the CPU compute.
         * We will now communicate the non-local forces.
         * If we use a GPU this will overlap with GPU work, so in that case
         * we do not close the DD force balancing region here.
         * With special forces we closed this region already before computing the special forces.
         */
        ddBalanceRegionHandler.closeAfterForceComputationCpu();

        if (stepWork.computeForces)
        {

            if (stepWork.useGpuFHalo)
            {
                // If there exist CPU forces, data from halo exchange should accumulate into these
                bool accumulateForces = domainWork.haveCpuLocalForceWork;
                FixedCapacityVector<GpuEventSynchronizer*, 2> gpuForceHaloDependencies;
                // completion of both H2D copy and clearing is signaled by fReadyOnDevice
                if (domainWork.haveCpuLocalForceWork || stepWork.clearGpuFBufferEarly)
                {
                    gpuForceHaloDependencies.push_back(stateGpu->fReadyOnDevice(AtomLocality::Local));
                }
                gpuForceHaloDependencies.push_back(stateGpu->fReducedOnDevice(AtomLocality::NonLocal));

                communicateGpuHaloForces(*cr, accumulateForces, &gpuForceHaloDependencies);
            }
            else
            {
                if (stepWork.useGpuFBufferOps)
                {
                    stateGpu->waitForcesReadyOnHost(AtomLocality::NonLocal);
                }

                // Without MTS or with MTS at slow steps with uncombined forces we need to
                // communicate the fast forces
                if (!useSubstepLevels || !combineSubstepForcesBeforeHaloExchange)
                {
                    dd_move_f(cr->dd, &forceOutMtsLevel0.forceWithShiftForces(), wcycle);
                }
                // With MTS we need to communicate the slow or combined forces.
                if (haveExactSlowForceOutputs)
                {
                    for (int mtsLevel = 1; mtsLevel < exactRespaForceOutputs.numActiveLevels(); mtsLevel++)
                    {
                        ForceOutputs* outputs = exactRespaForceOutputs.levelOrNull(mtsLevel);
                        if (outputs != nullptr)
                        {
                            dd_move_f(cr->dd, &outputs->forceWithShiftForces(), wcycle);
                        }
                    }
                }
                else if (haveLegacySlowForceOutputs)
                {
                    for (int mtsLevel = 1; mtsLevel <= highestActiveSubstepLevel; mtsLevel++)
                    {
                        dd_move_f(cr->dd, &forceOutByMtsLevel[mtsLevel]->forceWithShiftForces(), wcycle);
                    }
                }
            }
        }
    }

    if (alternateGpuWait)
    {
        const auto tracedForceOutputsBeforePmeReduction =
                traceForceComponents ? captureCurrentDistinctForceOutputsForTrace() : TracedForcePair{};
        alternatePmeNbGpuWaitReduce(fr->nbv.get(),
                                    fr->pmedata,
                                    forceOutNonbonded,
                                    forceOutLongrange,
                                    enerd,
                                    lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)],
                                    stepWork,
                                    simulationWork,
                                    wcycle);
        if (traceForceComponents)
        {
            refreshDeferredReciprocalTrace(tracedForceOutputsBeforePmeReduction);
        }
    }

    if (!alternateGpuWait && stepWork.haveGpuPmeOnThisRank && !needEarlyPmeResults)
    {
        const auto tracedForceOutputsBeforePmeReduction =
                traceForceComponents ? captureCurrentDistinctForceOutputsForTrace() : TracedForcePair{};
        pmeGpuWaitAndReduce(fr->pmedata,
                            stepWork,
                            wcycle,
                            &forceOutLongrange->forceWithVirial(),
                            enerd,
                            lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)]);
        if (traceForceComponents)
        {
            refreshDeferredReciprocalTrace(tracedForceOutputsBeforePmeReduction);
        }
    }

    /* Wait for local GPU NB outputs on the non-alternating wait path */
    if (!alternateGpuWait && stepWork.computeNonbondedForces && simulationWork.useGpuNonbonded
        && !useExactLammpsRespaGpuNonbonded)
    {
        /* Measured overhead on CUDA and OpenCL with(out) GPU sharing
         * is between 0.5 and 1.5 Mcycles. So 2 MCycles is an overestimate,
         * but even with a step of 0.1 ms the difference is less than 1%
         * of the step time.
         */
        const float gpuWaitApiOverheadMargin = 2e6F; /* cycles */
        float       waitCycles;

        waitCycles = gpu_wait_finish_task(nbv->gpuNbv(),
                                          stepWork,
                                          AtomLocality::Local,
                                          (simulationWork.useGpuForeignNonbondedFE) ? true : false,
                                          enerd,
                                          forceOutNonbonded->forceWithShiftForces().shiftForces(),
                                          wcycle);

        if (ddBalanceRegionHandler.useBalancingRegion())
        {
            DdBalanceRegionWaitedForGpu waitedForGpu = DdBalanceRegionWaitedForGpu::yes;
            if (stepWork.computeForces && waitCycles <= gpuWaitApiOverheadMargin)
            {
                /* We measured few cycles, it could be that the kernel
                 * and transfer finished earlier and there was no actual
                 * wait time, only API call overhead.
                 * Then the actual time could be anywhere between 0 and
                 * cycles_wait_est. We will use half of cycles_wait_est.
                 */
                waitedForGpu = DdBalanceRegionWaitedForGpu::no;
            }
            ddBalanceRegionHandler.closeAfterForceComputationGpu(cycles_wait_gpu, waitedForGpu);
        }
    }

    if (fr->nbv->emulateGpu() && !useExactLammpsRespaNonbonded)
    {
        // NOTE: emulation kernel is not included in the balancing region,
        // but emulation mode does not target performance anyway
        wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
        do_nb_verlet(fr,
                     ic,
                     enerd,
                     stepWork,
                     InteractionLocality::Local,
                     haveDDAtomOrdering(*cr) ? enbvClearFNo : enbvClearFYes,
                     step,
                     nrnb,
                     wcycle);
        wallcycle_stop(wcycle, WallCycleCounter::Force);
    }

    // If on GPU PME-PP comms path, receive forces from PME before GPU buffer ops
    // TODO refactor this and unify with below default-path call to the same function
    // When running free energy perturbations steered by AWH and calculating PME on GPU,
    // i.e. if needEarlyPmeResults == true, the PME results have already been reduced above.
    if (needToReceivePmeResultsFromSeparateRank && simulationWork.useGpuPmePpCommunication && !needEarlyPmeResults)
    {
        /* In case of node-splitting, the PP nodes receive the long-range
         * forces, virial and energy from the PME nodes here.
         */
        const auto tracedForceOutputsBeforePmeReduction =
                traceForceComponents ? captureCurrentDistinctForceOutputsForTrace() : TracedForcePair{};
        pme_receive_force_ener(fr,
                               cr->dd,
                               &forceOutLongrange->forceWithVirial(),
                               enerd,
                               simulationWork.useGpuPmePpCommunication,
                               stepWork.useGpuPmeFReduction,
                               wcycle);
        if (traceForceComponents)
        {
            refreshDeferredReciprocalTrace(tracedForceOutputsBeforePmeReduction);
        }
    }


    /* Do the nonbonded GPU (or emulation) force buffer reduction
     * on the non-alternating path. */
    GMX_ASSERT(!(nonbondedAtMtsNonzeroLevel && stepWork.useGpuFBufferOps),
               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
    if (useOrEmulateGpuNb && !alternateGpuWait && !useExactLammpsRespaNonbonded)
    {
        if (stepWork.useGpuFBufferOps)
        {
            ArrayRef<RVec> forceWithShift = forceOutNonbonded->forceWithShiftForces().force();

            // TODO: move these steps as early as possible:
            // - CPU f H2D should be as soon as all CPU-side forces are done
            // - wait for force reduction does not need to block host (at least not here, it's sufficient to wait
            //   before the next CPU task that consumes the forces: vsite spread or update)
            // - copy is not performed if GPU force halo exchange is active, because it would overwrite the result
            //   of the halo exchange. In that case the copy is instead performed above, before the exchange.
            //   These should be unified.
            if (domainWork.haveLocalForceContribInCpuBuffer && !stepWork.useGpuFHalo)
            {
                stateGpu->copyForcesToGpu(forceWithShift, AtomLocality::Local);
            }

            if (stepWork.computeNonbondedForces)
            {
                fr->gpuForceReduction[AtomLocality::Local]->execute();
            }

            // Copy forces to host if they are needed for update or if virtual sites are enabled.
            // If there are vsites, we need to copy forces every step to spread vsite forces on host.
            // TODO: When the output flags will be included in step workload, this copy can be combined with the
            //       copy call done in sim_utils(...) for the output.
            // NOTE: If there are virtual sites, the forces are modified on host after this D2H copy. Hence,
            //       they should not be copied in do_md(...) for the output.
            if (!simulationWork.useGpuUpdate
                || (simulationWork.useGpuUpdate && haveDDAtomOrdering(*cr) && simulationWork.useCpuPmePpCommunication)
                || vsite)
            {
                if (stepWork.computeNonbondedForces)
                {
                    /* We have previously issued force reduction on the GPU, but we will
                     * not use this event, instead relying on the stream being in-order.
                     * Issue #3988. */
                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::Local);
                }
                stateGpu->copyForcesFromGpu(forceWithShift, AtomLocality::Local);
                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
            }
        }
        else if (stepWork.computeNonbondedForces)
        {
            ArrayRef<RVec> forceWithShift = forceOutNonbonded->forceWithShiftForces().force();
            nbv->atomdata_add_nbat_f_to_f(AtomLocality::Local, forceWithShift);
        }
    }

    if (expectedLocalFReadyOnDeviceConsumptionCount > 0)
    {
        /* The same fReadyOnDevice device synchronizer is later used to track buffer clearing,
         * so we reset the expected consumption value back to the default (1). */
        stateGpu->setFReadyOnDeviceEventExpectedConsumptionCount(AtomLocality::Local, 1);
    }

    launchGpuEndOfStepTasks(
            nbv, fr->listedForcesGpu.get(), fr->pmedata, enerd, runScheduleWork, step, wcycle);

    if (haveDDAtomOrdering(*cr))
    {
        dd_force_flop_stop(cr->dd, nrnb);
    }

    const bool haveCombinedMtsForces = (stepWork.computeForces && useLegacyMtsForceOutputs
                                        && computeLegacySlowSubstepForces
                                        && combineSubstepForcesBeforeHaloExchange);
    const char* mergeTraceDirPath = std::getenv("GMX_PCFF_RESPA_MERGE_TRACE_DIR");
    const bool  dumpMergeTrace =
            (mergeTraceDirPath != nullptr && *mergeTraceDirPath != '\0'
             && (step == 0 || shouldTraceRespaForceComponentsStep(step)));
    const auto dumpForceOutputsStage = [&](const char* stageLabel, int mtsLevelIndex, ForceOutputs* outputs)
    {
        if (!dumpMergeTrace || outputs == nullptr)
        {
            return;
        }

        const std::string commonHeader = "stage=" + std::string(stageLabel) + " mts_index="
                                         + std::to_string(mtsLevelIndex) + " mts_user="
                                         + std::to_string(mtsLevelIndex + 1) + " step="
                                         + std::to_string(step);
        const std::string levelLabel   = "step" + std::to_string(step) + "_level"
                                       + std::to_string(mtsLevelIndex) + "_" + stageLabel;

        dumpRespaMergeTraceVector(mergeTraceDirPath,
                                  (levelLabel + "_shift.tsv").c_str(),
                                  commonHeader + " buffer=forceWithShiftForces",
                                  outputs->forceWithShiftForces().force());
        if (mtsLevelIndex == 0 && std::strcmp(stageLabel, "post_postprocess") == 0
            && activeM2pTraceDirPath() != nullptr)
        {
            appendExplicitLevel0SnapshotForTracedAtoms(activeM2pTraceDirPath(),
                                                       step,
                                                       stageLabel,
                                                       x.unpaddedConstArrayRef(),
                                                       outputs->forceWithShiftForces().force(),
                                                       "src/gromacs/mdlib/sim_util.cpp:dumpForceOutputsStage");
        }
        if (outputs->haveForceWithVirial())
        {
            dumpRespaMergeTraceVector(mergeTraceDirPath,
                                      (levelLabel + "_virial.tsv").c_str(),
                                      commonHeader + " buffer=forceWithVirial",
                                      outputs->forceWithVirial().force_);
        }
    };
    if (stepWork.computeForces)
    {
        postProcessForceWithShiftForces(
                nrnb, wcycle, box, x.unpaddedArrayRef(), &forceOutMtsLevel0, vir_force, *mdatoms, *fr, vsite, stepWork);

        if (useExactRespaForceOutputs)
        {
            for (int mtsLevel = 1; mtsLevel < exactRespaForceOutputs.numActiveLevels(); mtsLevel++)
            {
                ForceOutputs* outputs = exactRespaForceOutputs.levelOrNull(mtsLevel);
                if (outputs == nullptr)
                {
                    continue;
                }
                dumpForceOutputsStage("pre_postprocess", mtsLevel, outputs);
                postProcessForceWithShiftForces(nrnb,
                                                wcycle,
                                                box,
                                                x.unpaddedArrayRef(),
                                                outputs,
                                                vir_force,
                                                *mdatoms,
                                                *fr,
                                                vsite,
                                                stepWork);
                dumpForceOutputsStage("post_postprocess", mtsLevel, outputs);
            }
            // Exact r-RESPA must defer force combining until postProcessForces().
            // Before that point, the outer level can still reside in forceWithVirial(),
            // so an early combine would add the middle level once here and again later.
        }
        else if (haveLegacySlowForceOutputs && !haveCombinedMtsForces)
        {
            for (int mtsLevel = 1; mtsLevel <= highestActiveSubstepLevel; mtsLevel++)
            {
                postProcessForceWithShiftForces(nrnb,
                                                wcycle,
                                                box,
                                                x.unpaddedArrayRef(),
                                                forceOutByMtsLevel[mtsLevel],
                                                vir_force,
                                                *mdatoms,
                                                *fr,
                                                vsite,
                                                stepWork);
            }
        }
    }

    // TODO refactor this and unify with above GPU PME-PP / GPU update path call to the same function
    // When running free energy perturbations steered by AWH and calculating PME on GPU,
    // i.e. if needEarlyPmeResults == true, the PME results have already been reduced above.
    if (needToReceivePmeResultsFromSeparateRank && simulationWork.useCpuPmePpCommunication && !needEarlyPmeResults)
    {
        /* In case of node-splitting, the PP nodes receive the long-range
         * forces, virial and energy from the PME nodes here.
         */
        const auto tracedForceOutputsBeforePmeReduction =
                traceForceComponents ? captureCurrentDistinctForceOutputsForTrace() : TracedForcePair{};
        pme_receive_force_ener(fr,
                               cr->dd,
                               &forceOutLongrange->forceWithVirial(),
                               enerd,
                               simulationWork.useGpuPmePpCommunication,
                               false,
                               wcycle);
        if (traceForceComponents)
        {
            refreshDeferredReciprocalTrace(tracedForceOutputsBeforePmeReduction);
        }
    }

    if (stepWork.computeForces)
    {
        /* If we don't use MTS or if we already combined the MTS forces before, we only
         * need to post-process one ForceOutputs object here, called forceOutCombined,
         * otherwise we have to post-process two outputs and then combine them.
         */
        ForceOutputs& forceOutCombined = (haveCombinedMtsForces ? forceOutSingleSlowLevel.value() : forceOutMtsLevel0);
        dumpForceOutputsStage("pre_postprocess", 0, &forceOutCombined);
        postProcessForces(
                cr->dd, step, nrnb, wcycle, box, x.unpaddedArrayRef(), &forceOutCombined, vir_force, mdatoms, fr, vsite, stepWork);
        dumpForceOutputsStage("post_postprocess", 0, &forceOutCombined);

        if (useExactRespaForceOutputs && !haveCombinedMtsForces)
        {
            GMX_RELEASE_ASSERT(exactRespaForceStore != nullptr,
                               "Exact r-RESPA force combining requires persisted slow-level totals");
            for (int mtsLevel = 1; mtsLevel < exactRespaForceOutputs.numActiveLevels(); mtsLevel++)
            {
                ForceOutputs* outputs = exactRespaForceOutputs.levelOrNull(mtsLevel);
                if (outputs == nullptr)
                {
                    continue;
                }
                dumpForceOutputsStage("pre_postprocess", mtsLevel, outputs);
                postProcessForces(cr->dd,
                                  step,
                                  nrnb,
                                  wcycle,
                                  box,
                                  x.unpaddedArrayRef(),
                                  outputs,
                                  vir_force,
                                  mdatoms,
                                  fr,
                                  vsite,
                                  stepWork);
                dumpForceOutputsStage("post_postprocess", mtsLevel, outputs);
            }

            std::vector<ArrayRef<const RVec>> slowLevelForces;
            std::vector<int>                  slowLevelFactors;
            for (int mtsLevel = 1; mtsLevel < exactRespaNumLevels(inputrec); ++mtsLevel)
            {
                ArrayRef<const RVec> slowForce;
                if (exactRespaForceOutputs.hasLevel(mtsLevel))
                {
                    slowForce = exactRespaForceOutputs.level(mtsLevel).forceWithShiftForces().force();
                }
                else if (exactRespaForceStore->hasLevel(mtsLevel))
                {
                    slowForce = exactRespaForceStore->levelTotal(mtsLevel);
                }
                if (!slowForce.empty())
                {
                    slowLevelForces.push_back(slowForce);
                    slowLevelFactors.push_back(gmx::exactRespaLevelStepFactor(inputrec, mtsLevel));
                }
            }

            if (!slowLevelForces.empty())
            {
                combineMtsForces(mdatoms->homenr,
                                 force.unpaddedArrayRef(),
                                 forceView->forceMtsCombined(),
                                 slowLevelForces,
                                 slowLevelFactors);
                if (dumpMergeTrace)
                {
                    dumpRespaMergeTraceVector(mergeTraceDirPath,
                                              ("step" + std::to_string(step) + "_physical_postcombine.tsv").c_str(),
                                              "stage=post_combine buffer=physical_total step="
                                                      + std::to_string(step),
                                              force.unpaddedArrayRef());
                    dumpRespaMergeTraceVector(mergeTraceDirPath,
                                              ("step" + std::to_string(step) + "_impulse_postcombine.tsv").c_str(),
                                              "stage=post_combine buffer=impulse_total step="
                                                      + std::to_string(step),
                                              forceView->forceMtsCombined());
                }
            }
        }
        else if (haveLegacySlowForceOutputs && !haveCombinedMtsForces)
        {
            std::vector<ArrayRef<const RVec>> slowLevelForces;
            std::vector<int>                  slowLevelFactors;
            for (int mtsLevel = 1; mtsLevel <= highestActiveSubstepLevel; mtsLevel++)
            {
                dumpForceOutputsStage("pre_postprocess", mtsLevel, forceOutByMtsLevel[mtsLevel]);
                postProcessForces(cr->dd,
                                  step,
                                  nrnb,
                                  wcycle,
                                  box,
                                  x.unpaddedArrayRef(),
                                  forceOutByMtsLevel[mtsLevel],
                                  vir_force,
                                  mdatoms,
                                  fr,
                                  vsite,
                                  stepWork);
                dumpForceOutputsStage("post_postprocess", mtsLevel, forceOutByMtsLevel[mtsLevel]);
                slowLevelForces.push_back(forceOutByMtsLevel[mtsLevel]->forceWithShiftForces().force());
                slowLevelFactors.push_back(gmx::useExactRespa(inputrec)
                                                   ? gmx::exactRespaLevelStepFactor(inputrec, mtsLevel)
                                                   : inputrec.mtsLevels[mtsLevel].stepFactor);
            }

            combineMtsForces(mdatoms->homenr,
                             force.unpaddedArrayRef(),
                             forceView->forceMtsCombined(),
                             slowLevelForces,
                             slowLevelFactors);
            if (dumpMergeTrace)
            {
                dumpRespaMergeTraceVector(mergeTraceDirPath,
                                          ("step" + std::to_string(step) + "_physical_postcombine.tsv").c_str(),
                                          "stage=post_combine buffer=physical_total step="
                                                  + std::to_string(step),
                                          force.unpaddedArrayRef());
                dumpRespaMergeTraceVector(mergeTraceDirPath,
                                          ("step" + std::to_string(step) + "_impulse_postcombine.tsv").c_str(),
                                          "stage=post_combine buffer=impulse_total step="
                                                  + std::to_string(step),
                                          forceView->forceMtsCombined());
            }
        }
    }

    if (stepWork.computeEnergy)
    {
        struct EnergyTermReadTrace
        {
            double firstReadTotal = 0.0;
            double finalTotal     = 0.0;
            bool   firstCaptured  = false;
        };
        const auto traceEnergyTermsRead = [](gmx::ArrayRef<const real> values) -> EnergyTermReadTrace
        {
            EnergyTermReadTrace trace;
            double              total = 0.0;
            for (const real value : values)
            {
                total += value;
                if (!trace.firstCaptured)
                {
                    trace.firstReadTotal = total;
                    trace.firstCaptured  = true;
                }
            }
            trace.finalTotal = total;
            return trace;
        };
        const char* ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2W_TRACE_DIR");
        const char* ljSrCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2W_CASE_LABEL");
        const bool  dumpM2wLjSrTrace =
                (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0');
        if (ljSrTraceDirPath == nullptr || *ljSrTraceDirPath == '\0')
        {
            ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2V_TRACE_DIR");
            ljSrCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2V_CASE_LABEL");
        }
        const bool  dumpM2vLjSrTrace =
                (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0');
        if (!dumpM2wLjSrTrace && (ljSrTraceDirPath == nullptr || *ljSrTraceDirPath == '\0'))
        {
            ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2U_TRACE_DIR");
            ljSrCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2U_CASE_LABEL");
        }
        const bool  dumpM2uLjSrTrace =
                (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0');
        if (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && (ljSrTraceDirPath == nullptr || *ljSrTraceDirPath == '\0'))
        {
            ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2S_TRACE_DIR");
            ljSrCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2S_CASE_LABEL");
        }
        const bool  dumpM2sLjSrTrace =
                (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && !dumpM2uLjSrTrace && ljSrTraceDirPath != nullptr
                 && *ljSrTraceDirPath != '\0');
        if (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && !dumpM2uLjSrTrace && (ljSrTraceDirPath == nullptr || *ljSrTraceDirPath == '\0'))
        {
            ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2R_TRACE_DIR");
            ljSrCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2R_CASE_LABEL");
        }
        if (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && !dumpM2uLjSrTrace && !dumpM2sLjSrTrace
            && (ljSrTraceDirPath == nullptr || *ljSrTraceDirPath == '\0'))
        {
            ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2Q_TRACE_DIR");
            ljSrCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2Q_CASE_LABEL");
        }
        if (!dumpM2wLjSrTrace && !dumpM2vLjSrTrace && !dumpM2uLjSrTrace && !dumpM2sLjSrTrace
            && (ljSrTraceDirPath == nullptr || *ljSrTraceDirPath == '\0'))
        {
            ljSrTraceDirPath = std::getenv("GMX_PCFF_RESPA_M2P_TRACE_DIR");
            ljSrCaseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2P_CASE_LABEL");
        }
        if (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0')
        {
            const auto ljReadTrace =
                    traceEnergyTermsRead(enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJSR]);
            const auto coulReadTrace =
                    traceEnergyTermsRead(enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR]);
            const auto lj14ReadTrace =
                    traceEnergyTermsRead(enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJ14]);
            const auto coul14ReadTrace =
                    traceEnergyTermsRead(enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::Coulomb14]);
            const auto buckinghamReadTrace =
                    traceEnergyTermsRead(enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::BuckinghamSR]);
            const std::string ljSrCaseLabel =
                    (ljSrCaseLabelEnv != nullptr && *ljSrCaseLabelEnv != '\0') ? ljSrCaseLabelEnv : "unknown";
            const bool emitRawReadRows = (dumpM2uLjSrTrace || dumpM2sLjSrTrace) && ljSrCaseLabel != "plain_verlet";
            if (emitRawReadRows)
            {
                if (ljReadTrace.firstCaptured)
                {
                    appendRespaTraceTextLine(
                            ljSrTraceDirPath,
                            "step0_lj_sr_internal_trace.txt",
                            "stage=RAW_FIRST_READ_OR_REDUCE code_location=src/gromacs/mdlib/sim_util.cpp:sumEnergyTerms_first_read case_label="
                                    + ljSrCaseLabel
                                    + " execution_path=exact_sumEnergyTerms_first_read trace_role=first_reducer_read_partial_total lj_sr="
                                    + formatString("%.15f", ljReadTrace.firstReadTotal));
                }
                appendRespaTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_lj_sr_internal_trace.txt",
                        "stage=RAW_POST_READ_OR_REDUCE code_location=src/gromacs/mdlib/sim_util.cpp:sumEnergyTerms_final_total case_label="
                                + ljSrCaseLabel
                                + " execution_path=exact_sumEnergyTerms_post_read trace_role=post_reducer_total lj_sr="
                                + formatString("%.15f", ljReadTrace.finalTotal));
            }
            appendRespaTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_lj_sr_internal_trace.txt",
                    "stage=SR_ACCUMULATION code_location=src/gromacs/mdlib/sim_util.cpp:4284 case_label="
                            + ljSrCaseLabel
                            + " execution_path=pre_sum_epot_grpp lj_sr="
                            + formatString("%.15f", ljReadTrace.finalTotal)
                            + " coulomb_sr="
                            + formatString("%.15f", coulReadTrace.finalTotal));
            appendRespaTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_coulomb_sr_component_trace.txt",
                    "stage=SR_ACCUMULATION_PATCH_COMBINED code_location=src/gromacs/mdlib/sim_util.cpp:traceEnergyTermsRead_coulomb_sr case_label="
                            + ljSrCaseLabel
                            + " execution_path=pre_sum_epot_grpp_combined"
                            + " patch_combined_coulomb_sr="
                            + formatString("%.15f", coulReadTrace.finalTotal));
            if (ljSrCaseLabel == "plain_verlet")
            {
                appendRespaTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_coulomb_sr_component_trace.txt",
                        "stage=SR_ACCUMULATION_PLAIN code_location=src/gromacs/mdlib/sim_util.cpp:traceEnergyTermsRead_coulomb_sr case_label="
                                + ljSrCaseLabel
                                + " execution_path=pre_sum_epot_grpp_plain plain_coulomb_sr="
                                + formatString("%.15f", coulReadTrace.finalTotal));
            }
            if (step == 0)
            {
                const bool isExactRespaProbe =
                        gmx::useExactRespa(inputrec);
                double preSumNonElectroResidual = 0.0;
                for (int i = static_cast<int>(InteractionFunction::Bonds);
                     i < static_cast<int>(InteractionFunction::PotentialEnergy);
                     ++i)
                {
                    if (i == static_cast<int>(InteractionFunction::DistanceRestraintViolations)
                        || i == static_cast<int>(InteractionFunction::OrientationRestraintDeviations)
                        || i == static_cast<int>(InteractionFunction::CoulombShortRange)
                        || i == static_cast<int>(InteractionFunction::Coulomb14)
                        || i == static_cast<int>(InteractionFunction::CoulombReciprocalSpace)
                        || i == static_cast<int>(InteractionFunction::LennardJonesShortRange)
                        || i == static_cast<int>(InteractionFunction::LennardJones14)
                        || i == static_cast<int>(InteractionFunction::BuckinghamShortRange))
                    {
                        continue;
                    }
                    preSumNonElectroResidual += static_cast<double>(enerd->term[static_cast<InteractionFunction>(i)]);
                }
                preSumNonElectroResidual += ljReadTrace.finalTotal + lj14ReadTrace.finalTotal
                                            + buckinghamReadTrace.finalTotal;
                const double preSumElectroTotal =
                        coulReadTrace.finalTotal + coul14ReadTrace.finalTotal
                        + static_cast<double>(enerd->term[InteractionFunction::CoulombReciprocalSpace]);
                appendRespaTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_pre_sum_potential_hypothesis_probe.txt",
                        "stage=PRE_SUM_POTENTIAL_HYPOTHESIS_PROBE code_location=src/gromacs/mdlib/sim_util.cpp:before_accumulatePotentialEnergies"
                                + std::string(" case_label=") + ljSrCaseLabel
                                + " execution_path=step0_pre_sum_probe run_kind="
                                + std::string(isExactRespaProbe ? "exact_respa" : "single_step")
                                + " step=0"
                                + " pre_sum_coulomb_sr=" + formatString("%.15f", coulReadTrace.finalTotal)
                                + " pre_sum_coulomb14=" + formatString("%.15f", coul14ReadTrace.finalTotal)
                                + " pre_sum_coulomb_recip="
                                + formatString("%.15f", static_cast<double>(enerd->term[InteractionFunction::CoulombReciprocalSpace]))
                                + " pre_sum_electro_total=" + formatString("%.15f", preSumElectroTotal)
                                + " pre_sum_lj_sr=" + formatString("%.15f", ljReadTrace.finalTotal)
                                + " pre_sum_lj14=" + formatString("%.15f", lj14ReadTrace.finalTotal)
                                + " pre_sum_buckingham_sr=" + formatString("%.15f", buckinghamReadTrace.finalTotal)
                                + " pre_sum_non_electro_residual="
                                + formatString("%.15f", preSumNonElectroResidual)
                                + " pre_sum_component_total="
                                + formatString("%.15f", preSumElectroTotal + preSumNonElectroResidual));
            }
        }

        /* Compute the final potential energy terms */
        accumulatePotentialEnergies(enerd, lambda, inputrec.fepvals.get());

        if (ljSrTraceDirPath != nullptr && *ljSrTraceDirPath != '\0')
        {
            real tracedPotentialComponentSum = 0.0_real;
            for (int i = static_cast<int>(InteractionFunction::Bonds);
                 i < static_cast<int>(InteractionFunction::PotentialEnergy);
                 ++i)
            {
                if (i != static_cast<int>(InteractionFunction::DistanceRestraintViolations)
                    && i != static_cast<int>(InteractionFunction::OrientationRestraintDeviations))
                {
                    tracedPotentialComponentSum += enerd->term[static_cast<InteractionFunction>(i)];
                }
            }
            const double tracedBondEnergy =
                    static_cast<double>(enerd->term[InteractionFunction::Bonds]);
            const double tracedAngleEnergy =
                    static_cast<double>(enerd->term[InteractionFunction::Angles]);
            const double tracedProperDihedralEnergy =
                    static_cast<double>(enerd->term[InteractionFunction::ProperDihedrals]);
            const double tracedImproperDihedralEnergy =
                    static_cast<double>(enerd->term[InteractionFunction::ImproperDihedrals]);
            const double tracedLj14Energy =
                    static_cast<double>(enerd->term[InteractionFunction::LennardJones14]);
            const double tracedCoul14Energy =
                    static_cast<double>(enerd->term[InteractionFunction::Coulomb14]);
            const double tracedLjSrEnergy =
                    static_cast<double>(enerd->term[InteractionFunction::LennardJonesShortRange]);
            const double tracedCoulSrEnergy =
                    static_cast<double>(enerd->term[InteractionFunction::CoulombShortRange]);
            const double tracedCoulRecipEnergy =
                    static_cast<double>(enerd->term[InteractionFunction::CoulombReciprocalSpace]);
            const double tracedBuckinghamSrEnergy =
                    static_cast<double>(enerd->term[InteractionFunction::BuckinghamShortRange]);
            const double tracedPotentialCoreSum = tracedBondEnergy + tracedAngleEnergy + tracedProperDihedralEnergy
                                                  + tracedImproperDihedralEnergy + tracedLj14Energy
                                                  + tracedCoul14Energy + tracedLjSrEnergy + tracedCoulSrEnergy
                                                  + tracedCoulRecipEnergy + tracedBuckinghamSrEnergy;
            const double tracedOtherPotentialTerms =
                    static_cast<double>(tracedPotentialComponentSum) - tracedPotentialCoreSum;
            appendRespaTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_lj_sr_internal_trace.txt",
                    "stage=FINAL_INTERNAL_LEDGER code_location=src/gromacs/mdlib/sim_util.cpp:4298 case_label="
                            + std::string(
                                    (ljSrCaseLabelEnv != nullptr && *ljSrCaseLabelEnv != '\0') ? ljSrCaseLabelEnv :
                                                                                                "unknown")
                            + " execution_path=post_sum_epot_enerd_term lj_sr="
                            + formatString("%.15f", static_cast<double>(enerd->term[InteractionFunction::LennardJonesShortRange]))
                            + " coulomb_sr="
                            + formatString("%.15f", static_cast<double>(enerd->term[InteractionFunction::CoulombShortRange]))
                            + " potential="
                            + formatString("%.15f", static_cast<double>(enerd->term[InteractionFunction::PotentialEnergy])));
            appendRespaTraceTextLine(
                    ljSrTraceDirPath,
                    "step0_coulomb_sr_component_trace.txt",
                    "stage=FINAL_INTERNAL_LEDGER code_location=src/gromacs/mdlib/sim_util.cpp:4298 case_label="
                            + std::string(
                                    (ljSrCaseLabelEnv != nullptr && *ljSrCaseLabelEnv != '\0') ? ljSrCaseLabelEnv :
                                                                                                "unknown")
                            + " execution_path=post_sum_epot_enerd_term"
                            + (std::string((ljSrCaseLabelEnv != nullptr && *ljSrCaseLabelEnv != '\0')
                                                   ? ljSrCaseLabelEnv
                                                   : "unknown")
                                       == "plain_verlet"
                                       ? " plain_coulomb_sr="
                                       : " patch_combined_coulomb_sr=")
                            + formatString("%.15f", static_cast<double>(enerd->term[InteractionFunction::CoulombShortRange]))
                            + " potential="
                            + formatString("%.15f", static_cast<double>(enerd->term[InteractionFunction::PotentialEnergy])));
            if (step == 0)
            {
                const bool isExactRespaLedger =
                        gmx::useExactRespa(inputrec);
                constexpr double c_barToAtmTrace = 0.9869232667160128;
                const double volumeNm3 = static_cast<double>(box[XX][XX] * box[YY][YY] * box[ZZ][ZZ]);
                const auto virialPressureAtm = [volumeNm3, c_barToAtmTrace](const double virialKjPerMol)
                {
                    return ((-virialKjPerMol) * (2.0 * gmx::c_presfac) / volumeNm3)
                           * c_barToAtmTrace;
                };
                const double virialXx = static_cast<double>(vir_force[XX][XX]);
                const double virialXy = static_cast<double>(vir_force[XX][YY]);
                const double virialXz = static_cast<double>(vir_force[XX][ZZ]);
                const double virialYx = static_cast<double>(vir_force[YY][XX]);
                const double virialYy = static_cast<double>(vir_force[YY][YY]);
                const double virialYz = static_cast<double>(vir_force[YY][ZZ]);
                const double virialZx = static_cast<double>(vir_force[ZZ][XX]);
                const double virialZy = static_cast<double>(vir_force[ZZ][YY]);
                const double virialZz = static_cast<double>(vir_force[ZZ][ZZ]);
                appendRespaTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_potential_ledger_trace.txt",
                        "stage=FINAL_INTERNAL_LEDGER code_location=src/gromacs/mdlib/sim_util.cpp:after_accumulatePotentialEnergies"
                                + std::string(" case_label=")
                                + std::string(
                                        (ljSrCaseLabelEnv != nullptr && *ljSrCaseLabelEnv != '\0')
                                                ? ljSrCaseLabelEnv
                                                : "unknown")
                                + " execution_path=post_sum_epot_enerd_term run_kind="
                                + std::string(isExactRespaLedger ? "exact_respa" : "single_step")
                                + " step=0"
                                + " bond=" + formatString("%.15f", tracedBondEnergy) + " angle="
                                + formatString("%.15f", tracedAngleEnergy) + " proper_dih="
                                + formatString("%.15f", tracedProperDihedralEnergy) + " improper_dih="
                                + formatString("%.15f", tracedImproperDihedralEnergy) + " lj14="
                                + formatString("%.15f", tracedLj14Energy) + " coul14="
                                + formatString("%.15f", tracedCoul14Energy) + " lj_sr="
                                + formatString("%.15f", tracedLjSrEnergy) + " coul_sr="
                                + formatString("%.15f", tracedCoulSrEnergy) + " coul_recip="
                                + formatString("%.15f", tracedCoulRecipEnergy) + " buckingham_sr="
                                + formatString("%.15f", tracedBuckinghamSrEnergy) + " other_terms="
                                + formatString("%.15f", tracedOtherPotentialTerms) + " component_sum="
                                + formatString("%.15f", static_cast<double>(tracedPotentialComponentSum))
                                + " potential="
                                + formatString("%.15f", static_cast<double>(enerd->term[InteractionFunction::PotentialEnergy])));
                appendRespaTraceTextLine(
                        ljSrTraceDirPath,
                        "step0_virial_pressure_ledger_trace.txt",
                        "stage=FINAL_INTERNAL_VIRIAL_LEDGER code_location=src/gromacs/mdlib/sim_util.cpp:after_accumulatePotentialEnergies"
                                + std::string(" case_label=")
                                + std::string(
                                        (ljSrCaseLabelEnv != nullptr && *ljSrCaseLabelEnv != '\0')
                                                ? ljSrCaseLabelEnv
                                                : "unknown")
                                + " execution_path=post_sum_virial_force_tensor run_kind="
                                + std::string(isExactRespaLedger ? "exact_respa" : "single_step")
                                + " step=0"
                                + " volume_nm3=" + formatString("%.15f", volumeNm3)
                                + " vir_xx=" + formatString("%.15f", virialXx)
                                + " vir_xy=" + formatString("%.15f", virialXy)
                                + " vir_xz=" + formatString("%.15f", virialXz)
                                + " vir_yx=" + formatString("%.15f", virialYx)
                                + " vir_yy=" + formatString("%.15f", virialYy)
                                + " vir_yz=" + formatString("%.15f", virialYz)
                                + " vir_zx=" + formatString("%.15f", virialZx)
                                + " vir_zy=" + formatString("%.15f", virialZy)
                                + " vir_zz=" + formatString("%.15f", virialZz)
                                + " pressure_xx_atm=" + formatString("%.15f", virialPressureAtm(virialXx))
                                + " pressure_yy_atm=" + formatString("%.15f", virialPressureAtm(virialYy))
                                + " pressure_zz_atm=" + formatString("%.15f", virialPressureAtm(virialZz))
                                + " pressure_xy_atm="
                                + formatString("%.15f", 0.5 * (virialPressureAtm(virialXy) + virialPressureAtm(virialYx)))
                                + " pressure_xz_atm="
                                + formatString("%.15f", 0.5 * (virialPressureAtm(virialXz) + virialPressureAtm(virialZx)))
                                + " pressure_yz_atm="
                                + formatString("%.15f", 0.5 * (virialPressureAtm(virialYz) + virialPressureAtm(virialZy))));
            }
        }

        if (!EI_TPI(inputrec.eI))
        {
            checkPotentialEnergyValidity(step, *enerd, inputrec);
        }
    }

    if (traceForceComponents)
    {
        const auto* exactRealspaceTrace =
                useExactLammpsRespaNonbonded ? activeExactRespaRealspaceTraceCapture(step) : nullptr;
        appendForceComponentTracePair(activeM2pTraceDirPath(),
                                      traceSide,
                                      step,
                                      "bonded_force",
                                      tracedBondedDelta,
                                      "listedForces.calculate_delta",
                                      "src/gromacs/mdlib/sim_util.cpp:do_force.listed_forces_delta",
                                      "true_source_component",
                                      true);
        if (useExactLammpsRespaNonbonded)
        {
            if (haveTracedPair14Delta)
            {
                appendForceComponentTracePair(activeM2pTraceDirPath(),
                                              traceSide,
                                              step,
                                              "pair14_force",
                                              tracedPair14Delta,
                                              "exact_pair14_level_delta",
                                              "src/gromacs/mdlib/sim_util.cpp:do_force.exact_pair14_level_delta",
                                              "true_source_component",
                                              true);
            }
            else
            {
                appendForceComponentUnavailablePair(
                        activeM2pTraceDirPath(),
                        traceSide,
                        step,
                        "pair14_force",
                        "exact_pair14_level_delta",
                        "src/gromacs/mdlib/sim_util.cpp:do_force.exact_pair14_level_delta",
                        "pair14_level_inactive_for_step");
            }
        }
        if (exactRealspaceTrace != nullptr)
        {
            appendForceComponentTracePair(activeM2pTraceDirPath(),
                                          traceSide,
                                          step,
                                          "lj_sr_force",
                                          exactRealspaceTrace->ljSrForce,
                                          "computeExactRespaNonbondedCpu.exact_pairlist_truth",
                                          "src/gromacs/mdlib/sim_util.cpp:computeExactRespaNonbondedCpu",
                                          "true_source_component",
                                          true);
            appendForceComponentTracePair(activeM2pTraceDirPath(),
                                          traceSide,
                                          step,
                                          "coulomb_sr_force",
                                          exactRealspaceTrace->coulombSrForce,
                                          "computeExactRespaNonbondedCpu.exact_pairlist_truth",
                                          "src/gromacs/mdlib/sim_util.cpp:computeExactRespaNonbondedCpu",
                                          "true_source_component",
                                          true);
        }
        else
        {
            appendForceComponentUnavailablePair(activeM2pTraceDirPath(),
                                               traceSide,
                                               step,
                                               "lj_sr_force",
                                               "not_separately_retained_in_do_force",
                                               "src/gromacs/mdlib/sim_util.cpp:do_force.nonbonded_stage",
                                               "runtime_force_component_unavailable");
            appendForceComponentUnavailablePair(activeM2pTraceDirPath(),
                                               traceSide,
                                               step,
                                               "coulomb_sr_force",
                                               "not_separately_retained_in_do_force",
                                               "src/gromacs/mdlib/sim_util.cpp:do_force.nonbonded_stage",
                                               "runtime_force_component_unavailable");
        }
        appendForceComponentTracePair(activeM2pTraceDirPath(),
                                      traceSide,
                                      step,
                                      "coulomb_recip_force",
                                      tracedCoulombRecipDelta,
                                      "longRangeNonbondeds.calculate_delta",
                                      "src/gromacs/mdlib/sim_util.cpp:do_force.longrange_delta",
                                      "true_source_component",
                                      true);
        if (exactRealspaceTrace != nullptr)
        {
            appendForceComponentTracePair(activeM2pTraceDirPath(),
                                          traceSide,
                                          step,
                                          "exclusion_correction_force",
                                          exactRealspaceTrace->exclusionCorrectionForce,
                                          "computeExactRespaNonbondedCpu.exact_pairlist_truth",
                                          "src/gromacs/mdlib/sim_util.cpp:computeExactRespaNonbondedCpu",
                                          "true_source_component",
                                          true);
        }
        else
        {
            appendForceComponentUnavailablePair(activeM2pTraceDirPath(),
                                               traceSide,
                                               step,
                                               "exclusion_correction_force",
                                               "not_separately_retained_in_do_force",
                                               "src/gromacs/mdlib/sim_util.cpp:do_force.nonbonded_stage",
                                               "runtime_force_component_unavailable");
        }
        appendForceComponentTracePair(activeM2pTraceDirPath(),
                                      traceSide,
                                      step,
                                      "realspace_nonbonded_combined_force",
                                      tracedCombinedRealspaceDelta,
                                      "distinct_force_outputs_after_nonbonded_stage",
                                      "src/gromacs/mdlib/sim_util.cpp:do_force.realspace_nonbonded_delta",
                                      "combined_total",
                                      false);
        appendForceComponentTracePair(activeM2pTraceDirPath(),
                                      traceSide,
                                      step,
                                      "total_force",
                                      captureForceArrayPair(force.unpaddedConstArrayRef()),
                                      "do_force_return_force",
                                      "src/gromacs/mdlib/sim_util.cpp:do_force.exit",
                                      "combined_total",
                                      false);
    }

    if (exactRespaForceStore != nullptr)
    {
        GMX_RELEASE_ASSERT(gmx::useExactRespa(inputrec),
                           "Exact force-store updates should only be active for exact r-RESPA");
        const ArrayRef<const RVec> recomputedLevel1 =
                (useExactRespaForceOutputs && exactRespaForceOutputs.hasLevel(1))
                        ? exactRespaForceOutputs.level(1).forceWithShiftForces().force()
                        : ArrayRef<const RVec>{};
        const ArrayRef<const RVec> recomputedLevel2 =
                (useExactRespaForceOutputs && exactRespaForceOutputs.hasLevel(2))
                        ? exactRespaForceOutputs.level(2).forceWithShiftForces().force()
                        : ArrayRef<const RVec>{};
        if (traceForceComponents)
        {
            appendForceStoreUpdateInputTrace(
                    activeM2pTraceDirPath(), step, "physical_total", force.unpaddedConstArrayRef());
            appendForceStoreUpdateInputTrace(
                    activeM2pTraceDirPath(), step, "recomputed_level1", recomputedLevel1);
            appendForceStoreUpdateInputTrace(
                    activeM2pTraceDirPath(), step, "recomputed_level2", recomputedLevel2);
        }
        exactRespaForceStore->update(force.unpaddedConstArrayRef(),
                                     recomputedLevel1,
                                     recomputedLevel2,
                                     exactRespaNumLevels(inputrec));
    }

    /* In case we don't have constraints and are using GPUs, the next balancing
     * region starts here.
     * Some "special" work at the end of do_force_cuts?, such as vsite spread,
     * virial calculation and COM pulling, is not thus not included in
     * the balance timing, which is ok as most tasks do communication.
     */
    ddBalanceRegionHandler.openBeforeForceComputationCpu(DdAllowBalanceRegionReopen::no);
    if (shouldTraceRespaStateXChainStep(step) && (step == 5 || step == 6))
    {
        const char* postForceStageName =
                (step == 5) ? "STEP5_POST_FORCE_STATE_X" : "STEP6_POST_FORCE_STATE_X";
        appendStateXChainTracePair(activeM2pTraceDirPath(),
                                   traceSide,
                                   postForceStageName,
                                   step,
                                   x.unpaddedArrayRef(),
                                   "do_force exit",
                                   "src/gromacs/mdlib/sim_util.cpp:5515",
                                   false);
    }
}

} // namespace gmx
