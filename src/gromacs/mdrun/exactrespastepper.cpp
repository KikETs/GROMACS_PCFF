/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2025- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */

#include "gmxpre.h"

#include "legacysimulator.h"
#include "exactrespasoftstart.h"
#include "exactrespasteppertesting.h"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "gromacs/domdec/collect.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/ewald/pme.h"
#if GMX_GPU
#    include "gromacs/gpu_utils/devicebuffer.h"
#endif
#include "gromacs/math/functions.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/mdlib/exactrespaimagetracker.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/md_support.h"
#include "gromacs/mdlib/update_constrain_gpu.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdtypes/state_propagator_data_gpu.h"
#include "gromacs/mdlib/vcm.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#if GMX_GPU
#    include "gromacs/nbnxm/gpu_types_common.h"
#endif
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/exactrespaforcestore.h"
#include "gromacs/mdtypes/exactrespaschedule.h"
#include "gromacs/mdtypes/forcebuffers.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdrunoptions.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/taskassignment/include/gromacs/taskassignment/decidesimulationworkload.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/vectypes.h"

namespace gmx
{

extern thread_local const char* g_respaDoForceContextLabel;

namespace
{
thread_local ExactRespaRuntimeEventSink* g_exactRespaRuntimeEventSink = nullptr;
thread_local std::unique_ptr<std::ofstream> g_exactRespaRuntimeEventTraceStream;
thread_local std::string                    g_exactRespaRuntimeEventTracePath;
thread_local bool                           g_exactRespaRuntimeEventTraceHasContent = false;
std::mutex                                  g_exactRespaStateTraceMutex;
}

enum class ExactRespaUpdateOmpMode : int
{
    Auto,
    Off,
    On
};

static ExactRespaUpdateOmpMode exactRespaUpdateOmpMode()
{
    static const ExactRespaUpdateOmpMode mode = []()
    {
        const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_UPDATE_OMP");
        if (env == nullptr || *env == '\0')
        {
            return ExactRespaUpdateOmpMode::Auto;
        }
        return (std::strcmp(env, "0") == 0) ? ExactRespaUpdateOmpMode::Off
                                            : ExactRespaUpdateOmpMode::On;
    }();

    return mode;
}

static bool exactRespaFusedInitialDriftEnabled()
{
    static const bool enabled = []()
    {
        const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT");
        return env == nullptr || *env == '\0' || std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool exactRespaFusedUpdateVectorEnabled()
{
    static const bool enabled = []()
    {
        const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_FUSED_UPDATE_VECTOR");
        return env == nullptr || *env == '\0' || std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool exactRespaGpuResidentXProbeEnabled()
{
    static const bool enabled =
            std::getenv("GMX_PCFF_EXACT_RESPA_GPU_RESIDENT_X_PROBE") != nullptr;
    return enabled;
}

static bool exactRespaGpuDeviceKickProbeEnabled()
{
    static const bool enabled =
            std::getenv("GMX_PCFF_EXACT_RESPA_GPU_DEVICE_KICK_PROBE") != nullptr;
    return enabled;
}

static bool exactRespaGpuDeviceKickForceAuditEnabled()
{
    static const bool enabled =
            std::getenv("GMX_PCFF_EXACT_RESPA_GPU_DEVICE_KICK_FORCE_AUDIT") != nullptr;
    return enabled;
}

#if GMX_GPU_CUDA
static void auditExactRespaGpuDeviceKickForces(const ExactRespaForceStore& forceStore,
                                              nonbonded_verlet_t*         nbv,
                                              StatePropagatorDataGpu*     stateGpu,
                                              DeviceBuffer<RVec>          level0Force,
                                              DeviceBuffer<RVec>          level1Force,
                                              DeviceBuffer<RVec>          level2Force,
                                              const int                   numAtoms)
{
    if (!exactRespaGpuDeviceKickForceAuditEnabled())
    {
        return;
    }

    const DeviceStream* stream = stateGpu->getUpdateStream();
    GMX_RELEASE_ASSERT(stream != nullptr, "Exact r-RESPA force audit requires an update stream");
    const int nbnxmAtoms = gpuGetNBAtomData(nbv->gpuNbv())->numAtomsAlloc;
    std::vector<RVec> deviceLevel0(nbnxmAtoms);
    std::vector<RVec> deviceLevel1(nbnxmAtoms);
    std::vector<RVec> deviceLevel2(numAtoms);
    copyFromDeviceBuffer(deviceLevel0.data(),
                         &level0Force,
                         0,
                         nbnxmAtoms,
                         *stream,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    copyFromDeviceBuffer(deviceLevel1.data(),
                         &level1Force,
                         0,
                         nbnxmAtoms,
                         *stream,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    copyFromDeviceBuffer(deviceLevel2.data(),
                         &level2Force,
                         0,
                         numAtoms,
                         *stream,
                         GpuApiCallBehavior::Sync,
                         nullptr);

    const auto stateToNbnxm = nbv->getGridIndices();
    std::fprintf(stderr,
                 "exact-respa device force audit order_matches=%d mapped0=%d mapped1=%d\n",
                 nbv->localAtomOrderMatchesNbnxmOrder(),
                 stateToNbnxm.empty() ? -1 : stateToNbnxm[0],
                 stateToNbnxm.ssize() < 2 ? -1 : stateToNbnxm[1]);
    for (int level = 0; level < forceStore.numLevels(); ++level)
    {
        const auto hostForce = forceStore.levelTotal(level);
        double     sumSquaredDifference = 0;
        double     sumSquaredReference  = 0;
        double     maxAbsoluteDifference = 0;
        for (int atom = 0; atom < numAtoms; ++atom)
        {
            const int nbnxmAtom = stateToNbnxm[atom];
            const RVec& deviceForce = level == 0   ? deviceLevel0[nbnxmAtom]
                                      : level == 1 ? deviceLevel1[nbnxmAtom]
                                                   : deviceLevel2[atom];
            for (int dim = 0; dim < DIM; ++dim)
            {
                const double difference = deviceForce[dim] - hostForce[atom][dim];
                sumSquaredDifference += difference * difference;
                sumSquaredReference += hostForce[atom][dim] * hostForce[atom][dim];
                maxAbsoluteDifference = std::max(maxAbsoluteDifference, std::abs(difference));
            }
        }
        std::fprintf(stderr,
                     "exact-respa device force audit level=%d rel_l2=%.9g max_abs=%.9g\n",
                     level,
                     std::sqrt(sumSquaredDifference / std::max(sumSquaredReference, 1.0e-300)),
                     maxAbsoluteDifference);
    }
}
#endif

static bool exactRespaReturnNextVirialEnabled()
{
    static const bool enabled = []()
    {
        const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_RETURN_NEXT_VIRIAL");
        return env != nullptr && *env != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool exactRespaPostTrotterReplayNeedsNextVirialForTesting(const char* value)
{
    // md.cpp defaults an unset post-Trotter replay to sequence Three. Both
    // sequences Two and Three consume the pressure tensor from the newly
    // evaluated coordinates, so their replay requires the next-step virial.
    if (value == nullptr || *value == '\0')
    {
        return true;
    }
    return std::strcmp(value, "2") == 0 || std::strcmp(value, "two") == 0
           || std::strcmp(value, "3") == 0 || std::strcmp(value, "three") == 0
           || std::strcmp(value, "2,3") == 0 || std::strcmp(value, "two-three") == 0
           || std::strcmp(value, "3,2") == 0 || std::strcmp(value, "three-two") == 0;
}

static bool exactRespaPostTrotterReplayIncludesFinalHalf()
{
    static const bool enabled = [] {
        const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_POST_TROTTER");
        return exactRespaPostTrotterReplayNeedsNextVirialForTesting(value);
    }();
    return enabled;
}

static bool exactRespaMttkPostTrotterNeedsNextVirial(const t_inputrec& inputRecord,
                                                     const int64_t     baseStep)
{
    return inputRecord.exactRespa.enabled() && inputRecord.eI == IntegrationAlgorithm::VV
           && inputRecord.pressureCouplingOptions.epc == PressureCoupling::Mttk
           && exactRespaPostTrotterReplayIncludesFinalHalf()
           && (inputRecord.nsttcouple == 1 || ((baseStep + 1) % inputRecord.nsttcouple == 0));
}

static int exactRespaUpdateThreadOverride()
{
    static const int requestedThreads = []()
    {
        const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_UPDATE_OMP_THREADS");
        if (env == nullptr || *env == '\0')
        {
            return 0;
        }
        return std::max(0, std::atoi(env));
    }();
    return requestedThreads;
}

static bool exactRespaDirectUpdateFastPathEnabled()
{
    static const bool enabled = []()
    {
        const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_UPDATE_DIRECT_FASTPATH");
        return env == nullptr || *env == '\0' || std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static int exactRespaUpdateThreadCount(const int numAtoms)
{
    int numThreads = 1;
    switch (exactRespaUpdateOmpMode())
    {
        case ExactRespaUpdateOmpMode::Off: return 1;
        case ExactRespaUpdateOmpMode::On:
            numThreads = std::max(1, gmx_omp_nthreads_get(ModuleMultiThread::Update));
            break;
        case ExactRespaUpdateOmpMode::Auto:
            numThreads = std::max(1, gmx_omp_nthreads_get_simple_rvec_task(ModuleMultiThread::Update, numAtoms));
            break;
    }

    const int overrideThreads = exactRespaUpdateThreadOverride();
    if (overrideThreads > 0)
    {
        numThreads = std::min(numThreads, overrideThreads);
    }
    return numThreads;
}

static bool exactRespaCanUseDirectUpdatePath(const int                         numAtoms,
                                             const ArrayRef<const ParticleType> ptype,
                                             const ArrayRef<const RVec>         invMassPerDim)
{
    if (!exactRespaDirectUpdateFastPathEnabled() || numAtoms <= 0
        || ptype.ssize() < numAtoms || invMassPerDim.ssize() < numAtoms)
    {
        return false;
    }

    struct Cache
    {
        const ParticleType* ptypeData         = nullptr;
        const RVec*         invMassPerDimData = nullptr;
        int                 numAtoms          = 0;
        bool                directPath        = false;
    };
    static thread_local Cache cache;

    if (cache.ptypeData == ptype.data() && cache.invMassPerDimData == invMassPerDim.data()
        && cache.numAtoms == numAtoms)
    {
        return cache.directPath;
    }

    bool directPath = true;
    for (int atom = 0; atom < numAtoms; ++atom)
    {
        if (ptype[atom] == ParticleType::Shell || invMassPerDim[atom][XX] == 0
            || invMassPerDim[atom][YY] == 0 || invMassPerDim[atom][ZZ] == 0)
        {
            directPath = false;
            break;
        }
    }

    cache = Cache{ ptype.data(), invMassPerDim.data(), numAtoms, directPath };
    return directPath;
}

template<typename Body>
static void exactRespaUpdateForAtoms(const int numAtoms, const Body& body)
{
    const int numThreads = exactRespaUpdateThreadCount(numAtoms);
    if (numThreads <= 1)
    {
        for (int atom = 0; atom < numAtoms; atom++)
        {
            body(atom);
        }
        return;
    }

#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int atom = 0; atom < numAtoms; atom++)
    {
        body(atom);
    }
}

template<int NumKicks>
static void exactRespaFusedPlainKickDriftRange(const int                   beginAtom,
                                              const int                   endAtom,
                                              const RVec* gmx_restrict    invMassPerDim,
                                              const RVec* gmx_restrict    force0,
                                              const RVec* gmx_restrict    force1,
                                              const RVec* gmx_restrict    force2,
                                              const real                  scale0,
                                              const real                  scale1,
                                              const real                  scale2,
                                              const real                  driftDt,
                                              RVec* gmx_restrict          position,
                                              RVec* gmx_restrict          velocity)
{
    static_assert(NumKicks >= 1 && NumKicks <= ExactRespaForceStore::c_numStoredLevels);
    for (int atom = beginAtom; atom < endAtom; ++atom)
    {
        for (int d = 0; d < DIM; ++d)
        {
            const real inverseMass = invMassPerDim[atom][d];
            real       updatedVelocity = velocity[atom][d];
            updatedVelocity += scale0 * inverseMass * force0[atom][d];
            if constexpr (NumKicks >= 2)
            {
                updatedVelocity += scale1 * inverseMass * force1[atom][d];
            }
            if constexpr (NumKicks >= 3)
            {
                updatedVelocity += scale2 * inverseMass * force2[atom][d];
            }
            velocity[atom][d] = updatedVelocity;
            position[atom][d] += driftDt * updatedVelocity;
        }
    }
}

template<int NumKicks>
static void exactRespaFusedPlainKickDrift(const int                numAtoms,
                                         const RVec* gmx_restrict invMassPerDim,
                                         const RVec* gmx_restrict force0,
                                         const RVec* gmx_restrict force1,
                                         const RVec* gmx_restrict force2,
                                         const real               scale0,
                                         const real               scale1,
                                         const real               scale2,
                                         const real               driftDt,
                                         RVec* gmx_restrict       position,
                                         RVec* gmx_restrict       velocity)
{
    const int numThreads = exactRespaUpdateThreadCount(numAtoms);
    if (numThreads <= 1)
    {
        exactRespaFusedPlainKickDriftRange<NumKicks>(0,
                                                    numAtoms,
                                                    invMassPerDim,
                                                    force0,
                                                    force1,
                                                    force2,
                                                    scale0,
                                                    scale1,
                                                    scale2,
                                                    driftDt,
                                                    position,
                                                    velocity);
        return;
    }

#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int atom = 0; atom < numAtoms; ++atom)
    {
        for (int d = 0; d < DIM; ++d)
        {
            const real inverseMass = invMassPerDim[atom][d];
            real       updatedVelocity = velocity[atom][d];
            updatedVelocity += scale0 * inverseMass * force0[atom][d];
            if constexpr (NumKicks >= 2)
            {
                updatedVelocity += scale1 * inverseMass * force1[atom][d];
            }
            if constexpr (NumKicks >= 3)
            {
                updatedVelocity += scale2 * inverseMass * force2[atom][d];
            }
            velocity[atom][d] = updatedVelocity;
            position[atom][d] += driftDt * updatedVelocity;
        }
    }
}

void setExactRespaRuntimeEventSinkForTesting(ExactRespaRuntimeEventSink* sink)
{
    g_exactRespaRuntimeEventSink = sink;
}

namespace
{

struct ExactRespaStateTraceConfig
{
    bool        enabled     = false;
    std::string path;
    int         atomCount   = 8;
    int64_t     maxBaseStep = 16;
    bool        includePositions = true;
};

struct ExactRespaForceStoreTraceConfig
{
    bool        enabled     = false;
    std::string path;
    int         atomCount   = 8;
    int64_t     maxBaseStep = 16;
};

std::string exactRespaTraceSidecarPath(const std::string& path, const char* suffix)
{
    return path + suffix;
}

int canonicalAtomIndexForTraceAtom(const int*                  ddGlobalAtomIndices,
                                   const int                   ddGlobalAtomIndicesCount,
                                   const nonbonded_verlet_t*   nbv,
                                   const int                   atomIndex)
{
    if (nbv == nullptr || atomIndex < 0)
    {
        if (ddGlobalAtomIndices != nullptr && atomIndex < ddGlobalAtomIndicesCount)
        {
            return ddGlobalAtomIndices[atomIndex];
        }
        return atomIndex;
    }

    if (ddGlobalAtomIndices != nullptr && atomIndex < ddGlobalAtomIndicesCount)
    {
        return ddGlobalAtomIndices[atomIndex];
    }

    return atomIndex;
}

void initializeExactRespaAtomOrderSidecar(const std::string& path, const char* stageColumnName)
{
    std::ofstream output(path, std::ios::trunc);
    GMX_RELEASE_ASSERT(output.good(), "Could not open exact r-RESPA atom-order sidecar for writing");
    output << "# base_step\t" << stageColumnName << "\tatom\tcanonical_atom\n";
}

void appendExactRespaAtomOrderRows(const std::string&        path,
                                   const int64_t            baseStep,
                                   const char*              stageOrPhaseLabel,
                                   const int                atomCount,
                                   const int*               ddGlobalAtomIndices,
                                   const int                ddGlobalAtomIndicesCount,
                                   const nonbonded_verlet_t* nbv)
{
    std::ofstream output(path, std::ios::app);
    GMX_RELEASE_ASSERT(output.good(), "Could not open exact r-RESPA atom-order sidecar for appending");
    for (int atom = 0; atom < atomCount; ++atom)
    {
        output << baseStep << '\t' << stageOrPhaseLabel << '\t' << atom << '\t'
               << canonicalAtomIndexForTraceAtom(ddGlobalAtomIndices, ddGlobalAtomIndicesCount, nbv, atom)
               << '\n';
    }
}

const ExactRespaStateTraceConfig& exactRespaStateTraceConfig()
{
    static const ExactRespaStateTraceConfig config = []()
    {
        ExactRespaStateTraceConfig result;
        if (const char* value = std::getenv("GMX_EXACT_RESPA_STATE_TRACE_FILE"))
        {
            if (*value != '\0')
            {
                result.enabled = true;
                result.path    = value;
                if (const char* atomCount = std::getenv("GMX_EXACT_RESPA_STATE_TRACE_ATOMS"))
                {
                    result.atomCount = std::max(1, std::atoi(atomCount));
                }
                if (const char* maxBaseStep = std::getenv("GMX_EXACT_RESPA_STATE_TRACE_MAX_BASE_STEP"))
                {
                    result.maxBaseStep = std::max<int64_t>(0, std::atoll(maxBaseStep));
                }
                if (const char* includePositions =
                            std::getenv("GMX_EXACT_RESPA_STATE_TRACE_INCLUDE_POSITIONS"))
                {
                    result.includePositions = (std::atoi(includePositions) != 0);
                }

                const std::filesystem::path tracePath(result.path);
                if (tracePath.has_parent_path())
                {
                    std::filesystem::create_directories(tracePath.parent_path());
                }

                std::ofstream output(tracePath, std::ios::trunc);
                GMX_RELEASE_ASSERT(output.good(),
                                   "Could not open GMX_EXACT_RESPA_STATE_TRACE_FILE for writing");
                output << "# base_step\tstage\tatom\tc0\tc1\tc2\n";
                initializeExactRespaAtomOrderSidecar(
                        exactRespaTraceSidecarPath(result.path, ".atom_order.tsv"), "stage");
            }
        }
        return result;
    }();

    return config;
}

const ExactRespaForceStoreTraceConfig& exactRespaForceStoreTraceConfig()
{
    static const ExactRespaForceStoreTraceConfig config = []()
    {
        ExactRespaForceStoreTraceConfig result;
        if (const char* value = std::getenv("GMX_EXACT_RESPA_FORCESTORE_TRACE_FILE"))
        {
            if (*value != '\0')
            {
                result.enabled = true;
                result.path    = value;
                if (const char* atomCount = std::getenv("GMX_EXACT_RESPA_FORCESTORE_TRACE_ATOMS"))
                {
                    result.atomCount = std::max(1, std::atoi(atomCount));
                }
                else if (const char* stateAtomCount = std::getenv("GMX_EXACT_RESPA_STATE_TRACE_ATOMS"))
                {
                    result.atomCount = std::max(1, std::atoi(stateAtomCount));
                }
                if (const char* maxBaseStep = std::getenv("GMX_EXACT_RESPA_FORCESTORE_TRACE_MAX_BASE_STEP"))
                {
                    result.maxBaseStep = std::max<int64_t>(0, std::atoll(maxBaseStep));
                }
                else if (const char* stateMaxBaseStep = std::getenv("GMX_EXACT_RESPA_STATE_TRACE_MAX_BASE_STEP"))
                {
                    result.maxBaseStep = std::max<int64_t>(0, std::atoll(stateMaxBaseStep));
                }

                const std::filesystem::path tracePath(result.path);
                if (tracePath.has_parent_path())
                {
                    std::filesystem::create_directories(tracePath.parent_path());
                }

                std::ofstream output(tracePath, std::ios::trunc);
                GMX_RELEASE_ASSERT(output.good(),
                                   "Could not open GMX_EXACT_RESPA_FORCESTORE_TRACE_FILE for writing");
                output << "# base_step\tphase\tlevel\tatom\tfx\tfy\tfz\n";
                initializeExactRespaAtomOrderSidecar(
                        exactRespaTraceSidecarPath(result.path, ".atom_order.tsv"), "phase");
            }
        }
        return result;
    }();

    return config;
}

bool shouldRecordExactRespaStateTrace(const int64_t baseStep)
{
    const auto& config = exactRespaStateTraceConfig();
    return config.enabled && baseStep <= config.maxBaseStep;
}

void appendExactRespaStateTraceRows(const int64_t                     baseStep,
                                    const char*                       stage,
                                    const ArrayRef<const RVec>        values,
                                    const int*                        ddGlobalAtomIndices,
                                    const int                         ddGlobalAtomIndicesCount,
                                    const nonbonded_verlet_t*         nbv)
{
    if (!shouldRecordExactRespaStateTrace(baseStep))
    {
        return;
    }

    const auto& config = exactRespaStateTraceConfig();
    const int   atomCount =
            std::min<int>(config.atomCount, static_cast<int>(values.ssize()));

    std::lock_guard<std::mutex> lock(g_exactRespaStateTraceMutex);
    std::ofstream               output(config.path, std::ios::app);
    GMX_RELEASE_ASSERT(output.good(),
                       "Could not open GMX_EXACT_RESPA_STATE_TRACE_FILE for appending");
    output << std::setprecision(17);
    for (int atom = 0; atom < atomCount; ++atom)
    {
        output << baseStep << '\t' << stage << '\t' << atom << '\t' << values[atom][XX] << '\t'
               << values[atom][YY] << '\t' << values[atom][ZZ] << '\n';
    }
    appendExactRespaAtomOrderRows(
            exactRespaTraceSidecarPath(config.path, ".atom_order.tsv"),
            baseStep,
            stage,
            atomCount,
            ddGlobalAtomIndices,
            ddGlobalAtomIndicesCount,
            nbv);
}

bool shouldRecordExactRespaForceStoreTrace(const int64_t baseStep)
{
    const auto& config = exactRespaForceStoreTraceConfig();
    return config.enabled && baseStep <= config.maxBaseStep;
}

void appendExactRespaForceStoreTraceRows(const int64_t              baseStep,
                                         const char*                phaseLabel,
                                         const int                  level,
                                         const ArrayRef<const RVec> values,
                                         const int*                 ddGlobalAtomIndices,
                                         const int                  ddGlobalAtomIndicesCount,
                                         const nonbonded_verlet_t*  nbv)
{
    if (!shouldRecordExactRespaForceStoreTrace(baseStep))
    {
        return;
    }

    const auto& config = exactRespaForceStoreTraceConfig();
    const int   atomCount =
            std::min<int>(config.atomCount, static_cast<int>(values.ssize()));

    std::lock_guard<std::mutex> lock(g_exactRespaStateTraceMutex);
    std::ofstream               output(config.path, std::ios::app);
    GMX_RELEASE_ASSERT(output.good(),
                       "Could not open GMX_EXACT_RESPA_FORCESTORE_TRACE_FILE for appending");
    output << std::setprecision(17);
    for (int atom = 0; atom < atomCount; ++atom)
    {
        output << baseStep << '\t' << phaseLabel << '\t' << level << '\t' << atom << '\t'
               << values[atom][XX] << '\t' << values[atom][YY] << '\t' << values[atom][ZZ] << '\n';
    }
    appendExactRespaAtomOrderRows(exactRespaTraceSidecarPath(config.path, ".atom_order.tsv"),
                                  baseStep,
                                  phaseLabel,
                                  atomCount,
                                  ddGlobalAtomIndices,
                                  ddGlobalAtomIndicesCount,
                                  nbv);
}

gmx_enerdata_t& exactRespaNestedForceScratchEnerd(const gmx_enerdata_t& templateEnerd)
{
    using ForeignLambdaTable =
            gmx::EnumerationArray<FreeEnergyPerturbationCouplingType, std::vector<double>>;

    thread_local std::unique_ptr<gmx_enerdata_t> scratchEnerd;
    thread_local const ForeignLambdaTable*       scratchLambdaTable = nullptr;

    const auto* lambdaTable = templateEnerd.foreignLambdaTerms.allLambdasSource();
    if (!scratchEnerd || scratchEnerd->grpp.nener != templateEnerd.grpp.nener
        || scratchLambdaTable != lambdaTable)
    {
        scratchEnerd       = std::make_unique<gmx_enerdata_t>(templateEnerd);
        scratchLambdaTable = lambdaTable;
    }

    return *scratchEnerd;
}

enum class RespaKickPhase : int
{
    Initial,
    Final
};

ExactRespaRuntimeEventType exactRespaRuntimeEventTypeFromKickPhase(const RespaKickPhase phase)
{
    return (phase == RespaKickPhase::Initial) ? ExactRespaRuntimeEventType::InitialKick
                                              : ExactRespaRuntimeEventType::FinalKick;
}

const char* exactRespaRuntimeEventTypeLabel(const ExactRespaRuntimeEventType type)
{
    switch (type)
    {
        case ExactRespaRuntimeEventType::InitialKick: return "kick";
        case ExactRespaRuntimeEventType::Drift: return "drift";
        case ExactRespaRuntimeEventType::RefreshForce: return "force";
        case ExactRespaRuntimeEventType::FinalKick: return "final_kick";
    }

    return "unknown";
}

bool exactRespaRuntimeEventArtifactRequested()
{
    static const bool requested = []()
    {
        const char* traceFilePath = std::getenv("GMX_EXACT_RESPA_RUNTIME_EVENT_TRACE_FILE");
        return traceFilePath != nullptr && *traceFilePath != '\0';
    }();

    return requested;
}

bool exactRespaRuntimeEventInstrumentationEnabled()
{
    return g_exactRespaRuntimeEventSink != nullptr || exactRespaRuntimeEventArtifactRequested();
}

void maybeRecordExactRespaRuntimeEventArtifact(const ExactRespaRuntimeEvent& event)
{
    const char* traceFilePath = std::getenv("GMX_EXACT_RESPA_RUNTIME_EVENT_TRACE_FILE");
    if (traceFilePath == nullptr || *traceFilePath == '\0')
    {
        return;
    }

    const bool startingNewTrace = (event.baseStep == 0 && event.type == ExactRespaRuntimeEventType::InitialKick
                                   && g_exactRespaRuntimeEventTraceHasContent);
    if (!g_exactRespaRuntimeEventTraceStream || g_exactRespaRuntimeEventTracePath != traceFilePath
        || startingNewTrace)
    {
        const std::filesystem::path outputPath(traceFilePath);
        if (outputPath.has_parent_path())
        {
            std::filesystem::create_directories(outputPath.parent_path());
        }

        g_exactRespaRuntimeEventTraceStream =
                std::make_unique<std::ofstream>(outputPath, std::ios::out | std::ios::trunc);
        GMX_RELEASE_ASSERT(g_exactRespaRuntimeEventTraceStream->is_open(),
                           "Could not open exact r-RESPA runtime trace artifact");
        *g_exactRespaRuntimeEventTraceStream << "# base_step\tevent\tlevel\n";
        g_exactRespaRuntimeEventTracePath       = traceFilePath;
        g_exactRespaRuntimeEventTraceHasContent = false;
    }

    *g_exactRespaRuntimeEventTraceStream << event.baseStep << '\t'
                                         << exactRespaRuntimeEventTypeLabel(event.type) << '\t'
                                         << event.level << '\n';
    g_exactRespaRuntimeEventTraceStream->flush();
    g_exactRespaRuntimeEventTraceHasContent = true;
}

void recordExactRespaRuntimeEventForTesting(const int64_t                  baseStep,
                                            const ExactRespaRuntimeEventType type,
                                            const int                      level)
{
    if (!exactRespaRuntimeEventInstrumentationEnabled())
    {
        return;
    }

    const ExactRespaRuntimeEvent event{ baseStep, type, level };
    if (exactRespaRuntimeEventArtifactRequested())
    {
        maybeRecordExactRespaRuntimeEventArtifact(event);
    }

    if (g_exactRespaRuntimeEventSink == nullptr)
    {
        return;
    }
    g_exactRespaRuntimeEventSink->recordEvent(event);
}

void recordExactRespaRefreshEventsForTesting(const ExactRespaParameters& exactRespa, const int64_t baseStep)
{
    if (!exactRespaRuntimeEventInstrumentationEnabled())
    {
        return;
    }

    const ExactRespaBaseStepTrace trace = exactRespaBaseStepTrace(exactRespa, baseStep);
    for (const int level : trace.refreshedForceLevels)
    {
        recordExactRespaRuntimeEventForTesting(baseStep, ExactRespaRuntimeEventType::RefreshForce, level);
    }
}

ArrayRef<const RVec> exactRespaLevelForceOrEmpty(const ExactRespaForceStore* exactRespaForceStore, const int mtsLevel)
{
    if (exactRespaForceStore == nullptr || mtsLevel < 0 || mtsLevel >= ExactRespaForceStore::c_numStoredLevels
        || !exactRespaForceStore->hasLevel(mtsLevel))
    {
        return {};
    }

    return exactRespaForceStore->levelTotal(mtsLevel);
}

ArrayRef<const RVec> forceForExactRespaKickLevel(const ExactRespaForceStore*      exactRespaForceStore,
                                                 const int                        mtsLevel)
{
    GMX_RELEASE_ASSERT(exactRespaForceStore != nullptr, "Need exact r-RESPA force totals for kick selection");
    GMX_RELEASE_ASSERT(exactRespaForceStore->hasLevel(mtsLevel),
                       "Requested exact r-RESPA kick level should be stored");

    return exactRespaLevelForceOrEmpty(exactRespaForceStore, mtsLevel);
}

struct ExactRespaPreparedHalfKicks
{
    std::array<ArrayRef<const RVec>, ExactRespaForceStore::c_numStoredLevels> forcesPerKick = {};
    std::array<real, ExactRespaForceStore::c_numStoredLevels>                  dtPerKick     = {};
    std::array<int, ExactRespaForceStore::c_numStoredLevels>                   levelPerKick  = {};
    int                                                                        numKicks      = 0;
};

struct ExactRespaExtendedVvUpdate
{
    bool scaleVelocities     = false;
    bool scalePositions      = false;
    bool lammpsVelocityScale = false;
    bool lammpsRemapPositions = false;
    bool inlineBoxRemap      = false;
    real veta            = 0;
    real alpha           = 1;
};

static real exactRespaMttkVetaScale()
{
    static const real scale = [] {
        const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_MTTK_VETA_SCALE");
        if (env == nullptr || *env == '\0')
        {
            return 1.0_real;
        }
        char*      end    = nullptr;
        const real parsed = static_cast<real>(std::strtod(env, &end));
        GMX_RELEASE_ASSERT(end != env && (end == nullptr || *end == '\0'),
                           "GMX_PCFF_EXACT_RESPA_MTTK_VETA_SCALE must be a floating point value");
        return parsed;
    }();
    return scale;
}

static real exactRespaLammpsFixNhVelocityAlpha(const t_inputrec& inputRecord)
{
    static const double natomsOverride = [] {
        const char* natomsText = std::getenv("GMX_PCFF_MTTK_LAMMPS_NATOMS");
        if (natomsText == nullptr || *natomsText == '\0')
        {
            return 0.0;
        }
        char*        end    = nullptr;
        const double natoms = std::strtod(natomsText, &end);
        GMX_RELEASE_ASSERT(end != natomsText && (end == nullptr || *end == '\0') && natoms > 0,
                           "GMX_PCFF_MTTK_LAMMPS_NATOMS must be a positive floating point value");
        return natoms;
    }();
    if (natomsOverride > 0.0)
    {
        return 1.0_real + 1.0_real / static_cast<real>(natomsOverride);
    }

    GMX_RELEASE_ASSERT(inputRecord.opts.nrdf[0] > 0,
                       "LAMMPS MTTK velocity scaling requires either natoms or non-zero degrees of freedom");
    return 1.0_real + DIM / static_cast<real>(inputRecord.opts.nrdf[0]);
}

ExactRespaExtendedVvUpdate exactRespaExtendedVvUpdateFromState(const t_inputrec& inputRecord,
                                                               const t_state&    state)
{
    ExactRespaExtendedVvUpdate update;
    const bool mttkVv = inputRecord.eI == IntegrationAlgorithm::VV
                        && inputRecord.pressureCouplingOptions.epc == PressureCoupling::Mttk;
    static const char* mode = [] {
        const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_MTTK_EXTENDED_UPDATE");
        return (value != nullptr && *value != '\0') ? value : nullptr;
    }();
    if (mttkVv)
    {
        const bool useDefault = false;
        if (mode != nullptr && *mode != '\0' && std::strcmp(mode, "0") != 0
            && std::strcmp(mode, "none") != 0)
        {
            // LAMMPS FixNH r-RESPA remaps the box around each innermost drift.
            // Keeping the legacy GROMACS MTTK box update after the full drift
            // makes forces see the old box during pressure ramps and can drive
            // the barostat unstable.
            const bool velocityLammpsRemap = std::strcmp(mode, "velocity-lammps-remap") == 0;
            update.scaleVelocities = useDefault || std::strcmp(mode, "velocity-only") == 0
                                     || std::strcmp(mode, "velocity-position") == 0
                                     || std::strcmp(mode, "1") == 0;
            update.scalePositions = useDefault || std::strcmp(mode, "position-only") == 0
                                     || std::strcmp(mode, "velocity-position") == 0
                                     || std::strcmp(mode, "1") == 0;
            update.lammpsVelocityScale = velocityLammpsRemap;
            update.lammpsRemapPositions =
                    useDefault || std::strcmp(mode, "lammps-remap") == 0 || velocityLammpsRemap;
            update.inlineBoxRemap       = update.scalePositions || update.lammpsRemapPositions;
        }
    }
    update.veta = (update.scaleVelocities || update.scalePositions || update.lammpsVelocityScale
                   || update.lammpsRemapPositions)
                          ? state.veta * exactRespaMttkVetaScale()
                          : 0;
    if (update.scaleVelocities || update.scalePositions || update.lammpsVelocityScale)
    {
        GMX_RELEASE_ASSERT(inputRecord.opts.nrdf[0] > 0,
                           "MTTK exact r-RESPA update requires non-zero degrees of freedom");
        update.alpha = update.lammpsVelocityScale
                               ? exactRespaLammpsFixNhVelocityAlpha(inputRecord)
                               : 1.0_real + DIM / static_cast<real>(inputRecord.opts.nrdf[0]);
    }
    return update;
}

static bool exactRespaMttkInlineBoxRemapEnabled(const t_inputrec& inputRecord)
{
    if (inputRecord.eI != IntegrationAlgorithm::VV
        || inputRecord.pressureCouplingOptions.epc != PressureCoupling::Mttk
        || inputRecord.pressureCouplingOptions.epct != PressureCouplingType::Isotropic)
    {
        return false;
    }
    static const bool requested = [] {
        const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_MTTK_INLINE_BOX_REMAP");
        return env != nullptr && *env != '\0' && std::strcmp(env, "0") != 0;
    }();
    if (!requested)
    {
        return false;
    }
    static const bool realOnlyRequested = [] {
        const char* realOnlyEnv = std::getenv("GMX_PCFF_EWALD_REAL_ONLY");
        return realOnlyEnv != nullptr && *realOnlyEnv != '\0' && std::strcmp(realOnlyEnv, "0") != 0;
    }();
    if (usingPmeOrEwald(inputRecord.coulombtype) && !realOnlyRequested)
    {
        static const bool allowPmeInlineRemap = [] {
            const char* allowPmeEnv = std::getenv("GMX_PCFF_EXACT_RESPA_MTTK_INLINE_BOX_REMAP_PME");
            return allowPmeEnv != nullptr && *allowPmeEnv != '\0' && std::strcmp(allowPmeEnv, "0") != 0;
        }();
        if (allowPmeInlineRemap)
        {
            return true;
        }
        // Inline box remapping changes the box inside the exact-rRESPA drift.
        // Reciprocal PME state is not rebuilt at that point, so PME stages can
        // see inconsistent box/mesh state and drive the MTTK box to NaN.
        return false;
    }
    return true;
}

static bool exactRespaStatePbcWrappingEnabled()
{
    static const bool enabled = [] {
        const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_WRAP_STATE_IN_BOX");
        return env != nullptr && *env != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static void exactRespaMttkScaleBox(t_state& state, const real deltaT)
{
    const real veta  = state.veta * exactRespaMttkVetaScale();
    const real scale = std::exp(veta * deltaT);
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            state.box[i][j] *= scale;
            state.boxv[i][j] = veta * state.box[i][j];
        }
    }
}

static inline real exactRespaVvVelocityAfterHalfKick(const real currentVelocity,
                                                     const real invMass,
                                                     const real force,
                                                     const real dt,
                                                     const ExactRespaExtendedVvUpdate& extendedUpdate)
{
    if (!extendedUpdate.scaleVelocities)
    {
        return currentVelocity + 0.5_real * dt * invMass * force;
    }

    const real g   = 0.25_real * dt * extendedUpdate.veta * extendedUpdate.alpha;
    const real mv1 = std::exp(-g);
    const real mv2 = gmx::series_sinhx(g);
    return mv1 * (mv1 * currentVelocity + 0.5_real * dt * invMass * mv2 * force);
}

ExactRespaExtendedVvUpdate exactRespaPlainKickUpdate(ExactRespaExtendedVvUpdate update)
{
    update.scaleVelocities     = false;
    update.lammpsVelocityScale = false;
    return update;
}

int exactRespaMttkOuterStepLevel(const t_inputrec& inputRecord)
{
    const int pairSplitOuterLevel = inputRecord.exactRespa.forceLayout.outerLevel;
    if (pairSplitOuterLevel > 0)
    {
        return pairSplitOuterLevel;
    }

    // forceLayout.outerLevel describes the optional real-space pair splitting
    // window. A value of 0 is the "no explicit pair-split outer window" sentinel
    // used by the PolyGen 2-level schedule. FixNH/MTTK still couples on the
    // slowest r-RESPA level, matching LAMMPS initial_integrate_respa().
    return exactRespaNumLevels(inputRecord) - 1;
}

bool exactRespaPreparedKicksIncludeMttkOuterStep(const t_inputrec&                 inputRecord,
                                                 const ExactRespaPreparedHalfKicks& preparedHalfKicks)
{
    const int outerLevel = exactRespaMttkOuterStepLevel(inputRecord);
    for (int kickIndex = 0; kickIndex < preparedHalfKicks.numKicks; ++kickIndex)
    {
        if (preparedHalfKicks.levelPerKick[kickIndex] == outerLevel)
        {
            return true;
        }
    }
    return false;
}

real exactRespaOuterLevelDt(const t_inputrec& inputRecord)
{
    const int outerLevel = exactRespaMttkOuterStepLevel(inputRecord);
    GMX_RELEASE_ASSERT(outerLevel >= 0, "LAMMPS MTTK velocity scaling requires an outer r-RESPA level");
    return inputRecord.delta_t * exactRespaLevelStepFactor(inputRecord.exactRespa, outerLevel);
}

void applyLammpsMttkVelocityScale(const int                           homenr,
                                  const ArrayRef<const ParticleType> ptype,
                                  const ArrayRef<const RVec>         invMassPerDim,
                                  const real                         outerDt,
                                  const ExactRespaExtendedVvUpdate&  extendedUpdate,
                                  ArrayRef<RVec>                     velocity)
{
    if (!extendedUpdate.lammpsVelocityScale)
    {
        return;
    }

    // LAMMPS FixNH applies nh_v_press() once on each outer r-RESPA half step.
    // For isotropic boxes, its two quarter-step factors combine to
    // exp(-0.5*dt*(omega_dot + omega_dot/natoms)).
    const real scale = std::exp(-0.5_real * outerDt * extendedUpdate.veta * extendedUpdate.alpha);
    if (exactRespaCanUseDirectUpdatePath(homenr, ptype, invMassPerDim))
    {
        exactRespaUpdateForAtoms(homenr, [&](const int atom)
        {
            velocity[atom][XX] *= scale;
            velocity[atom][YY] *= scale;
            velocity[atom][ZZ] *= scale;
        });
        return;
    }

    exactRespaUpdateForAtoms(homenr, [&](const int atom)
    {
        if (ptype[atom] == ParticleType::Shell)
        {
            return;
        }
        for (int d = 0; d < DIM; ++d)
        {
            if (invMassPerDim[atom][d] != 0)
            {
                velocity[atom][d] *= scale;
            }
        }
    });
}

static inline real exactRespaVvPositionAfterDrift(const real currentPosition,
                                                  const real velocity,
                                                  const real dt,
                                                  const ExactRespaExtendedVvUpdate& extendedUpdate)
{
    if (!extendedUpdate.scalePositions && !extendedUpdate.lammpsRemapPositions)
    {
        return currentPosition + dt * velocity;
    }

    if (extendedUpdate.lammpsRemapPositions)
    {
        const real halfScale = std::exp(0.5_real * dt * extendedUpdate.veta);
        return halfScale * (halfScale * currentPosition + dt * velocity);
    }

    const real g   = 0.5_real * dt * extendedUpdate.veta;
    const real mr1 = std::exp(g);
    const real mr2 = gmx::series_sinhx(g);
    return mr1 * (mr1 * currentPosition + mr2 * dt * velocity);
}

void appendPreparedRespaHalfKick(const t_inputrec&           inputRecord,
                                 const int64_t               baseStep,
                                 const RespaKickPhase        phase,
                                 const int                   mtsLevel,
                                 const ExactRespaForceStore* exactRespaForceStore,
                                 const int*                  ddGlobalAtomIndices,
                                 const int                   ddGlobalAtomIndicesCount,
                                 const nonbonded_verlet_t*   nbv,
                                 ExactRespaPreparedHalfKicks* preparedHalfKicks)
{
    const auto forceForLevel = forceForExactRespaKickLevel(exactRespaForceStore, mtsLevel);
    appendExactRespaForceStoreTraceRows(
            baseStep,
            (phase == RespaKickPhase::Initial) ? "initial" : "final",
            mtsLevel,
            forceForLevel,
            ddGlobalAtomIndices,
            ddGlobalAtomIndicesCount,
            nbv);
    recordExactRespaRuntimeEventForTesting(
            baseStep, exactRespaRuntimeEventTypeFromKickPhase(phase), mtsLevel);

    GMX_RELEASE_ASSERT(preparedHalfKicks->numKicks < ExactRespaForceStore::c_numStoredLevels,
                       "Exact r-RESPA kick count should not exceed stored force levels");
    preparedHalfKicks->forcesPerKick[preparedHalfKicks->numKicks] = forceForLevel;
    preparedHalfKicks->dtPerKick[preparedHalfKicks->numKicks] =
            inputRecord.delta_t * exactRespaLevelStepFactor(inputRecord.exactRespa, mtsLevel);
    preparedHalfKicks->levelPerKick[preparedHalfKicks->numKicks] = mtsLevel;
    ++preparedHalfKicks->numKicks;
}

ExactRespaPreparedHalfKicks prepareRespaHalfKicks(const t_inputrec&                 inputRecord,
                                                  const int64_t                     baseStep,
                                                  const RespaKickPhase              phase,
                                                  const ExactRespaForceStore*       exactRespaForceStore,
                                                  const int*                        ddGlobalAtomIndices,
                                                  const int                         ddGlobalAtomIndicesCount,
                                                  const nonbonded_verlet_t*         nbv)
{
    ExactRespaPreparedHalfKicks preparedHalfKicks;
    if (!inputRecord.exactRespa.enabled())
    {
        appendPreparedRespaHalfKick(inputRecord,
                                    baseStep,
                                    phase,
                                    0,
                                    exactRespaForceStore,
                                    ddGlobalAtomIndices,
                                    ddGlobalAtomIndicesCount,
                                    nbv,
                                    &preparedHalfKicks);
        return preparedHalfKicks;
    }

    if (phase == RespaKickPhase::Initial)
    {
        const int highestInitialLevel = highestActiveExactRespaLevel(inputRecord.exactRespa, baseStep);
        for (int mtsLevel = highestInitialLevel; mtsLevel >= 0; --mtsLevel)
        {
            appendPreparedRespaHalfKick(inputRecord,
                                        baseStep,
                                        phase,
                                        mtsLevel,
                                        exactRespaForceStore,
                                        ddGlobalAtomIndices,
                                        ddGlobalAtomIndicesCount,
                                        nbv,
                                        &preparedHalfKicks);
        }
    }
    else
    {
        const int highestFinalLevel = highestActiveExactRespaLevel(inputRecord.exactRespa, baseStep + 1);
        for (int mtsLevel = 0; mtsLevel <= highestFinalLevel; ++mtsLevel)
        {
            appendPreparedRespaHalfKick(inputRecord,
                                        baseStep,
                                        phase,
                                        mtsLevel,
                                        exactRespaForceStore,
                                        ddGlobalAtomIndices,
                                        ddGlobalAtomIndicesCount,
                                        nbv,
                                        &preparedHalfKicks);
        }
    }

    return preparedHalfKicks;
}

void applyRespaVelocityHalfKick(const int                             homenr,
                                const ArrayRef<const ParticleType>   ptype,
                                const ArrayRef<const RVec>           invMassPerDim,
                                const ArrayRef<const RVec>&          force,
                                const real                           dt,
                                const ExactRespaExtendedVvUpdate&    extendedUpdate,
                                ArrayRef<RVec>                       velocity)
{
    if (exactRespaCanUseDirectUpdatePath(homenr, ptype, invMassPerDim))
    {
        exactRespaUpdateForAtoms(homenr, [&](const int atom)
        {
            velocity[atom][XX] = exactRespaVvVelocityAfterHalfKick(
                    velocity[atom][XX], invMassPerDim[atom][XX], force[atom][XX], dt, extendedUpdate);
            velocity[atom][YY] = exactRespaVvVelocityAfterHalfKick(
                    velocity[atom][YY], invMassPerDim[atom][YY], force[atom][YY], dt, extendedUpdate);
            velocity[atom][ZZ] = exactRespaVvVelocityAfterHalfKick(
                    velocity[atom][ZZ], invMassPerDim[atom][ZZ], force[atom][ZZ], dt, extendedUpdate);
        });
        return;
    }

    exactRespaUpdateForAtoms(homenr, [&](const int atom)
    {
        if (ptype[atom] == ParticleType::Shell)
        {
            return;
        }
        for (int d = 0; d < DIM; d++)
        {
            const real inverseMass = invMassPerDim[atom][d];
            if (inverseMass != 0)
            {
                velocity[atom][d] = exactRespaVvVelocityAfterHalfKick(
                        velocity[atom][d], inverseMass, force[atom][d], dt, extendedUpdate);
            }
        }
    });
}

void applyRespaVelocityHalfKicksFused(const int                             homenr,
                                      const ArrayRef<const ParticleType>   ptype,
                                      const ArrayRef<const RVec>           invMassPerDim,
                                      const std::array<ArrayRef<const RVec>, ExactRespaForceStore::c_numStoredLevels>&
                                                                            forcesPerKick,
                                      const std::array<real, ExactRespaForceStore::c_numStoredLevels>&
                                                                            dtPerKick,
                                     const int                             numKicks,
                                     const ExactRespaExtendedVvUpdate&     extendedUpdate,
                                     ArrayRef<RVec>                        velocity)
{
    GMX_RELEASE_ASSERT(numKicks >= 0 && numKicks <= ExactRespaForceStore::c_numStoredLevels,
                       "Exact r-RESPA fused kick count should be within the stored-level bound");
    if (numKicks == 0)
    {
        return;
    }

    if (numKicks == 1)
    {
        applyRespaVelocityHalfKick(
                homenr, ptype, invMassPerDim, forcesPerKick[0], dtPerKick[0], extendedUpdate, velocity);
        return;
    }

    if (extendedUpdate.scaleVelocities)
    {
        exactRespaUpdateForAtoms(homenr, [&](const int atom)
        {
            if (ptype[atom] == ParticleType::Shell)
            {
                return;
            }
            for (int d = 0; d < DIM; d++)
            {
                const real inverseMass = invMassPerDim[atom][d];
                if (inverseMass == 0)
                {
                    continue;
                }
                real updatedVelocity = velocity[atom][d];
                for (int kickIndex = 0; kickIndex < numKicks; ++kickIndex)
                {
                    updatedVelocity = exactRespaVvVelocityAfterHalfKick(updatedVelocity,
                                                                        inverseMass,
                                                                        forcesPerKick[kickIndex][atom][d],
                                                                        dtPerKick[kickIndex],
                                                                        extendedUpdate);
                }
                velocity[atom][d] = updatedVelocity;
            }
        });
        return;
    }

    if (numKicks == 2)
    {
        const auto force0 = forcesPerKick[0];
        const auto force1 = forcesPerKick[1];
        const real scale0 = 0.5_real * dtPerKick[0];
        const real scale1 = 0.5_real * dtPerKick[1];
        if (exactRespaCanUseDirectUpdatePath(homenr, ptype, invMassPerDim))
        {
            exactRespaUpdateForAtoms(homenr, [&](const int atom)
            {
                const real invMassX = invMassPerDim[atom][XX];
                const real invMassY = invMassPerDim[atom][YY];
                const real invMassZ = invMassPerDim[atom][ZZ];
                real       updatedVelocityX = velocity[atom][XX];
                real       updatedVelocityY = velocity[atom][YY];
                real       updatedVelocityZ = velocity[atom][ZZ];
                updatedVelocityX += scale0 * invMassX * force0[atom][XX];
                updatedVelocityX += scale1 * invMassX * force1[atom][XX];
                updatedVelocityY += scale0 * invMassY * force0[atom][YY];
                updatedVelocityY += scale1 * invMassY * force1[atom][YY];
                updatedVelocityZ += scale0 * invMassZ * force0[atom][ZZ];
                updatedVelocityZ += scale1 * invMassZ * force1[atom][ZZ];
                velocity[atom][XX] = updatedVelocityX;
                velocity[atom][YY] = updatedVelocityY;
                velocity[atom][ZZ] = updatedVelocityZ;
            });
            return;
        }
        exactRespaUpdateForAtoms(homenr, [&](const int atom)
        {
            if (ptype[atom] == ParticleType::Shell)
            {
                return;
            }
            for (int d = 0; d < DIM; d++)
            {
                const real inverseMass = invMassPerDim[atom][d];
                if (inverseMass == 0)
                {
                    continue;
                }
                velocity[atom][d] += scale0 * inverseMass * force0[atom][d];
                velocity[atom][d] += scale1 * inverseMass * force1[atom][d];
            }
        });
        return;
    }

    if (numKicks == 3)
    {
        const auto force0 = forcesPerKick[0];
        const auto force1 = forcesPerKick[1];
        const auto force2 = forcesPerKick[2];
        const real scale0 = 0.5_real * dtPerKick[0];
        const real scale1 = 0.5_real * dtPerKick[1];
        const real scale2 = 0.5_real * dtPerKick[2];
        if (exactRespaCanUseDirectUpdatePath(homenr, ptype, invMassPerDim))
        {
            exactRespaUpdateForAtoms(homenr, [&](const int atom)
            {
                const real invMassX = invMassPerDim[atom][XX];
                const real invMassY = invMassPerDim[atom][YY];
                const real invMassZ = invMassPerDim[atom][ZZ];
                real       updatedVelocityX = velocity[atom][XX];
                real       updatedVelocityY = velocity[atom][YY];
                real       updatedVelocityZ = velocity[atom][ZZ];
                updatedVelocityX += scale0 * invMassX * force0[atom][XX];
                updatedVelocityX += scale1 * invMassX * force1[atom][XX];
                updatedVelocityX += scale2 * invMassX * force2[atom][XX];
                updatedVelocityY += scale0 * invMassY * force0[atom][YY];
                updatedVelocityY += scale1 * invMassY * force1[atom][YY];
                updatedVelocityY += scale2 * invMassY * force2[atom][YY];
                updatedVelocityZ += scale0 * invMassZ * force0[atom][ZZ];
                updatedVelocityZ += scale1 * invMassZ * force1[atom][ZZ];
                updatedVelocityZ += scale2 * invMassZ * force2[atom][ZZ];
                velocity[atom][XX] = updatedVelocityX;
                velocity[atom][YY] = updatedVelocityY;
                velocity[atom][ZZ] = updatedVelocityZ;
            });
            return;
        }
        exactRespaUpdateForAtoms(homenr, [&](const int atom)
        {
            if (ptype[atom] == ParticleType::Shell)
            {
                return;
            }
            for (int d = 0; d < DIM; d++)
            {
                const real inverseMass = invMassPerDim[atom][d];
                if (inverseMass == 0)
                {
                    continue;
                }
                velocity[atom][d] += scale0 * inverseMass * force0[atom][d];
                velocity[atom][d] += scale1 * inverseMass * force1[atom][d];
                velocity[atom][d] += scale2 * inverseMass * force2[atom][d];
            }
        });
        return;
    }

    exactRespaUpdateForAtoms(homenr, [&](const int atom)
    {
        if (ptype[atom] == ParticleType::Shell)
        {
            return;
        }

        for (int d = 0; d < DIM; d++)
        {
            const real inverseMass = invMassPerDim[atom][d];
            if (inverseMass == 0)
            {
                continue;
            }

            for (int kickIndex = 0; kickIndex < numKicks; ++kickIndex)
            {
                velocity[atom][d] += 0.5_real * dtPerKick[kickIndex] * inverseMass
                                     * forcesPerKick[kickIndex][atom][d];
            }
        }
    });
}

void applyPreparedRespaHalfKicks(const int                             homenr,
                                 const ArrayRef<const ParticleType>   ptype,
                                 const ArrayRef<const RVec>           invMassPerDim,
                                 const ExactRespaPreparedHalfKicks&   preparedHalfKicks,
                                 const ExactRespaExtendedVvUpdate&    extendedUpdate,
                                 ArrayRef<RVec>                       velocity)
{
    if (preparedHalfKicks.numKicks == 1)
    {
        applyRespaVelocityHalfKick(homenr,
                                   ptype,
                                   invMassPerDim,
                                   preparedHalfKicks.forcesPerKick[0],
                                   preparedHalfKicks.dtPerKick[0],
                                   extendedUpdate,
                                   velocity);
    }
    else if (preparedHalfKicks.numKicks > 1)
    {
        applyRespaVelocityHalfKicksFused(homenr,
                                         ptype,
                                         invMassPerDim,
                                         preparedHalfKicks.forcesPerKick,
                                         preparedHalfKicks.dtPerKick,
                                         preparedHalfKicks.numKicks,
                                         extendedUpdate,
                                         velocity);
    }
}

void driftRespaPositions(const int                             homenr,
                         const ArrayRef<const ParticleType>   ptype,
                         const ArrayRef<const RVec>           invMassPerDim,
                         const real                           dt,
                         const ExactRespaExtendedVvUpdate&    extendedUpdate,
                         ArrayRef<RVec>                       position,
                         ArrayRef<const RVec>                 velocity)
{
    if (exactRespaCanUseDirectUpdatePath(homenr, ptype, invMassPerDim))
    {
        exactRespaUpdateForAtoms(homenr, [&](const int atom)
        {
            position[atom][XX] = exactRespaVvPositionAfterDrift(
                    position[atom][XX], velocity[atom][XX], dt, extendedUpdate);
            position[atom][YY] = exactRespaVvPositionAfterDrift(
                    position[atom][YY], velocity[atom][YY], dt, extendedUpdate);
            position[atom][ZZ] = exactRespaVvPositionAfterDrift(
                    position[atom][ZZ], velocity[atom][ZZ], dt, extendedUpdate);
        });
        return;
    }

    exactRespaUpdateForAtoms(homenr, [&](const int atom)
    {
        if (ptype[atom] == ParticleType::Shell)
        {
            return;
        }
        for (int d = 0; d < DIM; d++)
        {
            if (invMassPerDim[atom][d] != 0)
            {
                position[atom][d] = exactRespaVvPositionAfterDrift(
                        position[atom][d], velocity[atom][d], dt, extendedUpdate);
            }
        }
    });
}

void applyPreparedRespaInitialHalfKicksAndDrift(
        const t_inputrec&                  inputRecord,
        const int                           homenr,
        const ArrayRef<const ParticleType> ptype,
        const ArrayRef<const RVec>         invMassPerDim,
        const ExactRespaPreparedHalfKicks& preparedHalfKicks,
        const real                         driftDt,
        const ExactRespaExtendedVvUpdate&  extendedUpdate,
        ArrayRef<RVec>                     position,
        ArrayRef<RVec>                     velocity)
{
    GMX_RELEASE_ASSERT(preparedHalfKicks.numKicks >= 0
                               && preparedHalfKicks.numKicks
                                          <= ExactRespaForceStore::c_numStoredLevels,
                       "Exact r-RESPA fused initial kick count should be within the stored-level bound");
    if (preparedHalfKicks.numKicks == 0)
    {
        driftRespaPositions(homenr, ptype, invMassPerDim, driftDt, extendedUpdate, position, velocity);
        return;
    }

    if (extendedUpdate.lammpsVelocityScale)
    {
        if (exactRespaPreparedKicksIncludeMttkOuterStep(inputRecord, preparedHalfKicks))
        {
            applyLammpsMttkVelocityScale(homenr,
                                         ptype,
                                         invMassPerDim,
                                         exactRespaOuterLevelDt(inputRecord),
                                         extendedUpdate,
                                         velocity);
        }
        applyPreparedRespaHalfKicks(homenr,
                                    ptype,
                                    invMassPerDim,
                                    preparedHalfKicks,
                                    exactRespaPlainKickUpdate(extendedUpdate),
                                    velocity);
        driftRespaPositions(homenr, ptype, invMassPerDim, driftDt, extendedUpdate, position, velocity);
        return;
    }

    if (extendedUpdate.scaleVelocities || extendedUpdate.scalePositions)
    {
        exactRespaUpdateForAtoms(homenr, [&](const int atom)
        {
            if (ptype[atom] == ParticleType::Shell)
            {
                return;
            }
            for (int d = 0; d < DIM; d++)
            {
                const real inverseMass = invMassPerDim[atom][d];
                if (inverseMass == 0)
                {
                    continue;
                }

                real updatedVelocity = velocity[atom][d];
                for (int kickIndex = 0; kickIndex < preparedHalfKicks.numKicks; ++kickIndex)
                {
                    updatedVelocity =
                            exactRespaVvVelocityAfterHalfKick(updatedVelocity,
                                                              inverseMass,
                                                              preparedHalfKicks.forcesPerKick[kickIndex][atom][d],
                                                              preparedHalfKicks.dtPerKick[kickIndex],
                                                              extendedUpdate);
                }
                velocity[atom][d] = updatedVelocity;
                position[atom][d] = exactRespaVvPositionAfterDrift(
                        position[atom][d], updatedVelocity, driftDt, extendedUpdate);
            }
        });
        return;
    }

    const auto updateAtomWithGenericKickCount = [&](const int atom)
    {
        if (ptype[atom] == ParticleType::Shell)
        {
            return;
        }
        for (int d = 0; d < DIM; d++)
        {
            const real inverseMass = invMassPerDim[atom][d];
            if (inverseMass == 0)
            {
                continue;
            }

            real updatedVelocity = velocity[atom][d];
            for (int kickIndex = 0; kickIndex < preparedHalfKicks.numKicks; ++kickIndex)
            {
                updatedVelocity += 0.5_real * preparedHalfKicks.dtPerKick[kickIndex] * inverseMass
                                   * preparedHalfKicks.forcesPerKick[kickIndex][atom][d];
            }
            velocity[atom][d] = updatedVelocity;
            position[atom][d] += driftDt * updatedVelocity;
        }
    };

    if (preparedHalfKicks.numKicks == 1)
    {
        const auto force0 = preparedHalfKicks.forcesPerKick[0];
        const real scale0 = 0.5_real * preparedHalfKicks.dtPerKick[0];
        if (exactRespaCanUseDirectUpdatePath(homenr, ptype, invMassPerDim))
        {
            if (exactRespaFusedUpdateVectorEnabled())
            {
                exactRespaFusedPlainKickDrift<1>(homenr,
                                                 invMassPerDim.data(),
                                                 force0.data(),
                                                 nullptr,
                                                 nullptr,
                                                 scale0,
                                                 0,
                                                 0,
                                                 driftDt,
                                                 position.data(),
                                                 velocity.data());
            }
            else
            {
                exactRespaUpdateForAtoms(homenr, [&](const int atom)
                {
                    const real updatedVelocityX =
                            velocity[atom][XX] + scale0 * invMassPerDim[atom][XX] * force0[atom][XX];
                    const real updatedVelocityY =
                            velocity[atom][YY] + scale0 * invMassPerDim[atom][YY] * force0[atom][YY];
                    const real updatedVelocityZ =
                            velocity[atom][ZZ] + scale0 * invMassPerDim[atom][ZZ] * force0[atom][ZZ];
                    velocity[atom][XX] = updatedVelocityX;
                    velocity[atom][YY] = updatedVelocityY;
                    velocity[atom][ZZ] = updatedVelocityZ;
                    position[atom][XX] += driftDt * updatedVelocityX;
                    position[atom][YY] += driftDt * updatedVelocityY;
                    position[atom][ZZ] += driftDt * updatedVelocityZ;
                });
            }
            return;
        }
        exactRespaUpdateForAtoms(homenr, [&](const int atom)
        {
            if (ptype[atom] == ParticleType::Shell)
            {
                return;
            }
            for (int d = 0; d < DIM; d++)
            {
                const real inverseMass = invMassPerDim[atom][d];
                if (inverseMass == 0)
                {
                    continue;
                }
                const real updatedVelocity = velocity[atom][d] + scale0 * inverseMass * force0[atom][d];
                velocity[atom][d] = updatedVelocity;
                position[atom][d] += driftDt * updatedVelocity;
            }
        });
        return;
    }

    if (preparedHalfKicks.numKicks == 2)
    {
        const auto force0 = preparedHalfKicks.forcesPerKick[0];
        const auto force1 = preparedHalfKicks.forcesPerKick[1];
        const real scale0 = 0.5_real * preparedHalfKicks.dtPerKick[0];
        const real scale1 = 0.5_real * preparedHalfKicks.dtPerKick[1];
        if (exactRespaCanUseDirectUpdatePath(homenr, ptype, invMassPerDim))
        {
            if (exactRespaFusedUpdateVectorEnabled())
            {
                exactRespaFusedPlainKickDrift<2>(homenr,
                                                 invMassPerDim.data(),
                                                 force0.data(),
                                                 force1.data(),
                                                 nullptr,
                                                 scale0,
                                                 scale1,
                                                 0,
                                                 driftDt,
                                                 position.data(),
                                                 velocity.data());
            }
            else
            {
                exactRespaUpdateForAtoms(homenr, [&](const int atom)
                {
                    const real invMassX = invMassPerDim[atom][XX];
                    const real invMassY = invMassPerDim[atom][YY];
                    const real invMassZ = invMassPerDim[atom][ZZ];
                    real       updatedVelocityX = velocity[atom][XX];
                    real       updatedVelocityY = velocity[atom][YY];
                    real       updatedVelocityZ = velocity[atom][ZZ];
                    updatedVelocityX += scale0 * invMassX * force0[atom][XX];
                    updatedVelocityX += scale1 * invMassX * force1[atom][XX];
                    updatedVelocityY += scale0 * invMassY * force0[atom][YY];
                    updatedVelocityY += scale1 * invMassY * force1[atom][YY];
                    updatedVelocityZ += scale0 * invMassZ * force0[atom][ZZ];
                    updatedVelocityZ += scale1 * invMassZ * force1[atom][ZZ];
                    velocity[atom][XX] = updatedVelocityX;
                    velocity[atom][YY] = updatedVelocityY;
                    velocity[atom][ZZ] = updatedVelocityZ;
                    position[atom][XX] += driftDt * updatedVelocityX;
                    position[atom][YY] += driftDt * updatedVelocityY;
                    position[atom][ZZ] += driftDt * updatedVelocityZ;
                });
            }
            return;
        }
        exactRespaUpdateForAtoms(homenr, [&](const int atom)
        {
            if (ptype[atom] == ParticleType::Shell)
            {
                return;
            }
            for (int d = 0; d < DIM; d++)
            {
                const real inverseMass = invMassPerDim[atom][d];
                if (inverseMass == 0)
                {
                    continue;
                }
                real updatedVelocity = velocity[atom][d];
                updatedVelocity += scale0 * inverseMass * force0[atom][d];
                updatedVelocity += scale1 * inverseMass * force1[atom][d];
                velocity[atom][d] = updatedVelocity;
                position[atom][d] += driftDt * updatedVelocity;
            }
        });
        return;
    }

    if (preparedHalfKicks.numKicks == 3)
    {
        const auto force0 = preparedHalfKicks.forcesPerKick[0];
        const auto force1 = preparedHalfKicks.forcesPerKick[1];
        const auto force2 = preparedHalfKicks.forcesPerKick[2];
        const real scale0 = 0.5_real * preparedHalfKicks.dtPerKick[0];
        const real scale1 = 0.5_real * preparedHalfKicks.dtPerKick[1];
        const real scale2 = 0.5_real * preparedHalfKicks.dtPerKick[2];
        if (exactRespaCanUseDirectUpdatePath(homenr, ptype, invMassPerDim))
        {
            if (exactRespaFusedUpdateVectorEnabled())
            {
                exactRespaFusedPlainKickDrift<3>(homenr,
                                                 invMassPerDim.data(),
                                                 force0.data(),
                                                 force1.data(),
                                                 force2.data(),
                                                 scale0,
                                                 scale1,
                                                 scale2,
                                                 driftDt,
                                                 position.data(),
                                                 velocity.data());
            }
            else
            {
                exactRespaUpdateForAtoms(homenr, [&](const int atom)
                {
                    const real invMassX = invMassPerDim[atom][XX];
                    const real invMassY = invMassPerDim[atom][YY];
                    const real invMassZ = invMassPerDim[atom][ZZ];
                    real       updatedVelocityX = velocity[atom][XX];
                    real       updatedVelocityY = velocity[atom][YY];
                    real       updatedVelocityZ = velocity[atom][ZZ];
                    updatedVelocityX += scale0 * invMassX * force0[atom][XX];
                    updatedVelocityX += scale1 * invMassX * force1[atom][XX];
                    updatedVelocityX += scale2 * invMassX * force2[atom][XX];
                    updatedVelocityY += scale0 * invMassY * force0[atom][YY];
                    updatedVelocityY += scale1 * invMassY * force1[atom][YY];
                    updatedVelocityY += scale2 * invMassY * force2[atom][YY];
                    updatedVelocityZ += scale0 * invMassZ * force0[atom][ZZ];
                    updatedVelocityZ += scale1 * invMassZ * force1[atom][ZZ];
                    updatedVelocityZ += scale2 * invMassZ * force2[atom][ZZ];
                    velocity[atom][XX] = updatedVelocityX;
                    velocity[atom][YY] = updatedVelocityY;
                    velocity[atom][ZZ] = updatedVelocityZ;
                    position[atom][XX] += driftDt * updatedVelocityX;
                    position[atom][YY] += driftDt * updatedVelocityY;
                    position[atom][ZZ] += driftDt * updatedVelocityZ;
                });
            }
            return;
        }
        exactRespaUpdateForAtoms(homenr, [&](const int atom)
        {
            if (ptype[atom] == ParticleType::Shell)
            {
                return;
            }
            for (int d = 0; d < DIM; d++)
            {
                const real inverseMass = invMassPerDim[atom][d];
                if (inverseMass == 0)
                {
                    continue;
                }
                real updatedVelocity = velocity[atom][d];
                updatedVelocity += scale0 * inverseMass * force0[atom][d];
                updatedVelocity += scale1 * inverseMass * force1[atom][d];
                updatedVelocity += scale2 * inverseMass * force2[atom][d];
                velocity[atom][d] = updatedVelocity;
                position[atom][d] += driftDt * updatedVelocity;
            }
        });
        return;
    }

    if (exactRespaCanUseDirectUpdatePath(homenr, ptype, invMassPerDim))
    {
        exactRespaUpdateForAtoms(homenr, [&](const int atom)
        {
            for (int d = 0; d < DIM; d++)
            {
                const real inverseMass = invMassPerDim[atom][d];
                real       updatedVelocity = velocity[atom][d];
                for (int kickIndex = 0; kickIndex < preparedHalfKicks.numKicks; ++kickIndex)
                {
                    updatedVelocity += 0.5_real * preparedHalfKicks.dtPerKick[kickIndex] * inverseMass
                                       * preparedHalfKicks.forcesPerKick[kickIndex][atom][d];
                }
                velocity[atom][d] = updatedVelocity;
                position[atom][d] += driftDt * updatedVelocity;
            }
        });
        return;
    }

    exactRespaUpdateForAtoms(homenr, updateAtomWithGenericKickCount);
}

void driftRespaPositionsOnGpu(StatePropagatorDataGpu* stateGpu, UpdateConstrainGpu* gpuUpdater, const real dt)
{
    GMX_RELEASE_ASSERT(stateGpu != nullptr, "Exact r-RESPA GPU update requires GPU state.");
    GMX_RELEASE_ASSERT(gpuUpdater != nullptr, "Exact r-RESPA GPU update requires a GPU updater.");
    GMX_UNUSED_VALUE(stateGpu);
    gpuUpdater->driftOnly(dt);
}

void applyRespaHalfKicks(const t_inputrec&                 inputRecord,
                         const int64_t                     baseStep,
                         const RespaKickPhase              phase,
                         const int                         homenr,
                         const ArrayRef<const ParticleType> ptype,
                         const ArrayRef<const RVec>        invMassPerDim,
                         const ExactRespaForceStore*       exactRespaForceStore,
                         const int*                        ddGlobalAtomIndices,
                         const int                         ddGlobalAtomIndicesCount,
                         const nonbonded_verlet_t*         nbv,
                         ForceBuffers&                     forceBuffers,
                         const ExactRespaExtendedVvUpdate& extendedUpdate,
                         ArrayRef<RVec>                    velocity)
{
    const ExactRespaPreparedHalfKicks preparedHalfKicks = prepareRespaHalfKicks(
            inputRecord, baseStep, phase, exactRespaForceStore, ddGlobalAtomIndices, ddGlobalAtomIndicesCount, nbv);
    GMX_UNUSED_VALUE(forceBuffers);

    if (extendedUpdate.lammpsVelocityScale)
    {
        const bool includesOuterLevel =
                exactRespaPreparedKicksIncludeMttkOuterStep(inputRecord, preparedHalfKicks);
        if (phase == RespaKickPhase::Initial && includesOuterLevel)
        {
            applyLammpsMttkVelocityScale(homenr,
                                         ptype,
                                         invMassPerDim,
                                         exactRespaOuterLevelDt(inputRecord),
                                         extendedUpdate,
                                         velocity);
        }
        applyPreparedRespaHalfKicks(homenr,
                                    ptype,
                                    invMassPerDim,
                                    preparedHalfKicks,
                                    exactRespaPlainKickUpdate(extendedUpdate),
                                    velocity);
        if (phase == RespaKickPhase::Final && includesOuterLevel)
        {
            applyLammpsMttkVelocityScale(homenr,
                                         ptype,
                                         invMassPerDim,
                                         exactRespaOuterLevelDt(inputRecord),
                                         extendedUpdate,
                                         velocity);
        }
        return;
    }

    applyPreparedRespaHalfKicks(homenr, ptype, invMassPerDim, preparedHalfKicks, extendedUpdate, velocity);
}

void applySoftStartPreparedRespaHalfKicks(
        const t_inputrec&                  inputRecord,
        const int64_t                      baseStep,
        const RespaKickPhase               phase,
        const int                           homenr,
        const ArrayRef<const ParticleType> ptype,
        const ArrayRef<const RVec>         invMassPerDim,
        const ExactRespaPreparedHalfKicks& preparedHalfKicks,
        const t_mdatoms&                   mdatoms,
        const int*                         globalAtomIndices,
        const int                          globalAtomIndicesCount,
        const MpiComm&                     mpiComm,
        ExactRespaSoftStartState*          softStartState,
        ArrayRef<RVec>                     velocity)
{
    GMX_RELEASE_ASSERT(softStartState != nullptr && softStartState->config.enabled,
                       "Soft-start half-kicks require enabled soft-start state");
    GMX_RELEASE_ASSERT(velocity.ssize() >= homenr && ptype.ssize() >= homenr
                               && invMassPerDim.ssize() >= homenr,
                       "Soft-start half-kicks require all home-atom state");

    for (int kickIndex = 0; kickIndex < preparedHalfKicks.numKicks; ++kickIndex)
    {
        const int  level       = preparedHalfKicks.levelPerKick[kickIndex];
        const auto force       = preparedHalfKicks.forcesPerKick[kickIndex];
        const bool isOuterKick = level == softStartState->outerLevel;
        if (isOuterKick)
        {
            const int64_t boundaryStep =
                    (phase == RespaKickPhase::Initial) ? baseStep : baseStep + 1;
            GMX_RELEASE_ASSERT(
                    exactRespaSoftStartIsOuterBoundary(boundaryStep,
                                                       inputRecord.init_step,
                                                       softStartState->outerStepFactor),
                    "Soft-start slow half-kick must occur on a slowest-level boundary");
            const int64_t expectedBoundary =
                    (boundaryStep - inputRecord.init_step) / softStartState->outerStepFactor;
            if (phase == RespaKickPhase::Final)
            {
                // LAMMPS's recursive r-RESPA applies lower-level final kicks
                // before post_force_respa() evaluates fix langevin at the
                // slowest level. Refresh here, not immediately after do_force,
                // so the drag term sees the same boundary velocity.
                refreshExactRespaSoftStartLangevinForce(softStartState,
                                                         expectedBoundary,
                                                         mdatoms,
                                                         velocity,
                                                         globalAtomIndices,
                                                         globalAtomIndicesCount,
                                                         mpiComm);
            }
            GMX_RELEASE_ASSERT(softStartState->cachedBoundary == expectedBoundary,
                               "Soft-start slow half-kicks must reuse the Langevin force from their boundary refresh");
            GMX_RELEASE_ASSERT(softStartState->cachedLangevinForce.size()
                                       >= static_cast<size_t>(homenr),
                               "Soft-start slow half-kick requires a cached Langevin force per home atom");
        }

        exactRespaUpdateForAtoms(homenr, [&](const int atom)
        {
            if (ptype[atom] == ParticleType::Shell
                || (invMassPerDim[atom][XX] == 0 && invMassPerDim[atom][YY] == 0
                    && invMassPerDim[atom][ZZ] == 0))
            {
                return;
            }
            const RVec* langevinForce =
                    isOuterKick ? &softStartState->cachedLangevinForce[atom] : nullptr;
            applyExactRespaSoftStartHalfKick(preparedHalfKicks.dtPerKick[kickIndex],
                                             invMassPerDim[atom],
                                             force[atom],
                                             langevinForce,
                                             softStartState->maximumSpeedNmPerPs,
                                             &velocity[atom]);
        });
    }
}

} // namespace

void prepareExactRespaVelocityVerletObservables(
        const ExactRespaVelocityVerletObservablesContext& context)
{
    GMX_RELEASE_ASSERT(context.fr != nullptr, "Exact r-RESPA VV observables require a force record");
    GMX_RELEASE_ASSERT(context.state != nullptr, "Exact r-RESPA VV observables require state");
    GMX_RELEASE_ASSERT(context.mdatoms != nullptr, "Exact r-RESPA VV observables require mdatoms");
    GMX_RELEASE_ASSERT(context.nrnb != nullptr, "Exact r-RESPA VV observables require nrnb");
    GMX_RELEASE_ASSERT(context.vcm != nullptr, "Exact r-RESPA VV observables require vcm");
    GMX_RELEASE_ASSERT(context.enerd != nullptr, "Exact r-RESPA VV observables require energy data");
    GMX_RELEASE_ASSERT(context.ekind != nullptr, "Exact r-RESPA VV observables require kinetic data");
    GMX_RELEASE_ASSERT(context.gstat != nullptr, "Exact r-RESPA VV observables require global stats");
    GMX_RELEASE_ASSERT(context.nullSignaller != nullptr,
                       "Exact r-RESPA VV observables require a signaller");
    GMX_RELEASE_ASSERT(context.observablesReducer != nullptr,
                       "Exact r-RESPA VV observables require an observables reducer");
    GMX_RELEASE_ASSERT(context.sumEkinhOld != nullptr,
                       "Exact r-RESPA VV observables require bSumEkinhOld storage");
    GMX_RELEASE_ASSERT(context.savedConservedQuantity != nullptr,
                       "Exact r-RESPA VV observables require conserved quantity storage");
    GMX_RELEASE_ASSERT(context.lastEkin != nullptr,
                       "Exact r-RESPA VV observables require kinetic-energy storage");

    int cgloFlags = (context.calcGlobalStats ? CGLO_GSTAT : 0) | CGLO_TEMPERATURE | CGLO_SCALEEKIN;
    if (context.calcEner)
    {
        cgloFlags |= CGLO_ENERGY;
    }
    if (context.calcVir)
    {
        cgloFlags |= CGLO_PRESSURE;
    }

    compute_globals(context.gstat,
                    context.mpiComm,
                    &context.inputRecord,
                    context.fr,
                    context.ekind,
                    makeConstArrayRef(context.state->x),
                    makeConstArrayRef(context.state->v),
                    context.state->box,
                    context.mdatoms,
                    context.nrnb,
                    context.vcm,
                    context.wallCycle,
                    context.enerd,
                    context.forceVir,
                    context.shakeVir,
                    context.totalVir,
                    context.pres,
                    context.nullSignaller,
                    context.state->box,
                    context.sumEkinhOld,
                    cgloFlags,
                    context.step,
                    context.observablesReducer);

    *context.savedConservedQuantity = 0;
    *context.lastEkin               = context.enerd->term[InteractionFunction::KineticEnergy];
}

void LegacySimulator::prepareExactRespaVelocityVerletObservablesForStep(const t_inputrec& inputRecord,
                                                                        const int64_t     step,
                                                                        const MpiComm&    mpiComm,
                                                                        const t_mdatoms&  mdatoms,
                                                                        t_nrnb*           nrnb,
                                                                        t_vcm*            vcm,
                                                                        gmx_enerdata_t*   enerd,
                                                                        gmx_global_stat*  gstat,
                                                                        SimulationSignaller* nullSignaller,
                                                                        ObservablesReducer*  observablesReducer,
                                                                        tensor&           forceVir,
                                                                        tensor&           shakeVir,
                                                                        tensor&           totalVir,
                                                                        tensor&           pres,
                                                                        const bool        calcEner,
                                                                        const bool        calcVir,
                                                                        const bool        calcGlobalStats,
                                                                        gmx_bool*         sumEkinhOld,
                                                                        real*             savedConservedQuantity,
                                                                        real*             lastEkin)
{
    const ExactRespaVelocityVerletObservablesContext observablesContext{
            inputRecord,
            step,
            mpiComm,
            fr_,
            state_,
            &mdatoms,
            nrnb,
            vcm,
            wallCycleCounters_,
            enerd,
            ekind_,
            gstat,
            nullSignaller,
            observablesReducer,
            forceVir,
            shakeVir,
            totalVir,
            pres,
            calcEner,
            calcVir,
            calcGlobalStats,
            sumEkinhOld,
            savedConservedQuantity,
            lastEkin };
    prepareExactRespaVelocityVerletObservables(observablesContext);
}

void LegacySimulator::dispatchExactRespaVelocityVerletStep(const t_inputrec&              inputRecord,
                                                           const int64_t                  step,
                                                           const double                   time,
                                                           const t_mdatoms&              mdatoms,
                                                           const SimulationWorkload&     simulationWork,
                                                           const DomainLifetimeWorkload& domainWork,
                                                           ForceBuffers&                 forceBuffers,
                                                           ExactRespaForceStore&         exactRespaForceStore,
                                                           tensor&                       forceVir,
                                                           rvec&                         muTot,
                                                           gmx_enerdata_t&               enerd,
                                                           Awh*                          awh,
                                                           gmx_edsam*                    ed,
                                                           const DDBalanceRegionHandler& ddBalanceRegionHandler)
{
    const ExactRespaStepContext exactRespaStep{
            step,
            time,
            inputRecord,
            mdatoms,
            simulationWork,
            domainWork,
            forceBuffers,
            exactRespaForceStore,
            forceVir,
            muTot,
            enerd,
            awh,
            ed,
            ddBalanceRegionHandler };
    doExactRespaVelocityVerletStep(exactRespaStep);
    if (!simulationWork.useGpuUpdate)
    {
        wallcycle_stop(wallCycleCounters_, WallCycleCounter::Update);
    }
}

void LegacySimulator::dispatchExactRespaNestedPrototypeStep(const t_inputrec&              inputRecord,
                                                            const int64_t                  step,
                                                            const double                   time,
                                                            const t_mdatoms&              mdatoms,
                                                            const SimulationWorkload&     simulationWork,
                                                            const DomainLifetimeWorkload& domainWork,
                                                            ForceBuffers&                 forceBuffers,
                                                            ExactRespaForceStore&         exactRespaForceStore,
                                                            tensor&                       forceVir,
                                                            rvec&                         muTot,
                                                            gmx_enerdata_t&               enerd,
                                                            Awh*                          awh,
                                                            gmx_edsam*                    ed,
                                                            const DDBalanceRegionHandler& ddBalanceRegionHandler)
{
    const ExactRespaStepContext exactRespaStep{
            step,
            time,
            inputRecord,
            mdatoms,
            simulationWork,
            domainWork,
            forceBuffers,
            exactRespaForceStore,
            forceVir,
            muTot,
            enerd,
            awh,
            ed,
            ddBalanceRegionHandler };
    doExactRespaNestedPrototypeStep(exactRespaStep);
    if (!simulationWork.useGpuUpdate)
    {
        wallcycle_stop(wallCycleCounters_, WallCycleCounter::Update);
    }
}

void LegacySimulator::doExactRespaVelocityVerletStep(const ExactRespaStepContext& exactRespaStep)
{
    const t_inputrec& inputRecord = exactRespaStep.inputRecord;
    const bool        useGpuUpdate = exactRespaStep.simulationWork.useGpuUpdate;
    const bool        trackExactImages = exactRespaImageTrackerEnabled();
    int64_t           imageSidecarFinalStep = 0;
    if (trackExactImages)
    {
        GMX_RELEASE_ASSERT(!haveDDAtomOrdering(*cr_),
                           "Exact r-RESPA image sidecars require fixed no-DD global atom order");
        GMX_RELEASE_ASSERT(fr_->pbcType != PbcType::No,
                           "Exact r-RESPA image sidecars require periodic boundaries");
        GMX_RELEASE_ASSERT(inputRecord.nsteps >= 0,
                           "Exact r-RESPA image sidecars require a finite stage");
        GMX_RELEASE_ASSERT(inputRecord.init_step
                                   <= std::numeric_limits<int64_t>::max() - inputRecord.nsteps,
                           "Exact r-RESPA image sidecar final step overflow");
        imageSidecarFinalStep = inputRecord.init_step + inputRecord.nsteps;
        ensureExactRespaImageTrackerInitialized(
                exactRespaStep.step,
                state_->box,
                state_->x.arrayRefWithPadding().unpaddedConstArrayRef().subArray(
                        0, exactRespaStep.mdatoms.homenr));
    }
    const ExactRespaExtendedVvUpdate extendedUpdate =
            exactRespaExtendedVvUpdateFromState(inputRecord, *state_);
    const bool inlineMttkBoxRemap =
            extendedUpdate.inlineBoxRemap && exactRespaMttkInlineBoxRemapEnabled(inputRecord);
    GMX_RELEASE_ASSERT(!useGpuUpdate
                               || (!extendedUpdate.scaleVelocities && !extendedUpdate.scalePositions
                                   && !extendedUpdate.lammpsVelocityScale
                                   && !extendedUpdate.lammpsRemapPositions),
                       "Exact r-RESPA MTTK propagation currently requires CPU update.");
    const bool        traceState = shouldRecordExactRespaStateTrace(exactRespaStep.step);
    const bool        tracePositions = exactRespaStateTraceConfig().includePositions;
#if GMX_GPU_CUDA
    const bool useGpuDeviceKicks = useGpuUpdate && exactRespaGpuDeviceKickProbeEnabled() && !traceState;
#else
    const bool useGpuDeviceKicks = false;
    GMX_RELEASE_ASSERT(!useGpuUpdate || !exactRespaGpuDeviceKickProbeEnabled(),
                       "Exact r-RESPA direct GPU kicks currently require CUDA");
#endif
    bool              copiedCoordinatesFromGpuAfterDrift = false;
    const int*        ddGlobalAtomIndices =
            haveDDAtomOrdering(*cr_) ? cr_->dd->globalAtomIndices.data() : nullptr;
    const int         ddGlobalAtomIndicesCount =
            haveDDAtomOrdering(*cr_) ? static_cast<int>(cr_->dd->globalAtomIndices.size()) : 0;
    initializeExactRespaSoftStartState(
            inputRecord, useGpuUpdate, exactRespaStep.step, &exactRespaSoftStartState_);
    const bool softStartEnabled = exactRespaSoftStartState_.config.enabled;
    if (softStartEnabled && exactRespaSoftStartState_.cachedLangevinForce.empty())
    {
        refreshExactRespaSoftStartLangevinForce(
                &exactRespaSoftStartState_,
                0,
                exactRespaStep.mdatoms,
                state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                ddGlobalAtomIndices,
                ddGlobalAtomIndicesCount,
                cr_->commMyGroup);
        if (cr_->isSimulationMainRank())
        {
            FILE* report = (fpLog_ != nullptr) ? fpLog_ : stderr;
            std::fprintf(report,
                         "Exact r-RESPA LAMMPS-style soft-start enabled: xlimit=%g nm, "
                         "outer-dt=%g ps, vmax=%g nm/ps, T=%g K, damp=%g ps, "
                         "seed=%llu, zero-random=%s, update=CPU\n",
                         static_cast<double>(exactRespaSoftStartState_.config.xlimitNm),
                         static_cast<double>(exactRespaSoftStartState_.outerDtPs),
                         static_cast<double>(exactRespaSoftStartState_.maximumSpeedNmPerPs),
                         static_cast<double>(exactRespaSoftStartState_.config.temperatureK),
                         static_cast<double>(exactRespaSoftStartState_.config.dampingTimePs),
                         static_cast<unsigned long long>(exactRespaSoftStartState_.config.seed),
                         exactRespaSoftStartState_.config.zeroRandomForce ? "yes" : "no");
        }
    }
    const int64_t nextStep         = exactRespaStep.step + 1;
    const bool    nextStepIsNsStep = (inputRecord.nstlist > 0 && nextStep % inputRecord.nstlist == 0);
    const bool    nextStepNeedsVirialForMttkPost =
            exactRespaMttkPostTrotterNeedsNextVirial(inputRecord, exactRespaStep.step);
    const int nextLegacyForceFlags =
            GMX_FORCE_STATECHANGED | GMX_FORCE_ALLFORCES | (nextStepIsNsStep ? GMX_FORCE_NS : 0)
            | (nextStepNeedsVirialForMttkPost ? GMX_FORCE_VIRIAL : 0);

    MdrunScheduleWorkload nextRunSchedule = *runScheduleWork_;
    nextRunSchedule.stepWork = setupExactRespaStepWorkload(nextLegacyForceFlags,
                                                           inputRecord,
                                                           nextStep,
                                                           nextRunSchedule.domainWork,
                                                           nextRunSchedule.simulationWork);
    nextRunSchedule.exactRespaStepWork = setupExactRespaStepWork(nextLegacyForceFlags,
                                                                 inputRecord,
                                                                 nextStep,
                                                                 nextRunSchedule.domainWork,
                                                                 nextRunSchedule.simulationWork);
    if (useGpuDeviceKicks && nextRunSchedule.stepWork.computeLongRangeNonbondedForces
        && !nextRunSchedule.stepWork.computeEnergy && !nextRunSchedule.stepWork.computeVirial
        && !nextRunSchedule.stepWork.computeDhdl)
    {
        nextRunSchedule.stepWork.useGpuPmeFReduction = true;
    }

    if (traceState)
    {
        appendExactRespaStateTraceRows(exactRespaStep.step,
                                       "pre_initial_kick_velocity",
                                       state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                       ddGlobalAtomIndices,
                                       ddGlobalAtomIndicesCount,
                                       fr_->nbv.get());
    }

    if (useGpuUpdate)
    {
        if (!useGpuDeviceKicks)
        {
            applyRespaHalfKicks(inputRecord,
                                exactRespaStep.step,
                                RespaKickPhase::Initial,
                                exactRespaStep.mdatoms.homenr,
                                exactRespaStep.mdatoms.ptype,
                                exactRespaStep.mdatoms.invMassPerDim,
                                &exactRespaStep.exactRespaForceStore,
                                ddGlobalAtomIndices,
                                ddGlobalAtomIndicesCount,
                                fr_->nbv.get(),
                                exactRespaStep.forceBuffers,
                                extendedUpdate,
                                state_->v.arrayRefWithPadding().unpaddedArrayRef());
        }
        if (traceState)
        {
            appendExactRespaStateTraceRows(exactRespaStep.step,
                                           "post_initial_kick_velocity",
                                           state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                           ddGlobalAtomIndices,
                                           ddGlobalAtomIndicesCount,
                                           fr_->nbv.get());
        }
        if (traceState && tracePositions)
        {
            appendExactRespaStateTraceRows(exactRespaStep.step,
                                           "pre_drift_position",
                                           state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                           ddGlobalAtomIndices,
                                           ddGlobalAtomIndicesCount,
                                           fr_->nbv.get());
        }
        recordExactRespaRuntimeEventForTesting(exactRespaStep.step, ExactRespaRuntimeEventType::Drift, 0);
        GMX_RELEASE_ASSERT(exactRespaGpuUpdater_ != nullptr,
                           "Exact r-RESPA GPU update requires an initialized GPU updater.");
        GMX_RELEASE_ASSERT(fr_->stateGpu != nullptr,
                           "Exact r-RESPA GPU update requires GPU state buffers.");

#if GMX_GPU_CUDA
        if (useGpuDeviceKicks)
        {
            GMX_RELEASE_ASSERT(fr_->nbv != nullptr && fr_->nbv->gpuNbv() != nullptr,
                               "Exact r-RESPA direct GPU kicks require GPU NBNXM state");
            const bool refreshAtomOrder = exactRespaStep.step == inputRecord.init_step
                                          || (inputRecord.nstlist > 0
                                              && exactRespaStep.step % inputRecord.nstlist == 0);
            if (refreshAtomOrder)
            {
                exactRespaGpuUpdater_->setExactRespaNbnxmAtomOrder(fr_->nbv->getGridIndices());
            }

            NbnxmGpu* const nbnxmGpu = fr_->nbv->gpuNbv();
            const auto level0Force = gpu_get_exact_respa_multi_f(nbnxmGpu, 0);
            const auto level1Force = exactRespaNumLevels(inputRecord) > 1
                                             ? gpu_get_exact_respa_multi_f(nbnxmGpu, 1)
                                             : DeviceBuffer<RVec>{};
            const auto level2Force = exactRespaNumLevels(inputRecord) > 2
                                             ? pme_gpu_get_device_f(fr_->pmedata)
                                             : DeviceBuffer<RVec>{};
            const real level0HalfDt = 0.5_real * inputRecord.delta_t;
            const real level1HalfDt = exactRespaNumLevels(inputRecord) > 1
                                              ? 0.5_real * inputRecord.delta_t
                                                        * exactRespaLevelStepFactor(inputRecord.exactRespa, 1)
                                              : 0;
            const real level2HalfDt = exactRespaNumLevels(inputRecord) > 2
                                              ? 0.5_real * inputRecord.delta_t
                                                        * exactRespaLevelStepFactor(inputRecord.exactRespa, 2)
                                              : 0;
            if (exactRespaStep.step == inputRecord.init_step)
            {
                auditExactRespaGpuDeviceKickForces(exactRespaStep.exactRespaForceStore,
                                                   fr_->nbv.get(),
                                                   fr_->stateGpu,
                                                   level0Force,
                                                   level1Force,
                                                   level2Force,
                                                   exactRespaStep.mdatoms.homenr);
            }
            exactRespaGpuUpdater_->exactRespaKickAndDrift(
                    level0Force,
                    level1Force,
                    level2Force,
                    highestActiveExactRespaLevel(inputRecord.exactRespa, exactRespaStep.step),
                    level0HalfDt,
                    level1HalfDt,
                    level2HalfDt,
                    inputRecord.delta_t);
        }
        else
#endif
        {
            fr_->stateGpu->copyVelocitiesToGpu(state_->v.arrayRefWithPadding().unpaddedArrayRef(),
                                               AtomLocality::Local);
            driftRespaPositionsOnGpu(fr_->stateGpu, exactRespaGpuUpdater_, inputRecord.delta_t);
        }
        fr_->stateGpu->setXUpdatedOnDeviceEventExpectedConsumptionCount(1);
        if (traceState && tracePositions)
        {
            fr_->stateGpu->copyCoordinatesFromGpu(
                    state_->x.arrayRefWithPadding().unpaddedArrayRef(),
                    AtomLocality::Local,
                    exactRespaGpuUpdater_->xUpdatedOnDeviceEvent());
            fr_->stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
            copiedCoordinatesFromGpuAfterDrift = true;
        }
    }
    else
    {
        if (softStartEnabled)
        {
            const ExactRespaPreparedHalfKicks preparedHalfKicks = prepareRespaHalfKicks(
                    inputRecord,
                    exactRespaStep.step,
                    RespaKickPhase::Initial,
                    &exactRespaStep.exactRespaForceStore,
                    ddGlobalAtomIndices,
                    ddGlobalAtomIndicesCount,
                    fr_->nbv.get());
            applySoftStartPreparedRespaHalfKicks(
                    inputRecord,
                    exactRespaStep.step,
                    RespaKickPhase::Initial,
                    exactRespaStep.mdatoms.homenr,
                    exactRespaStep.mdatoms.ptype,
                    exactRespaStep.mdatoms.invMassPerDim,
                    preparedHalfKicks,
                    exactRespaStep.mdatoms,
                    ddGlobalAtomIndices,
                    ddGlobalAtomIndicesCount,
                    cr_->commMyGroup,
                    &exactRespaSoftStartState_,
                    state_->v.arrayRefWithPadding().unpaddedArrayRef());
            if (traceState)
            {
                appendExactRespaStateTraceRows(exactRespaStep.step,
                                               "post_initial_kick_velocity",
                                               state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                               ddGlobalAtomIndices,
                                               ddGlobalAtomIndicesCount,
                                               fr_->nbv.get());
                if (tracePositions)
                {
                    appendExactRespaStateTraceRows(exactRespaStep.step,
                                                   "pre_drift_position",
                                                   state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                                   ddGlobalAtomIndices,
                                                   ddGlobalAtomIndicesCount,
                                                   fr_->nbv.get());
                }
            }
            recordExactRespaRuntimeEventForTesting(
                    exactRespaStep.step, ExactRespaRuntimeEventType::Drift, 0);
            driftRespaPositions(exactRespaStep.mdatoms.homenr,
                                exactRespaStep.mdatoms.ptype,
                                exactRespaStep.mdatoms.invMassPerDim,
                                inputRecord.delta_t,
                                extendedUpdate,
                                state_->x.arrayRefWithPadding().unpaddedArrayRef(),
                                state_->v.arrayRefWithPadding().unpaddedArrayRef());
        }
        else if (traceState)
        {
            applyRespaHalfKicks(inputRecord,
                                exactRespaStep.step,
                                RespaKickPhase::Initial,
                                exactRespaStep.mdatoms.homenr,
                                exactRespaStep.mdatoms.ptype,
                                exactRespaStep.mdatoms.invMassPerDim,
                                &exactRespaStep.exactRespaForceStore,
                                ddGlobalAtomIndices,
                                ddGlobalAtomIndicesCount,
                                fr_->nbv.get(),
                                exactRespaStep.forceBuffers,
                                extendedUpdate,
                                state_->v.arrayRefWithPadding().unpaddedArrayRef());
            appendExactRespaStateTraceRows(exactRespaStep.step,
                                           "post_initial_kick_velocity",
                                           state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                           ddGlobalAtomIndices,
                                           ddGlobalAtomIndicesCount,
                                           fr_->nbv.get());
            if (tracePositions)
            {
                appendExactRespaStateTraceRows(exactRespaStep.step,
                                               "pre_drift_position",
                                               state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                               ddGlobalAtomIndices,
                                               ddGlobalAtomIndicesCount,
                                               fr_->nbv.get());
            }
            recordExactRespaRuntimeEventForTesting(exactRespaStep.step, ExactRespaRuntimeEventType::Drift, 0);
            if (inlineMttkBoxRemap)
            {
                exactRespaMttkScaleBox(*state_, 0.5_real * inputRecord.delta_t);
            }
            driftRespaPositions(exactRespaStep.mdatoms.homenr,
                                exactRespaStep.mdatoms.ptype,
                                exactRespaStep.mdatoms.invMassPerDim,
                                inputRecord.delta_t,
                                extendedUpdate,
                                state_->x.arrayRefWithPadding().unpaddedArrayRef(),
                                state_->v.arrayRefWithPadding().unpaddedArrayRef());
            if (inlineMttkBoxRemap)
            {
                exactRespaMttkScaleBox(*state_, 0.5_real * inputRecord.delta_t);
            }
        }
        else if (exactRespaFusedInitialDriftEnabled())
        {
            const ExactRespaPreparedHalfKicks preparedHalfKicks = prepareRespaHalfKicks(
                    inputRecord,
                    exactRespaStep.step,
                    RespaKickPhase::Initial,
                    &exactRespaStep.exactRespaForceStore,
                    ddGlobalAtomIndices,
                    ddGlobalAtomIndicesCount,
                    fr_->nbv.get());
            recordExactRespaRuntimeEventForTesting(exactRespaStep.step, ExactRespaRuntimeEventType::Drift, 0);
            if (inlineMttkBoxRemap)
            {
                exactRespaMttkScaleBox(*state_, 0.5_real * inputRecord.delta_t);
            }
            applyPreparedRespaInitialHalfKicksAndDrift(inputRecord,
                                                       exactRespaStep.mdatoms.homenr,
                                                       exactRespaStep.mdatoms.ptype,
                                                       exactRespaStep.mdatoms.invMassPerDim,
                                                       preparedHalfKicks,
                                                       inputRecord.delta_t,
                                                       extendedUpdate,
                                                       state_->x.arrayRefWithPadding().unpaddedArrayRef(),
                                                       state_->v.arrayRefWithPadding().unpaddedArrayRef());
            if (inlineMttkBoxRemap)
            {
                exactRespaMttkScaleBox(*state_, 0.5_real * inputRecord.delta_t);
            }
        }
        else
        {
            applyRespaHalfKicks(inputRecord,
                                exactRespaStep.step,
                                RespaKickPhase::Initial,
                                exactRespaStep.mdatoms.homenr,
                                exactRespaStep.mdatoms.ptype,
                                exactRespaStep.mdatoms.invMassPerDim,
                                &exactRespaStep.exactRespaForceStore,
                                ddGlobalAtomIndices,
                                ddGlobalAtomIndicesCount,
                                fr_->nbv.get(),
                                exactRespaStep.forceBuffers,
                                extendedUpdate,
                                state_->v.arrayRefWithPadding().unpaddedArrayRef());
            recordExactRespaRuntimeEventForTesting(exactRespaStep.step, ExactRespaRuntimeEventType::Drift, 0);
            if (inlineMttkBoxRemap)
            {
                exactRespaMttkScaleBox(*state_, 0.5_real * inputRecord.delta_t);
            }
            driftRespaPositions(exactRespaStep.mdatoms.homenr,
                                exactRespaStep.mdatoms.ptype,
                                exactRespaStep.mdatoms.invMassPerDim,
                                inputRecord.delta_t,
                                extendedUpdate,
                                state_->x.arrayRefWithPadding().unpaddedArrayRef(),
                                state_->v.arrayRefWithPadding().unpaddedArrayRef());
            if (inlineMttkBoxRemap)
            {
                exactRespaMttkScaleBox(*state_, 0.5_real * inputRecord.delta_t);
            }
        }
    }
    if (traceState && tracePositions)
    {
        appendExactRespaStateTraceRows(exactRespaStep.step,
                                       "post_drift_position",
                                       state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                       ddGlobalAtomIndices,
                                       ddGlobalAtomIndicesCount,
                                       fr_->nbv.get());
    }

    const bool pressureTrotterNeedsHostCoordinates =
            inputrecNptTrotter(&inputRecord) || inputrecNphTrotter(&inputRecord);
    const bool residentXHostRefresh = nextStepIsNsStep
                                      || (pressureTrotterNeedsHostCoordinates
                                          && inputRecord.nsttcouple > 0
                                          && nextStep % inputRecord.nsttcouple == 0)
                                      || (inputRecord.nstxout > 0 && nextStep % inputRecord.nstxout == 0)
                                      || (inputRecord.nstxout_compressed > 0
                                          && nextStep % inputRecord.nstxout_compressed == 0);
    if (useGpuUpdate && !copiedCoordinatesFromGpuAfterDrift
        && (!exactRespaGpuResidentXProbeEnabled() || residentXHostRefresh))
    {
        // Exact r-RESPA still refreshes forces through do_force() using host-visible coordinates.
        // GPU drift therefore has to materialize updated x on the host before every nested force step,
        // not only on neighbor-search steps.
        fr_->stateGpu->copyCoordinatesFromGpu(
                state_->x.arrayRefWithPadding().unpaddedArrayRef(),
                AtomLocality::Local,
                exactRespaGpuUpdater_->xUpdatedOnDeviceEvent());
        fr_->stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
        copiedCoordinatesFromGpuAfterDrift = true;
    }

    if (fr_->pbcType != PbcType::No && exactRespaStatePbcWrappingEnabled())
    {
        // Diagnostic-only path: per-atom wrapping can split bonded molecules
        // across PBC and make listed 1-4 distances exceed the table range.
        auto statePositions = state_->x.arrayRefWithPadding().unpaddedArrayRef().subArray(
                0, exactRespaStep.mdatoms.homenr);
        auto stateVelocities = state_->v.arrayRefWithPadding().unpaddedArrayRef().subArray(
                0, exactRespaStep.mdatoms.homenr);
        const int numThreads = gmx_omp_nthreads_get(ModuleMultiThread::Default);
        if (trackExactImages)
        {
            putAtomsInBoxAndTrackExactRespaImages(nextStep,
                                                  fr_->pbcType,
                                                  state_->box,
                                                  fr_->haveBoxDeformation,
                                                  inputRecord.deform,
                                                  statePositions,
                                                  stateVelocities,
                                                  numThreads);
        }
        else
        {
            put_atoms_in_box_omp(fr_->pbcType,
                                 state_->box,
                                 fr_->haveBoxDeformation,
                                 inputRecord.deform,
                                 statePositions,
                                 stateVelocities,
                                 numThreads);
        }
    }

    tensor         nextForceVir = { { 0 } };
    gmx_enerdata_t& nextEnerd   = exactRespaNestedForceScratchEnerd(exactRespaStep.enerd);
    rvec           nextMuTot    = { 0, 0, 0 };

    if (!useGpuUpdate)
    {
        wallcycle_stop(wallCycleCounters_, WallCycleCounter::Update);
    }

    {
        const char* previousForceContextLabel = g_respaDoForceContextLabel;
        g_respaDoForceContextLabel           = "exact_respa_next_step_force";
        do_force(fpLog_,
                 cr_,
                 inputRecord,
                 mdModulesNotifiers_,
                 exactRespaStep.awh,
                 enforcedRotation_,
                 imdSession_,
                 pullWork_,
                 nextStep,
                 nrnb_,
                 wallCycleCounters_,
                 top_,
                 state_->box,
                 state_->x.arrayRefWithPadding(),
                 state_->v.arrayRefWithPadding().unpaddedArrayRef(),
                 &state_->hist,
                 &exactRespaStep.forceBuffers.view(),
                 &exactRespaStep.exactRespaForceStore,
                 nextForceVir,
                 &exactRespaStep.mdatoms,
                 &nextEnerd,
                 state_->lambda,
                 fr_,
                 nextRunSchedule,
                 virtualSites_,
                 nextMuTot,
                 exactRespaStep.time + inputRecord.delta_t,
                 exactRespaStep.ed,
                 fr_->longRangeNonbondeds.get(),
                 exactRespaStep.ddBalanceRegionHandler);
        g_respaDoForceContextLabel = previousForceContextLabel;
    }
    if (trackExactImages)
    {
        maybeWriteFinalExactRespaImageSidecar(
                nextStep,
                imageSidecarFinalStep,
                state_->box,
                state_->x.arrayRefWithPadding().unpaddedConstArrayRef().subArray(
                        0, exactRespaStep.mdatoms.homenr));
    }
    if (exactRespaReturnNextVirialEnabled() || nextStepNeedsVirialForMttkPost)
    {
        copy_mat(nextForceVir, exactRespaStep.forceVir);
    }

    if (!useGpuUpdate)
    {
        wallcycle_start(wallCycleCounters_, WallCycleCounter::Update);
    }

    recordExactRespaRefreshEventsForTesting(inputRecord.exactRespa, exactRespaStep.step);

    if (softStartEnabled)
    {
        const int* finalGlobalAtomIndices =
                haveDDAtomOrdering(*cr_) ? cr_->dd->globalAtomIndices.data() : nullptr;
        const int finalGlobalAtomIndicesCount =
                haveDDAtomOrdering(*cr_) ? static_cast<int>(cr_->dd->globalAtomIndices.size()) : 0;
        const ExactRespaPreparedHalfKicks preparedHalfKicks = prepareRespaHalfKicks(
                inputRecord,
                exactRespaStep.step,
                RespaKickPhase::Final,
                &exactRespaStep.exactRespaForceStore,
                finalGlobalAtomIndices,
                finalGlobalAtomIndicesCount,
                fr_->nbv.get());
        applySoftStartPreparedRespaHalfKicks(
                inputRecord,
                exactRespaStep.step,
                RespaKickPhase::Final,
                exactRespaStep.mdatoms.homenr,
                exactRespaStep.mdatoms.ptype,
                exactRespaStep.mdatoms.invMassPerDim,
                preparedHalfKicks,
                exactRespaStep.mdatoms,
                finalGlobalAtomIndices,
                finalGlobalAtomIndicesCount,
                cr_->commMyGroup,
                &exactRespaSoftStartState_,
                state_->v.arrayRefWithPadding().unpaddedArrayRef());
    }
    else if (!useGpuDeviceKicks)
    {
        applyRespaHalfKicks(inputRecord,
                            exactRespaStep.step,
                            RespaKickPhase::Final,
                            exactRespaStep.mdatoms.homenr,
                            exactRespaStep.mdatoms.ptype,
                            exactRespaStep.mdatoms.invMassPerDim,
                            &exactRespaStep.exactRespaForceStore,
                            ddGlobalAtomIndices,
                            ddGlobalAtomIndicesCount,
                            fr_->nbv.get(),
                            exactRespaStep.forceBuffers,
                            extendedUpdate,
                            state_->v.arrayRefWithPadding().unpaddedArrayRef());
    }
#if GMX_GPU_CUDA
    else
    {
        NbnxmGpu* const nbnxmGpu = fr_->nbv->gpuNbv();
        const auto level0Force = gpu_get_exact_respa_multi_f(nbnxmGpu, 0);
        const auto level1Force = exactRespaNumLevels(inputRecord) > 1
                                         ? gpu_get_exact_respa_multi_f(nbnxmGpu, 1)
                                         : DeviceBuffer<RVec>{};
        const auto level2Force = exactRespaNumLevels(inputRecord) > 2
                                         ? pme_gpu_get_device_f(fr_->pmedata)
                                         : DeviceBuffer<RVec>{};
        const real level0HalfDt = 0.5_real * inputRecord.delta_t;
        const real level1HalfDt = exactRespaNumLevels(inputRecord) > 1
                                          ? 0.5_real * inputRecord.delta_t
                                                    * exactRespaLevelStepFactor(inputRecord.exactRespa, 1)
                                          : 0;
        const real level2HalfDt = exactRespaNumLevels(inputRecord) > 2
                                          ? 0.5_real * inputRecord.delta_t
                                                    * exactRespaLevelStepFactor(inputRecord.exactRespa, 2)
                                          : 0;
        const int highestActiveLevel =
                highestActiveExactRespaLevel(inputRecord.exactRespa, nextStep);
        if (nextStepIsNsStep)
        {
            exactRespaGpuUpdater_->setExactRespaNbnxmAtomOrder(fr_->nbv->getGridIndices());
        }
        const DeviceStream* updateStream = fr_->stateGpu->getUpdateStream();
        GMX_RELEASE_ASSERT(updateStream != nullptr && fr_->exactRespaLocalForcesReady != nullptr,
                           "Exact r-RESPA direct GPU kicks require local-force completion events");
        fr_->exactRespaLocalForcesReady->enqueueWaitEvent(*updateStream);
        if (highestActiveLevel >= 2)
        {
            GpuEventSynchronizer* pmeForcesReady =
                    pme_gpu_get_f_ready_synchronizer(fr_->pmedata);
            GMX_RELEASE_ASSERT(pmeForcesReady != nullptr,
                               "Exact r-RESPA direct GPU kicks require a PME-force completion event");
            pmeForcesReady->enqueueWaitEvent(*updateStream);
        }
        if (exactRespaStep.step == inputRecord.init_step)
        {
            auditExactRespaGpuDeviceKickForces(exactRespaStep.exactRespaForceStore,
                                               fr_->nbv.get(),
                                               fr_->stateGpu,
                                               level0Force,
                                               level1Force,
                                               level2Force,
                                               exactRespaStep.mdatoms.homenr);
        }
        exactRespaGpuUpdater_->exactRespaKick(
                level0Force,
                level1Force,
                level2Force,
                highestActiveLevel,
                level0HalfDt,
                level1HalfDt,
                level2HalfDt);
    }
#endif
    if (traceState)
    {
        appendExactRespaStateTraceRows(exactRespaStep.step,
                                       "post_final_kick_velocity",
                                       state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                       ddGlobalAtomIndices,
                                       ddGlobalAtomIndicesCount,
                                       fr_->nbv.get());
    }
    if (useGpuUpdate && !useGpuDeviceKicks)
    {
        fr_->stateGpu->copyVelocitiesToGpu(state_->v.arrayRefWithPadding().unpaddedArrayRef(),
                                           AtomLocality::Local);
    }
    GMX_UNUSED_VALUE(nextStepIsNsStep);
}

void LegacySimulator::doExactRespaNestedPrototypeStep(const ExactRespaStepContext& exactRespaStep)
{
    const t_inputrec& inputRecord = exactRespaStep.inputRecord;
    const bool        useGpuUpdate = exactRespaStep.simulationWork.useGpuUpdate;
    initializeExactRespaSoftStartState(
            inputRecord, useGpuUpdate, exactRespaStep.step, &exactRespaSoftStartState_);
    const ExactRespaExtendedVvUpdate extendedUpdate =
            exactRespaExtendedVvUpdateFromState(inputRecord, *state_);
    const int*        ddGlobalAtomIndices =
            haveDDAtomOrdering(*cr_) ? cr_->dd->globalAtomIndices.data() : nullptr;
    const int         ddGlobalAtomIndicesCount =
            haveDDAtomOrdering(*cr_) ? static_cast<int>(cr_->dd->globalAtomIndices.size()) : 0;

    if (exactRespaFusedInitialDriftEnabled())
    {
        const ExactRespaPreparedHalfKicks preparedHalfKicks = prepareRespaHalfKicks(
                inputRecord,
                exactRespaStep.step,
                RespaKickPhase::Initial,
                &exactRespaStep.exactRespaForceStore,
                ddGlobalAtomIndices,
                ddGlobalAtomIndicesCount,
                fr_->nbv.get());
        recordExactRespaRuntimeEventForTesting(exactRespaStep.step, ExactRespaRuntimeEventType::Drift, 0);
        applyPreparedRespaInitialHalfKicksAndDrift(inputRecord,
                                                   exactRespaStep.mdatoms.homenr,
                                                   exactRespaStep.mdatoms.ptype,
                                                   exactRespaStep.mdatoms.invMassPerDim,
                                                   preparedHalfKicks,
                                                   inputRecord.delta_t,
                                                   extendedUpdate,
                                                   state_->x.arrayRefWithPadding().unpaddedArrayRef(),
                                                   state_->v.arrayRefWithPadding().unpaddedArrayRef());
    }
    else
    {
        applyRespaHalfKicks(inputRecord,
                            exactRespaStep.step,
                            RespaKickPhase::Initial,
                            exactRespaStep.mdatoms.homenr,
                            exactRespaStep.mdatoms.ptype,
                            exactRespaStep.mdatoms.invMassPerDim,
                            &exactRespaStep.exactRespaForceStore,
                            ddGlobalAtomIndices,
                            ddGlobalAtomIndicesCount,
                            fr_->nbv.get(),
                            exactRespaStep.forceBuffers,
                            extendedUpdate,
                            state_->v.arrayRefWithPadding().unpaddedArrayRef());
        recordExactRespaRuntimeEventForTesting(exactRespaStep.step, ExactRespaRuntimeEventType::Drift, 0);
        driftRespaPositions(exactRespaStep.mdatoms.homenr,
                            exactRespaStep.mdatoms.ptype,
                            exactRespaStep.mdatoms.invMassPerDim,
                            inputRecord.delta_t,
                            extendedUpdate,
                            state_->x.arrayRefWithPadding().unpaddedArrayRef(),
                            state_->v.arrayRefWithPadding().unpaddedArrayRef());
    }
    const int64_t nextStep         = exactRespaStep.step + 1;
    const bool    nextStepIsNsStep = (inputRecord.nstlist > 0 && nextStep % inputRecord.nstlist == 0);
    const bool    nextStepNeedsVirialForMttkPost =
            exactRespaMttkPostTrotterNeedsNextVirial(inputRecord, exactRespaStep.step);
    const int nextLegacyForceFlags =
            GMX_FORCE_STATECHANGED | GMX_FORCE_ALLFORCES | (nextStepIsNsStep ? GMX_FORCE_NS : 0)
            | (nextStepNeedsVirialForMttkPost ? GMX_FORCE_VIRIAL : 0);

    MdrunScheduleWorkload nextRunSchedule = *runScheduleWork_;
    nextRunSchedule.stepWork = setupExactRespaStepWorkload(nextLegacyForceFlags,
                                                           inputRecord,
                                                           nextStep,
                                                           nextRunSchedule.domainWork,
                                                           nextRunSchedule.simulationWork);
    nextRunSchedule.exactRespaStepWork = setupExactRespaStepWork(nextLegacyForceFlags,
                                                                 inputRecord,
                                                                 nextStep,
                                                                 nextRunSchedule.domainWork,
                                                                 nextRunSchedule.simulationWork);

    tensor         nextForceVir = { { 0 } };
    gmx_enerdata_t& nextEnerd   = exactRespaNestedForceScratchEnerd(exactRespaStep.enerd);
    rvec           nextMuTot    = { 0, 0, 0 };

    if (!useGpuUpdate)
    {
        wallcycle_stop(wallCycleCounters_, WallCycleCounter::Update);
    }

    {
        const char* previousForceContextLabel = g_respaDoForceContextLabel;
        g_respaDoForceContextLabel           = "exact_respa_next_step_force";
        do_force(fpLog_,
                 cr_,
                 inputRecord,
                 mdModulesNotifiers_,
                 exactRespaStep.awh,
                 enforcedRotation_,
                 imdSession_,
                 pullWork_,
                 nextStep,
                 nrnb_,
                 wallCycleCounters_,
                 top_,
                 state_->box,
                 state_->x.arrayRefWithPadding(),
                 state_->v.arrayRefWithPadding().unpaddedArrayRef(),
                 &state_->hist,
                 &exactRespaStep.forceBuffers.view(),
                 &exactRespaStep.exactRespaForceStore,
                 nextForceVir,
                 &exactRespaStep.mdatoms,
                 &nextEnerd,
                 state_->lambda,
                 fr_,
                 nextRunSchedule,
                 virtualSites_,
                 nextMuTot,
                 exactRespaStep.time + inputRecord.delta_t,
                 exactRespaStep.ed,
                 fr_->longRangeNonbondeds.get(),
                 exactRespaStep.ddBalanceRegionHandler);
        g_respaDoForceContextLabel = previousForceContextLabel;
    }
    if (exactRespaReturnNextVirialEnabled() || nextStepNeedsVirialForMttkPost)
    {
        copy_mat(nextForceVir, exactRespaStep.forceVir);
    }

    if (!useGpuUpdate)
    {
        wallcycle_start(wallCycleCounters_, WallCycleCounter::Update);
    }

    recordExactRespaRefreshEventsForTesting(inputRecord.exactRespa, exactRespaStep.step);

    applyRespaHalfKicks(inputRecord,
                        exactRespaStep.step,
                        RespaKickPhase::Final,
                        exactRespaStep.mdatoms.homenr,
                        exactRespaStep.mdatoms.ptype,
                        exactRespaStep.mdatoms.invMassPerDim,
                        &exactRespaStep.exactRespaForceStore,
                        ddGlobalAtomIndices,
                        ddGlobalAtomIndicesCount,
                        fr_->nbv.get(),
                        exactRespaStep.forceBuffers,
                        extendedUpdate,
                        state_->v.arrayRefWithPadding().unpaddedArrayRef());
}
} // namespace gmx
