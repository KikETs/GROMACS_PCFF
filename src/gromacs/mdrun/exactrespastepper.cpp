/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2025- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */

#include "gmxpre.h"

#include "legacysimulator.h"
#include "exactrespasteppertesting.h"

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "gromacs/domdec/collect.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdlib/md_support.h"
#include "gromacs/mdlib/update_constrain_gpu.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdtypes/state_propagator_data_gpu.h"
#include "gromacs/mdlib/vcm.h"
#include "gromacs/nbnxm/nbnxm.h"
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
#include "gromacs/taskassignment/include/gromacs/taskassignment/decidesimulationworkload.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/vectypes.h"

namespace gmx
{

namespace
{
thread_local ExactRespaRuntimeEventSink* g_exactRespaRuntimeEventSink = nullptr;
thread_local std::unique_ptr<std::ofstream> g_exactRespaRuntimeEventTraceStream;
thread_local std::string                    g_exactRespaRuntimeEventTracePath;
thread_local bool                           g_exactRespaRuntimeEventTraceHasContent = false;
std::mutex                                  g_exactRespaStateTraceMutex;
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

    const auto localAtomOrder = nbv->getLocalAtomOrder();
    if (atomIndex < localAtomOrder.ssize())
    {
        return localAtomOrder[atomIndex];
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
    const ExactRespaRuntimeEvent event{ baseStep, type, level };
    maybeRecordExactRespaRuntimeEventArtifact(event);

    if (g_exactRespaRuntimeEventSink == nullptr)
    {
        return;
    }
    g_exactRespaRuntimeEventSink->recordEvent(event);
}

void recordExactRespaRefreshEventsForTesting(const ExactRespaParameters& exactRespa, const int64_t baseStep)
{
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

void applyRespaVelocityHalfKick(const int                             homenr,
                                const ArrayRef<const ParticleType>   ptype,
                                const ArrayRef<const RVec>           invMassPerDim,
                                const ArrayRef<const RVec>&          force,
                                const real                           dt,
                                ArrayRef<RVec>                       velocity)
{
    const real halfDt = 0.5 * dt;
    for (int atom = 0; atom < homenr; atom++)
    {
        if (ptype[atom] == ParticleType::Shell)
        {
            continue;
        }
        for (int d = 0; d < DIM; d++)
        {
            const real inverseMass = invMassPerDim[atom][d];
            if (inverseMass != 0)
            {
                velocity[atom][d] += halfDt * inverseMass * force[atom][d];
            }
        }
    }
}

void driftRespaPositions(const int                             homenr,
                         const ArrayRef<const ParticleType>   ptype,
                         const ArrayRef<const RVec>           invMassPerDim,
                         const real                           dt,
                         ArrayRef<RVec>                       position,
                         ArrayRef<const RVec>                 velocity)
{
    for (int atom = 0; atom < homenr; atom++)
    {
        if (ptype[atom] == ParticleType::Shell)
        {
            continue;
        }
        for (int d = 0; d < DIM; d++)
        {
            if (invMassPerDim[atom][d] != 0)
            {
                position[atom][d] += dt * velocity[atom][d];
            }
        }
    }
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
                         ArrayRef<RVec>                    velocity)
{
    const ExactRespaBaseStepTrace trace = exactRespaBaseStepTrace(inputRecord.exactRespa, baseStep);
    const std::vector<int>& kickLevels =
            (phase == RespaKickPhase::Initial) ? trace.initialKickLevels : trace.finalKickLevels;

    for (const int mtsLevel : kickLevels)
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

        const real scaledDt = inputRecord.delta_t * exactRespaLevelStepFactor(inputRecord.exactRespa, mtsLevel);
        GMX_UNUSED_VALUE(forceBuffers);
        applyRespaVelocityHalfKick(homenr, ptype, invMassPerDim, forceForLevel, scaledDt, velocity);
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
    wallcycle_stop(wallCycleCounters_, WallCycleCounter::Update);
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
    wallcycle_stop(wallCycleCounters_, WallCycleCounter::Update);
}

void LegacySimulator::doExactRespaVelocityVerletStep(const ExactRespaStepContext& exactRespaStep)
{
    const t_inputrec& inputRecord = exactRespaStep.inputRecord;
    const bool        useGpuUpdate = exactRespaStep.simulationWork.useGpuUpdate;
    const bool        traceState = shouldRecordExactRespaStateTrace(exactRespaStep.step);
    const bool        tracePositions = exactRespaStateTraceConfig().includePositions;
    bool              copiedCoordinatesFromGpuAfterDrift = false;
    const int*        ddGlobalAtomIndices =
            haveDDAtomOrdering(*cr_) ? cr_->dd->globalAtomIndices.data() : nullptr;
    const int         ddGlobalAtomIndicesCount =
            haveDDAtomOrdering(*cr_) ? static_cast<int>(cr_->dd->globalAtomIndices.size()) : 0;

    if (traceState)
    {
        appendExactRespaStateTraceRows(exactRespaStep.step,
                                       "pre_initial_kick_velocity",
                                       state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                       ddGlobalAtomIndices,
                                       ddGlobalAtomIndicesCount,
                                       fr_->nbv.get());
    }

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
                        state_->v.arrayRefWithPadding().unpaddedArrayRef());
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
    if (useGpuUpdate)
    {
        GMX_RELEASE_ASSERT(exactRespaGpuUpdater_ != nullptr,
                           "Exact r-RESPA GPU update requires an initialized GPU updater.");
        GMX_RELEASE_ASSERT(fr_->stateGpu != nullptr,
                           "Exact r-RESPA GPU update requires GPU state buffers.");

        fr_->stateGpu->copyVelocitiesToGpu(state_->v.arrayRefWithPadding().unpaddedArrayRef(),
                                           AtomLocality::Local);
        driftRespaPositionsOnGpu(fr_->stateGpu, exactRespaGpuUpdater_, inputRecord.delta_t);
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
        driftRespaPositions(exactRespaStep.mdatoms.homenr,
                            exactRespaStep.mdatoms.ptype,
                            exactRespaStep.mdatoms.invMassPerDim,
                            inputRecord.delta_t,
                            state_->x.arrayRefWithPadding().unpaddedArrayRef(),
                            state_->v.arrayRefWithPadding().unpaddedArrayRef());
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

    gmx_enerdata_t savedEnerd = exactRespaStep.enerd;
    tensor         savedForceVir;
    rvec           savedMuTot;
    copy_mat(exactRespaStep.forceVir, savedForceVir);
    copy_rvec(exactRespaStep.muTot, savedMuTot);

    const int64_t nextStep         = exactRespaStep.step + 1;
    const bool    nextStepIsNsStep = (inputRecord.nstlist > 0 && nextStep % inputRecord.nstlist == 0);
    const int nextLegacyForceFlags =
            GMX_FORCE_STATECHANGED | GMX_FORCE_ALLFORCES | (nextStepIsNsStep ? GMX_FORCE_NS : 0);

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

    if (useGpuUpdate && !copiedCoordinatesFromGpuAfterDrift)
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

    tensor         nextForceVir = { { 0 } };
    gmx_enerdata_t nextEnerd    = exactRespaStep.enerd;
    clear_rvec(exactRespaStep.muTot);

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
             exactRespaStep.muTot,
             exactRespaStep.time + inputRecord.delta_t,
             exactRespaStep.ed,
             fr_->longRangeNonbondeds.get(),
             exactRespaStep.ddBalanceRegionHandler);

    recordExactRespaRefreshEventsForTesting(inputRecord.exactRespa, exactRespaStep.step);
    exactRespaStep.enerd = std::move(savedEnerd);
    copy_mat(savedForceVir, exactRespaStep.forceVir);
    copy_rvec(savedMuTot, exactRespaStep.muTot);

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
                        state_->v.arrayRefWithPadding().unpaddedArrayRef());
    if (traceState)
    {
        appendExactRespaStateTraceRows(exactRespaStep.step,
                                       "post_final_kick_velocity",
                                       state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                       ddGlobalAtomIndices,
                                       ddGlobalAtomIndicesCount,
                                       fr_->nbv.get());
    }
    if (useGpuUpdate)
    {
        fr_->stateGpu->copyVelocitiesToGpu(state_->v.arrayRefWithPadding().unpaddedArrayRef(),
                                           AtomLocality::Local);
        GMX_UNUSED_VALUE(nextStepIsNsStep);
    }
}

void LegacySimulator::doExactRespaNestedPrototypeStep(const ExactRespaStepContext& exactRespaStep)
{
    const t_inputrec& inputRecord = exactRespaStep.inputRecord;
    const int*        ddGlobalAtomIndices =
            haveDDAtomOrdering(*cr_) ? cr_->dd->globalAtomIndices.data() : nullptr;
    const int         ddGlobalAtomIndicesCount =
            haveDDAtomOrdering(*cr_) ? static_cast<int>(cr_->dd->globalAtomIndices.size()) : 0;

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
                        state_->v.arrayRefWithPadding().unpaddedArrayRef());
    recordExactRespaRuntimeEventForTesting(exactRespaStep.step, ExactRespaRuntimeEventType::Drift, 0);
    driftRespaPositions(exactRespaStep.mdatoms.homenr,
                        exactRespaStep.mdatoms.ptype,
                        exactRespaStep.mdatoms.invMassPerDim,
                        inputRecord.delta_t,
                        state_->x.arrayRefWithPadding().unpaddedArrayRef(),
                        state_->v.arrayRefWithPadding().unpaddedArrayRef());

    gmx_enerdata_t savedEnerd = exactRespaStep.enerd;
    tensor         savedForceVir;
    rvec           savedMuTot;
    copy_mat(exactRespaStep.forceVir, savedForceVir);
    copy_rvec(exactRespaStep.muTot, savedMuTot);

    const int64_t nextStep         = exactRespaStep.step + 1;
    const bool    nextStepIsNsStep = (inputRecord.nstlist > 0 && nextStep % inputRecord.nstlist == 0);
    const int nextLegacyForceFlags =
            GMX_FORCE_STATECHANGED | GMX_FORCE_ALLFORCES | (nextStepIsNsStep ? GMX_FORCE_NS : 0);

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
    gmx_enerdata_t nextEnerd    = exactRespaStep.enerd;
    clear_rvec(exactRespaStep.muTot);

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
             exactRespaStep.muTot,
             exactRespaStep.time + inputRecord.delta_t,
             exactRespaStep.ed,
             fr_->longRangeNonbondeds.get(),
             exactRespaStep.ddBalanceRegionHandler);

    recordExactRespaRefreshEventsForTesting(inputRecord.exactRespa, exactRespaStep.step);
    exactRespaStep.enerd = std::move(savedEnerd);
    copy_mat(savedForceVir, exactRespaStep.forceVir);
    copy_rvec(savedMuTot, exactRespaStep.muTot);

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
                        state_->v.arrayRefWithPadding().unpaddedArrayRef());
}
} // namespace gmx
