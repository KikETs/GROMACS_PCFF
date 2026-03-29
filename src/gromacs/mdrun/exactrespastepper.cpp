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

#include <cstdint>
#include <utility>
#include <vector>

#include "gromacs/domdec/collect.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdlib/md_support.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdlib/vcm.h"
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
#include "gromacs/utility/vec.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/vectypes.h"

namespace gmx
{

namespace
{
thread_local ExactRespaRuntimeEventSink* g_exactRespaRuntimeEventSink = nullptr;
}

void setExactRespaRuntimeEventSinkForTesting(ExactRespaRuntimeEventSink* sink)
{
    g_exactRespaRuntimeEventSink = sink;
}

namespace
{

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

void recordExactRespaRuntimeEventForTesting(const int64_t                  baseStep,
                                            const ExactRespaRuntimeEventType type,
                                            const int                      level)
{
    if (g_exactRespaRuntimeEventSink == nullptr)
    {
        return;
    }
    g_exactRespaRuntimeEventSink->recordEvent({ baseStep, type, level });
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

void applyRespaHalfKicks(const t_inputrec&                 inputRecord,
                         const int64_t                     baseStep,
                         const RespaKickPhase              phase,
                         const int                         homenr,
                         const ArrayRef<const ParticleType> ptype,
                         const ArrayRef<const RVec>        invMassPerDim,
                         const ExactRespaForceStore*       exactRespaForceStore,
                         ForceBuffers&                     forceBuffers,
                         ArrayRef<RVec>                    velocity)
{
    const ExactRespaBaseStepTrace trace = exactRespaBaseStepTrace(inputRecord.exactRespa, baseStep);
    const std::vector<int>& kickLevels =
            (phase == RespaKickPhase::Initial) ? trace.initialKickLevels : trace.finalKickLevels;

    for (const int mtsLevel : kickLevels)
    {
        const auto forceForLevel = forceForExactRespaKickLevel(exactRespaForceStore, mtsLevel);
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

    applyRespaHalfKicks(inputRecord,
                        exactRespaStep.step,
                        RespaKickPhase::Initial,
                        exactRespaStep.mdatoms.homenr,
                        exactRespaStep.mdatoms.ptype,
                        exactRespaStep.mdatoms.invMassPerDim,
                        &exactRespaStep.exactRespaForceStore,
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
                        exactRespaStep.forceBuffers,
                        state_->v.arrayRefWithPadding().unpaddedArrayRef());
}

void LegacySimulator::doExactRespaNestedPrototypeStep(const ExactRespaStepContext& exactRespaStep)
{
    const t_inputrec& inputRecord = exactRespaStep.inputRecord;

    applyRespaHalfKicks(inputRecord,
                        exactRespaStep.step,
                        RespaKickPhase::Initial,
                        exactRespaStep.mdatoms.homenr,
                        exactRespaStep.mdatoms.ptype,
                        exactRespaStep.mdatoms.invMassPerDim,
                        &exactRespaStep.exactRespaForceStore,
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
                        exactRespaStep.forceBuffers,
                        state_->v.arrayRefWithPadding().unpaddedArrayRef());
}
} // namespace gmx
