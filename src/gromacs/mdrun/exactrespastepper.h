/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2025- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */
#ifndef GMX_MDRUN_EXACTRESPASTEPPER_H
#define GMX_MDRUN_EXACTRESPASTEPPER_H

#include <cstdint>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vectypes.h"

class gmx_ekindata_t;
struct gmx_global_stat;
struct gmx_wallcycle;
struct t_forcerec;
struct DDBalanceRegionHandler;
struct gmx_edsam;
struct gmx_enerdata_t;
struct t_inputrec;
struct t_mdatoms;
struct t_nrnb;
struct t_vcm;
class t_state;

namespace gmx
{
class Awh;
class DomainLifetimeWorkload;
class ExactRespaForceStore;
class ForceBuffers;
class MpiComm;
class MdrunScheduleWorkload;
class ObservablesReducer;
class SimulationWorkload;
class SimulationSignaller;

struct ExactRespaStepContext
{
    int64_t                      step = 0;
    double                       time = 0.0;
    const t_inputrec&            inputRecord;
    const t_mdatoms&             mdatoms;
    const SimulationWorkload&    simulationWork;
    const DomainLifetimeWorkload& domainWork;
    ForceBuffers&                forceBuffers;
    ExactRespaForceStore&        exactRespaForceStore;
    tensor&                      forceVir;
    rvec&                        muTot;
    gmx_enerdata_t&              enerd;
    Awh*                         awh = nullptr;
    gmx_edsam*                   ed = nullptr;
    const DDBalanceRegionHandler& ddBalanceRegionHandler;
};

struct ExactRespaVelocityVerletObservablesContext
{
    const t_inputrec&    inputRecord;
    int64_t              step = 0;
    const MpiComm&       mpiComm;
    t_forcerec*          fr = nullptr;
    t_state*             state = nullptr;
    const t_mdatoms*     mdatoms = nullptr;
    t_nrnb*              nrnb = nullptr;
    t_vcm*               vcm = nullptr;
    gmx_wallcycle*       wallCycle = nullptr;
    gmx_enerdata_t*      enerd = nullptr;
    gmx_ekindata_t*      ekind = nullptr;
    gmx_global_stat*     gstat = nullptr;
    SimulationSignaller* nullSignaller = nullptr;
    ObservablesReducer*  observablesReducer = nullptr;
    tensor&              forceVir;
    tensor&              shakeVir;
    tensor&              totalVir;
    tensor&              pres;
    bool                 calcEner = false;
    bool                 calcVir = false;
    bool                 calcGlobalStats = false;
    bool                 stopCenterOfMass = false;
    gmx_bool*            sumEkinhOld = nullptr;
    real*                savedConservedQuantity = nullptr;
    real*                lastEkin = nullptr;
};

void prepareExactRespaVelocityVerletObservables(
        const ExactRespaVelocityVerletObservablesContext& context);

} // namespace gmx

#endif
