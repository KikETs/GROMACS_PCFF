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
/*! \internal \file
 *
 * \brief Implements the integrator for normal molecular dynamics simulations
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_mdrun
 */
#include "gmxpre.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gromacs/applied_forces/awh/awh.h"
#include "gromacs/applied_forces/awh/read_params.h"
#include "gromacs/commandline/filenm.h"
#include "gromacs/compat/pointers.h"
#include "gromacs/domdec/collect.h"
#include "gromacs/domdec/dlbtiming.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_network.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/gpuhaloexchange.h"
#include "gromacs/domdec/localtopologychecker.h"
#include "gromacs/domdec/mdsetup.h"
#include "gromacs/domdec/partition.h"
#include "gromacs/essentialdynamics/edsam.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/ewald/pme_load_balancing.h"
#include "gromacs/ewald/pme_pp.h"
#include "gromacs/fileio/enxio.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/gpu_utils/device_stream_manager.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/imd/imd.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/boxmatrix.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/matrix.h"
#include "gromacs/math/paddedvector.h"
#include "gromacs/math/units.h"
#include "gromacs/mdlib/checkpointhandler.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/coupling.h"
#include "gromacs/mdlib/ebin.h"
#include "gromacs/mdlib/enerdata_utils.h"
#include "gromacs/mdlib/energyoutput.h"
#include "gromacs/mdlib/expanded.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdlib/forcerec.h"
#include "gromacs/mdlib/freeenergyparameters.h"
#include "gromacs/mdlib/md_support.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/mdgraph_gpu.h"
#include "gromacs/mdlib/mdoutf.h"
#include "gromacs/mdlib/membed.h"
#include "gromacs/mdlib/resethandler.h"
#include "gromacs/mdlib/sighandler.h"
#include "gromacs/mdlib/simulationsignal.h"
#include "gromacs/mdlib/stat.h"
#include "gromacs/mdlib/stophandler.h"
#include "gromacs/mdlib/tgroup.h"
#include "gromacs/mdlib/trajectory_writing.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdlib/update_constrain_gpu.h"
#include "gromacs/mdlib/update_vv.h"
#include "gromacs/mdlib/vcm.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdrunutility/freeenergy.h"
#include "gromacs/mdrunutility/handlerestart.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdrunutility/multisim.h"
#include "gromacs/mdrunutility/printtime.h"
#include "gromacs/mdtypes/awh_history.h"
#include "gromacs/mdtypes/awh_params.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/df_history.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/energyhistory.h"
#include "gromacs/mdtypes/exactrespaforcestore.h"
#include "gromacs/mdtypes/exactrespaschedule.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/forcebuffers.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/mdrunoptions.h"
#include "gromacs/mdtypes/multipletimestepping.h"
#include "gromacs/mdtypes/observableshistory.h"
#include "gromacs/mdtypes/observablesreducer.h"
#include "gromacs/mdtypes/pull_params.h"
#include "gromacs/mdtypes/pullhistory.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/mdtypes/state_propagator_data_gpu.h"
#include "gromacs/modularsimulator/energydata.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pulling/output.h"
#include "gromacs/pulling/pull.h"
#include "gromacs/swap/swapcoords.h"
#include "gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h"
#include "gromacs/taskassignment/include/gromacs/taskassignment/decidesimulationworkload.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/topology/topology_enums.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

#include "legacysimulator.h"
#include "replicaexchange.h"
#include "shellfc.h"

struct gmx_mdoutf;
struct gmx_shellfc_t;
struct pme_load_balancing_t;

using gmx::SimulationSignaller;

namespace gmx
{
thread_local bool g_respaSuppressDoForceStateXChain = false;
thread_local const char* g_respaDoForceContextLabel = nullptr;
thread_local int64_t g_respaCurrentDoForceStep = -1;
thread_local const int* g_respaCurrentGlobalAtomIndices = nullptr;
thread_local int g_respaCurrentGlobalAtomIndexCount = 0;
thread_local const int* g_respaLatestForceDumpGlobalAtomIndices = nullptr;
thread_local int g_respaLatestForceDumpGlobalAtomIndexCount = 0;
}

namespace
{

bool useNestedExactLammpsRespa(const t_inputrec& inputRecord)
{
    return gmx::useExactRespa(inputRecord);
}

bool useExactVelocityVerletLammpsRespa(const t_inputrec& inputRecord)
{
    return useNestedExactLammpsRespa(inputRecord) && inputRecord.eI == IntegrationAlgorithm::VV;
}

bool shouldUsePmeLoadBalancingForExactRespa(const t_inputrec& inputRecord)
{
    if (!gmx::useExactRespa(inputRecord))
    {
        return true;
    }

    const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_ALLOW_PME_TUNING");
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0
           && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0;
}

bool readEnvReal(const char* name, double* value)
{
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0')
    {
        return false;
    }

    char* end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (end == text || (end != nullptr && *end != '\0'))
    {
        gmx_fatal(FARGS, "Invalid floating point value for %s: '%s'", name, text);
    }
    *value = parsed;
    return true;
}

bool readEnvInt64(const char* name, int64_t* value)
{
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0')
    {
        return false;
    }

    char* end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (end == text || (end != nullptr && *end != '\0'))
    {
        gmx_fatal(FARGS, "Invalid integer value for %s: '%s'", name, text);
    }
    *value = static_cast<int64_t>(parsed);
    return true;
}

enum PcffFixNhMassMask : unsigned
{
    pcffFixNhMassNone                = 0,
    pcffFixNhThermostatMass          = 1U << 0,
    pcffFixNhPressureMass            = 1U << 1,
    pcffFixNhPressureThermostatMass  = 1U << 2,
    pcffFixNhAllMasses               = pcffFixNhThermostatMass | pcffFixNhPressureMass
                         | pcffFixNhPressureThermostatMass,
};

unsigned pcffLammpsFixNhMassMask()
{
    const char* value = std::getenv("GMX_PCFF_MTTK_MASS_MODE");
    if (value == nullptr || *value == '\0')
    {
        return pcffFixNhMassNone;
    }

    const std::string mode(value);
    if (mode == "0" || mode == "off" || mode == "false" || mode == "FALSE" || mode == "gromacs")
    {
        return pcffFixNhMassNone;
    }
    if (mode == "lammps" || mode == "lammps_fixnh" || mode == "lammps_pdamp")
    {
        return pcffFixNhAllMasses;
    }
    if (mode == "lammps_tchain" || mode == "lammps_thermostat")
    {
        return pcffFixNhThermostatMass;
    }
    if (mode == "lammps_pmass" || mode == "lammps_pressure")
    {
        return pcffFixNhPressureMass;
    }
    if (mode == "lammps_pchain" || mode == "lammps_pressure_thermostat")
    {
        return pcffFixNhPressureThermostatMass;
    }
    if (mode == "lammps_tchain_pmass")
    {
        return pcffFixNhThermostatMass | pcffFixNhPressureMass;
    }
    if (mode == "lammps_tchain_pchain")
    {
        return pcffFixNhThermostatMass | pcffFixNhPressureThermostatMass;
    }
    if (mode == "lammps_pmass_pchain")
    {
        return pcffFixNhPressureMass | pcffFixNhPressureThermostatMass;
    }

    gmx_fatal(FARGS,
              "Invalid GMX_PCFF_MTTK_MASS_MODE='%s'. Supported values are gromacs/off "
              "or lammps/lammps_fixnh/lammps_pdamp plus diagnostic component modes "
              "lammps_tchain, lammps_pmass, lammps_pchain, lammps_tchain_pmass, "
              "lammps_tchain_pchain, lammps_pmass_pchain.",
              value);
    return pcffFixNhMassNone;
}

bool pcffUseLammpsFixNhMassMode()
{
    return pcffLammpsFixNhMassMask() != pcffFixNhMassNone;
}

bool pcffUseLammpsFixNhPressureMassMode()
{
    return (pcffLammpsFixNhMassMask() & pcffFixNhPressureMass) != 0;
}

double pcffReadPositiveEnvRealOrDefault(const char* name, const double defaultValue)
{
    double value = defaultValue;
    if (!readEnvReal(name, &value))
    {
        if (value <= 0)
        {
            gmx_fatal(FARGS, "%s is required and must be positive.", name);
        }
        return value;
    }
    if (value <= 0)
    {
        gmx_fatal(FARGS, "%s must be positive.", name);
    }
    return value;
}

double pcffReadPositiveEnvRealRequired(const char* name)
{
    return pcffReadPositiveEnvRealOrDefault(name, -1.0);
}

double pcffLammpsFixNhPdampPs(const t_inputrec& inputRecord)
{
    return pcffReadPositiveEnvRealOrDefault("GMX_PCFF_MTTK_LAMMPS_PDAMP_PS",
                                            inputRecord.pressureCouplingOptions.tau_p);
}

double pcffLammpsFixNhPressureMassScale()
{
    return pcffReadPositiveEnvRealOrDefault("GMX_PCFF_MTTK_PRESSURE_MASS_SCALE", 1.0);
}

struct PcffContinuousRefPressureRamp
{
    bool   active     = false;
    double startBar   = 0;
    double endBar     = 0;
    double durationPs = 0;
};

PcffContinuousRefPressureRamp pcffContinuousRefPressureRampFromEnv(const t_inputrec& inputRecord)
{
    PcffContinuousRefPressureRamp ramp;
    if (inputRecord.pressureCouplingOptions.epc == PressureCoupling::No)
    {
        return ramp;
    }

    const bool haveStart = readEnvReal("GMX_PCFF_REFP_RAMP_START_BAR", &ramp.startBar);
    const bool haveEnd   = readEnvReal("GMX_PCFF_REFP_RAMP_END_BAR", &ramp.endBar);
    if (haveStart != haveEnd)
    {
        gmx_fatal(FARGS,
                  "Both GMX_PCFF_REFP_RAMP_START_BAR and GMX_PCFF_REFP_RAMP_END_BAR "
                  "must be set for continuous reference-pressure ramping.");
    }
    if (!haveStart)
    {
        return ramp;
    }

    ramp.active     = true;
    ramp.durationPs = inputRecord.nsteps > 0 ? inputRecord.nsteps * inputRecord.delta_t : 0;
    readEnvReal("GMX_PCFF_REFP_RAMP_DURATION_PS", &ramp.durationPs);
    if (ramp.durationPs <= 0)
    {
        gmx_fatal(FARGS,
                  "GMX_PCFF_REFP_RAMP_DURATION_PS must be positive for continuous "
                  "reference-pressure ramping.");
    }
    return ramp;
}

PressureCouplingOptions pressureCouplingOptionsWithPcffContinuousRefPressureRamp(
        const PcffContinuousRefPressureRamp& ramp,
        const t_inputrec&              inputRecord,
        const PressureCouplingOptions& basePressureCouplingOptions,
        const int64_t                  step)
{
    if (!ramp.active)
    {
        return basePressureCouplingOptions;
    }
    if (basePressureCouplingOptions.epct != PressureCouplingType::Isotropic)
    {
        gmx_fatal(FARGS,
                  "GMX_PCFF_REFP_RAMP_* currently supports only isotropic pressure coupling.");
    }

    const double relativeTimePs =
            std::clamp((step - inputRecord.init_step) * inputRecord.delta_t, 0.0, ramp.durationPs);
    const double fraction = relativeTimePs / ramp.durationPs;
    const real   refPBar  = static_cast<real>(ramp.startBar + (ramp.endBar - ramp.startBar) * fraction);

    PressureCouplingOptions current = basePressureCouplingOptions;
    for (int d = 0; d < DIM; d++)
    {
        current.ref_p[d][d] = refPBar;
    }
    return current;
}

struct PcffMttkReferenceCellReset
{
    bool    active        = false;
    int64_t intervalSteps = 0;
};

PcffMttkReferenceCellReset pcffMttkReferenceCellResetFromEnv(const t_inputrec& inputRecord)
{
    PcffMttkReferenceCellReset reset;
    if (!readEnvInt64("GMX_PCFF_MTTK_NRESET_STEPS", &reset.intervalSteps))
    {
        return reset;
    }
    if (reset.intervalSteps <= 0)
    {
        gmx_fatal(FARGS, "GMX_PCFF_MTTK_NRESET_STEPS must be positive.");
    }
    if (inputRecord.pressureCouplingOptions.epc != PressureCoupling::Mttk)
    {
        gmx_fatal(FARGS, "GMX_PCFF_MTTK_NRESET_STEPS requires pcoupl = MTTK.");
    }
    if (inputRecord.pressureCouplingOptions.epct != PressureCouplingType::Isotropic)
    {
        gmx_fatal(FARGS, "GMX_PCFF_MTTK_NRESET_STEPS currently supports only isotropic MTTK.");
    }

    reset.active = true;
    return reset;
}

bool pcffExactRespaMttkOuterPcoupleEnabled()
{
    const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_MTTK_OUTER_PCOUPLE");
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}

bool pcffExactRespaMttkInlineBoxRemapEnabled()
{
    const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_MTTK_INLINE_BOX_REMAP");
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}

bool pcffMttkReferenceCellResetThisStep(const PcffMttkReferenceCellReset& reset,
                                        const t_inputrec&                 inputRecord,
                                        const int64_t                     step)
{
    if (!reset.active)
    {
        return false;
    }

    const int64_t relativeStep = step - inputRecord.init_step;
    return relativeStep > 0 && relativeStep % reset.intervalSteps == 0;
}

enum class PcffExactRespaTrotterReplay
{
    None,
    Two,
    Three,
    TwoThenThree,
    ThreeThenTwo,
};

PcffExactRespaTrotterReplay pcffExactRespaTrotterReplayFromEnv(
        const char* envName, const PcffExactRespaTrotterReplay defaultValue)
{
    const char* value = std::getenv(envName);
    if (value == nullptr || *value == '\0')
    {
        return defaultValue;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "none") == 0
        || std::strcmp(value, "skip") == 0)
    {
        return PcffExactRespaTrotterReplay::None;
    }
    if (std::strcmp(value, "2") == 0 || std::strcmp(value, "two") == 0)
    {
        return PcffExactRespaTrotterReplay::Two;
    }
    if (std::strcmp(value, "3") == 0 || std::strcmp(value, "three") == 0)
    {
        return PcffExactRespaTrotterReplay::Three;
    }
    if (std::strcmp(value, "2,3") == 0 || std::strcmp(value, "two-three") == 0)
    {
        return PcffExactRespaTrotterReplay::TwoThenThree;
    }
    if (std::strcmp(value, "3,2") == 0 || std::strcmp(value, "three-two") == 0)
    {
        return PcffExactRespaTrotterReplay::ThreeThenTwo;
    }
    gmx_fatal(FARGS,
              "%s must be one of: none, two, three, two-three, three-two.",
              envName);
}

bool pcffExactRespaTrotterSequenceCouplesThisStep(const t_inputrec&     inputRecord,
                                                  const int64_t         step,
                                                  const TrotterSequence sequence)
{
    if (inputRecord.exactRespa.enabled() && inputRecord.eI == IntegrationAlgorithm::VV
        && (inputrecNvtTrotter(&inputRecord) || inputrecNptTrotter(&inputRecord)
            || inputrecNphTrotter(&inputRecord)))
    {
        if (sequence == TrotterSequence::Two)
        {
            return inputRecord.nsttcouple == 1 || do_per_step(step, inputRecord.nsttcouple);
        }
        return inputRecord.nsttcouple == 1 || do_per_step(step + 1, inputRecord.nsttcouple);
    }

    const int64_t stepEff = (sequence <= TrotterSequence::Two) ? step - 1 : step;
    return inputRecord.nsttcouple == 1
           || do_per_step(stepEff + inputRecord.nsttcouple, inputRecord.nsttcouple);
}

bool pcffExactRespaTrotterReplayCouplesThisStep(const t_inputrec&                 inputRecord,
                                                const int64_t                     step,
                                                const PcffExactRespaTrotterReplay replay)
{
    switch (replay)
    {
        case PcffExactRespaTrotterReplay::None: return false;
        case PcffExactRespaTrotterReplay::Two:
            return pcffExactRespaTrotterSequenceCouplesThisStep(
                    inputRecord, step, TrotterSequence::Two);
        case PcffExactRespaTrotterReplay::Three:
            return pcffExactRespaTrotterSequenceCouplesThisStep(
                    inputRecord, step, TrotterSequence::Three);
        case PcffExactRespaTrotterReplay::TwoThenThree:
            return pcffExactRespaTrotterSequenceCouplesThisStep(
                           inputRecord, step, TrotterSequence::Two)
                   || pcffExactRespaTrotterSequenceCouplesThisStep(
                           inputRecord, step, TrotterSequence::Three);
        case PcffExactRespaTrotterReplay::ThreeThenTwo:
            return pcffExactRespaTrotterSequenceCouplesThisStep(
                           inputRecord, step, TrotterSequence::Three)
                   || pcffExactRespaTrotterSequenceCouplesThisStep(
                           inputRecord, step, TrotterSequence::Two);
    }
    return false;
}

void applyPcffMttkReferenceCellReset(const t_inputrec&      inputRecord,
                                     t_state*              state,
                                     t_extmass*            massQ,
                                     const gmx_ekindata_t& ekind)
{
    GMX_RELEASE_ASSERT(state != nullptr, "Need simulation state for MTTK reference-cell reset");
    GMX_RELEASE_ASSERT(massQ != nullptr, "Need extended-mass state for MTTK reference-cell reset");

    set_box_rel(&inputRecord, state);
    state->vol0 = det(state->box);
    if (state->vol0 <= 0)
    {
        gmx_fatal(FARGS, "Cannot reset MTTK reference volume from a non-positive box volume.");
    }

    // LAMMPS FixNH::compute_sigma() only resets the reference cell at nreset;
    // omega_mass/etap_mass stay fixed unless their explicit *_mass_flag is set.
    if (!pcffUseLammpsFixNhPressureMassMode())
    {
        massQ->Winv = (gmx::c_presfac * trace(inputRecord.pressureCouplingOptions.compress) * gmx::c_boltz
                       * ekind.currentEnsembleTemperature())
                      / (DIM * state->vol0 * gmx::square(inputRecord.pressureCouplingOptions.tau_p / M_2PI));
    }
}

bool pcffResetNhMttkStateOnStartEnabled()
{
    const char* value = std::getenv("GMX_PCFF_RESET_NH_MTTK_STATE_ON_START");
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}

const char* pcffRestoreNhMttkStateEnergyPath()
{
    const char* value = std::getenv("GMX_PCFF_RESTORE_NH_MTTK_STATE_FROM_EDR");
    return (value != nullptr && *value != '\0') ? value : nullptr;
}

const char* pcffRestoreNhMttkStateLammpsFixVector()
{
    const char* value = std::getenv("GMX_PCFF_RESTORE_NH_MTTK_STATE_FROM_LAMMPS_FIX_VECTOR");
    return (value != nullptr && *value != '\0') ? value : nullptr;
}

real pcffRestoreNhMttkStateEnergyTime(const t_inputrec& inputRecord)
{
    const char* value = std::getenv("GMX_PCFF_RESTORE_NH_MTTK_STATE_TIME_PS");
    return (value != nullptr && *value != '\0') ? std::atof(value) : inputRecord.init_t;
}

void resetPcffNhMttkStateOnStart(const t_inputrec& inputRecord, t_state* state)
{
    GMX_RELEASE_ASSERT(state != nullptr, "Need simulation state for NH/MTTK state reset");

    std::fill(state->nosehoover_xi.begin(), state->nosehoover_xi.end(), 0.0);
    std::fill(state->nosehoover_vxi.begin(), state->nosehoover_vxi.end(), 0.0);
    std::fill(state->nhpres_xi.begin(), state->nhpres_xi.end(), 0.0);
    std::fill(state->nhpres_vxi.begin(), state->nhpres_vxi.end(), 0.0);

    if (inputRecord.pressureCouplingOptions.epc == PressureCoupling::Mttk)
    {
        state->veta = 0;
        clear_mat(state->boxv);
        state->vol0 = det(state->box);
        if (state->vol0 <= 0)
        {
            gmx_fatal(FARGS, "Cannot reset MTTK start state from a non-positive box volume.");
        }
    }
}

std::vector<double> parsePcffLammpsFixNhVector(const char* value)
{
    std::string text(value != nullptr ? value : "");
    for (char& c : text)
    {
        if (c == ',' || c == ';')
        {
            c = ' ';
        }
    }

    std::istringstream stream(text);
    std::vector<double> values;
    double parsed = 0;
    while (stream >> parsed)
    {
        values.push_back(parsed);
    }
    if (!stream.eof())
    {
        gmx_fatal(FARGS,
                  "Invalid GMX_PCFF_RESTORE_NH_MTTK_STATE_FROM_LAMMPS_FIX_VECTOR='%s'. "
                  "Expected comma- or whitespace-separated numbers.",
                  value);
    }
    if (values.size() < 6)
    {
        gmx_fatal(FARGS,
                  "GMX_PCFF_RESTORE_NH_MTTK_STATE_FROM_LAMMPS_FIX_VECTOR requires at least "
                  "six LAMMPS FixNH values for eta[0:3] and eta_dot[0:3], got %zu.",
                  values.size());
    }
    return values;
}

void copyScaledPcffLammpsFixNhValues(std::vector<double>*     target,
                                     const std::vector<double>& values,
                                     const int                  sourceOffset,
                                     const int                  count,
                                     const double               scale,
                                     const char*                targetName)
{
    GMX_RELEASE_ASSERT(target != nullptr, "Need a target vector for FixNH state restore");
    if (target->size() < static_cast<size_t>(count))
    {
        gmx_fatal(FARGS,
                  "Cannot restore LAMMPS FixNH state into %s: need at least %d values, have %zu.",
                  targetName,
                  count,
                  target->size());
    }
    if (values.size() < static_cast<size_t>(sourceOffset + count))
    {
        gmx_fatal(FARGS,
                  "LAMMPS FixNH state vector is too short for %s: need source values through "
                  "f_1[%d], got %zu values.",
                  targetName,
                  sourceOffset + count,
                  values.size());
    }
    for (int i = 0; i < count; i++)
    {
        (*target)[i] = values[sourceOffset + i] * scale;
    }
}

void restorePcffNhMttkStateFromLammpsFixVector(const t_inputrec& inputRecord, t_state* state)
{
    const char* fixVector = pcffRestoreNhMttkStateLammpsFixVector();
    if (fixVector == nullptr)
    {
        return;
    }
    GMX_RELEASE_ASSERT(state != nullptr, "Need simulation state for LAMMPS FixNH restore");

    const std::vector<double> values = parsePcffLammpsFixNhVector(fixVector);
    // LAMMPS real units store eta_dot/omega_dot/etap_dot in fs^-1; GROMACS uses ps^-1.
    constexpr double lammpsFsInvToGromacsPsInv = 1000.0;
    copyScaledPcffLammpsFixNhValues(&state->nosehoover_xi, values, 0, 3, 1.0, "nosehoover_xi");
    copyScaledPcffLammpsFixNhValues(&state->nosehoover_vxi,
                                    values,
                                    3,
                                    3,
                                    lammpsFsInvToGromacsPsInv,
                                    "nosehoover_vxi");

    if (inputRecord.pressureCouplingOptions.epc == PressureCoupling::Mttk)
    {
        if (values.size() < 14)
        {
            gmx_fatal(FARGS,
                      "MTTK restore requires LAMMPS FixNH values f_1[1] through f_1[14], "
                      "got %zu values.",
                      values.size());
        }
        state->veta = values[7] * lammpsFsInvToGromacsPsInv;
        clear_mat(state->boxv);
        for (int d = 0; d < DIM; d++)
        {
            state->boxv[d][d] = state->veta * state->box[d][d];
        }
        state->vol0 = det(state->box);
        if (state->vol0 <= 0)
        {
            gmx_fatal(FARGS, "Cannot restore MTTK state from a non-positive box volume.");
        }
        copyScaledPcffLammpsFixNhValues(&state->nhpres_xi, values, 8, 3, 1.0, "nhpres_xi");
        copyScaledPcffLammpsFixNhValues(&state->nhpres_vxi,
                                        values,
                                        11,
                                        3,
                                        lammpsFsInvToGromacsPsInv,
                                        "nhpres_vxi");
    }
}

void restorePcffNhMttkStateFromEnergy(const t_inputrec&        inputRecord,
                                      const SimulationGroups& simulationGroups,
                                      t_state*                state)
{
    const char* energyPath = pcffRestoreNhMttkStateEnergyPath();
    if (energyPath == nullptr)
    {
        return;
    }
    get_enx_state(std::filesystem::path(energyPath),
                  pcffRestoreNhMttkStateEnergyTime(inputRecord),
                  simulationGroups,
                  const_cast<t_inputrec*>(&inputRecord),
                  state);
}

bool useExactLammpsRespaForceOnlyContract(const t_inputrec& inputRecord)
{
    const bool forceOnlySchedule =
            inputRecord.nsteps > 0 && inputRecord.nstcalcenergy > inputRecord.nsteps
            && inputRecord.nstenergy > inputRecord.nsteps;

    return useExactVelocityVerletLammpsRespa(inputRecord)
           && gmx::isSupportedExactRespaHybridNbGpuInput(inputRecord) && forceOnlySchedule;
}

bool nestedExactLammpsRespaPrototypeEnabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("GMX_EXACT_RESPA_NESTED_PROTOTYPE");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool exactRespaReuseNextForceForLiveStepEnabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_REUSE_NEXT_FORCE");
        if (value == nullptr || *value == '\0')
        {
            return true;
        }
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0
               && std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

bool exactRespaReuseNextForceWithGpuEnabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_REUSE_NEXT_FORCE_GPU");
        if (value == nullptr || *value == '\0')
        {
            return true;
        }
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0
               && std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

bool exactRespaResidentXGpuUpdateProbeEnabled()
{
    static const bool enabled =
            std::getenv("GMX_PCFF_EXACT_RESPA_GPU_RESIDENT_X_PROBE") != nullptr;
    return enabled;
}

bool exactRespaDeviceKickGpuUpdateProbeEnabled()
{
    static const bool enabled =
            std::getenv("GMX_PCFF_EXACT_RESPA_GPU_DEVICE_KICK_PROBE") != nullptr;
    return enabled;
}

bool exactRespaFusedNvtTrotterGpuUpdateProbeEnabled()
{
    static const bool enabled =
            std::getenv("GMX_PCFF_EXACT_RESPA_GPU_FUSED_NVT_TROTTER") != nullptr;
    return enabled;
}

bool exactRespaSparseNvtObservablesGpuUpdateProbeEnabled()
{
    static const bool enabled =
            std::getenv("GMX_PCFF_EXACT_RESPA_GPU_SPARSE_NVT_OBSERVABLES") != nullptr;
    return enabled;
}

bool exactRespaGpuNvtKineticReductionProbeEnabled()
{
    static const bool enabled =
            std::getenv("GMX_PCFF_EXACT_RESPA_GPU_NVT_KINETIC_REDUCTION") != nullptr;
    return enabled;
}

bool exactRespaGpuDeferNvtPostTrotterProbeEnabled()
{
    static const bool enabled =
            std::getenv("GMX_PCFF_EXACT_RESPA_GPU_DEFER_NVT_POST_TROTTER") != nullptr;
    return enabled;
}

bool exactRespaReuseNextForceOnNeighborSearchEnabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_REUSE_NEXT_FORCE_ON_NS");
        if (value == nullptr || *value == '\0')
        {
            return true;
        }
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0
               && std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

bool exactRespaReuseNextForceForLongRangeEnabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_REUSE_NEXT_FORCE_LONGRANGE");
        if (value == nullptr || *value == '\0')
        {
            return true;
        }
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0
               && std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

bool exactRespaSkipUnusedCombinedForceRestoreEnabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_SKIP_UNUSED_COMBINED_RESTORE");
        if (value == nullptr || *value == '\0')
        {
            return true;
        }
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0
               && std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

const char* exactRespaReuseDecisionTraceFilePath()
{
    static const char* path = []() -> const char* {
        const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_REUSE_TRACE_FILE");
        return (value != nullptr && *value != '\0') ? value : nullptr;
    }();
    return path;
}

int exactRespaReuseDecisionTraceMaxStep()
{
    static const int maxStep = [] {
        const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_REUSE_TRACE_MAX_STEP");
        return (value != nullptr && *value != '\0') ? std::atoi(value) : 16;
    }();
    return maxStep;
}

void appendExactRespaReuseDecisionTrace(const int64_t step,
                                        const bool    canReuse,
                                        const bool    reuseEnabled,
                                        const bool    supportedVvRespa,
                                        const bool    gpuForceWorkPresent,
                                        const bool    reuseWithGpu,
                                        const bool    useGpuUpdate,
                                        const bool    hasForceStore,
                                        const int     forceStoreLevels,
                                        const bool    hasLevel0,
                                        const int     longRangeLevel,
                                        const bool    hasLongRangeLevel,
                                        const bool    canReuseNeighborSearchForce,
                                        const bool    canReuseLongRangeForce,
                                        const bool    firstStep,
                                        const bool    bNS,
                                        const bool    bNStList,
                                        const bool    exchanged,
                                        const bool    needRepartition,
                                        const bool    haveDomainDecomposition,
                                        const bool    stepDoNeighborSearch,
                                        const bool    stepComputeEnergy,
                                        const bool    stepComputeVirial,
                                        const bool    stepComputeDhdl,
                                        const bool    stepComputeLongRange,
                                        const bool    calcEnergy,
                                        const bool    calcVirial,
                                        const bool    computeDhdl,
                                        const bool    stopCenterOfMass,
                                        const bool    globalStat,
                                        const bool    haveShellfc,
                                        const bool    haveAwh,
                                        const bool    haveEd,
                                        const bool    haveConstr,
                                        const bool    haveVirtualSites,
                                        const bool    haveM2pTrace,
                                        const bool    haveTotalForceDump,
                                        const bool    havePerLevelForceDump,
                                        const bool    haveMtsCombinedForceDump)
{
    const char* path = exactRespaReuseDecisionTraceFilePath();
    if (path == nullptr || step > exactRespaReuseDecisionTraceMaxStep())
    {
        return;
    }

    static std::mutex traceMutex;
    std::lock_guard<std::mutex> lock(traceMutex);

    const std::filesystem::path tracePath(path);
    if (!tracePath.parent_path().empty())
    {
        std::filesystem::create_directories(tracePath.parent_path());
    }
    const bool needHeader = !std::filesystem::exists(tracePath) || std::filesystem::file_size(tracePath) == 0;
    std::ofstream output(tracePath, std::ios::app);
    if (!output)
    {
        return;
    }
    if (needHeader)
    {
        output
                << "step can_reuse reuse_enabled supported_vv_respa gpu_force_work_present reuse_with_gpu"
                << " use_gpu_update has_force_store force_store_levels has_level0 long_range_level"
                << " has_long_range_level can_reuse_ns can_reuse_long_range first_step bNS bNStList"
                << " exchanged need_repartition have_dd step_do_ns step_energy step_virial step_dhdl"
                << " step_long_range calc_energy calc_virial compute_dhdl stop_cm gstat shellfc awh ed"
                << " constr virtual_sites m2p_trace total_force_dump per_level_force_dump mts_combined_dump\n";
    }

    const auto bit = [](const bool value) { return value ? 1 : 0; };
    output << step << ' ' << bit(canReuse) << ' ' << bit(reuseEnabled) << ' '
           << bit(supportedVvRespa) << ' ' << bit(gpuForceWorkPresent) << ' ' << bit(reuseWithGpu)
           << ' ' << bit(useGpuUpdate) << ' ' << bit(hasForceStore) << ' ' << forceStoreLevels
           << ' ' << bit(hasLevel0) << ' ' << longRangeLevel << ' ' << bit(hasLongRangeLevel)
           << ' ' << bit(canReuseNeighborSearchForce) << ' ' << bit(canReuseLongRangeForce)
           << ' ' << bit(firstStep) << ' ' << bit(bNS) << ' ' << bit(bNStList) << ' '
           << bit(exchanged) << ' ' << bit(needRepartition) << ' ' << bit(haveDomainDecomposition)
           << ' ' << bit(stepDoNeighborSearch) << ' ' << bit(stepComputeEnergy) << ' '
           << bit(stepComputeVirial) << ' ' << bit(stepComputeDhdl) << ' '
           << bit(stepComputeLongRange) << ' ' << bit(calcEnergy) << ' ' << bit(calcVirial)
           << ' ' << bit(computeDhdl) << ' ' << bit(stopCenterOfMass) << ' ' << bit(globalStat)
           << ' ' << bit(haveShellfc) << ' ' << bit(haveAwh) << ' ' << bit(haveEd) << ' '
           << bit(haveConstr) << ' ' << bit(haveVirtualSites) << ' ' << bit(haveM2pTrace)
           << ' ' << bit(haveTotalForceDump) << ' ' << bit(havePerLevelForceDump) << ' '
           << bit(haveMtsCombinedForceDump) << '\n';
}

const char* totalForceDumpFilePath()
{
    static const char* path = []() -> const char* {
        if (const char* value = std::getenv("GMX_TOTAL_FORCE_DUMP_FILE"))
        {
            return value;
        }
        return std::getenv("GMX_EXACT_RESPA_TOTAL_FORCE_DUMP_FILE");
    }();
    return path;
}

const char* perLevelForceDumpFilePath()
{
    static const char* path = std::getenv("GMX_EXACT_RESPA_PER_LEVEL_FORCE_DUMP_FILE");
    return path;
}

const char* mtsCombinedForceDumpFilePath()
{
    static const char* path = std::getenv("GMX_EXACT_RESPA_MTS_COMBINED_FORCE_DUMP_FILE");
    return path;
}

void restoreExactRespaForcesFromStore(const t_inputrec&                 inputRecord,
                                      const gmx::ExactRespaForceStore& exactRespaForceStore,
                                      gmx::ArrayRef<gmx::RVec>         physicalForce,
                                      gmx::ArrayRef<gmx::RVec>         combinedForce)
{
    GMX_RELEASE_ASSERT(exactRespaForceStore.hasLevel(0),
                       "Exact r-RESPA force reuse requires a stored fast force level");
    const gmx::ArrayRef<const gmx::RVec> fastForce = exactRespaForceStore.levelTotal(0);
    GMX_RELEASE_ASSERT(fastForce.size() == physicalForce.size(),
                       "Stored exact r-RESPA fast force should match the force buffer size");
    GMX_RELEASE_ASSERT(combinedForce.empty() || combinedForce.size() == physicalForce.size(),
                       "Exact r-RESPA combined force buffer should match the physical force size");

    const bool restoreCombinedForce = !combinedForce.empty();
    const bool skipUnusedCombinedForceRestore =
            !restoreCombinedForce && exactRespaSkipUnusedCombinedForceRestoreEnabled();
    if (skipUnusedCombinedForceRestore)
    {
        const int numLevels = exactRespaForceStore.numLevels();
        const auto storedSlowForce = [&exactRespaForceStore, &physicalForce](
                                             const int level) -> gmx::ArrayRef<const gmx::RVec> {
            if (!exactRespaForceStore.hasLevel(level))
            {
                return {};
            }
            const gmx::ArrayRef<const gmx::RVec> slowForce = exactRespaForceStore.levelTotal(level);
            GMX_RELEASE_ASSERT(slowForce.size() == physicalForce.size(),
                               "Stored exact r-RESPA slow force should match the force buffer size");
            return slowForce;
        };
        if (numLevels == 1)
        {
            std::memcpy(physicalForce.data(), fastForce.data(), physicalForce.size() * sizeof(gmx::RVec));
            return;
        }
        const auto slowForce1 = (numLevels > 1) ? storedSlowForce(1) : gmx::ArrayRef<const gmx::RVec>{};
        if (numLevels == 2 && !slowForce1.empty())
        {
            for (int atom = 0; atom < physicalForce.ssize(); ++atom)
            {
                physicalForce[atom][XX] = fastForce[atom][XX] + slowForce1[atom][XX];
                physicalForce[atom][YY] = fastForce[atom][YY] + slowForce1[atom][YY];
                physicalForce[atom][ZZ] = fastForce[atom][ZZ] + slowForce1[atom][ZZ];
            }
            return;
        }
        const auto slowForce2 = (numLevels > 2) ? storedSlowForce(2) : gmx::ArrayRef<const gmx::RVec>{};
        if (numLevels == 3 && !slowForce1.empty() && !slowForce2.empty())
        {
            for (int atom = 0; atom < physicalForce.ssize(); ++atom)
            {
                physicalForce[atom][XX] =
                        fastForce[atom][XX] + slowForce1[atom][XX] + slowForce2[atom][XX];
                physicalForce[atom][YY] =
                        fastForce[atom][YY] + slowForce1[atom][YY] + slowForce2[atom][YY];
                physicalForce[atom][ZZ] =
                        fastForce[atom][ZZ] + slowForce1[atom][ZZ] + slowForce2[atom][ZZ];
            }
            return;
        }

        std::array<gmx::ArrayRef<const gmx::RVec>, gmx::ExactRespaForceStore::c_numStoredLevels>
                slowForces;
        for (int level = 1; level < numLevels; ++level)
        {
            slowForces[level] = storedSlowForce(level);
        }
        for (int atom = 0; atom < physicalForce.ssize(); ++atom)
        {
            gmx::RVec restoredPhysical = fastForce[atom];
            for (int level = 1; level < numLevels; ++level)
            {
                if (slowForces[level].empty())
                {
                    continue;
                }
                restoredPhysical += slowForces[level][atom];
            }
            physicalForce[atom] = restoredPhysical;
        }
        return;
    }

    std::array<gmx::ArrayRef<const gmx::RVec>, gmx::ExactRespaForceStore::c_numStoredLevels> slowForces;
    std::array<real, gmx::ExactRespaForceStore::c_numStoredLevels> combinedFactors = {};
    for (int level = 1; level < exactRespaForceStore.numLevels(); ++level)
    {
        if (!exactRespaForceStore.hasLevel(level))
        {
            continue;
        }
        const gmx::ArrayRef<const gmx::RVec> slowForce = exactRespaForceStore.levelTotal(level);
        GMX_RELEASE_ASSERT(slowForce.size() == physicalForce.size(),
                           "Stored exact r-RESPA slow force should match the force buffer size");
        slowForces[level]      = slowForce;
        combinedFactors[level] = gmx::exactRespaLevelStepFactor(inputRecord, level);
    }

    for (int atom = 0; atom < physicalForce.ssize(); ++atom)
    {
        gmx::RVec restoredPhysical = fastForce[atom];
        gmx::RVec restoredCombined = fastForce[atom];
        for (int level = 1; level < exactRespaForceStore.numLevels(); ++level)
        {
            if (slowForces[level].empty())
            {
                continue;
            }
            restoredPhysical += slowForces[level][atom];
            restoredCombined += combinedFactors[level] * slowForces[level][atom];
        }
        physicalForce[atom] = restoredPhysical;
        if (restoreCombinedForce)
        {
            combinedForce[atom] = restoredCombined;
        }
    }
}

int canonicalAtomIndexForForceDumpAtom(const int atomIndex)
{
    if (atomIndex < 0)
    {
        return atomIndex;
    }
    if (gmx::g_respaLatestForceDumpGlobalAtomIndices != nullptr
        && atomIndex < gmx::g_respaLatestForceDumpGlobalAtomIndexCount)
    {
        return gmx::g_respaLatestForceDumpGlobalAtomIndices[atomIndex];
    }
    return atomIndex;
}

std::optional<int64_t> exactRespaForceDumpIntervalOverride()
{
    static const std::optional<int64_t> interval = []() -> std::optional<int64_t> {
        const char* value = std::getenv("GMX_EXACT_RESPA_FORCE_DUMP_INTERVAL");
        if (value == nullptr || *value == '\0')
        {
            return std::nullopt;
        }

        char*      end    = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed <= 0)
        {
            return std::nullopt;
        }

        return static_cast<int64_t>(parsed);
    }();
    return interval;
}

bool shouldDumpExactRespaForceDiagnostics(const t_inputrec& inputRecord, const int64_t step)
{
    if (const auto interval = exactRespaForceDumpIntervalOverride())
    {
        return (step % *interval) == 0;
    }
    return do_per_step(step, inputRecord.nstenergy);
}

void appendExactRespaTotalForceRecord(const char*                        outputPath,
                                      const int64_t                      step,
                                      const real                         time,
                                      const int                          highestActiveMtsLevel,
                                      const gmx::ArrayRef<const gmx::RVec> totalForce)
{
    GMX_RELEASE_ASSERT(outputPath != nullptr && *outputPath != '\0', "Need a valid force dump path");
    std::filesystem::path path(outputPath);
    std::filesystem::create_directories(path.parent_path());

    std::ofstream output(path, std::ios::app);
    output << std::setprecision(17);
    for (gmx::Index atom = 0; atom < gmx::ssize(totalForce); ++atom)
    {
        output << step << '\t' << time << '\t' << highestActiveMtsLevel << '\t' << atom << '\t'
               << canonicalAtomIndexForForceDumpAtom(atom);
        for (int d = 0; d < DIM; ++d)
        {
            output << '\t' << totalForce[atom][d];
        }
        output << '\n';
    }
}

void appendExactRespaPerLevelForceRecord(const char*                        outputPath,
                                         const int64_t                      step,
                                         const real                         time,
                                         const int                          highestActiveMtsLevel,
                                         const int                          mtsLevel,
                                         const gmx::ArrayRef<const gmx::RVec> levelForce)
{
    GMX_RELEASE_ASSERT(outputPath != nullptr && *outputPath != '\0', "Need a valid per-level force dump path");
    std::filesystem::path path(outputPath);
    std::filesystem::create_directories(path.parent_path());

    std::ofstream output(path, std::ios::app);
    output << std::setprecision(17);
    for (gmx::Index atom = 0; atom < gmx::ssize(levelForce); ++atom)
    {
        output << step << '\t' << time << '\t' << highestActiveMtsLevel << '\t' << mtsLevel << '\t' << atom
               << '\t' << canonicalAtomIndexForForceDumpAtom(atom);
        for (int d = 0; d < DIM; ++d)
        {
            output << '\t' << levelForce[atom][d];
        }
        output << '\n';
    }
}

void maybeDumpTotalForceForDiagnostics(const t_inputrec&                 inputRecord,
                                       const int64_t                     step,
                                       const real                        time,
                                       gmx::ForceBuffersView*            forceView,
                                       const gmx::MdrunScheduleWorkload& runScheduleWork)
{
    const char* outputPath = totalForceDumpFilePath();
    if (outputPath == nullptr || *outputPath == '\0'
        || !shouldDumpExactRespaForceDiagnostics(inputRecord, step))
    {
        return;
    }

    if (gmx::useMtsSubstepping(inputRecord))
    {
        return;
    }

    if (gmx::useExactRespa(inputRecord))
    {
        const int highestActiveLevel =
                runScheduleWork.exactRespaStepWork.highestActiveLevel;
        if (highestActiveLevel <= 0)
        {
            return;
        }
        appendExactRespaTotalForceRecord(
                outputPath, step, time, highestActiveLevel, gmx::makeConstArrayRef(forceView->force()));
        return;
    }

    appendExactRespaTotalForceRecord(outputPath,
                                     step,
                                     time,
                                     0,
                                     gmx::makeConstArrayRef(forceView->force()));
}

void maybeDumpPerLevelForceForDiagnostics(const t_inputrec&                 inputRecord,
                                          const int64_t                     step,
                                          const real                        time,
                                          gmx::ForceBuffersView*            forceView,
                                          const gmx::ExactRespaForceStore*  exactRespaForceStore,
                                          const gmx::MdrunScheduleWorkload& runScheduleWork)
{
    const char* outputPath = perLevelForceDumpFilePath();
    if (outputPath == nullptr || *outputPath == '\0'
        || !shouldDumpExactRespaForceDiagnostics(inputRecord, step))
    {
        return;
    }

    if (!gmx::useExactRespa(inputRecord) || gmx::useMtsSubstepping(inputRecord))
    {
        return;
    }

    const int highestActiveLevel = runScheduleWork.exactRespaStepWork.highestActiveLevel;
    if (highestActiveLevel <= 0)
    {
        return;
    }

    GMX_RELEASE_ASSERT(forceView != nullptr, "Need exact r-RESPA force buffers for diagnostic dumps");
    GMX_RELEASE_ASSERT(exactRespaForceStore != nullptr, "Need exact r-RESPA force store for per-level dumps");
    GMX_RELEASE_ASSERT(exactRespaForceStore->hasLevel(0),
                       "Exact r-RESPA per-level force dump requires stored level totals");
    appendExactRespaPerLevelForceRecord(outputPath,
                                        step,
                                        time,
                                        highestActiveLevel,
                                        0,
                                        exactRespaForceStore->levelTotal(0));
    for (int mtsLevel = 1; mtsLevel <= highestActiveLevel; ++mtsLevel)
    {
        GMX_RELEASE_ASSERT(exactRespaForceStore->hasLevel(mtsLevel),
                           "Exact r-RESPA per-level force dump requires stored level totals");
        appendExactRespaPerLevelForceRecord(outputPath,
                                            step,
                                            time,
                                            highestActiveLevel,
                                            mtsLevel,
                                            exactRespaForceStore->levelTotal(mtsLevel));
    }
}

void maybeDumpMtsCombinedForceForDiagnostics(const t_inputrec&                 inputRecord,
                                             const int64_t                     step,
                                             const real                        time,
                                             gmx::ForceBuffersView*            forceView,
                                             const gmx::MdrunScheduleWorkload& runScheduleWork)
{
    const char* outputPath = mtsCombinedForceDumpFilePath();
    if (outputPath == nullptr || *outputPath == '\0'
        || !shouldDumpExactRespaForceDiagnostics(inputRecord, step))
    {
        return;
    }

    if (!gmx::useExactRespa(inputRecord) || gmx::useMtsSubstepping(inputRecord))
    {
        return;
    }

    GMX_RELEASE_ASSERT(forceView != nullptr, "Need exact r-RESPA force buffers for diagnostic dumps");
    const gmx::ArrayRef<const gmx::RVec> combinedForce = forceView->forceMtsCombined();
    if (combinedForce.empty())
    {
        return;
    }

    appendExactRespaTotalForceRecord(outputPath,
                                     step,
                                     time,
                                     runScheduleWork.exactRespaStepWork.highestActiveLevel,
                                     combinedForce);
}

struct Tp18eTraceConfig
{
    bool        enabled = false;
    std::string path;
};

std::mutex g_tp18eTraceMutex;

const Tp18eTraceConfig& tp18eTraceConfig()
{
    static const Tp18eTraceConfig config = []()
    {
        Tp18eTraceConfig result;
        if (const char* value = std::getenv("GMX_TP18E_TRACE_FILE"))
        {
            if (*value != '\0')
            {
                result.enabled = true;
                result.path    = value;

                std::filesystem::path tracePath(result.path);
                std::filesystem::create_directories(tracePath.parent_path());
                std::ofstream output(tracePath, std::ios::trunc);
                GMX_RELEASE_ASSERT(output.good(), "Could not open GMX_TP18E_TRACE_FILE for writing");
                output << "step,stage,using_mts_combined_force,force_l2,force_max_abs,"
                          "state_x_l2,state_x_max_abs,state_v_l2,state_v_max_abs,"
                          "xprime_available,xprime_l2,xprime_max_abs,"
                          "force_vir_trace,shake_vir_trace,total_vir_trace,pres_trace,"
                          "potential_energy_kj,kinetic_energy_kj,temperature_k\n";
            }
        }
        return result;
    }();

    return config;
}

std::pair<double, double> tp18eSummarizeBuffer(gmx::ArrayRef<const gmx::RVec> values)
{
    double squaredNormSum = 0.0;
    double maxAbs         = 0.0;

    for (const auto& value : values)
    {
        for (int d = 0; d < DIM; ++d)
        {
            const double component = value[d];
            squaredNormSum += component * component;
            maxAbs = std::max(maxAbs, std::abs(component));
        }
    }

    return { std::sqrt(squaredNormSum), maxAbs };
}

double tp18eTraceOfTensor(const tensor value)
{
    double trace = 0.0;
    for (int d = 0; d < DIM; ++d)
    {
        trace += value[d][d];
    }
    return trace;
}

void appendTp18eTraceRow(const int64_t                   step,
                         const char*                     stage,
                         const gmx::ArrayRef<const gmx::RVec> force,
                         const bool                      usingMtsCombinedForce,
                         const gmx::ArrayRef<const gmx::RVec> stateX,
                         const gmx::ArrayRef<const gmx::RVec> stateV,
                         const gmx::ArrayRef<const gmx::RVec> xPrime,
                         const bool                      haveXPrime,
                         const tensor                    forceVir,
                         const tensor                    shakeVir,
                         const tensor                    totalVir,
                         const tensor                    pres,
                         const gmx_enerdata_t*           enerd)
{
    const auto& config = tp18eTraceConfig();
    if (!config.enabled)
    {
        return;
    }

    const auto [forceL2, forceMaxAbs]     = tp18eSummarizeBuffer(force);
    const auto [stateXL2, stateXMaxAbs]   = tp18eSummarizeBuffer(stateX);
    const auto [stateVL2, stateVMaxAbs]   = tp18eSummarizeBuffer(stateV);
    const auto [xPrimeL2, xPrimeMaxAbs]   = haveXPrime ? tp18eSummarizeBuffer(xPrime) : std::pair<double, double>{ 0.0, 0.0 };
    const double forceVirTrace            = tp18eTraceOfTensor(forceVir);
    const double shakeVirTrace            = tp18eTraceOfTensor(shakeVir);
    const double totalVirTrace            = tp18eTraceOfTensor(totalVir);
    const double presTrace                = tp18eTraceOfTensor(pres);
    const double potentialEnergy          = (enerd != nullptr) ? enerd->term[InteractionFunction::PotentialEnergy] : 0.0;
    const double kineticEnergy            = (enerd != nullptr) ? enerd->term[InteractionFunction::KineticEnergy] : 0.0;
    const double temperature              = (enerd != nullptr) ? enerd->term[InteractionFunction::Temperature] : 0.0;

    std::lock_guard<std::mutex> lock(g_tp18eTraceMutex);
    std::ofstream               output(config.path, std::ios::app);
    GMX_RELEASE_ASSERT(output.good(), "Could not open GMX_TP18E_TRACE_FILE for appending");
    output << std::setprecision(17) << step << ',' << stage << ',' << static_cast<int>(usingMtsCombinedForce) << ','
           << forceL2 << ',' << forceMaxAbs << ','
           << stateXL2 << ',' << stateXMaxAbs << ','
           << stateVL2 << ',' << stateVMaxAbs << ','
           << static_cast<int>(haveXPrime) << ',' << xPrimeL2 << ',' << xPrimeMaxAbs << ','
           << forceVirTrace << ',' << shakeVirTrace << ',' << totalVirTrace << ',' << presTrace << ','
           << potentialEnergy << ',' << kineticEnergy << ',' << temperature << '\n';
}

struct Tp18gTraceConfig
{
    bool        enabled = false;
    std::string path;
};

std::mutex g_tp18gTraceMutex;

const Tp18gTraceConfig& tp18gTraceConfig()
{
    static const Tp18gTraceConfig config = []()
    {
        Tp18gTraceConfig result;
        if (const char* value = std::getenv("GMX_TP18G_TRACE_FILE"))
        {
            if (*value != '\0')
            {
                result.enabled = true;
                result.path    = value;

                std::filesystem::path tracePath(result.path);
                std::filesystem::create_directories(tracePath.parent_path());
                std::ofstream output(tracePath, std::ios::trunc);
                GMX_RELEASE_ASSERT(output.good(), "Could not open GMX_TP18G_TRACE_FILE for writing");
                output << "step,time_ps,stage,b_calc_ener,b_calc_ener_step,"
                          "pressure_coupling_is_no,pressure_coupling_consumer_active,"
                          "has_pressure_previous,pressure_previous_copy_executed,"
                          "energy_add_called,record_nonenergy_called,energy_print_called,"
                          "total_vir_trace,total_vir_l2,pres_trace,pres_l2,pressure_scalar,"
                          "potential_energy_kj,kinetic_energy_kj,temperature_k\n";
            }
        }
        return result;
    }();

    return config;
}

double tp18gL2OfTensor(const tensor value)
{
    double squaredNormSum = 0.0;
    for (int i = 0; i < DIM; ++i)
    {
        for (int j = 0; j < DIM; ++j)
        {
            const double component = value[i][j];
            squaredNormSum += component * component;
        }
    }
    return std::sqrt(squaredNormSum);
}

bool tp18gPressureCouplingConsumerActive(const t_inputrec* inputrec, const int64_t step)
{
    switch (inputrec->pressureCouplingOptions.epc)
    {
        case PressureCoupling::No: return false;
        case PressureCoupling::Berendsen:
        case PressureCoupling::CRescale:
            return do_per_step(step, inputrec->pressureCouplingOptions.nstpcouple);
        case PressureCoupling::ParrinelloRahman:
            return do_per_step(step + inputrec->pressureCouplingOptions.nstpcouple - 1,
                               inputrec->pressureCouplingOptions.nstpcouple);
        case PressureCoupling::Mttk: return true;
        default: return false;
    }
}

void appendTp18gTraceRow(const int64_t         step,
                         const double          time,
                         const char*           stage,
                         const bool            bCalcEner,
                         const bool            bCalcEnerStep,
                         const t_inputrec*     inputrec,
                         const bool            hasPressurePrevious,
                         const bool            pressurePreviousCopyExecuted,
                         const bool            energyAddCalled,
                         const bool            recordNonEnergyCalled,
                         const bool            energyPrintCalled,
                         const tensor          totalVir,
                         const tensor          pres,
                         const gmx_enerdata_t* enerd)
{
    const auto& config = tp18gTraceConfig();
    if (!config.enabled)
    {
        return;
    }

    const bool   pressureCouplingIsNo       = (inputrec->pressureCouplingOptions.epc == PressureCoupling::No);
    const bool   pressureCouplingActive     = tp18gPressureCouplingConsumerActive(inputrec, step);
    const double totalVirTrace              = tp18eTraceOfTensor(totalVir);
    const double totalVirL2                 = tp18gL2OfTensor(totalVir);
    const double presTrace                  = tp18eTraceOfTensor(pres);
    const double presL2                     = tp18gL2OfTensor(pres);
    const double pressureScalar             = (enerd != nullptr) ? enerd->term[InteractionFunction::Pressure] : 0.0;
    const double potentialEnergy            = (enerd != nullptr) ? enerd->term[InteractionFunction::PotentialEnergy] : 0.0;
    const double kineticEnergy              = (enerd != nullptr) ? enerd->term[InteractionFunction::KineticEnergy] : 0.0;
    const double temperature                = (enerd != nullptr) ? enerd->term[InteractionFunction::Temperature] : 0.0;

    std::lock_guard<std::mutex> lock(g_tp18gTraceMutex);
    std::ofstream               output(config.path, std::ios::app);
    GMX_RELEASE_ASSERT(output.good(), "Could not open GMX_TP18G_TRACE_FILE for appending");
    output << std::setprecision(17) << step << ',' << time << ',' << stage << ','
           << static_cast<int>(bCalcEner) << ',' << static_cast<int>(bCalcEnerStep) << ','
           << static_cast<int>(pressureCouplingIsNo) << ',' << static_cast<int>(pressureCouplingActive) << ','
           << static_cast<int>(hasPressurePrevious) << ',' << static_cast<int>(pressurePreviousCopyExecuted) << ','
           << static_cast<int>(energyAddCalled) << ',' << static_cast<int>(recordNonEnergyCalled) << ','
           << static_cast<int>(energyPrintCalled) << ','
           << totalVirTrace << ',' << totalVirL2 << ','
           << presTrace << ',' << presL2 << ','
           << pressureScalar << ',' << potentialEnergy << ',' << kineticEnergy << ',' << temperature << '\n';
}

struct PcffMttkStateTraceConfig
{
    bool        enabled = false;
    std::string path;
    int64_t     stride = 1;
};

std::mutex g_pcffMttkStateTraceMutex;

int64_t pcffReadPositiveEnvInt64OrDefault(const char* name, const int64_t defaultValue)
{
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0')
    {
        return defaultValue;
    }
    char*           end    = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (end == text || (end != nullptr && *end != '\0') || parsed <= 0)
    {
        gmx_fatal(FARGS, "%s must be a positive integer.", name);
    }
    return static_cast<int64_t>(parsed);
}

const PcffMttkStateTraceConfig& pcffMttkStateTraceConfig()
{
    static const PcffMttkStateTraceConfig config = []()
    {
        PcffMttkStateTraceConfig result;
        if (const char* value = std::getenv("GMX_PCFF_MTTK_STATE_TRACE_FILE"))
        {
            if (*value != '\0')
            {
                result.enabled = true;
                result.path    = value;
                result.stride  = pcffReadPositiveEnvInt64OrDefault(
                        "GMX_PCFF_MTTK_STATE_TRACE_STRIDE", 1);

                std::filesystem::path tracePath(result.path);
                std::filesystem::create_directories(tracePath.parent_path());
                std::ofstream output(tracePath, std::ios::trunc);
                GMX_RELEASE_ASSERT(output.good(),
                                   "Could not open GMX_PCFF_MTTK_STATE_TRACE_FILE for writing");
                output << "step,time_ps,stage,veta,volume_nm3,box_x,box_y,box_z,"
                          "boxv_x,boxv_y,boxv_z,ref_p_bar,pressure_bar,"
                          "pres_trace,pres_l2,total_vir_trace,total_vir_l2,"
                          "potential_energy_kj,kinetic_energy_kj,temperature_k,"
                          "nose_xi0,nose_xi1,nose_xi2,nose_vxi0,nose_vxi1,nose_vxi2,"
                          "nhpres_xi0,nhpres_xi1,nhpres_xi2,"
                          "nhpres_vxi0,nhpres_vxi1,nhpres_vxi2\n";
            }
        }
        return result;
    }();

    return config;
}

double pcffMttkTraceArrayValue(const std::vector<double>& values, const int index)
{
    return (index >= 0 && index < static_cast<int>(values.size())) ? values[index] : 0.0;
}

void appendPcffMttkStateTraceRow(const int64_t         step,
                                 const double          time,
                                 const char*           stage,
                                 const t_inputrec&     inputrec,
                                 const t_state*        state,
                                 const tensor          pres,
                                 const tensor          totalVir,
                                 const gmx_enerdata_t* enerd)
{
    const auto& config = pcffMttkStateTraceConfig();
    if (!config.enabled || state == nullptr || step % config.stride != 0)
    {
        return;
    }

    const double volume          = det(state->box);
    const double refPressure     = tp18eTraceOfTensor(inputrec.pressureCouplingOptions.ref_p) / DIM;
    const double pressureScalar  = (enerd != nullptr) ? enerd->term[InteractionFunction::Pressure] : 0.0;
    const double potentialEnergy = (enerd != nullptr) ? enerd->term[InteractionFunction::PotentialEnergy] : 0.0;
    const double kineticEnergy   = (enerd != nullptr) ? enerd->term[InteractionFunction::KineticEnergy] : 0.0;
    const double temperature     = (enerd != nullptr) ? enerd->term[InteractionFunction::Temperature] : 0.0;

    std::lock_guard<std::mutex> lock(g_pcffMttkStateTraceMutex);
    std::ofstream               output(config.path, std::ios::app);
    GMX_RELEASE_ASSERT(output.good(), "Could not open GMX_PCFF_MTTK_STATE_TRACE_FILE for appending");
    output << std::setprecision(17) << step << ',' << time << ',' << stage << ','
           << state->veta << ',' << volume << ','
           << state->box[XX][XX] << ',' << state->box[YY][YY] << ',' << state->box[ZZ][ZZ] << ','
           << state->boxv[XX][XX] << ',' << state->boxv[YY][YY] << ',' << state->boxv[ZZ][ZZ] << ','
           << refPressure << ',' << pressureScalar << ','
           << tp18eTraceOfTensor(pres) << ',' << tp18gL2OfTensor(pres) << ','
           << tp18eTraceOfTensor(totalVir) << ',' << tp18gL2OfTensor(totalVir) << ','
           << potentialEnergy << ',' << kineticEnergy << ',' << temperature << ','
           << pcffMttkTraceArrayValue(state->nosehoover_xi, 0) << ','
           << pcffMttkTraceArrayValue(state->nosehoover_xi, 1) << ','
           << pcffMttkTraceArrayValue(state->nosehoover_xi, 2) << ','
           << pcffMttkTraceArrayValue(state->nosehoover_vxi, 0) << ','
           << pcffMttkTraceArrayValue(state->nosehoover_vxi, 1) << ','
           << pcffMttkTraceArrayValue(state->nosehoover_vxi, 2) << ','
           << pcffMttkTraceArrayValue(state->nhpres_xi, 0) << ','
           << pcffMttkTraceArrayValue(state->nhpres_xi, 1) << ','
           << pcffMttkTraceArrayValue(state->nhpres_xi, 2) << ','
           << pcffMttkTraceArrayValue(state->nhpres_vxi, 0) << ','
           << pcffMttkTraceArrayValue(state->nhpres_vxi, 1) << ','
           << pcffMttkTraceArrayValue(state->nhpres_vxi, 2) << '\n';
}

bool canUseNestedExactLammpsRespa(const t_inputrec&                  inputRecord,
                                  const gmx::SimulationWorkload&     simulationWork,
                                  const gmx::DomainLifetimeWorkload& domainWork,
                                  const gmx_shellfc_t*               shellfc,
                                  gmx::Constraints*                  constr,
                                  const gmx::VirtualSitesHandler*    virtualSites,
                                  const bool                         useReplicaExchange)
{
    return inputRecord.eI == IntegrationAlgorithm::MD && inputRecord.etc == TemperatureCoupling::No
           && inputRecord.pressureCouplingOptions.epc == PressureCoupling::No
           && inputRecord.comm_mode == ComRemovalAlgorithm::No
           && (constr == nullptr || constr->numConstraintsTotal() == 0)
           && !simulationWork.havePpDomainDecomposition && !simulationWork.useGpuUpdate
           && !simulationWork.useGpuNonbonded && !simulationWork.useGpuPme && shellfc == nullptr
           && virtualSites == nullptr && !domainWork.haveSpecialForces && !useReplicaExchange
           && inputRecord.cos_accel == 0.0 && !inputRecord.useConstantAcceleration;
}

bool pcffExactRespaAllowLinearComRemoval()
{
    const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_ALLOW_LINEAR_COM_REMOVAL");
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0
           && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0;
}

bool exactVelocityVerletRespaSupportsComRemoval(const t_inputrec& inputRecord)
{
    return inputRecord.comm_mode == ComRemovalAlgorithm::No
           || (inputRecord.comm_mode == ComRemovalAlgorithm::Linear
               && pcffExactRespaAllowLinearComRemoval());
}

bool canUseExactLammpsRespaVelocityVerlet(const t_inputrec&                  inputRecord,
                                          const gmx::SimulationWorkload&     simulationWork,
                                          const gmx::DomainLifetimeWorkload& domainWork,
                                          const gmx_shellfc_t*               shellfc,
                                          gmx::Constraints*                  constr,
                                          const gmx::VirtualSitesHandler*    virtualSites,
                                          const bool                         useReplicaExchange)
{
    const bool usesSupportedExactRespaTemperatureCoupling =
            inputRecord.etc == TemperatureCoupling::No
            || inputRecord.etc == TemperatureCoupling::Berendsen
            || inputRecord.etc == TemperatureCoupling::VRescale
            || inputRecord.etc == TemperatureCoupling::NoseHoover;
    const bool usesSupportedExactRespaPressureCoupling =
            inputRecord.pressureCouplingOptions.epc == PressureCoupling::No
            || inputRecord.pressureCouplingOptions.epc == PressureCoupling::Berendsen
            || inputRecord.pressureCouplingOptions.epc == PressureCoupling::CRescale
            || inputRecord.pressureCouplingOptions.epc == PressureCoupling::Mttk;
    const bool usesSupportedExactRespaGpuForces =
            (!simulationWork.useGpuBonded || simulationWork.useGpuNonbonded)
            && (!simulationWork.useGpuPme
                || (simulationWork.useGpuNonbonded && simulationWork.useGpuBonded))
            && (!simulationWork.useGpuNonbonded
                || (!simulationWork.useGpuPme && !simulationWork.useGpuBonded)
                || (simulationWork.useGpuBonded && !simulationWork.useGpuPme)
                || (simulationWork.useGpuBonded && simulationWork.useGpuPme));
    const bool usesSupportedExactRespaGpuUpdate =
            !simulationWork.useGpuUpdate
            || (simulationWork.useGpuNonbonded && simulationWork.useGpuBonded
                && simulationWork.useGpuPme && !simulationWork.useMdGpuGraph);

    return inputRecord.eI == IntegrationAlgorithm::VV && usesSupportedExactRespaTemperatureCoupling
           && usesSupportedExactRespaPressureCoupling
           && exactVelocityVerletRespaSupportsComRemoval(inputRecord)
           && (constr == nullptr || constr->numConstraintsTotal() == 0)
           && !simulationWork.havePpDomainDecomposition && usesSupportedExactRespaGpuForces
           && usesSupportedExactRespaGpuUpdate && shellfc == nullptr
           && virtualSites == nullptr && !domainWork.haveSpecialForces && !useReplicaExchange
           && inputRecord.cos_accel == 0.0 && !inputRecord.useConstantAcceleration;
}

static const char* activeM2pTraceDirPath()
{
    static const char* traceDir = [] {
        const char* value = std::getenv("GMX_PCFF_RESPA_M2P_TRACE_DIR");
        return (value != nullptr && *value != '\0') ? value : nullptr;
    }();
    return traceDir;
}

static const std::vector<int>& configuredM2pTraceAtomIndices()
{
    static const std::vector<int> atomIndices = []() {
        std::vector<int> parsedAtomIndices;
        const char*      value = std::getenv("GMX_PCFF_RESPA_TRACE_ATOMS");
        if (value != nullptr && *value != '\0')
        {
            std::stringstream ss(value);
            std::string       item;
            while (std::getline(ss, item, ','))
            {
                if (item.empty())
                {
                    continue;
                }

                try
                {
                    size_t    endPos    = 0;
                    const int atomIndex = std::stoi(item, &endPos);
                    if (endPos == item.size() && atomIndex >= 0)
                    {
                        parsedAtomIndices.push_back(atomIndex);
                    }
                }
                catch (const std::exception&)
                {
                }
            }
        }

        if (parsedAtomIndices.empty())
        {
            return std::vector<int>{ 0, 5 };
        }
        return parsedAtomIndices;
    }();

    return atomIndices;
}

static std::vector<int> filteredM2pTraceAtomIndices(const int availableAtoms)
{
    std::vector<int> filteredAtomIndices;
    for (const int atomIndex : configuredM2pTraceAtomIndices())
    {
        if (atomIndex >= 0 && atomIndex < availableAtoms)
        {
            filteredAtomIndices.push_back(atomIndex);
        }
    }
    return filteredAtomIndices;
}

static bool skipDdPartitionForExactRespaExperiment()
{
    const char* value = std::getenv("GMX_PCFF_RESPA_EXPERIMENT_SKIP_DD_PARTITION");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

static void appendPreDoForceStateTrace(const char*                   traceDirPath,
                                       const int64_t                 step,
                                       const char*                   contextLabel,
                                       gmx::ArrayRef<const gmx::RVec> coordinates,
                                       gmx::ArrayRef<const gmx::RVec> velocities,
                                       const char*                   codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::ofstream output(traceDir / "predoforce_state_trace.txt", std::ios::app);
    const auto filteredAtoms = filteredM2pTraceAtomIndices(coordinates.ssize());
    for (const int atomIndex : filteredAtoms)
    {
        output << "step=" << step << " context_label="
               << ((contextLabel != nullptr && *contextLabel != '\0') ? contextLabel : "unspecified")
               << " atom=" << atomIndex << " px=" << gmx::formatString("%.15f", coordinates[atomIndex][XX])
               << " py=" << gmx::formatString("%.15f", coordinates[atomIndex][YY]) << " pz="
               << gmx::formatString("%.15f", coordinates[atomIndex][ZZ]) << " vx="
               << gmx::formatString("%.15f", velocities[atomIndex][XX]) << " vy="
               << gmx::formatString("%.15f", velocities[atomIndex][YY]) << " vz="
               << gmx::formatString("%.15f", velocities[atomIndex][ZZ]) << " code_location="
               << codeLocation << '\n';
    }
}

static void appendMdLoopBoundarySnapshotPair(const char*                   traceDirPath,
                                             const char*                   side,
                                             const char*                   stageName,
                                             const int64_t                 step,
                                             gmx::ArrayRef<const gmx::RVec> coordinates,
                                             const char*                   codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    const auto filteredAtoms = filteredM2pTraceAtomIndices(coordinates.ssize());
    if (filteredAtoms.empty())
    {
        return;
    }

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::ofstream output(traceDir / "multistep_md_loop_boundary_trace.txt", std::ios::app);
    for (const int atomIndex : filteredAtoms)
    {
        output << "side=" << side << " step=" << step << " stage=" << stageName << " atom=" << atomIndex
               << " x=" << std::setprecision(15) << coordinates[atomIndex][XX] << " y="
               << std::setprecision(15) << coordinates[atomIndex][YY] << " z=" << std::setprecision(15)
               << coordinates[atomIndex][ZZ] << " code_location=" << codeLocation
               << " snapshot_type=md_loop_boundary\n";
    }
}

static void appendLocalGlobalStateAliasTrace(const char* traceDirPath,
                                             const int64_t step,
                                             const char* contextLabel,
                                             const t_state* localState,
                                             const t_state* globalState,
                                             const char* codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0' || localState == nullptr || globalState == nullptr)
    {
        return;
    }

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::ofstream output(traceDir / "local_global_state_alias_trace.txt", std::ios::app);
    output << "step=" << step << " context_label="
           << ((contextLabel != nullptr && *contextLabel != '\0') ? contextLabel : "unspecified")
           << " local_state_ptr=" << static_cast<const void*>(localState)
           << " global_state_ptr=" << static_cast<const void*>(globalState)
           << " local_x_ptr=" << static_cast<const void*>(localState->x.data())
           << " global_x_ptr=" << static_cast<const void*>(globalState->x.data())
           << " local_v_ptr=" << static_cast<const void*>(localState->v.data())
           << " global_v_ptr=" << static_cast<const void*>(globalState->v.data())
           << " local_atoms=" << localState->numAtoms() << " global_atoms=" << globalState->numAtoms()
           << " code_location=" << codeLocation << '\n';

    const auto localCoords  = localState->x.constArrayRefWithPadding().unpaddedConstArrayRef();
    const auto localVels    = localState->v.constArrayRefWithPadding().unpaddedConstArrayRef();
    const auto globalCoords = globalState->x.constArrayRefWithPadding().unpaddedConstArrayRef();
    const auto globalVels   = globalState->v.constArrayRefWithPadding().unpaddedConstArrayRef();
    const int availableAtoms =
            std::min<int>(std::min(localCoords.ssize(), globalCoords.ssize()),
                          std::min(localVels.ssize(), globalVels.ssize()));
    const auto filteredAtoms = filteredM2pTraceAtomIndices(availableAtoms);
    for (const int atomIndex : filteredAtoms)
    {
        output << "step=" << step << " context_label="
               << ((contextLabel != nullptr && *contextLabel != '\0') ? contextLabel : "unspecified")
               << " atom=" << atomIndex << " local_px="
               << gmx::formatString("%.15f", localCoords[atomIndex][XX]) << " local_py="
               << gmx::formatString("%.15f", localCoords[atomIndex][YY]) << " local_pz="
               << gmx::formatString("%.15f", localCoords[atomIndex][ZZ]) << " local_vx="
               << gmx::formatString("%.15f", localVels[atomIndex][XX]) << " local_vy="
               << gmx::formatString("%.15f", localVels[atomIndex][YY]) << " local_vz="
               << gmx::formatString("%.15f", localVels[atomIndex][ZZ]) << " global_px="
               << gmx::formatString("%.15f", globalCoords[atomIndex][XX]) << " global_py="
               << gmx::formatString("%.15f", globalCoords[atomIndex][YY]) << " global_pz="
               << gmx::formatString("%.15f", globalCoords[atomIndex][ZZ]) << " global_vx="
               << gmx::formatString("%.15f", globalVels[atomIndex][XX]) << " global_vy="
               << gmx::formatString("%.15f", globalVels[atomIndex][YY]) << " global_vz="
               << gmx::formatString("%.15f", globalVels[atomIndex][ZZ]) << " code_location="
               << codeLocation << '\n';
    }
}

static void appendUpdateStateTrace(const char*                    traceDirPath,
                                   const int64_t                  step,
                                   const char*                    stageLabel,
                                   gmx::ArrayRef<const gmx::RVec> coordinates,
                                   gmx::ArrayRef<const gmx::RVec> velocities,
                                   const char*                    codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::ofstream output(traceDir / "update_state_trace.txt", std::ios::app);
    const auto filteredAtoms = filteredM2pTraceAtomIndices(coordinates.ssize());
    for (const int atomIndex : filteredAtoms)
    {
        output << "step=" << step << " stage=" << stageLabel << " atom=" << atomIndex
               << " px=" << gmx::formatString("%.15f", coordinates[atomIndex][XX]) << " py="
               << gmx::formatString("%.15f", coordinates[atomIndex][YY]) << " pz="
               << gmx::formatString("%.15f", coordinates[atomIndex][ZZ]) << " vx="
               << gmx::formatString("%.15f", velocities[atomIndex][XX]) << " vy="
               << gmx::formatString("%.15f", velocities[atomIndex][YY]) << " vz="
               << gmx::formatString("%.15f", velocities[atomIndex][ZZ]) << " code_location="
               << codeLocation << '\n';
    }
}

} // namespace

void gmx::LegacySimulator::do_md()
{
    // TODO Historically, the EM and MD "integrators" used different
    // names for the t_inputrec *parameter, but these must have the
    // same name, now that it's a member of a struct. We use this ir
    // alias to avoid a large ripple of nearly useless changes.
    // t_inputrec is being replaced by IMdpOptionsProvider, so this
    // will go away eventually.
    const t_inputrec* ir = inputRec_;

    const double t0 = ir->init_t;
    gmx_bool     bFirstStep, bInitStep, bLastStep = FALSE;
    gmx_bool     bDoExpanded = FALSE;
    tensor    force_vir = { { 0 } }, shake_vir = { { 0 } }, total_vir = { { 0 } }, pres = { { 0 } };
    rvec      mu_tot;
    Matrix3x3 pressureCouplingMu{ 0. }, parrinelloRahmanM{ 0. };
    gmx_repl_ex_t     repl_ex = nullptr;
    gmx_bool          bSumEkinhOld, bDoReplEx, bExchanged, bNeedRepartition;
    real              dvdl_constr;
    std::vector<RVec> cbuf;
    matrix            lastbox;
    int               lamnew = 0;
    /* for FEP */
    real      saved_conserved_quantity = 0;
    real      last_ekin                = 0;
    t_extmass MassQ;
    char      sbuf[STEPSTRSIZE], sbuf2[STEPSTRSIZE];

    bool bInteractiveMDstep = false;
    bool exactRespaDeviceKickHostVelocitiesCurrent = true;
    bool exactRespaGpuNvtKineticReadyForPreTrotter = false;
    bool exactRespaGpuNvtPostTrotterPending = false;
    int64_t exactRespaGpuNvtPostTrotterStep = -1;
    float exactRespaDeviceKickPendingPostTrotterScale = 1.0F;

    SimulationSignals signals;
    // Most global communication stages don't propagate mdrun
    // signals, and will use this object to achieve that.
    SimulationSignaller nullSignaller(nullptr, nullptr, nullptr, false, false);

    if (!mdrunOptions_.writeConfout)
    {
        // This is on by default, and the main known use case for
        // turning it off is for convenience in benchmarking, which is
        // something that should not show up in the general user
        // interface.
        GMX_LOG(mdLog_.info)
                .asParagraph()
                .appendText(
                        "The -noconfout functionality is deprecated, and may be removed in a "
                        "future version.");
    }

    /* md-vv uses averaged full step velocities for T-control
       md-vv-avek uses averaged half step velocities for T-control (but full step ekin for P control)
       md uses averaged half step kinetic energies to determine temperature unless defined otherwise by GMX_EKIN_AVE_VEL; */
    const bool bTrotter =
            (EI_VV(ir->eI)
             && (inputrecNptTrotter(ir) || inputrecNphTrotter(ir) || inputrecNvtTrotter(ir)));

    const bool bRerunMD = false;

    const int  nstglobalcomm   = computeGlobalCommunicationPeriod(mdLog_, ir, cr_->commMyGroup);
    const bool bGStatEveryStep = (nstglobalcomm == 1);

    const SimulationGroups* groups = &topGlobal_.groups;

    std::unique_ptr<EssentialDynamics> ed = nullptr;
    if (opt2bSet("-ei", nFile_, fnm_))
    {
        /* Initialize essential dynamics sampling */
        ed = init_edsam(mdLog_,
                        opt2fn_null("-ei", nFile_, fnm_),
                        opt2fn("-eo", nFile_, fnm_),
                        topGlobal_,
                        *ir,
                        cr_->commMyGroup,
                        cr_->dd,
                        constr_,
                        stateGlobal_,
                        observablesHistory_,
                        oenv_,
                        startingBehavior_);
    }
    else if (observablesHistory_->edsamHistory)
    {
        gmx_fatal(FARGS,
                  "The checkpoint is from a run with essential dynamics sampling, "
                  "but the current run did not specify the -ei option. "
                  "Either specify the -ei option to mdrun, or do not use this checkpoint file.");
    }

    const bool isMainRank = cr_->commMySim.isMainRank();

    int*                fep_state = isMainRank ? &stateGlobal_->fep_state : nullptr;
    gmx::ArrayRef<real> lambda    = isMainRank ? stateGlobal_->lambda : gmx::ArrayRef<real>{};
    initialize_lambdas(
            fpLog_, ir->efep, ir->bSimTemp, *ir->fepvals, ir->simtempvals->temperatures, ekind_, isMainRank, fep_state, lambda);
    Update upd(*ir, *ekind_, deform_);

    // Simulated annealing updates the reference temperature.
    const bool doSimulatedAnnealing = initSimulatedAnnealing(*ir, ekind_, &upd);

    const bool useReplicaExchange = (replExParams_.exchangeInterval > 0);

    t_fcdata& fcdata = *fr_->fcdata;

    // We should let all special algorithms use MDModules, so notifiers tells if we need to share
    bool simulationsShareState =
            (ms_ != nullptr)
            && mdModulesNotifiers_.simulationSetupNotifier_.haveSubscribers<const gmx_multisim_t*>();
    bool simulationsShareHamiltonian = false;
    int  nstSignalComm               = nstglobalcomm;
    {
        // TODO This implementation of ensemble orientation restraints is nasty because
        // a user can't just do multi-sim with single-sim orientation restraints.
        bool usingEnsembleRestraints =
                (fcdata.disres->nsystems > 1) || ((ms_ != nullptr) && fcdata.orires);
        bool awhUsesMultiSim = (ir->bDoAwh && ir->awhParams->shareBiasMultisim() && (ms_ != nullptr));

        // Replica exchange, ensemble restraints and AWH need all
        // simulations to remain synchronized, so they need
        // checkpoints and stop conditions to act on the same step, so
        // the propagation of such signals must take place between
        // simulations, not just within simulations.
        // TODO: Make algorithm initializers set these flags.
        simulationsShareState = simulationsShareState || useReplicaExchange
                                || usingEnsembleRestraints || awhUsesMultiSim;

        // With AWH with bias sharing each simulation uses an non-shared, but identical, Hamiltonian
        simulationsShareHamiltonian = useReplicaExchange || usingEnsembleRestraints;

        if (simulationsShareState)
        {
            // Inter-simulation signal communication does not need to happen
            // often, so we use a minimum of 200 steps to reduce overhead.
            const int c_minimumInterSimulationSignallingInterval = 200;
            nstSignalComm = gmx::divideRoundUp(c_minimumInterSimulationSignallingInterval, nstglobalcomm)
                            * nstglobalcomm;
        }
    }

    if (startingBehavior_ != StartingBehavior::RestartWithAppending)
    {
        pleaseCiteCouplingAlgorithms(fpLog_, *ir);
    }
    gmx_mdoutf*       outf = init_mdoutf(fpLog_,
                                   nFile_,
                                   fnm_,
                                   mdrunOptions_,
                                   cr_,
                                   outputProvider_,
                                   mdModulesNotifiers_,
                                   ir,
                                   topGlobal_,
                                   oenv_,
                                   wallCycleCounters_,
                                   startingBehavior_,
                                   simulationsShareState,
                                   ms_);
    gmx::EnergyOutput energyOutput(mdoutf_get_fp_ene(outf),
                                   topGlobal_,
                                   *ir,
                                   pullWork_,
                                   mdoutf_get_fp_dhdl(outf),
                                   false,
                                   startingBehavior_,
                                   simulationsShareHamiltonian,
                                   mdModulesNotifiers_);

    gmx_global_stat_t gstat = global_stat_init(ir);

    const auto& simulationWork     = runScheduleWork_->simulationWork;
    const bool  useGpuForPme       = simulationWork.useGpuPme;
    const bool  useGpuForNonbonded = simulationWork.useGpuNonbonded;
    const bool  useGpuForUpdate    = simulationWork.useGpuUpdate;

    /* Check for polarizable models and flexible constraints */
    gmx_shellfc_t* shellfc = init_shell_flexcon(fpLog_,
                                                topGlobal_,
                                                constr_ ? constr_->numFlexibleConstraints() : 0,
                                                ir->nstcalcenergy,
                                                haveDDAtomOrdering(*cr_),
                                                simulationWork);

    ObservablesReducer observablesReducer = observablesReducerBuilder_->build();

    const int numForceBufferLevels = simulationWork.useExactRespa
                                             ? gmx::exactRespaNumLevels(*ir)
                                             : (simulationWork.useLegacyMtsSubsteps()
                                                        ? static_cast<int>(ir->mtsLevels.size())
                                                        : 0);
    ForceBuffers f(numForceBufferLevels,
                   (simulationWork.useGpuFBufferOpsWhenAllowed || useGpuForUpdate)
                           ? PinningPolicy::PinnedIfSupported
                           : PinningPolicy::CannotBePinned);
    gmx::ExactRespaForceStore exactRespaForceStore;
    gmx::ExactRespaForceStore* exactRespaForceStorePtr =
            gmx::useExactRespa(*ir) ? &exactRespaForceStore : nullptr;
    const t_mdatoms* md = mdAtoms_->mdatoms();
    if (haveDDAtomOrdering(*cr_))
    {
        // Local state only becomes valid now.
        dd_init_local_state(*cr_->dd, stateGlobal_, state_);

        /* Distribute the charge groups over the nodes from the main node */
        dd_partition_system(fpLog_,
                            mdLog_,
                            ir->init_step,
                            cr_->dd,
                            TRUE,
                            stateGlobal_,
                            topGlobal_,
                            *ir,
                            mdModulesNotifiers_,
                            imdSession_,
                            pullWork_,
                            state_,
                            &f,
                            mdAtoms_,
                            top_,
                            fr_,
                            virtualSites_,
                            constr_,
                            nrnb_,
                            nullptr,
                            FALSE);
        upd.updateAfterPartition(state_->numAtoms(), md->cFREEZE, md->cTC, md->cACC);
        fr_->longRangeNonbondeds->updateAfterPartition(*md);
    }
    else
    {
        /* Generate and initialize new topology */
        mdAlgorithmsSetupAtomData(
                cr_->dd, *ir, topGlobal_, top_, fr_, &f, mdAtoms_, constr_, virtualSites_, shellfc);

        upd.updateAfterPartition(state_->numAtoms(), md->cFREEZE, md->cTC, md->cACC);
        fr_->longRangeNonbondeds->updateAfterPartition(*md);
    }

    // Now that the state is valid we can set up Parrinello-Rahman
    init_parrinellorahman(ir->pressureCouplingOptions,
                          ir->deform,
                          ir->delta_t * ir->pressureCouplingOptions.nstpcouple,
                          state_->box,
                          state_->box_rel,
                          state_->boxv,
                          &parrinelloRahmanM,
                          &pressureCouplingMu);

    std::unique_ptr<UpdateConstrainGpu> integrator;
    exactRespaGpuUpdater_ = nullptr;

    StatePropagatorDataGpu* stateGpu = fr_->stateGpu;

    // TODO: the assertions below should be handled by UpdateConstraintsBuilder.
    if (useGpuForUpdate)
    {
        const bool exactRespaGpuUpdate = useExactRespa(*ir) && ir->eI == IntegrationAlgorithm::VV;
        GMX_RELEASE_ASSERT(!haveDDAtomOrdering(*cr_) || ddUsesUpdateGroups(*cr_->dd)
                                   || constr_ == nullptr || constr_->numConstraintsTotal() == 0,
                           "Constraints in domain decomposition are only supported with update "
                           "groups if using GPU update.\n");
        GMX_RELEASE_ASSERT(ir->eConstrAlg != ConstraintAlgorithm::Shake || constr_ == nullptr
                                   || constr_->numConstraintsTotal() == 0,
                           "SHAKE is not supported with GPU update.");
        GMX_RELEASE_ASSERT(useGpuForPme || (useGpuForNonbonded && simulationWork.useGpuXBufferOpsWhenAllowed),
                           "Either PME or short-ranged non-bonded interaction tasks must run on "
                           "the GPU to use GPU update.\n");
        GMX_RELEASE_ASSERT(ir->eI == IntegrationAlgorithm::MD || exactRespaGpuUpdate,
                           "Only the md integrator and standalone exact r-RESPA md-vv are "
                           "supported with the GPU update.\n");
        GMX_RELEASE_ASSERT(!exactRespaGpuUpdate || constr_ == nullptr || constr_->numConstraintsTotal() == 0,
                           "Standalone exact r-RESPA GPU update currently supports unconstrained systems only.\n");
        GMX_RELEASE_ASSERT(
                exactRespaGpuUpdate || ir->etc != TemperatureCoupling::NoseHoover,
                "Nose-Hoover temperature coupling is not supported with the GPU update.\n");
        GMX_RELEASE_ASSERT(
                ir->pressureCouplingOptions.epc == PressureCoupling::No
                        || ir->pressureCouplingOptions.epc == PressureCoupling::ParrinelloRahman
                        || ir->pressureCouplingOptions.epc == PressureCoupling::Berendsen
                        || ir->pressureCouplingOptions.epc == PressureCoupling::CRescale,
                "Only Parrinello-Rahman, Berendsen, and C-rescale pressure coupling are supported "
                "with the GPU update.\n");
        GMX_RELEASE_ASSERT(!md->haveVsites,
                           "Virtual sites are not supported with the GPU update.\n");
        GMX_RELEASE_ASSERT(ed == nullptr,
                           "Essential dynamics is not supported with the GPU update.\n");
        GMX_RELEASE_ASSERT(!ir->bPull || !pull_have_constraint(*ir->pull),
                           "Constraints pulling is not supported with the GPU update.\n");
        GMX_RELEASE_ASSERT(fcdata.orires == nullptr,
                           "Orientation restraints are not supported with the GPU update.\n");
        GMX_RELEASE_ASSERT(
                ir->efep == FreeEnergyPerturbationType::No
                        || (!haveFepPerturbedMasses(topGlobal_) && !havePerturbedConstraints(topGlobal_)),
                "Free energy perturbation of masses and constraints are not supported with the GPU "
                "update.");

        if (constr_ != nullptr && constr_->numConstraintsTotal() > 0)
        {
            GMX_LOG(mdLog_.info)
                    .asParagraph()
                    .appendText("Updating coordinates and applying constraints on the GPU.");
        }
        else
        {
            GMX_LOG(mdLog_.info).asParagraph().appendText("Updating coordinates on the GPU.");
        }
        GMX_RELEASE_ASSERT(fr_->deviceStreamManager != nullptr,
                           "Device stream manager should be initialized in order to use GPU "
                           "update-constraints.");
        GMX_RELEASE_ASSERT(
                fr_->deviceStreamManager->streamIsValid(gmx::DeviceStreamType::UpdateAndConstraints),
                "Update stream should be initialized in order to use GPU "
                "update-constraints.");
        integrator = std::make_unique<UpdateConstrainGpu>(
                *ir,
                topGlobal_,
                ekind_->numTemperatureCouplingGroups(),
                fr_->deviceStreamManager->context(),
                fr_->deviceStreamManager->stream(gmx::DeviceStreamType::UpdateAndConstraints),
                wallCycleCounters_);
        exactRespaGpuUpdater_ = integrator.get();

        stateGpu->setXUpdatedOnDeviceEvent(integrator->xUpdatedOnDeviceEvent());

        integrator->setPbc(PbcType::Xyz, state_->box);
    }

    if (useGpuForPme || simulationWork.useGpuXBufferOpsWhenAllowed || useGpuForUpdate)
    {
        changePinningPolicy(&state_->x, PinningPolicy::PinnedIfSupported);
    }
    if (useGpuForUpdate)
    {
        changePinningPolicy(&state_->v, PinningPolicy::PinnedIfSupported);
    }

    // NOTE: The global state is no longer used at this point.
    // But state_global is still used as temporary storage space for writing
    // the global state to file and potentially for replica exchange.
    // (Global topology should persist.)

    update_mdatoms(mdAtoms_->mdatoms(), state_->lambda[FreeEnergyPerturbationCouplingType::Mass]);

    if (ir->bExpanded)
    {
        /* Check nstexpanded here, because the grompp check was broken */
        if (ir->expandedvals->nstexpanded % ir->nstcalcenergy != 0)
        {
            gmx_fatal(FARGS,
                      "With expanded ensemble, nstexpanded should be a multiple of nstcalcenergy");
        }
        init_expanded_ensemble(
                startingBehavior_ != StartingBehavior::NewSimulation, ir, state_->dfhist.get());
    }

    if (isMainRank)
    {
        EnergyData::initializeEnergyHistory(startingBehavior_, observablesHistory_, &energyOutput);
    }

    preparePrevStepPullCom(ir,
                           pullWork_,
                           md->massT,
                           state_,
                           stateGlobal_,
                           cr_->commMyGroup,
                           startingBehavior_ != StartingBehavior::NewSimulation);

    // TODO: Remove this by converting AWH into a ForceProvider
    auto awh = prepareAwhModule(fpLog_,
                                *ir,
                                stateGlobal_,
                                cr_->commMyGroup,
                                ms_,
                                startingBehavior_ != StartingBehavior::NewSimulation,
                                shellfc != nullptr,
                                opt2fn("-awh", nFile_, fnm_),
                                pullWork_);

    if (useReplicaExchange && isMainRank)
    {
        repl_ex = init_replica_exchange(fpLog_, ms_, topGlobal_.natoms, ir, replExParams_);
    }

    // PME tuning is only supported with PME for Coulomb. It is not supported with only LJ PME
    std::unique_ptr<PmeLoadBalancing> pmeLoadBal;
    if (mdrunOptions_.tunePme
        && shouldUsePmeLoadBalancingForExactRespa(*ir)
        && pmeTuningIsSupported(fr_->ic->coulomb.type, mdrunOptions_.reproducible, simulationWork))
    {
        pmeLoadBal = std::make_unique<PmeLoadBalancing>(
                cr_->dd, mdLog_, *ir, state_->box, *fr_->ic, *fr_->nbv, fr_->pmedata, simulationWork);
    }
    else if (mdrunOptions_.tunePme && gmx::useExactRespa(*ir))
    {
        GMX_LOG(mdLog_.info)
                .asParagraph()
                .appendText("PME load balancing disabled for exact r-RESPA. Set "
                            "GMX_PCFF_EXACT_RESPA_ALLOW_PME_TUNING=1 to restore PME tuning.");
    }

    if (!ir->bContinuation)
    {
        if (state_->hasEntry(StateEntry::V))
        {
            auto v = makeArrayRef(state_->v);
            /* Set the velocities of vsites, shells and frozen atoms to zero */
            for (int i = 0; i < md->homenr; i++)
            {
                if (md->ptype[i] == ParticleType::Shell)
                {
                    clear_rvec(v[i]);
                }
                else if (!md->cFREEZE.empty())
                {
                    for (int m = 0; m < DIM; m++)
                    {
                        if (ir->opts.nFreeze[md->cFREEZE[i]][m])
                        {
                            v[i][m] = 0;
                        }
                    }
                }
            }
        }

        if (constr_)
        {
            /* Constrain the initial coordinates and velocities */
            do_constrain_first(fpLog_,
                               constr_,
                               *ir,
                               md->homenr,
                               state_->x.arrayRefWithPadding(),
                               state_->v.arrayRefWithPadding(),
                               state_->box,
                               state_->lambda[FreeEnergyPerturbationCouplingType::Bonded]);
        }
    }

    const int nstfep = computeFepPeriod(*ir, replExParams_);

    /* Be REALLY careful about what flags you set here. You CANNOT assume
     * this is the first step, since we might be restarting from a checkpoint,
     * and in that case we should not do any modifications to the state.
     */
    const bool stopCenterOfMassMovementBeforeFirstStep =
            (ir->comm_mode != ComRemovalAlgorithm::No && !ir->bContinuation);

    // When restarting from a checkpoint, it can be appropriate to
    // initialize ekind from quantities in the checkpoint. Otherwise,
    // compute_globals must initialize ekind before the simulation
    // starts/restarts. However, only the main rank knows what was
    // found in the checkpoint file, so we have to communicate in
    // order to coordinate the restart.
    //
    // TODO Consider removing this communication if/when checkpoint
    // reading directly follows .tpr reading, because all ranks can
    // agree on hasReadEkinState at that time.
    bool hasReadEkinState = isMainRank ? stateGlobal_->ekinstate.hasReadEkinState : false;
    if (cr_->commMyGroup.isParallel())
    {
        gmx_bcast(sizeof(hasReadEkinState), &hasReadEkinState, cr_->commMyGroup.comm());
    }
    if (hasReadEkinState)
    {
        restore_ekinstate_from_state(
                cr_->commMyGroup, ekind_, isMainRank ? &stateGlobal_->ekinstate : nullptr);
    }

    unsigned int cglo_flags =
            (CGLO_TEMPERATURE | CGLO_GSTAT | (EI_VV(ir->eI) ? CGLO_PRESSURE : 0)
             | (EI_VV(ir->eI) ? CGLO_CONSTRAINT : 0) | (hasReadEkinState ? CGLO_READEKIN : 0));

    bSumEkinhOld = FALSE;

    t_vcm vcm(topGlobal_.groups, *ir, topGlobal_.natoms);
    reportComRemovalInfo(fpLog_, vcm);

    int64_t step     = ir->init_step;
    int64_t step_rel = 0;

    /* To minimize communication, compute_globals computes the COM velocity
     * and the kinetic energy for the velocities without COM motion removed.
     * Thus to get the kinetic energy without the COM contribution, we need
     * to call compute_globals twice.
     */
    for (int cgloIteration = 0; cgloIteration < (stopCenterOfMassMovementBeforeFirstStep ? 2 : 1);
         cgloIteration++)
    {
        unsigned int cglo_flags_iteration = cglo_flags;
        if (stopCenterOfMassMovementBeforeFirstStep && cgloIteration == 0)
        {
            cglo_flags_iteration |= CGLO_STOPCM;
            cglo_flags_iteration &= ~CGLO_TEMPERATURE;
        }
        compute_globals(gstat,
                        cr_->commMyGroup,
                        ir,
                        fr_,
                        ekind_,
                        makeConstArrayRef(state_->x),
                        makeConstArrayRef(state_->v),
                        state_->box,
                        md,
                        nrnb_,
                        &vcm,
                        nullptr,
                        enerd_,
                        force_vir,
                        shake_vir,
                        total_vir,
                        pres,
                        &nullSignaller,
                        state_->box,
                        &bSumEkinhOld,
                        cglo_flags_iteration,
                        step - 1, // Pass step-1 to signal that v is from minus a half step
                        &observablesReducer);
        // Clean up after pre-step use of compute_globals()
        observablesReducer.markAsReadyToReduce();

        if (cglo_flags_iteration & CGLO_STOPCM)
        {
            /* At initialization, do not pass x with acceleration-correction mode
             * to avoid (incorrect) correction of the initial coordinates.
             */
            auto x = (vcm.mode == ComRemovalAlgorithm::LinearAccelerationCorrection)
                             ? ArrayRef<RVec>{}
                             : makeArrayRef(state_->x);
            process_and_stopcm_grp(fpLog_, &vcm, *md, x, makeArrayRef(state_->v));
            inc_nrnb(nrnb_, eNR_STOPCM, md->homenr);
        }
    }
    if (ir->eI == IntegrationAlgorithm::VVAK)
    {
        /* a second call to get the half step temperature initialized as well */
        /* we do the same call as above, but turn the pressure off -- internally to
           compute_globals, this is recognized as a velocity verlet half-step
           kinetic energy calculation.  This minimized excess variables, but
           perhaps loses some logic?*/

        compute_globals(gstat,
                        cr_->commMyGroup,
                        ir,
                        fr_,
                        ekind_,
                        makeConstArrayRef(state_->x),
                        makeConstArrayRef(state_->v),
                        state_->box,
                        md,
                        nrnb_,
                        &vcm,
                        nullptr,
                        enerd_,
                        force_vir,
                        shake_vir,
                        total_vir,
                        pres,
                        &nullSignaller,
                        state_->box,
                        &bSumEkinhOld,
                        cglo_flags & ~CGLO_PRESSURE,
                        step,
                        &observablesReducer);
        // Clean up after pre-step use of compute_globals()
        observablesReducer.markAsReadyToReduce();
    }

    /* Calculate the initial half step temperature, and save the ekinh_old */
    if (startingBehavior_ == StartingBehavior::NewSimulation)
    {
        for (int i = 0; (i < ir->opts.ngtc); i++)
        {
            copy_mat(ekind_->tcstat[i].ekinh, ekind_->tcstat[i].ekinh_old);
        }
    }

    /* need to make an initiation call to get the Trotter variables set, as well as other constants
       for non-trotter temperature control */
    auto trotter_seq = init_npt_vars(ir, *ekind_, state_, &MassQ, bTrotter);
    if (pcffResetNhMttkStateOnStartEnabled())
    {
        resetPcffNhMttkStateOnStart(*ir, state_);
        if (isMainRank)
        {
            GMX_LOG(mdLog_.info)
                    .asParagraph()
                    .appendText("PCFF reset NH/MTTK extended state on start while preserving checkpoint x/v.");
        }
    }
    if (pcffRestoreNhMttkStateEnergyPath() != nullptr)
    {
        restorePcffNhMttkStateFromEnergy(*ir, topGlobal_.groups, state_);
        if (isMainRank)
        {
            GMX_LOG(mdLog_.info)
                    .asParagraph()
                    .appendTextFormatted(
                            "PCFF restored NH/MTTK extended state from energy file %s at t=%g ps.",
                            pcffRestoreNhMttkStateEnergyPath(),
                            pcffRestoreNhMttkStateEnergyTime(*ir));
        }
    }
    if (pcffRestoreNhMttkStateLammpsFixVector() != nullptr)
    {
        restorePcffNhMttkStateFromLammpsFixVector(*ir, state_);
        if (isMainRank)
        {
            GMX_LOG(mdLog_.info)
                    .asParagraph()
                    .appendText("PCFF restored NH/MTTK extended state from LAMMPS FixNH f_1 vector.");
        }
    }

    if (isMainRank)
    {
        if (!ir->bContinuation)
        {
            if (constr_ && ir->eConstrAlg == ConstraintAlgorithm::Lincs)
            {
                fprintf(fpLog_,
                        "RMS relative constraint deviation after constraining: %.2e\n",
                        constr_->rmsd());
            }
            if (EI_STATE_VELOCITY(ir->eI))
            {
                real temp = enerd_->term[InteractionFunction::Temperature];
                if (ir->eI != IntegrationAlgorithm::VV)
                {
                    /* Result of Ekin averaged over velocities of -half
                     * and +half step, while we only have -half step here.
                     */
                    temp *= 2;
                }
                fprintf(fpLog_, "Initial temperature: %g K\n", temp);
            }
        }

        char tbuf[20];
        fprintf(stderr, "starting mdrun '%s'\n", *(topGlobal_.name));
        if (ir->nsteps >= 0)
        {
            sprintf(tbuf, "%8.1f", (ir->init_step + ir->nsteps) * ir->delta_t);
        }
        else
        {
            sprintf(tbuf, "%s", "infinite");
        }
        if (ir->init_step > 0)
        {
            fprintf(stderr,
                    "%s steps, %s ps (continuing from step %s, %8.1f ps).\n",
                    gmx_step_str(ir->init_step + ir->nsteps, sbuf),
                    tbuf,
                    gmx_step_str(ir->init_step, sbuf2),
                    ir->init_step * ir->delta_t);
        }
        else
        {
            fprintf(stderr, "%s steps, %s ps.\n", gmx_step_str(ir->nsteps, sbuf), tbuf);
        }
        fprintf(fpLog_, "\n");
    }

    walltime_accounting_start_time(wallTimeAccounting_);
    wallcycle_start(wallCycleCounters_, WallCycleCounter::Run);
    print_start(fpLog_, cr_, wallTimeAccounting_, "mdrun");

    /***********************************************************
     *
     *             Loop over MD steps
     *
     ************************************************************/

    bFirstStep = TRUE;
    /* Skip the first Nose-Hoover integration when we get the state from tpx */
    bInitStep        = startingBehavior_ == StartingBehavior::NewSimulation || EI_VV(ir->eI);
    bSumEkinhOld     = FALSE;
    bExchanged       = FALSE;
    bNeedRepartition = FALSE;

    auto stopHandler = stopHandlerBuilder_->getStopHandlerMD(
            compat::not_null<SimulationSignal*>(&signals[eglsSTOPCOND]),
            simulationsShareState,
            isMainRank,
            ir->nstlist,
            mdrunOptions_.reproducible,
            nstSignalComm,
            mdrunOptions_.maximumHoursToRun,
            fpLog_,
            step,
            wallTimeAccounting_);

    real checkpointPeriod = mdrunOptions_.checkpointOptions.period;
    if (ir->bExpanded)
    {
        GMX_LOG(mdLog_.info)
                .asParagraph()
                .appendText(
                        "Expanded ensemble with the legacy simulator does not always "
                        "checkpoint correctly, so checkpointing is disabled. You will "
                        "not be able to do a checkpoint restart of this simulation. "
                        "If you use the modular simulator (e.g. by choosing md-vv integrator) "
                        "then checkpointing is enabled. See "
                        "https://gitlab.com/gromacs/gromacs/-/issues/4629 for details.");
        // Use a negative period to disable checkpointing.
        checkpointPeriod = -1;
    }
    auto checkpointHandler = std::make_unique<CheckpointHandler>(
            compat::make_not_null<SimulationSignal*>(&signals[eglsCHKPT]),
            simulationsShareState,
            ir->nstlist == 0,
            isMainRank,
            mdrunOptions_.writeConfout,
            checkpointPeriod);

    if (gmx::useExactRespa(*ir))
    {
        GMX_LOG(mdLog_.info)
                .asParagraph()
                .appendText("Standalone exact r-RESPA active (mts-mode = lammps-respa).");
    }
    t_inputrec* mutableInputRecord = const_cast<t_inputrec*>(ir);
    const PressureCouplingOptions basePressureCouplingOptions = ir->pressureCouplingOptions;
    const PcffContinuousRefPressureRamp pcffContinuousRefPressureRamp =
            pcffContinuousRefPressureRampFromEnv(*ir);
    const PcffMttkReferenceCellReset pcffMttkReferenceCellReset =
            pcffMttkReferenceCellResetFromEnv(*ir);
    if (pcffContinuousRefPressureRamp.active)
    {
        GMX_LOG(mdLog_.info)
                .asParagraph()
                .appendText(formatString("PCFF continuous reference-pressure ramp active: "
                                         "%.8g -> %.8g bar over %.8g ps.",
                                         pcffContinuousRefPressureRamp.startBar,
                                         pcffContinuousRefPressureRamp.endBar,
                                         pcffContinuousRefPressureRamp.durationPs));
    }
    if (pcffMttkReferenceCellReset.active)
    {
        GMX_LOG(mdLog_.info)
                .asParagraph()
                .appendText(formatString("PCFF MTTK reference-cell reset active: every %lld steps.",
                                         static_cast<long long>(pcffMttkReferenceCellReset.intervalSteps)));
    }
    if (pcffUseLammpsFixNhMassMode() && ir->pressureCouplingOptions.epc == PressureCoupling::Mttk)
    {
        const double natoms            = pcffReadPositiveEnvRealRequired("GMX_PCFF_MTTK_LAMMPS_NATOMS");
        const double pdampPs           = pcffLammpsFixNhPdampPs(*ir);
        const double pressureMassScale = pcffLammpsFixNhPressureMassScale();
        const bool   exactRespaActive  = ir->exactRespa.enabled();
        const int    pairSplitOuterLevel =
                exactRespaActive ? ir->exactRespa.forceLayout.outerLevel : -1;
        int mttkOuterStepLevel  = 0;
        int mttkOuterStepFactor = 1;
        if (exactRespaActive)
        {
            const int numRespaLevels = static_cast<int>(ir->exactRespa.levelStepFactors.size());
            mttkOuterStepLevel = pairSplitOuterLevel > 0 ? pairSplitOuterLevel : numRespaLevels - 1;
            if (mttkOuterStepLevel >= 0 && mttkOuterStepLevel < numRespaLevels)
            {
                mttkOuterStepFactor = ir->exactRespa.levelStepFactors[mttkOuterStepLevel];
            }
        }
        const char*  extendedUpdateMode =
                std::getenv("GMX_PCFF_EXACT_RESPA_MTTK_EXTENDED_UPDATE");
        const char* vetaScale = std::getenv("GMX_PCFF_EXACT_RESPA_MTTK_VETA_SCALE");
        GMX_LOG(mdLog_.info)
                .asParagraph()
                .appendText(formatString("PCFF LAMMPS FixNH MTTK mass mode active: natoms=%.10g, "
                                         "pdamp=%.10g ps, pressure-mass-scale=%.10g, extended-update=%s, "
                                         "veta-scale=%s, mttk-outer-step-level=%d, "
                                         "mttk-outer-step-factor=%d, pair-split-outer-level=%d.",
                                         natoms,
                                         pdampPs,
                                         pressureMassScale,
                                         (extendedUpdateMode != nullptr && extendedUpdateMode[0] != '\0')
                                                 ? extendedUpdateMode
                                                 : "unset",
                                         (vetaScale != nullptr && vetaScale[0] != '\0') ? vetaScale
                                                                                        : "1",
                                         exactRespaActive ? mttkOuterStepLevel + 1 : 0,
                                         mttkOuterStepFactor,
                                         pairSplitOuterLevel >= 0 ? pairSplitOuterLevel + 1 : 0));
    }

    const bool resetCountersIsLocal = true;
    auto       resetHandler         = std::make_unique<ResetHandler>(
            compat::make_not_null<SimulationSignal*>(&signals[eglsRESETCOUNTERS]),
            !resetCountersIsLocal,
            ir->nsteps,
            isMainRank,
            mdrunOptions_.timingOptions.resetHalfway,
            mdrunOptions_.maximumHoursToRun,
            mdLog_,
            wallCycleCounters_,
            wallTimeAccounting_);

    const DDBalanceRegionHandler ddBalanceRegionHandler(cr_->dd);

    const auto completeExactRespaPendingGpuNvtPostTrotter = [&](const bool calcGlobalStats)
    {
        if (!exactRespaGpuNvtPostTrotterPending)
        {
            return;
        }
#if GMX_GPU_CUDA
        GMX_RELEASE_ASSERT(exactRespaGpuUpdater_ != nullptr,
                           "Exact r-RESPA deferred GPU kinetic reduction needs an updater");
        const real kineticEnergy = exactRespaGpuUpdater_->finishExactRespaKineticEnergy();
        GMX_RELEASE_ASSERT(std::isfinite(kineticEnergy) && kineticEnergy >= 0,
                           "Exact r-RESPA GPU kinetic energy is invalid");

        t_grp_tcstat& tcstat = ekind_->tcstat[0];
        clear_mat(tcstat.ekinf);
        tcstat.ekinf[XX][XX]   = kineticEnergy;
        tcstat.ekinscalef_nhc = 1.0;
        ekind_->dekindl_old   = ekind_->dekindl;
        ekind_->dekindl       = 0;
        real dvdlKinetic      = 0;
        enerd_->term[InteractionFunction::Temperature] =
                sum_ekin(&ir->opts, ekind_, &dvdlKinetic, true, true);
        enerd_->dvdl_lin[FreeEnergyPerturbationCouplingType::Mass] =
                static_cast<double>(dvdlKinetic);
        enerd_->term[InteractionFunction::KineticEnergy] = ::trace(ekind_->ekin);
        ekind_->lastComputeGlobalsStep = exactRespaGpuNvtPostTrotterStep + 1;
        bSumEkinhOld                   = calcGlobalStats ? FALSE : TRUE;
        saved_conserved_quantity      = 0;
        last_ekin = enerd_->term[InteractionFunction::KineticEnergy];

        trotter_update(ir,
                       exactRespaGpuNvtPostTrotterStep,
                       ekind_,
                       state_,
                       total_vir,
                       0,
                       md->cTC,
                       md->invmass,
                       &MassQ,
                       trotter_seq,
                       TrotterSequence::Three);
        GMX_RELEASE_ASSERT(exactRespaDeviceKickPendingPostTrotterScale == 1.0F,
                           "Exact r-RESPA deferred post-Trotter scale was not consumed");
        exactRespaDeviceKickPendingPostTrotterScale =
                static_cast<float>(ekind_->tcstat[0].vscale_nhc);
        exactRespaGpuNvtPostTrotterPending = false;
        exactRespaGpuNvtPostTrotterStep    = -1;
#else
        GMX_RELEASE_ASSERT(false, "Deferred exact r-RESPA GPU kinetic reduction needs CUDA");
#endif
    };

    const auto applyExactRespaPendingPostTrotterScaleToDevice = [&]()
    {
#if GMX_GPU_CUDA
        if (useGpuForUpdate && exactRespaGpuUpdater_ != nullptr
            && exactRespaDeviceKickPendingPostTrotterScale != 1.0F)
        {
            exactRespaGpuUpdater_->exactRespaScaleVelocities(
                    static_cast<real>(exactRespaDeviceKickPendingPostTrotterScale));
            exactRespaDeviceKickPendingPostTrotterScale = 1.0F;
        }
#endif
    };

    if (isMainRank && isMultiSim(ms_) && !useReplicaExchange)
    {
        logInitialMultisimStatus(*ms_, cr_, mdLog_, simulationsShareState, ir->nsteps, ir->init_step);
    }

    bool usedMdGpuGraphLastStep = false;
    /* and stop now if we should */
    bLastStep = (bLastStep || (ir->nsteps >= 0 && step_rel > ir->nsteps));
    while (!bLastStep)
    {
        if (pcffContinuousRefPressureRamp.active)
        {
            mutableInputRecord->pressureCouplingOptions =
                    pressureCouplingOptionsWithPcffContinuousRefPressureRamp(
                            pcffContinuousRefPressureRamp, *ir, basePressureCouplingOptions, step);
        }
        if (pcffMttkReferenceCellResetThisStep(pcffMttkReferenceCellReset, *ir, step))
        {
            applyPcffMttkReferenceCellReset(*ir, state_, &MassQ, *ekind_);
        }
        appendPreDoForceStateTrace(activeM2pTraceDirPath(),
                                   step,
                                   "loop_start",
                                   state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                   state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                   "src/gromacs/mdrun/md.cpp:loop_start");
        /* Determine if this is a neighbor search step */
        const bool bNStList = (ir->nstlist > 0 && step % ir->nstlist == 0);

        if (pmeLoadBal && bNStList)
        {
            // This has to be here because PME load balancing is called so early.
            // TODO: Move to after all booleans are defined.
            if (useGpuForUpdate && !bFirstStep)
            {
                stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
            }
            /* PME grid + cut-off optimization with GPUs or PME nodes */
            pmeLoadBal->addCycles((mdrunOptions_.verbose && isMainRank) ? stderr : nullptr,
                                  fr_,
                                  state_->box,
                                  state_->x,
                                  wallCycleCounters_,
                                  step,
                                  step_rel);
        }
        appendPreDoForceStateTrace(activeM2pTraceDirPath(),
                                   step,
                                   "after_pme_loadbal",
                                   state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                   state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                   "src/gromacs/mdrun/md.cpp:after_pme_loadbal");

        wallcycle_start(wallCycleCounters_, WallCycleCounter::Step);

        bLastStep      = (step_rel == ir->nsteps);
        const double t = t0 + step * ir->delta_t;

        // TODO Refactor this, so that nstfep does not need a default value of zero
        if (ir->efep != FreeEnergyPerturbationType::No || ir->bSimTemp)
        {
            /* find and set the current lambdas */
            state_->lambda = currentLambdas(step, *(ir->fepvals), state_->fep_state);

            bDoExpanded = (do_per_step(step, ir->expandedvals->nstexpanded) && (ir->bExpanded)
                           && (!bFirstStep));
        }

        bDoReplEx = (useReplicaExchange && (step > 0) && !bLastStep
                     && do_per_step(step, replExParams_.exchangeInterval));

        if (doSimulatedAnnealing)
        {
            // Simulated annealing updates the reference temperature.
            update_annealing_target_temp(*ir, t, ekind_, &upd);
        }

        /* Stop Center of Mass motion */
        const bool bStopCM = (ir->comm_mode != ComRemovalAlgorithm::No && do_per_step(step, ir->nstcomm));

        /* Determine whether or not to do Neighbour Searching */
        const bool bNS = (bFirstStep || bNStList || bExchanged || bNeedRepartition);

        /* Note that the stopHandler will cause termination at nstglobalcomm
         * steps. Since this concides with nstcalcenergy, nsttcouple and/or
         * nstpcouple steps, we have computed the half-step kinetic energy
         * of the previous step and can always output energies at the last step.
         */
        const bool stopAfterCurrentStep = stopHandler->stoppingAfterCurrentStep(step);
        bool       deferStopUntilExactRespaOuterStep = false;
        if (!bLastStep && stopAfterCurrentStep && gmx::useExactRespa(*ir))
        {
            const int outerLevel = gmx::exactRespaNonbondedOuterLevel(*ir);
            deferStopUntilExactRespaOuterStep =
                    outerLevel > gmx::highestActiveMtsLevel(ir->mtsLevels, step);
        }
        // A second SIGINT/SIGTERM asks GROMACS to stop immediately, which can land on
        // an inner-only exact r-RESPA step. Defer by at most a few base steps so the
        // final virial/energy path still runs on the outer nonbonded boundary.
        bLastStep = bLastStep || (stopAfterCurrentStep && !deferStopUntilExactRespaOuterStep);

        /* do_log triggers energy and virial calculation. Because this leads
         * to different code paths, forces can be different. Thus for exact
         * continuation we should avoid extra log output.
         * Note that the || bLastStep can result in non-exact continuation
         * beyond the last step. But we don't consider that to be an issue.
         */
        const bool suppressEnergyVirialForForceOnlyExactRespa = useExactLammpsRespaForceOnlyContract(*ir);
        const bool do_log = !suppressEnergyVirialForForceOnlyExactRespa
                            && (do_per_step(step, ir->nstlog)
                                || (bFirstStep && startingBehavior_ == StartingBehavior::NewSimulation)
                                || bLastStep);
        const bool do_verbose =
                mdrunOptions_.verbose
                && (step % mdrunOptions_.verboseStepPrintInterval == 0 || bFirstStep || bLastStep);

        // On search steps, when doing the update on the GPU, copy
        // the coordinates and velocities to the host unless they are
        // already there (ie on the first step and after replica
        // exchange).
        if (useGpuForUpdate && bNS && !bFirstStep && !bExchanged)
        {
            if (usedMdGpuGraphLastStep)
            {
                // Wait on coordinates produced from GPU graph
                stateGpu->waitCoordinatesUpdatedOnDevice();
            }
            const bool searchStepRequiresHostVelocities =
                    do_per_step(step, ir->nstvout) || checkpointHandler->isCheckpointingStep();
            if (exactRespaGpuNvtKineticReadyForPreTrotter
                && searchStepRequiresHostVelocities)
            {
                completeExactRespaPendingGpuNvtPostTrotter(false);
                applyExactRespaPendingPostTrotterScaleToDevice();
            }
            if (!(exactRespaGpuNvtKineticReadyForPreTrotter
                  && !searchStepRequiresHostVelocities)
                && (!exactRespaDeviceKickGpuUpdateProbeEnabled()
                    || !exactRespaDeviceKickHostVelocitiesCurrent))
            {
                stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
                stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
                exactRespaDeviceKickHostVelocitiesCurrent = true;
            }
            stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
        }

        // We need to calculate virtual velocities if we are writing them in the current step.
        // They also need to be periodically updated. Every 1000 steps is arbitrary, but a reasonable number.
        // The reason why the velocities need to be updated regularly is that the virtual site coordinates
        // are updated using these velocities during integration. Those coordinates are used for, e.g., domain
        // decomposition. Before computing any forces the positions of the virtual sites are recalculated.
        // This fixes a bug, #4879, which was introduced in MR !979.
        const int  c_virtualSiteVelocityUpdateInterval = 1000;
        const bool needVirtualVelocitiesThisStep =
                (virtualSites_ != nullptr)
                && (do_per_step(step, ir->nstvout) || checkpointHandler->isCheckpointingStep()
                    || do_per_step(step, c_virtualSiteVelocityUpdateInterval));

        if (virtualSites_ != nullptr)
        {
            // Virtual sites need to be updated before domain decomposition and forces are calculated
            wallcycle_start(wallCycleCounters_, WallCycleCounter::VsiteConstr);
            // md-vv calculates virtual velocities once it has full-step real velocities
            virtualSites_->construct(state_->x,
                                     state_->v,
                                     state_->box,
                                     (!EI_VV(inputRec_->eI) && needVirtualVelocitiesThisStep)
                                             ? VSiteOperation::PositionsAndVelocities
                                             : VSiteOperation::Positions);
            wallcycle_stop(wallCycleCounters_, WallCycleCounter::VsiteConstr);
        }

        if (bNS && !(bFirstStep && ir->bContinuation))
        {
            /* Correct the new box if it is too skewed */
            const bool bMainState = inputrecDynamicBox(ir) && correct_box(fpLog_, step, state_->box);
            // If update is offloaded, and the box was changed either
            // above or in a replica exchange on the previous step,
            // the GPU Update object should be informed
            if (useGpuForUpdate && (bMainState || bExchanged))
            {
                integrator->setPbc(PbcType::Xyz, state_->box);
            }
            if (haveDDAtomOrdering(*cr_) && bMainState)
            {
                dd_collect_state(cr_->dd, state_, stateGlobal_);
            }

            if (haveDDAtomOrdering(*cr_))
            {
                appendPreDoForceStateTrace(activeM2pTraceDirPath(),
                                           step,
                                           "before_dd_partition_system",
                                           state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                           state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                           "src/gromacs/mdrun/md.cpp:before_dd_partition_system");
                const bool skipDdPartitionExperiment =
                        useExactVelocityVerletLammpsRespa(*ir) && skipDdPartitionForExactRespaExperiment();
                if (!skipDdPartitionExperiment)
                {
                    /* Repartition the domain decomposition */
                    dd_partition_system(fpLog_,
                                        mdLog_,
                                        step,
                                        cr_->dd,
                                        bMainState,
                                        stateGlobal_,
                                        topGlobal_,
                                        *ir,
                                        mdModulesNotifiers_,
                                        imdSession_,
                                        pullWork_,
                                        state_,
                                        &f,
                                        mdAtoms_,
                                        top_,
                                        fr_,
                                        virtualSites_,
                                        constr_,
                                        nrnb_,
                                        wallCycleCounters_,
                                        do_verbose && !(pmeLoadBal && pmeLoadBal->isPrintingLoad()));
                }
                appendPreDoForceStateTrace(activeM2pTraceDirPath(),
                                           step,
                                           "after_dd_partition_system",
                                           state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                           state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                           "src/gromacs/mdrun/md.cpp:after_dd_partition_system");
                if (!skipDdPartitionExperiment)
                {
                    upd.updateAfterPartition(state_->numAtoms(), md->cFREEZE, md->cTC, md->cACC);
                    fr_->longRangeNonbondeds->updateAfterPartition(*md);
                }
            }
        }
        appendPreDoForceStateTrace(activeM2pTraceDirPath(),
                                   step,
                                   "after_ns_partition_block",
                                   state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                   state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                   "src/gromacs/mdrun/md.cpp:after_ns_partition_block");

        // Allocate or re-size GPU halo exchange object, if necessary
        if (bNS && simulationWork.havePpDomainDecomposition && simulationWork.useGpuHaloExchange)
        {
            GMX_RELEASE_ASSERT(fr_->deviceStreamManager != nullptr,
                               "GPU device manager has to be initialized to use GPU "
                               "version of halo exchange.");
            constructGpuHaloExchange(
                    *cr_, *fr_->deviceStreamManager, wallCycleCounters_, simulationWork.useNvshmem);
        }
        appendPreDoForceStateTrace(activeM2pTraceDirPath(),
                                   step,
                                   "after_ns_setup",
                                   state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                   state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                   "src/gromacs/mdrun/md.cpp:after_ns_setup");

        if (isMainRank && do_log)
        {
            gmx::EnergyOutput::printHeader(fpLog_, step, t); /* can we improve the information printed here? */
        }

        if (ir->efep != FreeEnergyPerturbationType::No)
        {
            update_mdatoms(mdAtoms_->mdatoms(), state_->lambda[FreeEnergyPerturbationCouplingType::Mass]);
        }

        if (bExchanged)
        {
            /* We need the kinetic energy at minus the half step for determining
             * the full step kinetic energy and possibly for T-coupling.*/
            /* This may not be quite working correctly yet . . . . */
            int cgloFlagsExchanged = CGLO_GSTAT | CGLO_COMPUTEEKIN;
            compute_globals(gstat,
                            cr_->commMyGroup,
                            ir,
                            fr_,
                            ekind_,
                            makeConstArrayRef(state_->x),
                            makeConstArrayRef(state_->v),
                            state_->box,
                            md,
                            nrnb_,
                            &vcm,
                            wallCycleCounters_,
                            enerd_,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            &nullSignaller,
                            state_->box,
                            &bSumEkinhOld,
                            cgloFlagsExchanged,
                            step - 1, // Pass step-1 to indicate that v is from minus half a step
                            &observablesReducer);
        }
        clear_mat(force_vir);

        checkpointHandler->decideIfCheckpointingThisStep(bNS, bFirstStep, bLastStep);

        /* Determine the energy and pressure:
         * at nstcalcenergy steps and at energy output steps (set below).
         */

        const bool do_ene              = !suppressEnergyVirialForForceOnlyExactRespa
                              && (do_per_step(step, ir->nstenergy) || bLastStep);
        const bool needEnergyAndVirial = do_ene || do_log || bDoReplEx;

        const bool bCalcEnerStep =
                !suppressEnergyVirialForForceOnlyExactRespa && do_per_step(step, ir->nstcalcenergy);
        const bool bCalcVir      = [&]() -> bool
        {
            auto doPressureCoupling = [ir](int64_t s) -> bool
            {
                return ir->pressureCouplingOptions.epc != PressureCoupling::No
                       && do_per_step(s, ir->pressureCouplingOptions.nstpcouple);
            };
            const bool exactRespaOuterBoundaryPressureCoupling =
                    useExactVelocityVerletLammpsRespa(*ir)
                    && (ir->pressureCouplingOptions.epc == PressureCoupling::Berendsen
                        || ir->pressureCouplingOptions.epc == PressureCoupling::CRescale
                        || ir->pressureCouplingOptions.epc == PressureCoupling::Mttk);
            if (EI_VV(ir->eI) && (!bInitStep))
            {
                if (exactRespaOuterBoundaryPressureCoupling)
                {
                    // Standalone exact r-RESPA carries virial-producing long-range work only on
                    // outer-force boundaries. Requiring an extra pressure-coupling virial on the
                    // immediately following inner-only step would force non-outer semantics and
                    // reject every NPT setup with nstpcouple aligned to the outer cadence.
                    return bCalcEnerStep || needEnergyAndVirial || doPressureCoupling(step);
                }
                return bCalcEnerStep || needEnergyAndVirial || doPressureCoupling(step)
                       || doPressureCoupling(step - 1);
            }
            else
            {
                return bCalcEnerStep || needEnergyAndVirial || doPressureCoupling(step);
            }
        }();

        const bool bCalcEner = bCalcEnerStep || needEnergyAndVirial;

        // bCalcEner is only here for when the last step is not a multiple of nstfep
        const bool computeDHDL = ((ir->efep != FreeEnergyPerturbationType::No || ir->bSimTemp)
                                  && (do_per_step(step, nstfep) || bCalcEner));

        /* Do we need global communication ? */
        const bool bGStat =
                (bCalcVir || bCalcEner || bStopCM || do_per_step(step, nstglobalcomm)
                 || (EI_VV(ir->eI) && inputrecNvtTrotter(ir) && do_per_step(step - 1, nstglobalcomm)));

        unsigned int force_flags =
                (GMX_FORCE_STATECHANGED | GMX_FORCE_ALLFORCES | (bCalcVir ? GMX_FORCE_VIRIAL : 0)
                 | (bCalcEner ? GMX_FORCE_ENERGY : 0) | (computeDHDL ? GMX_FORCE_DHDL : 0));
        if ((simulationWork.useExactRespa || simulationWork.useLegacyMtsSubsteps())
            && !do_per_step(step, ir->nstfout))
        {
            // TODO: merge this with stepWork.useOnlyCombinedForceBuffer
            force_flags |= GMX_FORCE_DO_NOT_NEED_NORMAL_FORCE;
        }

        if (bNS)
        {
            if (fr_->listedForcesGpu)
            {
                fr_->listedForcesGpu->updateHaveInteractions(top_->idef);
            }
            runScheduleWork_->domainWork = setupDomainLifetimeWorkload(
                    *ir, *fr_, pullWork_, ed ? ed->getLegacyED() : nullptr, *md, simulationWork);
        }

        const int shellfcFlags = force_flags | (mdrunOptions_.verbose ? GMX_FORCE_ENERGY : 0);
        const int legacyForceFlags = ((shellfc) ? shellfcFlags : force_flags) | (bNS ? GMX_FORCE_NS : 0);

        runScheduleWork_->stepWork =
                gmx::useExactRespa(*ir)
                        ? setupExactRespaStepWorkload(
                                  legacyForceFlags, *ir, step, runScheduleWork_->domainWork, simulationWork)
                        : setupStepWorkload(
                                  legacyForceFlags, ir->mtsLevels, step, runScheduleWork_->domainWork, simulationWork);
        if (gmx::useExactRespa(*ir))
        {
            runScheduleWork_->exactRespaStepWork =
                    setupExactRespaStepWork(legacyForceFlags,
                                            *ir,
                                            step,
                                            runScheduleWork_->domainWork,
                                            simulationWork);
        }
        const bool reconstructExactRespaForceStore =
                gmx::useExactRespa(*ir) && bFirstStep
                && startingBehavior_ != StartingBehavior::NewSimulation
                && exactRespaForceStorePtr != nullptr && !exactRespaForceStorePtr->hasLevel(0)
                && gmx::exactRespaRestartRequiresForceStoreReconstruction(ir->exactRespa, step);
        MdrunScheduleWorkload forceRunScheduleWork = *runScheduleWork_;
        if (reconstructExactRespaForceStore)
        {
            // ExactRespaForceStore is process-local and is not serialized in checkpoints.
            // A continuation can start on an inner-only step, where the normal workload
            // assumes that the slow level totals from the previous outer step still exist.
            // Recompute every force level once at the checkpoint coordinates to rebuild the
            // store. An outer-boundary restart already schedules every level and must use that
            // normal path without a redundant bootstrap workload. Keep runScheduleWork_
            // unchanged so the resumed kick cadence still follows the actual checkpoint step.
            forceRunScheduleWork.stepWork =
                    setupExactRespaStepWorkload(legacyForceFlags,
                                                *ir,
                                                step,
                                                forceRunScheduleWork.domainWork,
                                                simulationWork,
                                                true);
            forceRunScheduleWork.exactRespaStepWork =
                    setupExactRespaStepWork(legacyForceFlags,
                                            *ir,
                                            step,
                                            forceRunScheduleWork.domainWork,
                                            simulationWork,
                                            true);
            if (isMainRank)
            {
                FILE* report = (fpLog_ != nullptr) ? fpLog_ : stderr;
                std::fprintf(report,
                             "Reconstructing all exact r-RESPA force levels after checkpoint "
                             "restart at step %lld.\n",
                             static_cast<long long>(step));
            }
        }
        appendPreDoForceStateTrace(activeM2pTraceDirPath(),
                                   step,
                                   "after_stepwork_setup",
                                   state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                   state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                   "src/gromacs/mdrun/md.cpp:after_stepwork_setup");

        const bool useExactVelocityVerletRespa =
                useExactVelocityVerletLammpsRespa(*ir);
        const bool useSupportedExactVelocityVerletRespa =
                useExactVelocityVerletRespa
                && canUseExactLammpsRespaVelocityVerlet(*ir,
                                                        simulationWork,
                                                        runScheduleWork_->domainWork,
                                                        shellfc,
                                                        constr_,
                                                        virtualSites_,
                                                        useReplicaExchange);
        const bool useExactRespaGpuNvtKineticReduction =
                useSupportedExactVelocityVerletRespa && useGpuForUpdate
                && exactRespaDeviceKickGpuUpdateProbeEnabled()
                && exactRespaFusedNvtTrotterGpuUpdateProbeEnabled()
                && exactRespaSparseNvtObservablesGpuUpdateProbeEnabled()
                && exactRespaGpuNvtKineticReductionProbeEnabled() && inputrecNvtTrotter(ir)
                && ir->opts.ngtc == 1 && cr_->commMySim.isSerial()
                && ir->comm_mode == ComRemovalAlgorithm::No && !fr_->haveBoxDeformation
                && md->nMassPerturbed == 0 && sizeof(real) == sizeof(float)
                && exactRespaGpuUpdater_ != nullptr;
        if (useExactVelocityVerletRespa && !useSupportedExactVelocityVerletRespa)
        {
            gmx_fatal(FARGS,
                      "Exact LAMMPS-style r-RESPA with integrator = %s currently only supports "
                      "NVE/NVT/NPT runs without constraints, unsupported COM removal, domain decomposition, virtual "
                      "sites, replica exchange, or other special-force modules, with either "
                      "CPU-only execution or the canonical GPU milestone layouts "
                      "(nb gpu; nb gpu + bonded gpu; nb gpu + bonded gpu + pme gpu; "
                      "nb gpu + bonded gpu + pme gpu + update gpu). "
                      "The supported thermostat/barostat subset is tcoupl = no/berendsen/v-rescale/nose-hoover "
                      "and pcoupl = no/berendsen/c-rescale/mttk. Linear COM removal is admitted only "
                      "when GMX_PCFF_EXACT_RESPA_ALLOW_LINEAR_COM_REMOVAL is enabled.",
                      enumValueToString(IntegrationAlgorithm::VV));
        }
        if (gmx::useExactRespa(*ir) && bCalcVir)
        {
            const int outerLevel = gmx::exactRespaNonbondedOuterLevel(*ir);
            if (outerLevel > runScheduleWork_->exactRespaStepWork.highestActiveLevel)
            {
                const std::string message = formatString(
                        "Exact LAMMPS-style r-RESPA currently requires virial-carrying steps "
                        "(energy output, final step, or pressure-coupling steps) to land on "
                        "an outer-force boundary. Step %lld activates only up to MTS level %d, "
                        "but the outer nonbonded level is %d.",
                        static_cast<long long>(step),
                        runScheduleWork_->exactRespaStepWork.highestActiveLevel + 1,
                        outerLevel + 1);
                gmx_fatal(FARGS, "%s", message.c_str());
            }
        }
        const bool doTemperatureScaling = (ir->etc != TemperatureCoupling::No
                                           && do_per_step(step + ir->nsttcouple - 1, ir->nsttcouple));

        /* With leap-frog type integrators we compute the kinetic energy
         * at a whole time step as the average of the half-time step kinetic
         * energies of two subsequent steps. Therefore we need to compute the
         * half step kinetic energy also if we need energies at the next step.
         */
        const bool needHalfStepKineticEnergy =
                (!EI_VV(ir->eI) && (do_per_step(step + 1, nstglobalcomm) || step_rel + 1 == ir->nsteps));

        // Parrinello-Rahman requires the pressure to be available before the update to compute
        // the velocity scaling matrix. Hence, it runs one step after the nstpcouple step.
        const bool doParrinelloRahman =
                (ir->pressureCouplingOptions.epc == PressureCoupling::ParrinelloRahman
                 && do_per_step(step + ir->pressureCouplingOptions.nstpcouple - 1,
                                ir->pressureCouplingOptions.nstpcouple));

        MdGpuGraph* mdGraph = simulationWork.useMdGpuGraph ? fr_->mdGraph[step % 2].get() : nullptr;

        if (simulationWork.useMdGpuGraph)
        {
            // Reset graph on search step (due to changing neighbour list etc)
            // or virial step (due to changing shifts and box).
            if (bNS || bCalcVir)
            {
                fr_->mdGraph[MdGraphEvenOrOddStep::EvenStep]->reset();
                fr_->mdGraph[MdGraphEvenOrOddStep::OddStep]->reset();
            }
            else
            {
                mdGraph->setUsedGraphLastStep(usedMdGpuGraphLastStep);
                bool canUseMdGpuGraphThisStep =
                        !bNS && !bCalcVir && !doTemperatureScaling && !doParrinelloRahman && !bGStat
                        && !needHalfStepKineticEnergy && !do_per_step(step, ir->nstxout)
                        && !do_per_step(step, ir->nstxout_compressed)
                        && !do_per_step(step, ir->nstvout) && !do_per_step(step, ir->nstfout)
                        && !checkpointHandler->isCheckpointingStep();
                if (mdGraph->captureThisStep(canUseMdGpuGraphThisStep))
                {
                    mdGraph->startRecord(stateGpu->getCoordinatesReadyOnDeviceEvent(
                            AtomLocality::Local, simulationWork, runScheduleWork_->stepWork));
                }
            }
        }
        if (!simulationWork.useMdGpuGraph || mdGraph->graphIsCapturingThisStep()
            || !mdGraph->useGraphThisStep())
        {
            appendPreDoForceStateTrace(activeM2pTraceDirPath(),
                                       step,
                                       "before_force_branch",
                                       state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                       state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                       "src/gromacs/mdrun/md.cpp:before_force_branch");
            if (step == 5)
            {
                appendMdLoopBoundarySnapshotPair(activeM2pTraceDirPath(),
                                                useSupportedExactVelocityVerletRespa ? "PATCH" : "PLAIN",
                                                "STEP5_LOOP_ENTRY_STATE_X",
                                                step,
                                                state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                                "src/gromacs/mdrun/md.cpp:STEP5_LOOP_ENTRY_STATE_X");
            }

            if (shellfc)
            {
                /* Now is the time to relax the shells */
                relax_shell_flexcon(fpLog_,
                                    cr_,
                                    mdrunOptions_.verbose,
                                    enforcedRotation_,
                                    step,
                                    ir,
                                    mdModulesNotifiers_,
                                    imdSession_,
                                    pullWork_,
                                    bNS,
                                    top_,
                                    constr_,
                                    enerd_,
                                    state_->numAtoms(),
                                    state_->x.arrayRefWithPadding(),
                                    state_->v.arrayRefWithPadding(),
                                    state_->box,
                                    state_->lambda,
                                    &state_->hist,
                                    &f.view(),
                                    force_vir,
                                    *md,
                                    fr_->longRangeNonbondeds.get(),
                                    nrnb_,
                                    wallCycleCounters_,
                                    shellfc,
                                    fr_,
                                    *runScheduleWork_,
                                    t,
                                    mu_tot,
                                    virtualSites_,
                                    ddBalanceRegionHandler);
            }
            else
            {
                /* The AWH history need to be saved _before_ doing force calculations where the AWH bias
                   is updated (or the AWH update will be performed twice for one step when continuing).
                   It would be best to call this update function from do_md_trajectory_writing but that
                   would occur after do_force. One would have to divide the update_awh function into one
                   function applying the AWH force and one doing the AWH bias update. The update AWH
                   bias function could then be called after do_md_trajectory_writing (then containing
                   update_awh_history). The checkpointing will in the future probably moved to the start
                   of the md loop which will rid of this issue. */
                if (awh && checkpointHandler->isCheckpointingStep() && isMainRank)
                {
                    awh->updateHistory(stateGlobal_->awhHistory.get());
                }

                /* The coordinates (x) are shifted (to get whole molecules)
                 * in do_force.
                 * This is parallellized as well, and does communication too.
                 * Check comments in sim_util.c
                 */
                {
                    const bool exactRespaGpuForceWorkPresent =
                            simulationWork.useGpuNonbonded || simulationWork.useGpuBonded
                            || simulationWork.useGpuPme;
                    const bool canReuseExactRespaLiveNeighborSearchForce =
                            exactRespaGpuForceWorkPresent
                            && exactRespaReuseNextForceOnNeighborSearchEnabled() && bNS && bNStList
                            && !bExchanged && !bNeedRepartition
                            && !simulationWork.havePpDomainDecomposition;
                    const int exactRespaLongRangeLevel =
                            gmx::useExactRespa(*ir) ? gmx::exactRespaLongrangeNonbondedLevel(*ir) : -1;
                    const int exactRespaForceStoreLevels =
                            (exactRespaForceStorePtr != nullptr) ? exactRespaForceStorePtr->numLevels()
                                                                 : 0;
                    const bool exactRespaForceStoreHasLevel0 =
                            exactRespaForceStorePtr != nullptr && exactRespaForceStorePtr->hasLevel(0);
                    const bool exactRespaForceStoreHasLongRange =
                            exactRespaForceStorePtr != nullptr && exactRespaLongRangeLevel >= 0
                            && exactRespaLongRangeLevel < exactRespaForceStorePtr->numLevels()
                            && exactRespaForceStorePtr->hasLevel(exactRespaLongRangeLevel);
                    const bool canReuseExactRespaLiveLongRangeForce =
                            exactRespaReuseNextForceForLongRangeEnabled() && !bNS
                            && exactRespaForceStoreHasLongRange;
                    const bool haveM2pTrace              = activeM2pTraceDirPath() != nullptr;
                    const bool haveTotalForceDump        = totalForceDumpFilePath() != nullptr;
                    const bool havePerLevelForceDump     = perLevelForceDumpFilePath() != nullptr;
                    const bool haveMtsCombinedForceDump  = mtsCombinedForceDumpFilePath() != nullptr;
                    const bool canReuseExactRespaLiveForce =
                            exactRespaReuseNextForceForLiveStepEnabled()
                            && useSupportedExactVelocityVerletRespa
                            && (!exactRespaGpuForceWorkPresent
                                || exactRespaReuseNextForceWithGpuEnabled())
                            && (!simulationWork.useGpuUpdate
                                || exactRespaResidentXGpuUpdateProbeEnabled())
                            && exactRespaForceStorePtr != nullptr
                            && exactRespaForceStoreHasLevel0
                            && exactRespaForceStoreLevels > 0
                            && !bFirstStep
                            && (!bNS || canReuseExactRespaLiveNeighborSearchForce)
                            && (!runScheduleWork_->stepWork.doNeighborSearch
                                || canReuseExactRespaLiveNeighborSearchForce)
                            && !runScheduleWork_->stepWork.computeEnergy
                            && !runScheduleWork_->stepWork.computeVirial
                            && !runScheduleWork_->stepWork.computeDhdl
                            && (!runScheduleWork_->stepWork.computeLongRangeNonbondedForces
                                || canReuseExactRespaLiveNeighborSearchForce
                                || canReuseExactRespaLiveLongRangeForce)
                            && !bCalcEner && !bCalcVir && !computeDHDL && !bStopCM
                            && shellfc == nullptr
                            && awh == nullptr && ed == nullptr && constr_ == nullptr
                            && virtualSites_ == nullptr && !haveM2pTrace && !haveTotalForceDump
                            && !havePerLevelForceDump && !haveMtsCombinedForceDump;
                    appendExactRespaReuseDecisionTrace(
                            step,
                            canReuseExactRespaLiveForce,
                            exactRespaReuseNextForceForLiveStepEnabled(),
                            useSupportedExactVelocityVerletRespa,
                            exactRespaGpuForceWorkPresent,
                            exactRespaReuseNextForceWithGpuEnabled(),
                            simulationWork.useGpuUpdate,
                            exactRespaForceStorePtr != nullptr,
                            exactRespaForceStoreLevels,
                            exactRespaForceStoreHasLevel0,
                            exactRespaLongRangeLevel,
                            exactRespaForceStoreHasLongRange,
                            canReuseExactRespaLiveNeighborSearchForce,
                            canReuseExactRespaLiveLongRangeForce,
                            bFirstStep,
                            bNS,
                            bNStList,
                            bExchanged,
                            bNeedRepartition,
                            simulationWork.havePpDomainDecomposition,
                            runScheduleWork_->stepWork.doNeighborSearch,
                            runScheduleWork_->stepWork.computeEnergy,
                            runScheduleWork_->stepWork.computeVirial,
                            runScheduleWork_->stepWork.computeDhdl,
                            runScheduleWork_->stepWork.computeLongRangeNonbondedForces,
                            bCalcEner,
                            bCalcVir,
                            computeDHDL,
                            bStopCM,
                            bGStat,
                            shellfc != nullptr,
                            awh != nullptr,
                            ed != nullptr,
                            constr_ != nullptr,
                            virtualSites_ != nullptr,
                            haveM2pTrace,
                            haveTotalForceDump,
                            havePerLevelForceDump,
                            haveMtsCombinedForceDump);
                    if (canReuseExactRespaLiveForce)
                    {
                        const bool restoreCombinedForce =
                                runScheduleWork_->exactRespaStepWork.haveSlowForceLevels
                                && runScheduleWork_->stepWork.computeSlowForces;
                        restoreExactRespaForcesFromStore(
                                *ir,
                                *exactRespaForceStorePtr,
                                f.view().force(),
                                restoreCombinedForce ? f.view().forceMtsCombined() : gmx::ArrayRef<gmx::RVec>{});
                    }
                    else
                    {
                        const char* previousForceContextLabel = g_respaDoForceContextLabel;
                        g_respaDoForceContextLabel           = "live_main_step";
                        appendPreDoForceStateTrace(activeM2pTraceDirPath(),
                                                   step,
                                                   g_respaDoForceContextLabel,
                                                   state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                                   state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                                   "src/gromacs/mdrun/md.cpp:live_main_step_pre_do_force");
                        do_force(fpLog_,
                                 cr_,
                                 *ir,
                                 mdModulesNotifiers_,
                                 awh.get(),
                                 enforcedRotation_,
                                 imdSession_,
                                 pullWork_,
                                 step,
                                 nrnb_,
                                 wallCycleCounters_,
                                 top_,
                                 state_->box,
                                 state_->x.arrayRefWithPadding(),
                                 state_->v.arrayRefWithPadding().unpaddedArrayRef(),
                                 &state_->hist,
                                 &f.view(),
                                 exactRespaForceStorePtr,
                                 force_vir,
                                 md,
                                 enerd_,
                                 state_->lambda,
                                 fr_,
                                 forceRunScheduleWork,
                                 virtualSites_,
                                 mu_tot,
                                 t,
                                 ed ? ed->getLegacyED() : nullptr,
                                 fr_->longRangeNonbondeds.get(),
                                 ddBalanceRegionHandler);
                        g_respaDoForceContextLabel = previousForceContextLabel;
                    }
                }
            }

            appendTp18eTraceRow(step,
                                "after_do_force_return",
                                f.view().forceWithPadding().unpaddedConstArrayRef(),
                                false,
                                makeConstArrayRef(state_->x),
                                makeConstArrayRef(state_->v),
                                gmx::ArrayRef<const gmx::RVec>{},
                                false,
                                force_vir,
                                shake_vir,
                                total_vir,
                                pres,
                                enerd_);
            if (step == 5)
            {
                appendMdLoopBoundarySnapshotPair(activeM2pTraceDirPath(),
                                                useSupportedExactVelocityVerletRespa ? "PATCH" : "PLAIN",
                                                "STEP5_POST_FORCE_STATE_X",
                                                step,
                                                state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                                "src/gromacs/mdrun/md.cpp:STEP5_POST_FORCE_STATE_X");
            }

            maybeDumpTotalForceForDiagnostics(*ir, step, t, &f.view(), *runScheduleWork_);
            maybeDumpPerLevelForceForDiagnostics(
                    *ir, step, t, &f.view(), exactRespaForceStorePtr, *runScheduleWork_);
            maybeDumpMtsCombinedForceForDiagnostics(*ir, step, t, &f.view(), *runScheduleWork_);

            // VV integrators do not need the following velocity half step
            // if it is the first step after starting from a checkpoint.
            // That is, the half step is needed on all other steps, and
            // also the first step when starting from a .tpr file.
            if (EI_VV(ir->eI))
            {
                if (useSupportedExactVelocityVerletRespa)
                {
                    const PcffExactRespaTrotterReplay upcomingPreTrotterReplay =
                            pcffExactRespaTrotterReplayFromEnv(
                                    "GMX_PCFF_EXACT_RESPA_PRE_TROTTER",
                                    PcffExactRespaTrotterReplay::Two);
                    const bool upcomingPreTrotterCouples =
                            bTrotter
                            && pcffExactRespaTrotterReplayCouplesThisStep(
                                    *ir, step, upcomingPreTrotterReplay);
                    const bool useSparseNvtObservables =
                            useGpuForUpdate && exactRespaDeviceKickGpuUpdateProbeEnabled()
                            && exactRespaSparseNvtObservablesGpuUpdateProbeEnabled()
                            && inputrecNvtTrotter(ir);
                    // Match VV's initial/checkpoint boundary: the supplied full-step
                    // velocities are retained on the first iteration. COM removal is
                    // applied once on subsequent scheduled full-step boundaries.
                    const bool stopExactRespaCenterOfMass = bStopCM && !bInitStep;
                    const bool prepareCpuVelocityVerletObservables =
                            !useSparseNvtObservables || bFirstStep || bCalcEner || bCalcVir
                            || stopExactRespaCenterOfMass
                            || (upcomingPreTrotterCouples
                                && !exactRespaGpuNvtKineticReadyForPreTrotter);
                    if (prepareCpuVelocityVerletObservables
                        && useGpuForUpdate && exactRespaDeviceKickGpuUpdateProbeEnabled()
                        && !exactRespaDeviceKickHostVelocitiesCurrent)
                    {
                        completeExactRespaPendingGpuNvtPostTrotter(bGStat);
                        applyExactRespaPendingPostTrotterScaleToDevice();
                        stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
                        stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
                        exactRespaDeviceKickHostVelocitiesCurrent = true;
                    }
                    if (prepareCpuVelocityVerletObservables)
                    {
                        prepareExactRespaVelocityVerletObservablesForStep(
                                *ir,
                                step,
                                cr_->commMyGroup,
                                *mdAtoms_->mdatoms(),
                                nrnb_,
                                &vcm,
                                enerd_,
                                gstat,
                                &nullSignaller,
                                &observablesReducer,
                                force_vir,
                                shake_vir,
                                total_vir,
                                pres,
                                bCalcEner,
                                bCalcVir,
                                bGStat,
                                stopExactRespaCenterOfMass,
                                &bSumEkinhOld,
                                &saved_conserved_quantity,
                                &last_ekin);
                        if (stopExactRespaCenterOfMass && useGpuForUpdate)
                        {
                            // A resident device kick must consume the corrected velocities.
                            stateGpu->copyVelocitiesToGpu(state_->v, AtomLocality::Local);
                            exactRespaDeviceKickPendingPostTrotterScale = 1.0F;
                            exactRespaDeviceKickHostVelocitiesCurrent = true;
                        }
                        // integrateVVFirstStep() normally preserves the current virial in the
                        // checkpoint state for the first step after an MTTK restart. The exact
                        // r-RESPA path bypasses that routine, so mirror its checkpoint hand-off
                        // here whenever this step actually produced a virial. Without this, the
                        // restart kludge below restores an all-zero tensor and writes a
                        // kinetic-only pressure for the checkpoint-boundary energy frame.
                        if (bCalcVir && (inputrecNptTrotter(ir) || inputrecNphTrotter(ir)))
                        {
                            copy_mat(shake_vir, state_->svir_prev);
                            copy_mat(force_vir, state_->fvir_prev);
                        }
                        exactRespaGpuNvtKineticReadyForPreTrotter = false;
                    }
                }
                else
                {
                    integrateVVFirstStep(step,
                                         bFirstStep,
                                         bInitStep,
                                         startingBehavior_,
                                         nstglobalcomm,
                                         ir,
                                         fr_,
                                         cr_->commMyGroup,
                                         cr_->dd,
                                         state_,
                                         mdAtoms_->mdatoms(),
                                         &fcdata,
                                         &MassQ,
                                         &vcm,
                                         enerd_,
                                         &observablesReducer,
                                         ekind_,
                                         gstat,
                                         &last_ekin,
                                         bCalcVir,
                                         total_vir,
                                         shake_vir,
                                         force_vir,
                                         pres,
                                         do_log,
                                         do_ene,
                                         bCalcEner,
                                         bGStat,
                                         bStopCM,
                                         bTrotter,
                                         bExchanged,
                                         &bSumEkinhOld,
                                         &saved_conserved_quantity,
                                         &f,
                                         &upd,
                                         constr_,
                                         &nullSignaller,
                                         trotter_seq,
                                         nrnb_,
                                         fpLog_,
                                         wallCycleCounters_);
                    if (virtualSites_ != nullptr && needVirtualVelocitiesThisStep)
                    {
                        // Positions were calculated earlier
                        wallcycle_start(wallCycleCounters_, WallCycleCounter::VsiteConstr);
                        virtualSites_->construct(state_->x, state_->v, state_->box, VSiteOperation::Velocities);
                        wallcycle_stop(wallCycleCounters_, WallCycleCounter::VsiteConstr);
                    }
                }
            }

            /* ########  END FIRST UPDATE STEP  ############## */
            /* ########  If doing VV, we now have v(dt) ###### */
            if (bDoExpanded)
            {
                /* perform extended ensemble sampling in lambda - we don't
                   actually move to the new state before outputting
                   statistics, but if performing simulated tempering, we
                   do update the velocities and the tau_t. */
                lamnew = ExpandedEnsembleDynamics(fpLog_,
                                                  *inputRec_,
                                                  *enerd_,
                                                  ekind_,
                                                  state_,
                                                  &MassQ,
                                                  state_->fep_state,
                                                  state_->dfhist.get(),
                                                  step,
                                                  state_->v.rvec_array(),
                                                  md->homenr,
                                                  md->cTC);
                /* history is maintained in state->dfhist, but state_global is what is sent to trajectory and log output */
                if (isMainRank)
                {
                    *stateGlobal_->dfhist = *state_->dfhist;
                }
            }

            // Copy coordinate from the GPU for the output/checkpointing if the update is offloaded
            // and coordinates have not already been copied for i) search or ii) CPU force tasks.
            if (useGpuForUpdate && !bNS && !runScheduleWork_->domainWork.haveCpuLocalForceWork
                && (do_per_step(step, ir->nstxout) || do_per_step(step, ir->nstxout_compressed)
                    || checkpointHandler->isCheckpointingStep()))
            {
                stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
            }
            // Copy velocities if needed for the output/checkpointing.
            // NOTE: Copy on the search steps is done at the beginning of the step.
            if (useGpuForUpdate && !bNS
                && (do_per_step(step, ir->nstvout) || checkpointHandler->isCheckpointingStep())
                && (!exactRespaDeviceKickGpuUpdateProbeEnabled()
                    || !exactRespaDeviceKickHostVelocitiesCurrent))
            {
                completeExactRespaPendingGpuNvtPostTrotter(bGStat);
                applyExactRespaPendingPostTrotterScaleToDevice();
                stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
                stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
                exactRespaDeviceKickHostVelocitiesCurrent = true;
            }
            // Copy forces for the output if the forces were reduced on the GPU (not the case on virial steps)
            // and update is offloaded hence forces are kept on the GPU for update and have not been
            // already transferred in do_force().
            // TODO: There should be an improved, explicit mechanism that ensures this copy is only executed
            //       when the forces are ready on the GPU -- the same synchronizer should be used as the one
            //       prior to GPU update.
            // TODO: When the output flags will be included in step workload, this copy can be combined with the
            //       copy call in do_force(...).
            // NOTE: The forces should not be copied here if the vsites are present, since they were modified
            //       on host after the D2H copy in do_force(...).
            if (runScheduleWork_->stepWork.useGpuFBufferOps
                && (simulationWork.useGpuUpdate && !virtualSites_) && do_per_step(step, ir->nstfout))
            {
                stateGpu->copyForcesFromGpu(f.view().force(), AtomLocality::Local);
                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
            }
            /* Now we have the energies and forces corresponding to the
             * coordinates at time t. We must output all of this before
             * the update.
             */
            const EkindataState ekindataState =
                    bGStat ? (bSumEkinhOld ? EkindataState::UsedNeedToReduce
                                           : EkindataState::UsedDoNotNeedToReduce)
                           : EkindataState::NotUsed;
            do_md_trajectory_writing(fpLog_,
                                     cr_,
                                     nFile_,
                                     fnm_,
                                     step,
                                     step_rel,
                                     t,
                                     ir,
                                     state_,
                                     stateGlobal_,
                                     observablesHistory_,
                                     topGlobal_,
                                     fr_,
                                     outf,
                                     energyOutput,
                                     ekind_,
                                     f.view().force(),
                                     checkpointHandler->isCheckpointingStep(),
                                     bRerunMD,
                                     bLastStep,
                                     mdrunOptions_.writeConfout,
                                     ekindataState);
            appendLocalGlobalStateAliasTrace(activeM2pTraceDirPath(),
                                             step,
                                             "after_trajectory_writing",
                                             state_,
                                             stateGlobal_,
                                             "src/gromacs/mdrun/md.cpp:after_trajectory_writing");
            appendPreDoForceStateTrace(activeM2pTraceDirPath(),
                                       step,
                                       "after_trajectory_writing",
                                       state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                       state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                       "src/gromacs/mdrun/md.cpp:after_trajectory_writing");
            /* Check if IMD step and do IMD communication, if bIMD is TRUE. */
            bInteractiveMDstep = imdSession_->run(step, bNS, state_->box, state_->x, t);
            appendPreDoForceStateTrace(activeM2pTraceDirPath(),
                                       step,
                                       "after_imd_run",
                                       state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                       state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                       "src/gromacs/mdrun/md.cpp:after_imd_run");

            /* kludge -- virial is lost with restart for MTTK NPT control. Must reload (saved earlier). */
            if (startingBehavior_ != StartingBehavior::NewSimulation && bFirstStep
                && (inputrecNptTrotter(ir) || inputrecNphTrotter(ir)))
            {
                copy_mat(state_->svir_prev, shake_vir);
                copy_mat(state_->fvir_prev, force_vir);
            }

            stopHandler->setSignal();
            resetHandler->setSignal(wallTimeAccounting_);

            if (bGStat || cr_->commMySim.isSerial())
            {
                /* In parallel we only have to check for checkpointing in steps
                 * where we do global communication,
                 *  otherwise the other nodes don't know.
                 */
                checkpointHandler->setSignal(wallTimeAccounting_);
            }

            /* #########   START SECOND UPDATE STEP ################# */

            /* at the start of step, randomize or scale the velocities ((if vv. Restriction of
               Andersen controlled in preprocessing */

            if (ETC_ANDERSEN(ir->etc)) /* keep this outside of update_tcouple because of the extra info required to pass */
            {
                gmx_bool bIfRandomize;
                bIfRandomize = update_randomize_velocities(
                        ir, step, cr_->dd, md->homenr, md->cTC, md->invmass, state_->v, &upd, constr_);
                /* if we have constraints, we have to remove the kinetic energy parallel to the bonds */
                if (constr_ && bIfRandomize)
                {
                    constrain_velocities(constr_, do_log || do_ene, step, state_, nullptr, false, nullptr);
                }
            }
            /* Box is changed in update() when we do pressure coupling,
             * but we should still use the old box for energy corrections and when
             * writing it to the energy file, so it matches the trajectory files for
             * the same timestep above. Make a copy in a separate array.
             */
            copy_mat(state_->box, lastbox);

            dvdl_constr = 0;

            if (!useGpuForUpdate)
            {
                wallcycle_start(wallCycleCounters_, WallCycleCounter::Update);
            }
            if (isMainRank)
            {
                appendPcffMttkStateTraceRow(
                        step, t, "before_pre_trotter", *ir, state_, pres, total_vir, enerd_);
            }
            /* UPDATE PRESSURE VARIABLES IN TROTTER FORMULATION WITH CONSTRAINTS */
            const auto replayExactRespaTrotter = [&](const PcffExactRespaTrotterReplay replay,
                                                     const bool skipHostVelocityScaling)
            {
                double velocityScale = 1.0;
                const auto runTrotterPart = [&](const TrotterSequence sequence)
                {
                    trotter_update(ir,
                                   step,
                                   ekind_,
                                   state_,
                                   total_vir,
                                   skipHostVelocityScaling ? 0 : md->homenr,
                                   md->cTC,
                                   md->invmass,
                                   &MassQ,
                                   trotter_seq,
                                   sequence);
                    if (ekind_->numTemperatureCouplingGroups() == 1)
                    {
                        velocityScale *= ekind_->tcstat[0].vscale_nhc;
                    }
                };

                switch (replay)
                {
                    case PcffExactRespaTrotterReplay::None: break;
                    case PcffExactRespaTrotterReplay::Two: runTrotterPart(TrotterSequence::Two); break;
                    case PcffExactRespaTrotterReplay::Three: runTrotterPart(TrotterSequence::Three); break;
                    case PcffExactRespaTrotterReplay::TwoThenThree:
                        runTrotterPart(TrotterSequence::Two);
                        runTrotterPart(TrotterSequence::Three);
                        break;
                    case PcffExactRespaTrotterReplay::ThreeThenTwo:
                        runTrotterPart(TrotterSequence::Three);
                        runTrotterPart(TrotterSequence::Two);
                        break;
                }
                return velocityScale;
            };

            float  exactRespaDeviceKickPreTrotterScale = 1.0F;
            bool   useExactRespaFusedNvtTrotterScaling = false;

            if (bTrotter)
            {
                const PcffExactRespaTrotterReplay preTrotterReplay =
                        useSupportedExactVelocityVerletRespa
                                ? pcffExactRespaTrotterReplayFromEnv(
                                          "GMX_PCFF_EXACT_RESPA_PRE_TROTTER",
                                          PcffExactRespaTrotterReplay::Two)
                                : PcffExactRespaTrotterReplay::Three;
                const bool preTrotterCouplesThisStep =
                        pcffExactRespaTrotterReplayCouplesThisStep(
                                *ir, step, preTrotterReplay);
                useExactRespaFusedNvtTrotterScaling =
                        useSupportedExactVelocityVerletRespa && useGpuForUpdate
                        && exactRespaDeviceKickGpuUpdateProbeEnabled()
                        && exactRespaFusedNvtTrotterGpuUpdateProbeEnabled()
                        && inputrecNvtTrotter(ir) && ir->opts.ngtc == 1
                        && (preTrotterReplay == PcffExactRespaTrotterReplay::Two
                            || preTrotterReplay == PcffExactRespaTrotterReplay::Three);
                const bool gpuStateWillBeResetFromHost =
                        bNS && (bFirstStep || haveDDAtomOrdering(*cr_) || bExchanged);
                const bool skipPreTrotterHostVelocityScaling =
                        useExactRespaGpuNvtKineticReduction && !gpuStateWillBeResetFromHost;
                if (useSupportedExactVelocityVerletRespa && useGpuForUpdate
                    && exactRespaDeviceKickGpuUpdateProbeEnabled()
                    && preTrotterCouplesThisStep && !bFirstStep
                    && !exactRespaGpuNvtKineticReadyForPreTrotter
                    && !exactRespaDeviceKickHostVelocitiesCurrent)
                {
                    stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
                    stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
                    exactRespaDeviceKickHostVelocitiesCurrent = true;
                }
                if (preTrotterCouplesThisStep && exactRespaGpuNvtPostTrotterPending)
                {
                    completeExactRespaPendingGpuNvtPostTrotter(bGStat);
                }
                const double preTrotterVelocityScale =
                        replayExactRespaTrotter(preTrotterReplay,
                                               skipPreTrotterHostVelocityScaling);
                if (useSupportedExactVelocityVerletRespa && useGpuForUpdate
                    && exactRespaDeviceKickGpuUpdateProbeEnabled()
                    && preTrotterCouplesThisStep)
                {
                    if (useExactRespaFusedNvtTrotterScaling)
                    {
                        exactRespaDeviceKickPreTrotterScale =
                                static_cast<float>(preTrotterVelocityScale);
                    }
                    else
                    {
                        stateGpu->copyVelocitiesToGpu(state_->v, AtomLocality::Local);
                        exactRespaDeviceKickPendingPostTrotterScale = 1.0F;
                    }
                    exactRespaDeviceKickHostVelocitiesCurrent =
                            !skipPreTrotterHostVelocityScaling;
                    exactRespaGpuNvtKineticReadyForPreTrotter = false;
                }
                if (isMainRank)
                {
                    appendPcffMttkStateTraceRow(
                            step, t, "after_pre_trotter", *ir, state_, pres, total_vir, enerd_);
                }
                /* We can only do Berendsen coupling after we have summed
                 * the kinetic energy or virial. Since the happens
                 * in global_state after update, we should only do it at
                 * step % nstlist = 1 with bGStatEveryStep=FALSE.
                 */
            }
            else
            {
                update_tcouple(step, ir, state_, ekind_, &MassQ, md->homenr, md->cTC);
                update_pcouple_before_coordinates(mdLog_,
                                                  step,
                                                  ir->pressureCouplingOptions,
                                                  ir->deform,
                                                  ir->delta_t,
                                                  state_,
                                                  &pressureCouplingMu,
                                                  &parrinelloRahmanM);
            }

            if (EI_VV(ir->eI))
            {
                if (useSupportedExactVelocityVerletRespa)
                {
                    if (useGpuForUpdate)
                    {
                        GMX_RELEASE_ASSERT(
                                integrator != nullptr && stateGpu != nullptr,
                                "Exact r-RESPA GPU update requires initialized GPU state.");
                        if (bNS && (bFirstStep || haveDDAtomOrdering(*cr_) || bExchanged))
                        {
                            integrator->set(stateGpu->getCoordinates(),
                                            stateGpu->getVelocities(),
                                            stateGpu->getForces(),
                                            top_->idef,
                                            *md);

                            // Search steps may reallocate GPU state buffers; velocities must be
                            // recopied so the half-kick starts from the current host state.
                            stateGpu->copyVelocitiesToGpu(state_->v, AtomLocality::Local);
                            exactRespaDeviceKickPendingPostTrotterScale = 1.0F;
                            exactRespaDeviceKickPreTrotterScale        = 1.0F;
                            exactRespaGpuNvtKineticReadyForPreTrotter = false;

                            if (!(runScheduleWork_->stepWork.haveGpuPmeOnThisRank
                                  || runScheduleWork_->stepWork.useGpuXBufferOps))
                            {
                                stateGpu->copyCoordinatesToGpu(state_->x, AtomLocality::Local);
                                stateGpu->consumeCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
                            }
                        }
                    }
#if GMX_GPU_CUDA
                    if (useGpuForUpdate && useExactRespaFusedNvtTrotterScaling)
                    {
                        exactRespaGpuUpdater_->setExactRespaVelocityScaling(
                                exactRespaDeviceKickPendingPostTrotterScale,
                                exactRespaDeviceKickPreTrotterScale);
                    }
#endif
                    dispatchExactRespaVelocityVerletStep(*ir,
                                                         step,
                                                         t,
                                                         *md,
                                                         simulationWork,
                                                         runScheduleWork_->domainWork,
                                                         f,
                                                         exactRespaForceStore,
                                                         force_vir,
                                                         mu_tot,
                                                         *enerd_,
                                                         awh.get(),
                                                         ed ? ed->getLegacyED() : nullptr,
                                                         ddBalanceRegionHandler);
                    exactRespaDeviceKickPendingPostTrotterScale = 1.0F;
                    if (useGpuForUpdate && exactRespaDeviceKickGpuUpdateProbeEnabled())
                    {
                        exactRespaDeviceKickHostVelocitiesCurrent = false;
                    }
                    if (isMainRank)
                    {
                        appendPcffMttkStateTraceRow(
                                step, t, "after_respa_step", *ir, state_, pres, total_vir, enerd_);
                    }
                    const PcffExactRespaTrotterReplay postTrotterReplay =
                            pcffExactRespaTrotterReplayFromEnv(
                                    "GMX_PCFF_EXACT_RESPA_POST_TROTTER",
                                    PcffExactRespaTrotterReplay::Three);
                    const bool postTrotterCouplesThisStep =
                            pcffExactRespaTrotterReplayCouplesThisStep(*ir, step, postTrotterReplay);
                    if (bTrotter && postTrotterCouplesThisStep)
                    {
                        const bool useGpuNvtKineticForPostTrotter =
                                useExactRespaGpuNvtKineticReduction && !bCalcEner && !bCalcVir
                                && postTrotterReplay == PcffExactRespaTrotterReplay::Three;
                        const bool deferGpuNvtPostTrotter =
                                useGpuNvtKineticForPostTrotter
                                && exactRespaGpuDeferNvtPostTrotterProbeEnabled() && !bLastStep;
                        if (useGpuNvtKineticForPostTrotter)
                        {
                            GMX_RELEASE_ASSERT(!exactRespaGpuNvtPostTrotterPending,
                                               "Exact r-RESPA GPU post-Trotter work is already pending");
                            GMX_RELEASE_ASSERT(exactRespaDeviceKickPendingPostTrotterScale == 1.0F,
                                               "Exact r-RESPA post-Trotter scale is still pending");
#if GMX_GPU
                            exactRespaGpuUpdater_->launchExactRespaKineticEnergy();
#else
                            GMX_RELEASE_ASSERT(false,
                                               "Exact r-RESPA GPU kinetic reduction requires a GPU build");
#endif
                            exactRespaGpuNvtPostTrotterPending = true;
                            exactRespaGpuNvtPostTrotterStep    = step;
                            exactRespaGpuNvtKineticReadyForPreTrotter = true;
                            exactRespaDeviceKickHostVelocitiesCurrent = false;

                            if (!deferGpuNvtPostTrotter)
                            {
                                completeExactRespaPendingGpuNvtPostTrotter(bGStat);
                                applyExactRespaPendingPostTrotterScaleToDevice();
                            }
                        }
                        else
                        {
                            if (useGpuForUpdate && exactRespaDeviceKickGpuUpdateProbeEnabled())
                            {
                                stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
                                stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
                                exactRespaDeviceKickHostVelocitiesCurrent = true;
                            }
                            prepareExactRespaVelocityVerletObservablesForStep(
                                    *ir,
                                    step + 1,
                                    cr_->commMyGroup,
                                    *mdAtoms_->mdatoms(),
                                    nrnb_,
                                    &vcm,
                                    enerd_,
                                    gstat,
                                    &nullSignaller,
                                    &observablesReducer,
                                    force_vir,
                                    shake_vir,
                                    total_vir,
                                    pres,
                                    bCalcEner,
                                    bCalcVir || postTrotterCouplesThisStep,
                                    bGStat,
                                    false, // COM removal belongs to the next full-step boundary.
                                    &bSumEkinhOld,
                                    &saved_conserved_quantity,
                                    &last_ekin);
                            m_add(force_vir, shake_vir, total_vir);
                            if (isMainRank)
                            {
                                appendPcffMttkStateTraceRow(step,
                                                            t,
                                                            "before_post_trotter",
                                                            *ir,
                                                            state_,
                                                            pres,
                                                            total_vir,
                                                            enerd_);
                            }
                            const double postTrotterVelocityScale =
                                    replayExactRespaTrotter(postTrotterReplay, false);
                            if (useGpuForUpdate && exactRespaDeviceKickGpuUpdateProbeEnabled())
                            {
                                const bool fusePostTrotterVelocityScale =
                                        useExactRespaFusedNvtTrotterScaling
                                        && (postTrotterReplay == PcffExactRespaTrotterReplay::Two
                                            || postTrotterReplay
                                                       == PcffExactRespaTrotterReplay::Three);
                                if (fusePostTrotterVelocityScale)
                                {
                                    exactRespaDeviceKickPendingPostTrotterScale =
                                            static_cast<float>(postTrotterVelocityScale);
                                }
                                else
                                {
                                    stateGpu->copyVelocitiesToGpu(state_->v, AtomLocality::Local);
                                    exactRespaDeviceKickPendingPostTrotterScale = 1.0F;
                                }
                                exactRespaDeviceKickHostVelocitiesCurrent = true;
                                exactRespaGpuNvtKineticReadyForPreTrotter = false;
                            }
                            if (isMainRank)
                            {
                                appendPcffMttkStateTraceRow(step,
                                                            t,
                                                            "after_post_trotter",
                                                            *ir,
                                                            state_,
                                                            pres,
                                                            total_vir,
                                                            enerd_);
                            }
                        }
                    }
                }
                else
                {
                    integrateVVSecondStep(step,
                                          ir,
                                          fr_,
                                          cr_->commMyGroup,
                                          cr_->dd,
                                          state_,
                                          mdAtoms_->mdatoms(),
                                          &fcdata,
                                          &MassQ,
                                          &vcm,
                                          pullWork_,
                                          enerd_,
                                          &observablesReducer,
                                          ekind_,
                                          gstat,
                                          &dvdl_constr,
                                          bCalcVir,
                                          total_vir,
                                          shake_vir,
                                          force_vir,
                                          pres,
                                          lastbox,
                                          do_log,
                                          do_ene,
                                          bGStat,
                                          &bSumEkinhOld,
                                          &f,
                                          &cbuf,
                                          &upd,
                                          constr_,
                                          &nullSignaller,
                                          trotter_seq,
                                          nrnb_,
                                          wallCycleCounters_);
                }
            }
            else
            {
                if (useGpuForUpdate)
                {
                    // On search steps, update handles to device vectors
                    // TODO: this condition has redundant / unnecessary clauses
                    if (bNS && (bFirstStep || haveDDAtomOrdering(*cr_) || bExchanged))
                    {
                        integrator->set(stateGpu->getCoordinates(),
                                        stateGpu->getVelocities(),
                                        stateGpu->getForces(),
                                        top_->idef,
                                        *md);

                        // Copy data to the GPU after buffers might have been reinitialized
                        /* The velocity copy is redundant if we had Center-of-Mass motion removed on
                         * the previous step. We don't check that now. */
                        stateGpu->copyVelocitiesToGpu(state_->v, AtomLocality::Local);
                    }

                    // Copy x to the GPU unless we have already transferred in do_force().
                    // We transfer in do_force() if a GPU force task requires x (PME or x buffer ops).
                    if (!(runScheduleWork_->stepWork.haveGpuPmeOnThisRank
                          || runScheduleWork_->stepWork.useGpuXBufferOps))
                    {
                        stateGpu->copyCoordinatesToGpu(state_->x, AtomLocality::Local);
                        // Coordinates are later used by the integrator running in the same stream.
                        stateGpu->consumeCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
                    }

                    if ((simulationWork.useGpuPme && simulationWork.useCpuPmePpCommunication)
                        || (!runScheduleWork_->stepWork.useGpuFBufferOps))
                    {
                        // The PME forces were received to the host, and reduced on the CPU with the
                        // rest of the forces computed on the GPU, so the final forces have to be
                        // copied back to the GPU. Or the buffer ops were not offloaded this step,
                        // so the forces are on the host and have to be copied
                        stateGpu->copyForcesToGpu(f.view().force(), AtomLocality::Local);
                    }
                    const bool doTemperatureScalingGpu =
                            (ir->etc != TemperatureCoupling::No
                             && do_per_step(step + ir->nsttcouple - 1, ir->nsttcouple));

                    // This applies Leap-Frog, LINCS and SETTLE in succession
                    integrator->integrate(
                            stateGpu->getLocalForcesReadyOnDeviceEvent(
                                    runScheduleWork_->stepWork, runScheduleWork_->simulationWork),
                            ir->delta_t,
                            true,
                            bCalcVir,
                            shake_vir,
                            doTemperatureScalingGpu,
                            ekind_->tcstat,
                            doParrinelloRahman,
                            ir->pressureCouplingOptions.nstpcouple * ir->delta_t,
                            parrinelloRahmanM);
                }
                else
                {
                    const bool useExactNestedRespa =
                            useNestedExactLammpsRespa(*ir) && nestedExactLammpsRespaPrototypeEnabled();
                    const bool useSupportedExactNestedRespa =
                            useExactNestedRespa
                            && canUseNestedExactLammpsRespa(*ir,
                                                            simulationWork,
                                                            runScheduleWork_->domainWork,
                                                            shellfc,
                                                            constr_,
                                                            virtualSites_,
                                                            useReplicaExchange);
                    if (useSupportedExactNestedRespa)
                    {
                        dispatchExactRespaNestedPrototypeStep(*ir,
                                                              step,
                                                              t,
                                                              *md,
                                                              simulationWork,
                                                              runScheduleWork_->domainWork,
                                                              f,
                                                              exactRespaForceStore,
                                                              force_vir,
                                                              mu_tot,
                                                              *enerd_,
                                                              awh.get(),
                                                              ed ? ed->getLegacyED() : nullptr,
                                                              ddBalanceRegionHandler);
                    }
                    else
                    {
                        /* With multiple time stepping we need to do an additional normal
                         * update step to obtain the virial and dH/dl, as the actual MTS integration
                         * using an acceleration where the slow forces are multiplied by mtsFactor.
                         * Using that acceleration would result in a virial with the slow
                         * force contribution would be a factor mtsFactor too large.
                         */
                        const bool separateVirialConstraining =
                                ((simulationWork.useExactRespa || simulationWork.useLegacyMtsSubsteps())
                                 && (bCalcVir || computeDHDL) && constr_ != nullptr);
                        if (separateVirialConstraining)
                        {
                            upd.update_for_constraint_virial(*ir,
                                                             md->homenr,
                                                             md->havePartiallyFrozenAtoms,
                                                             md->invmass,
                                                             md->invMassPerDim,
                                                             *state_,
                                                             f.view().forceWithPadding(),
                                                             *ekind_);

                            // Call apply() directly so we can avoid constraining the velocities
                            constr_->apply(false,
                                           step,
                                           1,
                                           1.0,
                                           state_->x.arrayRefWithPadding(),
                                           upd.xp()->arrayRefWithPadding(),
                                           {},
                                           state_->box,
                                           state_->lambda[FreeEnergyPerturbationCouplingType::Bonded],
                                           &dvdl_constr,
                                           {},
                                           bCalcVir,
                                           shake_vir,
                                           ConstraintVariable::Positions);
                        }

                        const bool usingMtsCombinedForce =
                                (simulationWork.useExactRespa
                                 && runScheduleWork_->exactRespaStepWork.haveSlowForceLevels)
                                || (simulationWork.useLegacyMtsSubsteps()
                                    && runScheduleWork_->stepWork.computeSlowForces);
                        ArrayRefWithPadding<const RVec> forceCombined =
                                usingMtsCombinedForce ? f.view().forceMtsCombinedWithPadding()
                                                      : f.view().forceWithPadding();
                        appendTp18eTraceRow(step,
                                            "before_update_coords",
                                            forceCombined.unpaddedConstArrayRef(),
                                            usingMtsCombinedForce,
                                            makeConstArrayRef(state_->x),
                                            makeConstArrayRef(state_->v),
                                            gmx::ArrayRef<const gmx::RVec>{},
                                            false,
                                            force_vir,
                                            shake_vir,
                                            total_vir,
                                            pres,
                                            enerd_);
                        appendUpdateStateTrace(activeM2pTraceDirPath(),
                                               step,
                                               "before_update_coords",
                                               state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                               state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                               "src/gromacs/mdrun/md.cpp:before_update_coords");
                        upd.update_coords(*ir,
                                          step,
                                          md->homenr,
                                          md->havePartiallyFrozenAtoms,
                                          md->ptype,
                                          md->invmass,
                                          md->invMassPerDim,
                                          state_,
                                          forceCombined,
                                          &fcdata,
                                          ekind_,
                                          parrinelloRahmanM,
                                          etrtPOSITION,
                                          cr_->dd,
                                          constr_ != nullptr);
                        appendTp18eTraceRow(step,
                                            "after_update_coords",
                                            forceCombined.unpaddedConstArrayRef(),
                                            usingMtsCombinedForce,
                                            makeConstArrayRef(state_->x),
                                            makeConstArrayRef(state_->v),
                                            upd.xp()->arrayRefWithPadding().unpaddedConstArrayRef(),
                                            true,
                                            force_vir,
                                            shake_vir,
                                            total_vir,
                                            pres,
                                            enerd_);
                        appendUpdateStateTrace(activeM2pTraceDirPath(),
                                               step,
                                               "after_update_coords",
                                               state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                               state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                               "src/gromacs/mdrun/md.cpp:after_update_coords");

                        wallcycle_stop(wallCycleCounters_, WallCycleCounter::Update);

                        constrain_coordinates(constr_,
                                              do_log || do_ene,
                                              step,
                                              state_,
                                              upd.xp()->arrayRefWithPadding(),
                                              separateVirialConstraining ? nullptr : &dvdl_constr,
                                              bCalcVir && !separateVirialConstraining,
                                              shake_vir);

                        upd.update_sd_second_half(*ir,
                                                  step,
                                                  &dvdl_constr,
                                                  md->homenr,
                                                  md->ptype,
                                                  md->invmass,
                                                  state_,
                                                  cr_->dd,
                                                  nrnb_,
                                                  wallCycleCounters_,
                                                  constr_,
                                                  do_log,
                                                  do_ene);
                        upd.finish_update(*ir,
                                          md->havePartiallyFrozenAtoms,
                                          md->homenr,
                                          state_,
                                          wallCycleCounters_,
                                          constr_ != nullptr);
                    }
                }

                if (ir->bPull && ir->pull->bSetPbcRefToPrevStepCOM)
                {
                    updatePrevStepPullCom(pullWork_, state_->pull_com_prev_step);
                }

                enerd_->term[InteractionFunction::dHdLambdaConstraint] += dvdl_constr;
            }
        }

        if (simulationWork.useMdGpuGraph)
        {
            GMX_ASSERT((mdGraph != nullptr), "MD GPU graph does not exist.");
            if (mdGraph->graphIsCapturingThisStep())
            {
                mdGraph->endRecord();
                // Force graph reinstantiation (instead of graph exec
                // update): with PME tuning, since the GPU kernels
                // chosen by the FFT library can vary with grid size;
                // or with an odd nstlist, since the odd/even step
                // pruning pattern will change
                bool forceGraphReinstantiation =
                        (pmeLoadBal && pmeLoadBal->isActive()) || ((ir->nstlist % 2) == 1);
                mdGraph->createExecutableGraph(forceGraphReinstantiation);
            }
            if (mdGraph->useGraphThisStep())
            {
                mdGraph->launchGraphMdStep(integrator->xUpdatedOnDeviceEvent());
            }
            if (bNS)
            {
                // TODO: merge disableForDomainIfAnyPpRankHasCpuForces() back into reset() when
                // domainWork initialization is moved out of do_force().
                fr_->mdGraph[MdGraphEvenOrOddStep::EvenStep]->disableForDomainIfAnyPpRankHasCpuForces(
                        runScheduleWork_->domainWork.haveCpuLocalForceWork);
                fr_->mdGraph[MdGraphEvenOrOddStep::OddStep]->disableForDomainIfAnyPpRankHasCpuForces(
                        runScheduleWork_->domainWork.haveCpuLocalForceWork);
            }
            usedMdGpuGraphLastStep = mdGraph->useGraphThisStep();
        }

        /* ############## IF NOT VV, Calculate globals HERE  ############ */
        /* With Leap-Frog we can skip compute_globals at
         * non-communication steps, but we need to calculate
         * the kinetic energy one step before communication.
         */
        {
            // Organize to do inter-simulation signalling on steps if
            // and when algorithms require it.
            const bool doInterSimSignal = (simulationsShareState && do_per_step(step, nstSignalComm));

            if (useGpuForUpdate)
            {
                const bool coordinatesRequiredForStopCM =
                        bStopCM && (bGStat || needHalfStepKineticEnergy || doInterSimSignal)
                        && !EI_VV(ir->eI);

                // Copy coordinates when needed to stop the CM motion or for replica exchange
                if (coordinatesRequiredForStopCM || bDoReplEx)
                {
                    stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
                    stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
                }

                // Copy velocities back to the host if:
                // - Globals are computed this step (includes the energy output steps).
                // - Temperature is needed for the next step.
                // - This is a replica exchange step (even though we will only need
                //     the velocities if an exchange succeeds)
                const bool exactRespaSparseNvtObservables =
                        useSupportedExactVelocityVerletRespa
                        && exactRespaDeviceKickGpuUpdateProbeEnabled()
                        && exactRespaSparseNvtObservablesGpuUpdateProbeEnabled()
                        && inputrecNvtTrotter(ir);
                const bool globalsRequireHostVelocities =
                        needHalfStepKineticEnergy || bDoReplEx
                        || (bGStat && !exactRespaSparseNvtObservables);
                if (globalsRequireHostVelocities
                    && (!exactRespaDeviceKickGpuUpdateProbeEnabled()
                        || !exactRespaDeviceKickHostVelocitiesCurrent))
                {
                    completeExactRespaPendingGpuNvtPostTrotter(bGStat);
                    applyExactRespaPendingPostTrotterScaleToDevice();
                    stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
                    stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
                    exactRespaDeviceKickHostVelocitiesCurrent = true;
                }
            }

            if (bGStat || needHalfStepKineticEnergy || doInterSimSignal)
            {
                // Since we're already communicating at this step, we
                // can propagate intra-simulation signals. Note that
                // check_nstglobalcomm has the responsibility for
                // choosing the value of nstglobalcomm that is one way
                // bGStat becomes true, so we can't get into a
                // situation where e.g. checkpointing can't be
                // signalled.
                bool doIntraSimSignal = true;
                SimulationSignaller signaller(&signals, cr_, ms_, doInterSimSignal, doIntraSimSignal);

                // ScopedTp18jPostUpdateComputeGlobalsTrace tp18jPostUpdateComputeGlobalsTraceScope;
                compute_globals(gstat,
                                cr_->commMyGroup,
                                ir,
                                fr_,
                                ekind_,
                                makeConstArrayRef(state_->x),
                                makeConstArrayRef(state_->v),
                                state_->box,
                                md,
                                nrnb_,
                                &vcm,
                                wallCycleCounters_,
                                enerd_,
                                force_vir,
                                shake_vir,
                                total_vir,
                                pres,
                                &signaller,
                                lastbox,
                                &bSumEkinhOld,
                                (bGStat ? CGLO_GSTAT : 0) | (!EI_VV(ir->eI) && bCalcEner ? CGLO_ENERGY : 0)
                                        | (!EI_VV(ir->eI) && bStopCM ? CGLO_STOPCM : 0)
                                        | (!EI_VV(ir->eI) ? CGLO_TEMPERATURE : 0)
                                        | (!EI_VV(ir->eI) ? CGLO_PRESSURE : 0) | CGLO_CONSTRAINT,
                                step,
                                &observablesReducer);
                appendTp18eTraceRow(step,
                                    "after_compute_globals",
                                    f.view().forceWithPadding().unpaddedConstArrayRef(),
                                    false,
                                    makeConstArrayRef(state_->x),
                                    makeConstArrayRef(state_->v),
                                    gmx::ArrayRef<const gmx::RVec>{},
                                    false,
                                    force_vir,
                                    shake_vir,
                                    total_vir,
                                    pres,
                                    enerd_);
                if (!EI_VV(ir->eI) && bStopCM)
                {
                    process_and_stopcm_grp(
                            fpLog_, &vcm, *md, makeArrayRef(state_->x), makeArrayRef(state_->v));
                    inc_nrnb(nrnb_, eNR_STOPCM, md->homenr);

                    // TODO: The special case of removing CM motion should be dealt more gracefully
                    if (useGpuForUpdate)
                    {
                        // Issue #3988, #4106.
                        stateGpu->resetCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
                        stateGpu->copyCoordinatesToGpu(state_->x, AtomLocality::Local);
                        // Here we block until the H2D copy completes because event sync with the
                        // force kernels that use the coordinates on the next steps is not implemented
                        // (not because of a race on state->x being modified on the CPU while H2D is in progress).
                        stateGpu->waitCoordinatesCopiedToDevice(AtomLocality::Local);
                        // If the COM removal changed the velocities on the CPU, this has to be accounted for.
                        if (vcm.mode != ComRemovalAlgorithm::No)
                        {
                            stateGpu->copyVelocitiesToGpu(state_->v, AtomLocality::Local);
                        }
                    }
                }
            }
        }

        /* #############  END CALC EKIN AND PRESSURE ################# */

        /* Note: this is OK, but there are some numerical precision issues with using the convergence of
           the virial that should probably be addressed eventually. state->veta has better properties,
           but what we actually need entering the new cycle is the new shake_vir value. Ideally, we could
           generate the new shake_vir, but test the veta value for convergence.  This will take some thought. */

        if (ir->efep != FreeEnergyPerturbationType::No && !EI_VV(ir->eI))
        {
            /* Sum up the foreign energy and dK/dl terms for md and sd.
               Currently done every step so that dH/dl is correct in the .edr */
            accumulateKineticLambdaComponents(enerd_, state_->lambda, *ir->fepvals);
        }

        const real currentSystemRefT =
                (haveEnsembleTemperature(*ir) ? ekind_->currentEnsembleTemperature() : 0.0_real);
        const bool scaleCoordinates = !useGpuForUpdate || bDoReplEx;
        const bool exactRespaMttkOuterPcouple =
                useSupportedExactVelocityVerletRespa
                && ir->pressureCouplingOptions.epc == PressureCoupling::Mttk
                && pcffExactRespaMttkOuterPcoupleEnabled();
        const bool exactRespaMttkInlineBoxRemap =
                useSupportedExactVelocityVerletRespa
                && ir->pressureCouplingOptions.epc == PressureCoupling::Mttk
                && ir->pressureCouplingOptions.epct == PressureCouplingType::Isotropic
                && pcffExactRespaMttkInlineBoxRemapEnabled();
        const bool doExactRespaMttkOuterPcouple =
                exactRespaMttkOuterPcouple
                && do_per_step(step + 1, ir->pressureCouplingOptions.nstpcouple);
        if (!exactRespaMttkInlineBoxRemap
            && (!exactRespaMttkOuterPcouple || doExactRespaMttkOuterPcouple))
        {
            const real pcoupleDeltaT = doExactRespaMttkOuterPcouple
                                               ? ir->delta_t * ir->pressureCouplingOptions.nstpcouple
                                               : ir->delta_t;
            update_pcouple_after_coordinates(fpLog_,
                                             step,
                                             ir->pressureCouplingOptions,
                                             ir->ld_seed,
                                             currentSystemRefT,
                                             ir->opts.nFreeze,
                                             ir->deform,
                                             pcoupleDeltaT,
                                             md->homenr,
                                             md->cFREEZE,
                                             pres,
                                             force_vir,
                                             shake_vir,
                                             &pressureCouplingMu,
                                             state_,
                                             nrnb_,
                                             upd.deform(),
                                             scaleCoordinates);
        }
        if (isMainRank)
        {
            appendTp18gTraceRow(step,
                                t,
                                "after_update_pcouple",
                                bCalcEner,
                                bCalcEnerStep,
                                inputRec_,
                                state_->hasEntry(StateEntry::PressurePrevious),
                                false,
                                false,
                                false,
                                false,
                                total_vir,
                                pres,
                                enerd_);
            appendPcffMttkStateTraceRow(
                    step, t, "after_update_pcouple", *ir, state_, pres, total_vir, enerd_);
        }

        const bool doBerendsenPressureCoupling =
                (inputRec_->pressureCouplingOptions.epc == PressureCoupling::Berendsen
                 && do_per_step(step, inputRec_->pressureCouplingOptions.nstpcouple));
        const bool doCRescalePressureCoupling =
                (inputRec_->pressureCouplingOptions.epc == PressureCoupling::CRescale
                 && do_per_step(step, inputRec_->pressureCouplingOptions.nstpcouple));
        if (useGpuForUpdate
            && (doBerendsenPressureCoupling || doCRescalePressureCoupling || doParrinelloRahman))
        {
            integrator->scaleCoordinates(pressureCouplingMu);
            if (doCRescalePressureCoupling)
            {
                integrator->scaleVelocities(invertBoxMatrix(pressureCouplingMu));
            }
            integrator->setPbc(PbcType::Xyz, state_->box);
            if (useSupportedExactVelocityVerletRespa)
            {
                // Exact r-RESPA re-enters host-side kicks and force refresh on every base step,
                // so any GPU-only box/velocity scaling has to be materialized on the host
                // before the next exact-respa substep.
                stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
                if (doCRescalePressureCoupling)
                {
                    stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
                    stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
                }
            }
        }

        /* ################# END UPDATE STEP 2 ################# */
        /* #### We now have r(t+dt) and v(t+dt/2)  ############# */

        /* The coordinates (x) were unshifted in update */
        if (!bGStat)
        {
            /* We will not sum ekinh_old,
             * so signal that we still have to do it.
             */
            bSumEkinhOld = TRUE;
        }

        if (bCalcEner)
        {
            /* #########  BEGIN PREPARING EDR OUTPUT  ###########  */

            /* use the directly determined last velocity, not actually the averaged half steps */
            if (bTrotter && ir->eI == IntegrationAlgorithm::VV)
            {
                enerd_->term[InteractionFunction::KineticEnergy] = last_ekin;
            }
            enerd_->term[InteractionFunction::TotalEnergy] =
                    enerd_->term[InteractionFunction::PotentialEnergy]
                    + enerd_->term[InteractionFunction::KineticEnergy];

            if (integratorHasConservedEnergyQuantity(ir))
            {
                if (EI_VV(ir->eI))
                {
                    enerd_->term[InteractionFunction::ConservedEnergy] =
                            enerd_->term[InteractionFunction::TotalEnergy] + saved_conserved_quantity;
                }
                else
                {
                    enerd_->term[InteractionFunction::ConservedEnergy] =
                            enerd_->term[InteractionFunction::TotalEnergy]
                            + NPT_energy(ir->pressureCouplingOptions,
                                         ir->etc,
                                         gmx::constArrayRefFromArray(ir->opts.nrdf, ir->opts.ngtc),
                                         *ekind_,
                                         inputrecNvtTrotter(ir) || inputrecNptTrotter(ir),
                                         state_,
                                         &MassQ);
                }
            }
            /* #########  END PREPARING EDR OUTPUT  ###########  */
        }

        /* Output stuff */
        if (isMainRank)
        {
            if (fpLog_ && do_log && bDoExpanded)
            {
                /* only needed if doing expanded ensemble */
                PrintFreeEnergyInfoToFile(fpLog_,
                                          ir->fepvals.get(),
                                          ir->expandedvals.get(),
                                          ir->bSimTemp ? ir->simtempvals.get() : nullptr,
                                          stateGlobal_->dfhist.get(),
                                          state_->fep_state,
                                          ir->nstlog,
                                          step);
            }
            if (bCalcEner)
            {
                const bool outputDHDL = (computeDHDL && do_per_step(step, ir->fepvals->nstdhdl));

                energyOutput.addDataAtEnergyStep(outputDHDL,
                                                 bCalcEnerStep,
                                                 t,
                                                 md->tmass,
                                                 enerd_,
                                                 ir->fepvals.get(),
                                                 lastbox,
                                                 PTCouplingArrays{ state_->boxv,
                                                                   state_->nosehoover_xi,
                                                                   state_->nosehoover_vxi,
                                                                   state_->nhpres_xi,
                                                                   state_->nhpres_vxi },
                                                 state_->fep_state,
                                                 total_vir,
                                                 pres,
                                                 ekind_,
                                                 mu_tot,
                                                 constr_);
                appendTp18gTraceRow(step,
                                    t,
                                    "after_energy_add",
                                    bCalcEner,
                                    bCalcEnerStep,
                                    inputRec_,
                                    state_->hasEntry(StateEntry::PressurePrevious),
                                    false,
                                    true,
                                    false,
                                    false,
                                    total_vir,
                                    pres,
                                    enerd_);
            }
            else
            {
                energyOutput.recordNonEnergyStep();
                appendTp18gTraceRow(step,
                                    t,
                                    "after_energy_add",
                                    bCalcEner,
                                    bCalcEnerStep,
                                    inputRec_,
                                    state_->hasEntry(StateEntry::PressurePrevious),
                                    false,
                                    false,
                                    true,
                                    false,
                                    total_vir,
                                    pres,
                                    enerd_);
            }

            gmx_bool do_dr = do_per_step(step, ir->nstdisreout);
            gmx_bool do_or = do_per_step(step, ir->nstorireout);

            if (doSimulatedAnnealing)
            {
                gmx::EnergyOutput::printAnnealingTemperatures(
                        do_log ? fpLog_ : nullptr, *groups, ir->opts, *ekind_);
            }
            if (do_log || do_ene || do_dr || do_or)
            {
                energyOutput.printStepToEnergyFile(mdoutf_get_fp_ene(outf),
                                                   do_ene,
                                                   do_dr,
                                                   do_or,
                                                   do_log ? fpLog_ : nullptr,
                                                   step,
                                                   t,
                                                   fr_->fcdata.get(),
                                                   awh.get());
            }
            appendTp18gTraceRow(step,
                                t,
                                "after_energy_print",
                                bCalcEner,
                                bCalcEnerStep,
                                inputRec_,
                                state_->hasEntry(StateEntry::PressurePrevious),
                                false,
                                false,
                                false,
                                (do_log || do_ene || do_dr || do_or),
                                total_vir,
                                pres,
                                enerd_);
            if (do_log && ((ir->bDoAwh && awh->hasFepLambdaDimension()) || ir->fepvals->delta_lambda != 0))
            {
                const bool isInitialOutput = false;
                printLambdaStateToLog(fpLog_, state_->lambda, isInitialOutput);
            }

            if (ir->bPull)
            {
                pull_print_output(pullWork_, step, t);
            }

            if (do_per_step(step, ir->nstlog))
            {
                if (std::fflush(fpLog_) != 0)
                {
                    gmx_fatal(FARGS, "Cannot flush logfile - maybe you are out of disk space?");
                }
            }
        }
        if (bDoExpanded)
        {
            /* Have to do this part _after_ outputting the logfile and the edr file */
            /* Gets written into the state at the beginning of next loop*/
            state_->fep_state = lamnew;
        }
        else if (ir->bDoAwh && awh->needForeignEnergyDifferences(step))
        {
            state_->fep_state = awh->fepLambdaState();
        }
        /* Print the remaining wall clock time for the run */
        if (isMainSimMainRank(ms_, isMainRank) && (do_verbose || gmx_got_usr_signal())
            && !(pmeLoadBal && pmeLoadBal->isPrintingLoad()))
        {
            if (shellfc)
            {
                fprintf(stderr, "\n");
            }
            print_time(stderr, wallTimeAccounting_, step, ir, cr_->commMySim);
        }

        /* Ion/water position swapping.
         * Not done in last step since trajectory writing happens before this call
         * in the MD loop and exchanges would be lost anyway. */
        bNeedRepartition = FALSE;
        if ((ir->eSwapCoords != SwapType::No) && (step > 0) && !bLastStep
            && do_per_step(step, ir->swap->nstswap))
        {
            bNeedRepartition = do_swapcoords(cr_->commMyGroup,
                                             step,
                                             t,
                                             ir,
                                             swap_,
                                             wallCycleCounters_,
                                             state_->x,
                                             state_->box,
                                             isMainRank && mdrunOptions_.verbose,
                                             bRerunMD);

            if (bNeedRepartition && haveDDAtomOrdering(*cr_))
            {
                dd_collect_state(cr_->dd, state_, stateGlobal_);
            }
        }

        /* Replica exchange */
        bExchanged = FALSE;
        if (bDoReplEx)
        {
            bExchanged =
                    replica_exchange(fpLog_, cr_, ms_, repl_ex, stateGlobal_, enerd_, state_, step, t);
        }

        if ((bExchanged || bNeedRepartition) && haveDDAtomOrdering(*cr_))
        {
            dd_partition_system(fpLog_,
                                mdLog_,
                                step,
                                cr_->dd,
                                TRUE,
                                stateGlobal_,
                                topGlobal_,
                                *ir,
                                mdModulesNotifiers_,
                                imdSession_,
                                pullWork_,
                                state_,
                                &f,
                                mdAtoms_,
                                top_,
                                fr_,
                                virtualSites_,
                                constr_,
                                nrnb_,
                                wallCycleCounters_,
                                FALSE);
            upd.updateAfterPartition(state_->numAtoms(), md->cFREEZE, md->cTC, md->cACC);
            fr_->longRangeNonbondeds->updateAfterPartition(*md);
            if (runScheduleWork_->stepWork.haveGpuPmeOnThisRank)
            {
                pme_gpu_prepare_computation(
                        fr_->pmedata, state_->box, simulationWork.haveDynamicBox, runScheduleWork_->stepWork);
            }
        }
        appendPreDoForceStateTrace(activeM2pTraceDirPath(),
                                   step,
                                   "before_step_increment",
                                   state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                   state_->v.arrayRefWithPadding().unpaddedConstArrayRef(),
                                   "src/gromacs/mdrun/md.cpp:before_step_increment");
        if (step == 4)
        {
            appendMdLoopBoundarySnapshotPair(activeM2pTraceDirPath(),
                                            useSupportedExactVelocityVerletRespa ? "PATCH" : "PLAIN",
                                            "STEP4_END_STATE_X",
                                            step,
                                            state_->x.arrayRefWithPadding().unpaddedConstArrayRef(),
                                            "src/gromacs/mdrun/md.cpp:STEP4_END_STATE_X");
        }

        bFirstStep = FALSE;
        bInitStep  = FALSE;

        /* #######  SET VARIABLES FOR NEXT ITERATION IF THEY STILL NEED IT ###### */
        /* With all integrators, except VV, we need to retain the pressure
         * at the current step for coupling at the next step.
         */
        if (state_->hasEntry(StateEntry::PressurePrevious)
            && (bGStatEveryStep
                || (ir->pressureCouplingOptions.nstpcouple > 0
                    && step % ir->pressureCouplingOptions.nstpcouple == 0)))
        {
            /* Store the pressure in t_state for pressure coupling
             * at the next MD step.
             */
            copy_mat(pres, state_->pres_prev);
        }
        if (isMainRank)
        {
            const bool pressurePreviousCopyExecuted =
                    (state_->hasEntry(StateEntry::PressurePrevious)
                     && (bGStatEveryStep
                         || (ir->pressureCouplingOptions.nstpcouple > 0
                             && step % ir->pressureCouplingOptions.nstpcouple == 0)));
            appendTp18gTraceRow(step,
                                t,
                                "after_pressure_prev_handoff",
                                bCalcEner,
                                bCalcEnerStep,
                                inputRec_,
                                state_->hasEntry(StateEntry::PressurePrevious),
                                pressurePreviousCopyExecuted,
                                false,
                                false,
                                false,
                                total_vir,
                                pres,
                                enerd_);
        }

        /* #######  END SET VARIABLES FOR NEXT ITERATION ###### */

        if ((membed_ != nullptr) && (!bLastStep))
        {
            rescale_membed(step_rel, membed_, as_rvec_array(stateGlobal_->x.data()));
        }

        const double cycles = wallcycle_stop(wallCycleCounters_, WallCycleCounter::Step);
        if (haveDDAtomOrdering(*cr_) && wallCycleCounters_)
        {
            dd_cycles_add(cr_->dd, cycles, ddCyclStep);
        }

        /* increase the MD step number */
        step++;
        step_rel++;
        observablesReducer.markAsReadyToReduce();

#if GMX_FAHCORE
        if (MAIN(cr))
        {
            fcReportProgress(ir->nsteps + ir->init_step, step);
        }
#endif

        resetHandler->resetCounters(step,
                                    step_rel,
                                    mdLog_,
                                    fpLog_,
                                    cr_,
                                    fr_->nbv.get(),
                                    nrnb_,
                                    fr_->pmedata,
                                    pmeLoadBal.get(),
                                    wallCycleCounters_,
                                    wallTimeAccounting_);

        /* If bIMD is TRUE, the main updates the IMD energy record and sends positions to VMD client */
        imdSession_->updateEnergyRecordAndSendPositionsAndEnergies(bInteractiveMDstep, step, bCalcEner);

        // any run that uses GPUs must be at least offloading nonbondeds
        const bool usingGpu = simulationWork.useGpuNonbonded;
        if (usingGpu)
        {
            // ensure that GPU errors do not propagate between MD steps
            checkPendingDeviceErrorBetweenSteps();
        }
    }
    /* End of main MD loop */
    if (pcffContinuousRefPressureRamp.active)
    {
        mutableInputRecord->pressureCouplingOptions = basePressureCouplingOptions;
    }

    /* Closing TNG files can include compressing data. Therefore it is good to do that
     * before stopping the time measurements. */
    mdoutf_tng_close(outf);

    /* Stop measuring walltime */
    walltime_accounting_end_time(wallTimeAccounting_);

    if (simulationWork.haveSeparatePmeRank)
    {
        /* Tell the PME only node to finish */
        gmx_pme_send_finish(cr_->dd);
    }

    // This is to free PP ranks gpuhaloexchange symmetric buffer `d_recvBuf_`
    // as calling its destruction happens very late causing hang as this is a collective
    // call, the PME side free of the same buffer happens quite early.
    if (cr_->commMySim.isParallel() && simulationWork.useNvshmem)
    {
        destroyGpuHaloExchangeNvshmemBuf(*cr_);
    }

    if (isMainRank)
    {
        if (ir->nstcalcenergy > 0)
        {
            energyOutput.printEnergyConservation(fpLog_, ir->simulation_part, EI_MD(ir->eI));

            gmx::EnergyOutput::printAnnealingTemperatures(fpLog_, *groups, ir->opts, *ekind_);
            energyOutput.printAverages(fpLog_, groups);
        }
    }
    done_mdoutf(outf);

    if (pmeLoadBal)
    {
        pmeLoadBal->printSettings();
        pmeLoadBal.reset(nullptr);
    }

    done_shellfc(fpLog_, shellfc, step_rel);

    if (useReplicaExchange && isMainRank)
    {
        print_replica_exchange_statistics(fpLog_, repl_ex);
    }

    walltime_accounting_set_nsteps_done(wallTimeAccounting_, step_rel);

    global_stat_destroy(gstat);
}
