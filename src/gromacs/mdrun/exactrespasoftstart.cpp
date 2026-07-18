/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */

#include "gmxpre.h"

#include "exactrespasoftstart.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "gromacs/math/units.h"
#include "gromacs/mdtypes/exactrespaschedule.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/random/seed.h"
#include "gromacs/random/threefry.h"
#include "gromacs/random/uniformrealdistribution.h"
#include "gromacs/topology/topology_enums.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/mpicomm.h"

namespace gmx
{
namespace
{

bool environmentFlag(const char* name, const bool defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return defaultValue;
    }
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0
           && std::strcmp(value, "no") != 0;
}

real positiveEnvironmentReal(const char* name, const char* alias, const real defaultValue)
{
    const char* value = std::getenv(name);
    if ((value == nullptr || *value == '\0') && alias != nullptr)
    {
        value = std::getenv(alias);
    }
    if (value == nullptr || *value == '\0')
    {
        return defaultValue;
    }

    char*        end    = nullptr;
    const double parsed = std::strtod(value, &end);
    GMX_RELEASE_ASSERT(end != value && end != nullptr && *end == '\0' && std::isfinite(parsed)
                               && parsed > 0,
                       "Exact r-RESPA soft-start numeric environment values must be finite and positive");
    return static_cast<real>(parsed);
}

uint64_t positiveEnvironmentSeed(const char* name, const uint64_t defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return defaultValue;
    }

    char*                    end    = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    GMX_RELEASE_ASSERT(end != value && end != nullptr && *end == '\0' && parsed > 0,
                       "Exact r-RESPA soft-start Langevin seed must be a positive integer");
    return static_cast<uint64_t>(parsed);
}

bool isIntegratedParticle(const t_mdatoms& mdatoms, const int atom)
{
    return atom >= 0 && atom < mdatoms.homenr && atom < static_cast<int>(mdatoms.massT.size())
           && atom < static_cast<int>(mdatoms.ptype.size()) && mdatoms.massT[atom] > 0
           && mdatoms.ptype[atom] != ParticleType::Shell;
}

int64_t globalAtomIndex(const int atom, const int* globalAtomIndices, const int globalAtomIndicesCount)
{
    if (globalAtomIndices != nullptr && atom < globalAtomIndicesCount)
    {
        return globalAtomIndices[atom];
    }
    return atom;
}

} // namespace

ExactRespaSoftStartConfig exactRespaSoftStartConfigFromEnvironment()
{
    ExactRespaSoftStartConfig config;
    config.enabled = environmentFlag("GMX_PCFF_EXACT_RESPA_SOFT_START", false);
    if (!config.enabled)
    {
        return config;
    }

    config.xlimitNm = positiveEnvironmentReal("GMX_PCFF_EXACT_RESPA_NVE_LIMIT_XMAX_NM",
                                               "GMX_PCFF_EXACT_RESPA_SOFT_XLIMIT_NM",
                                               config.xlimitNm);
    config.temperatureK = positiveEnvironmentReal(
            "GMX_PCFF_EXACT_RESPA_LANGEVIN_TEMP_K", nullptr, config.temperatureK);
    config.dampingTimePs = positiveEnvironmentReal(
            "GMX_PCFF_EXACT_RESPA_LANGEVIN_TAU_PS", nullptr, config.dampingTimePs);
    config.seed = positiveEnvironmentSeed("GMX_PCFF_EXACT_RESPA_LANGEVIN_SEED", config.seed);
    config.zeroRandomForce =
            environmentFlag("GMX_PCFF_EXACT_RESPA_LANGEVIN_ZERO_RANDOM", true);
    return config;
}

bool exactRespaSoftStartIsOuterBoundary(const int64_t baseStep,
                                        const int64_t initStep,
                                        const int     outerStepFactor)
{
    GMX_RELEASE_ASSERT(outerStepFactor > 0,
                       "Exact r-RESPA soft-start outer step factor must be positive");
    return baseStep >= initStep && (baseStep - initStep) % outerStepFactor == 0;
}

void initializeExactRespaSoftStartState(const t_inputrec&          inputRecord,
                                        const bool                 useGpuUpdate,
                                        const int64_t              firstBaseStep,
                                        ExactRespaSoftStartState*  state)
{
    GMX_RELEASE_ASSERT(state != nullptr, "Exact r-RESPA soft-start state must exist");
    if (state->configured)
    {
        return;
    }

    state->config     = exactRespaSoftStartConfigFromEnvironment();
    state->configured = true;
    if (!state->config.enabled)
    {
        return;
    }

    GMX_RELEASE_ASSERT(inputRecord.exactRespa.enabled(),
                       "Exact r-RESPA soft-start requires exact-respa = yes");
    GMX_RELEASE_ASSERT(inputRecord.eI == IntegrationAlgorithm::VV,
                       "LAMMPS-style exact r-RESPA soft-start requires integrator = md-vv");
    GMX_RELEASE_ASSERT(inputRecord.etc == TemperatureCoupling::No,
                       "Exact r-RESPA soft-start supplies its own Langevin thermostat; set tcoupl = no");
    GMX_RELEASE_ASSERT(inputRecord.pressureCouplingOptions.epc == PressureCoupling::No,
                       "Exact r-RESPA soft-start is an NVE/limit plus Langevin stage; set pcoupl = no");
    GMX_RELEASE_ASSERT(!useGpuUpdate,
                       "Exact r-RESPA soft-start requires CPU state update; use -update cpu");
    GMX_RELEASE_ASSERT(firstBaseStep == inputRecord.init_step,
                       "Exact r-RESPA soft-start is not checkpoint-resumable; restart the short soft stage from its initial state");

    state->outerLevel      = exactRespaNumLevels(inputRecord) - 1;
    state->outerStepFactor = exactRespaLevelStepFactor(inputRecord, state->outerLevel);
    state->outerDtPs       = inputRecord.delta_t * state->outerStepFactor;
    state->maximumSpeedNmPerPs = state->config.xlimitNm / state->outerDtPs;
    GMX_RELEASE_ASSERT(state->outerLevel > 0 && state->outerStepFactor > 1
                               && state->outerDtPs > 0 && std::isfinite(state->maximumSpeedNmPerPs),
                       "Exact r-RESPA soft-start requires a valid slowest r-RESPA level and time step");
    GMX_RELEASE_ASSERT(
            exactRespaSoftStartIsOuterBoundary(firstBaseStep, inputRecord.init_step, state->outerStepFactor),
            "Exact r-RESPA soft-start must begin on a slowest-level boundary");
}

double exactRespaSoftStartUniform(const uint64_t seed,
                                  const int64_t  outerBoundary,
                                  const int64_t  globalAtomIndexValue,
                                  const int      dimension)
{
    GMX_RELEASE_ASSERT(outerBoundary >= 0, "Soft-start Langevin boundary index must be non-negative");
    GMX_RELEASE_ASSERT(globalAtomIndexValue >= 0,
                       "Soft-start Langevin global atom index must be non-negative");
    GMX_RELEASE_ASSERT(dimension >= 0 && dimension < DIM,
                       "Soft-start Langevin dimension must be Cartesian");

    ThreeFry2x64<0> random(seed, RandomDomain::Thermostat);
    const uint64_t atomDimensionCounter =
            (static_cast<uint64_t>(globalAtomIndexValue) << 2U) | static_cast<uint64_t>(dimension);
    random.restart(static_cast<uint64_t>(outerBoundary), atomDimensionCounter);
    return generateCanonical<double, std::numeric_limits<double>::digits>(random);
}

RVec exactRespaSoftStartLangevinForce(const real        mass,
                                      const RVec&       velocity,
                                      const RVec&       centeredUniform,
                                      const RVec&       meanRandomForce,
                                      const real        temperatureK,
                                      const real        dampingTimePs,
                                      const real        outerDtPs)
{
    GMX_RELEASE_ASSERT(mass > 0 && temperatureK > 0 && dampingTimePs > 0 && outerDtPs > 0,
                       "Soft-start Langevin parameters and particle mass must be positive");

    const double randomAmplitude =
            std::sqrt(24.0 * static_cast<double>(mass) * static_cast<double>(c_boltz)
                      * static_cast<double>(temperatureK)
                      / (static_cast<double>(dampingTimePs) * static_cast<double>(outerDtPs)));
    RVec force;
    for (int dimension = 0; dimension < DIM; ++dimension)
    {
        const double drag = -static_cast<double>(mass) * static_cast<double>(velocity[dimension])
                            / static_cast<double>(dampingTimePs);
        const double randomForce = randomAmplitude * static_cast<double>(centeredUniform[dimension]);
        force[dimension] = static_cast<real>(drag + randomForce - meanRandomForce[dimension]);
    }
    return force;
}

bool applyExactRespaSoftStartHalfKick(const real        dt,
                                      const RVec&       inverseMassPerDim,
                                      const RVec&       physicalForce,
                                      const RVec*       langevinForce,
                                      const real        maximumSpeedNmPerPs,
                                      RVec*             velocity)
{
    GMX_RELEASE_ASSERT(dt > 0 && maximumSpeedNmPerPs > 0,
                       "Soft-start half-kick parameters must be positive");
    GMX_RELEASE_ASSERT(velocity != nullptr, "Soft-start half-kick velocity must exist");
    GMX_RELEASE_ASSERT(inverseMassPerDim[XX] > 0 || inverseMassPerDim[YY] > 0
                               || inverseMassPerDim[ZZ] > 0,
                       "Soft-start half-kick requires at least one integrated dimension");

    double speedSquared = 0;
    for (int dimension = 0; dimension < DIM; ++dimension)
    {
        const real extraForce = (langevinForce == nullptr) ? 0 : (*langevinForce)[dimension];
        if (inverseMassPerDim[dimension] > 0)
        {
            (*velocity)[dimension] += 0.5_real * dt * inverseMassPerDim[dimension]
                                      * (physicalForce[dimension] + extraForce);
        }
        speedSquared += static_cast<double>((*velocity)[dimension]) * (*velocity)[dimension];
    }

    const double maximumSpeedSquared = static_cast<double>(maximumSpeedNmPerPs)
                                       * maximumSpeedNmPerPs;
    if (speedSquared <= maximumSpeedSquared)
    {
        return false;
    }

    const real scale = static_cast<real>(maximumSpeedNmPerPs / std::sqrt(speedSquared));
    for (int dimension = 0; dimension < DIM; ++dimension)
    {
        (*velocity)[dimension] *= scale;
    }
    return true;
}

void refreshExactRespaSoftStartLangevinForce(ExactRespaSoftStartState* state,
                                             const int64_t             outerBoundary,
                                             const t_mdatoms&          mdatoms,
                                             const ArrayRef<const RVec> velocity,
                                             const int*                globalAtomIndices,
                                             const int                 globalAtomIndicesCount,
                                             const MpiComm&            mpiComm)
{
    GMX_RELEASE_ASSERT(state != nullptr && state->configured && state->config.enabled,
                       "Soft-start Langevin refresh requires enabled, configured state");
    GMX_RELEASE_ASSERT(outerBoundary >= 0, "Soft-start Langevin boundary must be non-negative");
    GMX_RELEASE_ASSERT(velocity.ssize() >= mdatoms.homenr,
                       "Soft-start Langevin refresh requires all home-atom velocities");

    state->cachedLangevinForce.assign(mdatoms.homenr, RVec{ 0, 0, 0 });
    std::vector<RVec> centeredUniform(mdatoms.homenr, RVec{ 0, 0, 0 });
    std::array<double, DIM + 1> globalRandomSumAndCount = { 0, 0, 0, 0 };

    for (int atom = 0; atom < mdatoms.homenr; ++atom)
    {
        if (!isIntegratedParticle(mdatoms, atom))
        {
            continue;
        }
        const int64_t atomIndex = globalAtomIndex(atom, globalAtomIndices, globalAtomIndicesCount);
        const double randomAmplitude =
                std::sqrt(24.0 * static_cast<double>(mdatoms.massT[atom])
                          * static_cast<double>(c_boltz)
                          * static_cast<double>(state->config.temperatureK)
                          / (static_cast<double>(state->config.dampingTimePs)
                             * static_cast<double>(state->outerDtPs)));
        for (int dimension = 0; dimension < DIM; ++dimension)
        {
            const real centered = static_cast<real>(
                    exactRespaSoftStartUniform(state->config.seed,
                                               outerBoundary,
                                               atomIndex,
                                               dimension)
                    - 0.5);
            centeredUniform[atom][dimension] = centered;
            globalRandomSumAndCount[dimension] += randomAmplitude * centered;
        }
        globalRandomSumAndCount[DIM] += 1.0;
    }

    RVec meanRandomForce = { 0, 0, 0 };
    if (state->config.zeroRandomForce)
    {
        mpiComm.sumReduce(globalRandomSumAndCount.size(), globalRandomSumAndCount.data());
        GMX_RELEASE_ASSERT(globalRandomSumAndCount[DIM] > 0,
                           "Cannot zero a soft-start Langevin force for zero integrated atoms");
        for (int dimension = 0; dimension < DIM; ++dimension)
        {
            meanRandomForce[dimension] = static_cast<real>(
                    globalRandomSumAndCount[dimension] / globalRandomSumAndCount[DIM]);
        }
    }

    for (int atom = 0; atom < mdatoms.homenr; ++atom)
    {
        if (!isIntegratedParticle(mdatoms, atom))
        {
            continue;
        }
        state->cachedLangevinForce[atom] = exactRespaSoftStartLangevinForce(
                mdatoms.massT[atom],
                velocity[atom],
                centeredUniform[atom],
                meanRandomForce,
                state->config.temperatureK,
                state->config.dampingTimePs,
                state->outerDtPs);
    }
    state->cachedBoundary = outerBoundary;
}

} // namespace gmx
