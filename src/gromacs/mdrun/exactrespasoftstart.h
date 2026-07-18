/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */
#ifndef GMX_MDRUN_EXACTRESPASOFTSTART_H
#define GMX_MDRUN_EXACTRESPASOFTSTART_H

#include <cstdint>

#include <limits>
#include <vector>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vectypes.h"

struct t_inputrec;
struct t_mdatoms;

namespace gmx
{

class MpiComm;

/*! \internal
 * \brief Opt-in parameters for the LAMMPS-style Eq01 soft-start update.
 *
 * The defaults reproduce the Eq01 settings used by the PolyGen validation:
 * fix nve/limit 0.10 and fix langevin 353 353 50 97531 zero yes, expressed
 * in GROMACS units (nm and ps).
 */
struct ExactRespaSoftStartConfig
{
    bool     enabled           = false;
    real     xlimitNm          = 0.01_real;
    real     temperatureK      = 353.0_real;
    real     dampingTimePs     = 0.05_real;
    uint64_t seed              = 97531;
    bool     zeroRandomForce   = true;
};

/*! \internal
 * \brief Per-simulation cache for the slow-level Langevin force.
 *
 * LAMMPS computes fix langevin only at the slowest r-RESPA force refresh and
 * reuses that force for the final and following initial slow half-kicks. This
 * cache deliberately lives on LegacySimulator rather than in process-global
 * state so separate mdrun invocations in one test process cannot contaminate
 * each other.
 */
struct ExactRespaSoftStartState
{
    bool                      configured      = false;
    ExactRespaSoftStartConfig config;
    int                       outerLevel      = -1;
    int                       outerStepFactor = 0;
    real                      outerDtPs       = 0;
    real                      maximumSpeedNmPerPs = 0;
    int64_t                   cachedBoundary = std::numeric_limits<int64_t>::min();
    std::vector<RVec>         cachedLangevinForce;
};

//! Reads the soft-start stage environment once for a simulation.
ExactRespaSoftStartConfig exactRespaSoftStartConfigFromEnvironment();

//! Initializes and validates the opt-in state. The disabled path has no side effects.
void initializeExactRespaSoftStartState(const t_inputrec&          inputRecord,
                                        bool                       useGpuUpdate,
                                        int64_t                    firstBaseStep,
                                        ExactRespaSoftStartState*  state);

//! Returns whether \p baseStep is a slowest-level boundary relative to init-step.
bool exactRespaSoftStartIsOuterBoundary(int64_t baseStep, int64_t initStep, int outerStepFactor);

/*! \brief Deterministic uniform variate used by the standalone GROMACS lane.
 *
 * The counter is keyed by slow boundary, global atom index and Cartesian
 * dimension, so atom/domain ordering does not change the generated variate.
 * This is algorithmic parity with LAMMPS's uniform force distribution, not a
 * replay of LAMMPS's processor-dependent Marsaglia stream.
 */
double exactRespaSoftStartUniform(uint64_t seed,
                                  int64_t  outerBoundary,
                                  int64_t  globalAtomIndex,
                                  int      dimension);

//! Computes drag plus uniform random force, after optional zero-force subtraction.
RVec exactRespaSoftStartLangevinForce(real        mass,
                                      const RVec& velocity,
                                      const RVec& centeredUniform,
                                      const RVec& meanRandomForce,
                                      real        temperatureK,
                                      real        dampingTimePs,
                                      real        outerDtPs);

/*! \brief Applies one velocity-Verlet half-kick and the absolute speed cap.
 *
 * The physical and optional Langevin forces are combined in one kick. Returning
 * true means the speed was limited. Applying this function sequentially for
 * every active r-RESPA level preserves LAMMPS's nonlinear kick/cap ordering.
 */
bool applyExactRespaSoftStartHalfKick(real        dt,
                                      const RVec& inverseMassPerDim,
                                      const RVec& physicalForce,
                                      const RVec* langevinForce,
                                      real        maximumSpeedNmPerPs,
                                      RVec*       velocity);

/*! \brief Refreshes the slow-level Langevin force using current boundary velocities.
 *
 * With zeroRandomForce enabled, the component-wise mean of the random (not drag)
 * force is removed globally, matching LAMMPS `fix langevin ... zero yes`.
 */
void refreshExactRespaSoftStartLangevinForce(ExactRespaSoftStartState* state,
                                             int64_t                   outerBoundary,
                                             const t_mdatoms&          mdatoms,
                                             ArrayRef<const RVec>      velocity,
                                             const int*                globalAtomIndices,
                                             int                       globalAtomIndicesCount,
                                             const MpiComm&            mpiComm);

} // namespace gmx

#endif
