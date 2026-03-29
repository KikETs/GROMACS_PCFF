/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2020- The GROMACS Authors
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
 */
#ifndef GMX_EXACTRESPASCHEDULE_H
#define GMX_EXACTRESPASCHEDULE_H

#include <cstdint>
#include <string>
#include <vector>

#include "gromacs/mdtypes/exactrespaparameters.h"

struct t_inputrec;

namespace gmx
{

//! Standalone exact-r-RESPA base-step ordering equivalent to LAMMPS recursive recurse().
struct ExactRespaBaseStepTrace
{
    //! Levels that apply their initial half-kick before the base-step drift, ordered slow -> fast.
    std::vector<int> initialKickLevels;
    //! Levels whose forces are refreshed after the drift to x(t + dt), ordered fast -> slow.
    std::vector<int> refreshedForceLevels;
    //! Levels that apply their final half-kick after the force refresh, ordered fast -> slow.
    std::vector<int> finalKickLevels;
};

//! Returns the highest active exact-r-RESPA level at \p step or 0 without standalone metadata.
int highestActiveExactRespaLevel(const ExactRespaParameters& exactRespa, int64_t step);

//! Returns the step factor for \p level from standalone exact-r-RESPA metadata.
int exactRespaLevelStepFactor(const ExactRespaParameters& exactRespa, int level);

//! Returns the number of standalone exact-r-RESPA levels.
int exactRespaNumLevels(const t_inputrec& ir);

//! Returns the step factor for \p level from the standalone input record metadata.
int exactRespaLevelStepFactor(const t_inputrec& ir, int level);

//! Returns the slowest exact-r-RESPA step factor or 1 when exact r-RESPA is disabled.
int exactRespaSlowestStepFactor(const t_inputrec& ir);

//! Returns whether standalone exact r-RESPA uses split real-space pair ownership.
bool exactRespaHasPairSplitting(const t_inputrec& ir);

//! Returns the exact-r-RESPA level that owns the full nonbonded pair contribution.
int exactRespaNonbondedFullLevel(const t_inputrec& ir);

//! Returns the exact-r-RESPA level that owns the inner real-space pair contribution.
int exactRespaNonbondedInnerLevel(const t_inputrec& ir);

//! Returns the exact-r-RESPA level that owns the middle real-space pair contribution.
int exactRespaNonbondedMiddleLevel(const t_inputrec& ir);

//! Returns the exact-r-RESPA level that owns the outer real-space pair contribution.
int exactRespaNonbondedOuterLevel(const t_inputrec& ir);

//! Returns the exact-r-RESPA level that owns long-range electrostatics / LJ treatment.
int exactRespaLongrangeNonbondedLevel(const t_inputrec& ir);

//! Returns the exact-r-RESPA level that owns pull forces in the standalone path.
int exactRespaPullLevel(const t_inputrec& ir);

//! Returns the exact-r-RESPA level that owns AWH forces in the standalone path.
int exactRespaAwhLevel(const t_inputrec& ir);

//! Returns the exact-r-RESPA factor for the real-space pair computation interval.
int exactRespaNonbondedMtsFactor(const t_inputrec& ir);

//! Returns whether standalone exact r-RESPA metadata satisfies exact-path validation.
bool haveValidExactRespaSetup(const t_inputrec& ir);

//! Returns exact-path validation errors for standalone exact r-RESPA metadata.
std::vector<std::string> checkExactRespaRequirements(const t_inputrec& ir);

//! Returns the exact-r-RESPA base-step event ordering for \p baseStep.
ExactRespaBaseStepTrace exactRespaBaseStepTrace(const ExactRespaParameters& exactRespa, int64_t baseStep);

} // namespace gmx

#endif
