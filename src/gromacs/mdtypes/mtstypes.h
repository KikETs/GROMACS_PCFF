/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
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
#ifndef GMX_MDTYPES_MTSTYPES_H
#define GMX_MDTYPES_MTSTYPES_H

#include <bitset>
#include <cstdint>
#include <vector>

namespace gmx
{

//! Selects the MTS semantics to apply.
enum class MtsMode : int
{
    Legacy,      //!< The original GROMACS force-group MTS behavior
    LammpsRespa, //!< Exact LAMMPS-style r-RESPA force-class behavior
    Count
};

//! The real-space nonbonded contribution to compute for one kernel launch.
enum class MtsNonbondedRespaContribution : int
{
    Full,
    Inner,
    Middle,
    Outer,
    Count
};

//! Force group available for selection for multiple time step integration.
enum class MtsForceGroups : int
{
    LongrangeNonbonded,
    Nonbonded,
    NonbondedInner,
    NonbondedMiddle,
    NonbondedOuter,
    Pair,
    Bond,
    Dihedral,
    Improper,
    Angle,
    Pull,
    Awh,
    Count
};

//! Setting for a single level for multiple time step integration.
struct MtsLevel
{
    std::bitset<static_cast<int>(MtsForceGroups::Count)> forceGroups;
    int                                                  stepFactor = 0;
};

//! Explicit base-step trace for mapping GROMACS base steps to LAMMPS recursive r-RESPA events.
struct LammpsRespaBaseStepTrace
{
    std::vector<int> initialKickLevels;
    std::vector<int> refreshedForceLevels;
    std::vector<int> finalKickLevels;
};

//! Maximum number of MTS levels supported by the CPU exact r-RESPA implementation.
constexpr int c_maxMtsLevels = 8;

} // namespace gmx

#endif
