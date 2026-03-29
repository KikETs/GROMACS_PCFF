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
#ifndef GMX_MULTIPLETIMESTEPPING_H
#define GMX_MULTIPLETIMESTEPPING_H

#include <cstdint>
#include <string>
#include <vector>

#include "gromacs/mdtypes/exactrespaparameters.h"
#include "gromacs/mdtypes/mtstypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/real.h"

struct t_inputrec;

namespace gmx
{

//! Names for the MTS execution modes
static const gmx::EnumerationArray<MtsMode, std::string> mtsModeNames = { "legacy", "lammps-respa" };

//! Names for the MTS force groups
static const gmx::EnumerationArray<MtsForceGroups, std::string> mtsForceGroupNames = {
    "longrange-nonbonded",
    "nonbonded",
    "nonbonded-inner",
    "nonbonded-middle",
    "nonbonded-outer",
    "pair",
    "bond",
    "dihedral",
    "improper",
    "angle",
    "pull",
    "awh"
};

/*! \brief Returns the MTS level at which a force group is to be computed
 *
 * \param[in] mtsLevels  List of force groups for each MTS level, can be empty without MTS
 * \param[in] mtsForceGroup  The force group to query the MTS level for
 */
int forceGroupMtsLevel(ArrayRef<const MtsLevel> mtsLevels, MtsForceGroups mtsForceGroup);

//! Returns the factor for the MTS level that owns \p mtsForceGroup or 1 without MTS
int forceGroupMtsFactor(ArrayRef<const MtsLevel> mtsLevels, MtsForceGroups mtsForceGroup);

//! Returns the highest active MTS level at \p step or 0 without MTS
int highestActiveMtsLevel(ArrayRef<const MtsLevel> mtsLevels, int64_t step);

//! Returns whether \p mtsLevel is active at \p step or always true without MTS
bool mtsLevelIsActive(ArrayRef<const MtsLevel> mtsLevels, int mtsLevel, int64_t step);

/*! \brief Returns the interval in steps at which the non-bonded pair forces are calculated
 *
 * Note: returns 1 when multiple time-stepping is not activated.
 */
int nonbondedMtsFactor(const t_inputrec& ir);

//! Returns the MTS level that owns a specific real-space nonbonded contribution
int nonbondedRespaContributionMtsLevel(const t_inputrec& ir, MtsNonbondedRespaContribution contribution);

//! Returns the base-step event ordering that corresponds to LAMMPS recursive r-RESPA
LammpsRespaBaseStepTrace lammpsRespaBaseStepTrace(ArrayRef<const MtsLevel> mtsLevels, int64_t baseStep);

//! Struct for passing the MTS mdp options to setupMtsLevels()
struct GromppMtsOpts
{
    //! The MTS execution mode
    MtsMode mode = MtsMode::Legacy;
    //! The number of MTS levels
    int numLevels = 0;
    //! The names of the force groups assigned by the user to levels 2..N, internal indices 1..N-1
    std::vector<std::string> levelForces;
    //! The step factors assigned by the user to levels 2..N, internal indices 1..N-1
    std::vector<int> levelFactors;
    //! Exact LAMMPS-style r-RESPA settings
    LammpsRespaParameters lammpsRespa;
};

/*! \brief Sets up and returns the MTS levels and checks requirements of MTS
 *
 * Appends errors about allowed input values ir to errorMessages, when not nullptr.
 *
 * \param[in]     mtsOpts        Options for setting the MTS levels
 * \param[in,out] errorMessages  List of error messages, can be nullptr
 */
std::vector<MtsLevel> setupMtsLevels(const GromppMtsOpts& mtsOpts, std::vector<std::string>* errorMessages);

/*! \brief Returns whether we use MTS and the MTS setup is internally valid
 *
 * Note that setupMtsLevels would have returned at least one error message
 * when this function returns false
 */
bool haveValidMtsSetup(const t_inputrec& ir);

/*! \brief Checks whether the MTS requirements on other algorithms and output frequencies are met
 *
 * Note: exits with an assertion failure when
 * ir.useMts == true && haveValidMtsSetup(ir) == false
 *
 * \param[in] ir  Complete input record
 * \returns list of error messages, empty when all MTS requirements are met
 */
std::vector<std::string> checkMtsRequirements(const t_inputrec& ir);

} // namespace gmx

#endif /* GMX_MULTIPLETIMESTEPPING_H */
