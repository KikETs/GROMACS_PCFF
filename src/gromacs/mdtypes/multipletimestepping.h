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

#include <bitset>
#include <cstdint>
#include <string>
#include <vector>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/real.h"

struct t_inputrec;

namespace gmx
{

//! Selects the MTS semantics to apply.
enum class MtsMode : int
{
    Legacy,      //!< The original GROMACS force-group MTS behavior
    LammpsRespa, //!< Exact LAMMPS-style r-RESPA force-class behavior
    Count
};

//! Names for the MTS execution modes
static const gmx::EnumerationArray<MtsMode, std::string> mtsModeNames = { "legacy", "lammps-respa" };

//! The real-space nonbonded contribution to compute for one kernel launch.
enum class MtsNonbondedRespaContribution : int
{
    Full,
    Inner,
    Middle,
    Outer,
    Count
};

//! Exact LAMMPS-style r-RESPA pair-splitting weights for a single interatomic distance.
struct LammpsRespaPairSplitWeights
{
    real inner  = 0;
    real middle = 0;
    real outer  = 1;
};

//! The force-output sink used by an exact LAMMPS-style r-RESPA nonbonded contribution.
enum class LammpsRespaNonbondedOutputSinkKind : int
{
    ShiftForce,
    ForceWithVirial,
    Count
};

//! Routing contract for one active exact LAMMPS-style r-RESPA nonbonded contribution.
struct LammpsRespaNonbondedOutputSink
{
    MtsNonbondedRespaContribution         contribution = MtsNonbondedRespaContribution::Full;
    int                                   mtsLevel     = -1;
    LammpsRespaNonbondedOutputSinkKind    sinkKind     = LammpsRespaNonbondedOutputSinkKind::ShiftForce;
    bool                                  accumulateEnergy = false;
};

//! Exact LAMMPS-style r-RESPA settings stored in the inputrec.
struct LammpsRespaParameters
{
    bool enabled = false;
    int  bondLevel = 0;
    int  angleLevel = 0;
    int  dihedralLevel = 0;
    int  improperLevel = 0;
    int  pair14Level = 0;
    int  pairLevel = 0;
    int  kspaceLevel = 0;
    int  innerLevel = -1;
    int  middleLevel = -1;
    int  outerLevel = -1;
    real innerOff = 0;
    real innerOn = 0;
    real outerOn = 0;
    real outerOff = 0;

    bool hasPairSplitting() const { return innerLevel >= 0 || middleLevel >= 0 || outerLevel >= 0; }
    bool hasMiddle() const { return middleLevel >= 0; }
};

//! Force group available for selection for multiple time step integration
enum class MtsForceGroups : int
{
    LongrangeNonbonded, //!< PME-mesh or Ewald for electrostatics and/or LJ
    Nonbonded,          //!< Unsplit real-space non-bonded pair interactions
    NonbondedInner,     //!< Inner real-space r-RESPA contribution
    NonbondedMiddle,    //!< Middle real-space r-RESPA contribution
    NonbondedOuter,     //!< Outer real-space r-RESPA contribution
    Pair,               //!< Listed 1-4 pair interactions
    Bond,               //!< Bond interactions
    Dihedral,           //!< Proper dihedrals, including cmap (not restraints)
    Improper,           //!< Improper dihedrals
    Angle,              //!< Bonded angle potentials (not restraints)
    Pull,               //!< COM pulling
    Awh,                //!< Accelerated weight histogram method
    Count               //!< The number of groups above
};

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

//! Setting for a single level for multiple time step integration
struct MtsLevel
{
    //! The force group selection for this level;
    std::bitset<static_cast<int>(MtsForceGroups::Count)> forceGroups;
    //! The factor between the base, fastest, time step and the time step for this level
    int stepFactor;
};

//! Explicit base-step trace for mapping GROMACS base steps to LAMMPS recursive r-RESPA events
struct LammpsRespaBaseStepTrace
{
    //! Levels that apply their initial half-kick before the base-step drift, ordered slow -> fast
    std::vector<int> initialKickLevels;
    //! Levels whose forces are refreshed after the drift to x(t + dt), ordered fast -> slow
    std::vector<int> refreshedForceLevels;
    //! Levels that apply their final half-kick after the force refresh, ordered fast -> slow
    std::vector<int> finalKickLevels;
};

//! Maximum number of MTS levels supported by the CPU exact r-RESPA implementation
constexpr int c_maxMtsLevels = 8;

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

//! Returns exact LAMMPS-style r-RESPA pair-splitting weights for distance \p r
LammpsRespaPairSplitWeights computeLammpsRespaPairSplitWeights(const t_inputrec& ir, real r);

//! Returns the active output sinks for exact LAMMPS-style r-RESPA real-space nonbonded contributions
std::vector<LammpsRespaNonbondedOutputSink> activeLammpsRespaNonbondedOutputSinks(const t_inputrec& ir,
                                                                                   int               highestActiveMtsLevel,
                                                                                   bool              computeVirial,
                                                                                   bool              computeEnergy);

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
