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
#include "gmxpre.h"

#include "multipletimestepping.h"

#include <algorithm>
#include <memory>
#include <optional>

#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/pull_params.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

namespace
{

void assignForceGroupToLevel(std::vector<MtsLevel>*                                           mtsLevels,
                             const int                                                        mtsLevel,
                             const MtsForceGroups                                             forceGroup,
                             std::bitset<static_cast<int>(MtsForceGroups::Count)>*            assignedForceGroups,
                             std::vector<std::string>*                                        errorMessages,
                             const char*                                                      forceGroupName)
{
    GMX_RELEASE_ASSERT(mtsLevels != nullptr, "Need valid MTS levels");
    GMX_RELEASE_ASSERT(assignedForceGroups != nullptr, "Need valid group assignment bitset");

    if (mtsLevel < 0 || mtsLevel >= static_cast<int>(mtsLevels->size()))
    {
        if (errorMessages)
        {
            errorMessages->push_back(
                    gmx::formatString("MTS level for '%s' should be within [1, %td]",
                                      forceGroupName,
                                      mtsLevels->size()));
        }
        return;
    }

    const int forceGroupIndex = static_cast<int>(forceGroup);
    if ((*assignedForceGroups)[forceGroupIndex])
    {
        if (errorMessages)
        {
            errorMessages->push_back(
                    gmx::formatString("MTS force group '%s' is assigned to more than one level", forceGroupName));
        }
        return;
    }

    (*mtsLevels)[mtsLevel].forceGroups.set(forceGroupIndex);
    assignedForceGroups->set(forceGroupIndex);
}

void validateStepFactors(ArrayRef<const MtsLevel> mtsLevels, std::vector<std::string>* errorMessages)
{
    for (int mtsLevel = 1; mtsLevel < static_cast<int>(mtsLevels.size()); mtsLevel++)
    {
        const int previousStepFactor = mtsLevels[mtsLevel - 1].stepFactor;
        const int stepFactor         = mtsLevels[mtsLevel].stepFactor;
        if (errorMessages && stepFactor <= previousStepFactor)
        {
            errorMessages->emplace_back(gmx::formatString(
                    "mts-level%d-factor should be larger than mts-level%d-factor", mtsLevel + 1, mtsLevel));
        }
        else if (errorMessages && stepFactor % previousStepFactor != 0)
        {
            errorMessages->emplace_back(gmx::formatString(
                    "mts-level%d-factor should be a multiple of mts-level%d-factor", mtsLevel + 1, mtsLevel));
        }
    }
}

void setupLegacyMtsLevels(const GromppMtsOpts&                mtsOpts,
                          std::vector<MtsLevel>*              mtsLevels,
                          std::vector<std::string>*           errorMessages,
                          std::bitset<static_cast<int>(MtsForceGroups::Count)>* assignedForceGroups)
{
    if (mtsOpts.levelForces.size() != static_cast<size_t>(mtsOpts.numLevels - 1)
        || mtsOpts.levelFactors.size() != static_cast<size_t>(mtsOpts.numLevels - 1))
    {
        if (errorMessages)
        {
            errorMessages->emplace_back("Internal MTS option parsing mismatch");
        }
        return;
    }

    for (int mtsLevel = 1; mtsLevel < mtsOpts.numLevels; mtsLevel++)
    {
        const std::string& levelForceString = mtsOpts.levelForces[mtsLevel - 1];
        (*mtsLevels)[mtsLevel].stepFactor   = mtsOpts.levelFactors[mtsLevel - 1];

        if (levelForceString.empty() && errorMessages)
        {
            errorMessages->emplace_back(
                    gmx::formatString("mts-level%d-forces should not be empty", mtsLevel + 1));
        }

        const std::vector<std::string> inputForceGroups = gmx::splitString(levelForceString);
        auto&                          forceGroups      = (*mtsLevels)[mtsLevel].forceGroups;
        for (const auto& inputForceGroup : inputForceGroups)
        {
            bool found     = false;
            int  nameIndex = 0;
            for (const auto& forceGroupName : gmx::mtsForceGroupNames)
            {
                if (gmx::equalCaseInsensitive(inputForceGroup, forceGroupName))
                {
                    if ((*assignedForceGroups)[nameIndex] && errorMessages)
                    {
                        errorMessages->push_back(gmx::formatString(
                                "MTS force group '%s' is assigned to more than one level",
                                inputForceGroup.c_str()));
                    }
                    forceGroups.set(nameIndex);
                    assignedForceGroups->set(nameIndex);
                    found = true;
                }
                nameIndex++;
            }
            if (!found && errorMessages)
            {
                errorMessages->push_back(
                        gmx::formatString("Unknown MTS force group '%s'", inputForceGroup.c_str()));
            }
        }
    }
}

void setupLammpsRespaLevels(const GromppMtsOpts&                mtsOpts,
                            std::vector<MtsLevel>*              mtsLevels,
                            std::vector<std::string>*           errorMessages,
                            std::bitset<static_cast<int>(MtsForceGroups::Count)>* assignedForceGroups)
{
    const auto& respa = mtsOpts.lammpsRespa;

    for (int mtsLevel = 1; mtsLevel < mtsOpts.numLevels; mtsLevel++)
    {
        if (!gmx::splitString(mtsOpts.levelForces[mtsLevel - 1]).empty() && errorMessages)
        {
            errorMessages->emplace_back(gmx::formatString(
                    "With mts-mode = %s, mts-level%d-forces should be left empty; force ownership is defined by the mts-respa-* options",
                    mtsModeNames[MtsMode::LammpsRespa].c_str(),
                    mtsLevel + 1));
        }
    }

    assignForceGroupToLevel(
            mtsLevels, respa.bondLevel, MtsForceGroups::Bond, assignedForceGroups, errorMessages, "bond");
    assignForceGroupToLevel(
            mtsLevels, respa.angleLevel, MtsForceGroups::Angle, assignedForceGroups, errorMessages, "angle");
    assignForceGroupToLevel(mtsLevels,
                            respa.dihedralLevel,
                            MtsForceGroups::Dihedral,
                            assignedForceGroups,
                            errorMessages,
                            "dihedral");
    assignForceGroupToLevel(mtsLevels,
                            respa.improperLevel,
                            MtsForceGroups::Improper,
                            assignedForceGroups,
                            errorMessages,
                            "improper");
    assignForceGroupToLevel(
            mtsLevels, respa.pair14Level, MtsForceGroups::Pair, assignedForceGroups, errorMessages, "pair");
    assignForceGroupToLevel(mtsLevels,
                            respa.kspaceLevel,
                            MtsForceGroups::LongrangeNonbonded,
                            assignedForceGroups,
                            errorMessages,
                            "kspace");

    if (respa.hasPairSplitting())
    {
        if (respa.innerLevel < 0 || respa.outerLevel < 0)
        {
            if (errorMessages)
            {
                errorMessages->emplace_back(
                        "Exact LAMMPS-style r-RESPA pair splitting requires both inner and outer levels");
            }
        }
        if (respa.hasMiddle() && !(respa.innerLevel <= respa.middleLevel && respa.middleLevel <= respa.outerLevel))
        {
            if (errorMessages)
            {
                errorMessages->emplace_back(
                        "With exact LAMMPS-style r-RESPA, inner/middle/outer levels should be nested");
            }
        }
        if (!respa.hasMiddle() && respa.innerLevel > respa.outerLevel)
        {
            if (errorMessages)
            {
                errorMessages->emplace_back(
                        "With exact LAMMPS-style r-RESPA, the inner level should not be slower than the outer level");
            }
        }

        assignForceGroupToLevel(mtsLevels,
                                respa.innerLevel,
                                MtsForceGroups::NonbondedInner,
                                assignedForceGroups,
                                errorMessages,
                                "inner");
        if (respa.hasMiddle())
        {
            assignForceGroupToLevel(mtsLevels,
                                    respa.middleLevel,
                                    MtsForceGroups::NonbondedMiddle,
                                    assignedForceGroups,
                                    errorMessages,
                                    "middle");
        }
        assignForceGroupToLevel(mtsLevels,
                                respa.outerLevel,
                                MtsForceGroups::NonbondedOuter,
                                assignedForceGroups,
                                errorMessages,
                                "outer");

        if (respa.hasMiddle())
        {
            if (!(respa.innerOff < respa.innerOn && respa.innerOn < respa.outerOn
                  && respa.outerOn < respa.outerOff))
            {
                if (errorMessages)
                {
                    errorMessages->emplace_back(
                            "Exact LAMMPS-style r-RESPA middle splitting requires inner-off < inner-on < outer-on < outer-off");
                }
            }
        }
        else if (!(respa.outerOn < respa.outerOff))
        {
            if (errorMessages)
            {
                errorMessages->emplace_back(
                        "Exact LAMMPS-style r-RESPA inner/outer splitting requires outer-on < outer-off");
            }
        }
    }
    else
    {
        assignForceGroupToLevel(mtsLevels,
                                respa.pairLevel,
                                MtsForceGroups::Nonbonded,
                                assignedForceGroups,
                                errorMessages,
                                "pair");
    }
}

} // namespace

int forceGroupMtsLevel(ArrayRef<const MtsLevel> mtsLevels, const MtsForceGroups mtsForceGroup)
{
    if (mtsLevels.empty())
    {
        return 0;
    }

    const int forceGroupIndex = static_cast<int>(mtsForceGroup);
    for (int mtsLevel = 0; mtsLevel < static_cast<int>(mtsLevels.size()); mtsLevel++)
    {
        if (mtsLevels[mtsLevel].forceGroups[forceGroupIndex])
        {
            return mtsLevel;
        }
    }

    GMX_RELEASE_ASSERT(false, "Each force group should belong to exactly one MTS level");
    return 0;
}

int forceGroupMtsFactor(ArrayRef<const MtsLevel> mtsLevels, const MtsForceGroups mtsForceGroup)
{
    return mtsLevels.empty() ? 1 : mtsLevels[forceGroupMtsLevel(mtsLevels, mtsForceGroup)].stepFactor;
}

int highestActiveMtsLevel(ArrayRef<const MtsLevel> mtsLevels, const int64_t step)
{
    if (mtsLevels.empty())
    {
        return 0;
    }

    int highestLevel = 0;
    for (int mtsLevel = 1; mtsLevel < static_cast<int>(mtsLevels.size()); mtsLevel++)
    {
        if (step % mtsLevels[mtsLevel].stepFactor == 0)
        {
            highestLevel = mtsLevel;
        }
        else
        {
            break;
        }
    }

    return highestLevel;
}

bool mtsLevelIsActive(ArrayRef<const MtsLevel> mtsLevels, const int mtsLevel, const int64_t step)
{
    GMX_RELEASE_ASSERT(mtsLevels.empty() || (mtsLevel >= 0 && mtsLevel < static_cast<int>(mtsLevels.size())),
                       "MTS level index should be valid");
    return (mtsLevels.empty() || highestActiveMtsLevel(mtsLevels, step) >= mtsLevel);
}

int nonbondedMtsFactor(const t_inputrec& ir)
{
    if (ir.useMts && !ir.mtsLevels.empty())
    {
        if (ir.mtsMode == MtsMode::LammpsRespa && ir.lammpsRespa.hasPairSplitting())
        {
            int factor = 1;
            for (const auto contribution : { MtsNonbondedRespaContribution::Inner,
                                             MtsNonbondedRespaContribution::Middle,
                                             MtsNonbondedRespaContribution::Outer })
            {
                const int mtsLevel = nonbondedRespaContributionMtsLevel(ir, contribution);
                if (mtsLevel >= 0)
                {
                    factor = std::max(factor, ir.mtsLevels[mtsLevel].stepFactor);
                }
            }
            return factor;
        }
        return forceGroupMtsFactor(ir.mtsLevels, MtsForceGroups::Nonbonded);
    }
    else
    {
        return 1;
    }
}

int nonbondedRespaContributionMtsLevel(const t_inputrec& ir, const MtsNonbondedRespaContribution contribution)
{
    if (!ir.useMts || ir.mtsLevels.empty())
    {
        return 0;
    }
    if (ir.mtsMode != MtsMode::LammpsRespa)
    {
        return (contribution == MtsNonbondedRespaContribution::Full)
                       ? forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::Nonbonded)
                       : -1;
    }

    switch (contribution)
    {
        case MtsNonbondedRespaContribution::Full:
            return ir.lammpsRespa.hasPairSplitting() ? -1 : ir.lammpsRespa.pairLevel;
        case MtsNonbondedRespaContribution::Inner: return ir.lammpsRespa.innerLevel;
        case MtsNonbondedRespaContribution::Middle: return ir.lammpsRespa.middleLevel;
        case MtsNonbondedRespaContribution::Outer: return ir.lammpsRespa.outerLevel;
        default: GMX_RELEASE_ASSERT(false, "Invalid real-space nonbonded r-RESPA contribution");
    }
    return -1;
}

LammpsRespaBaseStepTrace lammpsRespaBaseStepTrace(ArrayRef<const MtsLevel> mtsLevels, const int64_t baseStep)
{
    LammpsRespaBaseStepTrace trace;
    if (mtsLevels.empty())
    {
        trace.initialKickLevels = { 0 };
        trace.refreshedForceLevels = { 0 };
        trace.finalKickLevels = { 0 };
        return trace;
    }

    const int highestInitialLevel = highestActiveMtsLevel(mtsLevels, baseStep);
    for (int mtsLevel = highestInitialLevel; mtsLevel >= 0; mtsLevel--)
    {
        trace.initialKickLevels.push_back(mtsLevel);
    }

    const int highestFinalLevel = highestActiveMtsLevel(mtsLevels, baseStep + 1);
    for (int mtsLevel = 0; mtsLevel <= highestFinalLevel; mtsLevel++)
    {
        trace.refreshedForceLevels.push_back(mtsLevel);
        trace.finalKickLevels.push_back(mtsLevel);
    }

    return trace;
}

std::vector<MtsLevel> setupMtsLevels(const GromppMtsOpts& mtsOpts, std::vector<std::string>* errorMessages)
{
    std::vector<MtsLevel> mtsLevels;

    if (mtsOpts.numLevels < 2 || mtsOpts.numLevels > c_maxMtsLevels)
    {
        if (errorMessages)
        {
            errorMessages->emplace_back(
                    gmx::formatString("Only 2 <= mts-levels <= %d is supported", c_maxMtsLevels));
        }
    }
    else
    {
        mtsLevels.resize(mtsOpts.numLevels);
        std::bitset<static_cast<int>(MtsForceGroups::Count)> assignedForceGroups;

        mtsLevels[0].stepFactor = 1;
        for (int mtsLevel = 1; mtsLevel < mtsOpts.numLevels; mtsLevel++)
        {
            mtsLevels[mtsLevel].stepFactor =
                    (mtsLevel - 1 < static_cast<int>(mtsOpts.levelFactors.size()))
                            ? mtsOpts.levelFactors[mtsLevel - 1]
                            : 0;
        }

        if (mtsOpts.mode == MtsMode::LammpsRespa)
        {
            setupLammpsRespaLevels(mtsOpts, &mtsLevels, errorMessages, &assignedForceGroups);
        }
        else
        {
            setupLegacyMtsLevels(mtsOpts, &mtsLevels, errorMessages, &assignedForceGroups);
        }

        validateStepFactors(mtsLevels, errorMessages);

        // Level 0 owns the fast-force complement plus any force groups explicitly assigned there
        // (required for exact LAMMPS-style r-RESPA inner/bond ownership).
        mtsLevels[0].forceGroups |= ~assignedForceGroups;
    }

    return mtsLevels;
}

bool haveValidMtsSetup(const t_inputrec& ir)
{
    if (!ir.useMts || ir.mtsLevels.size() < 2 || ir.mtsLevels.size() > c_maxMtsLevels)
    {
        return false;
    }

    for (int mtsLevel = 1; mtsLevel < static_cast<int>(ir.mtsLevels.size()); mtsLevel++)
    {
        if (ir.mtsLevels[mtsLevel].stepFactor <= ir.mtsLevels[mtsLevel - 1].stepFactor
            || ir.mtsLevels[mtsLevel].stepFactor % ir.mtsLevels[mtsLevel - 1].stepFactor != 0)
        {
            return false;
        }
    }

    return true;
}

namespace
{

//! Checks whether \p nstValue is a multiple of the largest MTS step, returns an error string for parameter \p param when this is not the case
std::optional<std::string> checkMtsInterval(ArrayRef<const MtsLevel> mtsLevels, const char* param, const int nstValue)
{
    GMX_RELEASE_ASSERT(mtsLevels.size() >= 2, "Need at least two levels for MTS");

    const int mtsFactor = mtsLevels.back().stepFactor;
    if (nstValue % mtsFactor == 0)
    {
        return {};
    }
    else
    {
        return gmx::formatString(
                "With MTS, %s = %d should be a multiple of mts-factor = %d", param, nstValue, mtsFactor);
    }
}

} // namespace

std::vector<std::string> checkMtsRequirements(const t_inputrec& ir)
{
    std::vector<std::string> errorMessages;

    if (!ir.useMts)
    {
        return errorMessages;
    }

    GMX_RELEASE_ASSERT(haveValidMtsSetup(ir), "MTS setup should be valid here");

    ArrayRef<const MtsLevel> mtsLevels = ir.mtsLevels;

    if (ir.mtsMode == MtsMode::LammpsRespa)
    {
        if (static_cast<int>(mtsLevels.size()) != 3)
        {
            errorMessages.emplace_back(
                    "Exact LAMMPS-style r-RESPA CPU validation is currently frozen only for mts-levels = 3");
        }
        if (ir.eI != IntegrationAlgorithm::MD && ir.eI != IntegrationAlgorithm::VV)
        {
            errorMessages.push_back(
                    gmx::formatString("Exact LAMMPS-style r-RESPA is only supported with integrator %s or %s",
                                      enumValueToString(IntegrationAlgorithm::MD),
                                      enumValueToString(IntegrationAlgorithm::VV)));
        }
        if (ir.vdwtype != VanDerWaalsType::Cut)
        {
            errorMessages.emplace_back(
                    "Exact LAMMPS-style r-RESPA currently requires vdw-type = cut-off");
        }
        if (ir.vdw_modifier != InteractionModifiers::None)
        {
            errorMessages.emplace_back(
                    "Exact LAMMPS-style r-RESPA currently requires vdw-modifier = none");
        }
        if (!usingPmeOrEwald(ir.coulombtype))
        {
            errorMessages.emplace_back(
                    "Exact LAMMPS-style r-RESPA currently requires coulombtype = PME or Ewald");
        }
        if (ir.coulomb_modifier != InteractionModifiers::None)
        {
            errorMessages.emplace_back(
                    "Exact LAMMPS-style r-RESPA currently requires coulomb-modifier = none");
        }
        if (ir.lammpsRespa.hasPairSplitting())
        {
            const real shortestPairCutoff = std::min(ir.rcoulomb, ir.rvdw);
            if (ir.lammpsRespa.outerOff > shortestPairCutoff)
            {
                errorMessages.emplace_back(gmx::formatString(
                        "Exact LAMMPS-style r-RESPA requires mts-respa-outer-off = %g to be <= min(rcoulomb, rvdw) = %g",
                        ir.lammpsRespa.outerOff,
                        shortestPairCutoff));
            }
        }
        if (ir.nstlist > 1)
        {
            const real longestPairCutoff = std::max(ir.rcoulomb, ir.rvdw);
            if (ir.rlist <= longestPairCutoff)
            {
                errorMessages.emplace_back(gmx::formatString(
                        "Exact LAMMPS-style r-RESPA with nstlist = %d requires rlist = %g to be > max(rcoulomb, rvdw) = %g so the plain Verlet pairlist has a real buffer",
                        ir.nstlist,
                        ir.rlist,
                        longestPairCutoff));
            }
        }
    }
    else
    {
        if (ir.eI != IntegrationAlgorithm::MD)
        {
            errorMessages.push_back(
                    gmx::formatString("Multiple time stepping is only supported with integrator %s",
                                      enumValueToString(IntegrationAlgorithm::MD)));
        }
    }

    if ((usingFullElectrostatics(ir.coulombtype) || usingLJPme(ir.vdwtype))
        && forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::LongrangeNonbonded)
                   != static_cast<int>(ir.mtsLevels.size()) - 1)
    {
        errorMessages.emplace_back(gmx::formatString(
                "With long-range electrostatics and/or LJ treatment, the long-range part "
                "has to be part of the mts-level%d-forces",
                static_cast<int>(ir.mtsLevels.size())));
    }

    std::optional<std::string> mesg;
    if (ir.nstcalcenergy > 0)
    {
        if ((mesg = checkMtsInterval(mtsLevels, "nstcalcenergy", ir.nstcalcenergy)))
        {
            errorMessages.push_back(mesg.value());
        }
    }
    if ((mesg = checkMtsInterval(mtsLevels, "nstenergy", ir.nstenergy)))
    {
        errorMessages.push_back(mesg.value());
    }
    if ((mesg = checkMtsInterval(mtsLevels, "nstlog", ir.nstlog)))
    {
        errorMessages.push_back(mesg.value());
    }
    if ((mesg = checkMtsInterval(mtsLevels, "nstfout", ir.nstfout)))
    {
        errorMessages.push_back(mesg.value());
    }
    if (ir.efep != FreeEnergyPerturbationType::No)
    {
        if ((mesg = checkMtsInterval(mtsLevels, "nstdhdl", ir.fepvals->nstdhdl)))
        {
            errorMessages.push_back(mesg.value());
        }
    }
    const int nonbondedFactor = nonbondedMtsFactor(ir);
    if (nonbondedFactor > 1)
    {
        if (ir.nstlist % nonbondedFactor != 0)
        {
            errorMessages.push_back(gmx::formatString(
                    "With MTS, nstlist = %d should be a multiple of the nonbonded mts-factor = %d",
                    ir.nstlist,
                    nonbondedFactor));
        }
    }

    if (ir.bPull)
    {
        const int pullMtsLevel  = gmx::forceGroupMtsLevel(ir.mtsLevels, gmx::MtsForceGroups::Pull);
        const int mtsStepFactor = ir.mtsLevels[pullMtsLevel].stepFactor;
        if (ir.pull->nstxout % mtsStepFactor != 0)
        {
            errorMessages.emplace_back("pull-nstxout should be a multiple of mts-factor");
        }
        if (ir.pull->nstfout % mtsStepFactor != 0)
        {
            errorMessages.emplace_back("pull-nstfout should be a multiple of mts-factor");
        }
    }

    return errorMessages;
}

} // namespace gmx
