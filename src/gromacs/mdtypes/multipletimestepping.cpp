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

namespace
{

real respaSwitchIn(const real r, const real off, const real on)
{
    if (on <= off)
    {
        return (r >= on ? 1.0_real : 0.0_real);
    }
    if (r <= off)
    {
        return 0.0_real;
    }
    if (r >= on)
    {
        return 1.0_real;
    }

    const real x = (r - off) / (on - off);
    return x * x * (3.0_real - 2.0_real * x);
}

} // namespace

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

LammpsRespaPairSplitWeights computeLammpsRespaPairSplitWeights(const t_inputrec& ir, const real r)
{
    LammpsRespaPairSplitWeights weights;
    if (!useExactRespa(ir))
    {
        GMX_RELEASE_ASSERT(ir.mtsMode != MtsMode::LammpsRespa && !ir.lammpsRespa.enabled,
                           "Exact LAMMPS-style r-RESPA pair splitting should not use legacy MTS state");
        return weights;
    }

    assertExactRespaOwnsNoLegacyMtsState(ir);
    const auto& respa = ir.exactRespa.forceLayout;

    if (!respa.hasPairSplitting())
    {
        return weights;
    }

    if (respa.hasMiddle())
    {
        const real switchIntoMiddle = respaSwitchIn(r, respa.innerOff, respa.innerOn);
        const real switchIntoOuter  = respaSwitchIn(r, respa.outerOn, respa.outerOff);
        weights.inner               = 1.0_real - switchIntoMiddle;
        weights.middle              = switchIntoMiddle * (1.0_real - switchIntoOuter);
        weights.outer               = switchIntoOuter;
    }
    else
    {
        const real switchIntoOuter = respaSwitchIn(r, respa.outerOn, respa.outerOff);
        weights.inner              = 1.0_real - switchIntoOuter;
        weights.middle             = 0.0_real;
        weights.outer              = switchIntoOuter;
    }

    return weights;
}

std::vector<LammpsRespaNonbondedOutputSink> activeLammpsRespaNonbondedOutputSinks(const t_inputrec& ir,
                                                                                   const int         highestActiveMtsLevel,
                                                                                   const bool        computeVirial,
                                                                                   const bool        computeEnergy)
{
    std::vector<LammpsRespaNonbondedOutputSink> sinks;
    sinks.reserve(3);
    if (!useExactRespa(ir))
    {
        GMX_RELEASE_ASSERT(ir.mtsMode != MtsMode::LammpsRespa && !ir.lammpsRespa.enabled,
                           "Exact LAMMPS-style r-RESPA output sinks should not use legacy MTS state");
        return sinks;
    }

    assertExactRespaOwnsNoLegacyMtsState(ir);
    const auto& respa = ir.exactRespa.forceLayout;
    const auto appendSink = [&](const MtsNonbondedRespaContribution contribution)
    {
        const int mtsLevel = nonbondedRespaContributionMtsLevel(ir, contribution);
        if (mtsLevel < 0 || mtsLevel > highestActiveMtsLevel)
        {
            return;
        }

        const bool directVirialContribution =
                computeVirial && (contribution == MtsNonbondedRespaContribution::Outer
                                  || contribution == MtsNonbondedRespaContribution::Full);
        const bool accumulateEnergy =
                computeEnergy && (contribution == MtsNonbondedRespaContribution::Outer
                                  || contribution == MtsNonbondedRespaContribution::Full);

        sinks.push_back({ contribution,
                          mtsLevel,
                          directVirialContribution ? LammpsRespaNonbondedOutputSinkKind::ForceWithVirial
                                                   : LammpsRespaNonbondedOutputSinkKind::ShiftForce,
                          accumulateEnergy });
    };

    if (respa.hasPairSplitting())
    {
        for (const auto contribution : { MtsNonbondedRespaContribution::Inner,
                                         MtsNonbondedRespaContribution::Middle,
                                         MtsNonbondedRespaContribution::Outer })
        {
            appendSink(contribution);
        }
    }
    else
    {
        appendSink(MtsNonbondedRespaContribution::Full);
    }

    return sinks;
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

} // namespace gmx
