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

#include "gmxpre.h"

#include "gromacs/mdtypes/exactrespaschedule.h"
#include "gromacs/mdtypes/inputrec.h"

#include <algorithm>
#include <optional>

#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/pull_params.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

namespace
{

int highestActiveLevel(ArrayRef<const int> levelStepFactors, const int64_t step)
{
    if (levelStepFactors.empty())
    {
        return 0;
    }

    GMX_RELEASE_ASSERT(levelStepFactors[0] > 0, "Exact r-RESPA level 0 step factor must be positive");

    int highestLevel = 0;
    for (int level = 1; level < static_cast<int>(levelStepFactors.size()); ++level)
    {
        GMX_RELEASE_ASSERT(levelStepFactors[level] > 0,
                           "Exact r-RESPA step factors must be positive");
        if (step % levelStepFactors[level] == 0)
        {
            highestLevel = level;
        }
        else
        {
            break;
        }
    }

    return highestLevel;
}

std::optional<std::string> checkExactRespaInterval(const t_inputrec& ir, const char* param, const int nstValue)
{
    const int exactRespaFactor = exactRespaSlowestStepFactor(ir);
    if (nstValue % exactRespaFactor == 0)
    {
        return {};
    }

    return formatString("With exact-respa, %s = %d should be a multiple of exact-respa-factor = %d",
                        param,
                        nstValue,
                        exactRespaFactor);
}

} // namespace

int forceGroupMtsLevel(ArrayRef<const MtsLevel> mtsLevels, const MtsForceGroups mtsForceGroup)
{
    if (mtsLevels.empty())
    {
        return 0;
    }

    const int forceGroupIndex = static_cast<int>(mtsForceGroup);
    for (int mtsLevel = 0; mtsLevel < static_cast<int>(mtsLevels.size()); ++mtsLevel)
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

int nonbondedMtsFactor(const t_inputrec& ir)
{
    if (useExactRespa(ir))
    {
        return exactRespaNonbondedMtsFactor(ir);
    }

    if (ir.useMts && !ir.mtsLevels.empty())
    {
        return forceGroupMtsFactor(ir.mtsLevels, MtsForceGroups::Nonbonded);
    }

    return 1;
}

int nonbondedRespaContributionMtsLevel(const t_inputrec& ir, const MtsNonbondedRespaContribution contribution)
{
    if (useExactRespa(ir))
    {
        switch (contribution)
        {
            case MtsNonbondedRespaContribution::Full:
                return exactRespaHasPairSplitting(ir) ? -1 : exactRespaNonbondedFullLevel(ir);
            case MtsNonbondedRespaContribution::Inner: return exactRespaNonbondedInnerLevel(ir);
            case MtsNonbondedRespaContribution::Middle: return exactRespaNonbondedMiddleLevel(ir);
            case MtsNonbondedRespaContribution::Outer: return exactRespaNonbondedOuterLevel(ir);
            default: GMX_RELEASE_ASSERT(false, "Invalid real-space nonbonded r-RESPA contribution");
        }
    }

    if (!ir.useMts || ir.mtsLevels.empty())
    {
        return 0;
    }

    return (contribution == MtsNonbondedRespaContribution::Full)
                   ? forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::Nonbonded)
                   : -1;
}

int highestActiveExactRespaLevel(const ExactRespaParameters& exactRespa, const int64_t step)
{
    if (!exactRespa.enabled())
    {
        return 0;
    }

    GMX_RELEASE_ASSERT(!exactRespa.levelStepFactors.empty(),
                       "Exact r-RESPA requires standalone level step factors");
    return highestActiveLevel(exactRespa.levelStepFactors, step);
}

int exactRespaLevelStepFactor(const ExactRespaParameters& exactRespa, const int level)
{
    GMX_RELEASE_ASSERT(exactRespa.enabled(), "Exact r-RESPA level step factors require enabled metadata");
    GMX_RELEASE_ASSERT(level >= 0 && level < static_cast<int>(exactRespa.levelStepFactors.size()),
                       "Exact r-RESPA level index should be valid");

    const int stepFactor = exactRespa.levelStepFactors[level];
    GMX_RELEASE_ASSERT(stepFactor > 0, "Exact r-RESPA step factors must be positive");
    return stepFactor;
}

int exactRespaNumLevels(const t_inputrec& ir)
{
    GMX_RELEASE_ASSERT(ir.exactRespa.enabled(), "Standalone exact r-RESPA metadata must be enabled");
    GMX_RELEASE_ASSERT(!ir.exactRespa.levelStepFactors.empty(),
                       "Standalone exact r-RESPA metadata requires level step factors");
    return static_cast<int>(ir.exactRespa.levelStepFactors.size());
}

int exactRespaLevelStepFactor(const t_inputrec& ir, const int level)
{
    GMX_RELEASE_ASSERT(ir.exactRespa.enabled(), "Standalone exact r-RESPA metadata must be enabled");
    return exactRespaLevelStepFactor(ir.exactRespa, level);
}

int exactRespaSlowestStepFactor(const t_inputrec& ir)
{
    if (!ir.exactRespa.enabled())
    {
        return 1;
    }

    GMX_RELEASE_ASSERT(!ir.exactRespa.levelStepFactors.empty(),
                       "Standalone exact r-RESPA metadata requires level step factors");
    return ir.exactRespa.levelStepFactors.back();
}

bool exactRespaHasPairSplitting(const t_inputrec& ir)
{
    return ir.exactRespa.enabled() && ir.exactRespa.forceLayout.hasPairSplitting();
}

int exactRespaNonbondedFullLevel(const t_inputrec& ir)
{
    GMX_RELEASE_ASSERT(ir.exactRespa.enabled(), "Standalone exact r-RESPA metadata must be enabled");
    return ir.exactRespa.forceLayout.pairLevel;
}

int exactRespaNonbondedInnerLevel(const t_inputrec& ir)
{
    GMX_RELEASE_ASSERT(ir.exactRespa.enabled(), "Standalone exact r-RESPA metadata must be enabled");
    return ir.exactRespa.forceLayout.innerLevel;
}

int exactRespaNonbondedMiddleLevel(const t_inputrec& ir)
{
    GMX_RELEASE_ASSERT(ir.exactRespa.enabled(), "Standalone exact r-RESPA metadata must be enabled");
    return ir.exactRespa.forceLayout.middleLevel;
}

int exactRespaNonbondedOuterLevel(const t_inputrec& ir)
{
    GMX_RELEASE_ASSERT(ir.exactRespa.enabled(), "Standalone exact r-RESPA metadata must be enabled");
    return ir.exactRespa.forceLayout.outerLevel;
}

int exactRespaLongrangeNonbondedLevel(const t_inputrec& ir)
{
    GMX_RELEASE_ASSERT(ir.exactRespa.enabled(), "Standalone exact r-RESPA metadata must be enabled");
    return ir.exactRespa.forceLayout.kspaceLevel;
}

int exactRespaPullLevel(const t_inputrec& ir)
{
    GMX_RELEASE_ASSERT(ir.exactRespa.enabled(), "Standalone exact r-RESPA metadata must be enabled");
    return 0;
}

int exactRespaAwhLevel(const t_inputrec& ir)
{
    GMX_RELEASE_ASSERT(ir.exactRespa.enabled(), "Standalone exact r-RESPA metadata must be enabled");
    return 0;
}

int exactRespaNonbondedMtsFactor(const t_inputrec& ir)
{
    if (!ir.exactRespa.enabled())
    {
        return 1;
    }

    if (!exactRespaHasPairSplitting(ir))
    {
        return exactRespaLevelStepFactor(ir.exactRespa, exactRespaNonbondedFullLevel(ir));
    }

    int factor = 1;
    factor = std::max(factor, exactRespaLevelStepFactor(ir.exactRespa, exactRespaNonbondedInnerLevel(ir)));
    factor = std::max(factor, exactRespaLevelStepFactor(ir.exactRespa, exactRespaNonbondedOuterLevel(ir)));
    if (ir.exactRespa.forceLayout.hasMiddle())
    {
        factor = std::max(factor,
                          exactRespaLevelStepFactor(ir.exactRespa, exactRespaNonbondedMiddleLevel(ir)));
    }
    return factor;
}

bool haveValidMtsSetup(const t_inputrec& ir)
{
    if (useExactRespa(ir))
    {
        return haveValidExactRespaSetup(ir);
    }

    if (!useMtsSubstepping(ir) || ir.mtsLevels.size() < 2 || ir.mtsLevels.size() > c_maxMtsLevels)
    {
        return false;
    }

    for (int mtsLevel = 1; mtsLevel < static_cast<int>(ir.mtsLevels.size()); ++mtsLevel)
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

std::optional<std::string> checkMtsInterval(ArrayRef<const MtsLevel> mtsLevels, const char* param, const int nstValue)
{
    GMX_RELEASE_ASSERT(mtsLevels.size() >= 2, "Need at least two levels for MTS");

    const int mtsFactor = mtsLevels.back().stepFactor;
    if (nstValue % mtsFactor == 0)
    {
        return {};
    }

    return gmx::formatString(
            "With MTS, %s = %d should be a multiple of mts-factor = %d", param, nstValue, mtsFactor);
}

} // namespace

std::vector<std::string> checkMtsRequirements(const t_inputrec& ir)
{
    if (useExactRespa(ir))
    {
        return checkExactRespaRequirements(ir);
    }

    std::vector<std::string> errorMessages;
    if (!useMtsSubstepping(ir))
    {
        return errorMessages;
    }

    GMX_RELEASE_ASSERT(haveValidMtsSetup(ir), "MTS setup should be valid here");

    const ArrayRef<const MtsLevel> mtsLevels = ir.mtsLevels;
    if (ir.eI != IntegrationAlgorithm::MD)
    {
        errorMessages.push_back(
                gmx::formatString("Multiple time stepping is only supported with integrator %s",
                                  enumValueToString(IntegrationAlgorithm::MD)));
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
    if (nonbondedFactor > 1 && ir.nstlist % nonbondedFactor != 0)
    {
        errorMessages.push_back(gmx::formatString(
                "With MTS, nstlist = %d should be a multiple of the nonbonded mts-factor = %d",
                ir.nstlist,
                nonbondedFactor));
    }

    if (ir.bPull)
    {
        const int pullMtsLevel  = forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::Pull);
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

bool haveValidExactRespaSetup(const t_inputrec& ir)
{
    if (!ir.exactRespa.enabled())
    {
        return false;
    }
    if (ir.exactRespa.levelStepFactors.size() < 2 || ir.exactRespa.levelStepFactors.size() > c_maxMtsLevels)
    {
        return false;
    }

    for (int level = 1; level < exactRespaNumLevels(ir); ++level)
    {
        const int previousStepFactor = exactRespaLevelStepFactor(ir, level - 1);
        const int stepFactor         = exactRespaLevelStepFactor(ir, level);
        if (stepFactor <= previousStepFactor || stepFactor % previousStepFactor != 0)
        {
            return false;
        }
    }

    return checkExactRespaRequirements(ir).empty();
}

std::vector<std::string> checkExactRespaRequirements(const t_inputrec& ir)
{
    std::vector<std::string> errorMessages;
    if (!ir.exactRespa.enabled())
    {
        return errorMessages;
    }

    if (ir.exactRespa.levelStepFactors.size() < 2 || ir.exactRespa.levelStepFactors.size() > c_maxMtsLevels)
    {
        errorMessages.emplace_back(
                gmx::formatString("Only 2 <= exact-respa-levels <= %d is supported", c_maxMtsLevels));
        return errorMessages;
    }
    for (int level = 1; level < exactRespaNumLevels(ir); ++level)
    {
        const int previousStepFactor = exactRespaLevelStepFactor(ir, level - 1);
        const int stepFactor         = exactRespaLevelStepFactor(ir, level);
        if (stepFactor <= previousStepFactor || stepFactor % previousStepFactor != 0)
        {
            errorMessages.emplace_back(
                    "Exact r-RESPA step factors must increase monotonically and divide the next level");
            return errorMessages;
        }
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
        errorMessages.emplace_back("Exact LAMMPS-style r-RESPA currently requires vdw-type = cut-off");
    }
    if (ir.vdw_modifier != InteractionModifiers::None)
    {
        errorMessages.emplace_back("Exact LAMMPS-style r-RESPA currently requires vdw-modifier = none");
    }
    if (!(usingPmeOrEwald(ir.coulombtype) || ir.coulombtype == CoulombInteractionType::Cut))
    {
        errorMessages.emplace_back(
                "Exact LAMMPS-style r-RESPA currently requires coulombtype = PME, Ewald, or Cut-off");
    }
    if (ir.coulomb_modifier != InteractionModifiers::None)
    {
        errorMessages.emplace_back("Exact LAMMPS-style r-RESPA currently requires coulomb-modifier = none");
    }
    if (exactRespaHasPairSplitting(ir))
    {
        const real shortestPairCutoff = std::min(ir.rcoulomb, ir.rvdw);
        if (ir.exactRespa.forceLayout.outerOff > shortestPairCutoff)
        {
            errorMessages.emplace_back(gmx::formatString(
                    "Exact LAMMPS-style r-RESPA requires exact-respa-outer-off = %g to be <= min(rcoulomb, rvdw) = %g",
                    ir.exactRespa.forceLayout.outerOff,
                    shortestPairCutoff));
        }
        if (ir.exactRespa.forceLayout.hasMiddle())
        {
            if (!(ir.exactRespa.forceLayout.innerOff < ir.exactRespa.forceLayout.innerOn
                  && ir.exactRespa.forceLayout.innerOn < ir.exactRespa.forceLayout.outerOn
                  && ir.exactRespa.forceLayout.outerOn < ir.exactRespa.forceLayout.outerOff))
            {
                errorMessages.emplace_back(
                        "Exact LAMMPS-style r-RESPA middle splitting requires exact-respa-inner-off < exact-respa-inner-on < exact-respa-outer-on < exact-respa-outer-off");
            }
        }
        else if (!(ir.exactRespa.forceLayout.outerOn < ir.exactRespa.forceLayout.outerOff))
        {
            errorMessages.emplace_back(
                    "Exact LAMMPS-style r-RESPA inner/outer splitting requires exact-respa-outer-on < exact-respa-outer-off");
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
    if ((usingFullElectrostatics(ir.coulombtype) || usingLJPme(ir.vdwtype))
        && exactRespaLongrangeNonbondedLevel(ir) != exactRespaNumLevels(ir) - 1)
    {
        errorMessages.emplace_back(
                "With long-range electrostatics and/or LJ treatment, exact-respa-kspace-level should match the slowest exact-respa level");
    }

    std::optional<std::string> mesg;
    if (ir.nstcalcenergy > 0)
    {
        if ((mesg = checkExactRespaInterval(ir, "nstcalcenergy", ir.nstcalcenergy)))
        {
            errorMessages.push_back(*mesg);
        }
    }
    if ((mesg = checkExactRespaInterval(ir, "nstenergy", ir.nstenergy)))
    {
        errorMessages.push_back(*mesg);
    }
    if ((mesg = checkExactRespaInterval(ir, "nstlog", ir.nstlog)))
    {
        errorMessages.push_back(*mesg);
    }
    if ((mesg = checkExactRespaInterval(ir, "nstfout", ir.nstfout)))
    {
        errorMessages.push_back(*mesg);
    }
    if (ir.efep != FreeEnergyPerturbationType::No)
    {
        if ((mesg = checkExactRespaInterval(ir, "nstdhdl", ir.fepvals->nstdhdl)))
        {
            errorMessages.push_back(*mesg);
        }
    }

    const int nonbondedFactor = exactRespaNonbondedMtsFactor(ir);
    if (ir.nstlist % nonbondedFactor != 0)
    {
        errorMessages.push_back(gmx::formatString(
                "With exact-respa, nstlist = %d should be a multiple of the nonbonded factor = %d",
                ir.nstlist,
                nonbondedFactor));
    }
    if (ir.bPull)
    {
        const int pullStepFactor = exactRespaLevelStepFactor(ir, exactRespaPullLevel(ir));
        if (ir.pull->nstxout % pullStepFactor != 0)
        {
            errorMessages.emplace_back("pull-nstxout should be a multiple of exact-respa-factor");
        }
        if (ir.pull->nstfout % pullStepFactor != 0)
        {
            errorMessages.emplace_back("pull-nstfout should be a multiple of exact-respa-factor");
        }
    }

    return errorMessages;
}

ExactRespaBaseStepTrace exactRespaBaseStepTrace(const ExactRespaParameters& exactRespa, const int64_t baseStep)
{
    ExactRespaBaseStepTrace trace;
    if (!exactRespa.enabled())
    {
        trace.initialKickLevels = { 0 };
        trace.refreshedForceLevels = { 0 };
        trace.finalKickLevels = { 0 };
        return trace;
    }

    const int highestInitialLevel = highestActiveExactRespaLevel(exactRespa, baseStep);
    for (int level = highestInitialLevel; level >= 0; --level)
    {
        trace.initialKickLevels.push_back(level);
    }

    const int highestFinalLevel = highestActiveExactRespaLevel(exactRespa, baseStep + 1);
    for (int level = 0; level <= highestFinalLevel; ++level)
    {
        trace.refreshedForceLevels.push_back(level);
        trace.finalKickLevels.push_back(level);
    }

    return trace;
}

} // namespace gmx
