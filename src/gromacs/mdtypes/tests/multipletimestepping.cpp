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
/*! \internal \file
 * \brief
 * Tests for the MultipleTimeStepping class and stand-alone functions.
 *
 * \author berk Hess <hess@kth.se>
 * \ingroup module_mdtypes
 */
#include "gmxpre.h"

#include "gromacs/mdtypes/multipletimestepping.h"

#include <bitset>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/smalloc.h"

#include "testutils/testasserts.h"

namespace gmx
{

namespace test
{

namespace
{

//! Returns the number of parser + requirement errors for the MTS setup in \p ir
int countMtsLevelErrors(const GromppMtsOpts& mtsOpts, t_inputrec* ir)
{
    std::vector<std::string> errorMessages;
    ir->useMts        = true;
    ir->mtsMode       = mtsOpts.mode;
    ir->lammpsRespa   = mtsOpts.lammpsRespa;
    ir->mtsLevels     = setupMtsLevels(mtsOpts, &errorMessages);
    ir->exactRespa    = exactRespaParametersFromLegacyMts(ir->mtsMode, ir->mtsLevels, ir->lammpsRespa);

    if (useExactRespa(*ir) || haveValidMtsSetup(*ir))
    {
        std::vector<std::string> errorMessagesCheck = checkMtsRequirements(*ir);

        // Concatenate the two lists with error messages
        errorMessages.insert(errorMessages.end(), errorMessagesCheck.begin(), errorMessagesCheck.end());
    }

    return errorMessages.size();
}

//! brief Sets up the MTS levels in \p ir and tests whether the number of errors matches \p numExpectedErrors
void setAndCheckMtsLevels(const GromppMtsOpts& mtsOpts, t_inputrec* ir, const int numExpectedErrors)
{
    EXPECT_EQ(countMtsLevelErrors(mtsOpts, ir), numExpectedErrors);
}

} // namespace

//! Checks that only numLevels = 2 does not produce an error
TEST(MultipleTimeStepping, ChecksNumLevels)
{
    for (int numLevels : { 0, 1 })
    {
        GromppMtsOpts mtsOpts;
        mtsOpts.numLevels = numLevels;

        t_inputrec ir;

        EXPECT_GT(countMtsLevelErrors(mtsOpts, &ir), 0);
    }

    {
        GromppMtsOpts mtsOpts;
        mtsOpts.numLevels    = 2;
        mtsOpts.levelForces  = { "nonbonded" };
        mtsOpts.levelFactors = { 2 };

        t_inputrec ir;

        setAndCheckMtsLevels(mtsOpts, &ir, 0);
    }

    {
        GromppMtsOpts mtsOpts;
        mtsOpts.numLevels    = 3;
        mtsOpts.levelForces  = { "pair", "nonbonded" };
        mtsOpts.levelFactors = { 2, 4 };

        t_inputrec ir;

        setAndCheckMtsLevels(mtsOpts, &ir, 0);
    }

    {
        GromppMtsOpts mtsOpts;
        mtsOpts.numLevels = 4;

        t_inputrec ir;

        EXPECT_GT(countMtsLevelErrors(mtsOpts, &ir), 0);
    }
}

//! Test that each force group works
TEST(MultipleTimeStepping, SelectsForceGroups)
{
    for (int forceGroupIndex = 0; forceGroupIndex < static_cast<int>(MtsForceGroups::Count);
         forceGroupIndex++)
    {
        const MtsForceGroups forceGroup = static_cast<MtsForceGroups>(forceGroupIndex);
        SCOPED_TRACE("Testing force group " + mtsForceGroupNames[forceGroup]);

        GromppMtsOpts mtsOpts;
        mtsOpts.numLevels    = 2;
        mtsOpts.levelForces  = { mtsForceGroupNames[forceGroup] };
        mtsOpts.levelFactors = { 2 };

        t_inputrec ir;

        setAndCheckMtsLevels(mtsOpts, &ir, 0);

        EXPECT_EQ(ir.mtsLevels[1].forceGroups.count(), 1);
        EXPECT_EQ(ir.mtsLevels[1].forceGroups[forceGroupIndex], true);
    }
}

//! Checks that factor is checked
TEST(MultipleTimeStepping, ChecksStepFactor)
{
    for (int stepFactor = 0; stepFactor <= 3; stepFactor++)
    {
        GromppMtsOpts mtsOpts;
        mtsOpts.numLevels    = 2;
        mtsOpts.levelForces  = { "nonbonded" };
        mtsOpts.levelFactors = { stepFactor };

        t_inputrec ir;

        setAndCheckMtsLevels(mtsOpts, &ir, stepFactor < 2 ? 1 : 0);
    }
}

namespace
{

GromppMtsOpts simpleMtsOpts()
{
    GromppMtsOpts mtsOpts;
    mtsOpts.numLevels    = 2;
    mtsOpts.levelForces  = { "nonbonded" };
    mtsOpts.levelFactors = { 4 };

    return mtsOpts;
}

GromppMtsOpts exactLammpsRespaOpts()
{
    GromppMtsOpts mtsOpts;
    mtsOpts.mode         = MtsMode::LammpsRespa;
    mtsOpts.numLevels    = 3;
    mtsOpts.levelForces  = { "", "" };
    mtsOpts.levelFactors = { 2, 4 };
    mtsOpts.lammpsRespa.enabled       = true;
    mtsOpts.lammpsRespa.bondLevel     = 0;
    mtsOpts.lammpsRespa.angleLevel    = 1;
    mtsOpts.lammpsRespa.dihedralLevel = 1;
    mtsOpts.lammpsRespa.improperLevel = 1;
    mtsOpts.lammpsRespa.pair14Level   = 1;
    mtsOpts.lammpsRespa.kspaceLevel   = 2;
    mtsOpts.lammpsRespa.innerLevel    = 0;
    mtsOpts.lammpsRespa.middleLevel   = 1;
    mtsOpts.lammpsRespa.outerLevel    = 2;
    mtsOpts.lammpsRespa.innerOff      = 0.30;
    mtsOpts.lammpsRespa.innerOn       = 0.45;
    mtsOpts.lammpsRespa.outerOn       = 0.60;
    mtsOpts.lammpsRespa.outerOff      = 0.80;

    return mtsOpts;
}

void configureExactLammpsRespaInputRecord(t_inputrec* ir)
{
    ir->eI               = IntegrationAlgorithm::MD;
    ir->coulombtype      = CoulombInteractionType::Pme;
    ir->coulomb_modifier = InteractionModifiers::None;
    ir->vdwtype          = VanDerWaalsType::Cut;
    ir->vdw_modifier     = InteractionModifiers::None;
    ir->rcoulomb         = 0.9;
    ir->rvdw             = 0.9;
}

} // namespace

TEST(MultipleTimeStepping, ChecksPmeIsAtLastLevel)
{
    const GromppMtsOpts mtsOpts = simpleMtsOpts();

    t_inputrec ir;
    ir.coulombtype = CoulombInteractionType::Pme;

    setAndCheckMtsLevels(mtsOpts, &ir, 1);
}

TEST(MultipleTimeStepping, AcceptsVelocityVerletOnlyForExactLammpsRespa)
{
    {
        const GromppMtsOpts mtsOpts = exactLammpsRespaOpts();

        t_inputrec ir;
        configureExactLammpsRespaInputRecord(&ir);
        ir.eI        = IntegrationAlgorithm::VV;
        ir.useMts    = true;
        ir.mtsMode   = mtsOpts.mode;
        ir.lammpsRespa = mtsOpts.lammpsRespa;
        ir.mtsLevels = setupMtsLevels(mtsOpts, nullptr);
        ir.exactRespa = exactRespaParametersFromLegacyMts(ir.mtsMode, ir.mtsLevels, ir.lammpsRespa);

        EXPECT_TRUE(checkMtsRequirements(ir).empty());
    }

    {
        const GromppMtsOpts mtsOpts = simpleMtsOpts();

        t_inputrec ir;
        ir.eI        = IntegrationAlgorithm::VV;
        ir.useMts    = true;
        ir.mtsMode   = mtsOpts.mode;
        ir.mtsLevels = setupMtsLevels(mtsOpts, nullptr);

        EXPECT_THAT(checkMtsRequirements(ir), ::testing::Not(::testing::IsEmpty()));
    }
}

//! Test fixture base for parametrizing interval tests
using MtsIntervalTestParams = std::tuple<std::string, int>;
class MtsIntervalTest : public ::testing::Test, public ::testing::WithParamInterface<MtsIntervalTestParams>
{
public:
    MtsIntervalTest()
    {
        const auto  params        = GetParam();
        const auto& parameterName = std::get<0>(params);
        const auto  interval      = std::get<1>(params);
        numExpectedErrors_        = (interval == 4 ? 0 : 1);

        if (parameterName == "nstcalcenergy")
        {
            ir_.nstcalcenergy = interval;
        }
        else if (parameterName == "nstenergy")
        {
            ir_.nstenergy = interval;
        }
        else if (parameterName == "nstfout")
        {
            ir_.nstfout = interval;
        }
        else if (parameterName == "nstlist")
        {
            ir_.nstlist = interval;
        }
        else if (parameterName == "nstdhdl")
        {
            ir_.efep             = FreeEnergyPerturbationType::Yes;
            ir_.fepvals->nstdhdl = interval;
        }
        else

        {
            GMX_RELEASE_ASSERT(false, "unknown parameter name");
        }
    }

    t_inputrec ir_;
    int        numExpectedErrors_;
};

TEST_P(MtsIntervalTest, Works)
{
    const GromppMtsOpts mtsOpts = simpleMtsOpts();

    setAndCheckMtsLevels(mtsOpts, &ir_, numExpectedErrors_);
}

INSTANTIATE_TEST_SUITE_P(
        ChecksStepInterval,
        MtsIntervalTest,
        ::testing::Combine(
                ::testing::Values("nstcalcenergy", "nstenergy", "nstfout", "nstlist", "nstdhdl"),
                ::testing::Values(3, 4, 5)));

// Check that correct input does not produce errors
TEST(MultipleTimeStepping, ChecksIntegrator)
{
    const GromppMtsOpts mtsOpts = simpleMtsOpts();

    t_inputrec ir;
    ir.eI = IntegrationAlgorithm::BD;

    setAndCheckMtsLevels(mtsOpts, &ir, 1);
}

TEST(MultipleTimeStepping, ParsesThreeLevelSchedule)
{
    GromppMtsOpts mtsOpts;
    mtsOpts.numLevels    = 3;
    mtsOpts.levelForces  = { "pair dihedral angle", "nonbonded longrange-nonbonded" };
    mtsOpts.levelFactors = { 2, 4 };

    t_inputrec ir;

    setAndCheckMtsLevels(mtsOpts, &ir, 0);
    ASSERT_EQ(ir.mtsLevels.size(), 3);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::Pair), 1);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::Dihedral), 1);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::Angle), 1);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::Nonbonded), 2);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::LongrangeNonbonded), 2);
}

TEST(MultipleTimeStepping, RejectsNonNestedThreeLevelSchedule)
{
    GromppMtsOpts mtsOpts;
    mtsOpts.numLevels    = 3;
    mtsOpts.levelForces  = { "pair", "longrange-nonbonded" };
    mtsOpts.levelFactors = { 3, 4 };

    t_inputrec ir;

    setAndCheckMtsLevels(mtsOpts, &ir, 1);
}

TEST(MultipleTimeStepping, ReportsHighestActiveLevelForNestedSchedule)
{
    GromppMtsOpts mtsOpts;
    mtsOpts.numLevels    = 3;
    mtsOpts.levelForces  = { "pair dihedral angle", "nonbonded longrange-nonbonded" };
    mtsOpts.levelFactors = { 2, 4 };

    t_inputrec ir;

    setAndCheckMtsLevels(mtsOpts, &ir, 0);
    EXPECT_EQ(highestActiveMtsLevel(ir.mtsLevels, 0), 2);
    EXPECT_EQ(highestActiveMtsLevel(ir.mtsLevels, 1), 0);
    EXPECT_EQ(highestActiveMtsLevel(ir.mtsLevels, 2), 1);
    EXPECT_EQ(highestActiveMtsLevel(ir.mtsLevels, 3), 0);
    EXPECT_EQ(highestActiveMtsLevel(ir.mtsLevels, 4), 2);
}

TEST(MultipleTimeStepping, RejectsTwoLevelExactLammpsRespaSchedule)
{
    GromppMtsOpts mtsOpts;
    mtsOpts.mode         = MtsMode::LammpsRespa;
    mtsOpts.numLevels    = 2;
    mtsOpts.levelForces  = { "" };
    mtsOpts.levelFactors = { 2 };
    mtsOpts.lammpsRespa.enabled       = true;
    mtsOpts.lammpsRespa.bondLevel     = 0;
    mtsOpts.lammpsRespa.angleLevel    = 0;
    mtsOpts.lammpsRespa.dihedralLevel = 0;
    mtsOpts.lammpsRespa.improperLevel = 0;
    mtsOpts.lammpsRespa.pair14Level   = 0;
    mtsOpts.lammpsRespa.kspaceLevel   = 1;
    mtsOpts.lammpsRespa.innerLevel    = -1;
    mtsOpts.lammpsRespa.middleLevel   = -1;
    mtsOpts.lammpsRespa.outerLevel    = -1;
    mtsOpts.lammpsRespa.pairLevel     = 1;

    t_inputrec ir;
    configureExactLammpsRespaInputRecord(&ir);
    setAndCheckMtsLevels(mtsOpts, &ir, 1);
}

TEST(MultipleTimeStepping, ParsesExactLammpsRespaSchedule)
{
    const GromppMtsOpts mtsOpts = exactLammpsRespaOpts();

    t_inputrec ir;
    configureExactLammpsRespaInputRecord(&ir);

    setAndCheckMtsLevels(mtsOpts, &ir, 0);
    ir.exactRespa = exactRespaParametersFromLegacyMts(ir.mtsMode, ir.mtsLevels, ir.lammpsRespa);
    ASSERT_EQ(ir.mtsLevels.size(), 3);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::Bond), 0);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::Angle), 1);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::Dihedral), 1);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::Improper), 1);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::Pair), 1);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::LongrangeNonbonded), 2);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::NonbondedInner), 0);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::NonbondedMiddle), 1);
    EXPECT_EQ(forceGroupMtsLevel(ir.mtsLevels, MtsForceGroups::NonbondedOuter), 2);
    EXPECT_EQ(nonbondedRespaContributionMtsLevel(ir, MtsNonbondedRespaContribution::Inner), 0);
    EXPECT_EQ(nonbondedRespaContributionMtsLevel(ir, MtsNonbondedRespaContribution::Middle), 1);
    EXPECT_EQ(nonbondedRespaContributionMtsLevel(ir, MtsNonbondedRespaContribution::Outer), 2);
    EXPECT_EQ(nonbondedMtsFactor(ir), 4);
}

TEST(MultipleTimeStepping, ComputesExactLammpsRespaPairSplitWeightsAcrossTransitionRegions)
{
    const GromppMtsOpts mtsOpts = exactLammpsRespaOpts();

    t_inputrec ir;
    configureExactLammpsRespaInputRecord(&ir);

    setAndCheckMtsLevels(mtsOpts, &ir, 0);
    constexpr double splitWeightTolerance = 1e-6;

    const auto innerOnlyWeights = computeLammpsRespaPairSplitWeights(ir, 0.25_real);
    EXPECT_NEAR(innerOnlyWeights.inner, 1.0, splitWeightTolerance);
    EXPECT_NEAR(innerOnlyWeights.middle, 0.0, splitWeightTolerance);
    EXPECT_NEAR(innerOnlyWeights.outer, 0.0, splitWeightTolerance);

    const auto innerMiddleBlend = computeLammpsRespaPairSplitWeights(ir, 0.375_real);
    EXPECT_NEAR(innerMiddleBlend.inner, 0.5, splitWeightTolerance);
    EXPECT_NEAR(innerMiddleBlend.middle, 0.5, splitWeightTolerance);
    EXPECT_NEAR(innerMiddleBlend.outer, 0.0, splitWeightTolerance);

    const auto middleOuterBlend = computeLammpsRespaPairSplitWeights(ir, 0.70_real);
    EXPECT_NEAR(middleOuterBlend.inner, 0.0, splitWeightTolerance);
    EXPECT_NEAR(middleOuterBlend.middle, 0.5, splitWeightTolerance);
    EXPECT_NEAR(middleOuterBlend.outer, 0.5, splitWeightTolerance);

    const auto outerOnlyWeights = computeLammpsRespaPairSplitWeights(ir, 0.90_real);
    EXPECT_NEAR(outerOnlyWeights.inner, 0.0, splitWeightTolerance);
    EXPECT_NEAR(outerOnlyWeights.middle, 0.0, splitWeightTolerance);
    EXPECT_NEAR(outerOnlyWeights.outer, 1.0, splitWeightTolerance);
}

TEST(MultipleTimeStepping, ReportsActiveExactLammpsRespaNonbondedOutputSinks)
{
    const GromppMtsOpts mtsOpts = exactLammpsRespaOpts();

    t_inputrec ir;
    configureExactLammpsRespaInputRecord(&ir);

    setAndCheckMtsLevels(mtsOpts, &ir, 0);

    const auto sinks = activeLammpsRespaNonbondedOutputSinks(ir, 2, true, true);
    ASSERT_EQ(sinks.size(), 3);

    EXPECT_EQ(sinks[0].contribution, MtsNonbondedRespaContribution::Inner);
    EXPECT_EQ(sinks[0].mtsLevel, 0);
    EXPECT_EQ(sinks[0].sinkKind, LammpsRespaNonbondedOutputSinkKind::ShiftForce);
    EXPECT_FALSE(sinks[0].accumulateEnergy);

    EXPECT_EQ(sinks[1].contribution, MtsNonbondedRespaContribution::Middle);
    EXPECT_EQ(sinks[1].mtsLevel, 1);
    EXPECT_EQ(sinks[1].sinkKind, LammpsRespaNonbondedOutputSinkKind::ShiftForce);
    EXPECT_FALSE(sinks[1].accumulateEnergy);

    EXPECT_EQ(sinks[2].contribution, MtsNonbondedRespaContribution::Outer);
    EXPECT_EQ(sinks[2].mtsLevel, 2);
    EXPECT_EQ(sinks[2].sinkKind, LammpsRespaNonbondedOutputSinkKind::ForceWithVirial);
    EXPECT_TRUE(sinks[2].accumulateEnergy);
}

TEST(MultipleTimeStepping, FiltersExactLammpsRespaNonbondedOutputSinksByActiveLevelAndVirialNeed)
{
    const GromppMtsOpts mtsOpts = exactLammpsRespaOpts();

    t_inputrec ir;
    configureExactLammpsRespaInputRecord(&ir);

    setAndCheckMtsLevels(mtsOpts, &ir, 0);

    const auto sinks = activeLammpsRespaNonbondedOutputSinks(ir, 1, false, false);
    ASSERT_EQ(sinks.size(), 2);

    EXPECT_EQ(sinks[0].contribution, MtsNonbondedRespaContribution::Inner);
    EXPECT_EQ(sinks[0].mtsLevel, 0);
    EXPECT_EQ(sinks[0].sinkKind, LammpsRespaNonbondedOutputSinkKind::ShiftForce);
    EXPECT_FALSE(sinks[0].accumulateEnergy);

    EXPECT_EQ(sinks[1].contribution, MtsNonbondedRespaContribution::Middle);
    EXPECT_EQ(sinks[1].mtsLevel, 1);
    EXPECT_EQ(sinks[1].sinkKind, LammpsRespaNonbondedOutputSinkKind::ShiftForce);
    EXPECT_FALSE(sinks[1].accumulateEnergy);
}

TEST(MultipleTimeStepping, DerivesPerLaunchWorkloadForExactNonbondedContribution)
{
    StepWorkload stepWork;
    stepWork.computeForces             = true;
    stepWork.computeNonbondedForces    = true;
    stepWork.computeEnergy             = true;
    stepWork.computeVirial             = true;
    stepWork.computeSlowForces         = true;
    stepWork.highestActiveMtsLevel     = 2;

    const auto innerWork = stepWork.withExactNonbondedContribution(MtsNonbondedRespaContribution::Inner);
    EXPECT_EQ(innerWork.nonbondedRespaContribution, MtsNonbondedRespaContribution::Inner);
    EXPECT_TRUE(innerWork.computeForces);
    EXPECT_TRUE(innerWork.computeNonbondedForces);
    EXPECT_FALSE(innerWork.computeEnergy);
    EXPECT_FALSE(innerWork.computeVirial);

    const auto middleWork = stepWork.withExactNonbondedContribution(MtsNonbondedRespaContribution::Middle);
    EXPECT_EQ(middleWork.nonbondedRespaContribution, MtsNonbondedRespaContribution::Middle);
    EXPECT_FALSE(middleWork.computeEnergy);
    EXPECT_FALSE(middleWork.computeVirial);

    const auto outerWork = stepWork.withExactNonbondedContribution(MtsNonbondedRespaContribution::Outer);
    EXPECT_EQ(outerWork.nonbondedRespaContribution, MtsNonbondedRespaContribution::Outer);
    EXPECT_TRUE(outerWork.computeEnergy);
    EXPECT_TRUE(outerWork.computeVirial);

    const auto fullWork = stepWork.withExactNonbondedContribution(MtsNonbondedRespaContribution::Full);
    EXPECT_EQ(fullWork.nonbondedRespaContribution, MtsNonbondedRespaContribution::Full);
    EXPECT_TRUE(fullWork.computeEnergy);
    EXPECT_TRUE(fullWork.computeVirial);
}

TEST(MultipleTimeStepping, RejectsLegacyForceListsInExactLammpsRespaMode)
{
    GromppMtsOpts mtsOpts = exactLammpsRespaOpts();
    mtsOpts.levelForces   = { "pair", "longrange-nonbonded" };

    t_inputrec ir;
    configureExactLammpsRespaInputRecord(&ir);

    setAndCheckMtsLevels(mtsOpts, &ir, 2);
}

TEST(MultipleTimeStepping, RejectsExactLammpsRespaWhenOuterCutoffExceedsPairCutoff)
{
    const GromppMtsOpts mtsOpts = exactLammpsRespaOpts();

    t_inputrec ir;
    configureExactLammpsRespaInputRecord(&ir);
    ir.rcoulomb = 0.7;

    setAndCheckMtsLevels(mtsOpts, &ir, 1);
}

} // namespace test
} // namespace gmx
