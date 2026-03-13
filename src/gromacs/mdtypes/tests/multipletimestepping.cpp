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

    if (haveValidMtsSetup(*ir))
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

struct ScalarRespaState
{
    double x = 0.0;
    double v = 0.0;
};

double levelDt(const std::vector<MtsLevel>& mtsLevels, const int mtsLevel, const double baseDt)
{
    return baseDt * mtsLevels[mtsLevel].stepFactor;
}

double springForce(const double springConstant, const double position)
{
    return -springConstant * position;
}

void halfKick(ScalarRespaState* state, const double force, const double dt)
{
    state->v += 0.5 * dt * force;
}

void drift(ScalarRespaState* state, const double dt)
{
    state->x += dt * state->v;
}

void integrateReferenceRecursively(const std::vector<MtsLevel>& mtsLevels,
                                   const std::vector<double>&   levelDt,
                                   const std::vector<double>&   forces,
                                   const int                    level,
                                   ScalarRespaState*            state)
{
    const int loops =
            (level + 1 == static_cast<int>(mtsLevels.size()))
                    ? 1
                    : mtsLevels[level + 1].stepFactor / mtsLevels[level].stepFactor;
    for (int iloop = 0; iloop < loops; ++iloop)
    {
        halfKick(state, forces[level], levelDt[level]);
        if (level == 0)
        {
            drift(state, levelDt[0]);
        }
        else
        {
            integrateReferenceRecursively(mtsLevels, levelDt, forces, level - 1, state);
        }
        halfKick(state, forces[level], levelDt[level]);
    }
}

ScalarRespaState integrateWithFlattenedTrace(const std::vector<MtsLevel>& mtsLevels,
                                             const std::vector<double>&   forces,
                                             const double                 baseDt,
                                             const int                    numBaseSteps)
{
    ScalarRespaState state;
    for (int baseStep = 0; baseStep < numBaseSteps; ++baseStep)
    {
        const LammpsRespaBaseStepTrace trace = lammpsRespaBaseStepTrace(mtsLevels, baseStep);
        for (const int level : trace.initialKickLevels)
        {
            halfKick(&state, forces[level], levelDt(mtsLevels, level, baseDt));
        }
        drift(&state, baseDt);
        for (const int level : trace.finalKickLevels)
        {
            halfKick(&state, forces[level], levelDt(mtsLevels, level, baseDt));
        }
    }
    return state;
}

void integrateReferenceRecursivelyWithDynamicForces(const std::vector<MtsLevel>& mtsLevels,
                                                    const std::vector<double>&   levelDt,
                                                    const std::vector<double>&   springConstants,
                                                    const int                    level,
                                                    ScalarRespaState*            state,
                                                    std::vector<double>*         forces)
{
    const int loops =
            (level + 1 == static_cast<int>(mtsLevels.size()))
                    ? 1
                    : mtsLevels[level + 1].stepFactor / mtsLevels[level].stepFactor;
    for (int iloop = 0; iloop < loops; ++iloop)
    {
        halfKick(state, (*forces)[level], levelDt[level]);
        if (level == 0)
        {
            drift(state, levelDt[0]);
            (*forces)[0] = springForce(springConstants[0], state->x);
        }
        else
        {
            integrateReferenceRecursivelyWithDynamicForces(
                    mtsLevels, levelDt, springConstants, level - 1, state, forces);
            (*forces)[level] = springForce(springConstants[level], state->x);
        }
        halfKick(state, (*forces)[level], levelDt[level]);
    }
}

ScalarRespaState integrateRecursivelyWithDynamicForces(const std::vector<MtsLevel>& mtsLevels,
                                                       const std::vector<double>&   springConstants,
                                                       const double                 baseDt,
                                                       const int                    numBaseSteps)
{
    std::vector<double> levelDtByIndex(mtsLevels.size());
    std::vector<double> forces(mtsLevels.size());
    ScalarRespaState    state{ 1.0, 0.0 };
    for (int level = 0; level < static_cast<int>(mtsLevels.size()); ++level)
    {
        levelDtByIndex[level] = levelDt(mtsLevels, level, baseDt);
        forces[level]         = springForce(springConstants[level], state.x);
    }

    const int outerLoops = numBaseSteps / mtsLevels.back().stepFactor;
    for (int i = 0; i < outerLoops; ++i)
    {
        integrateReferenceRecursivelyWithDynamicForces(
                mtsLevels, levelDtByIndex, springConstants, static_cast<int>(mtsLevels.size()) - 1, &state, &forces);
    }
    return state;
}

ScalarRespaState integrateWithFlattenedTraceAndDynamicForces(const std::vector<MtsLevel>& mtsLevels,
                                                             const std::vector<double>&   springConstants,
                                                             const double                 baseDt,
                                                             const int                    numBaseSteps)
{
    std::vector<double> forces(mtsLevels.size());
    ScalarRespaState    state{ 1.0, 0.0 };
    for (int level = 0; level < static_cast<int>(mtsLevels.size()); ++level)
    {
        forces[level] = springForce(springConstants[level], state.x);
    }

    for (int baseStep = 0; baseStep < numBaseSteps; ++baseStep)
    {
        const LammpsRespaBaseStepTrace trace = lammpsRespaBaseStepTrace(mtsLevels, baseStep);
        for (const int level : trace.initialKickLevels)
        {
            halfKick(&state, forces[level], levelDt(mtsLevels, level, baseDt));
        }
        drift(&state, baseDt);
        for (const int level : trace.refreshedForceLevels)
        {
            forces[level] = springForce(springConstants[level], state.x);
        }
        for (const int level : trace.finalKickLevels)
        {
            halfKick(&state, forces[level], levelDt(mtsLevels, level, baseDt));
        }
    }
    return state;
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

TEST(MultipleTimeStepping, FlattenedBaseStepTraceMatchesRecursiveLammpsReference)
{
    const GromppMtsOpts mtsOpts = exactLammpsRespaOpts();

    t_inputrec ir;
    configureExactLammpsRespaInputRecord(&ir);
    ir.useMts      = true;
    ir.mtsMode     = mtsOpts.mode;
    ir.lammpsRespa = mtsOpts.lammpsRespa;
    ir.mtsLevels   = setupMtsLevels(mtsOpts, nullptr);

    const std::vector<double> forces = { 1.25, -0.5, 3.0 };
    const double              baseDt = 0.0025;
    const int numBaseSteps = ir.mtsLevels.back().stepFactor;

    std::vector<double> recursiveLevelDt(ir.mtsLevels.size());
    for (int level = 0; level < static_cast<int>(ir.mtsLevels.size()); ++level)
    {
        recursiveLevelDt[level] = levelDt(ir.mtsLevels, level, baseDt);
    }

    ScalarRespaState recursiveState;
    integrateReferenceRecursively(ir.mtsLevels,
                                  recursiveLevelDt,
                                  forces,
                                  static_cast<int>(ir.mtsLevels.size()) - 1,
                                  &recursiveState);
    const ScalarRespaState flattenedState =
            integrateWithFlattenedTrace(ir.mtsLevels, forces, baseDt, numBaseSteps);

    EXPECT_NEAR(flattenedState.x, recursiveState.x, 1e-12);
    EXPECT_NEAR(flattenedState.v, recursiveState.v, 1e-12);
}

TEST(MultipleTimeStepping, FlattenedBaseStepTraceMatchesRecursiveLammpsReferenceWithDynamicForces)
{
    const GromppMtsOpts mtsOpts = exactLammpsRespaOpts();

    t_inputrec ir;
    configureExactLammpsRespaInputRecord(&ir);
    ir.useMts      = true;
    ir.mtsMode     = mtsOpts.mode;
    ir.lammpsRespa = mtsOpts.lammpsRespa;
    ir.mtsLevels   = setupMtsLevels(mtsOpts, nullptr);

    const std::vector<double> springConstants = { 1.25, 0.5, 0.125 };
    const double              baseDt          = 0.0025;
    const int                 numBaseSteps    = ir.mtsLevels.back().stepFactor * 5;

    const ScalarRespaState recursiveState =
            integrateRecursivelyWithDynamicForces(ir.mtsLevels, springConstants, baseDt, numBaseSteps);
    const ScalarRespaState flattenedState =
            integrateWithFlattenedTraceAndDynamicForces(ir.mtsLevels, springConstants, baseDt, numBaseSteps);

    EXPECT_NEAR(flattenedState.x, recursiveState.x, 1e-12);
    EXPECT_NEAR(flattenedState.v, recursiveState.v, 1e-12);
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

TEST(MultipleTimeStepping, ReportsLammpsRespaBaseStepTraceForThreeLevelSchedule)
{
    const auto exactMtsOpts = exactLammpsRespaOpts();

    t_inputrec ir;
    configureExactLammpsRespaInputRecord(&ir);
    setAndCheckMtsLevels(exactMtsOpts, &ir, 0);

    const auto trace0 = lammpsRespaBaseStepTrace(ir.mtsLevels, 0);
    EXPECT_THAT(trace0.initialKickLevels, ::testing::ElementsAre(2, 1, 0));
    EXPECT_THAT(trace0.refreshedForceLevels, ::testing::ElementsAre(0));
    EXPECT_THAT(trace0.finalKickLevels, ::testing::ElementsAre(0));

    const auto trace1 = lammpsRespaBaseStepTrace(ir.mtsLevels, 1);
    EXPECT_THAT(trace1.initialKickLevels, ::testing::ElementsAre(0));
    EXPECT_THAT(trace1.refreshedForceLevels, ::testing::ElementsAre(0, 1));
    EXPECT_THAT(trace1.finalKickLevels, ::testing::ElementsAre(0, 1));

    const auto trace2 = lammpsRespaBaseStepTrace(ir.mtsLevels, 2);
    EXPECT_THAT(trace2.initialKickLevels, ::testing::ElementsAre(1, 0));
    EXPECT_THAT(trace2.refreshedForceLevels, ::testing::ElementsAre(0));
    EXPECT_THAT(trace2.finalKickLevels, ::testing::ElementsAre(0));

    const auto trace3 = lammpsRespaBaseStepTrace(ir.mtsLevels, 3);
    EXPECT_THAT(trace3.initialKickLevels, ::testing::ElementsAre(0));
    EXPECT_THAT(trace3.refreshedForceLevels, ::testing::ElementsAre(0, 1, 2));
    EXPECT_THAT(trace3.finalKickLevels, ::testing::ElementsAre(0, 1, 2));
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
