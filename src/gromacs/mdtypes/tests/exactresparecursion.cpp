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
 */
/*! \internal \file
 * \brief Dedicated exact-r-RESPA recursion semantics tests.
 *
 * These tests intentionally avoid the legacy multipletimestepping.cpp test
 * file so standalone exact-r-RESPA coverage does not depend on MTS-only
 * fixtures.
 */
#include "gmxpre.h"

#include "gromacs/mdtypes/exactrespaschedule.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/taskassignment/decidesimulationworkload.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/mdlib/force_flags.h"

namespace gmx
{
namespace test
{
namespace
{

ExactRespaParameters threeLevelExactRespa()
{
    ExactRespaParameters exactRespa;
    exactRespa.levelStepFactors      = { 1, 2, 4 };
    exactRespa.forceLayout.enabled   = true;
    exactRespa.forceLayout.bondLevel = 0;
    exactRespa.forceLayout.angleLevel = 1;
    exactRespa.forceLayout.dihedralLevel = 1;
    exactRespa.forceLayout.improperLevel = 1;
    exactRespa.forceLayout.pair14Level = 1;
    exactRespa.forceLayout.pairLevel = 1;
    exactRespa.forceLayout.kspaceLevel = 2;
    exactRespa.forceLayout.innerLevel = 0;
    exactRespa.forceLayout.middleLevel = 1;
    exactRespa.forceLayout.outerLevel = 2;
    exactRespa.forceLayout.innerOff = 0.30;
    exactRespa.forceLayout.innerOn = 0.45;
    exactRespa.forceLayout.outerOn = 0.60;
    exactRespa.forceLayout.outerOff = 0.80;
    return exactRespa;
}

ExactRespaParameters nonBinaryThreeLevelExactRespa()
{
    ExactRespaParameters exactRespa = threeLevelExactRespa();
    exactRespa.levelStepFactors     = { 1, 3, 6 };
    return exactRespa;
}

void configureExactRespaInputRecord(t_inputrec* ir)
{
    ir->eI               = IntegrationAlgorithm::MD;
    ir->coulombtype      = CoulombInteractionType::Pme;
    ir->coulomb_modifier = InteractionModifiers::None;
    ir->vdwtype          = VanDerWaalsType::Cut;
    ir->vdw_modifier     = InteractionModifiers::None;
    ir->rcoulomb         = 0.9;
    ir->rvdw             = 0.9;
}

enum class RespaEventType
{
    InitialKick,
    Drift,
    RefreshForce,
    FinalKick
};

struct RespaEvent
{
    RespaEventType type;
    int            level;
};

int loopCount(const ExactRespaParameters& exactRespa, const int level)
{
    return (level + 1 == static_cast<int>(exactRespa.levelStepFactors.size()))
                   ? 1
                   : exactRespa.levelStepFactors[level + 1] / exactRespa.levelStepFactors[level];
}

// Mirrors the scalar event order in LAMMPS `Respa::recurse` together with
// `FixNVE::initial_integrate_respa` and `FixNVE::final_integrate_respa`.
void appendLammpsReferenceEvents(const ExactRespaParameters& exactRespa,
                                 const int                  level,
                                 std::vector<RespaEvent>*   events)
{
    ASSERT_NE(events, nullptr);

    for (int iloop = 0; iloop < loopCount(exactRespa, level); ++iloop)
    {
        events->push_back({ RespaEventType::InitialKick, level });
        if (level == 0)
        {
            events->push_back({ RespaEventType::Drift, 0 });
        }
        else
        {
            appendLammpsReferenceEvents(exactRespa, level - 1, events);
        }
        events->push_back({ RespaEventType::RefreshForce, level });
        events->push_back({ RespaEventType::FinalKick, level });
    }
}

std::vector<RespaEvent> flattenedExactRespaEvents(const ExactRespaParameters& exactRespa, const int numBaseSteps)
{
    std::vector<RespaEvent> events;
    for (int baseStep = 0; baseStep < numBaseSteps; ++baseStep)
    {
        const ExactRespaBaseStepTrace trace = exactRespaBaseStepTrace(exactRespa, baseStep);
        for (const int level : trace.initialKickLevels)
        {
            events.push_back({ RespaEventType::InitialKick, level });
        }
        events.push_back({ RespaEventType::Drift, 0 });
        for (const int level : trace.finalKickLevels)
        {
            events.push_back({ RespaEventType::RefreshForce, level });
            events.push_back({ RespaEventType::FinalKick, level });
        }
    }
    return events;
}

std::string describeEvent(const RespaEvent& event)
{
    switch (event.type)
    {
        case RespaEventType::InitialKick: return "initial-kick(L" + std::to_string(event.level) + ")";
        case RespaEventType::Drift: return "drift";
        case RespaEventType::RefreshForce: return "refresh-force(L" + std::to_string(event.level) + ")";
        case RespaEventType::FinalKick: return "final-kick(L" + std::to_string(event.level) + ")";
    }

    return "unknown";
}

std::vector<std::string> describeEvents(const std::vector<RespaEvent>& events)
{
    std::vector<std::string> labels;
    labels.reserve(events.size());
    for (const RespaEvent& event : events)
    {
        labels.push_back(describeEvent(event));
    }
    return labels;
}

std::vector<ExactRespaBaseStepTrace> referenceBaseStepTraces(const std::vector<RespaEvent>& events)
{
    std::vector<ExactRespaBaseStepTrace> traces;
    size_t                               eventIndex = 0;

    while (eventIndex < events.size())
    {
        ExactRespaBaseStepTrace trace;
        while (eventIndex < events.size() && events[eventIndex].type == RespaEventType::InitialKick)
        {
            trace.initialKickLevels.push_back(events[eventIndex].level);
            ++eventIndex;
        }

        if (eventIndex >= events.size())
        {
            ADD_FAILURE() << "Reference event stream ended before drift";
            return {};
        }
        if (events[eventIndex].type != RespaEventType::Drift)
        {
            ADD_FAILURE() << "Expected drift event, got " << describeEvent(events[eventIndex]);
            return {};
        }
        ++eventIndex;

        while (eventIndex < events.size() && events[eventIndex].type == RespaEventType::RefreshForce)
        {
            trace.refreshedForceLevels.push_back(events[eventIndex].level);
            ++eventIndex;
            if (eventIndex >= events.size())
            {
                ADD_FAILURE() << "Reference event stream ended before final kick";
                return {};
            }
            if (events[eventIndex].type != RespaEventType::FinalKick)
            {
                ADD_FAILURE() << "Expected final kick event, got " << describeEvent(events[eventIndex]);
                return {};
            }
            trace.finalKickLevels.push_back(events[eventIndex].level);
            ++eventIndex;
        }

        traces.push_back(std::move(trace));
    }

    return traces;
}

std::vector<ExactRespaBaseStepTrace> referenceBaseStepTraces(const ExactRespaParameters& exactRespa,
                                                             const int                   numOuterCycles)
{
    std::vector<RespaEvent> referenceEvents;
    const int highestLevel = static_cast<int>(exactRespa.levelStepFactors.size()) - 1;
    for (int outerCycle = 0; outerCycle < numOuterCycles; ++outerCycle)
    {
        appendLammpsReferenceEvents(exactRespa, highestLevel, &referenceEvents);
    }
    return referenceBaseStepTraces(referenceEvents);
}

std::string describeTrace(const ExactRespaBaseStepTrace& trace)
{
    auto describeLevels = [](const std::vector<int>& levels)
    {
        std::string result = "[";
        for (size_t i = 0; i < levels.size(); ++i)
        {
            if (i > 0)
            {
                result += ",";
            }
            result += std::to_string(levels[i]);
        }
        result += "]";
        return result;
    };

    return "initial=" + describeLevels(trace.initialKickLevels) + " refresh="
           + describeLevels(trace.refreshedForceLevels) + " final=" + describeLevels(trace.finalKickLevels);
}

} // namespace

TEST(ExactRespaStandalone, SelectsWithoutUseMts)
{
    t_inputrec ir;
    ir.exactRespa.levelStepFactors = { 1, 2, 4 };
    ir.exactRespa.forceLayout.enabled = true;

    EXPECT_FALSE(ir.useMts);
    EXPECT_TRUE(useExactRespa(ir));
    EXPECT_FALSE(useMtsSubstepping(ir));
}

TEST(ExactRespaStandalone, ValidatesWithoutLegacyMtsFields)
{
    t_inputrec ir;
    configureExactRespaInputRecord(&ir);
    ir.exactRespa = threeLevelExactRespa();

    EXPECT_FALSE(ir.useMts);
    EXPECT_TRUE(useExactRespa(ir));
    EXPECT_TRUE(haveValidExactRespaSetup(ir));
    EXPECT_TRUE(checkExactRespaRequirements(ir).empty());
    EXPECT_EQ(exactRespaNonbondedMtsFactor(ir), 4);
    EXPECT_EQ(exactRespaNonbondedInnerLevel(ir), 0);
    EXPECT_EQ(exactRespaNonbondedMiddleLevel(ir), 1);
    EXPECT_EQ(exactRespaNonbondedOuterLevel(ir), 2);
}

TEST(ExactRespaStandalone, ExactRespaStepWorkUsesStandaloneExactSchedule)
{
    t_inputrec ir;
    ir.exactRespa.levelStepFactors = { 1, 2, 4 };
    ir.exactRespa.forceLayout.enabled = true;

    SimulationWorkload simulationWork;
    simulationWork.useExactRespa = true;

    const ExactRespaStepWork exactStepWork = setupExactRespaStepWork(
            GMX_FORCE_ALLFORCES, ir, 0, DomainLifetimeWorkload{}, simulationWork);

    EXPECT_EQ(exactStepWork.highestActiveLevel, 2);
    EXPECT_TRUE(exactStepWork.haveSlowForceLevels);
    EXPECT_FALSE(exactStepWork.combineForcesBeforeHaloExchange);
}

TEST(ExactRespaStandalone, SimulationWorkloadSeparatesLegacyMtsAndExactSubsteps)
{
    SimulationWorkload simulationWork;
    simulationWork.useMts        = false;
    simulationWork.useExactRespa = true;

    EXPECT_FALSE(simulationWork.useLegacyMtsSubsteps());
    EXPECT_TRUE(simulationWork.useExactRespa);
}

TEST(ExactRespaStandalone, ExactStepWorkloadDoesNotPopulateLegacyMtsFlags)
{
    t_inputrec ir;
    configureExactRespaInputRecord(&ir);
    ir.exactRespa = threeLevelExactRespa();

    SimulationWorkload simulationWork;
    simulationWork.useExactRespa = true;

    const StepWorkload stepWork = setupExactRespaStepWorkload(
            GMX_FORCE_ALLFORCES, ir, 0, DomainLifetimeWorkload{}, simulationWork);
    const ExactRespaStepWork exactStepWork = setupExactRespaStepWork(
            GMX_FORCE_ALLFORCES, ir, 0, DomainLifetimeWorkload{}, simulationWork);

    EXPECT_EQ(stepWork.highestActiveMtsLevel, 0);
    EXPECT_FALSE(stepWork.computeSlowForces);
    EXPECT_FALSE(stepWork.combineMtsForcesBeforeHaloExchange);
    EXPECT_TRUE(stepWork.computeLongRangeNonbondedForces);

    EXPECT_EQ(exactStepWork.highestActiveLevel, 2);
    EXPECT_TRUE(exactStepWork.haveSlowForceLevels);
}

TEST(ExactRespaRecursion, MatchesLAMMPSRecursiveEventOrderForThreeLevelSchedule)
{
    const ExactRespaParameters exactRespa = threeLevelExactRespa();
    const auto referenceTraces = referenceBaseStepTraces(exactRespa, 1);

    ASSERT_EQ(referenceTraces.size(), static_cast<size_t>(exactRespa.levelStepFactors.back()));
    for (int baseStep = 0; baseStep < exactRespa.levelStepFactors.back(); ++baseStep)
    {
        const ExactRespaBaseStepTrace actualTrace = exactRespaBaseStepTrace(exactRespa, baseStep);
        EXPECT_THAT(actualTrace.initialKickLevels,
                    ::testing::ElementsAreArray(referenceTraces[baseStep].initialKickLevels))
                << "baseStep=" << baseStep << " " << describeTrace(actualTrace);
        EXPECT_THAT(actualTrace.refreshedForceLevels,
                    ::testing::ElementsAreArray(referenceTraces[baseStep].refreshedForceLevels))
                << "baseStep=" << baseStep << " " << describeTrace(actualTrace);
        EXPECT_THAT(actualTrace.finalKickLevels,
                    ::testing::ElementsAreArray(referenceTraces[baseStep].finalKickLevels))
                << "baseStep=" << baseStep << " " << describeTrace(actualTrace);
    }
}

TEST(ExactRespaRecursion, MatchesLAMMPSRecursiveEventOrderAcrossTwoOuterCycles)
{
    const ExactRespaParameters exactRespa = threeLevelExactRespa();

    std::vector<RespaEvent> referenceEvents;
    appendLammpsReferenceEvents(
            exactRespa, static_cast<int>(exactRespa.levelStepFactors.size()) - 1, &referenceEvents);
    appendLammpsReferenceEvents(
            exactRespa, static_cast<int>(exactRespa.levelStepFactors.size()) - 1, &referenceEvents);

    const std::vector<RespaEvent> flattenedEvents =
            flattenedExactRespaEvents(exactRespa, 2 * exactRespa.levelStepFactors.back());

    EXPECT_THAT(describeEvents(flattenedEvents), ::testing::ElementsAreArray(describeEvents(referenceEvents)));
}

TEST(ExactRespaRecursion, MatchesLAMMPSRecursiveBaseStepTraceForNonBinaryNestedSchedule)
{
    const ExactRespaParameters exactRespa = nonBinaryThreeLevelExactRespa();
    const auto                 referenceTraces = referenceBaseStepTraces(exactRespa, 1);

    ASSERT_EQ(referenceTraces.size(), static_cast<size_t>(exactRespa.levelStepFactors.back()));
    for (int baseStep = 0; baseStep < exactRespa.levelStepFactors.back(); ++baseStep)
    {
        const ExactRespaBaseStepTrace actualTrace = exactRespaBaseStepTrace(exactRespa, baseStep);
        EXPECT_THAT(actualTrace.initialKickLevels,
                    ::testing::ElementsAreArray(referenceTraces[baseStep].initialKickLevels))
                << "baseStep=" << baseStep << " " << describeTrace(actualTrace);
        EXPECT_THAT(actualTrace.refreshedForceLevels,
                    ::testing::ElementsAreArray(referenceTraces[baseStep].refreshedForceLevels))
                << "baseStep=" << baseStep << " " << describeTrace(actualTrace);
        EXPECT_THAT(actualTrace.finalKickLevels,
                    ::testing::ElementsAreArray(referenceTraces[baseStep].finalKickLevels))
                << "baseStep=" << baseStep << " " << describeTrace(actualTrace);
    }
}

} // namespace test
} // namespace gmx
