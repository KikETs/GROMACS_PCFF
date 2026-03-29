/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */

#include "gmxpre.h"

#include <filesystem>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/mdrun/exactrespasteppertesting.h"
#include "gromacs/mdtypes/exactrespaparameters.h"
#include "gromacs/mdtypes/exactrespaschedule.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/testfilemanager.h"

#include "moduletest.h"

namespace gmx
{
namespace test
{
namespace
{

constexpr double c_fourierSpacingNm = 0.08;

std::filesystem::path repoRoot()
{
    std::filesystem::path root = TestFileManager::getInputDataDirectory();
    for (int i = 0; i < 4; ++i)
    {
        root = root.parent_path();
    }
    return root;
}

std::filesystem::path exactRespaFixtureRoot(const std::string& systemId)
{
    return repoRoot() / "tests" / "reference_results" / "m6_respa" / systemId;
}

std::string makeExactRespaRuntimeEventMdp(const int level2Factor, const int level3Factor, const int outerCycles)
{
    std::ostringstream mdp;
    mdp << "title                   = exact respa runtime event trace\n"
        << "integrator              = md-vv\n"
        << "dt                      = 0.0005\n"
        << "nsteps                  = " << (outerCycles * level3Factor) << "\n"
        << "constraints             = none\n"
        << "cutoff-scheme           = Verlet\n"
        << "nstlist                 = " << level3Factor << "\n"
        << "rlist                   = 0.99\n"
        << "rvdw                    = 0.9\n"
        << "rcoulomb                = 0.9\n"
        << "vdwtype                 = Cut-off\n"
        << "vdw-modifier            = none\n"
        << "coulombtype             = PME\n"
        << "coulomb-modifier        = none\n"
        << "ewald-rtol              = 1e-6\n"
        << "pme-order               = 4\n"
        << "fourierspacing          = " << c_fourierSpacingNm << "\n"
        << "epsilon-r               = 1\n"
        << "pbc                     = xyz\n"
        << "tcoupl                  = no\n"
        << "pcoupl                  = no\n"
        << "comm-mode               = none\n"
        << "verlet-buffer-tolerance = -1\n"
        << "gen-vel                 = no\n"
        << "nstcalcenergy           = " << level3Factor << "\n"
        << "nstenergy               = " << level3Factor << "\n"
        << "nstlog                  = " << level3Factor << "\n"
        << "nstxout                 = 0\n"
        << "nstvout                 = 0\n"
        << "nstfout                 = 0\n"
        << "nstxout-compressed      = 0\n"
        << "exact-respa             = yes\n"
        << "exact-respa-levels      = 3\n"
        << "exact-respa-level2-factor = " << level2Factor << "\n"
        << "exact-respa-level3-factor = " << level3Factor << "\n"
        << "exact-respa-bond-level  = 1\n"
        << "exact-respa-angle-level = 1\n"
        << "exact-respa-dihedral-level = 1\n"
        << "exact-respa-improper-level = 1\n"
        << "exact-respa-pair14-level = 2\n"
        << "exact-respa-pair-level  = 3\n"
        << "exact-respa-kspace-level = 3\n"
        << "exact-respa-inner-level = 1\n"
        << "exact-respa-middle-level = 2\n"
        << "exact-respa-outer-level = 3\n"
        << "exact-respa-inner-off   = 0.30\n"
        << "exact-respa-inner-on    = 0.45\n"
        << "exact-respa-outer-on    = 0.60\n"
        << "exact-respa-outer-off   = 0.80\n";
    return mdp.str();
}

ExactRespaParameters makeExactRespaParameters(const int level2Factor, const int level3Factor)
{
    ExactRespaParameters exactRespa;
    exactRespa.levelStepFactors = { 1, level2Factor, level3Factor };
    return exactRespa;
}

int loopCount(const ExactRespaParameters& exactRespa, const int level)
{
    return (level + 1 == static_cast<int>(exactRespa.levelStepFactors.size()))
                   ? 1
                   : exactRespa.levelStepFactors[level + 1] / exactRespa.levelStepFactors[level];
}

void appendLammpsReferenceEvents(const ExactRespaParameters&         exactRespa,
                                 const int                          level,
                                 std::vector<ExactRespaRuntimeEvent>* events)
{
    ASSERT_NE(events, nullptr);

    for (int iloop = 0; iloop < loopCount(exactRespa, level); ++iloop)
    {
        events->push_back({ 0, ExactRespaRuntimeEventType::InitialKick, level });
        if (level == 0)
        {
            events->push_back({ 0, ExactRespaRuntimeEventType::Drift, 0 });
        }
        else
        {
            appendLammpsReferenceEvents(exactRespa, level - 1, events);
        }
        events->push_back({ 0, ExactRespaRuntimeEventType::RefreshForce, level });
        events->push_back({ 0, ExactRespaRuntimeEventType::FinalKick, level });
    }
}

std::vector<ExactRespaBaseStepTrace> referenceBaseStepTraces(const std::vector<ExactRespaRuntimeEvent>& events)
{
    std::vector<ExactRespaBaseStepTrace> traces;
    size_t                               eventIndex = 0;

    while (eventIndex < events.size())
    {
        ExactRespaBaseStepTrace trace;
        while (eventIndex < events.size() && events[eventIndex].type == ExactRespaRuntimeEventType::InitialKick)
        {
            trace.initialKickLevels.push_back(events[eventIndex].level);
            ++eventIndex;
        }

        if (eventIndex >= events.size() || events[eventIndex].type != ExactRespaRuntimeEventType::Drift)
        {
            ADD_FAILURE() << "Reference event stream ended before drift";
            return {};
        }
        ++eventIndex;

        while (eventIndex < events.size() && events[eventIndex].type == ExactRespaRuntimeEventType::RefreshForce)
        {
            trace.refreshedForceLevels.push_back(events[eventIndex].level);
            ++eventIndex;
            if (eventIndex >= events.size() || events[eventIndex].type != ExactRespaRuntimeEventType::FinalKick)
            {
                ADD_FAILURE() << "Reference event stream ended before final kick";
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
                                                             const int                   outerCycles)
{
    std::vector<ExactRespaRuntimeEvent> events;
    const int                           highestLevel = static_cast<int>(exactRespa.levelStepFactors.size()) - 1;
    for (int outerCycle = 0; outerCycle < outerCycles; ++outerCycle)
    {
        appendLammpsReferenceEvents(exactRespa, highestLevel, &events);
    }
    return referenceBaseStepTraces(events);
}

std::vector<ExactRespaRuntimeEvent> flattenBaseStepTraces(const std::vector<ExactRespaBaseStepTrace>& traces)
{
    std::vector<ExactRespaRuntimeEvent> events;
    for (Index baseStep = 0; baseStep < ssize(traces); ++baseStep)
    {
        const auto& trace = traces[baseStep];
        for (const int level : trace.initialKickLevels)
        {
            events.push_back({ baseStep, ExactRespaRuntimeEventType::InitialKick, level });
        }
        events.push_back({ baseStep, ExactRespaRuntimeEventType::Drift, 0 });
        for (const int level : trace.refreshedForceLevels)
        {
            events.push_back({ baseStep, ExactRespaRuntimeEventType::RefreshForce, level });
        }
        for (const int level : trace.finalKickLevels)
        {
            events.push_back({ baseStep, ExactRespaRuntimeEventType::FinalKick, level });
        }
    }
    return events;
}

std::vector<ExactRespaRuntimeEvent> lammpsReferenceEventsForExecutedBaseSteps(
        const ExactRespaParameters& exactRespa, const int numExecutedBaseSteps)
{
    const int outerCycles = (numExecutedBaseSteps + exactRespa.levelStepFactors.back() - 1)
                            / exactRespa.levelStepFactors.back();
    auto traces = referenceBaseStepTraces(exactRespa, outerCycles);
    traces.resize(numExecutedBaseSteps);
    return flattenBaseStepTraces(traces);
}

std::string describeEvent(const ExactRespaRuntimeEvent& event)
{
    switch (event.type)
    {
        case ExactRespaRuntimeEventType::InitialKick:
            return formatString("step%lld:initial-kick(L%d)",
                                static_cast<long long>(event.baseStep),
                                event.level);
        case ExactRespaRuntimeEventType::Drift:
            return formatString("step%lld:drift", static_cast<long long>(event.baseStep));
        case ExactRespaRuntimeEventType::RefreshForce:
            return formatString("step%lld:refresh-force(L%d)",
                                static_cast<long long>(event.baseStep),
                                event.level);
        case ExactRespaRuntimeEventType::FinalKick:
            return formatString("step%lld:final-kick(L%d)",
                                static_cast<long long>(event.baseStep),
                                event.level);
    }

    return "unknown";
}

std::vector<std::string> describeEvents(const std::vector<ExactRespaRuntimeEvent>& events)
{
    std::vector<std::string> labels;
    labels.reserve(events.size());
    for (const auto& event : events)
    {
        labels.push_back(describeEvent(event));
    }
    return labels;
}

class CollectingExactRespaRuntimeEventSink final : public ExactRespaRuntimeEventSink
{
public:
    void recordEvent(const ExactRespaRuntimeEvent& event) override
    {
        events.push_back(event);
    }

    std::vector<ExactRespaRuntimeEvent> events;
};

class ExactRespaRuntimeEventSinkGuard
{
public:
    explicit ExactRespaRuntimeEventSinkGuard(ExactRespaRuntimeEventSink* sink) : sink_(sink)
    {
        setExactRespaRuntimeEventSinkForTesting(sink_);
    }

    ~ExactRespaRuntimeEventSinkGuard()
    {
        setExactRespaRuntimeEventSinkForTesting(nullptr);
    }

private:
    ExactRespaRuntimeEventSink* sink_;
};

using ExactRespaRuntimeEventParams = std::tuple<const char*, int, int, int>;

class ExactRespaRuntimeEventOrderTest :
    public MdrunTestFixture,
    public ::testing::WithParamInterface<ExactRespaRuntimeEventParams>
{
};

TEST_P(ExactRespaRuntimeEventOrderTest, MatchesLAMMPSReferenceEventOrder)
{
    const auto [systemIdRaw, level2Factor, level3Factor, outerCycles] = GetParam();
    const std::string systemId(systemIdRaw);

    runner_.topFileName_ = (exactRespaFixtureRoot(systemId) / "topol.top").string();
    runner_.groFileName_ = (exactRespaFixtureRoot(systemId) / "initial_nve.gro").string();
    runner_.useStringAsMdpFile(makeExactRespaRuntimeEventMdp(level2Factor, level3Factor, outerCycles));
    runner_.setMaxWarn(1);

    ASSERT_EQ(0, runner_.callGrompp()) << "grompp failed for exact runtime event trace " << systemId;

    CollectingExactRespaRuntimeEventSink sink;
    ExactRespaRuntimeEventSinkGuard      sinkGuard(&sink);

    CommandLine mdrunCaller;
    mdrunCaller.append("mdrun");
    mdrunCaller.append("-reprod");
    ASSERT_EQ(0, runner_.callMdrun(mdrunCaller)) << "mdrun failed for exact runtime event trace " << systemId;

    const int  executedBaseSteps = outerCycles * level3Factor + 1;
    const auto referenceEvents = lammpsReferenceEventsForExecutedBaseSteps(
            makeExactRespaParameters(level2Factor, level3Factor), executedBaseSteps);
    EXPECT_EQ(describeEvents(sink.events), describeEvents(referenceEvents));
}

std::string exactRespaRuntimeEventCaseName(
        const testing::TestParamInfo<ExactRespaRuntimeEventParams>& info)
{
    return formatString("%s_L2_%d_L3_%d_cycles_%d",
                        std::get<0>(info.param),
                        std::get<1>(info.param),
                        std::get<2>(info.param),
                        std::get<3>(info.param));
}

INSTANTIATE_TEST_SUITE_P(PcffExactRespaRuntimeEvents,
                         ExactRespaRuntimeEventOrderTest,
                         ::testing::Values(ExactRespaRuntimeEventParams{ "small_oligomer", 2, 4, 1 },
                                           ExactRespaRuntimeEventParams{ "small_oligomer", 2, 4, 2 },
                                           ExactRespaRuntimeEventParams{ "small_oligomer", 3, 6, 1 }),
                         exactRespaRuntimeEventCaseName);

} // namespace
} // namespace test
} // namespace gmx
