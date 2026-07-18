/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */

#include "gmxpre.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <limits>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "gromacs/math/units.h"
#include "gromacs/mdrun/exactrespasoftstart.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/utility/gmxassert.h"

#include "testutils/setenv.h"

namespace gmx
{
namespace test
{
namespace
{

class ScopedEnvironmentVariable
{
public:
    ScopedEnvironmentVariable(const std::string& name, const std::optional<std::string>& value) : name_(name)
    {
        if (const char* previousValue = std::getenv(name.c_str()))
        {
            previousValue_ = previousValue;
        }
        if (value.has_value())
        {
            GMX_RELEASE_ASSERT(gmxSetenv(name.c_str(), value->c_str(), true) == 0,
                               "Failed to set soft-start test environment variable");
        }
        else
        {
            GMX_RELEASE_ASSERT(gmxUnsetenv(name.c_str()) == 0,
                               "Failed to unset soft-start test environment variable");
        }
    }

    ~ScopedEnvironmentVariable()
    {
        if (previousValue_.has_value())
        {
            GMX_RELEASE_ASSERT(gmxSetenv(name_.c_str(), previousValue_->c_str(), true) == 0,
                               "Failed to restore soft-start test environment variable");
        }
        else
        {
            GMX_RELEASE_ASSERT(gmxUnsetenv(name_.c_str()) == 0,
                               "Failed to clear soft-start test environment variable");
        }
    }

private:
    std::string                name_;
    std::optional<std::string> previousValue_;
};

TEST(ExactRespaSoftStartConfigTest, IsDisabledUnlessStageEnvironmentOptsIn)
{
    const ScopedEnvironmentVariable enable("GMX_PCFF_EXACT_RESPA_SOFT_START", std::nullopt);
    EXPECT_FALSE(exactRespaSoftStartConfigFromEnvironment().enabled);
}

TEST(ExactRespaSoftStartConfigTest, OptInDefaultsMatchEq01LAMMPSSettings)
{
    const ScopedEnvironmentVariable enable("GMX_PCFF_EXACT_RESPA_SOFT_START", std::string("1"));
    const ScopedEnvironmentVariable xlimit("GMX_PCFF_EXACT_RESPA_NVE_LIMIT_XMAX_NM", std::nullopt);
    const ScopedEnvironmentVariable xlimitAlias("GMX_PCFF_EXACT_RESPA_SOFT_XLIMIT_NM", std::nullopt);
    const ScopedEnvironmentVariable temperature("GMX_PCFF_EXACT_RESPA_LANGEVIN_TEMP_K", std::nullopt);
    const ScopedEnvironmentVariable damping("GMX_PCFF_EXACT_RESPA_LANGEVIN_TAU_PS", std::nullopt);
    const ScopedEnvironmentVariable seed("GMX_PCFF_EXACT_RESPA_LANGEVIN_SEED", std::nullopt);
    const ScopedEnvironmentVariable zero("GMX_PCFF_EXACT_RESPA_LANGEVIN_ZERO_RANDOM", std::nullopt);

    const ExactRespaSoftStartConfig config = exactRespaSoftStartConfigFromEnvironment();
    EXPECT_TRUE(config.enabled);
    EXPECT_NEAR(config.xlimitNm, 0.01, 1.0e-6);
    EXPECT_NEAR(config.temperatureK, 353.0, 1.0e-5);
    EXPECT_NEAR(config.dampingTimePs, 0.05, 1.0e-6);
    EXPECT_EQ(config.seed, 97531U);
    EXPECT_TRUE(config.zeroRandomForce);
}

TEST(ExactRespaSoftStartConfigTest, StageEnvironmentCanOverrideEveryEq01Parameter)
{
    const ScopedEnvironmentVariable enable("GMX_PCFF_EXACT_RESPA_SOFT_START", std::string("1"));
    const ScopedEnvironmentVariable xlimit("GMX_PCFF_EXACT_RESPA_NVE_LIMIT_XMAX_NM", std::string("0.02"));
    const ScopedEnvironmentVariable temperature("GMX_PCFF_EXACT_RESPA_LANGEVIN_TEMP_K", std::string("400"));
    const ScopedEnvironmentVariable damping("GMX_PCFF_EXACT_RESPA_LANGEVIN_TAU_PS", std::string("0.1"));
    const ScopedEnvironmentVariable seed("GMX_PCFF_EXACT_RESPA_LANGEVIN_SEED", std::string("12345"));
    const ScopedEnvironmentVariable zero("GMX_PCFF_EXACT_RESPA_LANGEVIN_ZERO_RANDOM", std::string("0"));

    const ExactRespaSoftStartConfig config = exactRespaSoftStartConfigFromEnvironment();
    EXPECT_TRUE(config.enabled);
    EXPECT_NEAR(config.xlimitNm, 0.02, 1.0e-6);
    EXPECT_NEAR(config.temperatureK, 400.0, 1.0e-5);
    EXPECT_NEAR(config.dampingTimePs, 0.1, 1.0e-6);
    EXPECT_EQ(config.seed, 12345U);
    EXPECT_FALSE(config.zeroRandomForce);
}

TEST(ExactRespaSoftStartConfigTest, RejectsGpuStateUpdate)
{
    const ScopedEnvironmentVariable enable("GMX_PCFF_EXACT_RESPA_SOFT_START", std::string("1"));
    t_inputrec                     inputRecord;
    inputRecord.eI                    = IntegrationAlgorithm::VV;
    inputRecord.etc                   = TemperatureCoupling::No;
    inputRecord.pressureCouplingOptions.epc = PressureCoupling::No;
    inputRecord.exactRespa.levelStepFactors = { 1, 4 };
    inputRecord.exactRespa.forceLayout.enabled = true;
    inputRecord.delta_t              = 0.000125;
    inputRecord.init_step            = 0;
    ExactRespaSoftStartState state;

    EXPECT_DEATH_IF_SUPPORTED(
            initializeExactRespaSoftStartState(inputRecord, true, 0, &state),
            "requires CPU state update");
}

TEST(ExactRespaSoftStartConfigTest, RejectsCheckpointResumeInsideSoftStage)
{
    const ScopedEnvironmentVariable enable("GMX_PCFF_EXACT_RESPA_SOFT_START", std::string("1"));
    t_inputrec                     inputRecord;
    inputRecord.eI                    = IntegrationAlgorithm::VV;
    inputRecord.etc                   = TemperatureCoupling::No;
    inputRecord.pressureCouplingOptions.epc = PressureCoupling::No;
    inputRecord.exactRespa.levelStepFactors = { 1, 4 };
    inputRecord.exactRespa.forceLayout.enabled = true;
    inputRecord.delta_t              = 0.000125;
    inputRecord.init_step            = 0;
    ExactRespaSoftStartState state;

    EXPECT_DEATH_IF_SUPPORTED(
            initializeExactRespaSoftStartState(inputRecord, false, 4, &state),
            "not checkpoint-resumable");
}

TEST(ExactRespaSoftStartRandomTest, IsDeterministicAndKeyedByGlobalAtomIndex)
{
    const double atom17 = exactRespaSoftStartUniform(97531, 9, 17, XX);
    const double atom42 = exactRespaSoftStartUniform(97531, 9, 42, XX);

    EXPECT_DOUBLE_EQ(atom17, exactRespaSoftStartUniform(97531, 9, 17, XX));
    EXPECT_DOUBLE_EQ(atom42, exactRespaSoftStartUniform(97531, 9, 42, XX));
    EXPECT_NE(atom17, atom42);
    EXPECT_NE(atom17, exactRespaSoftStartUniform(97531, 10, 17, XX));
    EXPECT_NE(atom17, exactRespaSoftStartUniform(97531, 9, 17, YY));
}

TEST(ExactRespaSoftStartLangevinTest, UsesLAMMPSUniformForceAndDragFormula)
{
    constexpr real mass        = 12.0_real;
    constexpr real temperature = 353.0_real;
    constexpr real dampingTime = 0.05_real;
    constexpr real outerDt     = 0.0005_real;
    const RVec     velocity    = { 1.0, -2.0, 3.0 };
    const RVec     centered    = { 0.5, -0.25, 0.0 };
    const RVec     randomMean  = { 1.0, 2.0, 3.0 };

    const RVec force = exactRespaSoftStartLangevinForce(
            mass, velocity, centered, randomMean, temperature, dampingTime, outerDt);
    const double amplitude =
            std::sqrt(24.0 * mass * c_boltz * temperature / (dampingTime * outerDt));
    for (int dimension = 0; dimension < DIM; ++dimension)
    {
        const double expected = -mass * velocity[dimension] / dampingTime
                                + amplitude * centered[dimension] - randomMean[dimension];
        const double tolerance = 500.0 * std::numeric_limits<real>::epsilon()
                                 * std::max(1.0, std::abs(expected));
        EXPECT_NEAR(force[dimension], expected, tolerance);
    }
}

TEST(ExactRespaSoftStartLangevinTest, ZeroYesRemovesComponentWiseRandomForceMean)
{
    constexpr real temperature  = 353.0_real;
    constexpr real dampingTime  = 0.05_real;
    constexpr real outerDt      = 0.0005_real;
    const RVec     zeroVelocity = { 0.0, 0.0, 0.0 };
    const RVec     centeredA    = { 0.4, -0.2, 0.1 };
    const RVec     centeredB    = { -0.1, 0.3, -0.4 };
    constexpr real massA        = 4.0_real;
    constexpr real massB        = 16.0_real;
    const double amplitudeA =
            std::sqrt(24.0 * massA * c_boltz * temperature / (dampingTime * outerDt));
    const double amplitudeB =
            std::sqrt(24.0 * massB * c_boltz * temperature / (dampingTime * outerDt));
    RVec meanRandomForce;
    for (int dimension = 0; dimension < DIM; ++dimension)
    {
        meanRandomForce[dimension] = static_cast<real>(
                0.5 * (amplitudeA * centeredA[dimension] + amplitudeB * centeredB[dimension]));
    }

    const RVec forceA = exactRespaSoftStartLangevinForce(
            massA, zeroVelocity, centeredA, meanRandomForce, temperature, dampingTime, outerDt);
    const RVec forceB = exactRespaSoftStartLangevinForce(
            massB, zeroVelocity, centeredB, meanRandomForce, temperature, dampingTime, outerDt);
    for (int dimension = 0; dimension < DIM; ++dimension)
    {
        const double forceScale = std::max(std::abs(static_cast<double>(forceA[dimension])),
                                           std::abs(static_cast<double>(forceB[dimension])));
        EXPECT_NEAR(forceA[dimension] + forceB[dimension],
                    0.0,
                    2000.0 * std::numeric_limits<real>::epsilon() * std::max(1.0, forceScale));
    }
}

TEST(ExactRespaSoftStartLimitTest, CapsAbsoluteSpeedFromXlimitOverOuterDt)
{
    const RVec inverseMass = { 1.0, 1.0, 1.0 };
    const RVec zeroForce   = { 0.0, 0.0, 0.0 };
    RVec       velocity    = { 30.0, 40.0, 0.0 };

    EXPECT_TRUE(applyExactRespaSoftStartHalfKick(
            0.0005, inverseMass, zeroForce, nullptr, 0.01 / 0.0005, &velocity));
    EXPECT_NEAR(velocity[XX], 12.0, 1.0e-12);
    EXPECT_NEAR(velocity[YY], 16.0, 1.0e-12);
    EXPECT_NEAR(velocity[ZZ], 0.0, 1.0e-12);
}

TEST(ExactRespaSoftStartLimitTest, AppliesCapAfterEverySequentialRespaHalfKick)
{
    const RVec inverseMass = { 1.0, 1.0, 1.0 };
    const RVec outerForce  = { 30.0, 0.0, 0.0 };
    const RVec innerForce  = { -30.0, 0.0, 0.0 };
    RVec       velocity    = { 0.0, 0.0, 0.0 };

    EXPECT_TRUE(applyExactRespaSoftStartHalfKick(
            2.0, inverseMass, outerForce, nullptr, 20.0, &velocity));
    EXPECT_FALSE(applyExactRespaSoftStartHalfKick(
            2.0, inverseMass, innerForce, nullptr, 20.0, &velocity));
    EXPECT_NEAR(velocity[XX], -10.0, 1.0e-12);

    // A fused (+30 -30) kick would yield zero and is therefore observably wrong.
    EXPECT_NE(velocity[XX], 0.0);
}

TEST(ExactRespaSoftStartLimitTest, CombinesPhysicalAndLangevinForceBeforeOneCap)
{
    const RVec inverseMass   = { 1.0, 1.0, 1.0 };
    const RVec physicalForce = { 30.0, 0.0, 0.0 };
    const RVec langevinForce = { -30.0, 0.0, 0.0 };
    RVec       velocity      = { 0.0, 0.0, 0.0 };

    EXPECT_FALSE(applyExactRespaSoftStartHalfKick(
            2.0, inverseMass, physicalForce, &langevinForce, 20.0, &velocity));
    EXPECT_NEAR(velocity[XX], 0.0, 1.0e-12);
}

TEST(ExactRespaSoftStartScheduleTest, Eq01SoftStageHasExpectedSlowForceBoundaries)
{
    constexpr int outerStepFactor = 4;
    constexpr int baseSteps       = 66400; // 8.3 ps / 0.000125 ps
    int           boundaryCount   = 0;
    for (int baseStep = 0; baseStep <= baseSteps; ++baseStep)
    {
        boundaryCount += exactRespaSoftStartIsOuterBoundary(baseStep, 0, outerStepFactor) ? 1 : 0;
    }
    EXPECT_EQ(boundaryCount, 16601);
    EXPECT_EQ(baseSteps / outerStepFactor, 16600);
}

} // namespace
} // namespace test
} // namespace gmx
