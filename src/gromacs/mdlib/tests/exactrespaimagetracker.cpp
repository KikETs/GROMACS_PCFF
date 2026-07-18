/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */

#include "gmxpre.h"

#include "gromacs/mdlib/exactrespaimagetracker.h"

#include <filesystem>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{
namespace
{

ExactRespaImageBox orthorhombicBox(const double x, const double y, const double z)
{
    return { { { x, 0, 0 }, { 0, y, 0 }, { 0, 0, z } } };
}

void copyBoxToMatrix(const ExactRespaImageBox& source, matrix destination)
{
    for (int i = 0; i < DIM; ++i)
    {
        for (int j = 0; j < DIM; ++j)
        {
            destination[i][j] = source[i][j];
        }
    }
}

ExactRespaImageSidecar makeSidecar(const int64_t          step,
                                   const ExactRespaImageBox& box,
                                   ArrayRef<const DVec>    statePositions,
                                   ArrayRef<const ExactRespaAtomImage> images)
{
    EXPECT_EQ(statePositions.size(), images.size());
    ExactRespaImageSidecar sidecar;
    sidecar.step = step;
    sidecar.box  = box;
    sidecar.atoms.resize(statePositions.size());
    for (Index atom = 0; atom < statePositions.ssize(); ++atom)
    {
        sidecar.atoms[atom].globalAtomIndex    = atom;
        sidecar.atoms[atom].statePosition      = statePositions[atom];
        sidecar.atoms[atom].image              = images[atom];
        sidecar.atoms[atom].continuousPosition =
                exactRespaContinuousPosition(statePositions[atom], images[atom], box);
    }
    return sidecar;
}

TEST(ExactRespaImageTrackerMath, RecoversMultiBoxCrossingsInBothDirections)
{
    const ExactRespaImageBox box = orthorhombicBox(10.0, 20.0, 30.0);
    const DVec from = { 31.25, -41.5, 60.125 };
    const DVec to   = { 1.25, 18.5, 0.125 };

    const auto shift = exactRespaIntegerLatticeShift(from, to, box);
    ASSERT_TRUE(shift.has_value());
    EXPECT_EQ(*shift, (ExactRespaAtomImage{ 3, -3, 2 }));
    EXPECT_EQ(exactRespaContinuousPosition(to, *shift, box), from);
}

TEST(ExactRespaImageTrackerMath, RecoversTriclinicRowVectorShift)
{
    const ExactRespaImageBox box =
            { { { 10.0, 0.0, 0.0 }, { 1.5, 8.0, 0.0 }, { -0.5, 2.0, 6.0 } } };
    const DVec                state = { 2.0, 3.0, 4.0 };
    const ExactRespaAtomImage image = { -2, 3, -4 };
    const DVec continuous = exactRespaContinuousPosition(state, image, box);

    const auto recovered = exactRespaIntegerLatticeShift(continuous, state, box);
    ASSERT_TRUE(recovered.has_value());
    EXPECT_EQ(*recovered, image);
}

TEST(ExactRespaImageTrackerMath, RejectsNonLatticeDifference)
{
    const ExactRespaImageBox box = orthorhombicBox(10.0, 10.0, 10.0);
    EXPECT_FALSE(exactRespaIntegerLatticeShift(DVec{ 1.25, 2.0, 3.0 },
                                               DVec{ 1.0, 2.0, 3.0 },
                                               box)
                         .has_value());
}

class ExactRespaImageTrackerTest : public ::testing::Test
{
protected:
    TestFileManager fileManager_;
};

TEST_F(ExactRespaImageTrackerTest, TracksExactWrapsAndCurrentNptBox)
{
    const auto inputPath  = fileManager_.getTemporaryFilePath("images-in.sidecar");
    const auto outputPath = fileManager_.getTemporaryFilePath("images-out.sidecar");

    ExactRespaImageBox box = orthorhombicBox(10.0, 10.0, 10.0);
    const std::vector<DVec> inputState = { { 11.25, -0.5, 30.125 },
                                           { -20.25, 21.5, 9.75 } };
    const std::vector<ExactRespaAtomImage> inputImages(inputState.size(), { 0, 0, 0 });
    writeExactRespaImageSidecarAtomically(
            inputPath, makeSidecar(10, box, inputState, inputImages));

    std::vector<RVec> state = { { 11.25_real, -0.5_real, 30.125_real },
                                { -20.25_real, 21.5_real, 9.75_real } };
    matrix simulationBox;
    copyBoxToMatrix(box, simulationBox);

    ExactRespaImageTracker tracker(inputPath, outputPath);
    tracker.putAtomsInBoxAndTrack(10,
                                  PbcType::Xyz,
                                  simulationBox,
                                  false,
                                  nullptr,
                                  state,
                                  ArrayRef<RVec>{},
                                  1);
    ASSERT_EQ(tracker.imagesForTesting(state).size(), 2U);
    EXPECT_EQ(tracker.imagesForTesting(state)[0], (ExactRespaAtomImage{ 1, -1, 3 }));
    EXPECT_EQ(tracker.imagesForTesting(state)[1], (ExactRespaAtomImage{ -3, 2, 0 }));

    // Isotropic MTTK remapping scales both the ordinary state and the box. The
    // integer images remain unchanged and therefore scale the continuous state
    // exactly, rather than being reconstructed from an endpoint displacement.
    box = orthorhombicBox(11.0, 11.0, 11.0);
    copyBoxToMatrix(box, simulationBox);
    for (auto& position : state)
    {
        position *= 1.1_real;
    }

    // Cross multiple boxes in both directions before the next actual wrap.
    state[0][XX] += 22.0_real;
    state[0][YY] -= 33.0_real;
    tracker.putAtomsInBoxAndTrack(11,
                                  PbcType::Xyz,
                                  simulationBox,
                                  false,
                                  nullptr,
                                  state,
                                  ArrayRef<RVec>{},
                                  1);
    EXPECT_EQ(tracker.imagesForTesting(state)[0], (ExactRespaAtomImage{ 3, -4, 3 }));

    tracker.maybeWriteFinal(12, 12, simulationBox, state);
    const ExactRespaImageSidecar output = readExactRespaImageSidecar(outputPath);
    ASSERT_EQ(output.step, 12);
    ASSERT_EQ(output.atoms.size(), 2U);
    EXPECT_EQ(output.atoms[0].image, (ExactRespaAtomImage{ 3, -4, 3 }));

    const DVec expectedContinuous = exactRespaContinuousPosition(
            DVec{ static_cast<double>(state[0][XX]),
                  static_cast<double>(state[0][YY]),
                  static_cast<double>(state[0][ZZ]) },
            ExactRespaAtomImage{ 3, -4, 3 },
            box);
    EXPECT_EQ(output.atoms[0].continuousPosition, expectedContinuous);

    // The legacy stepper can invoke its endpoint hook again while completing
    // final bookkeeping. Once the atomic sidecar is written, later steps are
    // a no-op rather than an erroneous passed-final failure.
    EXPECT_NO_THROW(tracker.maybeWriteFinal(13, 12, simulationBox, state));
}

TEST_F(ExactRespaImageTrackerTest, RejectedTrialCrossingDoesNotCorruptAcceptedBuffer)
{
    const auto inputPath  = fileManager_.getTemporaryFilePath("accepted-in.sidecar");
    const auto outputPath = fileManager_.getTemporaryFilePath("accepted-out.sidecar");
    const ExactRespaImageBox box = orthorhombicBox(10.0, 10.0, 10.0);
    const std::vector<DVec> inputState = { { 1.25, 2.0, 3.0 } };
    const std::vector<ExactRespaAtomImage> inputImages = { { 0, 0, 0 } };
    writeExactRespaImageSidecarAtomically(
            inputPath, makeSidecar(0, box, inputState, inputImages));

    matrix simulationBox;
    copyBoxToMatrix(box, simulationBox);
    std::vector<RVec> acceptedBase = { { 1.25_real, 2.0_real, 3.0_real } };
    std::vector<RVec> trial        = { { 21.25_real, 2.0_real, 3.0_real } };
    ExactRespaImageTracker tracker(inputPath, outputPath);
    tracker.ensureInitialized(0, simulationBox, acceptedBase);

    // Model do_em_step: every reused trial buffer inherits the counters from
    // the accepted source before its newly derived coordinates are evaluated.
    tracker.inheritCoordinateBuffer(acceptedBase, trial);
    tracker.putAtomsInBoxAndTrack(0,
                                  PbcType::Xyz,
                                  simulationBox,
                                  false,
                                  nullptr,
                                  trial,
                                  ArrayRef<RVec>{},
                                  1);
    ASSERT_EQ(tracker.imagesForTesting(trial).size(), 1U);
    EXPECT_EQ(tracker.imagesForTesting(trial)[0], (ExactRespaAtomImage{ 2, 0, 0 }));

    // Reusing the same trial buffer must overwrite, not extend, the rejected
    // trial's counters.
    trial[0] = acceptedBase[0] + RVec{ 0.25_real, 0.0_real, 0.0_real };
    tracker.inheritCoordinateBuffer(acceptedBase, trial);
    EXPECT_EQ(tracker.imagesForTesting(trial)[0], (ExactRespaAtomImage{ 0, 0, 0 }));

    // Model accepting the second trial by making its coordinate buffer s_min.
    // Final output must select that buffer's overwritten counters rather than
    // the rejected evaluation previously performed in the same storage.
    EXPECT_EQ(tracker.imagesForTesting(acceptedBase)[0], (ExactRespaAtomImage{ 0, 0, 0 }));
    tracker.maybeWriteFinal(1, 1, simulationBox, trial);
    const ExactRespaImageSidecar output = readExactRespaImageSidecar(outputPath);
    ASSERT_EQ(output.atoms.size(), 1U);
    EXPECT_EQ(output.atoms[0].image, (ExactRespaAtomImage{ 0, 0, 0 }));
    EXPECT_EQ(output.atoms[0].continuousPosition, (DVec{ 1.5, 2.0, 3.0 }));
}

TEST_F(ExactRespaImageTrackerTest, RejectsCheckpointWithoutMatchingStepSidecar)
{
    const auto inputPath  = fileManager_.getTemporaryFilePath("resume-in.sidecar");
    const auto outputPath = fileManager_.getTemporaryFilePath("resume-out.sidecar");
    const ExactRespaImageBox box = orthorhombicBox(10.0, 10.0, 10.0);
    const std::vector<DVec> stateDouble = { { 1.0, 2.0, 3.0 } };
    const std::vector<ExactRespaAtomImage> images = { { 0, 0, 0 } };
    writeExactRespaImageSidecarAtomically(
            inputPath, makeSidecar(100, box, stateDouble, images));

    matrix simulationBox;
    copyBoxToMatrix(box, simulationBox);
    const std::vector<RVec> state = { { 1.0_real, 2.0_real, 3.0_real } };
    ExactRespaImageTracker  tracker(inputPath, outputPath);
    EXPECT_THROW(tracker.ensureInitialized(101, simulationBox, state), InvalidInputError);
}

TEST_F(ExactRespaImageTrackerTest, WritesExactProvidedStateAtEarlyEmEndpoint)
{
    const auto inputPath  = fileManager_.getTemporaryFilePath("em-in.sidecar");
    const auto outputPath = fileManager_.getTemporaryFilePath("em-out.sidecar");
    const ExactRespaImageBox box = orthorhombicBox(8.0, 9.0, 10.0);
    const std::vector<DVec> inputState = { { 9.0, -1.0, 2.0 } };
    const std::vector<ExactRespaAtomImage> inputImages = { { 0, 0, 0 } };
    writeExactRespaImageSidecarAtomically(
            inputPath, makeSidecar(0, box, inputState, inputImages));

    matrix simulationBox;
    copyBoxToMatrix(box, simulationBox);
    std::vector<RVec> state = { { 9.0_real, -1.0_real, 2.0_real } };
    ExactRespaImageTracker tracker(inputPath, outputPath);
    tracker.putAtomsInBoxAndTrack(0,
                                  PbcType::Xyz,
                                  simulationBox,
                                  false,
                                  nullptr,
                                  state,
                                  ArrayRef<RVec>{},
                                  1);

    // Model a minimizer that converges at step 3, before its requested nsteps.
    state[0] += RVec{ 0.125_real, -0.25_real, 0.5_real };
    tracker.maybeWriteFinal(3, 3, simulationBox, state);
    const ExactRespaImageSidecar output = readExactRespaImageSidecar(outputPath);

    ASSERT_EQ(output.step, 3);
    ASSERT_EQ(output.atoms.size(), 1U);
    EXPECT_DOUBLE_EQ(output.atoms[0].statePosition[XX], state[0][XX]);
    EXPECT_DOUBLE_EQ(output.atoms[0].statePosition[YY], state[0][YY]);
    EXPECT_DOUBLE_EQ(output.atoms[0].statePosition[ZZ], state[0][ZZ]);
    EXPECT_EQ(output.atoms[0].continuousPosition,
              exactRespaContinuousPosition(output.atoms[0].statePosition,
                                            output.atoms[0].image,
                                            output.box));
}

} // namespace
} // namespace test
} // namespace gmx
