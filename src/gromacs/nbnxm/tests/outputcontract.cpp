/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */

/*! \internal \file
 * \brief Tests for exact r-RESPA NBNXM output routing contracts.
 *
 * \ingroup module_nbnxm
 */

#include "gmxpre.h"

#include <algorithm>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/atominfo.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/multipletimestepping.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/nbnxm_enums.h"
#include "gromacs/nbnxm/pairlistparams.h"
#include "gromacs/nbnxm/pairlistset.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/nbnxm/pairsearch.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

class NbnxmOutputContractTest : public ::testing::Test
{
protected:
    std::vector<RVec> forceInner_ = std::vector<RVec>(4);
    std::vector<RVec> forceMiddle_ = std::vector<RVec>(4);
    std::vector<RVec> forceOuter_ = std::vector<RVec>(4);
    std::vector<RVec> forceAlternative_ = std::vector<RVec>(4);
    std::vector<RVec> shiftOuter_ = std::vector<RVec>(c_numShiftVectors);
    std::vector<RVec> shiftAlternative_ = std::vector<RVec>(c_numShiftVectors);
    ForceWithVirial   directVirialOuter_{ forceOuter_, true };
    ForceWithVirial   directVirialAlternative_{ forceAlternative_, true };
    std::vector<real> vdwEnergy_ = std::vector<real>(1);
    std::vector<real> coulombEnergy_ = std::vector<real>(1);

    NbnxmOutputSink shiftSink(MtsNonbondedRespaContribution contribution, std::vector<RVec>& force)
    {
        return { contribution,
                 LammpsRespaNonbondedOutputSinkKind::ShiftForce,
                 AtomLocality::Local,
                 force,
                 {},
                 nullptr };
    }

    NbnxmOutputSink shiftSink(MtsNonbondedRespaContribution contribution,
                              std::vector<RVec>&            force,
                              std::vector<RVec>&            shift)
    {
        return { contribution,
                 LammpsRespaNonbondedOutputSinkKind::ShiftForce,
                 AtomLocality::Local,
                 force,
                 shift,
                 nullptr };
    }

    NbnxmOutputSink directVirialSink(MtsNonbondedRespaContribution contribution,
                                     ForceWithVirial*              directVirial)
    {
        return { contribution,
                 LammpsRespaNonbondedOutputSinkKind::ForceWithVirial,
                 AtomLocality::Local,
                 directVirial->force_,
                 {},
                 directVirial };
    }

    NbnxmOutputContract threeLevelForceOnlyContract()
    {
        NbnxmOutputContract contract;
        contract.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Inner, forceInner_));
        contract.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Middle, forceMiddle_));
        contract.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Outer, forceOuter_));
        return contract;
    }

};

std::unique_ptr<nonbonded_verlet_t> makeMinimalNbnxm(const int numOutputBuffers)
{
    const PinningPolicy   pinPolicy = PinningPolicy::CannotBePinned;
    constexpr int         numAtoms  = 8;
    gmx_omp_nthreads_set(ModuleMultiThread::Pairsearch, numOutputBuffers);
    gmx_omp_nthreads_set(ModuleMultiThread::Nonbonded, numOutputBuffers);

    const NbnxmKernelSetup kernelSetup{ NbnxmKernelType::Cpu4x4_PlainC,
                                        EwaldExclusionType::NotSet };
    const PairlistParams   pairlistParams(kernelSetup.kernelType, {}, false, 1.0_real, false);
    auto                   pairlistSets =
            std::make_unique<PairlistSets>(pairlistParams, false, 0, pinPolicy);
    auto pairSearch = std::make_unique<PairSearch>(PbcType::Xyz,
                                                   false,
                                                   nullptr,
                                                   nullptr,
                                                   pairlistParams.pairlistType,
                                                   false,
                                                   true,
                                                   numOutputBuffers,
                                                   pinPolicy);

    const std::vector<real> nbfp = { 0.0_real, 0.0_real };
    auto                    atomData =
            std::make_unique<nbnxn_atomdata_t>(pinPolicy,
                                               MDLogger(),
                                               kernelSetup.kernelType,
                                               std::nullopt,
                                               LJCombinationRule::None,
                                               nbfp,
                                               false,
                                               1,
                                               numOutputBuffers);
    atomData->resizeCoordinateBuffer(numAtoms, 0);
    atomData->resizeForceBuffers();

    auto nbv = std::make_unique<nonbonded_verlet_t>(
            std::move(pairlistSets), std::move(pairSearch), std::move(atomData), kernelSetup, nullptr);

    const matrix box = { { 4.0_real, 0.0_real, 0.0_real },
                         { 0.0_real, 4.0_real, 0.0_real },
                         { 0.0_real, 0.0_real, 4.0_real } };
    const RVec   lowerCorner{ 0.0_real, 0.0_real, 0.0_real };
    const RVec   upperCorner{ 4.0_real, 4.0_real, 4.0_real };
    std::vector<RVec> coordinates(numAtoms);
    for (int atomIndex = 0; atomIndex < numAtoms; atomIndex++)
    {
        coordinates[atomIndex] = { 0.2_real + 0.1_real * atomIndex,
                                   0.3_real + 0.1_real * atomIndex,
                                   0.4_real + 0.1_real * atomIndex };
    }
    const std::vector<int32_t> atomInfo(numAtoms, sc_atomInfo_HasVdw);
    nbv->putAtomsOnGrid(box,
                        0,
                        lowerCorner,
                        upperCorner,
                        nullptr,
                        Range<int>(0, numAtoms),
                        numAtoms,
                        numAtoms / 64.0_real,
                        atomInfo,
                        coordinates,
                        nullptr);

    return nbv;
}

TEST_F(NbnxmOutputContractTest, ResolvesActiveContributionSinkWithoutGpuState)
{
    const NbnxmOutputContract contract = threeLevelForceOnlyContract();

    const auto& innerSink = nbnxmOutputSinkForContribution(contract, MtsNonbondedRespaContribution::Inner);
    const auto& middleSink =
            nbnxmOutputSinkForContribution(contract, MtsNonbondedRespaContribution::Middle);
    const auto& outerSink = nbnxmOutputSinkForContribution(contract, MtsNonbondedRespaContribution::Outer);

    EXPECT_EQ(innerSink.force.data(), forceInner_.data());
    EXPECT_EQ(middleSink.force.data(), forceMiddle_.data());
    EXPECT_EQ(outerSink.force.data(), forceOuter_.data());
}

TEST_F(NbnxmOutputContractTest, RejectsMissingOrDuplicateContributionSink)
{
    NbnxmOutputContract missingMiddle = threeLevelForceOnlyContract();
    missingMiddle.sinks.erase(missingMiddle.sinks.begin() + 1);
    EXPECT_THROW_GMX(nbnxmOutputSinkForContribution(missingMiddle, MtsNonbondedRespaContribution::Middle),
                     InternalError);

    NbnxmOutputContract duplicateOuter = threeLevelForceOnlyContract();
    duplicateOuter.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Outer, forceAlternative_));
    EXPECT_THROW_GMX(nbnxmOutputSinkForContribution(duplicateOuter, MtsNonbondedRespaContribution::Outer),
                     InternalError);
}

TEST_F(NbnxmOutputContractTest, AcceptsOuterDirectVirialAndEnergyOwnership)
{
    NbnxmOutputContract contract;
    contract.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Inner, forceInner_));
    contract.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Middle, forceMiddle_));
    contract.sinks.push_back(
            directVirialSink(MtsNonbondedRespaContribution::Outer, &directVirialOuter_));
    contract.virial.accumulateVirial    = true;
    contract.virial.contribution        = MtsNonbondedRespaContribution::Outer;
    contract.virial.sinkKind            = LammpsRespaNonbondedOutputSinkKind::ForceWithVirial;
    contract.virial.directVirialOutput  = &directVirialOuter_;
    contract.energy.accumulateEnergy    = true;
    contract.energy.contribution        = MtsNonbondedRespaContribution::Outer;
    contract.energy.vdwEnergy           = vdwEnergy_;
    contract.energy.coulombEnergy       = coulombEnergy_;

    const auto& outerSink = nbnxmOutputSinkForContribution(contract, MtsNonbondedRespaContribution::Outer);

    EXPECT_EQ(outerSink.directVirialOutput, &directVirialOuter_);
}

TEST_F(NbnxmOutputContractTest, ResolvesNativeMultiContributionSinks)
{
    NbnxmOutputContract contract = threeLevelForceOnlyContract();
    contract.kind = NbnxmOutputContractKind::NativeMultiContribution;
    contract.nativeMultiContribution.contributions = { MtsNonbondedRespaContribution::Inner,
                                                       MtsNonbondedRespaContribution::Middle,
                                                       MtsNonbondedRespaContribution::Outer };

    const auto sinks = nbnxmOutputSinksForNativeMultiContribution(contract);

    ASSERT_EQ(sinks.size(), 3);
    EXPECT_EQ(sinks[0]->contribution, MtsNonbondedRespaContribution::Inner);
    EXPECT_EQ(sinks[0]->force.data(), forceInner_.data());
    EXPECT_EQ(sinks[1]->contribution, MtsNonbondedRespaContribution::Middle);
    EXPECT_EQ(sinks[1]->force.data(), forceMiddle_.data());
    EXPECT_EQ(sinks[2]->contribution, MtsNonbondedRespaContribution::Outer);
    EXPECT_EQ(sinks[2]->force.data(), forceOuter_.data());
}

TEST_F(NbnxmOutputContractTest, AcceptsNativeOuterDirectVirialAndEnergyOwnership)
{
    NbnxmOutputContract contract;
    contract.kind = NbnxmOutputContractKind::NativeMultiContribution;
    contract.nativeMultiContribution.contributions = { MtsNonbondedRespaContribution::Inner,
                                                       MtsNonbondedRespaContribution::Middle,
                                                       MtsNonbondedRespaContribution::Outer };
    contract.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Inner, forceInner_));
    contract.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Middle, forceMiddle_));
    contract.sinks.push_back(
            directVirialSink(MtsNonbondedRespaContribution::Outer, &directVirialOuter_));
    contract.virial.accumulateVirial   = true;
    contract.virial.contribution       = MtsNonbondedRespaContribution::Outer;
    contract.virial.sinkKind           = LammpsRespaNonbondedOutputSinkKind::ForceWithVirial;
    contract.virial.directVirialOutput = &directVirialOuter_;
    contract.energy.accumulateEnergy   = true;
    contract.energy.contribution       = MtsNonbondedRespaContribution::Outer;
    contract.energy.vdwEnergy          = vdwEnergy_;
    contract.energy.coulombEnergy      = coulombEnergy_;

    const auto sinks = nbnxmOutputSinksForNativeMultiContribution(contract);

    ASSERT_EQ(sinks.size(), 3);
    EXPECT_EQ(sinks[2]->directVirialOutput, &directVirialOuter_);
}

TEST_F(NbnxmOutputContractTest, RejectsNativeMultiContributionContractMismatch)
{
    NbnxmOutputContract missingNativeContribution = threeLevelForceOnlyContract();
    missingNativeContribution.kind = NbnxmOutputContractKind::NativeMultiContribution;
    missingNativeContribution.nativeMultiContribution.contributions = {
        MtsNonbondedRespaContribution::Inner,
        MtsNonbondedRespaContribution::Middle
    };
    EXPECT_THROW_GMX(nbnxmOutputSinksForNativeMultiContribution(missingNativeContribution),
                     InternalError);

    NbnxmOutputContract duplicateNativeContribution = threeLevelForceOnlyContract();
    duplicateNativeContribution.kind = NbnxmOutputContractKind::NativeMultiContribution;
    duplicateNativeContribution.nativeMultiContribution.contributions = {
        MtsNonbondedRespaContribution::Inner,
        MtsNonbondedRespaContribution::Middle,
        MtsNonbondedRespaContribution::Middle
    };
    EXPECT_THROW_GMX(nbnxmOutputSinksForNativeMultiContribution(duplicateNativeContribution),
                     InternalError);

    NbnxmOutputContract mixedContract = threeLevelForceOnlyContract();
    mixedContract.nativeMultiContribution.contributions = { MtsNonbondedRespaContribution::Inner,
                                                            MtsNonbondedRespaContribution::Outer };
    EXPECT_THROW_GMX(nbnxmOutputSinkForContribution(mixedContract,
                                                    MtsNonbondedRespaContribution::Inner),
                     InternalError);
}

TEST_F(NbnxmOutputContractTest, RejectsLookupThroughWrongExecutionModel)
{
    NbnxmOutputContract nativeContract = threeLevelForceOnlyContract();
    nativeContract.kind = NbnxmOutputContractKind::NativeMultiContribution;
    nativeContract.nativeMultiContribution.contributions = { MtsNonbondedRespaContribution::Inner,
                                                             MtsNonbondedRespaContribution::Middle,
                                                             MtsNonbondedRespaContribution::Outer };
    EXPECT_THROW_GMX(nbnxmOutputSinkForContribution(nativeContract,
                                                    MtsNonbondedRespaContribution::Inner),
                     InternalError);

    NbnxmOutputContract perContributionContract = threeLevelForceOnlyContract();
    EXPECT_THROW_GMX(nbnxmOutputSinksForNativeMultiContribution(perContributionContract),
                     InternalError);
}

TEST_F(NbnxmOutputContractTest, AcceptsOuterShiftForceVirialOwnership)
{
    NbnxmOutputContract contract;
    contract.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Inner, forceInner_));
    contract.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Middle, forceMiddle_));
    contract.sinks.push_back(
            shiftSink(MtsNonbondedRespaContribution::Outer, forceOuter_, shiftOuter_));
    contract.virial.accumulateVirial = true;
    contract.virial.contribution     = MtsNonbondedRespaContribution::Outer;
    contract.virial.sinkKind         = LammpsRespaNonbondedOutputSinkKind::ShiftForce;
    contract.virial.shiftForces      = shiftOuter_;

    const auto& outerSink = nbnxmOutputSinkForContribution(contract, MtsNonbondedRespaContribution::Outer);

    EXPECT_EQ(outerSink.shiftForces.data(), shiftOuter_.data());
}

TEST_F(NbnxmOutputContractTest, RejectsInnerVirialOrEnergyOwnership)
{
    NbnxmOutputContract virialInner = threeLevelForceOnlyContract();
    virialInner.virial.accumulateVirial = true;
    virialInner.virial.contribution     = MtsNonbondedRespaContribution::Inner;
    virialInner.virial.sinkKind         = LammpsRespaNonbondedOutputSinkKind::ShiftForce;
    virialInner.virial.shiftForces      = shiftOuter_;
    EXPECT_THROW_GMX(nbnxmOutputSinkForContribution(virialInner, MtsNonbondedRespaContribution::Inner),
                     InternalError);

    NbnxmOutputContract energyInner = threeLevelForceOnlyContract();
    energyInner.energy.accumulateEnergy = true;
    energyInner.energy.contribution     = MtsNonbondedRespaContribution::Inner;
    energyInner.energy.vdwEnergy        = vdwEnergy_;
    energyInner.energy.coulombEnergy    = coulombEnergy_;
    EXPECT_THROW_GMX(nbnxmOutputSinkForContribution(energyInner, MtsNonbondedRespaContribution::Inner),
                     InternalError);
}

TEST_F(NbnxmOutputContractTest, RejectsVirialOwnerSinkMismatch)
{
    NbnxmOutputContract shiftMismatch;
    shiftMismatch.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Inner, forceInner_));
    shiftMismatch.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Middle, forceMiddle_));
    shiftMismatch.sinks.push_back(
            shiftSink(MtsNonbondedRespaContribution::Outer, forceOuter_, shiftOuter_));
    shiftMismatch.virial.accumulateVirial = true;
    shiftMismatch.virial.contribution     = MtsNonbondedRespaContribution::Outer;
    shiftMismatch.virial.sinkKind         = LammpsRespaNonbondedOutputSinkKind::ShiftForce;
    shiftMismatch.virial.shiftForces      = shiftAlternative_;
    EXPECT_THROW_GMX(nbnxmOutputSinkForContribution(shiftMismatch, MtsNonbondedRespaContribution::Outer),
                     InternalError);

    NbnxmOutputContract directMismatch;
    directMismatch.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Inner, forceInner_));
    directMismatch.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Middle, forceMiddle_));
    directMismatch.sinks.push_back(
            directVirialSink(MtsNonbondedRespaContribution::Outer, &directVirialOuter_));
    directMismatch.virial.accumulateVirial   = true;
    directMismatch.virial.contribution       = MtsNonbondedRespaContribution::Outer;
    directMismatch.virial.sinkKind           = LammpsRespaNonbondedOutputSinkKind::ForceWithVirial;
    directMismatch.virial.directVirialOutput = &directVirialAlternative_;
    EXPECT_THROW_GMX(nbnxmOutputSinkForContribution(directMismatch, MtsNonbondedRespaContribution::Outer),
                     InternalError);
}

TEST_F(NbnxmOutputContractTest, NativeBoundaryAllocatesStorageBeforeReduction)
{
    auto nbv = makeMinimalNbnxm(1);

    NbnxmOutputContract contract = threeLevelForceOnlyContract();
    contract.kind = NbnxmOutputContractKind::NativeMultiContribution;
    contract.nativeMultiContribution.contributions = { MtsNonbondedRespaContribution::Inner,
                                                       MtsNonbondedRespaContribution::Middle,
                                                       MtsNonbondedRespaContribution::Outer };

    EXPECT_EQ(nbv->nbat().numNativeMultiContributionOutputSets(), 0);
    nbv->atomdata_add_nbat_f_to_native_multi_outputs(contract);
    EXPECT_EQ(nbv->nbat().numNativeMultiContributionOutputSets(), 3);
    EXPECT_EQ(nbv->nbat().nativeMultiContributionOutputBuffers(2).size(), 1);
}

TEST_F(NbnxmOutputContractTest, NativeBoundaryReducesContributionIndexedStorageToSinks)
{
    auto nbv = makeMinimalNbnxm(1);
    for (RVec& force : forceInner_)
    {
        force = { 0.0_real, 0.0_real, 0.0_real };
    }
    for (RVec& force : forceMiddle_)
    {
        force = { 0.0_real, 0.0_real, 0.0_real };
    }
    for (RVec& force : forceOuter_)
    {
        force = { 0.0_real, 0.0_real, 0.0_real };
    }
    for (RVec& shiftForce : shiftOuter_)
    {
        shiftForce = { 0.0_real, 0.0_real, 0.0_real };
    }

    NbnxmOutputContract contract = threeLevelForceOnlyContract();
    contract.kind = NbnxmOutputContractKind::NativeMultiContribution;
    contract.nativeMultiContribution.contributions = { MtsNonbondedRespaContribution::Inner,
                                                       MtsNonbondedRespaContribution::Middle,
                                                       MtsNonbondedRespaContribution::Outer };
    contract.sinks[2] = shiftSink(MtsNonbondedRespaContribution::Outer, forceOuter_, shiftOuter_);

    nbv->nbat().ensureNativeMultiContributionOutputBuffers(3);
    auto innerBuffers = nbv->nbat().nativeMultiContributionOutputBuffers(0);
    auto middleBuffers = nbv->nbat().nativeMultiContributionOutputBuffers(1);
    auto outerBuffers = nbv->nbat().nativeMultiContributionOutputBuffers(2);
    for (auto outputBuffers : { innerBuffers, middleBuffers, outerBuffers })
    {
        for (nbnxn_atomdata_output_t& outputBuffer : outputBuffers)
        {
            std::fill(outputBuffer.f.begin(), outputBuffer.f.end(), 0.0_real);
            std::fill(outputBuffer.fshift.begin(), outputBuffer.fshift.end(), 0.0_real);
        }
    }
    innerBuffers[0].f[0] = 1.0_real;
    innerBuffers[0].f[1] = 2.0_real;
    innerBuffers[0].f[2] = 3.0_real;
    outerBuffers[0].f[3] = 4.0_real;
    outerBuffers[0].f[4] = 5.0_real;
    outerBuffers[0].f[5] = 6.0_real;
    outerBuffers[0].fshift[XX] = 7.0_real;
    outerBuffers[0].fshift[YY] = 8.0_real;
    outerBuffers[0].fshift[ZZ] = 9.0_real;

    nbv->atomdata_add_nbat_f_to_native_multi_outputs(contract);

    EXPECT_EQ(forceInner_[0][XX], 1.0_real);
    EXPECT_EQ(forceInner_[0][YY], 2.0_real);
    EXPECT_EQ(forceInner_[0][ZZ], 3.0_real);
    EXPECT_EQ(forceOuter_[1][XX], 4.0_real);
    EXPECT_EQ(forceOuter_[1][YY], 5.0_real);
    EXPECT_EQ(forceOuter_[1][ZZ], 6.0_real);
    EXPECT_EQ(shiftOuter_[0][XX], 7.0_real);
    EXPECT_EQ(shiftOuter_[0][YY], 8.0_real);
    EXPECT_EQ(shiftOuter_[0][ZZ], 9.0_real);
}

TEST_F(NbnxmOutputContractTest, NativeBoundaryAllowsThreadedLocalFullSpanReduction)
{
    auto nbv = makeMinimalNbnxm(2);
    for (RVec& force : forceInner_)
    {
        force = { 0.0_real, 0.0_real, 0.0_real };
    }
    for (RVec& force : forceMiddle_)
    {
        force = { 0.0_real, 0.0_real, 0.0_real };
    }

    NbnxmOutputContract contract;
    contract.kind = NbnxmOutputContractKind::NativeMultiContribution;
    contract.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Inner, forceInner_));
    contract.sinks.push_back(shiftSink(MtsNonbondedRespaContribution::Middle, forceMiddle_));
    contract.nativeMultiContribution.contributions = { MtsNonbondedRespaContribution::Inner,
                                                       MtsNonbondedRespaContribution::Middle };

    nbv->nbat().ensureNativeMultiContributionOutputBuffers(2);
    auto contributionBuffers = nbv->nbat().nativeMultiContributionOutputBuffers(0);
    ASSERT_EQ(contributionBuffers.size(), 2);
    for (nbnxn_atomdata_output_t& outputBuffer : contributionBuffers)
    {
        std::fill(outputBuffer.f.begin(), outputBuffer.f.end(), 0.0_real);
    }

    gmx_bitmask_t combinedThreadMask;
    bitmask_init_bit(&combinedThreadMask, 0);
    bitmask_set_bit(&combinedThreadMask, 1);
    nbv->nbat().bufferFlags().assign(1, combinedThreadMask);

    contributionBuffers[0].f[0] = 1.0_real;
    contributionBuffers[0].f[1] = 2.0_real;
    contributionBuffers[0].f[2] = 3.0_real;
    contributionBuffers[1].f[0] = 10.0_real;
    contributionBuffers[1].f[1] = 20.0_real;
    contributionBuffers[1].f[2] = 30.0_real;

    nbv->atomdata_add_nbat_f_to_native_multi_outputs(contract);

    EXPECT_EQ(forceInner_[0][XX], 11.0_real);
    EXPECT_EQ(forceInner_[0][YY], 22.0_real);
    EXPECT_EQ(forceInner_[0][ZZ], 33.0_real);
}

TEST(NbnxmAtomDataNativeOutputStorageTest, AllocatesContributionIndexedOutputBuffers)
{
    const std::vector<real> nbfp = { 0.0_real, 0.0_real };
    nbnxn_atomdata_t        atomData(PinningPolicy::CannotBePinned,
                              MDLogger(),
                              NbnxmKernelType::Cpu4x4_PlainC,
                              std::nullopt,
                              LJCombinationRule::None,
                              nbfp,
                              false,
                              1,
                              2);
    atomData.resizeCoordinateBuffer(5, 0);
    atomData.resizeForceBuffers();

    atomData.ensureNativeMultiContributionOutputBuffers(3);

    EXPECT_EQ(atomData.numNativeMultiContributionOutputSets(), 3);
    auto firstContributionBuffers = atomData.nativeMultiContributionOutputBuffers(0);
    auto thirdContributionBuffers = atomData.nativeMultiContributionOutputBuffers(2);
    ASSERT_EQ(firstContributionBuffers.size(), atomData.outputBuffers().size());
    ASSERT_EQ(thirdContributionBuffers.size(), atomData.outputBuffers().size());

    const int paddedSize =
            (atomData.numAtoms() + NBNXN_BUFFERFLAG_SIZE - 1) / NBNXN_BUFFERFLAG_SIZE
            * NBNXN_BUFFERFLAG_SIZE;
    EXPECT_EQ(firstContributionBuffers[0].f.size(), paddedSize * atomData.fstride);
    EXPECT_EQ(thirdContributionBuffers[1].f.size(), paddedSize * atomData.fstride);

    firstContributionBuffers[0].f[0] = 1.25_real;
    thirdContributionBuffers[0].f[0] = 2.5_real;
    EXPECT_EQ(firstContributionBuffers[0].f[0], 1.25_real);
    EXPECT_EQ(thirdContributionBuffers[0].f[0], 2.5_real);
}

TEST(NbnxmAtomDataNativeOutputStorageTest, ResizesAndReconfiguresContributionIndexedOutputBuffers)
{
    const std::vector<real> nbfp = { 0.0_real, 0.0_real };
    nbnxn_atomdata_t        atomData(PinningPolicy::CannotBePinned,
                              MDLogger(),
                              NbnxmKernelType::Cpu4x4_PlainC,
                              std::nullopt,
                              LJCombinationRule::None,
                              nbfp,
                              false,
                              1,
                              2);
    atomData.resizeCoordinateBuffer(5, 0);
    atomData.ensureNativeMultiContributionOutputBuffers(2);
    auto contributionBuffers = atomData.nativeMultiContributionOutputBuffers(1);
    contributionBuffers[0].f[0] = 3.0_real;

    atomData.ensureNativeMultiContributionOutputBuffers(2);
    EXPECT_EQ(atomData.nativeMultiContributionOutputBuffers(1)[0].f[0], 3.0_real);

    atomData.resizeCoordinateBuffer(19, 0);
    atomData.resizeForceBuffers();
    const int paddedSize =
            (atomData.numAtoms() + NBNXN_BUFFERFLAG_SIZE - 1) / NBNXN_BUFFERFLAG_SIZE
            * NBNXN_BUFFERFLAG_SIZE;
    EXPECT_EQ(atomData.nativeMultiContributionOutputBuffers(1)[0].f.size(),
              paddedSize * atomData.fstride);

    atomData.ensureNativeMultiContributionOutputBuffers(1);
    EXPECT_EQ(atomData.numNativeMultiContributionOutputSets(), 1);
    EXPECT_THROW_GMX(atomData.nativeMultiContributionOutputBuffers(1), InternalError);
}

TEST(NbnxmAtomDataNativeOutputStorageTest, StagesNormalOutputBuffersIntoContributionIndexedStorage)
{
    const std::vector<real> nbfp = { 0.0_real, 0.0_real };
    nbnxn_atomdata_t        atomData(PinningPolicy::CannotBePinned,
                              MDLogger(),
                              NbnxmKernelType::Cpu4x4_PlainC,
                              std::nullopt,
                              LJCombinationRule::None,
                              nbfp,
                              false,
                              1,
                              1);
    atomData.resizeCoordinateBuffer(5, 0);
    atomData.resizeForceBuffers();
    atomData.ensureNativeMultiContributionOutputBuffers(2);

    atomData.outputBuffer(0).f[0]      = 11.0_real;
    atomData.outputBuffer(0).f[1]      = 12.0_real;
    atomData.outputBuffer(0).fshift[0] = 13.0_real;
    atomData.outputBuffer(0).Vvdw[0]   = 14.0_real;
    atomData.outputBuffer(0).Vc[0]     = 15.0_real;

    atomData.copyOutputBuffersToNativeMultiContributionOutputBuffers(1);

    const auto stagedBuffers = atomData.nativeMultiContributionOutputBuffers(1);
    EXPECT_EQ(stagedBuffers[0].f[0], 11.0_real);
    EXPECT_EQ(stagedBuffers[0].f[1], 12.0_real);
    EXPECT_EQ(stagedBuffers[0].fshift[0], 13.0_real);
    EXPECT_EQ(stagedBuffers[0].Vvdw[0], 14.0_real);
    EXPECT_EQ(stagedBuffers[0].Vc[0], 15.0_real);
}

} // namespace
} // namespace test
} // namespace gmx
