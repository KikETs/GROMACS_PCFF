/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2012- The GROMACS Authors
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

/* Doxygen gets confused (buggy) about the block in this file in combination with
 * the  namespace prefix, and thinks store is documented here.
 * This will solve itself with the second-generation nbnxn kernels, so for now
 * we just tell Doxygen to stay out.
 */
#ifndef DOXYGEN

/* This is the innermost loop contents for the NBNxM SIMD kernels.
 */
{
    /* Inner loop specific constexpr variables */
    static_assert(nR % 2 == 0);
    constexpr int c_nRLJ = (c_iLJInteractions == ILJInteractions::None
                                    ? 0
                                    : nR / (c_iLJInteractions == ILJInteractions::Half ? 2 : 1));
    /* When calculating RF or Ewald interactions we calculate the electrostatic/LJ
     * forces on excluded atom pairs here in the non-bonded loops.
     * But when energies and/or virial is required we calculate them
     * separately to as then it is easier to separate the energy and virial
     * contributions.
     */
    constexpr bool c_haveExclusionForces =
            (c_calculateCoulombInteractions || haveLJEwaldGeometric) && c_needToCheckExclusions;

    /* The force times 1/r */
    std::array<SimdReal, nR> fScalarV;

    /* j-cluster index */
    const int cj = l_cj[cjind].cj;

    /* Atom indices (of the first atom in the cluster) */
    const int gmx_unused aj = cj * c_jClusterSize;

    const int ajx =
            (c_jClusterSize == c_stride ? aj * DIM : (cj >> 1) * DIM * c_stride + (cj & 1) * c_jClusterSize);
    const int ajy = ajx + c_stride;
    const int ajz = ajy + c_stride;

    /* Interaction (non-exclusion) mask of all 1's or 0's */
    const auto interactV = loadSimdPairInteractionMasks<c_needToCheckExclusions, kernelLayout>(
            static_cast<int>(l_cj[cjind].excl), exclusionFilterV);

    /* load j atom coordinates */
    SimdReal jx_S = loadJAtomData<kernelLayout>(x, ajx);
    SimdReal jy_S = loadJAtomData<kernelLayout>(x, ajy);
    SimdReal jz_S = loadJAtomData<kernelLayout>(x, ajz);

    /* Calculate distance */
    const auto dxV = genArr<nR>([&](int i) { return ixV[i] - jx_S; });
    const auto dyV = genArr<nR>([&](int i) { return iyV[i] - jy_S; });
    const auto dzV = genArr<nR>([&](int i) { return izV[i] - jz_S; });

    /* rsq = dx*dx + dy*dy + dz*dz */
    auto rSquaredV = genArr<nR>([&](int i) { return norm2(dxV[i], dyV[i], dzV[i]); });

    /* Do the cut-off check */
    auto withinCutoffV = genBoolArr<nR>([&](int i) { return rSquaredV[i] < cutoffSquared; });

    if constexpr (c_needToCheckExclusions)
    {
        if constexpr (c_haveExclusionForces)
        {
            /* Only remove the (sub-)diagonal to avoid double counting exclusion forces */
            diagonalMasker.maskArray(ci_sh, cj, withinCutoffV);
        }
        else
        {
            /* No exclusion forces: remove all excluded atom pairs from the list */
            withinCutoffV = genBoolArr<nR>([&](int i) { return withinCutoffV[i] && interactV[i]; });
        }
    }

#    ifdef COUNT_PAIRS
    npair += pairCountWithinCutoff(rSquaredV, cutoffSquared);
#    endif

    // Ensure the distances do not fall below the limit where r^-12 overflows.
    // This should never happen for normal interactions.
    rSquaredV = genArr<nR>([&](int i) { return gmx::max(rSquaredV[i], minDistanceSquared); });

    /* Calculate 1/r */
    std::array<SimdReal, nR> rInvV;
#    if !GMX_DOUBLE
    rInvV = genArr<nR>([&](int i) { return invsqrt(rSquaredV[i]); });
#    else
    for (int i = 0; i < nR; i += 2)
    {
        invsqrtPair(rSquaredV[i], rSquaredV[i + 1], &rInvV[i], &rInvV[i + 1]);
    }
#    endif

    std::array<SimdReal, nR> qqV;
    if constexpr (c_calculateCoulombInteractions)
    {
        /* Load parameters for j atom */
        const SimdReal jq_S = loadJAtomData<kernelLayout>(q, aj);
        qqV                 = genArr<nR>([&](int i) { return chargeIV[i] * jq_S; });
    }

    /* Set rinv to zero for r beyond the cut-off */
    rInvV = genArr<nR>([&](int i) { return selectByMask(rInvV[i], withinCutoffV[i]); });

    const auto rInvSquaredV = genArr<nR>([&](int i) { return rInvV[i] * rInvV[i]; });

    /* frcoul = qi*qj*(1/r - fsub)*r */
    std::array<SimdReal, nR> frCoulombV;
    std::array<SimdReal, nR> vCoulombV;
    std::array<SimdReal, nR> rInvExclV;

    if constexpr (c_calculateCoulombInteractions)
    {
        /* Note that here we calculate force*r, not the usual force/r.
         * This allows avoiding masking the reaction-field contribution,
         * as frcoul is later multiplied by rinvsq which has been
         * masked with the cut-off check.
         */

        /* Only add 1/r for non-excluded atom pairs */
        if constexpr (c_haveExclusionForces)
        {
            rInvExclV = genArr<nR>([&](int i) { return selectByMask(rInvV[i], interactV[i]); });
        }
        else
        {
            /* We hope that the compiler optimizes rInvExclV away */
            rInvExclV = rInvV;
        }

        /* Electrostatic interactions, frcoul =  qi*qj*(1/r - fsub)*r */
        if constexpr (!calculateEnergies)
        {
            frCoulombV = coulombCalculator.template force<nR>(rSquaredV, rInvV, rInvExclV, withinCutoffV);

            frCoulombV = genArr<nR>([&](int i) { return qqV[i] * frCoulombV[i]; });
        }
        else
        {
            // The potential (RF or Ewald reciprocal) we need to subtract from 1/r
            std::array<SimdReal, nR> vCoulombCorrectionV;

            coulombCalculator.template forceAndCorrectionEnergy<nR>(
                    rSquaredV, rInvV, rInvExclV, withinCutoffV, frCoulombV, vCoulombCorrectionV);

            frCoulombV = genArr<nR>([&](int i) { return qqV[i] * frCoulombV[i]; });

            if constexpr (coulombType != KernelCoulombType::RF)
            {
                if constexpr (c_needToCheckExclusions)
                {
                    vCoulombCorrectionV = genArr<nR>(
                            [&](int i) {
                                return vCoulombCorrectionV[i] + selectByMask(ewaldShift, interactV[i]);
                            });
                }
                else
                {
                    vCoulombCorrectionV =
                            genArr<nR>([&](int i) { return vCoulombCorrectionV[i] + ewaldShift; });
                }
            }

            /* Combine Coulomb and correction terms */
            vCoulombV = genArr<nR>([&](int i)
                                   { return qqV[i] * (rInvExclV[i] - vCoulombCorrectionV[i]); });

            /* Mask energy for cut-off and diagonal */
            vCoulombV =
                    genArr<nR>([&](int i) { return selectByMask(vCoulombV[i], withinCutoffV[i]); });
        }
    }

    /* Lennard-Jones interaction */
    constexpr bool calculateLJInteractions = (c_iLJInteractions != ILJInteractions::None);
    std::array<SimdReal, calculateLJInteractions ? c_nRLJ : 0> gmx_unused frLJV;
    std::array<SimdReal, (calculateLJInteractions && calculateEnergies) ? c_nRLJ : 0> gmx_unused vLJV;

    if constexpr (calculateLJInteractions)
    {
        std::array<SimdBool, c_nRLJ> withinVdwCutoffV;
        if constexpr (haveVdwCutoffCheck)
        {
            withinVdwCutoffV =
                    genBoolArr<c_nRLJ>([&](int i) { return rSquaredV[i] < vdwCutoffSquared; });
        }

        /* Index for loading LJ parameters, complicated when interleaving */
        int aj2;
        if constexpr (ljCombinationRule != LJCombinationRule::None || haveLJEwaldGeometric)
        {
            if constexpr (GMX_SIMD_REAL_WIDTH == c_numJClustersPerSimdRegister * c_stride)
            {
                aj2 = aj * 2;
            }
            else
            {
                aj2 = (cj >> 1) * 2 * c_stride + (cj & 1) * c_jClusterSize;
            }
        }

        if constexpr (ljCombinationRule != LJCombinationRule::LorentzBerthelot)
        {
            /* We use C6 and C12 */
            std::array<SimdReal, c_nRLJ> c6V;
            std::array<SimdReal, c_nRLJ> c12V;

            if constexpr (ljCombinationRule == LJCombinationRule::None)
            {
                // Load 6*C6 and 6*C12 for all pairs
                for (int i = 0; i < c_nRLJ; i++)
                {
                    if constexpr (c_numJClustersPerSimdRegister == 1)
                    {
                        gatherLoadTranspose<c_simdBestPairAlignment>(
                                nbfpI[i], type + aj, &c6V[i], &c12V[i]);
                    }
                    else
                    {
                        gatherLoadTransposeHsimd<c_simdBestPairAlignment>(
                                nbfpI[i * 2], nbfpI[i * 2 + 1], type + aj, &c6V[i], &c12V[i]);
                    }
                }
            }

            if constexpr (ljCombinationRule == LJCombinationRule::Geometric)
            {
                // Load j-atom sqrt(6*C6) and sqrt(12*C12)
                SimdReal c6J  = loadJAtomData<kernelLayout>(ljc, aj2 + 0);
                SimdReal c12J = loadJAtomData<kernelLayout>(ljc, aj2 + c_stride);
                // Compute the combined 6*C6 and 12*C12
                c6V  = genArr<c_nRLJ>([&](int i) { return c6GeomV[i] * c6J; });
                c12V = genArr<c_nRLJ>([&](int i) { return c12GeomV[i] * c12J; });
            }

            // Compute the Lennard Jones force and optionally the energy
            ljCalculator.template forceC6C12<c_nRLJ, c_haveExclusionForces>(
                    rSquaredV, rInvV, rInvSquaredV, interactV, c6V, c12V, sixth_S, repulsionFraction_S, frLJV, vLJV);
        }

        if constexpr (ljCombinationRule == LJCombinationRule::LorentzBerthelot)
        {
            const SimdReal halfSigmaJ   = loadJAtomData<kernelLayout>(ljc, aj2 + 0);
            const SimdReal sqrtEpsilonJ = loadJAtomData<kernelLayout>(ljc, aj2 + c_stride);

            const auto sigmaV = genArr<c_nRLJ>([&](int i) { return halfSigmaIV[i] + halfSigmaJ; });
            const auto epsilonV =
                    genArr<c_nRLJ>([&](int i) { return sqrtEpsilonIV[i] * sqrtEpsilonJ; });

            ljCalculator.template forceSigmaEpsilon<c_nRLJ, c_haveExclusionForces, haveVdwCutoffCheck>(
                    rInvV, interactV, withinVdwCutoffV.data(), sigmaV, epsilonV, sixth_S, twelveth_S, frLJV, vLJV);
        }

        if constexpr (calculateEnergies && c_needToCheckExclusions)
        {
            /* The potential shift should be removed for excluded pairs */
            vLJV = genArr<c_nRLJ>([&](int i) { return selectByMask(vLJV[i], interactV[i]); });
        }

        if constexpr (haveLJEwaldGeometric)
        {
            /* Determine C6 for the grid using the geometric combination rule */
            const SimdReal c6J     = loadJAtomData<kernelLayout>(ljc, aj2 + 0);
            const auto     c6GridV = genArr<c_nRLJ>([&](int i) { return c6GeomV[i] * c6J; });

            addLennardJonesEwaldCorrections<c_nRLJ, c_needToCheckExclusions, calculateEnergies>(
                    rSquaredV, rInvSquaredV, interactV, withinCutoffV.data(), c6GridV, ljEwaldParams, sixth_S, frLJV, vLJV);
        }

        if constexpr (haveVdwCutoffCheck)
        {
            /* frLJ is multiplied later by rinvsq, which is masked for the Coulomb
             * cut-off, but if the VdW cut-off is shorter, we need to mask with that.
             */
            frLJV = genArr<c_nRLJ>([&](int i) { return selectByMask(frLJV[i], withinVdwCutoffV[i]); });
        }

        if constexpr (calculateEnergies)
        {
            /* The potential shift should be removed for pairs beyond cut-off */
            vLJV = genArr<c_nRLJ>(
                    [&](int i) {
                        return selectByMask(
                                vLJV[i], haveVdwCutoffCheck ? withinVdwCutoffV[i] : withinCutoffV[i]);
                    });
        }

    } // calculateLJInteractions

    energyAccumulator.template addEnergies<c_calculateCoulombInteractions ? nR : 0, c_nRLJ, kernelLayout, c_iClusterSize>(
            cj, vCoulombV, vLJV);

    if (exactRespaNativeMultiActive)
    {
        const SimdReal one_S(1.0_real);
        const SimdReal two_S(2.0_real);
        const SimdReal three_S(3.0_real);
        const auto exactRespaSwitchInV = [&](const SimdReal& r, const real off, const real on)
        {
            const SimdReal off_S(off);
            const SimdReal on_S(on);

            if (on <= off)
            {
                return blend(zero_S, one_S, on_S <= r);
            }

            const SimdReal switchFraction = (r - off_S) / (on_S - off_S);
            SimdReal       switched =
                    switchFraction * switchFraction * (three_S - two_S * switchFraction);
            switched                = blend(switched, zero_S, r <= off_S);
            switched                = blend(switched, one_S, on_S <= r);
            return switched;
        };

        const auto directForceScalarV = genArr<nR>(
                [&](int i)
                {
                    SimdReal directScalar = zero_S;
                    if constexpr (c_iLJInteractions != ILJInteractions::None)
                    {
                        if (i < c_nRLJ)
                        {
                            directScalar = directScalar + rInvSquaredV[i] * frLJV[i];
                        }
                    }
                    if constexpr (c_calculateCoulombInteractions)
                    {
                        directScalar = directScalar + rInvSquaredV[i] * qqV[i] * rInvExclV[i];
                    }
                    return directScalar;
                });

        const auto correctionForceScalarV = genArr<nR>(
                [&](int i)
                {
                    if constexpr (c_calculateCoulombInteractions)
                    {
                        return rInvSquaredV[i] * frCoulombV[i] - rInvSquaredV[i] * qqV[i] * rInvExclV[i];
                    }
                    return zero_S;
                });

        const auto distanceV = genArr<nR>([&](int i) { return rSquaredV[i] * rInvV[i]; });
        const auto switchIntoOuterV =
                genArr<nR>([&](int i)
                           { return exactRespaSwitchInV(distanceV[i],
                                                        ic.exactRespaCpuPairSplit.outerOn,
                                                        ic.exactRespaCpuPairSplit.outerOff); });
        const auto switchIntoMiddleV =
                genArr<nR>([&](int i)
                           {
                               if (ic.exactRespaCpuPairSplit.hasMiddle)
                               {
                                   return exactRespaSwitchInV(distanceV[i],
                                                             ic.exactRespaCpuPairSplit.innerOff,
                                                             ic.exactRespaCpuPairSplit.innerOn);
                               }
                               return switchIntoOuterV[i];
                           });
        const auto innerWeightV = genArr<nR>([&](int i) { return one_S - switchIntoMiddleV[i]; });
        const auto middleWeightV =
                genArr<nR>([&](int i)
                           {
                               if (!ic.exactRespaCpuPairSplit.hasMiddle)
                               {
                                   return zero_S;
                               }
                               return switchIntoMiddleV[i] * (one_S - switchIntoOuterV[i]);
                           });

        const bool twoContributionInnerMiddle =
                exactRespaNativeContributionCount == 2
                && exactRespaCpuPairSplitNativeMultiContribution(ic, 0)
                           == MtsNonbondedRespaContribution::Inner
                && exactRespaCpuPairSplitNativeMultiContribution(ic, 1)
                           == MtsNonbondedRespaContribution::Middle;
        if (twoContributionInnerMiddle
            && exactRespaCpuPairSplitNativeMultiTwoContributionFastPathEnabled())
        {
            const auto innerScalarV =
                    genArr<nR>([&](int i) { return innerWeightV[i] * directForceScalarV[i]; });
            const auto middleScalarV =
                    genArr<nR>([&](int i) { return middleWeightV[i] * directForceScalarV[i]; });

            const auto innerTxV = genArr<nR>([&](int i) { return innerScalarV[i] * dxV[i]; });
            const auto innerTyV = genArr<nR>([&](int i) { return innerScalarV[i] * dyV[i]; });
            const auto innerTzV = genArr<nR>([&](int i) { return innerScalarV[i] * dzV[i]; });
            const auto middleTxV = genArr<nR>([&](int i) { return middleScalarV[i] * dxV[i]; });
            const auto middleTyV = genArr<nR>([&](int i) { return middleScalarV[i] * dyV[i]; });
            const auto middleTzV = genArr<nR>([&](int i) { return middleScalarV[i] * dzV[i]; });

            exactRespaNativeForceIXV[0] =
                    genArr<nR>([&](int i) { return exactRespaNativeForceIXV[0][i] + innerTxV[i]; });
            exactRespaNativeForceIYV[0] =
                    genArr<nR>([&](int i) { return exactRespaNativeForceIYV[0][i] + innerTyV[i]; });
            exactRespaNativeForceIZV[0] =
                    genArr<nR>([&](int i) { return exactRespaNativeForceIZV[0][i] + innerTzV[i]; });
            exactRespaNativeForceIXV[1] =
                    genArr<nR>([&](int i) { return exactRespaNativeForceIXV[1][i] + middleTxV[i]; });
            exactRespaNativeForceIYV[1] =
                    genArr<nR>([&](int i) { return exactRespaNativeForceIYV[1][i] + middleTyV[i]; });
            exactRespaNativeForceIZV[1] =
                    genArr<nR>([&](int i) { return exactRespaNativeForceIZV[1][i] + middleTzV[i]; });

            real* innerForce = exactRespaNativeForces[0];
            real* middleForce = exactRespaNativeForces[1];
            if constexpr (c_numJClustersPerSimdRegister == 1)
            {
                store(innerForce + ajx, load<SimdReal>(innerForce + ajx) - sumArray(innerTxV));
                store(innerForce + ajy, load<SimdReal>(innerForce + ajy) - sumArray(innerTyV));
                store(innerForce + ajz, load<SimdReal>(innerForce + ajz) - sumArray(innerTzV));
                store(middleForce + ajx, load<SimdReal>(middleForce + ajx) - sumArray(middleTxV));
                store(middleForce + ajy, load<SimdReal>(middleForce + ajy) - sumArray(middleTyV));
                store(middleForce + ajz, load<SimdReal>(middleForce + ajz) - sumArray(middleTzV));
            }
            else
            {
                decr3Hsimd(innerForce + aj * DIM,
                           sumArray(innerTxV),
                           sumArray(innerTyV),
                           sumArray(innerTzV));
                decr3Hsimd(middleForce + aj * DIM,
                           sumArray(middleTxV),
                           sumArray(middleTyV),
                           sumArray(middleTzV));
            }
        }
        else
        {
            for (int contributionIndex = 0; contributionIndex < exactRespaNativeContributionCount;
                 ++contributionIndex)
            {
                const auto contribution =
                        exactRespaCpuPairSplitNativeMultiContribution(ic, contributionIndex);
                const auto directWeightV = genArr<nR>(
                        [&](int i)
                        {
                            switch (contribution)
                            {
                                case MtsNonbondedRespaContribution::Inner: return innerWeightV[i];
                                case MtsNonbondedRespaContribution::Middle: return middleWeightV[i];
                                case MtsNonbondedRespaContribution::Outer: return switchIntoOuterV[i];
                                case MtsNonbondedRespaContribution::Full: return one_S;
                                case MtsNonbondedRespaContribution::Count:
                                    GMX_RELEASE_ASSERT(false, "Invalid exact r-RESPA CPU contribution");
                                    return zero_S;
                            }

                            GMX_RELEASE_ASSERT(false, "Unhandled exact r-RESPA CPU contribution");
                            return zero_S;
                        });

                const auto contributionScalarV = genArr<nR>(
                        [&](int i)
                        {
                            SimdReal fScalar = directWeightV[i] * directForceScalarV[i];
                            if (exactRespaCpuPairSplitContributionAddsCorrection(contribution))
                            {
                                fScalar = fScalar + correctionForceScalarV[i];
                            }
                            return fScalar;
                        });
                const auto contributionTxV =
                        genArr<nR>([&](int i) { return contributionScalarV[i] * dxV[i]; });
                const auto contributionTyV =
                        genArr<nR>([&](int i) { return contributionScalarV[i] * dyV[i]; });
                const auto contributionTzV =
                        genArr<nR>([&](int i) { return contributionScalarV[i] * dzV[i]; });

                exactRespaNativeForceIXV[contributionIndex] =
                        genArr<nR>([&](int i)
                                   { return exactRespaNativeForceIXV[contributionIndex][i] + contributionTxV[i]; });
                exactRespaNativeForceIYV[contributionIndex] =
                        genArr<nR>([&](int i)
                                   { return exactRespaNativeForceIYV[contributionIndex][i] + contributionTyV[i]; });
                exactRespaNativeForceIZV[contributionIndex] =
                        genArr<nR>([&](int i)
                                   { return exactRespaNativeForceIZV[contributionIndex][i] + contributionTzV[i]; });

                real* nativeForce = exactRespaNativeForces[contributionIndex];
                if constexpr (c_numJClustersPerSimdRegister == 1)
                {
                    store(nativeForce + ajx, load<SimdReal>(nativeForce + ajx) - sumArray(contributionTxV));
                    store(nativeForce + ajy, load<SimdReal>(nativeForce + ajy) - sumArray(contributionTyV));
                    store(nativeForce + ajz, load<SimdReal>(nativeForce + ajz) - sumArray(contributionTzV));
                }
                else
                {
                    decr3Hsimd(nativeForce + aj * DIM,
                               sumArray(contributionTxV),
                               sumArray(contributionTyV),
                               sumArray(contributionTzV));
                }
            }
        }
    }
    else if (exactRespaCpuPairSplitLaunchActive(ic))
    {
        const SimdReal one_S(1.0_real);
        const SimdReal two_S(2.0_real);
        const SimdReal three_S(3.0_real);
        const auto exactRespaSwitchInV = [&](const SimdReal& r, const real off, const real on)
        {
            const SimdReal off_S(off);
            const SimdReal on_S(on);

            if (on <= off)
            {
                return blend(zero_S, one_S, on_S <= r);
            }

            const SimdReal switchFraction = (r - off_S) / (on_S - off_S);
            SimdReal       switched =
                    switchFraction * switchFraction * (three_S - two_S * switchFraction);
            switched                = blend(switched, zero_S, r <= off_S);
            switched                = blend(switched, one_S, on_S <= r);
            return switched;
        };

        const auto directWeightV = genArr<nR>(
                [&](int i)
                {
                    const SimdReal rV = rSquaredV[i] * rInvV[i];
                    switch (ic.exactRespaCpuPairSplit.contribution)
                    {
                        case MtsNonbondedRespaContribution::Inner:
                            if (ic.exactRespaCpuPairSplit.hasMiddle)
                            {
                                return one_S
                                       - exactRespaSwitchInV(rV,
                                                             ic.exactRespaCpuPairSplit.innerOff,
                                                             ic.exactRespaCpuPairSplit.innerOn);
                            }
                            return one_S
                                   - exactRespaSwitchInV(rV,
                                                         ic.exactRespaCpuPairSplit.outerOn,
                                                         ic.exactRespaCpuPairSplit.outerOff);
                        case MtsNonbondedRespaContribution::Middle:
                        {
                            if (!ic.exactRespaCpuPairSplit.hasMiddle)
                            {
                                return zero_S;
                            }
                            const SimdReal switchIntoMiddle = exactRespaSwitchInV(
                                    rV, ic.exactRespaCpuPairSplit.innerOff, ic.exactRespaCpuPairSplit.innerOn);
                            const SimdReal switchIntoOuter = exactRespaSwitchInV(
                                    rV, ic.exactRespaCpuPairSplit.outerOn, ic.exactRespaCpuPairSplit.outerOff);
                            return switchIntoMiddle * (one_S - switchIntoOuter);
                        }
                        case MtsNonbondedRespaContribution::Outer:
                            return exactRespaSwitchInV(
                                    rV, ic.exactRespaCpuPairSplit.outerOn, ic.exactRespaCpuPairSplit.outerOff);
                        case MtsNonbondedRespaContribution::Full: return one_S;
                        case MtsNonbondedRespaContribution::Count:
                            GMX_RELEASE_ASSERT(false, "Invalid exact r-RESPA CPU contribution");
                            return zero_S;
                    }

                    GMX_RELEASE_ASSERT(false, "Unhandled exact r-RESPA CPU contribution");
                    return zero_S;
                });

        const auto directForceScalarV = genArr<nR>(
                [&](int i)
                {
                    SimdReal directScalar = zero_S;
                    if constexpr (c_iLJInteractions != ILJInteractions::None)
                    {
                        if (i < c_nRLJ)
                        {
                            directScalar = directScalar + rInvSquaredV[i] * frLJV[i];
                        }
                    }
                    if constexpr (c_calculateCoulombInteractions)
                    {
                        directScalar = directScalar + rInvSquaredV[i] * qqV[i] * rInvExclV[i];
                    }
                    return directScalar;
                });

        const auto correctionForceScalarV = genArr<nR>(
                [&](int i)
                {
                    if constexpr (c_calculateCoulombInteractions)
                    {
                        return rInvSquaredV[i] * frCoulombV[i] - rInvSquaredV[i] * qqV[i] * rInvExclV[i];
                    }
                    return zero_S;
                });

        fScalarV = genArr<nR>(
                [&](int i)
                {
                    SimdReal fScalar = directWeightV[i] * directForceScalarV[i];
                    if (exactRespaCpuPairSplitAddsCorrection(ic))
                    {
                        fScalar = fScalar + correctionForceScalarV[i];
                    }
                    return fScalar;
                });
    }
    else
    {
        if constexpr (c_iLJInteractions != ILJInteractions::None)
        {
            if constexpr (c_calculateCoulombInteractions)
            {
                fScalarV = genArr<nR>(
                        [&](int i) {
                            return rInvSquaredV[i] * (i < c_nRLJ ? frCoulombV[i] + frLJV[i] : frCoulombV[i]);
                        });
            }
            else
            {
                // Note that here c_nRLJ=nR (otherwise this wouldn't compile)
                fScalarV = genArr<c_nRLJ>([&](int i) { return rInvSquaredV[i] * frLJV[i]; });
            }
        }
        else
        {
            fScalarV = genArr<nR>([&](int i) { return rInvSquaredV[i] * frCoulombV[i]; });
        }
    }

    if (!exactRespaNativeMultiActive)
    {
        /* Calculate temporary vectorial force */
        const auto txV = genArr<nR>([&](int i) { return fScalarV[i] * dxV[i]; });
        const auto tyV = genArr<nR>([&](int i) { return fScalarV[i] * dyV[i]; });
        const auto tzV = genArr<nR>([&](int i) { return fScalarV[i] * dzV[i]; });

        /* Increment i atom force */
        forceIXV = genArr<nR>([&](int i) { return forceIXV[i] + txV[i]; });
        forceIYV = genArr<nR>([&](int i) { return forceIYV[i] + tyV[i]; });
        forceIZV = genArr<nR>([&](int i) { return forceIZV[i] + tzV[i]; });

        /* Decrement j atom force */
        if constexpr (c_numJClustersPerSimdRegister == 1)
        {
            store(f + ajx, load<SimdReal>(f + ajx) - sumArray(txV));
            store(f + ajy, load<SimdReal>(f + ajy) - sumArray(tyV));
            store(f + ajz, load<SimdReal>(f + ajz) - sumArray(tzV));
        }
        else
        {
            decr3Hsimd(f + aj * DIM, sumArray(txV), sumArray(tyV), sumArray(tzV));
        }
    }
}

#endif // !DOXYGEN
