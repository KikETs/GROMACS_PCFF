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

/* When calculating RF or Ewald interactions we calculate the electrostatic
 * forces and energies on excluded atom pairs here in the non-bonded loops.
 */
#if defined CHECK_EXCLS && (defined CALC_COULOMB || defined LJ_EWALD)
#    define EXCL_FORCES
#endif

bool m2pPlain4x4MultiStepCoulombPairTraceEnabled();
void noteM2pPlain4x4MultiStepCoulombPairContribution(int         pairI,
                                                     int         pairJ,
                                                     int         energyIndex,
                                                     int         shiftIndex,
                                                     real        coordIX,
                                                     real        coordIY,
                                                     real        coordIZ,
                                                     real        coordJX,
                                                     real        coordJY,
                                                     real        coordJZ,
                                                     real        shiftX,
                                                     real        shiftY,
                                                     real        shiftZ,
                                                     real        dx,
                                                     real        dy,
                                                     real        dz,
                                                     real        rsq,
                                                     int         clusterI,
                                                     int         clusterJ,
                                                     int         localI,
                                                     int         localJ,
                                                     real        qq,
                                                     real        interact,
                                                     real        rinv,
                                                     int         tableIndex,
                                                     real        frac,
                                                     real        fexcl,
                                                     real        vcorr,
                                                     real        vcoul,
                                                     const char* codeLocation);
bool m2pPlain4x4RealspaceForceSubcomponentTraceEnabled();
void noteM2pPlain4x4RealspaceForceSubcomponents(int  ai,
                                                int  aj,
                                                real ljFx,
                                                real ljFy,
                                                real ljFz,
                                                real coulombSrFx,
                                                real coulombSrFy,
                                                real coulombSrFz,
                                                real exclusionCorrectionFx,
                                                real exclusionCorrectionFy,
                                                real exclusionCorrectionFz,
                                                real combinedFx,
                                                real combinedFy,
                                                real combinedFz);
bool m2pPlain4x4ExclusionEquivalenceTraceEnabled();
void noteM2pPlain4x4ExclusionEquivalencePair(int         ai,
                                             int         aj,
                                             real        interact,
                                             real        excludedMask,
                                             real        skipmask,
                                             real        qq,
                                             int         tableIndex,
                                             real        frac,
                                             real        fexcl,
                                             real        vcorr,
                                             real        correctionScalarUnmasked,
                                             real        correctionScalarEffective,
                                             real        correctionForceUnmaskedFx,
                                             real        correctionForceUnmaskedFy,
                                             real        correctionForceUnmaskedFz,
                                             real        correctionForceEffectiveFx,
                                             real        correctionForceEffectiveFy,
                                             real        correctionForceEffectiveFz,
                                             real        combinedForceFx,
                                             real        combinedForceFy,
                                             real        combinedForceFz,
                                             const char* sinkTarget,
                                             bool        sinkWriteExecuted,
                                             const char* codeLocation);
void noteM2pPlain4x4Step2PairTotal(int         ai,
                                   int         aj,
                                   real        r,
                                   real        rawLjScalar,
                                   real        bareCoulombScalar,
                                   real        correctionScalar,
                                   real        ljFx,
                                   real        ljFy,
                                   real        ljFz,
                                   real        coulombFx,
                                   real        coulombFy,
                                   real        coulombFz,
                                   real        correctionFx,
                                   real        correctionFy,
                                   real        correctionFz,
                                   real        totalFx,
                                   real        totalFy,
                                   real        totalFz,
                                   real        qq,
                                   real        rinv,
                                   int         tableIndex,
                                   real        frac,
                                   real        fexcl,
                                   real        vcorr,
                                   const char* codeLocation);

{
    const int cj = l_cj[cjind].cj;

#if UNROLLI == 1
    constexpr int i = 0;
#else
    for (int i = 0; i < UNROLLI; i++)
#endif
    {
        const int ai = ci * UNROLLI + i;

        const int gmx_unused type_i_off = type[ai] * ntype2;

#if UNROLLJ == 1
        constexpr int j = 0;
#else
        for (int j = 0; j < UNROLLJ; j++)
#endif
        {
            real FrLJ6 = 0, FrLJ12 = 0, frLJ = 0;
            const real repulsionPower = static_cast<real>(ic.vdw.repulsionPower);

            /* A multiply mask used to zero an interaction
             * when either the distance cutoff is exceeded, or
             * (if appropriate) the i and j indices are
             * unsuitable for this kind of inner loop. */
#ifdef CHECK_EXCLS
            /* A multiply mask used to zero an interaction
             * when that interaction should be excluded
             * (e.g. because of bonding). */
            const real interact = static_cast<real>((l_cj[cjind].excl >> (i * UNROLLI + j)) & 1);
            const real excludedMask = 1.0 - interact;
#    ifndef EXCL_FORCES
            real skipmask = interact;
#    else
            real skipmask = (cj == ci_sh && j <= i) ? 0.0 : 1.0;
#    endif
#else
            constexpr real interact = 1.0;
            constexpr real excludedMask = 0.0;
            real           skipmask = interact;
#endif

            real gmx_unused VLJ = 0;

            const int aj = cj * UNROLLJ + j;

            const real dx = xi[i * XI_STRIDE + XX] - x[aj * X_STRIDE + XX];
            const real dy = xi[i * XI_STRIDE + YY] - x[aj * X_STRIDE + YY];
            const real dz = xi[i * XI_STRIDE + ZZ] - x[aj * X_STRIDE + ZZ];

            real rsq = dx * dx + dy * dy + dz * dz;

            /* Prepare to enforce the cut-off. */
            skipmask = (rsq >= rcut2) ? 0 : skipmask;
            /* 9 flops for r^2 + cut-off check */

            // Ensure the distances do not fall below the limit where r^-12 overflows.
            // This should never happen for normal interactions.
            rsq = std::max(rsq, c_nbnxnMinDistanceSquared);

#ifdef COUNT_PAIRS
            npair++;
#endif

            real rinv = gmx::invsqrt(rsq);
            /* 5 flops for invsqrt */

            /* Partially enforce the cut-off (and perhaps
             * exclusions) to avoid possible overflow of
             * rinvsix when computing LJ, and/or overflowing
             * the Coulomb table during lookup. */
            rinv = rinv * skipmask;

            const real rinvsq = rinv * rinv;

#ifdef ENERGY_GROUPS
            const int egpJ = nbatParams.energyGroupsPerCluster->getEnergyGroup(cj, j);
#endif

#ifdef HALF_LJ
            if (UNROLLI > 1 && i < UNROLLI / 2)
#endif
            {
                const real c6  = nbfp[type_i_off + type[aj] * 2];
                const real c12 = nbfp[type_i_off + type[aj] * 2 + 1];

#if defined LJ_CUT || defined LJ_FORCE_SWITCH || defined LJ_POT_SWITCH
                real rinvsix = interact * rinvsq * rinvsq * rinvsq;
                FrLJ6        = c6 * rinvsix;
                FrLJ12       = c12
                         * (repulsionPower == 12.0 ? rinvsix * rinvsix
                                                   : interact * std::pow(rinv, repulsionPower));
                frLJ         = FrLJ12 - FrLJ6;
                /* 7 flops for r^-2 + LJ force */
#    if defined CALC_ENERGIES || defined LJ_POT_SWITCH
                VLJ = (FrLJ12 + interact * c12 * ic.vdw.repulsionShift.cpot) / repulsionPower
                      - (FrLJ6 + interact * c6 * ic.vdw.dispersionShift.cpot) / 6;
                /* 7 flops for LJ energy */
#    endif
#endif

#if defined LJ_FORCE_SWITCH || defined LJ_POT_SWITCH
                /* Force or potential switching from ic.rvdw_switch */
                real r   = rsq * rinv;
                real rsw = r - ic.vdw.switchDistance;
                rsw      = (rsw >= 0.0 ? rsw : 0.0);
#endif
#ifdef LJ_FORCE_SWITCH
                frLJ += -c6 * (ic.vdw.dispersionShift.c2 + ic.vdw.dispersionShift.c3 * rsw) * rsw * rsw * r
                        + c12 * (ic.vdw.repulsionShift.c2 + ic.vdw.repulsionShift.c3 * rsw) * rsw * rsw * r;
#    if defined CALC_ENERGIES
                VLJ += -c6 * (-ic.vdw.dispersionShift.c2 / 3 - ic.vdw.dispersionShift.c3 / 4 * rsw)
                               * rsw * rsw * rsw
                       + c12 * (-ic.vdw.repulsionShift.c2 / 3 - ic.vdw.repulsionShift.c3 / 4 * rsw)
                                 * rsw * rsw * rsw;
#    endif
#endif

#if defined CALC_ENERGIES || defined LJ_POT_SWITCH
                /* Masking should be done after force switching,
                 * but before potential switching.
                 */
                /* Need to zero the interaction if there should be exclusion. */
                const real gmx_unused m2wRawLjTerm = VLJ;
                VLJ = VLJ * interact;
#endif

#ifdef LJ_POT_SWITCH
                {
                    const real sw  = 1.0 + (swV3 + (swV4 + swV5 * rsw) * rsw) * rsw * rsw * rsw;
                    const real dsw = (swF2 + (swF3 + swF4 * rsw) * rsw) * rsw * rsw;

                    frLJ = frLJ * sw - r * VLJ * dsw;
                    VLJ *= sw;
                }
#endif

#ifdef LJ_EWALD
                {
#    ifdef LJ_EWALD_COMB_GEOM
                    const real c6grid = ljc[type[ai] * 2] * ljc[type[aj] * 2];
#    elif defined LJ_EWALD_COMB_LB
                    real c6grid = NAN;
                    {
                        /* These sigma and epsilon are scaled to give 6*C6 */
                        const real sigma   = ljc[type[ai] * 2] + ljc[type[aj] * 2];
                        const real epsilon = ljc[type[ai] * 2 + 1] * ljc[type[aj] * 2 + 1];

                        const real sigma2 = sigma * sigma;
                        c6grid            = epsilon * sigma2 * sigma2 * sigma2;
                    }
#    else
#        error "No LJ Ewald combination rule defined"
#    endif

#    ifdef CHECK_EXCLS
                    /* Recalculate rinvsix without exclusion mask */
                    const real rinvsix_nm = rinvsq * rinvsq * rinvsq;
#    else
                    const real rinvsix_nm = rinvsix;
#    endif
                    const real cr2 = lje_coeff2 * rsq;
#    if GMX_DOUBLE
                    const real expmcr2 = exp(-cr2);
#    else
                    const real expmcr2 = expf(-cr2);
#    endif
                    const real poly = 1 + cr2 + 0.5 * cr2 * cr2;

                    /* Subtract the grid force from the total LJ force */
                    frLJ += c6grid * (rinvsix_nm - expmcr2 * (rinvsix_nm * poly + lje_coeff6_6));
#    ifdef CALC_ENERGIES
                    /* Shift should only be applied to real LJ pairs */
                    const real sh_mask = lje_vc * interact;

                    VLJ += c6grid / 6 * (rinvsix_nm * (1 - expmcr2 * poly) + sh_mask);
#    endif
                }
#endif /* LJ_EWALD */

#ifdef VDW_CUTOFF_CHECK
                /* Mask for VdW cut-off shorter than Coulomb cut-off */
                const real gmx_unused m2wCutoffMask = (rsq < rvdw2) ? 1.0 : 0.0;
                {
                    real skipmask_rvdw = (rsq < rvdw2) ? 1.0 : 0.0;
                    frLJ *= skipmask_rvdw;
#    ifdef CALC_ENERGIES
                    VLJ *= skipmask_rvdw;
#    endif
                }
#else
#    if defined CALC_ENERGIES
                /* Need to zero the interaction if r >= rcut */
                const real gmx_unused m2wCutoffMask = skipmask;
                VLJ = VLJ * skipmask;
                /* 1 more flop for LJ energy */
#    endif
#endif /* VDW_CUTOFF_CHECK */


#ifdef CALC_ENERGIES
#    ifdef GMX_PCFF_RESPA_M2Q_PLAIN_RAW_TRACE_ENABLED
                if (m2qPlainEarliestRawStageEnabled)
                {
                    m2qPlainEarliestRawLjLocal += VLJ;
                }
#    endif
#    ifdef GMX_PCFF_RESPA_M2R_PLAIN_TRACE_ENABLED
                if (m2rPlainKernelLocalStageEnabled)
                {
                    m2rPlainKernelLocalLjLocal += VLJ;
                }
#    endif
#    ifdef GMX_PCFF_RESPA_M2V_PLAIN_TRACE_ENABLED
                if (m2vPlain4x4AlignedEventTraceEnabled() && VLJ != 0.0)
                {
                    noteM2vPlain4x4AlignedEvent(VLJ);
                }
#    endif
#    ifdef GMX_PCFF_RESPA_M2W_PLAIN_TRACE_ENABLED
                if (m2wPlain4x4AlignedEventTraceEnabled() && VLJ != 0.0)
                {
                    noteM2wPlain4x4AlignedEvent(ai,
                                                aj,
                                                type[ai],
                                                type[aj],
                                                ci,
                                                cj,
                                                i,
                                                j,
                                                c6,
                                                c12,
                                                rsq,
                                                rsq * rinv,
                                                m2wRawLjTerm,
                                                interact * m2wCutoffMask,
                                                VLJ);
                }
#    endif
#    ifdef GMX_PCFF_RESPA_M2X_PLAIN_TRACE_ENABLED
                if (m2xPlain4x4GeometryTraceEnabled() && VLJ != 0.0)
                {
                    M2xPlain4x4GeometryEventData m2xData;
                    m2xData.pairI          = ai;
                    m2xData.pairJ          = aj;
                    m2xData.typeI          = type[ai];
                    m2xData.typeJ          = type[aj];
                    m2xData.ciIndex        = ci;
                    m2xData.cjIndex        = cj;
                    m2xData.iIndex         = i;
                    m2xData.jIndex         = j;
                    m2xData.shiftIndex     = ish;
                    m2xData.coordISourceX  = x[(ci * UNROLLI + i) * X_STRIDE + XX];
                    m2xData.coordISourceY  = x[(ci * UNROLLI + i) * X_STRIDE + YY];
                    m2xData.coordISourceZ  = x[(ci * UNROLLI + i) * X_STRIDE + ZZ];
                    m2xData.coordJSourceX  = x[aj * X_STRIDE + XX];
                    m2xData.coordJSourceY  = x[aj * X_STRIDE + YY];
                    m2xData.coordJSourceZ  = x[aj * X_STRIDE + ZZ];
                    m2xData.shiftX         = shiftvec[ishf + XX];
                    m2xData.shiftY         = shiftvec[ishf + YY];
                    m2xData.shiftZ         = shiftvec[ishf + ZZ];
                    m2xData.coordIShiftedX = xi[i * XI_STRIDE + XX];
                    m2xData.coordIShiftedY = xi[i * XI_STRIDE + YY];
                    m2xData.coordIShiftedZ = xi[i * XI_STRIDE + ZZ];
                    m2xData.dx             = dx;
                    m2xData.dy             = dy;
                    m2xData.dz             = dz;
                    m2xData.rsq            = rsq;
                    m2xData.r              = rsq * rinv;
                    m2xData.rawLjTerm      = m2wRawLjTerm;
                    m2xData.finalEventLj   = VLJ;
                    noteM2xPlain4x4GeometryEvent(m2xData);
                }
#    endif
#    ifdef ENERGY_GROUPS
                if (m2pPlain4x4LjContractReplayEnabled() && VLJ != 0.0)
                {
                    noteM2pPlain4x4LjContractReplayPairContribution(VLJ);
                }
                Vvdw[egp_sh_i[i] + egpJ] += VLJ;
#        ifdef GMX_PCFF_RESPA_M2S_PLAIN_TRACE_ENABLED
                if (m2sPlain4x4InternalTraceEnabled() && VLJ != 0.0)
                {
                    noteM2sPlain4x4FirstWriteTargetTotal(
                            Vvdw, nbatParams.numEnergyGroups * nbatParams.numEnergyGroups);
                }
#        endif
#        ifdef GMX_PCFF_RESPA_M2U_PLAIN_TRACE_ENABLED
                if (m2uPlain4x4WriteOrdinalTraceEnabled() && VLJ != 0.0)
                {
                    noteM2uPlain4x4WriteTargetTotal(
                            Vvdw, nbatParams.numEnergyGroups * nbatParams.numEnergyGroups);
                }
#        endif
#    else
                if (m2pPlain4x4LjContractReplayEnabled() && VLJ != 0.0)
                {
                    noteM2pPlain4x4LjContractReplayPairContribution(VLJ);
                }
                Vvdw_ci += VLJ;
#        ifdef GMX_PCFF_RESPA_M2S_PLAIN_TRACE_ENABLED
                if (m2sPlain4x4InternalTraceEnabled() && VLJ != 0.0)
                {
                    noteM2sPlain4x4FirstWriteTargetTotal(&Vvdw_ci, 1);
                }
#        endif
                /* 1 flop for LJ energy addition */
#    endif
#endif
            }

#ifdef CALC_COULOMB
            /* Enforce the cut-off and perhaps exclusions. In
             * those cases, rinv is zero because of skipmask,
             * but fcoul and vcoul will later be non-zero (in
             * both RF and table cases) because of the
             * contributions that do not depend on rinv. These
             * contributions cannot be allowed to accumulate
             * to the force and potential, and the easiest way
             * to do this is to zero the charges in
             * advance. */
            const real qq = skipmask * qi[i] * q[aj];

#    ifdef CALC_COUL_RF
            real fcoul = qq * (interact * (rinv * rinvsq - k_rf2));
            /* 4 flops for RF force */
#        ifdef CALC_ENERGIES
            real vcoul = qq * (interact * rinv + reactionFieldCoefficient * rsq - reactionFieldShift);
            /* 4 flops for RF energy */
#        endif
#    endif

#    ifdef CALC_COUL_TAB
            const real rs   = rsq * rinv * tab_coul_scale;
            const int  ri   = int(rs);
            const real frac = rs - static_cast<real>(ri);
#        if !GMX_DOUBLE
            /* fexcl = F_i + frac * (F_(i+1)-F_i) */
            const real fexcl = tab_coul_FDV0[ri * 4] + frac * tab_coul_FDV0[ri * 4 + 1];
#        else
            /* fexcl = (1-frac) * F_i + frac * F_(i+1) */
            const real fexcl = (1 - frac) * tab_coul_F[ri] + frac * tab_coul_F[ri + 1];
#        endif
            real fcoul = interact * rinvsq - fexcl;
            /* 7 flops for float 1/r-table force */
#        ifdef CALC_ENERGIES
#            if !GMX_DOUBLE
            const real vcorr = tab_coul_FDV0[ri * 4 + 2] - halfsp * frac * (tab_coul_FDV0[ri * 4] + fexcl);
            const real vcoulUnmasked = qq * (rinv - ic.coulomb.ewaldShift - vcorr);
            real vcoul = qq * (interact * (rinv - ic.coulomb.ewaldShift) - vcorr);
            /* 7 flops for float 1/r-table energy (8 with excls) */
#            else
            const real vcorr = tab_coul_V[ri] - halfsp * frac * (tab_coul_F[ri] + fexcl);
            const real vcoulUnmasked = qq * (rinv - ic.coulomb.ewaldShift - vcorr);
            real vcoul = qq * (interact * (rinv - ic.coulomb.ewaldShift) - vcorr);
#            endif
            if (m2pPlain4x4CoulombProducerTraceEnabled() && excludedMask != 0.0 && vcoulUnmasked != 0.0)
            {
#        ifdef ENERGY_GROUPS
                const int coulProducerEnergyIndex = egp_sh_i[i] + egpJ;
#        else
                const int coulProducerEnergyIndex = 0;
#        endif
                noteM2pPlain4x4CoulombProducer(ai,
                                               aj,
                                               coulProducerEnergyIndex,
                                               excludedMask,
                                               qq,
                                               interact,
                                               rinv,
                                               ic.coulomb.ewaldShift,
                                               ri,
                                               frac,
                                               fexcl,
                                               vcorr,
                                               vcoul,
                                               vcoulUnmasked,
                                               "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:380");
            }
#        endif
            fcoul *= qq * rinv;
#    endif

#    ifdef CALC_ENERGIES
#        ifdef ENERGY_GROUPS
            const int coulEnergyIndex = egp_sh_i[i] + egpJ;
#            ifdef CALC_COUL_TAB
            if (m2pPlain4x4MultiStepCoulombPairTraceEnabled() && vcoul != 0.0)
            {
                noteM2pPlain4x4MultiStepCoulombPairContribution(ai,
                                                                aj,
                                                                coulEnergyIndex,
                                                                ish,
                                                                x[(ci * UNROLLI + i) * X_STRIDE + XX],
                                                                x[(ci * UNROLLI + i) * X_STRIDE + YY],
                                                                x[(ci * UNROLLI + i) * X_STRIDE + ZZ],
                                                                x[aj * X_STRIDE + XX],
                                                                x[aj * X_STRIDE + YY],
                                                                x[aj * X_STRIDE + ZZ],
                                                                shiftvec[ishf + XX],
                                                                shiftvec[ishf + YY],
                                                                shiftvec[ishf + ZZ],
                                                                dx,
                                                                dy,
                                                                dz,
                                                                rsq,
                                                                ci,
                                                                cj,
                                                                i,
                                                                j,
                                                                qq,
                                                                interact,
                                                                rinv,
                                                                ri,
                                                                frac,
                                                                fexcl,
                                                                vcorr,
                                                                vcoul,
                                                                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:431");
            }
#            endif
            if (m2pPlain4x4CoulombContractReplayEnabled() && vcoul != 0.0)
            {
                noteM2pPlain4x4CoulombContractReplayPairContribution(coulEnergyIndex, vcoul);
            }
            if (m2pPlain4x4CoulombFirstWriteTraceEnabled() && vcoul != 0.0)
            {
                const real targetBefore = Vc[coulEnergyIndex];
                const real targetAfter  = targetBefore + vcoul;
                noteM2pPlain4x4CoulombFirstWrite(targetBefore,
                                                 vcoul,
                                                 targetAfter,
                                                 coulEnergyIndex,
                                                 "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:396");
            }
            Vc[coulEnergyIndex] += vcoul;
#        else
#            ifdef CALC_COUL_TAB
            if (m2pPlain4x4MultiStepCoulombPairTraceEnabled() && vcoul != 0.0)
            {
                noteM2pPlain4x4MultiStepCoulombPairContribution(ai,
                                                                aj,
                                                                0,
                                                                ish,
                                                                x[(ci * UNROLLI + i) * X_STRIDE + XX],
                                                                x[(ci * UNROLLI + i) * X_STRIDE + YY],
                                                                x[(ci * UNROLLI + i) * X_STRIDE + ZZ],
                                                                x[aj * X_STRIDE + XX],
                                                                x[aj * X_STRIDE + YY],
                                                                x[aj * X_STRIDE + ZZ],
                                                                shiftvec[ishf + XX],
                                                                shiftvec[ishf + YY],
                                                                shiftvec[ishf + ZZ],
                                                                dx,
                                                                dy,
                                                                dz,
                                                                rsq,
                                                                ci,
                                                                cj,
                                                                i,
                                                                j,
                                                                qq,
                                                                interact,
                                                                rinv,
                                                                ri,
                                                                frac,
                                                                fexcl,
                                                                vcorr,
                                                                vcoul,
                                                                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:449");
            }
#            endif
            if (m2pPlain4x4CoulombContractReplayEnabled() && vcoul != 0.0)
            {
                noteM2pPlain4x4CoulombContractReplayPairContribution(0, vcoul);
            }
            Vc_ci += vcoul;
            /* 1 flop for Coulomb energy addition */
#        endif
#    endif
#endif

#ifdef CALC_COULOMB
            const real rForRespa = rsq * rinv;
            /* 2 flops for scalar LJ+Coulomb force if !HALF_LJ || (i < UNROLLI / 2) */
#    ifdef HALF_LJ
            const real ljScalarUnsplit = (i < UNROLLI / 2) ? frLJ * rinvsq : 0;
#    else
            const real ljScalarUnsplit = frLJ * rinvsq;
#    endif
            const real totalCoulombScalar = fcoul;
            const real directCoulombScalar = interact * qq * rinvsq * rinv;
            const real correctionScalar = totalCoulombScalar - directCoulombScalar;
            real       fscal = ljScalarUnsplit + totalCoulombScalar;
            if (exactRespaCpuPairSplitLaunchActive(ic))
            {
                const real directWeight = exactRespaCpuPairSplitWeight(ic, rForRespa);
                fscal = directWeight * (ljScalarUnsplit + directCoulombScalar)
                        + (exactRespaCpuPairSplitAddsCorrection(ic) ? correctionScalar : 0.0_real);
            }
#else
            const real rForRespa = rsq * rinv;
            const real ljScalarUnsplit = frLJ * rinvsq;
            real       fscal           = ljScalarUnsplit;
            if (exactRespaCpuPairSplitLaunchActive(ic))
            {
                fscal = exactRespaCpuPairSplitWeight(ic, rForRespa) * ljScalarUnsplit;
            }
#endif
            const real fx = fscal * dx;
            const real fy = fscal * dy;
            const real fz = fscal * dz;

            if (!exactRespaNativeMultiActive && m2pPlain4x4RealspaceForceSubcomponentTraceEnabled())
            {
                real ljScalar = ljScalarUnsplit;
#ifdef CALC_COULOMB
                real coulombSrScalar          = directCoulombScalar;
                real exclusionCorrectionScalar = correctionScalar;
                if (exactRespaCpuPairSplitLaunchActive(ic))
                {
                    const real directWeight = exactRespaCpuPairSplitWeight(ic, rForRespa);
                    ljScalar *= directWeight;
                    coulombSrScalar *= directWeight;
                    exclusionCorrectionScalar =
                            exactRespaCpuPairSplitAddsCorrection(ic) ? exclusionCorrectionScalar : 0.0_real;
                }
#else
                const real coulombSrScalar          = 0;
                const real exclusionCorrectionScalar = 0;
#endif
	                noteM2pPlain4x4RealspaceForceSubcomponents(ai,
	                                                            aj,
	                                                            ljScalar * dx,
                                                            ljScalar * dy,
                                                            ljScalar * dz,
                                                            coulombSrScalar * dx,
                                                            coulombSrScalar * dy,
                                                            coulombSrScalar * dz,
                                                            exclusionCorrectionScalar * dx,
                                                            exclusionCorrectionScalar * dy,
                                                            exclusionCorrectionScalar * dz,
	                                                            fx,
	                                                            fy,
		                                                            fz);

	                if (m2pPlain4x4RealspaceForceSubcomponentTraceEnabled())
	                {
	                    const real plainRawLjScalar = 
#ifdef HALF_LJ
	                            (i < UNROLLI / 2) ? frLJ : 0;
#else
	                            frLJ;
#endif
#ifdef CALC_COULOMB
#    ifdef CALC_COUL_TAB
	                    real plainBareCoulombScalar = directCoulombScalar / rinvsq;
	                    real plainCorrectionScalar  = (rinv != 0.0) ? (correctionScalar / rinvsq) : 0.0;
	                    if (exactRespaCpuPairSplitLaunchActive(ic))
	                    {
	                        const real directWeight = exactRespaCpuPairSplitWeight(ic, rForRespa);
	                        plainBareCoulombScalar *= directWeight;
	                        plainCorrectionScalar =
	                                exactRespaCpuPairSplitAddsCorrection(ic) ? plainCorrectionScalar : 0.0_real;
	                    }
	                    const real plainQq               = qq;
	                    const int  plainTableIndex        = ri;
	                    const real plainFrac              = frac;
	                    const real plainFexcl             = fexcl;
#        ifdef CALC_ENERGIES
	                    const real plainVcorr             = vcorr;
#        else
	                    const real plainVcorr             = 0.0;
#        endif
#    else
	                    const real plainBareCoulombScalar = 0.0;
	                    const real plainCorrectionScalar  = 0.0;
	                    const real plainQq               = qq;
	                    const int  plainTableIndex        = -1;
	                    const real plainFrac              = 0.0;
	                    const real plainFexcl             = 0.0;
	                    const real plainVcorr             = 0.0;
#    endif
#else
	                    const real plainBareCoulombScalar = 0.0;
	                    const real plainCorrectionScalar  = 0.0;
	                    const real plainQq               = 0.0;
	                    const int  plainTableIndex        = -1;
	                    const real plainFrac              = 0.0;
	                    const real plainFexcl             = 0.0;
	                    const real plainVcorr             = 0.0;
#endif
	                    const real rForTrace = (rinv != 0.0) ? (rsq * rinv) : 0.0;

	                    noteM2pPlain4x4Step2PairTotal(ai,
	                                                  aj,
	                                                  rForTrace,
	                                                  plainRawLjScalar,
	                                                  plainBareCoulombScalar,
	                                                  plainCorrectionScalar,
	                                                  ljScalar * dx,
	                                                  ljScalar * dy,
	                                                  ljScalar * dz,
	                                                  coulombSrScalar * dx,
	                                                  coulombSrScalar * dy,
	                                                  coulombSrScalar * dz,
	                                                  exclusionCorrectionScalar * dx,
	                                                  exclusionCorrectionScalar * dy,
	                                                  exclusionCorrectionScalar * dz,
	                                                  fx,
	                                                  fy,
	                                                  fz,
	                                                  plainQq,
	                                                  rinv,
	                                                  plainTableIndex,
	                                                  plainFrac,
	                                                  plainFexcl,
	                                                  plainVcorr,
	                                                  "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:plain_step2_pair_total_trace");
	                }
	            }

	            if (!exactRespaNativeMultiActive && m2pPlain4x4ExclusionEquivalenceTraceEnabled())
	            {
#ifdef CALC_COULOMB
#    ifdef CALC_COUL_TAB
	                const real correctionScalarUnmasked = correctionScalar;
	                real       correctionScalarEffective = correctionScalar;
	                if (exactRespaCpuPairSplitLaunchActive(ic) && !exactRespaCpuPairSplitAddsCorrection(ic))
	                {
	                    correctionScalarEffective = 0.0_real;
	                }
	                const real correctionQq              = qq;
	                const int  correctionTableIndex      = ri;
	                const real correctionFrac            = frac;
	                const real correctionFexcl           = fexcl;
#        ifdef CALC_ENERGIES
	                const real correctionVcorr           = vcorr;
#        else
	                const real correctionVcorr           = 0;
#        endif
#    else
	                const real correctionScalarUnmasked = 0;
	                const real correctionScalarEffective = 0;
	                const real correctionQq              = qq;
	                const int  correctionTableIndex      = -1;
	                const real correctionFrac            = 0;
	                const real correctionFexcl           = 0;
	                const real correctionVcorr           = 0;
#    endif
#else
	                const real correctionScalarUnmasked = 0;
	                const real correctionScalarEffective = 0;
	                const real correctionQq              = 0;
	                const int  correctionTableIndex      = -1;
	                const real correctionFrac            = 0;
	                const real correctionFexcl           = 0;
	                const real correctionVcorr           = 0;
#endif
	                noteM2pPlain4x4ExclusionEquivalencePair(ai,
	                                                        aj,
	                                                        interact,
	                                                        excludedMask,
	                                                        skipmask,
	                                                        correctionQq,
	                                                        correctionTableIndex,
	                                                        correctionFrac,
	                                                        correctionFexcl,
	                                                        correctionVcorr,
	                                                        correctionScalarUnmasked,
	                                                        correctionScalarEffective,
	                                                        correctionScalarUnmasked * dx,
	                                                        correctionScalarUnmasked * dy,
	                                                        correctionScalarUnmasked * dz,
	                                                        correctionScalarEffective * dx,
	                                                        correctionScalarEffective * dy,
	                                                        correctionScalarEffective * dz,
	                                                        fx,
	                                                        fy,
	                                                        fz,
	                                                        "nbat_force_array_via_fscal",
	                                                        correctionScalarEffective != 0.0,
	                                                        "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:plain_exclusion_equivalence_trace");
	            }

            if (exactRespaNativeMultiActive)
            {
                const bool twoContributionInnerMiddle =
                        exactRespaNativeContributionCount == 2
                        && gmx::exactRespaCpuPairSplitNativeMultiContribution(ic, 0)
                                   == MtsNonbondedRespaContribution::Inner
                        && gmx::exactRespaCpuPairSplitNativeMultiContribution(ic, 1)
                                   == MtsNonbondedRespaContribution::Middle;
                if (twoContributionInnerMiddle
                    && gmx::exactRespaCpuPairSplitNativeMultiTwoContributionFastPathEnabled())
                {
                    const real nativeDirectForceScalar =
#ifdef CALC_COULOMB
                            ljScalarUnsplit + directCoulombScalar;
#else
                            ljScalarUnsplit;
#endif
                    const real innerFscal =
                            gmx::exactRespaCpuPairSplitWeightForContribution(
                                    ic, MtsNonbondedRespaContribution::Inner, rForRespa)
                            * nativeDirectForceScalar
                            + (gmx::exactRespaCpuPairSplitContributionAddsCorrection(
                                       MtsNonbondedRespaContribution::Inner)
                                           ?
#ifdef CALC_COULOMB
                                                   correctionScalar
#else
                                                   0.0_real
#endif
                                           : 0.0_real);
                    const real middleFscal =
                            gmx::exactRespaCpuPairSplitWeightForContribution(
                                    ic, MtsNonbondedRespaContribution::Middle, rForRespa)
                            * nativeDirectForceScalar
                            + (gmx::exactRespaCpuPairSplitContributionAddsCorrection(
                                       MtsNonbondedRespaContribution::Middle)
                                           ?
#ifdef CALC_COULOMB
                                                   correctionScalar
#else
                                                   0.0_real
#endif
                                           : 0.0_real);
                    const real innerFx = innerFscal * dx;
                    const real innerFy = innerFscal * dy;
                    const real innerFz = innerFscal * dz;
                    const real middleFx = middleFscal * dx;
                    const real middleFy = middleFscal * dy;
                    const real middleFz = middleFscal * dz;
                    exactRespaNativeFi[0][i * FI_STRIDE + XX] += innerFx;
                    exactRespaNativeFi[0][i * FI_STRIDE + YY] += innerFy;
                    exactRespaNativeFi[0][i * FI_STRIDE + ZZ] += innerFz;
                    exactRespaNativeForces[0][aj * F_STRIDE + XX] -= innerFx;
                    exactRespaNativeForces[0][aj * F_STRIDE + YY] -= innerFy;
                    exactRespaNativeForces[0][aj * F_STRIDE + ZZ] -= innerFz;
                    exactRespaNativeFi[1][i * FI_STRIDE + XX] += middleFx;
                    exactRespaNativeFi[1][i * FI_STRIDE + YY] += middleFy;
                    exactRespaNativeFi[1][i * FI_STRIDE + ZZ] += middleFz;
                    exactRespaNativeForces[1][aj * F_STRIDE + XX] -= middleFx;
                    exactRespaNativeForces[1][aj * F_STRIDE + YY] -= middleFy;
                    exactRespaNativeForces[1][aj * F_STRIDE + ZZ] -= middleFz;
                }
                else
                {
                    for (int contributionIndex = 0;
                         contributionIndex < exactRespaNativeContributionCount;
                         ++contributionIndex)
                    {
                        const auto contribution =
                                gmx::exactRespaCpuPairSplitNativeMultiContribution(ic, contributionIndex);
#ifdef CALC_COULOMB
                        const real directWeight =
                                gmx::exactRespaCpuPairSplitWeightForContribution(ic, contribution, rForRespa);
                        const real contributionFscal =
                                directWeight * (ljScalarUnsplit + directCoulombScalar)
                                + (gmx::exactRespaCpuPairSplitContributionAddsCorrection(contribution)
                                           ? correctionScalar
                                           : 0.0_real);
#else
                        const real contributionFscal =
                                gmx::exactRespaCpuPairSplitWeightForContribution(ic, contribution, rForRespa)
                                * ljScalarUnsplit;
#endif
                        const real contributionFx = contributionFscal * dx;
                        const real contributionFy = contributionFscal * dy;
                        const real contributionFz = contributionFscal * dz;
                        exactRespaNativeFi[contributionIndex][i * FI_STRIDE + XX] += contributionFx;
                        exactRespaNativeFi[contributionIndex][i * FI_STRIDE + YY] += contributionFy;
                        exactRespaNativeFi[contributionIndex][i * FI_STRIDE + ZZ] += contributionFz;
                        exactRespaNativeForces[contributionIndex][aj * F_STRIDE + XX] -= contributionFx;
                        exactRespaNativeForces[contributionIndex][aj * F_STRIDE + YY] -= contributionFy;
                        exactRespaNativeForces[contributionIndex][aj * F_STRIDE + ZZ] -= contributionFz;
                    }
                }
            }
            else
            {
                /* Increment i-atom force */
                fi[i * FI_STRIDE + XX] += fx;
                fi[i * FI_STRIDE + YY] += fy;
                fi[i * FI_STRIDE + ZZ] += fz;
                /* Decrement j-atom force */
                f[aj * F_STRIDE + XX] -= fx;
                f[aj * F_STRIDE + YY] -= fy;
                f[aj * F_STRIDE + ZZ] -= fz;
                /* 9 flops for force addition */
            }
        }
    }
}

#undef interact
#undef EXCL_FORCES
