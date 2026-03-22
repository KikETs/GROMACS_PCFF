/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 1991- The GROMACS Authors
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
#include "gmxpre.h"

#include "config.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gromacs/applied_forces/awh/awh.h"
#include "gromacs/domdec/dlbtiming.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/gpuhaloexchange.h"
#include "gromacs/domdec/haloexchange.h"
#include "gromacs/domdec/partition.h"
#include "gromacs/essentialdynamics/edsam.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/ewald/pme_coordinate_receiver_gpu.h"
#include "gromacs/ewald/pme_pp.h"
#include "gromacs/ewald/pme_pp_comm_gpu.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/imd/imd.h"
#include "gromacs/listed_forces/disre.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/listed_forces/orires.h"
#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/mdlib/calcmu.h"
#include "gromacs/mdlib/calcvir.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/dispersioncorrection.h"
#include "gromacs/mdlib/enerdata_utils.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdlib/forcerec.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdlib/wall.h"
#include "gromacs/mdlib/wholemoleculetransform.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/atominfo.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forcebuffers.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/multipletimestepping.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/mdtypes/state_propagator_data_gpu.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/pairlist.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pulling/pull.h"
#include "gromacs/pulling/pull_rotation.h"
#include "gromacs/timing/cyclecounter.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/timing/wallcyclereporting.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/booltype.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/fixedcapacityvector.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/range.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/sysinfo.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vecdump.h"
#include "gromacs/utility/vectypes.h"

#include "gpuforcereduction.h"

class GpuEventSynchronizer;
struct gmx_edsam;
struct gmx_enfrot;
struct gmx_multisim_t;
struct gmx_pme_t;
struct interaction_const_t;
struct pull_t;

namespace gmx
{

// TODO: this environment variable allows us to verify before release
// that on less common architectures the total cost of polling is not larger than
// a blocking wait (so polling does not introduce overhead when the static
// PME-first ordering would suffice).
static const bool c_disableAlternatingWait = (std::getenv("GMX_DISABLE_ALTERNATING_GPU_WAIT") != nullptr);

static void sum_forces(ArrayRef<RVec> f, ArrayRef<const RVec> forceToAdd)
{
    GMX_ASSERT(f.size() >= forceToAdd.size(), "Accumulation buffer should be sufficiently large");
    const int end = forceToAdd.size();

    int gmx_unused nt = gmx_omp_nthreads_get(ModuleMultiThread::Default);
#pragma omp parallel for num_threads(nt) schedule(static)
    for (int i = 0; i < end; i++)
    {
        rvec_inc(f[i], forceToAdd[i]);
    }
}

static void calc_virial(int                         start,
                        int                         homenr,
                        const rvec                  x[],
                        const ForceWithShiftForces& forceWithShiftForces,
                        tensor                      vir_part,
                        const matrix                box,
                        t_nrnb*                     nrnb,
                        const t_forcerec*           fr,
                        PbcType                     pbcType)
{
    /* The short-range virial from surrounding boxes */
    const rvec* fshift          = as_rvec_array(forceWithShiftForces.shiftForces().data());
    const rvec* shiftVecPointer = as_rvec_array(fr->shift_vec.data());
    calc_vir(c_numShiftVectors, shiftVecPointer, fshift, vir_part, pbcType == PbcType::Screw, box);
    inc_nrnb(nrnb, eNR_VIRIAL, c_numShiftVectors);

    /* Calculate partial virial, for local atoms only, based on short range.
     * Total virial is computed in global_stat, called from do_md
     */
    const rvec* f = as_rvec_array(forceWithShiftForces.force().data());
    f_calc_vir(start, start + homenr, x, f, vir_part, box);
    inc_nrnb(nrnb, eNR_VIRIAL, homenr);

    if (debug)
    {
        pr_rvecs(debug, 0, "vir_part", vir_part, DIM);
    }
}

static void pull_potential_wrapper(const MpiComm&       mpiComm,
                                   const t_inputrec&    ir,
                                   const matrix         box,
                                   ArrayRef<const RVec> x,
                                   const t_mdatoms*     mdatoms,
                                   gmx_enerdata_t*      enerd,
                                   pull_t*              pull_work,
                                   const real*          lambda,
                                   double               t,
                                   gmx_wallcycle*       wcycle)
{
    t_pbc pbc;
    real  dvdl;

    /* Calculate the center of mass forces, this requires communication,
     * which is why pull_potential is called close to other communication.
     */
    wallcycle_start(wcycle, WallCycleCounter::PullPot);
    set_pbc(&pbc, ir.pbcType, box);
    dvdl = 0;
    enerd->term[InteractionFunction::CenterOfMassPullingEnergy] +=
            pull_potential(pull_work,
                           mdatoms->massT,
                           pbc,
                           mpiComm,
                           t,
                           lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Restraint)],
                           x,
                           &dvdl);
    enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Restraint] += dvdl;
    wallcycle_stop(wcycle, WallCycleCounter::PullPot);
}

static void pme_receive_force_ener(t_forcerec*      fr,
                                   gmx_domdec_t*    dd,
                                   ForceWithVirial* forceWithVirial,
                                   gmx_enerdata_t*  enerd,
                                   bool             useGpuPmePpComms,
                                   bool             receivePmeForceToGpu,
                                   gmx_wallcycle*   wcycle)
{
    real  e_q, e_lj, dvdl_q, dvdl_lj;
    float cycles_ppdpme, cycles_seppme;

    cycles_ppdpme = wallcycle_stop(wcycle, WallCycleCounter::PpDuringPme);
    dd_cycles_add(dd, cycles_ppdpme, ddCyclPPduringPME);

    /* In case of node-splitting, the PP nodes receive the long-range
     * forces, virial and energy from the PME nodes here.
     */
    wallcycle_start(wcycle, WallCycleCounter::PpPmeWaitRecvF);
    dvdl_q  = 0;
    dvdl_lj = 0;
    gmx_pme_receive_f(fr->pmePpCommGpu.get(),
                      dd,
                      fr->pmeForceReceiveBuffer,
                      forceWithVirial,
                      &e_q,
                      &e_lj,
                      &dvdl_q,
                      &dvdl_lj,
                      useGpuPmePpComms,
                      receivePmeForceToGpu,
                      &cycles_seppme);
    enerd->term[InteractionFunction::CoulombReciprocalSpace] += e_q;
    enerd->term[InteractionFunction::LennardJonesReciprocalSpace] += e_lj;
    enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Coul] += dvdl_q;
    enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Vdw] += dvdl_lj;

    if (wcycle)
    {
        dd_cycles_add(dd, cycles_seppme, ddCyclPME);
    }
    wallcycle_stop(wcycle, WallCycleCounter::PpPmeWaitRecvF);
}

static void print_large_forces(FILE*                fp,
                               const t_mdatoms*     md,
                               const gmx_domdec_t*  dd,
                               int64_t              step,
                               real                 forceTolerance,
                               ArrayRef<const RVec> x,
                               ArrayRef<const RVec> f)
{
    real  force2Tolerance = square(forceTolerance);
    Index numNonFinite    = 0;
    for (int i = 0; i < md->homenr; i++)
    {
        real force2    = norm2(f[i]);
        bool nonFinite = !std::isfinite(force2);
        if (force2 >= force2Tolerance || nonFinite)
        {
            fprintf(fp,
                    "step %" PRId64 " atom %6d  x %8.3f %8.3f %8.3f  force %12.5e\n",
                    step,
                    ddglatnr(dd, i),
                    x[i][XX],
                    x[i][YY],
                    x[i][ZZ],
                    std::sqrt(force2));
        }
        if (nonFinite)
        {
            numNonFinite++;
        }
    }
    if (numNonFinite > 0)
    {
        /* Note that with MPI this fatal call on one rank might interrupt
         * the printing on other ranks. But we can only avoid that with
         * an expensive MPI barrier that we would need at each step.
         */
        gmx_fatal(FARGS, "At step %" PRId64 " detected non-finite forces on %td atoms", step, numNonFinite);
    }
}

//! When necessary, spreads forces on vsites and computes the virial for \p forceOutputs->forceWithShiftForces()
static void postProcessForceWithShiftForces(t_nrnb*              nrnb,
                                            gmx_wallcycle*       wcycle,
                                            const matrix         box,
                                            ArrayRef<const RVec> x,
                                            ForceOutputs*        forceOutputs,
                                            tensor               vir_force,
                                            const t_mdatoms&     mdatoms,
                                            const t_forcerec&    fr,
                                            VirtualSitesHandler* vsite,
                                            const StepWorkload&  stepWork)
{
    ForceWithShiftForces& forceWithShiftForces = forceOutputs->forceWithShiftForces();

    /* If we have NoVirSum forces, but we do not calculate the virial,
     * we later sum the forceWithShiftForces buffer together with
     * the noVirSum buffer and spread the combined vsite forces at once.
     */
    if (vsite && (!forceOutputs->haveForceWithVirial() || stepWork.computeVirial))
    {
        using VirialHandling = VirtualSitesHandler::VirialHandling;

        auto                 f      = forceWithShiftForces.force();
        auto                 fshift = forceWithShiftForces.shiftForces();
        const VirialHandling virialHandling =
                (stepWork.computeVirial ? VirialHandling::Pbc : VirialHandling::None);
        vsite->spreadForces(x, f, virialHandling, fshift, nullptr, nrnb, box, wcycle);
        forceWithShiftForces.haveSpreadVsiteForces() = true;
    }

    if (stepWork.computeVirial)
    {
        /* Calculation of the virial must be done after vsites! */
        calc_virial(
                0, mdatoms.homenr, as_rvec_array(x.data()), forceWithShiftForces, vir_force, box, nrnb, &fr, fr.pbcType);
    }
}

//! Spread, compute virial for and sum forces, when necessary
static void postProcessForces(const gmx_domdec_t*  dd,
                              int64_t              step,
                              t_nrnb*              nrnb,
                              gmx_wallcycle*       wcycle,
                              const matrix         box,
                              ArrayRef<const RVec> x,
                              ForceOutputs*        forceOutputs,
                              tensor               vir_force,
                              const t_mdatoms*     mdatoms,
                              const t_forcerec*    fr,
                              VirtualSitesHandler* vsite,
                              const StepWorkload&  stepWork)
{
    // Extract the final output force buffer, which is also the buffer for forces with shift forces
    ArrayRef<RVec> f = forceOutputs->forceWithShiftForces().force();

    if (forceOutputs->haveForceWithVirial())
    {
        auto& forceWithVirial = forceOutputs->forceWithVirial();

        if (vsite)
        {
            /* Spread the mesh force on virtual sites to the other particles...
             * This is parallellized. MPI communication is performed
             * if the constructing atoms aren't local.
             */
            GMX_ASSERT(!stepWork.computeVirial || f.data() != forceWithVirial.force_.data(),
                       "We need separate force buffers for shift and virial forces when "
                       "computing the virial");
            GMX_ASSERT(!stepWork.computeVirial
                               || forceOutputs->forceWithShiftForces().haveSpreadVsiteForces(),
                       "We should spread the force with shift forces separately when computing "
                       "the virial");
            const VirtualSitesHandler::VirialHandling virialHandling =
                    (stepWork.computeVirial ? VirtualSitesHandler::VirialHandling::NonLinear
                                            : VirtualSitesHandler::VirialHandling::None);
            matrix virial = { { 0 } };
            vsite->spreadForces(x, forceWithVirial.force_, virialHandling, {}, virial, nrnb, box, wcycle);
            forceWithVirial.addVirialContribution(virial);
        }

        if (stepWork.computeVirial)
        {
            /* Now add the forces, this is local */
            sum_forces(f, forceWithVirial.force_);

            /* Add the direct virial contributions */
            GMX_ASSERT(
                    forceWithVirial.computeVirial_,
                    "forceWithVirial should request virial computation when we request the virial");
            m_add(vir_force, forceWithVirial.getVirial(), vir_force);

            if (debug)
            {
                pr_rvecs(debug, 0, "vir_force", vir_force, DIM);
            }
        }
    }
    else
    {
        GMX_ASSERT(vsite == nullptr || forceOutputs->forceWithShiftForces().haveSpreadVsiteForces(),
                   "We should have spread the vsite forces (earlier)");
    }

    if (fr->print_force >= 0)
    {
        print_large_forces(stderr, mdatoms, dd, step, fr->print_force, x, f);
    }
}

static void do_nb_verlet(t_forcerec*                fr,
                         const interaction_const_t* ic,
                         gmx_enerdata_t*            enerd,
                         const StepWorkload&        stepWork,
                         const InteractionLocality  ilocality,
                         const int                  clearF,
                         const int64_t              step,
                         t_nrnb*                    nrnb,
                         gmx_wallcycle*             wcycle)
{
    if (!stepWork.computeNonbondedForces)
    {
        /* skip non-bonded calculation */
        return;
    }

    nonbonded_verlet_t* nbv = fr->nbv.get();

    /* GPU kernel launch overhead is already timed separately */
    if (!nbv->useGpu())
    {
        /* When dynamic pair-list  pruning is requested, we need to prune
         * at nstlistPrune steps.
         */
        if (nbv->isDynamicPruningStepCpu(step))
        {
            /* Prune the pair-list beyond fr->ic->rlistPrune using
             * the current coordinates of the atoms.
             */
            wallcycle_sub_start(wcycle, WallCycleSubCounter::NonbondedPruning);
            nbv->dispatchPruneKernelCpu(ilocality, fr->shift_vec);
            wallcycle_sub_stop(wcycle, WallCycleSubCounter::NonbondedPruning);
        }
    }

    nbv->dispatchNonbondedKernel(
            ilocality,
            *ic,
            stepWork,
            clearF,
            fr->shift_vec,
            enerd->grpp.energyGroupPairTerms[fr->haveBuckingham ? NonBondedEnergyTerms::BuckinghamSR
                                                                : NonBondedEnergyTerms::LJSR],
            enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR],
            nrnb);
}

struct LammpsRespaSplitWeights
{
    real inner  = 0;
    real middle = 0;
    real outer  = 1;
};

static real respaSwitchIn(const real r, const real off, const real on)
{
    if (on <= off)
    {
        return (r >= on ? 1.0_real : 0.0_real);
    }
    if (r <= off)
    {
        return 0.0_real;
    }
    if (r >= on)
    {
        return 1.0_real;
    }

    const real x = (r - off) / (on - off);
    return x * x * (3.0_real - 2.0_real * x);
}

static LammpsRespaSplitWeights computeLammpsRespaSplitWeights(const t_inputrec& inputrec, const real r)
{
    LammpsRespaSplitWeights weights;
    const auto&             respa = inputrec.lammpsRespa;

    if (!respa.hasPairSplitting())
    {
        return weights;
    }

    if (respa.hasMiddle())
    {
        const real switchIntoMiddle = respaSwitchIn(r, respa.innerOff, respa.innerOn);
        const real switchIntoOuter  = respaSwitchIn(r, respa.outerOn, respa.outerOff);
        weights.inner               = 1.0_real - switchIntoMiddle;
        weights.middle              = switchIntoMiddle * (1.0_real - switchIntoOuter);
        weights.outer               = switchIntoOuter;
    }
    else
    {
        const real switchIntoOuter = respaSwitchIn(r, respa.outerOn, respa.outerOff);
        weights.inner              = 1.0_real - switchIntoOuter;
        weights.middle             = 0.0_real;
        weights.outer              = switchIntoOuter;
    }

    return weights;
}

static void accumulatePairVirial(const RVec& dx, const RVec& force, matrix virial)
{
    for (int dim1 = 0; dim1 < DIM; dim1++)
    {
        for (int dim2 = 0; dim2 < DIM; dim2++)
        {
            virial[dim1][dim2] -= 0.5_real * dx[dim1] * force[dim2];
        }
    }
}

static int energyGroupPairIndex(const int ai, const int aj, const t_forcerec& fr, const t_mdatoms& mdatoms)
{
    if (mdatoms.nenergrp <= 1)
    {
        return 0;
    }

    const int gidI = fr.atomInfo[ai] & gmx::sc_atomInfo_EnergyGroupIdMask;
    const int gidJ = fr.atomInfo[aj] & gmx::sc_atomInfo_EnergyGroupIdMask;
    return gidI * mdatoms.nenergrp + gidJ;
}

static void computePmeRealSpaceCoulombComponents(const interaction_const_t::CoulombSettings& coulomb,
                                                 const EwaldCorrectionTables*                 coulombTables,
                                                 const real                                   qq,
                                                 const real                                   r,
                                                 const real                                   rinv,
                                                 const real                                   factorCoulomb,
                                                 real*                                        bareCoulombScalar,
                                                 real*                                        correctionScalar,
                                                 real*                                        fullCoulombEnergy)
{
    GMX_RELEASE_ASSERT(coulombTables != nullptr, "PME real-space split requires Coulomb Ewald tables");

    const real scaledR    = r * coulombTables->scale;
    const int  tableIndex = static_cast<int>(scaledR);
    const real frac       = scaledR - tableIndex;
    const real halfsp     = 0.5_real / coulombTables->scale;

#if !GMX_DOUBLE
    const real* table = coulombTables->tableFDV0.data();
    const real  fexcl = table[tableIndex * 4] + frac * table[tableIndex * 4 + 1];
    const real  vcorr = table[tableIndex * 4 + 2] - halfsp * frac * (table[tableIndex * 4] + fexcl);
#else
    const real* tableF = coulombTables->tableF.data();
    const real* tableV = coulombTables->tableV.data();
    const real  fexcl  = (1 - frac) * tableF[tableIndex] + frac * tableF[tableIndex + 1];
    const real  vcorr  = tableV[tableIndex] - halfsp * frac * (tableF[tableIndex] + fexcl);
#endif

    *bareCoulombScalar = factorCoulomb * qq * rinv;
    *correctionScalar  = -qq * fexcl / rinv;
    *fullCoulombEnergy = qq * (factorCoulomb * (rinv - coulomb.ewaldShift) - vcorr);
}

static real computePmeSelfEnergy(const interaction_const_t& ic)
{
    GMX_RELEASE_ASSERT(ic.coulombEwaldTables, "PME self-energy requires Coulomb Ewald tables");
    return 0.5_real
#if !GMX_DOUBLE
           * ic.coulombEwaldTables->tableFDV0[2]
#else
           * ic.coulombEwaldTables->tableV[0]
#endif
            ;
}

static void dumpRespaMergeTraceVector(const char*               traceDirPath,
                                      const char*               fileName,
                                      const std::string&        header,
                                      ArrayRef<const gmx::RVec> forceBuffer)
{
    GMX_RELEASE_ASSERT(traceDirPath != nullptr && *traceDirPath != '\0', "Need a valid merge trace directory");

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::filesystem::path outputPath = traceDir / fileName;

    FILE* dumpFile = std::fopen(outputPath.string().c_str(), "w");
    if (dumpFile == nullptr)
    {
        gmx_fatal(FARGS, "Could not open merge trace output '%s' for writing", outputPath.string().c_str());
    }

    std::fprintf(dumpFile, "# %s\n", header.c_str());
    for (int atom = 0; atom < forceBuffer.ssize(); ++atom)
    {
        std::fprintf(dumpFile,
                     "%d\t%.17g\t%.17g\t%.17g\n",
                     atom,
                     forceBuffer[atom][XX],
                     forceBuffer[atom][YY],
                     forceBuffer[atom][ZZ]);
    }
    std::fclose(dumpFile);
}

static void dumpRespaTraceEvent(const char*        traceDirPath,
                                const char*        fileName,
                                const std::string& header,
                                const int          atomI,
                                const gmx::RVec&   forceI,
                                const int          atomJ,
                                const gmx::RVec&   forceJ)
{
    GMX_RELEASE_ASSERT(traceDirPath != nullptr && *traceDirPath != '\0', "Need a valid trace directory");

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::filesystem::path outputPath = traceDir / fileName;

    FILE* dumpFile = std::fopen(outputPath.string().c_str(), "w");
    if (dumpFile == nullptr)
    {
        gmx_fatal(FARGS, "Could not open trace event output '%s' for writing", outputPath.string().c_str());
    }

    std::fprintf(dumpFile, "# %s\n", header.c_str());
    std::fprintf(dumpFile, "%d\t%.17g\t%.17g\t%.17g\n", atomI, forceI[XX], forceI[YY], forceI[ZZ]);
    std::fprintf(dumpFile, "%d\t%.17g\t%.17g\t%.17g\n", atomJ, forceJ[XX], forceJ[YY], forceJ[ZZ]);
    std::fclose(dumpFile);
}

static std::string formatPointerValue(const void* ptr)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "0x%" PRIxPTR, reinterpret_cast<std::uintptr_t>(ptr));
    return std::string(buffer);
}

static void writeRespaTraceTextFile(const char* traceDirPath, const char* fileName, const std::string& contents)
{
    GMX_RELEASE_ASSERT(traceDirPath != nullptr && *traceDirPath != '\0', "Need a valid trace directory");

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::filesystem::path outputPath = traceDir / fileName;

    FILE* dumpFile = std::fopen(outputPath.string().c_str(), "w");
    if (dumpFile == nullptr)
    {
        gmx_fatal(FARGS, "Could not open trace text output '%s' for writing", outputPath.string().c_str());
    }

    std::fputs(contents.c_str(), dumpFile);
    std::fclose(dumpFile);
}

static void appendRespaTraceTextLine(const char* traceDirPath, const char* fileName, const std::string& line)
{
    GMX_RELEASE_ASSERT(traceDirPath != nullptr && *traceDirPath != '\0',
                       "Need a valid r-RESPA trace directory");

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::filesystem::path outputPath = traceDir / fileName;

    FILE* dumpFile = std::fopen(outputPath.string().c_str(), "a");
    if (dumpFile == nullptr)
    {
        gmx_fatal(FARGS, "Could not open trace text output '%s' for appending", outputPath.string().c_str());
    }

    std::fprintf(dumpFile, "%s\n", line.c_str());
    std::fclose(dumpFile);
}

static void computeLammpsRespaNonbondedCpu(const t_inputrec&                inputrec,
                                           const InteractionDefinitions&    idef,
                                           t_forcerec*                      fr,
                                           const t_mdatoms&                 mdatoms,
                                           ArrayRef<const RVec>             coordinates,
                                           ArrayRef<ForceOutputs*>          forceOutByMtsLevel,
                                           gmx_enerdata_t*                  enerd,
                                           const StepWorkload&              stepWork)
{
    GMX_RELEASE_ASSERT(fr->plainPairlistRange.has_value(),
                       "Exact LAMMPS-style r-RESPA requires a plain pairlist");
    GMX_RELEASE_ASSERT(fr->efep == FreeEnergyPerturbationType::No,
                       "Exact LAMMPS-style r-RESPA does not support free-energy perturbation yet");
    GMX_RELEASE_ASSERT(fr->ic->vdw.type == VanDerWaalsType::Cut,
                       "Exact LAMMPS-style r-RESPA currently supports cut-off Van der Waals only");
    GMX_RELEASE_ASSERT(fr->ic->vdw.modifier == InteractionModifiers::None,
                       "Exact LAMMPS-style r-RESPA currently supports unmodified real-space LJ only");
    GMX_RELEASE_ASSERT(usingPmeOrEwald(fr->ic->coulomb.type),
                       "Exact LAMMPS-style r-RESPA currently supports Coulomb long-range treatment only");
    GMX_RELEASE_ASSERT(fr->ic->coulomb.modifier == InteractionModifiers::None,
                       "Exact LAMMPS-style r-RESPA currently supports unmodified real-space Coulomb only");

    const auto& plainPairlist = fr->nbv->plainPairlist(fr->plainPairlistRange.value(), fr->shift_vec);

    struct ContributionAccumulator
    {
        MtsNonbondedRespaContribution contribution;
        ForceOutputs*                 outputs = nullptr;
        ArrayRef<RVec>                force;
        ArrayRef<RVec>                shift;
        ForceWithVirial*              forceWithVirial = nullptr;
        bool                          accumulateEnergy = false;
        matrix                        virial           = { { 0 } };
    };

    std::vector<ContributionAccumulator> activeContributions;

    const auto appendContribution = [&](const MtsNonbondedRespaContribution contribution)
    {
        const int mtsLevel = nonbondedRespaContributionMtsLevel(inputrec, contribution);
        if (mtsLevel < 0 || mtsLevel > stepWork.highestActiveMtsLevel || mtsLevel >= forceOutByMtsLevel.ssize()
            || forceOutByMtsLevel[mtsLevel] == nullptr)
        {
            return;
        }

        ForceOutputs* outputs = forceOutByMtsLevel[mtsLevel];

        ContributionAccumulator accumulator;
        accumulator.contribution = contribution;
        accumulator.outputs      = outputs;
        accumulator.accumulateEnergy =
                (contribution == MtsNonbondedRespaContribution::Outer
                 || contribution == MtsNonbondedRespaContribution::Full)
                && stepWork.computeEnergy;

        const bool directVirialContribution =
                stepWork.computeVirial
                && (contribution == MtsNonbondedRespaContribution::Outer
                    || contribution == MtsNonbondedRespaContribution::Full);
        if (directVirialContribution)
        {
            GMX_RELEASE_ASSERT(outputs->haveForceWithVirial(),
                               "Exact LAMMPS-style r-RESPA outer forces require a direct-virial buffer");
            accumulator.forceWithVirial = &outputs->forceWithVirial();
            accumulator.force           = accumulator.forceWithVirial->force_;
        }
        else
        {
            accumulator.force = outputs->forceWithShiftForces().force();
            accumulator.shift = outputs->forceWithShiftForces().shiftForces();
        }

        activeContributions.push_back(accumulator);
    };

    appendContribution(MtsNonbondedRespaContribution::Inner);
    appendContribution(MtsNonbondedRespaContribution::Middle);
    appendContribution(MtsNonbondedRespaContribution::Outer);

    ContributionAccumulator* outerAccumulator = nullptr;
    for (auto& accumulator : activeContributions)
    {
        if (accumulator.contribution == MtsNonbondedRespaContribution::Outer)
        {
            outerAccumulator = &accumulator;
            break;
        }
    }

    GMX_RELEASE_ASSERT(!stepWork.computeVirial
                               || std::any_of(activeContributions.begin(),
                                              activeContributions.end(),
                                              [](const ContributionAccumulator& accumulator)
                                              {
                                                  return accumulator.contribution
                                                                 == MtsNonbondedRespaContribution::Outer
                                                         || accumulator.contribution
                                                                    == MtsNonbondedRespaContribution::Full;
                                              }),
                       "Exact LAMMPS-style r-RESPA virial steps require the outer contribution to be active");

    const real coulombCutoff2   = gmx::square(fr->ic->coulomb.cutoff);
    const real vdwCutoff2       = gmx::square(fr->ic->vdw.cutoff);
    const real repulsionPower   = static_cast<real>(fr->ic->vdw.repulsionPower);
    const int  ntype2           = 2 * fr->ntype;
    auto&      vdwEnergyTerms   = enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJSR];
    auto&      coulEnergyTerms  = enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR];
    const bool debugExactRespa  = (std::getenv("GMX_PCFF_RESPA_DEBUG") != nullptr);
    const char* excludedCorrectionForceDumpPath =
            std::getenv("GMX_PCFF_RESPA_EXCLUDED_FORCE_DUMP_FILE");
    const bool dumpExcludedCorrectionForce =
            (excludedCorrectionForceDumpPath != nullptr && *excludedCorrectionForceDumpPath != '\0');
    const char* earlyAccumTraceDirPath = std::getenv("GMX_PCFF_RESPA_EARLY_TRACE_DIR");
    static bool dumpedEarlyAccumTrace  = false;
    const bool dumpEarlyAccumTrace =
            (earlyAccumTraceDirPath != nullptr && *earlyAccumTraceDirPath != '\0' && !dumpedEarlyAccumTrace);
    const char* pairWriteProofDirPath = std::getenv("GMX_PCFF_RESPA_PAIR_WRITE_PROOF_DIR");
    static bool dumpedPairWriteProof  = false;
    const bool dumpPairWriteProof =
            (pairWriteProofDirPath != nullptr && *pairWriteProofDirPath != '\0' && !dumpedPairWriteProof);
    const char* downstreamContractTraceDirPath = std::getenv("GMX_PCFF_RESPA_DOWNSTREAM_CONTRACT_TRACE_DIR");
    static bool dumpedDownstreamContractTrace  = false;
    const bool dumpDownstreamContract =
            (downstreamContractTraceDirPath != nullptr && *downstreamContractTraceDirPath != '\0'
             && !dumpedDownstreamContractTrace);
    const bool outerAliasesShift =
            (outerAccumulator != nullptr && outerAccumulator->outputs != nullptr
             && outerAccumulator->force.data()
                        == outerAccumulator->outputs->forceWithShiftForces().force().data());
    const real pmeSelfEnergy    = computePmeSelfEnergy(*fr->ic);
    std::vector<RVec> excludedCorrectionForce;
    if (dumpExcludedCorrectionForce)
    {
        excludedCorrectionForce.resize(coordinates.size());
        for (auto& force : excludedCorrectionForce)
        {
            clear_rvec(force);
        }
    }
    if (dumpEarlyAccumTrace && outerAccumulator != nullptr && outerAccumulator->forceWithVirial != nullptr)
    {
        dumpRespaMergeTraceVector(earlyAccumTraceDirPath,
                                  "step0_level2_initial_outer_virial.tsv",
                                  "stage=initial_outer mts_index=2 mts_user=3 buffer=forceWithVirial alias_with_shift="
                                          + std::string(outerAliasesShift ? "true" : "false"),
                                  outerAccumulator->forceWithVirial->force_);
    }
    if (dumpPairWriteProof)
    {
        std::string contents;
        const auto  appendLine = [&contents](const std::string& line) { contents += line + "\n"; };
        for (int mtsLevel = 0; mtsLevel < forceOutByMtsLevel.ssize(); ++mtsLevel)
        {
            const ForceOutputs* outputs = forceOutByMtsLevel[mtsLevel];
            if (outputs == nullptr)
            {
                appendLine("level=" + std::to_string(mtsLevel) + " active=false");
                continue;
            }
            ForceOutputs* mutableOutputs = const_cast<ForceOutputs*>(outputs);
            appendLine("level=" + std::to_string(mtsLevel) + " active=true");
            appendLine("level=" + std::to_string(mtsLevel) + " shift_force_ptr="
                       + formatPointerValue(mutableOutputs->forceWithShiftForces().force().data()));
            appendLine("level=" + std::to_string(mtsLevel) + " shift_shift_ptr="
                       + formatPointerValue(mutableOutputs->forceWithShiftForces().shiftForces().data()));
            appendLine("level=" + std::to_string(mtsLevel) + " have_virial="
                       + std::string(outputs->haveForceWithVirial() ? "true" : "false"));
            if (outputs->haveForceWithVirial())
            {
                appendLine("level=" + std::to_string(mtsLevel) + " virial_force_ptr="
                           + formatPointerValue(mutableOutputs->forceWithVirial().force_.data()));
            }
        }
        appendLine("outer_accumulator_present=" + std::string(outerAccumulator != nullptr ? "true" : "false"));
        if (outerAccumulator != nullptr)
        {
            appendLine("outer_accumulator_force_ptr=" + formatPointerValue(outerAccumulator->force.data()));
            appendLine("outer_accumulator_shift_ptr=" + formatPointerValue(outerAccumulator->shift.data()));
            appendLine("outer_accumulator_has_virial="
                       + std::string(outerAccumulator->forceWithVirial != nullptr ? "true" : "false"));
            if (outerAccumulator->forceWithVirial != nullptr)
            {
                appendLine("outer_accumulator_virial_ptr="
                           + formatPointerValue(outerAccumulator->forceWithVirial->force_.data()));
            }
            if (outerAccumulator->outputs != nullptr)
            {
                appendLine("outer_outputs_shift_force_ptr="
                           + formatPointerValue(outerAccumulator->outputs->forceWithShiftForces().force().data()));
                appendLine("outer_outputs_shift_shift_ptr="
                           + formatPointerValue(outerAccumulator->outputs->forceWithShiftForces().shiftForces().data()));
            }
            appendLine("outer_aliases_shift=" + std::string(outerAliasesShift ? "true" : "false"));
        }
        appendLine("excluded_correction_force_dump_enabled="
                   + std::string(dumpExcludedCorrectionForce ? "true" : "false"));
        appendLine("excluded_correction_force_dump_ptr=" + formatPointerValue(excludedCorrectionForce.data()));
        writeRespaTraceTextFile(pairWriteProofDirPath, "step0_force_storage_identity.txt", contents);
    }
    const auto pairKey = [](const int ai, const int aj)
    {
        const uint32_t first  = static_cast<uint32_t>(std::min(ai, aj));
        const uint32_t second = static_cast<uint32_t>(std::max(ai, aj));
        return (static_cast<uint64_t>(first) << 32) | second;
    };
    const uint64_t targetPairKey  = pairKey(0, 1);
    const uint64_t controlPairKey = pairKey(0, 4);
    struct PairEntryInfo
    {
        int      ordinal    = -1;
        int      ai         = -1;
        int      aj         = -1;
        int      shiftIndex = -1;
        uint64_t key        = 0;
    };
    constexpr int c_maxControlPairWriteProofs = 8;
    std::vector<PairEntryInfo> firstPairEntries;
    firstPairEntries.reserve(c_maxControlPairWriteProofs);
    for (int ordinal = 0; ordinal < static_cast<int>(plainPairlist.pairs.size())
                          && ordinal < c_maxControlPairWriteProofs;
         ++ordinal)
    {
        const auto& entry = plainPairlist.pairs[ordinal];
        firstPairEntries.push_back(
                PairEntryInfo{ ordinal, entry.first.first, entry.first.second, entry.second, pairKey(entry.first.first, entry.first.second) });
    }
    std::optional<PairEntryInfo> firstExcludedEntry;
    if (!plainPairlist.excludedPairs.empty())
    {
        const auto& entry = plainPairlist.excludedPairs.front();
        firstExcludedEntry.emplace(
                PairEntryInfo{ 0, entry.first.first, entry.first.second, entry.second, pairKey(entry.first.first, entry.first.second) });
    }
    if (dumpPairWriteProof)
    {
        std::string contents;
        const auto  appendLine = [&contents](const std::string& line) { contents += line + "\n"; };
        appendLine("kind=pairlist_preview list=pairs count=" + std::to_string(plainPairlist.pairs.size()));
        for (const auto& pairEntry : firstPairEntries)
        {
            appendLine("kind=pairs ordinal=" + std::to_string(pairEntry.ordinal) + " ai="
                       + std::to_string(pairEntry.ai) + " aj=" + std::to_string(pairEntry.aj) + " shift_index="
                       + std::to_string(pairEntry.shiftIndex));
        }
        appendLine("kind=pairlist_preview list=excludedPairs count="
                   + std::to_string(plainPairlist.excludedPairs.size()));
        if (firstExcludedEntry.has_value())
        {
            appendLine("kind=excludedPairs ordinal=0 ai=" + std::to_string(firstExcludedEntry->ai) + " aj="
                       + std::to_string(firstExcludedEntry->aj) + " shift_index="
                       + std::to_string(firstExcludedEntry->shiftIndex));
        }
        writeRespaTraceTextFile(pairWriteProofDirPath, "step0_plain_pairlist_preview.txt", contents);
    }
    std::unordered_set<uint64_t> listedPairKeys;
    if (debugExactRespa)
    {
        const auto appendInteractionList = [&listedPairKeys, &idef, &pairKey](const InteractionFunction ftype)
        {
            const auto& iatoms = idef.il[ftype].iatoms;
            const int   stride = NRAL(ftype) + 1;
            for (int index = 0; index < static_cast<int>(iatoms.size()); index += stride)
            {
                listedPairKeys.insert(pairKey(iatoms[index + 1], iatoms[index + 2]));
            }
        };
        appendInteractionList(InteractionFunction::LennardJones14);
        appendInteractionList(InteractionFunction::LennardJonesCoulomb14Q);
        appendInteractionList(InteractionFunction::LennardJonesCoulombNonBondedPairs);
    }
    if (dumpPairWriteProof)
    {
        const auto countOccurrences =
                [&pairKey](const auto& entries, const uint64_t keyToCount) -> int
        {
            return static_cast<int>(std::count_if(entries.begin(),
                                                 entries.end(),
                                                 [&pairKey, keyToCount](const auto& entry)
                                                 { return pairKey(entry.first.first, entry.first.second) == keyToCount; }));
        };

        std::string contents;
        const auto  appendLine = [&contents](const std::string& line) { contents += line + "\n"; };
        if (firstExcludedEntry.has_value())
        {
            appendLine("kind=excluded ordinal=0 ai=" + std::to_string(firstExcludedEntry->ai) + " aj="
                       + std::to_string(firstExcludedEntry->aj) + " shift_index="
                       + std::to_string(firstExcludedEntry->shiftIndex) + " in_plain_pairs="
                       + std::to_string(countOccurrences(plainPairlist.pairs, firstExcludedEntry->key))
                       + " in_plain_excluded="
                       + std::to_string(countOccurrences(plainPairlist.excludedPairs, firstExcludedEntry->key))
                       + " in_debug_listed_pair_keys="
                       + std::string(listedPairKeys.count(firstExcludedEntry->key) != 0 ? "true" : "false"));
        }
        for (const auto& pairEntry : firstPairEntries)
        {
            appendLine("kind=pairs ordinal=" + std::to_string(pairEntry.ordinal) + " ai="
                       + std::to_string(pairEntry.ai) + " aj=" + std::to_string(pairEntry.aj)
                       + " shift_index=" + std::to_string(pairEntry.shiftIndex) + " in_plain_pairs="
                       + std::to_string(countOccurrences(plainPairlist.pairs, pairEntry.key))
                       + " in_plain_excluded="
                       + std::to_string(countOccurrences(plainPairlist.excludedPairs, pairEntry.key))
                       + " in_debug_listed_pair_keys="
                       + std::string(listedPairKeys.count(pairEntry.key) != 0 ? "true" : "false"));
        }
        writeRespaTraceTextFile(pairWriteProofDirPath, "step0_pair_key_membership_scan.txt", contents);
    }

    const auto countPairKeyOccurrences = [&pairKey](const auto& entries, const uint64_t keyToCount) -> int
    {
        return static_cast<int>(std::count_if(entries.begin(),
                                              entries.end(),
                                              [&pairKey, keyToCount](const auto& entry)
                                              { return pairKey(entry.first.first, entry.first.second) == keyToCount; }));
    };
    const auto firstOrdinalForKey = [&pairKey](const auto& entries, const uint64_t keyToFind) -> int
    {
        for (int ordinal = 0; ordinal < static_cast<int>(entries.size()); ++ordinal)
        {
            if (pairKey(entries[ordinal].first.first, entries[ordinal].first.second) == keyToFind)
            {
                return ordinal;
            }
        }
        return -1;
    };

    if (dumpDownstreamContract)
    {
        appendRespaTraceTextLine(
                downstreamContractTraceDirPath,
                "step0_downstream_contract_trace.txt",
                "stage=excluded_pairs_dispatch_contract list=excludedPairs factor_coulomb=0 factor_lj=0 include_rule=always_true "
                        "target_in_list="
                        + std::string(countPairKeyOccurrences(plainPairlist.excludedPairs, targetPairKey) > 0 ? "true"
                                                                                                              : "false")
                        + " target_ordinal="
                        + std::to_string(firstOrdinalForKey(plainPairlist.excludedPairs, targetPairKey))
                        + " target_occurrences="
                        + std::to_string(countPairKeyOccurrences(plainPairlist.excludedPairs, targetPairKey))
                        + " control_in_list="
                        + std::string(countPairKeyOccurrences(plainPairlist.excludedPairs, controlPairKey) > 0 ? "true"
                                                                                                                : "false")
                        + " control_ordinal="
                        + std::to_string(firstOrdinalForKey(plainPairlist.excludedPairs, controlPairKey))
                        + " control_occurrences="
                        + std::to_string(countPairKeyOccurrences(plainPairlist.excludedPairs, controlPairKey))
                        + " semantic_role=excluded_membership_reintroduced_into_exact_nonbonded_consumer");
        appendRespaTraceTextLine(
                downstreamContractTraceDirPath,
                "step0_downstream_contract_trace.txt",
                "stage=pairs_dispatch_contract list=pairs factor_coulomb=1 factor_lj=1 include_rule=always_true "
                        "target_in_list="
                        + std::string(countPairKeyOccurrences(plainPairlist.pairs, targetPairKey) > 0 ? "true" : "false")
                        + " target_ordinal=" + std::to_string(firstOrdinalForKey(plainPairlist.pairs, targetPairKey))
                        + " target_occurrences="
                        + std::to_string(countPairKeyOccurrences(plainPairlist.pairs, targetPairKey))
                        + " control_in_list="
                        + std::string(countPairKeyOccurrences(plainPairlist.pairs, controlPairKey) > 0 ? "true" : "false")
                        + " control_ordinal=" + std::to_string(firstOrdinalForKey(plainPairlist.pairs, controlPairKey))
                        + " control_occurrences="
                        + std::to_string(countPairKeyOccurrences(plainPairlist.pairs, controlPairKey))
                        + " semantic_role=standard_physical_nonbonded_consumer");
    }

    struct PairDebugStats
    {
        const char* label        = nullptr;
        int         count        = 0;
        double      ljEnergy     = 0;
        double      coulEnergy   = 0;
        double      qqSum        = 0;
        double      selfEnergy   = 0;
    };

    const auto processPairlist = [&](const auto& pairEntries,
                                     const real factorCoulomb,
                                     const real factorLj,
                                     const auto& includePair,
                                     PairDebugStats* debugStats = nullptr)
    {
        bool dumpedFirstExcludedWrite = false;
        int  pairOrdinal             = 0;
        bool dumpedDownstreamTargetEval = false;
        bool dumpedDownstreamControlEval = false;
        for (const auto& entry : pairEntries)
        {
            const int ai         = entry.first.first;
            const int aj         = entry.first.second;
            const int shiftIndex = entry.second;

            if (!includePair(ai, aj))
            {
                continue;
            }

            const bool isExcludedPairlist = (factorCoulomb == 0.0_real && factorLj == 0.0_real);

            RVec dx;
            for (int dim = 0; dim < DIM; dim++)
            {
                dx[dim] = coordinates[ai][dim] - coordinates[aj][dim] + fr->shift_vec[shiftIndex][dim];
            }

            real rsq = iprod(dx, dx);
            rsq      = std::max(rsq, c_nbnxnMinDistanceSquared);

            const real rinv   = gmx::invsqrt(rsq);
            const real rinvsq = rinv * rinv;
            const real r      = rsq * rinv;

            const auto splitWeights = computeLammpsRespaSplitWeights(inputrec, r);

            real rawLjScalar = 0;
            real rawLjEnergy = 0;
            if (factorLj != 0.0_real && rsq < vdwCutoff2)
            {
                const int  typeI     = mdatoms.typeA[ai];
                const int  typeJ     = mdatoms.typeA[aj];
                const real c6        = fr->nbfp[typeI * ntype2 + typeJ * 2];
                const real cRepulsive = fr->nbfp[typeI * ntype2 + typeJ * 2 + 1];
                const real rinvsix   = rinvsq * rinvsq * rinvsq;
                const real repulsiveTerm =
                        (repulsionPower == 12.0_real ? rinvsix * rinvsix : std::pow(rinv, repulsionPower));
                rawLjScalar = cRepulsive * repulsiveTerm - c6 * rinvsix;
                rawLjEnergy = cRepulsive * repulsiveTerm / repulsionPower - c6 * rinvsix / 6.0_real;
            }

            real bareCoulombScalar = 0;
            real correctionScalar  = 0;
            real fullCoulombEnergy = 0;
            real qq                = 0;
            if (rsq < coulombCutoff2)
            {
                qq = mdatoms.chargeA[ai] * mdatoms.chargeA[aj] * fr->ic->coulomb.epsfac;
                if (qq != 0.0_real)
                {
                    computePmeRealSpaceCoulombComponents(fr->ic->coulomb,
                                                         fr->ic->coulombEwaldTables.get(),
                                                         qq,
                                                         r,
                                                         rinv,
                                                         factorCoulomb,
                                                         &bareCoulombScalar,
                                                         &correctionScalar,
                                                         &fullCoulombEnergy);
                }
            }

            if (debugStats != nullptr)
            {
                debugStats->count++;
                debugStats->ljEnergy += rawLjEnergy * factorLj;
                debugStats->coulEnergy += fullCoulombEnergy;
                debugStats->qqSum += qq;
            }

            const real innerScalar =
                    bareCoulombScalar * splitWeights.inner + factorLj * rawLjScalar * splitWeights.inner;
            const real middleScalar =
                    bareCoulombScalar * splitWeights.middle + factorLj * rawLjScalar * splitWeights.middle;
            const real outerScalar =
                    correctionScalar + bareCoulombScalar * splitWeights.outer + factorLj * rawLjScalar * splitWeights.outer;
            const real fullScalar = correctionScalar + bareCoulombScalar + factorLj * rawLjScalar;

            const bool isTargetPair = (ai == 0 && aj == 1);
            const bool isControlPair = (ai == 0 && aj == 4);
            if (dumpDownstreamContract
                && ((isTargetPair && !dumpedDownstreamTargetEval) || (isControlPair && !dumpedDownstreamControlEval)))
            {
                appendRespaTraceTextLine(
                        downstreamContractTraceDirPath,
                        "step0_downstream_contract_trace.txt",
                        "stage=consumer_pair_eval pair_list="
                                + std::string(factorCoulomb == 0.0_real && factorLj == 0.0_real ? "excludedPairs"
                                                                                                 : "pairs")
                                + " ordinal=" + std::to_string(pairOrdinal) + " ai=" + std::to_string(ai) + " aj="
                                + std::to_string(aj) + " shift_index=" + std::to_string(shiftIndex)
                                + " include_pair=true factor_coulomb=" + gmx::toString(factorCoulomb)
                                + " factor_lj=" + gmx::toString(factorLj) + " correction_scalar="
                                + gmx::toString(correctionScalar) + " bare_coulomb_scalar="
                                + gmx::toString(bareCoulombScalar) + " raw_lj_scalar="
                                + gmx::toString(rawLjScalar) + " inner_scalar=" + gmx::toString(innerScalar)
                                + " middle_scalar=" + gmx::toString(middleScalar) + " outer_scalar="
                                + gmx::toString(outerScalar) + " full_scalar=" + gmx::toString(fullScalar)
                                + " outer_force_write_eligible="
                                + std::string(outerScalar != 0.0_real && outerAccumulator != nullptr ? "true"
                                                                                                      : "false")
                                + " outer_force_buffer="
                                + std::string((outerAccumulator != nullptr && outerAccumulator->forceWithVirial != nullptr)
                                                      ? "forceWithVirial"
                                                      : "none")
                                + " semantic_role="
                                + std::string(factorCoulomb == 0.0_real && factorLj == 0.0_real
                                                      ? "excluded_membership_promoted_to_physical_outer_consumer"
                                                      : "standard_physical_nonbonded_consumer"));
                if (isTargetPair)
                {
                    dumpedDownstreamTargetEval = true;
                }
                if (isControlPair)
                {
                    dumpedDownstreamControlEval = true;
                }
            }

            if (dumpExcludedCorrectionForce && factorCoulomb == 0.0_real && factorLj == 0.0_real
                && correctionScalar != 0.0_real)
            {
                RVec correctionForce;
                svmul(correctionScalar * rinvsq, dx, correctionForce);
                rvec_inc(excludedCorrectionForce[ai], correctionForce);
                rvec_dec(excludedCorrectionForce[aj], correctionForce);
            }

            for (auto& accumulator : activeContributions)
            {
                real scalar = 0;
                switch (accumulator.contribution)
                {
                    case MtsNonbondedRespaContribution::Inner:
                        scalar = innerScalar;
                        break;
                    case MtsNonbondedRespaContribution::Middle:
                        scalar = middleScalar;
                        break;
                    case MtsNonbondedRespaContribution::Outer:
                        scalar = outerScalar;
                        break;
                    case MtsNonbondedRespaContribution::Full:
                        scalar = fullScalar;
                        break;
                    default: GMX_RELEASE_ASSERT(false, "Unexpected nonbonded r-RESPA contribution");
                }

                if (scalar == 0.0_real)
                {
                    continue;
                }

                RVec force;
                svmul(scalar * rinvsq, dx, force);
                const bool shouldDumpExcludedPairWrite =
                        dumpPairWriteProof && isExcludedPairlist
                        && accumulator.contribution == MtsNonbondedRespaContribution::Outer && pairOrdinal == 0;
                const bool shouldDumpControlPairWrite =
                        dumpPairWriteProof && !isExcludedPairlist
                        && accumulator.contribution == MtsNonbondedRespaContribution::Outer
                        && pairOrdinal < c_maxControlPairWriteProofs;
                if ((shouldDumpExcludedPairWrite || shouldDumpControlPairWrite)
                    && accumulator.forceWithVirial != nullptr)
                {
                    const std::string prefix = shouldDumpExcludedPairWrite
                                                       ? "step0_outer_excluded_write_ord000"
                                                       : "step0_outer_pairs_write_ord"
                                                                 + std::to_string(pairOrdinal);
                    const std::string header =
                            "stage="
                            + std::string(shouldDumpExcludedPairWrite ? "first_excluded_outer_write_boundary"
                                                                      : "control_pairs_outer_write_boundary")
                            + " pair_list=" + std::string(isExcludedPairlist ? "excludedPairs" : "pairs")
                            + " ordinal=" + std::to_string(pairOrdinal) + " contribution=outer buffer=forceWithVirial"
                            + " accumulator_ptr=" + formatPointerValue(accumulator.force.data())
                            + " virial_ptr="
                            + formatPointerValue(accumulator.forceWithVirial->force_.data()) + " ai="
                            + std::to_string(ai) + " aj=" + std::to_string(aj) + " shift_index="
                            + std::to_string(shiftIndex) + " scalar=" + gmx::toString(scalar)
                            + " correction_scalar=" + gmx::toString(correctionScalar) + " qq="
                            + gmx::toString(qq) + " r=" + gmx::toString(r);
                    dumpRespaMergeTraceVector(pairWriteProofDirPath,
                                              (prefix + "_before.tsv").c_str(),
                                              header + " snapshot=before",
                                              accumulator.force);
                    dumpRespaTraceEvent(pairWriteProofDirPath,
                                        (prefix + "_event.tsv").c_str(),
                                        header + " snapshot=event",
                                        ai,
                                        force,
                                        aj,
                                        RVec(-force[XX], -force[YY], -force[ZZ]));
                }
                if (dumpEarlyAccumTrace && factorCoulomb == 0.0_real && factorLj == 0.0_real
                    && accumulator.contribution == MtsNonbondedRespaContribution::Outer && !dumpedFirstExcludedWrite)
                {
                    dumpRespaTraceEvent(
                            earlyAccumTraceDirPath,
                            "step0_outer_first_excluded_write.tsv",
                            "stage=first_excluded_outer_write pair_list=excludedPairs contribution=outer buffer=forceWithVirial alias_with_shift="
                                    + std::string(outerAliasesShift ? "true" : "false") + " ai="
                                    + std::to_string(ai) + " aj=" + std::to_string(aj) + " shift_index="
                                    + std::to_string(shiftIndex) + " scalar=" + gmx::toString(scalar)
                                    + " correction_scalar=" + gmx::toString(correctionScalar) + " qq="
                                    + gmx::toString(qq) + " r=" + gmx::toString(r),
                            ai,
                            force,
                            aj,
                            RVec(-force[XX], -force[YY], -force[ZZ]));
                    dumpedFirstExcludedWrite = true;
                }
                rvec_inc(accumulator.force[ai], force);
                rvec_dec(accumulator.force[aj], force);
                if ((shouldDumpExcludedPairWrite || shouldDumpControlPairWrite)
                    && accumulator.forceWithVirial != nullptr)
                {
                    const std::string prefix = shouldDumpExcludedPairWrite
                                                       ? "step0_outer_excluded_write_ord000"
                                                       : "step0_outer_pairs_write_ord"
                                                                 + std::to_string(pairOrdinal);
                    const std::string header =
                            "stage="
                            + std::string(shouldDumpExcludedPairWrite ? "first_excluded_outer_write_boundary"
                                                                      : "control_pairs_outer_write_boundary")
                            + " pair_list=" + std::string(isExcludedPairlist ? "excludedPairs" : "pairs")
                            + " ordinal=" + std::to_string(pairOrdinal) + " contribution=outer buffer=forceWithVirial"
                            + " accumulator_ptr=" + formatPointerValue(accumulator.force.data())
                            + " virial_ptr="
                            + formatPointerValue(accumulator.forceWithVirial->force_.data()) + " ai="
                            + std::to_string(ai) + " aj=" + std::to_string(aj) + " shift_index="
                            + std::to_string(shiftIndex) + " scalar=" + gmx::toString(scalar)
                            + " correction_scalar=" + gmx::toString(correctionScalar) + " qq="
                            + gmx::toString(qq) + " r=" + gmx::toString(r);
                    dumpRespaMergeTraceVector(pairWriteProofDirPath,
                                              (prefix + "_after.tsv").c_str(),
                                              header + " snapshot=after",
                                              accumulator.force);
                }

                if (!accumulator.shift.empty() && shiftIndex != c_centralShiftIndex)
                {
                    rvec_inc(accumulator.shift[shiftIndex], force);
                    rvec_dec(accumulator.shift[c_centralShiftIndex], force);
                }

                if (accumulator.accumulateEnergy)
                {
                    const int energyIndex = energyGroupPairIndex(ai, aj, *fr, mdatoms);
                    vdwEnergyTerms[energyIndex] += factorLj * rawLjEnergy;
                    coulEnergyTerms[energyIndex] += fullCoulombEnergy;
                }

                if (stepWork.computeVirial && accumulator.forceWithVirial != nullptr)
                {
                    accumulatePairVirial(dx, force, accumulator.virial);
                }
            }

            pairOrdinal++;
        }
    };

    PairDebugStats pairStats{ "pairs", 0, 0, 0, 0 };
    PairDebugStats excludedStats{ "excludedPairs", 0, 0, 0, 0 };
    int            listedPairsInPairlist         = 0;
    int            listedPairsInExcludedPairlist = 0;
    int            duplicatePairs                = 0;
    int            duplicateExcludedPairs        = 0;
    if (debugExactRespa)
    {
        std::unordered_set<uint64_t> uniquePairs;
        std::unordered_set<uint64_t> uniqueExcludedPairs;
        for (const auto& entry : plainPairlist.pairs)
        {
            duplicatePairs += !uniquePairs.insert(pairKey(entry.first.first, entry.first.second)).second;
            listedPairsInPairlist += listedPairKeys.count(pairKey(entry.first.first, entry.first.second)) != 0;
        }
        for (const auto& entry : plainPairlist.excludedPairs)
        {
            duplicateExcludedPairs +=
                    !uniqueExcludedPairs.insert(pairKey(entry.first.first, entry.first.second)).second;
            listedPairsInExcludedPairlist +=
                    listedPairKeys.count(pairKey(entry.first.first, entry.first.second)) != 0;
        }
    }
    processPairlist(plainPairlist.pairs,
                    1.0_real,
                    1.0_real,
                    [](const int, const int) { return true; },
                    debugExactRespa ? &pairStats : nullptr);
    if (dumpEarlyAccumTrace && outerAccumulator != nullptr && outerAccumulator->forceWithVirial != nullptr)
    {
        dumpRespaMergeTraceVector(earlyAccumTraceDirPath,
                                  "step0_level2_after_pairs_virial.tsv",
                                  "stage=after_pairs_dispatch mts_index=2 mts_user=3 buffer=forceWithVirial alias_with_shift="
                                          + std::string(outerAliasesShift ? "true" : "false"),
                                  outerAccumulator->forceWithVirial->force_);
    }
    /* PME real-space bookkeeping for exact PME parity requires all excluded pairs,
     * not only the listed 1-4 subset. */
    processPairlist(plainPairlist.excludedPairs,
                    0.0_real,
                    0.0_real,
                    [](const int, const int) { return true; },
                    debugExactRespa ? &excludedStats : nullptr);
    if (dumpDownstreamContract)
    {
        dumpedDownstreamContractTrace = true;
    }
    if (dumpEarlyAccumTrace && outerAccumulator != nullptr && outerAccumulator->forceWithVirial != nullptr)
    {
        dumpRespaMergeTraceVector(earlyAccumTraceDirPath,
                                  "step0_level2_after_excluded_pairs_virial.tsv",
                                  "stage=after_excluded_pairs_dispatch mts_index=2 mts_user=3 buffer=forceWithVirial alias_with_shift="
                                          + std::string(outerAliasesShift ? "true" : "false"),
                                  outerAccumulator->forceWithVirial->force_);
        dumpedEarlyAccumTrace = true;
    }
    if (dumpPairWriteProof)
    {
        dumpedPairWriteProof = true;
    }

    for (auto& accumulator : activeContributions)
    {
        if (!accumulator.accumulateEnergy)
        {
            continue;
        }

        for (int atom = 0; atom < fr->natoms_force_constr; ++atom)
        {
            const real charge = mdatoms.chargeA[atom];
            if (charge == 0.0_real)
            {
                continue;
            }

            const int energyIndex = energyGroupPairIndex(atom, atom, *fr, mdatoms);
            const real selfEnergy = -fr->ic->coulomb.epsfac * charge * charge * pmeSelfEnergy;
            coulEnergyTerms[energyIndex] += selfEnergy;
            if (debugExactRespa)
            {
                pairStats.selfEnergy += selfEnergy;
            }
        }
    }

    if (debugExactRespa)
    {
        std::fprintf(stderr,
                     "GMX_PCFF_RESPA_DEBUG listedPairOverlaps pairs=%d excludedPairs=%d duplicatePairs=%d duplicateExcludedPairs=%d\n",
                     listedPairsInPairlist,
                     listedPairsInExcludedPairlist,
                     duplicatePairs,
                     duplicateExcludedPairs);
        std::fprintf(stderr,
                     "GMX_PCFF_RESPA_DEBUG pairs count=%d lj=% .9f coul=% .9f qq=% .9f self=% .9f\n",
                     pairStats.count,
                     pairStats.ljEnergy,
                     pairStats.coulEnergy,
                     pairStats.qqSum,
                     pairStats.selfEnergy);
        std::fprintf(stderr,
                     "GMX_PCFF_RESPA_DEBUG excludedPairs count=%d lj=% .9f coul=% .9f qq=% .9f self=% .9f\n",
                     excludedStats.count,
                     excludedStats.ljEnergy,
                     excludedStats.coulEnergy,
                     excludedStats.qqSum,
                     excludedStats.selfEnergy);
    }

    if (dumpExcludedCorrectionForce)
    {
        static bool dumpedExcludedCorrectionForce = false;
        if (!dumpedExcludedCorrectionForce)
        {
            const int outerMtsLevel =
                    nonbondedRespaContributionMtsLevel(inputrec, MtsNonbondedRespaContribution::Outer);
            FILE* dumpFile = std::fopen(excludedCorrectionForceDumpPath, "w");
            if (dumpFile == nullptr)
            {
                gmx_fatal(FARGS,
                          "Could not open GMX_PCFF_RESPA_EXCLUDED_FORCE_DUMP_FILE '%s' for writing",
                          excludedCorrectionForceDumpPath);
            }
            std::fprintf(
                    dumpFile,
                    "# contribution=excluded_pairs_coulomb_correction level_user=%d level_index=%d initial_buffer=%s postprocess=postProcessForces final_merge=combineMtsForces\n",
                    outerMtsLevel + 1,
                    outerMtsLevel,
                    stepWork.computeVirial ? "forceWithVirial" : "forceWithShiftForces");
            for (int atom = 0; atom < static_cast<int>(excludedCorrectionForce.size()); ++atom)
            {
                std::fprintf(dumpFile,
                             "%d\t%.17g\t%.17g\t%.17g\n",
                             atom,
                             excludedCorrectionForce[atom][XX],
                             excludedCorrectionForce[atom][YY],
                             excludedCorrectionForce[atom][ZZ]);
            }
            std::fclose(dumpFile);
            dumpedExcludedCorrectionForce = true;
        }
    }

    for (auto& accumulator : activeContributions)
    {
        if (accumulator.forceWithVirial != nullptr)
        {
            accumulator.forceWithVirial->addVirialContribution(accumulator.virial);
        }
    }
}

static inline void clearRVecs(ArrayRef<RVec> v, const bool useOpenmpThreading)
{
    int nth = gmx_omp_nthreads_get_simple_rvec_task(ModuleMultiThread::Default, v.ssize());

    /* Note that we would like to avoid this conditional by putting it
     * into the omp pragma instead, but then we still take the full
     * omp parallel for overhead (at least with gcc5).
     */
    if (!useOpenmpThreading || nth == 1)
    {
        for (RVec& elem : v)
        {
            clear_rvec(elem);
        }
    }
    else
    {
#pragma omp parallel for num_threads(nth) schedule(static)
        for (Index i = 0; i < v.ssize(); i++)
        {
            clear_rvec(v[i]);
        }
    }
}

/*! \brief Return an estimate of the average kinetic energy or 0 when unreliable
 *
 * \param groupOptions  Group options, containing T-coupling options
 */
static real averageKineticEnergyEstimate(const t_grpopts& groupOptions)
{
    real nrdfCoupled   = 0;
    real nrdfUncoupled = 0;
    real kineticEnergy = 0;
    for (int g = 0; g < groupOptions.ngtc; g++)
    {
        if (groupOptions.tau_t[g] >= 0)
        {
            nrdfCoupled += groupOptions.nrdf[g];
            kineticEnergy += groupOptions.nrdf[g] * 0.5 * groupOptions.ref_t[g] * c_boltz;
        }
        else
        {
            nrdfUncoupled += groupOptions.nrdf[g];
        }
    }

    /* This conditional with > also catches nrdf=0 */
    if (nrdfCoupled > nrdfUncoupled)
    {
        return kineticEnergy * (nrdfCoupled + nrdfUncoupled) / nrdfCoupled;
    }
    else
    {
        return 0;
    }
}

/*! \brief This routine checks that the potential energy is finite.
 *
 * Always checks that the potential energy is finite. If step equals
 * inputrec.init_step also checks that the magnitude of the potential energy
 * is reasonable. Terminates with a fatal error when a check fails.
 * Note that passing this check does not guarantee finite forces,
 * since those use slightly different arithmetics. But in most cases
 * there is just a narrow coordinate range where forces are not finite
 * and energies are finite.
 *
 * \param[in] step      The step number, used for checking and printing
 * \param[in] enerd     The energy data; the non-bonded group energies need to be added to
 *                      \c enerd.term[InteractionFunction::PotentialEnergy] before calling this
 * routine \param[in] inputrec  The input record
 */
static void checkPotentialEnergyValidity(int64_t step, const gmx_enerdata_t& enerd, const t_inputrec& inputrec)
{
    /* Threshold valid for comparing absolute potential energy against
     * the kinetic energy. Normally one should not consider absolute
     * potential energy values, but with a factor of one million
     * we should never get false positives.
     */
    constexpr real c_thresholdFactor = 1e6;

    bool energyIsNotFinite    = !std::isfinite(enerd.term[InteractionFunction::PotentialEnergy]);
    real averageKineticEnergy = 0;
    /* We only check for large potential energy at the initial step,
     * because that is by far the most likely step for this too occur
     * and because computing the average kinetic energy is not free.
     * Note: nstcalcenergy >> 1 often does not allow to catch large energies
     * before they become NaN.
     */
    if (step == inputrec.init_step && EI_DYNAMICS(inputrec.eI))
    {
        averageKineticEnergy = averageKineticEnergyEstimate(inputrec.opts);
    }

    if (energyIsNotFinite
        || (averageKineticEnergy > 0
            && enerd.term[InteractionFunction::PotentialEnergy] > c_thresholdFactor * averageKineticEnergy))
    {
        GMX_THROW(InternalError(formatString(
                "Step %" PRId64
                ": The total potential energy is %g, which is %s. The LJ and electrostatic "
                "contributions to the energy are %g and %g, respectively. A %s potential energy "
                "can be caused by overlapping interactions in bonded interactions or very large%s "
                "coordinate values. Usually this is caused by a badly- or non-equilibrated initial "
                "configuration, incorrect interactions or parameters in the topology.",
                step,
                enerd.term[InteractionFunction::PotentialEnergy],
                energyIsNotFinite ? "not finite" : "extremely high",
                enerd.term[InteractionFunction::LennardJonesShortRange],
                enerd.term[InteractionFunction::CoulombShortRange],
                energyIsNotFinite ? "non-finite" : "very high",
                energyIsNotFinite ? " or Nan" : "")));
    }
}

/*! \brief Compute forces and/or energies for special algorithms
 *
 * The intention is to collect all calls to algorithms that compute
 * forces on local atoms only and that do not contribute to the local
 * virial sum (but add their virial contribution separately).
 * Eventually these should likely all become ForceProviders.
 * Within this function the intention is to have algorithms that do
 * global communication at the end, so global barriers within the MD loop
 * are as close together as possible.
 *
 * \param[in]     fplog            The log file
 * \param[in]     mpiComm          The communication object for my group
 * \param[in]     dd               Pointer to the domdec object, is nullptr when DD is not in use
 * \param[in]     inputrec         The input record
 * \param[in]     awh              The Awh module (nullptr if none in use).
 * \param[in]     enforcedRotation Enforced rotation module.
 * \param[in]     imdSession       The IMD session
 * \param[in]     pull_work        The pull work structure.
 * \param[in]     step             The current MD step
 * \param[in]     t                The current time
 * \param[in,out] wcycle           Wallcycle accounting struct
 * \param[in,out] forceProviders   Pointer to a list of force providers
 * \param[in]     box              The unit cell
 * \param[in]     x                The coordinates
 * \param[in]     mdatoms          Per atom properties
 * \param[in]     lambda           Array of free-energy lambda values
 * \param[in]     stepWork         Step schedule flags
 * \param[in,out] forceWithVirialMtsLevel0  Force and virial for MTS level0 forces
 * \param[in,out] forceWithVirialMtsLevel1  Force and virial for MTS level1 forces, can be nullptr
 * \param[in,out] enerd            Energy buffer
 * \param[in,out] ed               Essential dynamics pointer
 * \param[in]     didNeighborSearch  Tells if we did neighbor searching this step,
 *                                   used for ED sampling
 *
 * \todo Remove didNeighborSearch, which is used incorrectly.
 * \todo Convert all other algorithms called here to ForceProviders.
 */
static void computeSpecialForces(FILE*                fplog,
                                 const MpiComm&       mpiComm,
                                 const gmx_domdec_t*  dd,
                                 const t_inputrec&    inputrec,
                                 Awh*                 awh,
                                 gmx_enfrot*          enforcedRotation,
                                 ImdSession*          imdSession,
                                 pull_t*              pull_work,
                                 int64_t              step,
                                 double               t,
                                 gmx_wallcycle*       wcycle,
                                 ForceProviders*      forceProviders,
                                 const matrix         box,
                                 ArrayRef<const RVec> x,
                                 const t_mdatoms*     mdatoms,
                                 ArrayRef<const real> lambda,
                                 const StepWorkload&  stepWork,
                                 ArrayRef<ForceWithVirial*> forceWithVirialPerMtsLevel,
                                 gmx_enerdata_t*      enerd,
                                 gmx_edsam*           ed,
                                 bool                 didNeighborSearch)
{
    /* NOTE: Currently all ForceProviders only provide forces.
     *       When they also provide energies, remove this conditional.
     */
    if (stepWork.computeForces)
    {
        ForceProviderInput  forceProviderInput(x,
                                              mdatoms->homenr,
                                              makeArrayRef(mdatoms->chargeA).subArray(0, mdatoms->homenr),
                                              makeArrayRef(mdatoms->massT).subArray(0, mdatoms->homenr),
                                              t,
                                              step,
                                              box,
                                              mpiComm,
                                              dd);
        ForceProviderOutput forceProviderOutput(forceWithVirialPerMtsLevel[0], enerd);

        /* Collect forces from modules */
        forceProviders->calculateForces(forceProviderInput, &forceProviderOutput);
    }

    const int  pullMtsLevel = forceGroupMtsLevel(inputrec.mtsLevels, MtsForceGroups::Pull);
    const bool doPulling    = (inputrec.bPull && pull_have_potential(*pull_work)
                            && pullMtsLevel <= stepWork.highestActiveMtsLevel);

    /* pull_potential_wrapper(), awh->applyBiasForcesAndUpdateBias(), pull_apply_forces()
     * have to be called in this order
     */
    // Note: this condition is mirrored in haveSpecialForces()
    if (doPulling)
    {
        pull_potential_wrapper(
                mpiComm, inputrec, box, x, mdatoms, enerd, pull_work, lambda.data(), t, wcycle);
    }
    // Note: the awh condition is mirrored in haveSpecialForces()
    if (awh && pullMtsLevel <= stepWork.highestActiveMtsLevel)
    {
        const bool          needForeignEnergyDifferences = awh->needForeignEnergyDifferences(step);
        std::vector<double> foreignLambdaDeltaH, foreignLambdaDhDl;
        if (needForeignEnergyDifferences)
        {
            enerd->foreignLambdaTerms.finalizePotentialContributions(
                    enerd->dvdl_lin, lambda, *inputrec.fepvals);
            std::tie(foreignLambdaDeltaH, foreignLambdaDhDl) = enerd->foreignLambdaTerms.getTerms(mpiComm);
        }

        enerd->term[InteractionFunction::CenterOfMassPullingEnergy] += awh->applyBiasForcesAndUpdateBias(
                inputrec.pbcType, foreignLambdaDeltaH, foreignLambdaDhDl, box, t, step, wcycle, fplog);
    }
    // Note: this condition is mirrored in haveSpecialForces()
    if (doPulling)
    {
        wallcycle_start_nocount(wcycle, WallCycleCounter::PullPot);
        pull_apply_forces(pull_work, mdatoms->massT, mpiComm, forceWithVirialPerMtsLevel[pullMtsLevel]);
        wallcycle_stop(wcycle, WallCycleCounter::PullPot);
    }

    /* Add the forces from enforced rotation potentials (if any) */
    // Note: this condition is mirrored in haveSpecialForces()
    if (inputrec.bRot)
    {
        wallcycle_start(wcycle, WallCycleCounter::RotAdd);
        enerd->term[InteractionFunction::CenterOfMassPullingEnergy] +=
                add_rot_forces(enforcedRotation, forceWithVirialPerMtsLevel[0]->force_, mpiComm, step, t);
        wallcycle_stop(wcycle, WallCycleCounter::RotAdd);
    }

    // Note: this condition is mirrored in haveSpecialForces()
    if (ed)
    {
        /* Note that since init_edsam() is called after the initialization
         * of forcerec, edsam doesn't request the noVirSum force buffer.
         * Thus if no other algorithm (e.g. PME) requires it, the forces
         * here will contribute to the virial.
         */
        do_flood(mpiComm, inputrec, x, forceWithVirialPerMtsLevel[0]->force_, ed, box, step, didNeighborSearch);
    }

    /* Add forces from interactive molecular dynamics (IMD), if any */
    // Note: this condition is mirrored in haveSpecialForces()
    if (inputrec.bIMD && stepWork.computeForces)
    {
        imdSession->applyForces(forceWithVirialPerMtsLevel[0]->force_);
    }
}

/*! \brief Launch the prepare_step and spread stages of PME GPU.
 *
 * \param[in]  pmedata              The PME structure
 * \param[in]  box                  The box matrix
 * \param[in]  simulationWork       Simulation schedule flags
 * \param[in]  stepWork             Step schedule flags
 * \param[in]  xReadyOnDevice       Event synchronizer indicating that the coordinates are ready in the device memory.
 * \param[in]  lambdaQ              The Coulomb lambda of the current state.
 * \param[in]  useMdGpuGraph        Whether MD GPU Graph is in use.
 * \param[in]  wcycle               The wallcycle structure
 */
static inline void launchPmeGpuSpread(gmx_pme_t*                pmedata,
                                      const matrix              box,
                                      const SimulationWorkload& simulationWork,
                                      const StepWorkload&       stepWork,
                                      GpuEventSynchronizer*     xReadyOnDevice,
                                      const real                lambdaQ,
                                      bool                      useMdGpuGraph,
                                      gmx_wallcycle*            wcycle)
{
    wallcycle_start(wcycle, WallCycleCounter::PmeGpuMesh);
    pme_gpu_prepare_computation(pmedata, box, simulationWork.haveDynamicBox, stepWork);
    bool                      useGpuDirectComm         = false;
    PmeCoordinateReceiverGpu* pmeCoordinateReceiverGpu = nullptr;
    pme_gpu_launch_spread(
            pmedata, xReadyOnDevice, wcycle, lambdaQ, useGpuDirectComm, pmeCoordinateReceiverGpu, useMdGpuGraph);
    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
}

/*! \brief Launch the FFT and gather stages of PME GPU
 *
 * This function only implements setting the output forces (no accumulation).
 *
 * \param[in]  pmedata        The PME structure
 * \param[in]  lambdaQ        The Coulomb lambda of the current system state.
 * \param[in]  wcycle         The wallcycle structure
 * \param[in]  stepWork       Step schedule flags
 */
static void launchPmeGpuFftAndGather(gmx_pme_t*          pmedata,
                                     const real          lambdaQ,
                                     gmx_wallcycle*      wcycle,
                                     const StepWorkload& stepWork)
{
    wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
    pme_gpu_launch_complex_transforms(pmedata, wcycle, stepWork);
    pme_gpu_launch_gather(pmedata, wcycle, lambdaQ, stepWork.computeVirial);
    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
}

/*! \brief
 * Blocks until PME GPU tasks are completed, and gets the output forces and virial/energy
 * (if they were to be computed).
 *
 * \param[in]  pme             The PME data structure.
 * \param[in]  stepWork        The required work for this simulation step
 * \param[in]  wcycle          The wallclock counter.
 * \param[out] forceWithVirial The output force and virial
 * \param[out] enerd           The output energies
 * \param[in]  lambdaQ         The Coulomb lambda to use when calculating the results.
 */
static void pmeGpuWaitAndReduce(gmx_pme_t*          pme,
                                const StepWorkload& stepWork,
                                gmx_wallcycle*      wcycle,
                                ForceWithVirial*    forceWithVirial,
                                gmx_enerdata_t*     enerd,
                                const real          lambdaQ)
{
    wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);

    pme_gpu_wait_and_reduce(pme, stepWork, wcycle, forceWithVirial, enerd, lambdaQ);

    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
}

/*! \brief
 *  Polling wait for either of the PME or nonbonded GPU tasks.
 *
 * Instead of a static order in waiting for GPU tasks, this function
 * polls checking which of the two tasks completes first, and does the
 * associated force buffer reduction overlapped with the other task.
 * By doing that, unlike static scheduling order, it can always overlap
 * one of the reductions, regardless of the GPU task completion order.
 *
 * \param[in]     nbv              Nonbonded verlet structure
 * \param[in,out] pmedata          PME module data
 * \param[in,out] forceOutputsNonbonded  Force outputs for the non-bonded forces and shift forces
 * \param[in,out] forceOutputsPme  Force outputs for the PME forces and virial
 * \param[in,out] enerd            Energy data structure results are reduced into
 * \param[in]     lambdaQ          The Coulomb lambda of the current system state.
 * \param[in]     stepWork         Step schedule flags
 * \param[in]     simulationWork   Simulation schedule flags
 * \param[in]     wcycle           The wallcycle structure
 */
static void alternatePmeNbGpuWaitReduce(nonbonded_verlet_t*       nbv,
                                        gmx_pme_t*                pmedata,
                                        ForceOutputs*             forceOutputsNonbonded,
                                        ForceOutputs*             forceOutputsPme,
                                        gmx_enerdata_t*           enerd,
                                        const real                lambdaQ,
                                        const StepWorkload&       stepWork,
                                        const SimulationWorkload& simulationWork,
                                        gmx_wallcycle*            wcycle)
{
    bool isPmeGpuDone = false;
    bool isNbGpuDone  = false;

    while (!isPmeGpuDone || !isNbGpuDone)
    {
        if (!isPmeGpuDone)
        {
            wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
            GpuTaskCompletion completionType =
                    (isNbGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
            isPmeGpuDone = pme_gpu_try_finish_task(
                    pmedata, stepWork, wcycle, &forceOutputsPme->forceWithVirial(), enerd, lambdaQ, completionType);
            wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
        }

        if (!isNbGpuDone)
        {
            auto&             forceBuffersNonbonded = forceOutputsNonbonded->forceWithShiftForces();
            GpuTaskCompletion completionType =
                    (isPmeGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
            // To get the wcycle call count right, when in GpuTaskCompletion::Check mode,
            // we start without counting and only when the task finished we issue a
            // start/stop to increment.
            // GpuTaskCompletion::Wait mode the timing is expected to be done in the caller.
            wallcycle_start_nocount(wcycle, WallCycleCounter::WaitGpuNbL);
            isNbGpuDone = gpu_try_finish_task(
                    nbv->gpuNbv(),
                    stepWork,
                    AtomLocality::Local,
                    enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJSR].data(),
                    enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR].data(),
                    (simulationWork.useGpuForeignNonbondedFE)
                            ? &enerd->dvdl_nonlin[FreeEnergyPerturbationCouplingType::Vdw]
                            : &enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Vdw],
                    (simulationWork.useGpuForeignNonbondedFE)
                            ? &enerd->dvdl_nonlin[FreeEnergyPerturbationCouplingType::Coul]
                            : &enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Coul],
                    forceBuffersNonbonded.shiftForces(),
                    &enerd->foreignLambdaTerms,
                    completionType);
            wallcycle_stop(wcycle, WallCycleCounter::WaitGpuNbL);

            if (isNbGpuDone)
            {
                wallcycle_increment_event_count(wcycle, WallCycleCounter::WaitGpuNbL);
                nbv->atomdata_add_nbat_f_to_f(AtomLocality::Local, forceBuffersNonbonded.force());
            }
        }
    }
}

/*! \brief Set up the different force buffers; also does clearing.
 *
 * \param[in] forceHelperBuffers        Helper force buffers
 * \param[in] force                     force array
 * \param[in] domainWork                Domain lifetime workload flags
 * \param[in] stepWork                  Step schedule flags
 * \param[in] havePpDomainDecomposition Whether we have a PP domain decomposition
 * \param[out] wcycle                   wallcycle recording structure
 *
 * \returns                             Cleared force output structure
 */
static ForceOutputs setupForceOutputs(ForceHelperBuffers*           forceHelperBuffers,
                                      ArrayRefWithPadding<RVec>     force,
                                      const DomainLifetimeWorkload& domainWork,
                                      const StepWorkload&           stepWork,
                                      const bool                    havePpDomainDecomposition,
                                      gmx_wallcycle*                wcycle)
{
    /* NOTE: We assume fr->shiftForces is all zeros here */
    ForceWithShiftForces forceWithShiftForces(
            force, stepWork.computeVirial, forceHelperBuffers->shiftForces());

    if (stepWork.computeForces
        && (domainWork.haveCpuLocalForceWork || !stepWork.useGpuFBufferOps
            || (havePpDomainDecomposition && !stepWork.useGpuFHalo)))
    {
        wallcycle_sub_start(wcycle, WallCycleSubCounter::ClearForceBuffer);
        /* Clear the short- and long-range forces */
        clearRVecs(forceWithShiftForces.force(), true);

        /* Clear the shift forces */
        clearRVecs(forceWithShiftForces.shiftForces(), false);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::ClearForceBuffer);
    }

    /* If we need to compute the virial, we might need a separate
     * force buffer for algorithms for which the virial is calculated
     * directly, such as PME. Otherwise, forceWithVirial uses the
     * the same force (f in legacy calls) buffer as other algorithms.
     */
    const bool useSeparateForceWithVirialBuffer =
            (stepWork.computeForces
             && (stepWork.computeVirial && forceHelperBuffers->haveDirectVirialContributions()));
    /* forceWithVirial uses the local atom range only */
    ForceWithVirial forceWithVirial(useSeparateForceWithVirialBuffer
                                            ? forceHelperBuffers->forceBufferForDirectVirialContributions()
                                            : force.unpaddedArrayRef(),
                                    stepWork.computeVirial);

    if (useSeparateForceWithVirialBuffer)
    {
        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::ClearForceBuffer);
        /* TODO: update comment
         * We only compute forces on local atoms. Note that vsites can
         * spread to non-local atoms, but that part of the buffer is
         * cleared separately in the vsite spreading code.
         */
        clearRVecs(forceWithVirial.force_, true);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::ClearForceBuffer);
    }


    return ForceOutputs(
            forceWithShiftForces, forceHelperBuffers->haveDirectVirialContributions(), forceWithVirial);
}

/* \brief Launch end-of-step GPU tasks: buffer clearing and rolling pruning.
 *
 */
static void launchGpuEndOfStepTasks(nonbonded_verlet_t*          nbv,
                                    ListedForcesGpu*             listedForcesGpu,
                                    gmx_pme_t*                   pmedata,
                                    gmx_enerdata_t*              enerd,
                                    const MdrunScheduleWorkload& runScheduleWork,
                                    int64_t                      step,
                                    gmx_wallcycle*               wcycle)
{
    if (runScheduleWork.simulationWork.useGpuNonbonded && runScheduleWork.stepWork.computeNonbondedForces)
    {
        /* Launch pruning before buffer clearing because the API overhead of the
         * clear kernel launches can leave the GPU idle while it could be running
         * the prune kernel.
         */
        if (nbv->isDynamicPruningStepGpu(step))
        {
            nbv->dispatchPruneKernelGpu(step);
        }

        /* now clear the GPU outputs while we finish the step on the CPU */
        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        gpu_clear_outputs(nbv->gpuNbv(), runScheduleWork.stepWork.computeVirial);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
    }

    if (runScheduleWork.stepWork.haveGpuPmeOnThisRank)
    {
        wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
        bool gpuGraphWithSeparatePmeRank = false;
        pme_gpu_finish_step(pmedata, gpuGraphWithSeparatePmeRank, wcycle);
        wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
    }

    if (runScheduleWork.domainWork.haveGpuBondedWork && runScheduleWork.stepWork.computeEnergy)
    {
        // in principle this should be included in the DD balancing region,
        // but generally it is infrequent so we'll omit it for the sake of
        // simpler code
        listedForcesGpu->waitAccumulateEnergyTerms(enerd);

        listedForcesGpu->clearEnergies();
    }
}

/*! \brief Compute the number of times the "local coordinates ready on device" GPU event will be used as a synchronization point.
 *
 * When some work is offloaded to GPU, force calculation should wait for the atom coordinates to
 * be ready on the device. The coordinates can come either from H2D copy at the beginning of the step,
 * or from the GPU integration at the end of the previous step.
 *
 * In GROMACS, we usually follow the "mark once - wait once" approach. But this event is "consumed"
 * (that is, waited upon either on host or on the device) multiple times, since many tasks
 * in different streams depend on the coordinates.
 *
 * This function return the number of times the event will be consumed based on this step's workload.
 *
 * \param simulationWork            Simulation workload flags.
 * \param stepWork                  Step workload flags.
 * \param domainWork                Domain workload flags.
 * \param pmeSendCoordinatesFromGpu Whether peer-to-peer communication is used for PME coordinates.
 * \return
 */
static int getExpectedLocalXReadyOnDeviceConsumptionCount(const SimulationWorkload& simulationWork,
                                                          const StepWorkload&       stepWork,
                                                          const DomainLifetimeWorkload& domainWork,
                                                          const bool pmeSendCoordinatesFromGpu)
{
    int result = 0;
    if (stepWork.computeLongRangeNonbondedForces)
    {
        if (pmeSendCoordinatesFromGpu)
        {
            GMX_ASSERT(simulationWork.haveSeparatePmeRank,
                       "GPU PME PP communications require having a separate PME rank");
            // Event is consumed by gmx_pme_send_coordinates for GPU
            // PME PP Communications when the domain has home atoms.
            result++;
        }
        if (stepWork.haveGpuPmeOnThisRank)
        {
            // Event is consumed by launchPmeGpuSpread
            result++;
        }
        if (stepWork.computeNonbondedForces && stepWork.useGpuXBufferOps)
        {
            // Event is consumed by convertCoordinatesGpu
            result++;
        }
    }
    if (stepWork.useGpuXHalo)
    {
        // Event is consumed by communicateGpuHaloCoordinates
        result++;
        if (GMX_THREAD_MPI) // Issue #4262
        {
            result++;
        }
    }
    if (stepWork.clearGpuFBufferEarly && simulationWork.useGpuUpdate)
    {
        // Event is consumed by force clearing which waits for the update to complete
        result++;
    }
    if (simulationWork.useGpuUpdate && domainWork.haveCpuLocalForceWork)
    {
        // Event is consumed when waiting for it on the CPU prior to CPU force buffer clearing.
        // The actual data dependency does not involve coordinates, but we use this event
        // as an "end-of-step" mark.
        if (!(stepWork.doNeighborSearch || simulationWork.useCpuHaloExchange
              || (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu)))
        {
            result++;
        }
    }
    return result;
}

/*! \brief Compute the number of times the "local forces ready on device" GPU event will be used as a synchronization point.
 *
 * In GROMACS, we usually follow the "mark once - wait once" approach. But this event is "consumed"
 * (that is, waited upon either on host or on the device) multiple times, since many tasks
 * in different streams depend on the local forces.
 *
 * \param simulationWork Simulation workload flags.
 * \param domainWork Domain workload flags.
 * \param stepWork Step workload flags.
 * \param useOrEmulateGpuNb Whether GPU non-bonded calculations are used or emulated.
 * \param alternateGpuWait Whether alternating wait/reduce scheme is used.
 * \return The number of times the event will be consumed based on this step's workload.
 */
static int getExpectedLocalFReadyOnDeviceConsumptionCount(const SimulationWorkload& simulationWork,
                                                          const DomainLifetimeWorkload& domainWork,
                                                          const StepWorkload&           stepWork,
                                                          bool useOrEmulateGpuNb,
                                                          bool alternateGpuWait)
{
    int  counter = 0;
    bool eventUsedInGpuForceReduction =
            (domainWork.haveCpuLocalForceWork
             || (simulationWork.havePpDomainDecomposition && !simulationWork.useGpuHaloExchange));
    bool gpuForceReductionUsed = useOrEmulateGpuNb && !alternateGpuWait && stepWork.useGpuFBufferOps
                                 && stepWork.computeNonbondedForces;
    if (gpuForceReductionUsed && eventUsedInGpuForceReduction)
    {
        counter++;
    }
    bool gpuForceHaloUsed = simulationWork.havePpDomainDecomposition && stepWork.computeForces
                            && stepWork.useGpuFHalo;
    if (gpuForceHaloUsed)
    {
        counter++;
    }
    return counter;
}

//! \brief Data structure to hold dipole-related data and staging arrays
struct DipoleData
{
    //! Dipole staging for fast summing over MPI
    DVec muStaging[2] = { { 0.0, 0.0, 0.0 } };
    //! Dipole staging for states A and B (index 0 and 1 resp.)
    RVec muStateAB[2] = { { 0.0_real, 0.0_real, 0.0_real } };
};


static void reduceAndUpdateMuTot(DipoleData*                   dipoleData,
                                 const MpiComm&                mpiComm,
                                 const bool                    haveFreeEnergy,
                                 ArrayRef<const real>          lambda,
                                 rvec                          muTotal,
                                 const DDBalanceRegionHandler& ddBalanceRegionHandler)
{
    if (mpiComm.isParallel())
    {
        mpiComm.sumReduce(2 * DIM, dipoleData->muStaging[0]);
        ddBalanceRegionHandler.reopenRegionCpu();
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            dipoleData->muStateAB[i][j] = dipoleData->muStaging[i][j];
        }
    }

    if (!haveFreeEnergy)
    {
        copy_rvec(dipoleData->muStateAB[0], muTotal);
    }
    else
    {
        for (int j = 0; j < DIM; j++)
        {
            muTotal[j] = (1.0 - lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)])
                                 * dipoleData->muStateAB[0][j]
                         + lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)]
                                   * dipoleData->muStateAB[1][j];
        }
    }
}

/*! \brief Combines MTS level0..N force buffers into a physical and an impulse-combined force buffer.
 *
 * \param[in]     numAtoms              The number of atoms to combine forces for
 * \param[in,out] forceMtsLevel0        Input: F_level0, output: sum_j F_levelj
 * \param[in,out] forceMtsCombined      Output: F_level0 + sum_{j>0} factor_j * F_levelj
 * \param[in]     slowLevelForces       Force buffers for the active non-zero MTS levels
 * \param[in]     slowLevelStepFactors  Absolute step factors for the active non-zero MTS levels
 */
static void combineMtsForces(const int                                numAtoms,
                             ArrayRef<RVec>                           forceMtsLevel0,
                             ArrayRef<RVec>                           forceMtsCombined,
                             const std::vector<ArrayRef<const RVec>>& slowLevelForces,
                             ArrayRef<const int>                      slowLevelStepFactors)
{
    GMX_RELEASE_ASSERT(slowLevelForces.size() == slowLevelStepFactors.size(),
                       "Need the same number of slow force buffers and MTS factors");

    const int gmx_unused numThreads = gmx_omp_nthreads_get(ModuleMultiThread::Default);
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < numAtoms; i++)
    {
        RVec physicalForce = forceMtsLevel0[i];
        RVec combinedForce = forceMtsLevel0[i];
        for (int slowLevelIndex = 0; slowLevelIndex < static_cast<int>(slowLevelForces.size()); slowLevelIndex++)
        {
            physicalForce += slowLevelForces[slowLevelIndex][i];
            combinedForce += static_cast<real>(slowLevelStepFactors[slowLevelIndex])
                             * slowLevelForces[slowLevelIndex][i];
        }
        forceMtsLevel0[i]   = physicalForce;
        forceMtsCombined[i] = combinedForce;
    }
}

/*! \brief Setup for the local GPU force reduction:
 * reinitialization plus the registration of forces and dependencies.
 *
 * \param [in] runScheduleWork     Schedule workload flag structure
 * \param [in] nbv                 Non-bonded Verlet object
 * \param [in] stateGpu            GPU state propagator object
 * \param [in] gpuForceReduction   GPU force reduction object
 * \param [in] pmePpCommGpu        PME-PP GPU communication object
 * \param [in] pmedata             PME data object
 * \param [in] dd                  Domain decomposition object
 */
static void setupLocalGpuForceReduction(const MdrunScheduleWorkload& runScheduleWork,
                                        nonbonded_verlet_t*          nbv,
                                        StatePropagatorDataGpu*      stateGpu,
                                        GpuForceReduction*           gpuForceReduction,
                                        PmePpCommGpu*                pmePpCommGpu,
                                        const gmx_pme_t*             pmedata,
                                        const gmx_domdec_t*          dd)
{
    GMX_ASSERT(!runScheduleWork.simulationWork.useMts,
               "GPU force reduction is not compatible with MTS");

    // (re-)initialize local GPU force reduction
    const bool accumulate = runScheduleWork.domainWork.haveCpuLocalForceWork
                            || runScheduleWork.simulationWork.havePpDomainDecomposition;
    const int atomStart = 0;
    gpuForceReduction->reinit(stateGpu->getForces(),
                              nbv->getNumAtoms(AtomLocality::Local),
                              nbv->getGridIndices(),
                              atomStart,
                              accumulate,
                              stateGpu->fReducedOnDevice(AtomLocality::Local));

    // register forces and add dependencies
    gpuForceReduction->registerNbnxmForce(gpu_get_f(nbv->gpuNbv()));

    DeviceBuffer<RVec>    pmeForcePtr;
    GpuEventSynchronizer* pmeSynchronizer     = nullptr;
    bool                  havePmeContribution = false;

    if (runScheduleWork.simulationWork.haveGpuPmeOnPpRank())
    {
        pmeForcePtr = pme_gpu_get_device_f(pmedata);
        if (pmeForcePtr)
        {
            pmeSynchronizer     = pme_gpu_get_f_ready_synchronizer(pmedata);
            havePmeContribution = true;
        }
    }
    else if (runScheduleWork.simulationWork.useGpuPmePpCommunication)
    {
        std::optional<DeviceBuffer<RVec>> pmeStagingPtr = pmePpCommGpu->getGpuForceStagingPtr();
        if (pmeStagingPtr)
        {
            pmeForcePtr = pmeStagingPtr.value();
            GMX_ASSERT(pmeForcePtr, "PME force for reduction has no data");
            if (GMX_THREAD_MPI)
            {
                pmeSynchronizer = pmePpCommGpu->getForcesReadySynchronizer().value();
            }
            havePmeContribution = true;
        }
    }

    if (havePmeContribution)
    {
        gpuForceReduction->registerRvecForce(pmeForcePtr);
        if (runScheduleWork.simulationWork.useNvshmem)
        {
            DeviceBuffer<uint64_t> forcesReadyNvshmemFlags = pmePpCommGpu->getGpuForcesSyncObj();
            gpuForceReduction->registerForcesReadyNvshmemFlags(forcesReadyNvshmemFlags);
        }

        if (!runScheduleWork.simulationWork.useGpuPmePpCommunication || GMX_THREAD_MPI)
        {
            GMX_ASSERT(pmeSynchronizer != nullptr, "PME force ready cuda event should not be NULL");
            gpuForceReduction->addDependency(pmeSynchronizer);
        }
    }

    if (runScheduleWork.domainWork.haveCpuLocalForceWork
        || (runScheduleWork.simulationWork.havePpDomainDecomposition
            && !runScheduleWork.simulationWork.useGpuHaloExchange))
    {
        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::Local));
    }

    if (runScheduleWork.simulationWork.useGpuHaloExchange)
    {
        if (runScheduleWork.simulationWork.useNvshmem)
        {
            GMX_RELEASE_ASSERT(dd->gpuHaloExchangeNvshmemHelper != nullptr,
                               "NVSHMEM helper should be initialized when using NVSHMEM");
            gpuForceReduction->addDependency(
                    dd->gpuHaloExchangeNvshmemHelper->getForcesReadyOnDeviceEvent());
        }
        else
        {
            gpuForceReduction->addDependency(dd->gpuHaloExchange[0][0]->getForcesReadyOnDeviceEvent());
        }
    }
}

/*! \brief Setup for the non-local GPU force reduction:
 * reinitialization plus the registration of forces and dependencies.
 *
 * \param [in] runScheduleWork     Schedule workload flag structure
 * \param [in] nbv                 Non-bonded Verlet object
 * \param [in] stateGpu            GPU state propagator object
 * \param [in] gpuForceReduction   GPU force reduction object
 * \param [in] dd                  Domain decomposition object
 */
static void setupNonLocalGpuForceReduction(const MdrunScheduleWorkload& runScheduleWork,
                                           nonbonded_verlet_t*          nbv,
                                           StatePropagatorDataGpu*      stateGpu,
                                           GpuForceReduction*           gpuForceReduction,
                                           const gmx_domdec_t*          dd)
{
    // (re-)initialize non-local GPU force reduction
    const bool accumulate = runScheduleWork.domainWork.haveCpuNonLocalForceWork;
    const int  atomStart  = dd_numHomeAtoms(*dd);
    gpuForceReduction->reinit(stateGpu->getForces(),
                              nbv->getNumAtoms(AtomLocality::NonLocal),
                              nbv->getGridIndices(),
                              atomStart,
                              accumulate,
                              stateGpu->fReducedOnDevice(AtomLocality::NonLocal));

    // register forces and add dependencies
    gpuForceReduction->registerNbnxmForce(gpu_get_f(nbv->gpuNbv()));

    if (runScheduleWork.domainWork.haveCpuNonLocalForceWork)
    {
        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::NonLocal));
    }
}


/*! \brief Return the number of local atoms.
 */
static int getLocalAtomCount(const gmx_domdec_t* dd, const t_mdatoms& mdatoms, bool havePPDomainDecomposition)
{
    GMX_ASSERT(!(havePPDomainDecomposition && (dd == nullptr)),
               "Can't have PP decomposition with dd uninitialized!");
    return havePPDomainDecomposition ? dd_numAtomsZones(*dd) : mdatoms.homenr;
}

/*! \brief Does pair search and closely related activities required on search steps.
 */
static void doPairSearch(const t_commrec*             cr,
                         const t_inputrec&            inputrec,
                         const MDModulesNotifiers&    mdModulesNotifiers,
                         int64_t                      step,
                         t_nrnb*                      nrnb,
                         gmx_wallcycle*               wcycle,
                         const gmx_localtop_t&        top,
                         const matrix                 box,
                         ArrayRefWithPadding<RVec>    x,
                         ArrayRef<RVec>               v,
                         const t_mdatoms&             mdatoms,
                         t_forcerec*                  fr,
                         const MdrunScheduleWorkload& runScheduleWork)
{
    nonbonded_verlet_t* nbv = fr->nbv.get();

    StatePropagatorDataGpu* stateGpu = fr->stateGpu;

    const SimulationWorkload&     simulationWork = runScheduleWork.simulationWork;
    const StepWorkload&           stepWork       = runScheduleWork.stepWork;
    const DomainLifetimeWorkload& domainWork     = runScheduleWork.domainWork;

    if (needStateGpu(simulationWork))
    {
        // TODO refactor this to do_md, after partitioning.
        //
        // Does global communication and symmetric reallocation with NVSHMEM
        stateGpu->reinit(mdatoms.homenr,
                         getLocalAtomCount(cr->dd, mdatoms, simulationWork.havePpDomainDecomposition),
                         cr->commMySim.comm());
        if (simulationWork.useGpuHaloExchange && runScheduleWork.simulationWork.useNvshmem)
        {
            // Does global communication and symmetric reallocation
            reinitGpuHaloExchangeNvshmem(*cr);
        }
    }

    if (simulationWork.haveGpuPmeOnPpRank())
    {
        GMX_ASSERT(needStateGpu(simulationWork), "StatePropagatorDataGpu is needed");
        // TODO: This should be moved into PME setup function ( pme_gpu_prepare_computation(...) )
        pme_gpu_set_device_x(fr->pmedata, stateGpu->getCoordinates());
    }

    if (fr->pbcType != PbcType::No)
    {
        const bool calcCGCM = (stepWork.stateChanged && !haveDDAtomOrdering(*cr));
        if (calcCGCM)
        {
            put_atoms_in_box_omp(fr->pbcType,
                                 box,
                                 fr->haveBoxDeformation,
                                 inputrec.deform,
                                 x.unpaddedArrayRef().subArray(0, mdatoms.homenr),
                                 v.empty() ? ArrayRef<RVec>{} : v.subArray(0, mdatoms.homenr),
                                 gmx_omp_nthreads_get(ModuleMultiThread::Default));
            inc_nrnb(nrnb, eNR_SHIFTX, mdatoms.homenr);
        }

        if (!haveDDAtomOrdering(*cr))
        {
            // Atoms might have changed periodic image, signal MDModules
            MDModulesAtomsRedistributedSignal mdModulesAtomsRedistributedSignal(
                    box, x.unpaddedArrayRef().subArray(0, mdatoms.homenr), std::nullopt);
            mdModulesNotifiers.simulationRunNotifier_.notify(mdModulesAtomsRedistributedSignal);
        }
    }

    if (fr->wholeMoleculeTransform && stepWork.stateChanged)
    {
        fr->wholeMoleculeTransform->updateForAtomPbcJumps(x.unpaddedArrayRef(), box);
    }

    wallcycle_start(wcycle, WallCycleCounter::NS);
    if (!haveDDAtomOrdering(*cr))
    {
        const rvec vzero       = { 0.0_real, 0.0_real, 0.0_real };
        const rvec boxDiagonal = { box[XX][XX], box[YY][YY], box[ZZ][ZZ] };
        wallcycle_sub_start(wcycle, WallCycleSubCounter::NBSGridLocal);
        nbv->putAtomsOnGrid(box,
                            0,
                            vzero,
                            boxDiagonal,
                            nullptr,
                            { 0, mdatoms.homenr },
                            mdatoms.homenr,
                            -1,
                            fr->atomInfo,
                            x.unpaddedArrayRef(),
                            nullptr);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::NBSGridLocal);
    }
    else
    {
        wallcycle_sub_start(wcycle, WallCycleSubCounter::NBSGridNonLocal);
        if (!nbv->localAtomOrderMatchesNbnxmOrder())
        {
            nbnxn_put_on_grid_nonlocal(nbv, getDomdecZones(*cr->dd), fr->atomInfo, x.unpaddedArrayRef());
        }
        nbv->convertCoordinates(AtomLocality::NonLocal, x.unpaddedArrayRef());
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::NBSGridNonLocal);
    }

    nbv->setAtomProperties(mdatoms.typeA, mdatoms.chargeA, fr->atomInfo, mdatoms.typeB, mdatoms.chargeB);

    wallcycle_stop(wcycle, WallCycleCounter::NS);

    /* initialize the GPU nbnxm atom data and bonded data structures */
    if (simulationWork.useGpuNonbonded)
    {
        // Note: cycle counting only nononbondeds, GPU listed forces counts internally
        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        gpu_init_atomdata(nbv->gpuNbv(), &nbv->nbat());
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);

        if (fr->listedForcesGpu)
        {
            /* Now we put all atoms on the grid, we can assign bonded
             * interactions to the GPU, where the grid order is
             * needed. Also the xq, f and fshift device buffers have
             * been reallocated if needed, so the bonded code can
             * learn about them. */
            // TODO the xq, f, and fshift buffers are now shared
            // resources, so they should be maintained by a
            // higher-level object than the nb module.
            fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
                    nbv->getGridIndices(), top.idef, gpuGetNBAtomData(nbv->gpuNbv()));
        }
    }

    wallcycle_start_nocount(wcycle, WallCycleCounter::NS);
    wallcycle_sub_start(wcycle, WallCycleSubCounter::NBSSearchLocal);
    /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
    const bool alsoMakePlainPairlist = fr->plainPairlistRange.has_value();
    nbv->constructPairlist(InteractionLocality::Local, top.excls, alsoMakePlainPairlist, step, nrnb);

    nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::Local);

    wallcycle_sub_stop(wcycle, WallCycleSubCounter::NBSSearchLocal);
    wallcycle_stop(wcycle, WallCycleCounter::NS);

    if (simulationWork.useGpuXBufferOpsWhenAllowed)
    {
        nbv->atomdata_init_copy_x_to_nbat_x_gpu();
    }

    if (simulationWork.useGpuFBufferOpsWhenAllowed)
    {
        // with MPI, direct GPU communication, and separate PME ranks we need
        // gmx_pme_send_coordinates() to be called before we can set up force reduction
        bool delaySetupLocalGpuForceReduction = GMX_MPI && simulationWork.useGpuPmePpCommunication;
        if (!delaySetupLocalGpuForceReduction)
        {
            setupLocalGpuForceReduction(runScheduleWork,
                                        nbv,
                                        stateGpu,
                                        fr->gpuForceReduction[AtomLocality::Local].get(),
                                        fr->pmePpCommGpu.get(),
                                        fr->pmedata,
                                        cr->dd);
        }

        if (simulationWork.havePpDomainDecomposition)
        {
            setupNonLocalGpuForceReduction(runScheduleWork,
                                           nbv,
                                           stateGpu,
                                           fr->gpuForceReduction[AtomLocality::NonLocal].get(),
                                           cr->dd);
        }
    }

    /* do non-local pair search */
    if (simulationWork.havePpDomainDecomposition)
    {
        wallcycle_start_nocount(wcycle, WallCycleCounter::NS);
        wallcycle_sub_start(wcycle, WallCycleSubCounter::NBSSearchNonLocal);
        /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
        nbv->constructPairlist(InteractionLocality::NonLocal, top.excls, alsoMakePlainPairlist, step, nrnb);

        nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::NonLocal);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::NBSSearchNonLocal);
        wallcycle_stop(wcycle, WallCycleCounter::NS);
        // TODO refactor this GPU halo exchange re-initialisation
        // to location in do_md where GPU halo exchange is
        // constructed at partitioning, after above stateGpu
        // re-initialization has similarly been refactored
        // because with NVSHMEM the ordering of synchronous
        // global operations must be preserved.
        if (simulationWork.useGpuHaloExchange)
        {
            reinitGpuHaloExchange(*cr, stateGpu->getCoordinates(), stateGpu->getForces());
        }
    }

    // With FEP we set up the reduction over threads for local+non-local simultaneously,
    // so we need to do that here after the local and non-local pairlist construction.
    if (domainWork.haveCpuNonbondedFreeEnergyWork)
    {
        wallcycle_sub_start(wcycle, WallCycleSubCounter::NonbondedFep);
        nbv->setupFepThreadedForceBuffer(fr->natoms_force_constr);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::NonbondedFep);
    }

    if (alsoMakePlainPairlist)
    {
        const auto& plainPairlist = nbv->plainPairlist(fr->plainPairlistRange.value(), fr->shift_vec);
        MDModulesPairlistConstructedSignal mdModulesPairlistConstructedSignal(
                plainPairlist.pairs, plainPairlist.excludedPairs, mdatoms.typeA);
        mdModulesNotifiers.simulationRunNotifier_.notify(mdModulesPairlistConstructedSignal);
    }
}

void do_force(FILE*                         fplog,
              const t_commrec*              cr,
              const t_inputrec&             inputrec,
              const MDModulesNotifiers&     mdModulesNotifiers,
              Awh*                          awh,
              gmx_enfrot*                   enforcedRotation,
              ImdSession*                   imdSession,
              pull_t*                       pull_work,
              int64_t                       step,
              t_nrnb*                       nrnb,
              gmx_wallcycle*                wcycle,
              const gmx_localtop_t*         top,
              const matrix                  box,
              ArrayRefWithPadding<RVec>     x,
              ArrayRef<RVec>                v,
              const history_t*              hist,
              ForceBuffersView*             forceView,
              tensor                        vir_force,
              const t_mdatoms*              mdatoms,
              gmx_enerdata_t*               enerd,
              ArrayRef<const real>          lambda,
              t_forcerec*                   fr,
              const MdrunScheduleWorkload&  runScheduleWork,
              VirtualSitesHandler*          vsite,
              rvec                          muTotal,
              double                        t,
              gmx_edsam*                    ed,
              CpuPpLongRangeNonbondeds*     longRangeNonbondeds,
              const DDBalanceRegionHandler& ddBalanceRegionHandler)
{
    ArrayRefWithPadding<RVec> force = forceView->forceWithPadding();
    GMX_ASSERT(force.unpaddedArrayRef().ssize() >= fr->natoms_force_constr,
               "The size of the force buffer should be at least the number of atoms to compute "
               "forces for");

    nonbonded_verlet_t*  nbv = fr->nbv.get();
    interaction_const_t* ic  = fr->ic.get();

    StatePropagatorDataGpu* stateGpu = fr->stateGpu;

    const SimulationWorkload& simulationWork = runScheduleWork.simulationWork;

    const DomainLifetimeWorkload& domainWork = runScheduleWork.domainWork;

    const StepWorkload& stepWork = runScheduleWork.stepWork;

    const bool pmeSendCoordinatesFromGpu =
            simulationWork.useGpuPmePpCommunication && !stepWork.doNeighborSearch;

    const bool reinitGpuPmePpComms = simulationWork.useGpuPmePpCommunication && stepWork.doNeighborSearch;
    if (stepWork.computePmeOnSeparateRank && stepWork.doNeighborSearch)
    {
        // We call the gmx_pme_send_coordinates early for reinit case
        // in order for nvshmem collective calls in StatePropagatorDataGpu::Impl::reinit
        // to be in sync with PME-PP
        gmx_pme_send_coordinates(fr,
                                 cr->dd,
                                 box,
                                 x.unpaddedArrayRef(),
                                 lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)],
                                 lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Vdw)],
                                 (stepWork.computeVirial || stepWork.computeEnergy),
                                 step,
                                 simulationWork.useGpuPmePpCommunication,
                                 reinitGpuPmePpComms,
                                 pmeSendCoordinatesFromGpu,
                                 stepWork.useGpuPmeFReduction,
                                 nullptr,
                                 simulationWork.useMdGpuGraph,
                                 wcycle);
    }

    if (stepWork.doNeighborSearch)
    {
        doPairSearch(cr, inputrec, mdModulesNotifiers, step, nrnb, wcycle, *top, box, x, v, *mdatoms, fr, runScheduleWork);

        /* At a search step we need to start the first balancing region
         * somewhere early inside the step after communication during domain
         * decomposition (and not during the previous step as usual).
         */
        ddBalanceRegionHandler.openBeforeForceComputationCpu(DdAllowBalanceRegionReopen::yes);
    }

    auto* localXReadyOnDevice = (stepWork.haveGpuPmeOnThisRank || stepWork.useGpuXBufferOps
                                 || simulationWork.useGpuUpdate || pmeSendCoordinatesFromGpu)
                                        ? stateGpu->getCoordinatesReadyOnDeviceEvent(
                                                  AtomLocality::Local, simulationWork, stepWork)
                                        : nullptr;

    if (stepWork.clearGpuFBufferEarly)
    {
        // GPU Force halo exchange will set a subset of local atoms with remote non-local data.
        // First clear local portion of force array, so that untouched atoms are zero.
        // The dependency for this is that forces from previous timestep have been consumed,
        // which is satisfied when localXReadyOnDevice has been marked for GPU update case.
        // For CPU update, the forces are consumed by the beginning of the step, so no extra sync needed.
        GpuEventSynchronizer* dependency = simulationWork.useGpuUpdate ? localXReadyOnDevice : nullptr;
        stateGpu->clearForcesOnGpu(AtomLocality::Local, dependency);
    }

    clear_mat(vir_force);

    if (fr->pbcType != PbcType::No)
    {
        /* Compute shift vectors every step,
         * because of pressure coupling or box deformation!
         */
        if (simulationWork.haveDynamicBox && stepWork.stateChanged)
        {
            calc_shifts(box, fr->shift_vec);
        }
    }
    nbnxn_atomdata_copy_shiftvec(simulationWork.haveDynamicBox, fr->shift_vec, &nbv->nbat());


    GMX_ASSERT(simulationWork.useGpuHaloExchange
                       == ((cr->dd != nullptr)
                           && (!cr->dd->gpuHaloExchange[0].empty()
                               || cr->dd->gpuHaloExchangeNvshmemHelper != nullptr)),
               "The GPU halo exchange is active, but it has not been constructed.");

    bool gmx_used_in_debug haveCopiedXFromGpu = false;
    // Copy coordinate from the GPU if update is on the GPU and there
    // are forces to be computed on the CPU, or for the computation of
    // virial, or if host-side data will be transferred from this task
    // to a remote task for halo exchange or PME-PP communication. At
    // search steps the current coordinates are already on the host,
    // hence copy is not needed.
    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch
        && (runScheduleWork.domainWork.haveCpuLocalForceWork || stepWork.computeVirial
            || simulationWork.useCpuPmePpCommunication || simulationWork.useCpuHaloExchange
            || simulationWork.computeMuTot))
    {
        stateGpu->copyCoordinatesFromGpu(x.unpaddedArrayRef(), AtomLocality::Local);
        haveCopiedXFromGpu = true;
    }

    // Coordinates on the device are needed if PME or BufferOps are offloaded.
    // The local coordinates can be copied right away.
    // NOTE: Consider moving this copy to right after they are updated and constrained,
    //       if the later is not offloaded.
    if (stepWork.haveGpuPmeOnThisRank || stepWork.useGpuXBufferOps || pmeSendCoordinatesFromGpu)
    {
        GMX_ASSERT(stateGpu != nullptr, "stateGpu should not be null");
        const int expectedLocalXReadyOnDeviceConsumptionCount =
                getExpectedLocalXReadyOnDeviceConsumptionCount(
                        simulationWork, stepWork, domainWork, pmeSendCoordinatesFromGpu);

        // We need to copy coordinates when:
        // 1. Update is not offloaded
        // 2. The buffers were reinitialized on search step
        if (!simulationWork.useGpuUpdate || stepWork.doNeighborSearch)
        {
            stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(),
                                           AtomLocality::Local,
                                           expectedLocalXReadyOnDeviceConsumptionCount);
        }
        else if (simulationWork.useGpuUpdate)
        {
            stateGpu->setXUpdatedOnDeviceEventExpectedConsumptionCount(
                    expectedLocalXReadyOnDeviceConsumptionCount);
        }
    }

    if (stepWork.computePmeOnSeparateRank && !stepWork.doNeighborSearch)
    {
        /* Send particle coordinates to the pme nodes */
        if (!pmeSendCoordinatesFromGpu && simulationWork.useGpuUpdate)
        {
            GMX_ASSERT(haveCopiedXFromGpu,
                       "a wait should only be triggered if copy has been scheduled");
            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
        }

        gmx_pme_send_coordinates(fr,
                                 cr->dd,
                                 box,
                                 x.unpaddedArrayRef(),
                                 lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)],
                                 lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Vdw)],
                                 (stepWork.computeVirial || stepWork.computeEnergy),
                                 step,
                                 simulationWork.useGpuPmePpCommunication,
                                 reinitGpuPmePpComms,
                                 pmeSendCoordinatesFromGpu,
                                 stepWork.useGpuPmeFReduction,
                                 pmeSendCoordinatesFromGpu ? localXReadyOnDevice : nullptr,
                                 simulationWork.useMdGpuGraph,
                                 wcycle);
    }

    if (simulationWork.useGpuFBufferOpsWhenAllowed && stepWork.doNeighborSearch)
    {
        // with MPI, direct GPU communication, and separate PME ranks we need
        // gmx_pme_send_coordinates() to be called before we can set up force reduction
        bool doSetupLocalGpuForceReduction = GMX_MPI && simulationWork.useGpuPmePpCommunication;
        if (doSetupLocalGpuForceReduction)
        {
            setupLocalGpuForceReduction(runScheduleWork,
                                        fr->nbv.get(),
                                        stateGpu,
                                        fr->gpuForceReduction[AtomLocality::Local].get(),
                                        fr->pmePpCommGpu.get(),
                                        fr->pmedata,
                                        cr->dd);
        }
    }

    if (stepWork.haveGpuPmeOnThisRank)
    {
        launchPmeGpuSpread(fr->pmedata,
                           box,
                           simulationWork,
                           stepWork,
                           localXReadyOnDevice,
                           lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)],
                           simulationWork.useMdGpuGraph,
                           wcycle);
    }


    if (!stepWork.doNeighborSearch && !EI_TPI(inputrec.eI) && stepWork.computeNonbondedForces)
    {
        if (stepWork.useGpuXBufferOps)
        {
            GMX_ASSERT(stateGpu, "stateGpu should be valid when buffer ops are offloaded");
            nbv->convertCoordinatesGpu(AtomLocality::Local, stateGpu->getCoordinates(), localXReadyOnDevice);
        }
        else
        {
            if (simulationWork.useGpuUpdate)
            {
                GMX_ASSERT(stateGpu, "need a valid stateGpu object");
                GMX_ASSERT(haveCopiedXFromGpu,
                           "a wait should only be triggered if copy has been scheduled");
                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
            }
            nbv->convertCoordinates(AtomLocality::Local, x.unpaddedArrayRef());
        }
    }

    if (simulationWork.useGpuNonbonded && (stepWork.computeNonbondedForces || domainWork.haveGpuBondedWork))
    {
        ddBalanceRegionHandler.openBeforeForceComputationGpu();

        wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
        wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        gpu_upload_shiftvec(nbv->gpuNbv(), &nbv->nbat());
        if (!stepWork.useGpuXBufferOps)
        {
            gpu_copy_xq_to_gpu(nbv->gpuNbv(), &nbv->nbat(), AtomLocality::Local);
        }
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
        // with X buffer ops offloaded to the GPU on all but the search steps

        // bonded work not split into separate local and non-local, so with DD
        // we can only launch the kernel after non-local coordinates have been received.
        if (domainWork.haveGpuBondedWork && !simulationWork.havePpDomainDecomposition)
        {
            fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
        }

        /* launch local nonbonded work on GPU */
        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::Local, enbvClearFNo, step, nrnb, wcycle);
        /* launch local nonbonded free energy work on GPU */
        if (domainWork.haveGpuNonbondedFreeEnergyWork && stepWork.computeNonbondedForces)
        {
            nbv->dispatchFreeEnergyGpuKernels(InteractionLocality::Local, simulationWork, stepWork);
        }
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
    }

    if (stepWork.haveGpuPmeOnThisRank)
    {
        // In PME GPU and mixed mode we launch FFT / gather after the
        // X copy/transform to allow overlap as well as after the GPU NB
        // launch to avoid FFT launch overhead hijacking the CPU and delaying
        // the nonbonded kernel.
        launchPmeGpuFftAndGather(fr->pmedata,
                                 lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)],
                                 wcycle,
                                 stepWork);
    }

    /* Communicate coordinates and sum dipole if necessary */
    if (simulationWork.havePpDomainDecomposition)
    {
        GpuEventSynchronizer* gpuCoordinateHaloLaunched = nullptr;
        if (!stepWork.doNeighborSearch)
        {
            if (stepWork.useGpuXHalo)
            {
                // The following must be called after local setCoordinates (which records an event
                // when the coordinate data has been copied to the device).
                gpuCoordinateHaloLaunched = communicateGpuHaloCoordinates(*cr, box, localXReadyOnDevice);

                if (domainWork.haveCpuNonLocalForceWork)
                {
                    // non-local part of coordinate buffer must be copied back to host for CPU work
                    stateGpu->copyCoordinatesFromGpu(
                            x.unpaddedArrayRef(), AtomLocality::NonLocal, gpuCoordinateHaloLaunched);
                }
            }
            else
            {
                if (simulationWork.useGpuUpdate)
                {
                    GMX_ASSERT(haveCopiedXFromGpu,
                               "a wait should only be triggered if copy has been scheduled");
                    const bool haveAlreadyWaited =
                            (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
                    if (!haveAlreadyWaited)
                    {
                        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
                    }
                }

                if (cr->dd->haloExchange)
                {
                    wallcycle_start(wcycle, WallCycleCounter::MoveX);
                    cr->dd->haloExchange->moveX(box, x.unpaddedArrayRef());
                    wallcycle_stop(wcycle, WallCycleCounter::MoveX);
                }
                else
                {
                    dd_move_x(cr->dd, box, x.unpaddedArrayRef(), wcycle);
                }
            }
        }

        if (stepWork.useGpuXBufferOps)
        {
            if (!stepWork.useGpuXHalo)
            {
                stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(), AtomLocality::NonLocal);
            }
            GpuEventSynchronizer* xReadyOnDeviceEvent = stateGpu->getCoordinatesReadyOnDeviceEvent(
                    AtomLocality::NonLocal, simulationWork, stepWork, gpuCoordinateHaloLaunched);
            if (stepWork.useGpuXHalo && domainWork.haveCpuNonLocalForceWork)
            {
                /* We already enqueued an event for Gpu Halo exchange completion into the
                 * NonLocal stream when D2H copying the coordinates. */
                xReadyOnDeviceEvent = nullptr;
            }
            nbv->convertCoordinatesGpu(
                    AtomLocality::NonLocal, stateGpu->getCoordinates(), xReadyOnDeviceEvent);
        }
        else if (!stepWork.doNeighborSearch)
        {
            nbv->convertCoordinates(AtomLocality::NonLocal, x.unpaddedArrayRef());
        }

        if (simulationWork.useGpuNonbonded)
        {

            if (!stepWork.useGpuXBufferOps)
            {
                wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
                wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
                gpu_copy_xq_to_gpu(nbv->gpuNbv(), &nbv->nbat(), AtomLocality::NonLocal);
                wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
                wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
            }

            if (domainWork.haveGpuBondedWork)
            {
                fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
            }

            /* launch non-local nonbonded tasks on GPU */
            wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
            wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
            do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::NonLocal, enbvClearFNo, step, nrnb, wcycle);
            /* launch non-local nonbonded free energy tsaks on GPU */
            if (domainWork.haveGpuNonbondedFreeEnergyWork && stepWork.computeNonbondedForces)
            {
                nbv->dispatchFreeEnergyGpuKernels(InteractionLocality::NonLocal, simulationWork, stepWork);
            }
            wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
            wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
        }
    }

    if (simulationWork.useGpuNonbonded && stepWork.computeNonbondedForces)
    {
        /* launch D2H copy-back F */
        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);

        if (simulationWork.havePpDomainDecomposition)
        {
            gpu_launch_cpyback(nbv->gpuNbv(), &nbv->nbat(), stepWork, AtomLocality::NonLocal);
        }
        gpu_launch_cpyback(nbv->gpuNbv(), &nbv->nbat(), stepWork, AtomLocality::Local);
        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);

        if (domainWork.haveGpuBondedWork && stepWork.computeEnergy)
        {
            fr->listedForcesGpu->launchEnergyTransfer();
        }
        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
    }

    ArrayRef<const RVec> xWholeMolecules;
    if (fr->wholeMoleculeTransform)
    {
        xWholeMolecules = fr->wholeMoleculeTransform->wholeMoleculeCoordinates(x.unpaddedArrayRef(), box);
    }

    // The CPU force buffer force clearing needs to happen after the previous step
    // force reduction has already completed. To minimize the cross-step dependencies
    // we wait on the coordinates to be updated on the device which is sufficient but
    // a later event than what we strictly need to synchronize with.
    if (simulationWork.useGpuUpdate && domainWork.haveCpuLocalForceWork)
    {
        const bool coordinatesAlreadyUpdatedOnDevice =
                stepWork.doNeighborSearch || simulationWork.useCpuHaloExchange
                || (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
        if (!coordinatesAlreadyUpdatedOnDevice)
        {
            stateGpu->waitCoordinatesUpdatedOnDevice();
        }
    }

    /* Start the force cycle counter.
     * Note that a different counter is used for dynamic load balancing.
     */
    wallcycle_start(wcycle, WallCycleCounter::Force);

    /* Set up and clear force outputs:
     * forceOutMtsLevel0:  everything except what is in the other two outputs
     * forceOutMtsLevel1:  PME-mesh and listed-forces group 1
     * forceOutNonbonded: non-bonded forces
     * Without multiple time stepping all point to the same object.
     * With multiple time-stepping the use is different for MTS fast (level0 only) and slow steps.
     */
    ForceOutputs forceOutMtsLevel0 = setupForceOutputs(
            &fr->forceHelperBuffers[0], force, domainWork, stepWork, simulationWork.havePpDomainDecomposition, wcycle);
    const bool useMultiLevelMts = (simulationWork.useMts && inputrec.mtsLevels.size() > 2);
    const int  longrangeMtsLevel =
            simulationWork.useMts ? forceGroupMtsLevel(inputrec.mtsLevels, MtsForceGroups::LongrangeNonbonded) : 0;

    std::optional<ForceOutputs>              forceOutSingleSlowLevel;
    std::vector<std::optional<ForceOutputs>> forceOutMultiLevel;
    std::vector<ForceOutputs*>               forceOutByMtsLevel(
            simulationWork.useMts ? inputrec.mtsLevels.size() : 1, nullptr);
    forceOutByMtsLevel[0] = &forceOutMtsLevel0;

    if (simulationWork.useMts && stepWork.computeSlowForces)
    {
        if (!useMultiLevelMts)
        {
            forceOutSingleSlowLevel.emplace(setupForceOutputs(&fr->forceHelperBuffers[1],
                                                              forceView->forceMtsCombinedWithPadding(),
                                                              domainWork,
                                                              stepWork,
                                                              simulationWork.havePpDomainDecomposition,
                                                              wcycle));
            forceOutByMtsLevel[1] = &forceOutSingleSlowLevel.value();
        }
        else
        {
            forceOutMultiLevel.resize(inputrec.mtsLevels.size());
            for (int mtsLevel = 1; mtsLevel <= stepWork.highestActiveMtsLevel; mtsLevel++)
            {
                forceOutMultiLevel[mtsLevel].emplace(setupForceOutputs(&fr->forceHelperBuffers[mtsLevel],
                                                                       forceView->forceForMtsLevelWithPadding(mtsLevel),
                                                                       domainWork,
                                                                       stepWork,
                                                                       simulationWork.havePpDomainDecomposition,
                                                                       wcycle));
                forceOutByMtsLevel[mtsLevel] = &forceOutMultiLevel[mtsLevel].value();
            }
        }
    }

    ForceOutputs* forceOutMtsLevel1 = simulationWork.useMts
                                              ? (stepWork.computeLongRangeNonbondedForces
                                                         ? forceOutByMtsLevel[longrangeMtsLevel]
                                                         : nullptr)
                                              : &forceOutMtsLevel0;
    GMX_ASSERT(!stepWork.computeLongRangeNonbondedForces
                       || forceOutMtsLevel1 == nullptr
                       || forceOutMtsLevel1->haveForceWithVirial(),
               "Active long-range nonbonded work requires a force-with-virial output buffer");

    ForceOutputs* forceOutNonbonded = &forceOutMtsLevel0;
    if (simulationWork.useMts && simulationWork.nonbondedMtsLevel > 0 && stepWork.computeNonbondedForces)
    {
        forceOutNonbonded = forceOutByMtsLevel[simulationWork.nonbondedMtsLevel];
    }
    std::vector<ForceWithVirial*> forceWithVirialByMtsLevel(forceOutByMtsLevel.size(), nullptr);
    forceWithVirialByMtsLevel[0] = &forceOutMtsLevel0.forceWithVirial();
    for (int mtsLevel = 1; mtsLevel <= stepWork.highestActiveMtsLevel && mtsLevel < static_cast<int>(forceOutByMtsLevel.size());
         mtsLevel++)
    {
        if (forceOutByMtsLevel[mtsLevel] != nullptr)
        {
            forceWithVirialByMtsLevel[mtsLevel] = &forceOutByMtsLevel[mtsLevel]->forceWithVirial();
        }
    }

    if (inputrec.bPull && pull_have_constraint(*pull_work))
    {
        clear_pull_forces(pull_work);
    }
    wallcycle_stop(wcycle, WallCycleCounter::Force);

    // For the rest of the CPU tasks that depend on GPU-update produced coordinates,
    // this wait ensures that the D2H transfer is complete.
    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch)
    {
        const bool needCoordsOnHost = (runScheduleWork.domainWork.haveCpuLocalForceWork
                                       || stepWork.computeVirial || simulationWork.computeMuTot);
        const bool haveAlreadyWaited =
                simulationWork.useCpuHaloExchange
                || (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
        if (needCoordsOnHost && !haveAlreadyWaited)
        {
            GMX_ASSERT(haveCopiedXFromGpu,
                       "a wait should only be triggered if copy has been scheduled");
            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
        }
    }

    DipoleData dipoleData;

    if (simulationWork.computeMuTot)
    {
        const int start = 0;

        /* Calculate total (local) dipole moment in a temporary common array.
         * This makes it possible to sum them over nodes faster.
         */
        ArrayRef<const RVec> xRef = (xWholeMolecules.empty() ? x.unpaddedArrayRef() : xWholeMolecules);
        calc_mu(start,
                mdatoms->homenr,
                xRef,
                mdatoms->chargeA,
                mdatoms->chargeB,
                mdatoms->nChargePerturbed != 0,
                dipoleData.muStaging[0],
                dipoleData.muStaging[1]);

        reduceAndUpdateMuTot(&dipoleData,
                             cr->commMyGroup,
                             (fr->efep != FreeEnergyPerturbationType::No),
                             lambda,
                             muTotal,
                             ddBalanceRegionHandler);
    }

    /* Reset energies */
    reset_enerdata(enerd);

    if (haveDDAtomOrdering(*cr) && simulationWork.haveSeparatePmeRank)
    {
        wallcycle_start(wcycle, WallCycleCounter::PpDuringPme);
        dd_force_flop_start(cr->dd, nrnb);
    }

    if (inputrec.bRot)
    {
        wallcycle_start(wcycle, WallCycleCounter::Rot);
        do_rotation(cr->commMyGroup, cr->dd, enforcedRotation, box, x.unpaddedConstArrayRef(), t, step, stepWork.doNeighborSearch);
        wallcycle_stop(wcycle, WallCycleCounter::Rot);
    }

    /* We calculate the non-bonded forces, when done on the CPU, here.
     * We do this before calling do_force_lowlevel, because in that
     * function, the listed forces are calculated before PME, which
     * does communication.  With this order, non-bonded and listed
     * force calculation imbalance can be balanced out by the domain
     * decomposition load balancing.
     */

    const bool useOrEmulateGpuNb = simulationWork.useGpuNonbonded || fr->nbv->emulateGpu();
    const bool useExactLammpsRespaNonbonded =
            stepWork.computeNonbondedForces && inputrec.useMts && inputrec.mtsMode == MtsMode::LammpsRespa
            && inputrec.lammpsRespa.hasPairSplitting();

    GMX_RELEASE_ASSERT(!useExactLammpsRespaNonbonded || !useOrEmulateGpuNb,
                       "Exact LAMMPS-style r-RESPA is CPU-only");
    GMX_RELEASE_ASSERT(!useExactLammpsRespaNonbonded || !domainWork.haveCpuNonbondedFreeEnergyWork,
                       "Exact LAMMPS-style r-RESPA does not support nonbonded free-energy work yet");

    if (useExactLammpsRespaNonbonded)
    {
        wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
        computeLammpsRespaNonbondedCpu(
                inputrec, top->idef, fr, *mdatoms, x.unpaddedArrayRef(), forceOutByMtsLevel, enerd, stepWork);
        wallcycle_stop(wcycle, WallCycleCounter::Force);
    }
    else if (!useOrEmulateGpuNb)
    {
        wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
        do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::Local, enbvClearFYes, step, nrnb, wcycle);
        wallcycle_stop(wcycle, WallCycleCounter::Force);
    }

    if (stepWork.useGpuXHalo && domainWork.haveCpuNonLocalForceWork)
    {
        /* Wait for non-local coordinate data to be copied from device */
        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::NonLocal);
    }

    wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
    if (domainWork.haveCpuNonbondedFreeEnergyWork && stepWork.computeNonbondedForces)
    {
        /* Calculate the local and non-local free energy interactions on CPU here. */
        nbv->dispatchFreeEnergyCpuKernels(x,
                                          &forceOutNonbonded->forceWithShiftForces(),
                                          fr->use_simd_kernels,
                                          fr->ntype,
                                          *fr->ic,
                                          fr->shift_vec,
                                          fr->nbfp,
                                          fr->ljpme_c6grid,
                                          mdatoms->chargeA,
                                          mdatoms->chargeB,
                                          mdatoms->typeA,
                                          mdatoms->typeB,
                                          lambda,
                                          enerd,
                                          stepWork,
                                          nrnb);
    }

    if (stepWork.computeNonbondedForces && !useOrEmulateGpuNb && !useExactLammpsRespaNonbonded)
    {
        if (simulationWork.havePpDomainDecomposition)
        {
            do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::NonLocal, enbvClearFNo, step, nrnb, wcycle);
        }

        if (stepWork.computeForces)
        {
            /* Add all the non-bonded force to the normal force array.
             * This can be split into a local and a non-local part when overlapping
             * communication with calculation with domain decomposition.
             */
            wallcycle_stop(wcycle, WallCycleCounter::Force);
            nbv->atomdata_add_nbat_f_to_f(AtomLocality::All,
                                          forceOutNonbonded->forceWithShiftForces().force());
            wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
        }

        /* If there are multiple fshift output buffers we need to reduce them */
        if (stepWork.computeVirial)
        {
            /* This is not in a subcounter because it takes a
               negligible and constant-sized amount of time */
            nbnxn_atomdata_add_nbat_fshift_to_fshift(
                    nbv->nbat(), forceOutNonbonded->forceWithShiftForces().shiftForces());
        }
    }

    // Compute wall interactions, when present.
    // Note: should be moved to special forces.
    if (inputrec.nwall && stepWork.computeNonbondedForces)
    {
        /* foreign lambda component for walls */
        real dvdl_walls = do_walls(inputrec,
                                   *fr,
                                   box,
                                   mdatoms->typeA,
                                   mdatoms->typeB,
                                   mdatoms->cENER,
                                   mdatoms->homenr,
                                   mdatoms->nPerturbed,
                                   x.unpaddedConstArrayRef(),
                                   &forceOutMtsLevel0.forceWithVirial(),
                                   lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Vdw)],
                                   enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJSR],
                                   nrnb);
        enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Vdw] += dvdl_walls;
    }

    if (stepWork.computeListedForces)
    {
        /* Check whether we need to take into account PBC in listed interactions */
        bool needMolPbc = false;
        for (const auto& listedForces : fr->listedForces)
        {
            if (listedForces.haveCpuListedForces(*fr->fcdata))
            {
                needMolPbc = fr->bMolPBC;
            }
        }

        t_pbc pbc;

        if (needMolPbc)
        {
            /* Since all atoms are in the rectangular or triclinic unit-cell,
             * only single box vector shifts (2 in x) are required.
             */
            set_pbc_dd(&pbc, fr->pbcType, haveDDAtomOrdering(*cr) ? &cr->dd->numCells : nullptr, TRUE, box);
        }

        const int numActiveMtsLevels = simulationWork.useMts ? (stepWork.highestActiveMtsLevel + 1) : 1;
        for (int mtsIndex = 0; mtsIndex < numActiveMtsLevels; mtsIndex++)
        {
            ListedForces& listedForces = fr->listedForces[mtsIndex];
            ForceOutputs& forceOut     = *forceOutByMtsLevel[mtsIndex];
            listedForces.calculate(wcycle,
                                   box,
                                   x,
                                   xWholeMolecules,
                                   fr->fcdata.get(),
                                   hist,
                                   &forceOut,
                                   fr,
                                   &pbc,
                                   enerd,
                                   nrnb,
                                   lambda,
                                   mdatoms->chargeA,
                                   mdatoms->chargeB,
                                   makeConstArrayRef(mdatoms->bPerturbed),
                                   mdatoms->cENER,
                                   mdatoms->nPerturbed,
                                   haveDDAtomOrdering(*cr) ? cr->dd->globalAtomIndices.data() : nullptr,
                                   stepWork);
        }
    }

    if (stepWork.computeLongRangeNonbondedForces)
    {
        const char* earlyAccumTraceDirPath = std::getenv("GMX_PCFF_RESPA_EARLY_TRACE_DIR");
        const bool dumpEarlyAccumTrace =
                (earlyAccumTraceDirPath != nullptr && *earlyAccumTraceDirPath != '\0' && step == 0);
        const bool outerAliasesShift =
                (forceOutMtsLevel1 != nullptr && forceOutMtsLevel1->haveForceWithVirial()
                 && forceOutMtsLevel1->forceWithVirial().force_.data()
                            == forceOutMtsLevel1->forceWithShiftForces().force().data());
        if (dumpEarlyAccumTrace && forceOutMtsLevel1 != nullptr)
        {
            dumpRespaMergeTraceVector(
                    earlyAccumTraceDirPath,
                    "step0_level2_before_longrange_virial.tsv",
                    "stage=before_longrange_nonbonded mts_index=2 mts_user=3 buffer=forceWithVirial alias_with_shift="
                            + std::string(outerAliasesShift ? "true" : "false"),
                    forceOutMtsLevel1->forceWithVirial().force_);
        }
        longRangeNonbondeds->calculate(fr->pmedata,
                                       cr,
                                       x.unpaddedConstArrayRef(),
                                       &forceOutMtsLevel1->forceWithVirial(),
                                       enerd,
                                       box,
                                       lambda,
                                       dipoleData.muStateAB,
                                       stepWork,
                                       ddBalanceRegionHandler);
        if (dumpEarlyAccumTrace && forceOutMtsLevel1 != nullptr)
        {
            dumpRespaMergeTraceVector(
                    earlyAccumTraceDirPath,
                    "step0_level2_after_longrange_virial.tsv",
                    "stage=after_longrange_nonbonded mts_index=2 mts_user=3 buffer=forceWithVirial alias_with_shift="
                            + std::string(outerAliasesShift ? "true" : "false"),
                    forceOutMtsLevel1->forceWithVirial().force_);
        }
    }

    wallcycle_stop(wcycle, WallCycleCounter::Force);

    // VdW dispersion correction, only computed on main rank to avoid double counting
    if ((stepWork.computeEnergy || stepWork.computeVirial) && fr->dispersionCorrection
        && cr->commMySim.isMainRank())
    {
        // Calculate long range corrections to pressure and energy
        const DispersionCorrection::Correction correction = fr->dispersionCorrection->calculate(
                box, lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Vdw)]);

        if (stepWork.computeEnergy)
        {
            enerd->term[InteractionFunction::DispersionCorrection] = correction.energy;
            enerd->term[InteractionFunction::dVvanderWaalsdLambda] += correction.dvdl;
            enerd->dvdl_lin[FreeEnergyPerturbationCouplingType::Vdw] += correction.dvdl;
        }
        if (stepWork.computeVirial)
        {
            correction.correctVirial(vir_force);
            enerd->term[InteractionFunction::PressureDispersionCorrection] = correction.pressure;
        }
    }

    const bool needToReceivePmeResultsFromSeparateRank = stepWork.computePmeOnSeparateRank;
    const bool needToReceivePmeResults =
            (stepWork.haveGpuPmeOnThisRank || needToReceivePmeResultsFromSeparateRank);

    /* When running free energy perturbations steered by AWH and doing PME calculations on the
     * GPU we must wait for the PME calculation (dhdl) results to finish before sampling the
     * FEP dimension with AWH. */
    const bool needEarlyPmeResults = (awh != nullptr && awh->hasFepLambdaDimension() && needToReceivePmeResults
                                      && stepWork.computeEnergy && stepWork.computeLongRangeNonbondedForces);
    if (needEarlyPmeResults)
    {
        if (stepWork.haveGpuPmeOnThisRank)
        {
            pmeGpuWaitAndReduce(fr->pmedata,
                                stepWork,
                                wcycle,
                                &forceOutMtsLevel1->forceWithVirial(),
                                enerd,
                                lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)]);
        }
        else if (needToReceivePmeResultsFromSeparateRank)
        {
            /* In case of node-splitting, the PP nodes receive the long-range
             * forces, virial and energy from the PME nodes here.
             */
            pme_receive_force_ener(fr,
                                   cr->dd,
                                   &forceOutMtsLevel1->forceWithVirial(),
                                   enerd,
                                   simulationWork.useGpuPmePpCommunication,
                                   stepWork.useGpuPmeFReduction,
                                   wcycle);
        }
    }

    if (domainWork.haveSpecialForces)
    {
        // Communication often happens for special forces, so we should close the balancing region here
        ddBalanceRegionHandler.closeAfterForceComputationCpu();

        computeSpecialForces(fplog,
                             cr->commMyGroup,
                             cr->dd,
                             inputrec,
                             awh,
                             enforcedRotation,
                             imdSession,
                             pull_work,
                             step,
                             t,
                             wcycle,
                             fr->forceProviders,
                             box,
                             x.unpaddedArrayRef(),
                             mdatoms,
                             lambda,
                             stepWork,
                             forceWithVirialByMtsLevel,
                             enerd,
                             ed,
                             stepWork.doNeighborSearch);
    }

    if (simulationWork.havePpDomainDecomposition && stepWork.computeForces && stepWork.useGpuFHalo
        && domainWork.haveCpuLocalForceWork)
    {
        stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(), AtomLocality::Local);
    }

    GMX_ASSERT(!(simulationWork.nonbondedMtsLevel > 0 && stepWork.useGpuFBufferOps),
               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
    GMX_ASSERT(!(nonbondedAtMtsNonzeroLevel && stepWork.useGpuFHalo),
               "The schedule below does not allow for nonbonded MTS with GPU halo exchange");
    // Will store the amount of cycles spent waiting for the GPU that
    // will be later used in the DLB accounting.
    float cycles_wait_gpu = 0;
    if (useOrEmulateGpuNb && stepWork.computeNonbondedForces && !useExactLammpsRespaNonbonded)
    {
        auto& forceWithShiftForces = forceOutNonbonded->forceWithShiftForces();

        /* wait for non-local forces (or calculate in emulation mode) */
        if (simulationWork.havePpDomainDecomposition)
        {
            if (simulationWork.useGpuNonbonded)
            {
                cycles_wait_gpu +=
                        gpu_wait_finish_task(nbv->gpuNbv(),
                                             stepWork,
                                             AtomLocality::NonLocal,
                                             (simulationWork.useGpuForeignNonbondedFE) ? true : false,
                                             enerd,
                                             forceWithShiftForces.shiftForces(),
                                             wcycle);
            }
            else
            {
                wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
                do_nb_verlet(
                        fr, ic, enerd, stepWork, InteractionLocality::NonLocal, enbvClearFYes, step, nrnb, wcycle);
                wallcycle_stop(wcycle, WallCycleCounter::Force);
            }

            if (stepWork.useGpuFBufferOps)
            {
                if (domainWork.haveCpuNonLocalForceWork)
                {
                    stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
                                              AtomLocality::NonLocal);
                }


                fr->gpuForceReduction[AtomLocality::NonLocal]->execute();

                if (!stepWork.useGpuFHalo)
                {
                    /* We don't explicitly wait for the forces to be reduced on device,
                     * but wait for them to finish copying to CPU instead.
                     * So, we manually consume the event, see Issue #3988. */
                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::NonLocal);
                    // copy from GPU input for dd_move_f()
                    stateGpu->copyForcesFromGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
                                                AtomLocality::NonLocal);
                }
            }
            else
            {
                nbv->atomdata_add_nbat_f_to_f(AtomLocality::NonLocal, forceWithShiftForces.force());
            }

            if (fr->nbv->emulateGpu() && stepWork.computeVirial)
            {
                nbnxn_atomdata_add_nbat_fshift_to_fshift(nbv->nbat(), forceWithShiftForces.shiftForces());
            }
        }
    }

    /* Combining the forces for multiple time stepping before the halo exchange, when possible,
     * avoids an extra halo exchange (when DD is used) and post-processing step.
     */
    if (stepWork.combineMtsForcesBeforeHaloExchange)
    {
        const std::vector<ArrayRef<const RVec>> slowLevelForces = {
            forceOutByMtsLevel[1]->forceWithShiftForces().force()
        };
        const std::array<int, 1> slowLevelFactors = { inputrec.mtsLevels[1].stepFactor };
        wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
        combineMtsForces(getLocalAtomCount(cr->dd, *mdatoms, simulationWork.havePpDomainDecomposition),
                         force.unpaddedArrayRef(),
                         forceView->forceMtsCombined(),
                         slowLevelForces,
                         slowLevelFactors);
        wallcycle_stop(wcycle, WallCycleCounter::Force);
    }

    // With both nonbonded and PME offloaded a GPU on the same rank, we use
    // an alternating wait/reduction scheme.
    // When running free energy perturbations steered by AWH and calculating PME on GPU,
    // i.e. if needEarlyPmeResults == true, the PME results have already been reduced above.
    const bool alternateGpuWait = (!c_disableAlternatingWait && stepWork.haveGpuPmeOnThisRank
                                   && simulationWork.useGpuNonbonded && !simulationWork.havePpDomainDecomposition
                                   && !stepWork.useGpuFBufferOps && !needEarlyPmeResults);


    const int expectedLocalFReadyOnDeviceConsumptionCount = getExpectedLocalFReadyOnDeviceConsumptionCount(
            simulationWork, domainWork, stepWork, useOrEmulateGpuNb, alternateGpuWait);
    // If expectedLocalFReadyOnDeviceConsumptionCount == 0, stateGpu can be uninitialized
    if (expectedLocalFReadyOnDeviceConsumptionCount > 0)
    {
        stateGpu->setFReadyOnDeviceEventExpectedConsumptionCount(
                AtomLocality::Local, expectedLocalFReadyOnDeviceConsumptionCount);
    }

    if (simulationWork.havePpDomainDecomposition)
    {
        /* We are done with the CPU compute.
         * We will now communicate the non-local forces.
         * If we use a GPU this will overlap with GPU work, so in that case
         * we do not close the DD force balancing region here.
         * With special forces we closed this region already before computing the special forces.
         */
        ddBalanceRegionHandler.closeAfterForceComputationCpu();

        if (stepWork.computeForces)
        {

            if (stepWork.useGpuFHalo)
            {
                // If there exist CPU forces, data from halo exchange should accumulate into these
                bool accumulateForces = domainWork.haveCpuLocalForceWork;
                FixedCapacityVector<GpuEventSynchronizer*, 2> gpuForceHaloDependencies;
                // completion of both H2D copy and clearing is signaled by fReadyOnDevice
                if (domainWork.haveCpuLocalForceWork || stepWork.clearGpuFBufferEarly)
                {
                    gpuForceHaloDependencies.push_back(stateGpu->fReadyOnDevice(AtomLocality::Local));
                }
                gpuForceHaloDependencies.push_back(stateGpu->fReducedOnDevice(AtomLocality::NonLocal));

                communicateGpuHaloForces(*cr, accumulateForces, &gpuForceHaloDependencies);
            }
            else
            {
                if (stepWork.useGpuFBufferOps)
                {
                    stateGpu->waitForcesReadyOnHost(AtomLocality::NonLocal);
                }

                // Without MTS or with MTS at slow steps with uncombined forces we need to
                // communicate the fast forces
                if (!simulationWork.useMts || !stepWork.combineMtsForcesBeforeHaloExchange)
                {
                    dd_move_f(cr->dd, &forceOutMtsLevel0.forceWithShiftForces(), wcycle);
                }
                // With MTS we need to communicate the slow or combined (in forceOutMtsLevel1) forces
                if (simulationWork.useMts && stepWork.computeSlowForces)
                {
                    for (int mtsLevel = 1; mtsLevel <= stepWork.highestActiveMtsLevel; mtsLevel++)
                    {
                        dd_move_f(cr->dd, &forceOutByMtsLevel[mtsLevel]->forceWithShiftForces(), wcycle);
                    }
                }
            }
        }
    }

    if (alternateGpuWait)
    {
        alternatePmeNbGpuWaitReduce(fr->nbv.get(),
                                    fr->pmedata,
                                    forceOutNonbonded,
                                    forceOutMtsLevel1,
                                    enerd,
                                    lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)],
                                    stepWork,
                                    simulationWork,
                                    wcycle);
    }

    if (!alternateGpuWait && stepWork.haveGpuPmeOnThisRank && !needEarlyPmeResults)
    {
        pmeGpuWaitAndReduce(fr->pmedata,
                            stepWork,
                            wcycle,
                            &forceOutMtsLevel1->forceWithVirial(),
                            enerd,
                            lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)]);
    }

    /* Wait for local GPU NB outputs on the non-alternating wait path */
    if (!alternateGpuWait && stepWork.computeNonbondedForces && simulationWork.useGpuNonbonded)
    {
        /* Measured overhead on CUDA and OpenCL with(out) GPU sharing
         * is between 0.5 and 1.5 Mcycles. So 2 MCycles is an overestimate,
         * but even with a step of 0.1 ms the difference is less than 1%
         * of the step time.
         */
        const float gpuWaitApiOverheadMargin = 2e6F; /* cycles */
        float       waitCycles;

        waitCycles = gpu_wait_finish_task(nbv->gpuNbv(),
                                          stepWork,
                                          AtomLocality::Local,
                                          (simulationWork.useGpuForeignNonbondedFE) ? true : false,
                                          enerd,
                                          forceOutNonbonded->forceWithShiftForces().shiftForces(),
                                          wcycle);

        if (ddBalanceRegionHandler.useBalancingRegion())
        {
            DdBalanceRegionWaitedForGpu waitedForGpu = DdBalanceRegionWaitedForGpu::yes;
            if (stepWork.computeForces && waitCycles <= gpuWaitApiOverheadMargin)
            {
                /* We measured few cycles, it could be that the kernel
                 * and transfer finished earlier and there was no actual
                 * wait time, only API call overhead.
                 * Then the actual time could be anywhere between 0 and
                 * cycles_wait_est. We will use half of cycles_wait_est.
                 */
                waitedForGpu = DdBalanceRegionWaitedForGpu::no;
            }
            ddBalanceRegionHandler.closeAfterForceComputationGpu(cycles_wait_gpu, waitedForGpu);
        }
    }

    if (fr->nbv->emulateGpu() && !useExactLammpsRespaNonbonded)
    {
        // NOTE: emulation kernel is not included in the balancing region,
        // but emulation mode does not target performance anyway
        wallcycle_start_nocount(wcycle, WallCycleCounter::Force);
        do_nb_verlet(fr,
                     ic,
                     enerd,
                     stepWork,
                     InteractionLocality::Local,
                     haveDDAtomOrdering(*cr) ? enbvClearFNo : enbvClearFYes,
                     step,
                     nrnb,
                     wcycle);
        wallcycle_stop(wcycle, WallCycleCounter::Force);
    }

    // If on GPU PME-PP comms path, receive forces from PME before GPU buffer ops
    // TODO refactor this and unify with below default-path call to the same function
    // When running free energy perturbations steered by AWH and calculating PME on GPU,
    // i.e. if needEarlyPmeResults == true, the PME results have already been reduced above.
    if (needToReceivePmeResultsFromSeparateRank && simulationWork.useGpuPmePpCommunication && !needEarlyPmeResults)
    {
        /* In case of node-splitting, the PP nodes receive the long-range
         * forces, virial and energy from the PME nodes here.
         */
        pme_receive_force_ener(fr,
                               cr->dd,
                               &forceOutMtsLevel1->forceWithVirial(),
                               enerd,
                               simulationWork.useGpuPmePpCommunication,
                               stepWork.useGpuPmeFReduction,
                               wcycle);
    }


    /* Do the nonbonded GPU (or emulation) force buffer reduction
     * on the non-alternating path. */
    GMX_ASSERT(!(nonbondedAtMtsNonzeroLevel && stepWork.useGpuFBufferOps),
               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
    if (useOrEmulateGpuNb && !alternateGpuWait)
    {
        if (stepWork.useGpuFBufferOps)
        {
            ArrayRef<RVec> forceWithShift = forceOutNonbonded->forceWithShiftForces().force();

            // TODO: move these steps as early as possible:
            // - CPU f H2D should be as soon as all CPU-side forces are done
            // - wait for force reduction does not need to block host (at least not here, it's sufficient to wait
            //   before the next CPU task that consumes the forces: vsite spread or update)
            // - copy is not performed if GPU force halo exchange is active, because it would overwrite the result
            //   of the halo exchange. In that case the copy is instead performed above, before the exchange.
            //   These should be unified.
            if (domainWork.haveLocalForceContribInCpuBuffer && !stepWork.useGpuFHalo)
            {
                stateGpu->copyForcesToGpu(forceWithShift, AtomLocality::Local);
            }

            if (stepWork.computeNonbondedForces)
            {
                fr->gpuForceReduction[AtomLocality::Local]->execute();
            }

            // Copy forces to host if they are needed for update or if virtual sites are enabled.
            // If there are vsites, we need to copy forces every step to spread vsite forces on host.
            // TODO: When the output flags will be included in step workload, this copy can be combined with the
            //       copy call done in sim_utils(...) for the output.
            // NOTE: If there are virtual sites, the forces are modified on host after this D2H copy. Hence,
            //       they should not be copied in do_md(...) for the output.
            if (!simulationWork.useGpuUpdate
                || (simulationWork.useGpuUpdate && haveDDAtomOrdering(*cr) && simulationWork.useCpuPmePpCommunication)
                || vsite)
            {
                if (stepWork.computeNonbondedForces)
                {
                    /* We have previously issued force reduction on the GPU, but we will
                     * not use this event, instead relying on the stream being in-order.
                     * Issue #3988. */
                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::Local);
                }
                stateGpu->copyForcesFromGpu(forceWithShift, AtomLocality::Local);
                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
            }
        }
        else if (stepWork.computeNonbondedForces)
        {
            ArrayRef<RVec> forceWithShift = forceOutNonbonded->forceWithShiftForces().force();
            nbv->atomdata_add_nbat_f_to_f(AtomLocality::Local, forceWithShift);
        }
    }

    if (expectedLocalFReadyOnDeviceConsumptionCount > 0)
    {
        /* The same fReadyOnDevice device synchronizer is later used to track buffer clearing,
         * so we reset the expected consumption value back to the default (1). */
        stateGpu->setFReadyOnDeviceEventExpectedConsumptionCount(AtomLocality::Local, 1);
    }

    launchGpuEndOfStepTasks(
            nbv, fr->listedForcesGpu.get(), fr->pmedata, enerd, runScheduleWork, step, wcycle);

    if (haveDDAtomOrdering(*cr))
    {
        dd_force_flop_stop(cr->dd, nrnb);
    }

    const bool haveCombinedMtsForces = (stepWork.computeForces && simulationWork.useMts && stepWork.computeSlowForces
                                        && stepWork.combineMtsForcesBeforeHaloExchange);
    const char* mergeTraceDirPath = std::getenv("GMX_PCFF_RESPA_MERGE_TRACE_DIR");
    const bool  dumpMergeTrace    = (mergeTraceDirPath != nullptr && *mergeTraceDirPath != '\0' && step == 0);
    const auto dumpForceOutputsStage = [&](const char* stageLabel, int mtsLevelIndex, ForceOutputs* outputs)
    {
        if (!dumpMergeTrace || outputs == nullptr)
        {
            return;
        }

        const std::string commonHeader = "stage=" + std::string(stageLabel) + " mts_index="
                                         + std::to_string(mtsLevelIndex) + " mts_user="
                                         + std::to_string(mtsLevelIndex + 1);
        const std::string levelLabel   = "step0_level" + std::to_string(mtsLevelIndex) + "_" + stageLabel;

        dumpRespaMergeTraceVector(mergeTraceDirPath,
                                  (levelLabel + "_shift.tsv").c_str(),
                                  commonHeader + " buffer=forceWithShiftForces",
                                  outputs->forceWithShiftForces().force());
        if (outputs->haveForceWithVirial())
        {
            dumpRespaMergeTraceVector(mergeTraceDirPath,
                                      (levelLabel + "_virial.tsv").c_str(),
                                      commonHeader + " buffer=forceWithVirial",
                                      outputs->forceWithVirial().force_);
        }
    };
    if (stepWork.computeForces)
    {
        postProcessForceWithShiftForces(
                nrnb, wcycle, box, x.unpaddedArrayRef(), &forceOutMtsLevel0, vir_force, *mdatoms, *fr, vsite, stepWork);

        if (simulationWork.useMts && stepWork.computeSlowForces && !haveCombinedMtsForces)
        {
            for (int mtsLevel = 1; mtsLevel <= stepWork.highestActiveMtsLevel; mtsLevel++)
            {
                postProcessForceWithShiftForces(nrnb,
                                                wcycle,
                                                box,
                                                x.unpaddedArrayRef(),
                                                forceOutByMtsLevel[mtsLevel],
                                                vir_force,
                                                *mdatoms,
                                                *fr,
                                                vsite,
                                                stepWork);
            }
        }
    }

    // TODO refactor this and unify with above GPU PME-PP / GPU update path call to the same function
    // When running free energy perturbations steered by AWH and calculating PME on GPU,
    // i.e. if needEarlyPmeResults == true, the PME results have already been reduced above.
    if (needToReceivePmeResultsFromSeparateRank && simulationWork.useCpuPmePpCommunication && !needEarlyPmeResults)
    {
        /* In case of node-splitting, the PP nodes receive the long-range
         * forces, virial and energy from the PME nodes here.
         */
        pme_receive_force_ener(fr,
                               cr->dd,
                               &forceOutMtsLevel1->forceWithVirial(),
                               enerd,
                               simulationWork.useGpuPmePpCommunication,
                               false,
                               wcycle);
    }

    if (stepWork.computeForces)
    {
        /* If we don't use MTS or if we already combined the MTS forces before, we only
         * need to post-process one ForceOutputs object here, called forceOutCombined,
         * otherwise we have to post-process two outputs and then combine them.
         */
        ForceOutputs& forceOutCombined = (haveCombinedMtsForces ? forceOutSingleSlowLevel.value() : forceOutMtsLevel0);
        dumpForceOutputsStage("pre_postprocess", 0, &forceOutCombined);
        postProcessForces(
                cr->dd, step, nrnb, wcycle, box, x.unpaddedArrayRef(), &forceOutCombined, vir_force, mdatoms, fr, vsite, stepWork);
        dumpForceOutputsStage("post_postprocess", 0, &forceOutCombined);

        if (simulationWork.useMts && stepWork.computeSlowForces && !haveCombinedMtsForces)
        {
            std::vector<ArrayRef<const RVec>> slowLevelForces;
            std::vector<int>                  slowLevelFactors;
            for (int mtsLevel = 1; mtsLevel <= stepWork.highestActiveMtsLevel; mtsLevel++)
            {
                dumpForceOutputsStage("pre_postprocess", mtsLevel, forceOutByMtsLevel[mtsLevel]);
                postProcessForces(cr->dd,
                                  step,
                                  nrnb,
                                  wcycle,
                                  box,
                                  x.unpaddedArrayRef(),
                                  forceOutByMtsLevel[mtsLevel],
                                  vir_force,
                                  mdatoms,
                                  fr,
                                  vsite,
                                  stepWork);
                dumpForceOutputsStage("post_postprocess", mtsLevel, forceOutByMtsLevel[mtsLevel]);
                slowLevelForces.push_back(forceOutByMtsLevel[mtsLevel]->forceWithShiftForces().force());
                slowLevelFactors.push_back(inputrec.mtsLevels[mtsLevel].stepFactor);
            }

            combineMtsForces(mdatoms->homenr,
                             force.unpaddedArrayRef(),
                             forceView->forceMtsCombined(),
                             slowLevelForces,
                             slowLevelFactors);
            if (dumpMergeTrace)
            {
                dumpRespaMergeTraceVector(mergeTraceDirPath,
                                          "step0_physical_postcombine.tsv",
                                          "stage=post_combine buffer=physical_total",
                                          force.unpaddedArrayRef());
                dumpRespaMergeTraceVector(mergeTraceDirPath,
                                          "step0_impulse_postcombine.tsv",
                                          "stage=post_combine buffer=impulse_total",
                                          forceView->forceMtsCombined());
            }
        }
    }

    if (stepWork.computeEnergy)
    {
        /* Compute the final potential energy terms */
        accumulatePotentialEnergies(enerd, lambda, inputrec.fepvals.get());

        if (!EI_TPI(inputrec.eI))
        {
            checkPotentialEnergyValidity(step, *enerd, inputrec);
        }
    }

    /* In case we don't have constraints and are using GPUs, the next balancing
     * region starts here.
     * Some "special" work at the end of do_force_cuts?, such as vsite spread,
     * virial calculation and COM pulling, is not thus not included in
     * the balance timing, which is ok as most tasks do communication.
     */
    ddBalanceRegionHandler.openBeforeForceComputationCpu(DdAllowBalanceRegionReopen::no);
}

} // namespace gmx
