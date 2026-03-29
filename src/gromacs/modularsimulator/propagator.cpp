/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2019- The GROMACS Authors
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
 * \brief Defines the propagator element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "propagator.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

#include "modularsimulator.h"
#include "simulatoralgorithm.h"
#include "statepropagatordata.h"

namespace gmx
{
namespace
{
const char* activeM2pTraceDirPath()
{
    const char* traceDir = std::getenv("GMX_PCFF_RESPA_M2P_TRACE_DIR");
    return (traceDir != nullptr && *traceDir != '\0') ? traceDir : nullptr;
}

bool shouldTracePositionUpdateStep(const Step step)
{
    const char* traceDir = activeM2pTraceDirPath();
    const char* value    = std::getenv("GMX_PCFF_RESPA_TRACE_POSITION_UPDATE_STEPS");
    if (traceDir == nullptr || value == nullptr || *value == '\0')
    {
        return false;
    }

    std::stringstream ss(value);
    std::string       item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty() && step == std::stoll(item))
        {
            return true;
        }
    }
    return false;
}

bool shouldTraceXvfStageStep(const Step step)
{
    const char* traceDir = activeM2pTraceDirPath();
    const char* value    = std::getenv("GMX_PCFF_RESPA_TRACE_XVF_STEPS");
    if (traceDir == nullptr || value == nullptr || *value == '\0')
    {
        return false;
    }

    std::stringstream ss(value);
    std::string       item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty() && step == std::stoll(item))
        {
            return true;
        }
    }
    return false;
}

bool shouldTraceInitialKickAuditStep(const Step step)
{
    const char* traceDir = activeM2pTraceDirPath();
    const char* value    = std::getenv("GMX_PCFF_RESPA_TRACE_INITIAL_KICK_AUDIT");
    return step == 0 && traceDir != nullptr && value != nullptr && *value != '\0';
}

const char* xvfPreUpdateStageName(const Step step)
{
    return (step == 0) ? "STEP0_PRE_UPDATE_XVF"
           : (step == 1) ? "STEP1_PRE_UPDATE_XVF"
           : (step == 2) ? "STEP2_PRE_UPDATE_XVF"
           : (step == 3) ? "STEP3_PRE_UPDATE_XVF"
           : (step == 4) ? "STEP4_PRE_UPDATE_XVF"
           : (step == 5) ? "STEP5_PRE_UPDATE_XVF"
           : (step == 6) ? "STEP6_PRE_UPDATE_XVF"
           : (step == 7) ? "STEP7_PRE_UPDATE_XVF"
           : (step == 8) ? "STEP8_PRE_UPDATE_XVF"
           : (step == 9) ? "STEP9_PRE_UPDATE_XVF"
           : (step == 10) ? "STEP10_PRE_UPDATE_XVF"
           : (step == 11) ? "STEP11_PRE_UPDATE_XVF"
           : (step == 12) ? "STEP12_PRE_UPDATE_XVF"
           : (step == 13) ? "STEP13_PRE_UPDATE_XVF"
                         : nullptr;
}

const char* xvfUpdateInputStageName(const Step step)
{
    return (step == 0) ? "STEP0_UPDATE_INPUT_XVF"
           : (step == 1) ? "STEP1_UPDATE_INPUT_XVF"
           : (step == 2) ? "STEP2_UPDATE_INPUT_XVF"
           : (step == 3) ? "STEP3_UPDATE_INPUT_XVF"
           : (step == 4) ? "STEP4_UPDATE_INPUT_XVF"
           : (step == 5) ? "STEP5_UPDATE_INPUT_XVF"
           : (step == 6) ? "STEP6_UPDATE_INPUT_XVF"
           : (step == 7) ? "STEP7_UPDATE_INPUT_XVF"
           : (step == 8) ? "STEP8_UPDATE_INPUT_XVF"
           : (step == 9) ? "STEP9_UPDATE_INPUT_XVF"
           : (step == 10) ? "STEP10_UPDATE_INPUT_XVF"
           : (step == 11) ? "STEP11_UPDATE_INPUT_XVF"
           : (step == 12) ? "STEP12_UPDATE_INPUT_XVF"
           : (step == 13) ? "STEP13_UPDATE_INPUT_XVF"
                         : nullptr;
}

const char* xvfPostPositionCommitStageName(const Step step)
{
    return (step == 0) ? "STEP0_POST_POSITION_COMMIT_XVF"
           : (step == 1) ? "STEP1_POST_POSITION_COMMIT_XVF"
           : (step == 2) ? "STEP2_POST_POSITION_COMMIT_XVF"
           : (step == 3) ? "STEP3_POST_POSITION_COMMIT_XVF"
           : (step == 4) ? "STEP4_POST_POSITION_COMMIT_XVF"
           : (step == 5) ? "STEP5_POST_POSITION_COMMIT_XVF"
           : (step == 6) ? "STEP6_POST_POSITION_COMMIT_XVF"
           : (step == 7) ? "STEP7_POST_POSITION_COMMIT_XVF"
           : (step == 8) ? "STEP8_POST_POSITION_COMMIT_XVF"
           : (step == 9) ? "STEP9_POST_POSITION_COMMIT_XVF"
           : (step == 10) ? "STEP10_POST_POSITION_COMMIT_XVF"
           : (step == 11) ? "STEP11_POST_POSITION_COMMIT_XVF"
           : (step == 12) ? "STEP12_POST_POSITION_COMMIT_XVF"
           : (step == 13) ? "STEP13_POST_POSITION_COMMIT_XVF"
                         : nullptr;
}

thread_local Step g_positionUpdateTraceCurrentStep = -1;

template<typename Callable>
void runWithPositionUpdateTraceStep(const Step step, Callable&& callable)
{
    g_positionUpdateTraceCurrentStep = step;
    callable();
    g_positionUpdateTraceCurrentStep = -1;
}

void appendPositionUpdateTraceAtom(const char*      traceDirPath,
                                   const char*      side,
                                   const char*      rowName,
                                   Step             step,
                                   int              atomIndex,
                                   const RVec&      x,
                                   const RVec&      v,
                                   const RVec&      f,
                                   real             dt,
                                   const char*      writerName,
                                   const char*      codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::ofstream output(traceDir / "multistep_position_update_contract_trace.txt", std::ios::app);
    output << "side=" << side << " row=" << rowName << " step=" << step << " atom=" << atomIndex
           << " x=" << std::setprecision(15) << x[XX] << " y=" << std::setprecision(15) << x[YY]
           << " z=" << std::setprecision(15) << x[ZZ] << " vx=" << std::setprecision(15) << v[XX]
           << " vy=" << std::setprecision(15) << v[YY] << " vz=" << std::setprecision(15) << v[ZZ]
           << " fx=" << std::setprecision(15) << f[XX] << " fy=" << std::setprecision(15) << f[YY]
           << " fz=" << std::setprecision(15) << f[ZZ] << " dt=" << std::setprecision(15) << dt
           << " writer=" << writerName << " code_location=" << codeLocation << "\n";
}

void appendXvfStageTraceAtom(const char*    traceDirPath,
                             const char*    side,
                             const char*    stageName,
                             Step           step,
                             int            atomIndex,
                             const RVec&    x,
                             const RVec&    v,
                             const RVec&    f,
                             const char*    writerName,
                             const char*    codeLocation,
                             const char*    snapshotType,
                             const char*    boundaryKind)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::ofstream output(traceDir / "multistep_xvf_stage_trace.txt", std::ios::app);
    output << "side=" << side << " stage=" << stageName << " step=" << step << " atom=" << atomIndex
           << " x=" << std::setprecision(15) << x[XX] << " y=" << std::setprecision(15) << x[YY]
           << " z=" << std::setprecision(15) << x[ZZ] << " vx=" << std::setprecision(15) << v[XX]
           << " vy=" << std::setprecision(15) << v[YY] << " vz=" << std::setprecision(15) << v[ZZ]
           << " fx=" << std::setprecision(15) << f[XX] << " fy=" << std::setprecision(15) << f[YY]
           << " fz=" << std::setprecision(15) << f[ZZ] << " writer=" << writerName
           << " code_location=" << codeLocation << " snapshot_type=" << snapshotType
           << " boundary_kind=" << boundaryKind << "\n";
}

void appendInitialKickAuditAtom(const char* traceDirPath,
                                const char* side,
                                Step        step,
                                int         atomIndex,
                                const RVec& velocityBefore,
                                const RVec& forceUsed,
                                real        dtUsed,
                                const RVec& velocityAfter,
                                const char* writerName,
                                const char* codeLocation)
{
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::ofstream output(traceDir / "step0_initial_kick_audit_trace.txt", std::ios::app);
    output << "side=" << side << " step=" << step << " phase=Initial atom=" << atomIndex
           << " kick_order=0 level_index=-1 kick_levels=plain_total force_source_label=plainTotalForce"
           << " dt_used=" << std::setprecision(15) << dtUsed << " half_dt_used=" << std::setprecision(15)
           << (0.5 * dtUsed) << " velocity_before_x=" << std::setprecision(15) << velocityBefore[XX]
           << " velocity_before_y=" << std::setprecision(15) << velocityBefore[YY]
           << " velocity_before_z=" << std::setprecision(15) << velocityBefore[ZZ] << " force_x="
           << std::setprecision(15) << forceUsed[XX] << " force_y=" << std::setprecision(15)
           << forceUsed[YY] << " force_z=" << std::setprecision(15) << forceUsed[ZZ]
           << " dv_x=" << std::setprecision(15) << (velocityAfter[XX] - velocityBefore[XX])
           << " dv_y=" << std::setprecision(15) << (velocityAfter[YY] - velocityBefore[YY])
           << " dv_z=" << std::setprecision(15) << (velocityAfter[ZZ] - velocityBefore[ZZ])
           << " velocity_after_x=" << std::setprecision(15) << velocityAfter[XX]
           << " velocity_after_y=" << std::setprecision(15) << velocityAfter[YY]
           << " velocity_after_z=" << std::setprecision(15) << velocityAfter[ZZ] << " writer="
           << writerName << " code_location=" << codeLocation << "\n";
}

// Names of integration steps, only used locally for error messages
constexpr EnumerationArray<IntegrationStage, const char*> integrationStepNames = {
    "IntegrationStage::PositionsOnly",   "IntegrationStage::VelocitiesOnly",
    "IntegrationStage::LeapFrog",        "IntegrationStage::VelocityVerletPositionsAndVelocities",
    "IntegrationStage::ScaleVelocities", "IntegrationStage::ScalePositions"
};
} // namespace

/*! \brief Update velocities
 *
 * To maximize the ability of the compiler to optimize, all the arrays
 * of RVec should be annotated with gmx_restrict, so the compiler knows
 * there is no aliasing, and for the same reason we do not use
 * ArrayRef<RVec> for them. */
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues>
static void inline updateVelocities(int                      a,
                                    real                     dt,
                                    real                     lambdaStart,
                                    real                     lambdaEnd,
                                    const RVec* gmx_restrict invMassPerDim,
                                    RVec* gmx_restrict       v,
                                    const RVec* gmx_restrict f,
                                    const RVec&              diagPR,
                                    const Matrix3x3&         matrixPR)
{
    RVec parrinelloRahmanScaledVelocity;
    if (parrinelloRahmanVelocityScaling == ParrinelloRahmanVelocityScaling::Anisotropic)
    {
        parrinelloRahmanScaledVelocity = matrixPR * v[a];
    }
    for (int d = 0; d < DIM; d++)
    {
        // TODO: Extract this into policy classes
        if (numStartVelocityScalingValues != NumVelocityScalingValues::None
            && parrinelloRahmanVelocityScaling == ParrinelloRahmanVelocityScaling::No)
        {
            v[a][d] *= lambdaStart;
        }
        if (numStartVelocityScalingValues != NumVelocityScalingValues::None
            && parrinelloRahmanVelocityScaling == ParrinelloRahmanVelocityScaling::Diagonal)
        {
            v[a][d] *= (lambdaStart - diagPR[d]);
        }
        if (numStartVelocityScalingValues != NumVelocityScalingValues::None
            && parrinelloRahmanVelocityScaling == ParrinelloRahmanVelocityScaling::Anisotropic)
        {
            v[a][d] = lambdaStart * v[a][d] - parrinelloRahmanScaledVelocity[d];
        }
        if (numStartVelocityScalingValues == NumVelocityScalingValues::None
            && parrinelloRahmanVelocityScaling == ParrinelloRahmanVelocityScaling::Diagonal)
        {
            v[a][d] *= (1 - diagPR[d]);
        }
        if (numStartVelocityScalingValues == NumVelocityScalingValues::None
            && parrinelloRahmanVelocityScaling == ParrinelloRahmanVelocityScaling::Anisotropic)
        {
            v[a][d] -= parrinelloRahmanScaledVelocity[d];
        }
        v[a][d] += f[a][d] * invMassPerDim[a][d] * dt;
        if (numEndVelocityScalingValues != NumVelocityScalingValues::None)
        {
            v[a][d] *= lambdaEnd;
        }
    }
}

/*! \brief Update positions
 *
 * To maximize the ability of the compiler to optimize, all the arrays
 * of RVec should be annotated with gmx_restrict, so the compiler knows
 * there is no aliasing, and for the same reason we do not use
 * ArrayRef<RVec> for them. */
static void inline updatePositions(int                      a,
                                   real                     dt,
                                   const RVec* gmx_restrict x,
                                   RVec* gmx_restrict       xprime,
                                   const RVec* gmx_restrict v)
{
    for (int d = 0; d < DIM; d++)
    {
        xprime[a][d] = x[a][d] + v[a][d] * dt;
    }
}

/*! \brief Scale velocities
 *
 * To maximize the ability of the compiler to optimize, all the arrays
 * of RVec should be annotated with gmx_restrict, so the compiler knows
 * there is no aliasing, and for the same reason we do not use
 * ArrayRef<RVec> for them. */
template<NumVelocityScalingValues numStartVelocityScalingValues>
static void inline scaleVelocities(int a, real lambda, RVec* gmx_restrict v)
{
    if (numStartVelocityScalingValues != NumVelocityScalingValues::None)
    {
        for (int d = 0; d < DIM; d++)
        {
            v[a][d] *= lambda;
        }
    }
}

/*! \brief Scale positions
 *
 * To maximize the ability of the compiler to optimize, all the arrays
 * of RVec should be annotated with gmx_restrict, so the compiler knows
 * there is no aliasing, and for the same reason we do not use
 * ArrayRef<RVec> for them. */
template<NumPositionScalingValues numPositionScalingValues>
static void inline scalePositions(int a, real lambda, RVec* gmx_restrict x)
{
    if (numPositionScalingValues != NumPositionScalingValues::None)
    {
        for (int d = 0; d < DIM; d++)
        {
            x[a][d] *= lambda;
        }
    }
}

//! Is the PR matrix diagonal?
template<ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling>
static inline bool canTreatPRScalingMatrixAsDiagonal(const Matrix3x3& matrixPR)
{
    if (parrinelloRahmanVelocityScaling != ParrinelloRahmanVelocityScaling::Anisotropic)
    {
        return false;
    }
    else
    {
        return (matrixPR(YY, XX) == 0 && matrixPR(ZZ, XX) == 0 && matrixPR(ZZ, YY) == 0);
    }
}

//! Propagation (position only)
template<>
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues,
         NumPositionScalingValues        numPositionScalingValues>
void Propagator<IntegrationStage::PositionsOnly>::run()
{
    wallcycle_start(wcycle_, WallCycleCounter::Update);

    RVec*       xp = statePropagatorData_->positionsView().paddedArrayRef().data();
    const RVec* x  = statePropagatorData_->constPositionsView().paddedArrayRef().data();
    const RVec* v  = statePropagatorData_->constVelocitiesView().paddedArrayRef().data();

    int nth    = gmx_omp_nthreads_get(ModuleMultiThread::Update);
    int homenr = mdAtoms_->mdatoms()->homenr;

#pragma omp parallel for num_threads(nth) schedule(static) default(none) shared(nth, homenr, x, xp, v)
    for (int th = 0; th < nth; th++)
    {
        try
        {
            int start_th, end_th;
            getThreadAtomRange(nth, th, homenr, &start_th, &end_th);

            for (int a = start_th; a < end_th; a++)
            {
                updatePositions(a, timestep_, x, xp, v);
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    wallcycle_stop(wcycle_, WallCycleCounter::Update);
}

//! Propagation (scale position only)
template<>
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues,
         NumPositionScalingValues        numPositionScalingValues>
void Propagator<IntegrationStage::ScalePositions>::run()
{
    wallcycle_start(wcycle_, WallCycleCounter::Update);

    RVec* x = statePropagatorData_->positionsView().paddedArrayRef().data();

    const real lambda =
            (numPositionScalingValues == NumPositionScalingValues::Single) ? positionScaling_[0] : 1.0;

    int nth    = gmx_omp_nthreads_get(ModuleMultiThread::Update);
    int homenr = mdAtoms_->mdatoms()->homenr;

#pragma omp parallel for num_threads(nth) schedule(static) default(none) shared(nth, homenr, x) \
        firstprivate(lambda)
    for (int th = 0; th < nth; th++)
    {
        try
        {
            int start_th, end_th;
            getThreadAtomRange(nth, th, homenr, &start_th, &end_th);

            for (int a = start_th; a < end_th; a++)
            {
                scalePositions<numPositionScalingValues>(
                        a,
                        (numPositionScalingValues == NumPositionScalingValues::Multiple)
                                ? positionScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                : lambda,
                        x);
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    wallcycle_stop(wcycle_, WallCycleCounter::Update);
}

//! Propagation (velocity only)
template<>
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues,
         NumPositionScalingValues        numPositionScalingValues>
void Propagator<IntegrationStage::VelocitiesOnly>::run()
{
    wallcycle_start(wcycle_, WallCycleCounter::Update);

    RVec*                      v = statePropagatorData_->velocitiesView().paddedArrayRef().data();
    const RVec*                f = statePropagatorData_->constForcesView().force().data();
    const ArrayRef<const RVec> invMassPerDim = mdAtoms_->mdatoms()->invMassPerDim;

    const real lambdaStart = (numStartVelocityScalingValues == NumVelocityScalingValues::Single)
                                     ? startVelocityScaling_[0]
                                     : 1.0;
    const real lambdaEnd   = (numEndVelocityScalingValues == NumVelocityScalingValues::Single)
                                     ? endVelocityScaling_[0]
                                     : 1.0;

    const bool treatPRScalingMatrixAsDiagonal =
            canTreatPRScalingMatrixAsDiagonal<parrinelloRahmanVelocityScaling>(matrixPR_);
    const RVec diagonalOfPRScalingMatrix = treatPRScalingMatrixAsDiagonal ? diagonal(matrixPR_) : RVec{};

    const int nth    = gmx_omp_nthreads_get(ModuleMultiThread::Update);
    const int homenr = mdAtoms_->mdatoms()->homenr;

#pragma omp parallel for num_threads(nth) schedule(static) default(none) shared(v, f, invMassPerDim) \
        shared(nth, homenr, lambdaStart, lambdaEnd, treatPRScalingMatrixAsDiagonal, diagonalOfPRScalingMatrix)
    for (int th = 0; th < nth; th++)
    {
        try
        {
            int start_th, end_th;
            getThreadAtomRange(nth, th, homenr, &start_th, &end_th);

            for (int a = start_th; a < end_th; a++)
            {
                if (treatPRScalingMatrixAsDiagonal)
                {
                    updateVelocities<numStartVelocityScalingValues, ParrinelloRahmanVelocityScaling::Diagonal, numEndVelocityScalingValues>(
                            a,
                            timestep_,
                            numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaStart,
                            numEndVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? endVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaEnd,
                            invMassPerDim.data(),
                            v,
                            f,
                            diagonalOfPRScalingMatrix,
                            matrixPR_);
                }
                else
                {
                    updateVelocities<numStartVelocityScalingValues, parrinelloRahmanVelocityScaling, numEndVelocityScalingValues>(
                            a,
                            timestep_,
                            numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaStart,
                            numEndVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? endVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaEnd,
                            invMassPerDim.data(),
                            v,
                            f,
                            diagonalOfPRScalingMatrix,
                            matrixPR_);
                }
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    wallcycle_stop(wcycle_, WallCycleCounter::Update);
}

//! Propagation (leapfrog case - position and velocity)
template<>
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues,
         NumPositionScalingValues        numPositionScalingValues>
void Propagator<IntegrationStage::LeapFrog>::run()
{
    wallcycle_start(wcycle_, WallCycleCounter::Update);

    RVec*       xp = statePropagatorData_->positionsView().paddedArrayRef().data();
    const RVec* x  = statePropagatorData_->constPositionsView().paddedArrayRef().data();
    RVec*       v  = statePropagatorData_->velocitiesView().paddedArrayRef().data();
    const RVec* f  = statePropagatorData_->constForcesView().force().data();
    const ArrayRef<const RVec> invMassPerDim = mdAtoms_->mdatoms()->invMassPerDim;

    const real lambdaStart = (numStartVelocityScalingValues == NumVelocityScalingValues::Single)
                                     ? startVelocityScaling_[0]
                                     : 1.0;
    const real lambdaEnd   = (numEndVelocityScalingValues == NumVelocityScalingValues::Single)
                                     ? endVelocityScaling_[0]
                                     : 1.0;

    const bool treatPRScalingMatrixAsDiagonal =
            canTreatPRScalingMatrixAsDiagonal<parrinelloRahmanVelocityScaling>(matrixPR_);
    const RVec diagonalOfPRScalingMatrix = treatPRScalingMatrixAsDiagonal ? diagonal(matrixPR_) : RVec{};

    const int nth    = gmx_omp_nthreads_get(ModuleMultiThread::Update);
    const int homenr = mdAtoms_->mdatoms()->homenr;

#pragma omp parallel for num_threads(nth) schedule(static) default(none) \
        shared(x, xp, v, f, invMassPerDim)                               \
        firstprivate(nth, homenr, lambdaStart, lambdaEnd, treatPRScalingMatrixAsDiagonal, diagonalOfPRScalingMatrix)
    for (int th = 0; th < nth; th++)
    {
        try
        {
            int start_th, end_th;
            getThreadAtomRange(nth, th, homenr, &start_th, &end_th);

            for (int a = start_th; a < end_th; a++)
            {
                if (treatPRScalingMatrixAsDiagonal)
                {
                    updateVelocities<numStartVelocityScalingValues, ParrinelloRahmanVelocityScaling::Diagonal, numEndVelocityScalingValues>(
                            a,
                            timestep_,
                            numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaStart,
                            numEndVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? endVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaEnd,
                            invMassPerDim.data(),
                            v,
                            f,
                            diagonalOfPRScalingMatrix,
                            matrixPR_);
                }
                else
                {
                    updateVelocities<numStartVelocityScalingValues, parrinelloRahmanVelocityScaling, numEndVelocityScalingValues>(
                            a,
                            timestep_,
                            numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaStart,
                            numEndVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? endVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaEnd,
                            invMassPerDim.data(),
                            v,
                            f,
                            diagonalOfPRScalingMatrix,
                            matrixPR_);
                }
                updatePositions(a, timestep_, x, xp, v);
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    wallcycle_stop(wcycle_, WallCycleCounter::Update);
}

//! Propagation (velocity verlet stage 2 - velocity and position)
template<>
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues,
         NumPositionScalingValues        numPositionScalingValues>
void Propagator<IntegrationStage::VelocityVerletPositionsAndVelocities>::run()
{
    wallcycle_start(wcycle_, WallCycleCounter::Update);

    RVec*       xp = statePropagatorData_->positionsView().paddedArrayRef().data();
    const RVec* x  = statePropagatorData_->constPositionsView().paddedArrayRef().data();
    RVec*       v  = statePropagatorData_->velocitiesView().paddedArrayRef().data();
    const RVec* f  = statePropagatorData_->constForcesView().force().data();
    const ArrayRef<const RVec> invMassPerDim = mdAtoms_->mdatoms()->invMassPerDim;

    const real lambdaStart = (numStartVelocityScalingValues == NumVelocityScalingValues::Single)
                                     ? startVelocityScaling_[0]
                                     : 1.0;
    const real lambdaEnd   = (numEndVelocityScalingValues == NumVelocityScalingValues::Single)
                                     ? endVelocityScaling_[0]
                                     : 1.0;

    const bool treatPRScalingMatrixAsDiagonal =
            canTreatPRScalingMatrixAsDiagonal<parrinelloRahmanVelocityScaling>(matrixPR_);
    const RVec diagonalOfPRScalingMatrix = treatPRScalingMatrixAsDiagonal ? diagonal(matrixPR_) : RVec{};

    const int nth    = gmx_omp_nthreads_get(ModuleMultiThread::Update);
    const int homenr = mdAtoms_->mdatoms()->homenr;
    const Step traceStep = g_positionUpdateTraceCurrentStep;
    const bool tracePositionUpdate = shouldTracePositionUpdateStep(traceStep);
    const bool traceXvfStage       = shouldTraceXvfStageStep(traceStep);
    const bool traceInitialKickAudit = shouldTraceInitialKickAuditStep(traceStep);

#pragma omp parallel for num_threads(nth) schedule(static) default(none) \
        shared(x, xp, v, f, invMassPerDim)                               \
        firstprivate(nth, homenr, lambdaStart, lambdaEnd, treatPRScalingMatrixAsDiagonal, diagonalOfPRScalingMatrix, traceStep, tracePositionUpdate, traceXvfStage, traceInitialKickAudit)
    for (int th = 0; th < nth; th++)
    {
        try
        {
            int start_th, end_th;
            getThreadAtomRange(nth, th, homenr, &start_th, &end_th);

            for (int a = start_th; a < end_th; a++)
            {
                const bool traceThisAtomInitialKick = traceInitialKickAudit && (a == 0 || a == 5);
                RVec       velocityBeforeKick       = {};
                RVec       forceBeforeKick          = {};
                if (traceThisAtomInitialKick)
                {
                    copy_rvec(v[a], velocityBeforeKick);
                    copy_rvec(f[a], forceBeforeKick);
                }
                if (treatPRScalingMatrixAsDiagonal)
                {
                    updateVelocities<numStartVelocityScalingValues, ParrinelloRahmanVelocityScaling::Diagonal, numEndVelocityScalingValues>(
                            a,
                            0.5 * timestep_,
                            numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaStart,
                            numEndVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? endVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaEnd,
                            invMassPerDim.data(),
                            v,
                            f,
                            diagonalOfPRScalingMatrix,
                            matrixPR_);
                }
                else
                {
                    updateVelocities<numStartVelocityScalingValues, parrinelloRahmanVelocityScaling, numEndVelocityScalingValues>(
                            a,
                            0.5 * timestep_,
                            numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaStart,
                            numEndVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? endVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaEnd,
                            invMassPerDim.data(),
                            v,
                            f,
                            diagonalOfPRScalingMatrix,
                            matrixPR_);
                }
                if (traceThisAtomInitialKick)
                {
                    appendInitialKickAuditAtom(activeM2pTraceDirPath(),
                                               "PLAIN",
                                               traceStep,
                                               a,
                                               velocityBeforeKick,
                                               forceBeforeKick,
                                               timestep_,
                                               v[a],
                                               "Propagator<IntegrationStage::VelocityVerletPositionsAndVelocities>::run",
                                               "src/gromacs/modularsimulator/propagator.cpp:651");
                }
                if ((tracePositionUpdate || traceXvfStage) && (a == 0 || a == 5))
                {
                    const RVec xBefore = x[a];
                    const RVec vBefore = v[a];
                    const RVec fBefore = f[a];
                    if (traceXvfStage)
                    {
                        if (const char* stageName = xvfPreUpdateStageName(traceStep); stageName != nullptr)
                        {
                            appendXvfStageTraceAtom(activeM2pTraceDirPath(),
                                                    "PLAIN",
                                                    stageName,
                                                    traceStep,
                                                    a,
                                                    xBefore,
                                                    vBefore,
                                                    fBefore,
                                                    "Propagator<IntegrationStage::VelocityVerletPositionsAndVelocities>::run",
                                                    "src/gromacs/modularsimulator/propagator.cpp:658",
                                                    "boundary_read",
                                                    "read");
                        }
                        if (const char* stageName = xvfUpdateInputStageName(traceStep); stageName != nullptr)
                        {
                            appendXvfStageTraceAtom(activeM2pTraceDirPath(),
                                                    "PLAIN",
                                                    stageName,
                                                    traceStep,
                                                    a,
                                                    xBefore,
                                                    vBefore,
                                                    fBefore,
                                                    "Propagator<IntegrationStage::VelocityVerletPositionsAndVelocities>::run",
                                                    "src/gromacs/modularsimulator/propagator.cpp:671",
                                                    "boundary_read",
                                                    "read");
                        }
                    }
                    appendPositionUpdateTraceAtom(activeM2pTraceDirPath(),
                                                  "PLAIN",
                                                  "UPDATE_INPUT",
                                                  traceStep,
                                                  a,
                                                  xBefore,
                                                  vBefore,
                                                  fBefore,
                                                  timestep_,
                                                  "Propagator<IntegrationStage::VelocityVerletPositionsAndVelocities>::run",
                                                  "src/gromacs/modularsimulator/propagator.cpp:604");
                    updatePositions(a, timestep_, x, xp, v);
                    if (traceXvfStage)
                    {
                        if (const char* stageName = xvfPostPositionCommitStageName(traceStep);
                            stageName != nullptr)
                        {
                            appendXvfStageTraceAtom(activeM2pTraceDirPath(),
                                                    "PLAIN",
                                                    stageName,
                                                    traceStep,
                                                    a,
                                                    xp[a],
                                                    vBefore,
                                                    fBefore,
                                                    "Propagator<IntegrationStage::VelocityVerletPositionsAndVelocities>::run",
                                                    "src/gromacs/modularsimulator/propagator.cpp:690",
                                                    "post_write_commit",
                                                    "write");
                        }
                    }
                    appendPositionUpdateTraceAtom(activeM2pTraceDirPath(),
                                                  "PLAIN",
                                                  "UPDATE_OUTPUT",
                                                  traceStep,
                                                  a,
                                                  xp[a],
                                                  vBefore,
                                                  fBefore,
                                                  timestep_,
                                                  "Propagator<IntegrationStage::VelocityVerletPositionsAndVelocities>::run",
                                                  "src/gromacs/modularsimulator/propagator.cpp:616");
                }
                else
                {
                    updatePositions(a, timestep_, x, xp, v);
                }
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    wallcycle_stop(wcycle_, WallCycleCounter::Update);
}

//! Scaling (velocity scaling only)
template<>
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues,
         NumPositionScalingValues        numPositionScalingValues>
void Propagator<IntegrationStage::ScaleVelocities>::run()
{
    if (numStartVelocityScalingValues == NumVelocityScalingValues::None)
    {
        return;
    }
    wallcycle_start(wcycle_, WallCycleCounter::Update);

    RVec* v = statePropagatorData_->velocitiesView().paddedArrayRef().data();

    const real lambdaStart = (numStartVelocityScalingValues == NumVelocityScalingValues::Single)
                                     ? startVelocityScaling_[0]
                                     : 1.0;

    const int nth    = gmx_omp_nthreads_get(ModuleMultiThread::Update);
    const int homenr = mdAtoms_->mdatoms()->homenr;

#pragma omp parallel for num_threads(nth) schedule(static) default(none) \
        shared(v, lambdaStart, nth, homenr)
    for (int th = 0; th < nth; th++)
    {
        try
        {
            int start_th = 0;
            int end_th   = 0;
            getThreadAtomRange(nth, th, homenr, &start_th, &end_th);

            for (int a = start_th; a < end_th; a++)
            {
                scaleVelocities<numStartVelocityScalingValues>(
                        a,
                        numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                : lambdaStart,
                        v);
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    wallcycle_stop(wcycle_, WallCycleCounter::Update);
}

template<IntegrationStage integrationStage>
Propagator<integrationStage>::Propagator(double               timestep,
                                         StatePropagatorData* statePropagatorData,
                                         const MDAtoms*       mdAtoms,
                                         gmx_wallcycle*       wcycle) :
    timestep_(timestep),
    statePropagatorData_(statePropagatorData),
    doSingleStartVelocityScaling_(false),
    doGroupStartVelocityScaling_(false),
    doSingleEndVelocityScaling_(false),
    doGroupEndVelocityScaling_(false),
    scalingStepVelocity_(-1),
    matrixPR_{ 0 },
    scalingStepPR_(-1),
    mdAtoms_(mdAtoms),
    wcycle_(wcycle)
{
}

template<IntegrationStage integrationStage>
void Propagator<integrationStage>::scheduleTask(Step                       step,
                                                Time gmx_unused            time,
                                                const RegisterRunFunction& registerRunFunction)
{
    const bool doSingleVScalingThisStep =
            (doSingleStartVelocityScaling_ && (step == scalingStepVelocity_));
    const bool doGroupVScalingThisStep = (doGroupStartVelocityScaling_ && (step == scalingStepVelocity_));

    if (integrationStage == IntegrationStage::ScaleVelocities)
    {
        // IntegrationStage::ScaleVelocities only needs to run if some kind of
        // velocity scaling is needed on the current step.
        if (!doSingleVScalingThisStep && !doGroupVScalingThisStep)
        {
            return;
        }
    }

    if (integrationStage == IntegrationStage::ScalePositions)
    {
        // IntegrationStage::ScalePositions only needs to run if
        // position scaling is needed on the current step.
        if (step != scalingStepPosition_)
        {
            return;
        }
        // Since IntegrationStage::ScalePositions is the only stage for which position scaling
        // is implemented we handle it here to avoid enlarging the decision tree below.
        if (doSinglePositionScaling_)
        {
            registerRunFunction(
                    [this]()
                    {
                        run<NumVelocityScalingValues::None,
                            ParrinelloRahmanVelocityScaling::No,
                            NumVelocityScalingValues::None,
                            NumPositionScalingValues::Single>();
                    });
        }
        else if (doGroupPositionScaling_)
        {
            registerRunFunction(
                    [this]()
                    {
                        run<NumVelocityScalingValues::None,
                            ParrinelloRahmanVelocityScaling::No,
                            NumVelocityScalingValues::None,
                            NumPositionScalingValues::Multiple>();
                    });
        }
    }

    const bool doParrinelloRahmanThisStep = (step == scalingStepPR_);

    if (doSingleVScalingThisStep)
    {
        if (doParrinelloRahmanThisStep)
        {
            if (doSingleEndVelocityScaling_)
            {
            registerRunFunction(
                    [this, step]()
                    {
                        runWithPositionUpdateTraceStep(step,
                                                       [this]()
                                                       {
                                                           run<NumVelocityScalingValues::Single,
                                                               ParrinelloRahmanVelocityScaling::Anisotropic,
                                                               NumVelocityScalingValues::Single,
                                                               NumPositionScalingValues::None>();
                                                       });
                    });
            }
            else
            {
            registerRunFunction(
                    [this, step]()
                    {
                        runWithPositionUpdateTraceStep(step,
                                                       [this]()
                                                       {
                                                           run<NumVelocityScalingValues::Single,
                                                               ParrinelloRahmanVelocityScaling::Anisotropic,
                                                               NumVelocityScalingValues::None,
                                                               NumPositionScalingValues::None>();
                                                       });
                    });
            }
        }
        else
        {
            if (doSingleEndVelocityScaling_)
            {
                registerRunFunction(
                        [this, step]()
                        {
                            runWithPositionUpdateTraceStep(step,
                                                           [this]()
                                                           {
                                                               run<NumVelocityScalingValues::Single,
                                                                   ParrinelloRahmanVelocityScaling::No,
                                                                   NumVelocityScalingValues::Single,
                                                                   NumPositionScalingValues::None>();
                                                           });
                        });
            }
            else
            {
                registerRunFunction(
                        [this, step]()
                        {
                            runWithPositionUpdateTraceStep(step,
                                                           [this]()
                                                           {
                                                               run<NumVelocityScalingValues::Single,
                                                                   ParrinelloRahmanVelocityScaling::No,
                                                                   NumVelocityScalingValues::None,
                                                                   NumPositionScalingValues::None>();
                                                           });
                        });
            }
        }
    }
    else if (doGroupVScalingThisStep)
    {
        if (doParrinelloRahmanThisStep)
        {
            if (doGroupEndVelocityScaling_)
            {
                registerRunFunction(
                        [this, step]()
                        {
                            runWithPositionUpdateTraceStep(step,
                                                           [this]()
                                                           {
                                                               run<NumVelocityScalingValues::Multiple,
                                                                   ParrinelloRahmanVelocityScaling::Anisotropic,
                                                                   NumVelocityScalingValues::Multiple,
                                                                   NumPositionScalingValues::None>();
                                                           });
                        });
            }
            else
            {
                registerRunFunction(
                        [this, step]()
                        {
                            runWithPositionUpdateTraceStep(step,
                                                           [this]()
                                                           {
                                                               run<NumVelocityScalingValues::Multiple,
                                                                   ParrinelloRahmanVelocityScaling::Anisotropic,
                                                                   NumVelocityScalingValues::None,
                                                                   NumPositionScalingValues::None>();
                                                           });
                        });
            }
        }
        else
        {
            if (doGroupEndVelocityScaling_)
            {
                registerRunFunction(
                        [this, step]()
                        {
                            runWithPositionUpdateTraceStep(step,
                                                           [this]()
                                                           {
                                                               run<NumVelocityScalingValues::Multiple,
                                                                   ParrinelloRahmanVelocityScaling::No,
                                                                   NumVelocityScalingValues::Multiple,
                                                                   NumPositionScalingValues::None>();
                                                           });
                        });
            }
            else
            {
                registerRunFunction(
                        [this, step]()
                        {
                            runWithPositionUpdateTraceStep(step,
                                                           [this]()
                                                           {
                                                               run<NumVelocityScalingValues::Multiple,
                                                                   ParrinelloRahmanVelocityScaling::No,
                                                                   NumVelocityScalingValues::None,
                                                                   NumPositionScalingValues::None>();
                                                           });
                        });
            }
        }
    }
    else
    {
        if (doParrinelloRahmanThisStep)
        {
            registerRunFunction(
                    [this, step]()
                    {
                        runWithPositionUpdateTraceStep(step,
                                                       [this]()
                                                       {
                                                           run<NumVelocityScalingValues::None,
                                                               ParrinelloRahmanVelocityScaling::Anisotropic,
                                                               NumVelocityScalingValues::None,
                                                               NumPositionScalingValues::None>();
                                                       });
                    });
        }
        else
        {
            registerRunFunction(
                    [this, step]()
                    {
                        runWithPositionUpdateTraceStep(step,
                                                       [this]()
                                                       {
                                                           run<NumVelocityScalingValues::None,
                                                               ParrinelloRahmanVelocityScaling::No,
                                                               NumVelocityScalingValues::None,
                                                               NumPositionScalingValues::None>();
                                                       });
                    });
        }
    }
}

template<IntegrationStage integrationStage>
constexpr bool hasStartVelocityScaling()
{
    return (integrationStage == IntegrationStage::VelocitiesOnly
            || integrationStage == IntegrationStage::LeapFrog
            || integrationStage == IntegrationStage::VelocityVerletPositionsAndVelocities
            || integrationStage == IntegrationStage::ScaleVelocities);
}

template<IntegrationStage integrationStage>
constexpr bool hasEndVelocityScaling()
{
    return (hasStartVelocityScaling<integrationStage>()
            && integrationStage != IntegrationStage::ScaleVelocities);
}

template<IntegrationStage integrationStage>
constexpr bool hasPositionScaling()
{
    return (integrationStage == IntegrationStage::ScalePositions);
}

template<IntegrationStage integrationStage>
constexpr bool hasParrinelloRahmanScaling()
{
    return (integrationStage == IntegrationStage::VelocitiesOnly
            || integrationStage == IntegrationStage::LeapFrog
            || integrationStage == IntegrationStage::VelocityVerletPositionsAndVelocities);
}

template<IntegrationStage integrationStage>
void Propagator<integrationStage>::setNumVelocityScalingVariables(int numVelocityScalingVariables,
                                                                  ScaleVelocities scaleVelocities)
{
    GMX_RELEASE_ASSERT(
            hasStartVelocityScaling<integrationStage>() || hasEndVelocityScaling<integrationStage>(),
            formatString("Velocity scaling not implemented for %s", integrationStepNames[integrationStage])
                    .c_str());
    GMX_RELEASE_ASSERT(startVelocityScaling_.empty(),
                       "Number of velocity scaling variables cannot be changed once set.");

    const bool scaleEndVelocities = (scaleVelocities == ScaleVelocities::PreStepAndPostStep);
    startVelocityScaling_.resize(numVelocityScalingVariables, 1.);
    if (scaleEndVelocities)
    {
        endVelocityScaling_.resize(numVelocityScalingVariables, 1.);
    }
    doSingleStartVelocityScaling_ = numVelocityScalingVariables == 1;
    doGroupStartVelocityScaling_  = numVelocityScalingVariables > 1;
    doSingleEndVelocityScaling_   = doSingleStartVelocityScaling_ && scaleEndVelocities;
    doGroupEndVelocityScaling_    = doGroupStartVelocityScaling_ && scaleEndVelocities;
}

template<IntegrationStage integrationStage>
void Propagator<integrationStage>::setNumPositionScalingVariables(int numPositionScalingVariables)
{
    GMX_RELEASE_ASSERT(hasPositionScaling<integrationStage>(),
                       formatString("Position scaling not implemented for %s",
                                    integrationStepNames[integrationStage])
                               .c_str());
    GMX_RELEASE_ASSERT(positionScaling_.empty(),
                       "Number of position scaling variables cannot be changed once set.");
    positionScaling_.resize(numPositionScalingVariables, 1.);
    doSinglePositionScaling_ = (numPositionScalingVariables == 1);
    doGroupPositionScaling_  = (numPositionScalingVariables > 1);
}

template<IntegrationStage integrationStage>
ArrayRef<real> Propagator<integrationStage>::viewOnStartVelocityScaling()
{
    GMX_RELEASE_ASSERT(hasStartVelocityScaling<integrationStage>(),
                       formatString("Start velocity scaling not implemented for %s",
                                    integrationStepNames[integrationStage])
                               .c_str());
    GMX_RELEASE_ASSERT(!startVelocityScaling_.empty(),
                       "Number of velocity scaling variables not set.");

    return startVelocityScaling_;
}

template<IntegrationStage integrationStage>
ArrayRef<real> Propagator<integrationStage>::viewOnEndVelocityScaling()
{
    GMX_RELEASE_ASSERT(hasEndVelocityScaling<integrationStage>(),
                       formatString("End velocity scaling not implemented for %s",
                                    integrationStepNames[integrationStage])
                               .c_str());
    GMX_RELEASE_ASSERT(!endVelocityScaling_.empty(),
                       "Number of velocity scaling variables not set.");

    return endVelocityScaling_;
}

template<IntegrationStage integrationStage>
ArrayRef<real> Propagator<integrationStage>::viewOnPositionScaling()
{
    GMX_RELEASE_ASSERT(hasPositionScaling<integrationStage>(),
                       formatString("Position scaling not implemented for %s",
                                    integrationStepNames[integrationStage])
                               .c_str());
    GMX_RELEASE_ASSERT(!positionScaling_.empty(), "Number of position scaling variables not set.");

    return positionScaling_;
}

template<IntegrationStage integrationStage>
PropagatorCallback Propagator<integrationStage>::velocityScalingCallback()
{
    GMX_RELEASE_ASSERT(
            hasStartVelocityScaling<integrationStage>() || hasEndVelocityScaling<integrationStage>(),
            formatString("Velocity scaling not implemented for %s", integrationStepNames[integrationStage])
                    .c_str());

    return [this](Step step) { scalingStepVelocity_ = step; };
}

template<IntegrationStage integrationStage>
PropagatorCallback Propagator<integrationStage>::positionScalingCallback()
{
    GMX_RELEASE_ASSERT(hasPositionScaling<integrationStage>(),
                       formatString("Position scaling not implemented for %s",
                                    integrationStepNames[integrationStage])
                               .c_str());

    return [this](Step step) { scalingStepPosition_ = step; };
}

template<IntegrationStage integrationStage>
Matrix3x3* Propagator<integrationStage>::viewOnPRScalingMatrix()
{
    GMX_RELEASE_ASSERT(hasParrinelloRahmanScaling<integrationStage>(),
                       formatString("Parrinello-Rahman scaling not implemented for %s",
                                    integrationStepNames[integrationStage])
                               .c_str());
    return &matrixPR_;
}

template<IntegrationStage integrationStage>
PropagatorCallback Propagator<integrationStage>::prScalingCallback()
{
    GMX_RELEASE_ASSERT(hasParrinelloRahmanScaling<integrationStage>(),
                       formatString("Parrinello-Rahman scaling not implemented for %s",
                                    integrationStepNames[integrationStage])
                               .c_str());

    return [this](Step step) { scalingStepPR_ = step; };
}

template<IntegrationStage integrationStage>
static PropagatorConnection getConnection(Propagator<integrationStage>* propagator,
                                          const PropagatorTag&          propagatorTag)
{
    PropagatorConnection propagatorConnection{ propagatorTag };

    if constexpr (hasStartVelocityScaling<integrationStage>() || hasEndVelocityScaling<integrationStage>())
    {
        propagatorConnection.setNumVelocityScalingVariables =
                [propagator](int num, ScaleVelocities scaleVelocities)
        { propagator->setNumVelocityScalingVariables(num, scaleVelocities); };
        propagatorConnection.getVelocityScalingCallback = [propagator]()
        { return propagator->velocityScalingCallback(); };
    }
    if constexpr (hasStartVelocityScaling<integrationStage>())
    {
        propagatorConnection.getViewOnStartVelocityScaling = [propagator]()
        { return propagator->viewOnStartVelocityScaling(); };
    }
    if constexpr (hasEndVelocityScaling<integrationStage>())
    {
        propagatorConnection.getViewOnEndVelocityScaling = [propagator]()
        { return propagator->viewOnEndVelocityScaling(); };
    }
    if constexpr (hasPositionScaling<integrationStage>())
    {
        propagatorConnection.setNumPositionScalingVariables = [propagator](int num)
        { propagator->setNumPositionScalingVariables(num); };
        propagatorConnection.getViewOnPositionScaling = [propagator]()
        { return propagator->viewOnPositionScaling(); };
        propagatorConnection.getPositionScalingCallback = [propagator]()
        { return propagator->positionScalingCallback(); };
    }
    if constexpr (hasParrinelloRahmanScaling<integrationStage>())
    {
        propagatorConnection.getViewOnPRScalingMatrix = [propagator]()
        { return propagator->viewOnPRScalingMatrix(); };
        propagatorConnection.getPRScalingCallback = [propagator]()
        { return propagator->prScalingCallback(); };
    }

    return propagatorConnection;
}

// doxygen is confused by the two definitions
//! \cond
template<IntegrationStage integrationStage>
ISimulatorElement* Propagator<integrationStage>::getElementPointerImpl(
        LegacySimulatorData*                    legacySimulatorData,
        ModularSimulatorAlgorithmBuilderHelper* builderHelper,
        StatePropagatorData*                    statePropagatorData,
        EnergyData gmx_unused*                  energyData,
        FreeEnergyPerturbationData gmx_unused*  freeEnergyPerturbationData,
        GlobalCommunicationHelper gmx_unused*   globalCommunicationHelper,
        ObservablesReducer* /* observablesReducer */,
        const PropagatorTag& propagatorTag,
        TimeStep             timestep)
{
    GMX_RELEASE_ASSERT(!(integrationStage == IntegrationStage::ScaleVelocities
                         || integrationStage == IntegrationStage::ScalePositions)
                               || (timestep == 0.0),
                       "Scaling elements don't propagate the system.");
    auto* element    = builderHelper->storeElement(std::make_unique<Propagator<integrationStage>>(
            timestep, statePropagatorData, legacySimulatorData->mdAtoms_, legacySimulatorData->wallCycleCounters_));
    auto* propagator = static_cast<Propagator<integrationStage>*>(element);
    builderHelper->registerPropagator(getConnection<integrationStage>(propagator, propagatorTag));
    return element;
}

template<IntegrationStage integrationStage>
ISimulatorElement* Propagator<integrationStage>::getElementPointerImpl(
        LegacySimulatorData*                    legacySimulatorData,
        ModularSimulatorAlgorithmBuilderHelper* builderHelper,
        StatePropagatorData*                    statePropagatorData,
        EnergyData*                             energyData,
        FreeEnergyPerturbationData*             freeEnergyPerturbationData,
        GlobalCommunicationHelper*              globalCommunicationHelper,
        ObservablesReducer*                     observablesReducer,
        const PropagatorTag&                    propagatorTag)
{
    GMX_RELEASE_ASSERT(
            integrationStage == IntegrationStage::ScaleVelocities
                    || integrationStage == IntegrationStage::ScalePositions,
            "Adding a propagator without time step is only allowed for scaling elements");
    return getElementPointerImpl(legacySimulatorData,
                                 builderHelper,
                                 statePropagatorData,
                                 energyData,
                                 freeEnergyPerturbationData,
                                 globalCommunicationHelper,
                                 observablesReducer,
                                 propagatorTag,
                                 TimeStep(0.0));
}
//! \endcond

// Explicit template initializations
template class Propagator<IntegrationStage::PositionsOnly>;
template class Propagator<IntegrationStage::VelocitiesOnly>;
template class Propagator<IntegrationStage::LeapFrog>;
template class Propagator<IntegrationStage::VelocityVerletPositionsAndVelocities>;
template class Propagator<IntegrationStage::ScaleVelocities>;
template class Propagator<IntegrationStage::ScalePositions>;

} // namespace gmx
