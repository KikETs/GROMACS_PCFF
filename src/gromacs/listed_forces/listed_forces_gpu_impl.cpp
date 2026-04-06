/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2018- The GROMACS Authors
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
 *
 * \brief Implements GPU bonded lists for non-GPU builds
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_listed_forces
 */

#include "gmxpre.h"

#include "config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gromacs/gpu_utils/capabilities.h"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/multipletimestepping.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/message_string_collector.h"
#include "gromacs/utility/vectypes.h"

class DeviceContext;
class DeviceStream;
struct gmx_enerdata_t;
struct gmx_ffparams_t;
struct gmx_wallcycle;

namespace gmx
{
struct NBAtomDataGpu;
class StepWorkload;

//! Returns whether there are any interactions in ilists suitable for a GPU.
static bool someInteractionsCanRunOnGpu(const InteractionLists& ilists)
{
    // Perturbation is not implemented in the GPU bonded
    // kernels. If all the interactions were actually
    // perturbed, then that will be detected later on each
    // domain, and work will never run on the GPU. This is
    // very unlikely to occur, and has little run-time cost,
    // so we don't complicate the code by catering for it
    // here.
    return std::any_of(fTypesOnGpu.begin(),
                       fTypesOnGpu.end(),
                       [ilists](InteractionFunction fType)
                       { return !ilists[static_cast<int>(fType)].iatoms.empty(); });
}

//! Returns whether there are any bonded interactions in the global topology suitable for a GPU.
static bool bondedInteractionsCanRunOnGpu(const gmx_mtop_t& mtop)
{
    // Check the regular molecule types
    for (const auto& moltype : mtop.moltype)
    {
        if (someInteractionsCanRunOnGpu(moltype.ilist))
        {
            return true;
        }
    }
    // Check the inter-molecular interactions.
    if (mtop.intermolecular_ilist)
    {
        if (someInteractionsCanRunOnGpu(*mtop.intermolecular_ilist))
        {
            return true;
        }
    }
    return false;
}

//! Returns whether \p fType appears anywhere in the topology.
static bool topologyContainsInteractionType(const gmx_mtop_t& mtop, const InteractionFunction fType)
{
    for (const auto& moltype : mtop.moltype)
    {
        if (!moltype.ilist[fType].iatoms.empty())
        {
            return true;
        }
    }
    return (mtop.intermolecular_ilist != nullptr && !(*mtop.intermolecular_ilist)[fType].iatoms.empty());
}

//! Returns whether any of the listed interaction types appear anywhere in the topology.
static bool topologyContainsAnyInteractionType(const gmx_mtop_t&                    mtop,
                                               const std::initializer_list<InteractionFunction> fTypes)
{
    return std::any_of(fTypes.begin(),
                       fTypes.end(),
                       [&mtop](const InteractionFunction fType)
                       { return topologyContainsInteractionType(mtop, fType); });
}

bool buildSupportsListedForcesGpu(std::string* error)
{
    MessageStringCollector errorReasons;
    // Before changing the prefix string, make sure that it is not searched for in regression tests.
    errorReasons.startContext("Bonded interactions on GPU are not supported in:");
    errorReasons.appendIf(GMX_DOUBLE, "Double precision build of GROMACS");
    errorReasons.appendIf(!GpuConfigurationCapabilities::Bonded, "Current GPU backend");
    errorReasons.appendIf(!GMX_GPU, "CPU-only build of GROMACS");
    errorReasons.finishContext();
    if (error != nullptr)
    {
        *error = errorReasons.toString();
    }
    return errorReasons.isEmpty();
}

bool inputSupportsListedForcesGpu(const t_inputrec& ir, const gmx_mtop_t& mtop, std::string* error)
{
    MessageStringCollector errorReasons;
    const bool isExactLammpsRespa = useExactRespa(ir);
    const bool hasLennardJones14Pairs =
            topologyContainsInteractionType(mtop, InteractionFunction::LennardJones14);
    const bool hasListedPairGpuSemantics =
            topologyContainsAnyInteractionType(mtop,
                                              { InteractionFunction::LennardJones14,
                                                InteractionFunction::LennardJonesCoulomb14Q,
                                                InteractionFunction::Coulomb14 });
    const bool hasExactHybridUnsupportedGpuBondedTypes =
            topologyContainsAnyInteractionType(mtop,
                                              { InteractionFunction::Bonds,
                                                InteractionFunction::Angles,
                                                InteractionFunction::UreyBradleyPotential,
                                                InteractionFunction::ProperDihedrals,
                                                InteractionFunction::RyckaertBellemansDihedrals,
                                                InteractionFunction::ImproperDihedrals,
                                                InteractionFunction::PeriodicImproperDihedrals });
    const bool hasExactHybridUnsupportedPairInteractionTypes =
            topologyContainsAnyInteractionType(mtop,
                                              { InteractionFunction::LennardJonesCoulomb14Q,
                                                InteractionFunction::Coulomb14 });
    const bool repulsionPowerSupported =
            std::abs(mtop.ffparams.reppow - 12.0) <= 10 * GMX_DOUBLE_EPS
            || (isExactLammpsRespa && std::abs(mtop.ffparams.reppow - 9.0) <= 10 * GMX_DOUBLE_EPS);
    // Before changing the prefix string, make sure that it is not searched for in regression tests.
    errorReasons.startContext("Bonded interactions can not be computed on a GPU:");

    errorReasons.appendIf(!bondedInteractionsCanRunOnGpu(mtop),
                          "None of the bonded types are implemented on the GPU.");
    errorReasons.appendIf(
            !EI_DYNAMICS(ir.eI),
            "Cannot compute bonded interactions on a GPU, because GPU implementation requires "
            "a dynamical integrator (md, sd, etc).");
    errorReasons.appendIf(EI_MIMIC(ir.eI), "MiMiC");
    if (isExactLammpsRespa)
    {
        errorReasons.appendIf(
                !hasLennardJones14Pairs,
                "Exact LAMMPS-style r-RESPA hybrid bonded GPU is only admitted in the pair14-only "
                "narrow mode, so the topology must contain Lennard-Jones 1-4 listed pairs.");
        errorReasons.appendIf(
                hasExactHybridUnsupportedGpuBondedTypes,
                "Exact LAMMPS-style r-RESPA hybrid bonded GPU is only admitted in the pair14-only "
                "narrow mode. GPU-capable bond/angle/dihedral/improper kernels must remain on the CPU.");
        errorReasons.appendIf(
                hasExactHybridUnsupportedPairInteractionTypes,
                "Exact LAMMPS-style r-RESPA hybrid bonded GPU is only admitted for "
                "InteractionFunction::LennardJones14 listed pairs. Coulomb14 and "
                "LennardJonesCoulomb14Q remain unsupported in the exact GPU path.");
    }
    else if (ir.useMts)
    {
        errorReasons.append("Cannot run with multiple time stepping");
    }
    // There is one energy group for each wall and those are not used for 1-4 interactions
    errorReasons.appendIf((ir.opts.ngener - ir.nwall > 1), "Cannot run with multiple energy groups");
    errorReasons.appendIf(hasListedPairGpuSemantics && !repulsionPowerSupported,
                          isExactLammpsRespa
                                  ? "Exact LAMMPS-style r-RESPA hybrid bonded GPU only supports 12-6 "
                                    "and 9-6 listed 1-4 Lennard-Jones semantics."
                                  : "The bonded GPU path only supports 12-6 listed 1-4 Lennard-Jones "
                                    "semantics outside the exact LAMMPS-style r-RESPA pair14-only mode.");
    errorReasons.finishContext();
    if (error != nullptr)
    {
        *error = errorReasons.toString();
    }
    return errorReasons.isEmpty();
}

#if !GMX_GPU || GMX_GPU_OPENCL

class ListedForcesGpu::Impl
{
};

ListedForcesGpu::ListedForcesGpu(const gmx_ffparams_t& /* ffparams */,
                                 const float /* electrostaticsScaleFactor */,
                                 const int /* numEnergyGroupsForListedForces */,
                                 const DeviceContext& /* deviceContext */,
                                 const DeviceStream& /* deviceStream */,
                                 gmx_wallcycle* /* wcycle */) :
    impl_(nullptr)
{
}

ListedForcesGpu::~ListedForcesGpu() = default;

void ListedForcesGpu::updateHaveInteractions(const InteractionDefinitions& /*idef*/) {}

void ListedForcesGpu::updateInteractionListsAndDeviceBuffers(ArrayRef<const int> /* nbnxnAtomOrder */,
                                                             const InteractionDefinitions& /* idef */,
                                                             NBAtomDataGpu* /* nbnxmAtomDataGpu */)
{
}

void ListedForcesGpu::setPbc(PbcType /* pbcType */, const matrix /* box */, bool /* canMoleculeSpanPbc */)
{
}

bool ListedForcesGpu::haveInteractions() const
{
    return !impl_;
}

void ListedForcesGpu::launchKernel(const gmx::StepWorkload& /* stepWork */) {}

void ListedForcesGpu::setPbcAndlaunchKernel(PbcType /* pbcType */,
                                            const matrix /* box */,
                                            bool /* canMoleculeSpanPbc */,
                                            const gmx::StepWorkload& /* stepWork */)
{
}

void ListedForcesGpu::launchEnergyTransfer() {}

void ListedForcesGpu::waitAccumulateEnergyTerms(gmx_enerdata_t* /* enerd */) {}

void ListedForcesGpu::clearEnergies() {}

#endif // !GMX_GPU || GMX_GPU_OPENCL

} // namespace gmx
