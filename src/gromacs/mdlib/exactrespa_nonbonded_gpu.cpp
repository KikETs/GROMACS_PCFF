#include "config.h"
#include "gmxpre.h"

#include "exactrespa_nonbonded_gpu.h"

#if GMX_GPU_CUDA

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "exactrespa_nonbonded_gpu_internal.h"

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/exactrespaparameters.h"
#include "gromacs/mdtypes/exactrespaschedule.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/nbnxm/cuda/nbnxm_cuda_types.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/vec.h"

namespace gmx
{
namespace
{

bool anyActiveLevel(const ExactRespaGpuOutputView& outputView)
{
    for (const auto& level : outputView.levels)
    {
        if (level.active)
        {
            return true;
        }
    }
    return false;
}

bool anyDirectVirialLevel(const ExactRespaGpuOutputView& outputView)
{
    for (const auto& level : outputView.levels)
    {
        if (level.active && level.directVirialOutput != nullptr)
        {
            return true;
        }
    }
    return false;
}

float computePmeSelfEnergy(const interaction_const_t& interactionConstants)
{
    GMX_RELEASE_ASSERT(interactionConstants.coulombEwaldTables != nullptr,
                       "PME self-energy requires Coulomb Ewald tables");
    return 0.5F * interactionConstants.coulombEwaldTables->tableFDV0[2];
}

const char* activeM2pTraceDirPath()
{
    const char* traceDir = std::getenv("GMX_PCFF_RESPA_M2P_TRACE_DIR");
    return (traceDir != nullptr && *traceDir != '\0') ? traceDir : nullptr;
}

bool respaTraceFlagEnabled(const char* envVarName)
{
    const char* value = std::getenv(envVarName);
    return (value != nullptr && *value != '\0');
}

const std::vector<int64_t>& respaCpuCorrectionTraceSteps()
{
    static const std::vector<int64_t> steps = []()
    {
        std::vector<int64_t> parsedSteps;
        const char* value = std::getenv("GMX_PCFF_RESPA_TRACE_CPU_CORRECTION_ENERGIES_STEPS");
        if (value == nullptr || *value == '\0')
        {
            return parsedSteps;
        }

        std::stringstream ss(value);
        std::string       item;
        while (std::getline(ss, item, ','))
        {
            if (!item.empty())
            {
                parsedSteps.push_back(std::stoll(item));
            }
        }
        return parsedSteps;
    }();
    return steps;
}

bool shouldTraceCpuCorrectionEnergiesStep(const int64_t step)
{
    if (!respaTraceFlagEnabled("GMX_PCFF_RESPA_TRACE_CPU_CORRECTION_ENERGIES")
        || activeM2pTraceDirPath() == nullptr)
    {
        return false;
    }

    const auto& traceSteps = respaCpuCorrectionTraceSteps();
    if (!traceSteps.empty())
    {
        return std::find(traceSteps.begin(), traceSteps.end(), step) != traceSteps.end();
    }

    return step == 0;
}

void writeRespaTraceTextFile(const char* traceDirPath, const char* fileName, const char* contents)
{
    GMX_RELEASE_ASSERT(traceDirPath != nullptr && *traceDirPath != '\0',
                       "Need a valid r-RESPA trace directory");

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::filesystem::path outputPath = traceDir / fileName;

    FILE* dumpFile = std::fopen(outputPath.string().c_str(), "w");
    GMX_RELEASE_ASSERT(dumpFile != nullptr, "Could not open r-RESPA trace output for writing");
    std::fprintf(dumpFile, "%s", contents);
    std::fclose(dumpFile);
}

void appendRespaTraceTextLine(const char* traceDirPath, const char* fileName, const std::string& line)
{
    GMX_RELEASE_ASSERT(traceDirPath != nullptr && *traceDirPath != '\0',
                       "Need a valid r-RESPA trace directory");

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::filesystem::path outputPath = traceDir / fileName;

    FILE* dumpFile = std::fopen(outputPath.string().c_str(), "a");
    GMX_RELEASE_ASSERT(dumpFile != nullptr, "Could not open r-RESPA trace output for appending");
    std::fprintf(dumpFile, "%s\n", line.c_str());
    std::fclose(dumpFile);
}

void appendCpuCorrectionEnergyTrace(const int64_t step,
                                    const int level,
                                    const char* actualBackend,
                                    const double reciprocalEnergy,
                                    const double selfEnergy,
                                    const double excludedCorrectionEnergy,
                                    const double shortRangePairEnergy,
                                    const double shortRangeTotalEnergy,
                                    const int reciprocalCount,
                                    const int selfCount,
                                    const int excludedCount,
                                    const int pairCount)
{
    const char* traceDirPath = activeM2pTraceDirPath();
    if (traceDirPath == nullptr || *traceDirPath == '\0')
    {
        return;
    }

    static std::once_flag traceHeaderOnce;
    std::call_once(traceHeaderOnce,
                   [traceDirPath]()
                   {
                       writeRespaTraceTextFile(traceDirPath,
                                               "cpu_correction_energy_trace.tsv",
                                               "#step\tlevel\tactual_backend\tterm\tenergy_kj_mol\tinteraction_count\tdiagnostic_origin\n");
                   });

    struct TermRow
    {
        const char* name;
        double      energy;
        int         count;
    };

    const std::array<TermRow, 5> termRows = {
        { { "coulomb_pairs_short_range", shortRangePairEnergy, pairCount },
          { "coulomb_excluded_correction", excludedCorrectionEnergy, excludedCount },
          { "coulomb_self", selfEnergy, selfCount },
          { "coulomb_short_range_total", shortRangeTotalEnergy, pairCount + excludedCount + selfCount },
          { "coulomb_reciprocal", reciprocalEnergy, reciprocalCount } }
    };

    for (const TermRow& row : termRows)
    {
        std::ostringstream line;
        line.setf(std::ios::scientific);
        line.precision(17);
        line << step << '\t' << level << '\t' << actualBackend << '\t' << row.name << '\t' << row.energy << '\t'
             << row.count << '\t' << "runtime_energy_split";
        appendRespaTraceTextLine(traceDirPath, "cpu_correction_energy_trace.tsv", line.str());
    }
}

void addVirialContributionFromFlat(ForceWithVirial* output, const float* values, const int level)
{
    matrix virial;
    clear_mat(virial);
    const int offset = level * DIM * DIM;
    for (int dim1 = 0; dim1 < DIM; ++dim1)
    {
        for (int dim2 = 0; dim2 < DIM; ++dim2)
        {
            virial[dim1][dim2] = values[offset + dim1 * DIM + dim2];
        }
    }
    output->addVirialContribution(virial);
}

std::vector<ExactRespaGpuPairEntry> packPairEntries(const PlainPairlist& plainPairlist)
{
    std::vector<ExactRespaGpuPairEntry> pairEntries;
    pairEntries.reserve(plainPairlist.pairs.size() + plainPairlist.excludedPairs.size());

    for (const auto& entry : plainPairlist.pairs)
    {
        pairEntries.push_back({ entry.first.first, entry.first.second, entry.second, 0 });
    }
    for (const auto& entry : plainPairlist.excludedPairs)
    {
        pairEntries.push_back({ entry.first.first, entry.first.second, entry.second, 1 });
    }

    return pairEntries;
}

template<typename ValueType>
void freeDeviceBufferIfAllocated(DeviceBuffer<ValueType>* buffer)
{
    if (buffer != nullptr && *buffer != nullptr)
    {
        freeDeviceBuffer(buffer);
    }
}

} // namespace

bool exactRespaNonbondedGpuSupported(const t_inputrec& inputrec, const t_forcerec& fr)
{
    return (sizeof(real) == sizeof(float)) && useExactRespa(inputrec) && exactRespaHasPairSplitting(inputrec)
           && fr.plainPairlistRange.has_value() && fr.nbv != nullptr;
}

void computeExactRespaNonbondedGpu(const t_inputrec&              inputrec,
                                   t_forcerec*                    fr,
                                   const t_mdatoms&               mdatoms,
                                   ArrayRef<const RVec>           coordinates,
                                   const ExactRespaGpuOutputView& outputView,
                                   gmx_enerdata_t*                enerd,
                                   const StepWorkload&            stepWork,
                                   const int64_t                  step)
{
    static_assert(sizeof(real) == sizeof(float),
                  "CUDA exact r-RESPA GPU path currently supports mixed/single precision builds only");

    GMX_RELEASE_ASSERT(sizeof(real) == sizeof(float),
                       "CUDA exact r-RESPA GPU path currently supports mixed/single precision builds only");
    GMX_RELEASE_ASSERT(useExactRespa(inputrec),
                       "CUDA exact r-RESPA GPU path requires exact r-RESPA to be enabled");
    GMX_RELEASE_ASSERT(exactRespaHasPairSplitting(inputrec),
                       "CUDA exact r-RESPA GPU path requires standalone exact pair splitting");
    GMX_RELEASE_ASSERT(fr->ic->coulombEwaldTables,
                       "CUDA exact r-RESPA GPU path requires PME/Ewald tables");
    GMX_RELEASE_ASSERT(fr->plainPairlistRange.has_value(),
                       "CUDA exact r-RESPA GPU path requires a plain pairlist");
    GMX_RELEASE_ASSERT(fr->nbv != nullptr,
                       "CUDA exact r-RESPA GPU path requires a nonbonded Verlet object");
    GMX_RELEASE_ASSERT(fr->nbv->gpuNbv() != nullptr,
                       "CUDA exact r-RESPA GPU path requires an initialized GPU nonbonded object");
    GMX_RELEASE_ASSERT(mdatoms.nenergrp <= 1,
                       "CUDA exact r-RESPA GPU path currently supports a single energy group");

    if (!anyActiveLevel(outputView))
    {
        return;
    }

    const auto& plainPairlist = fr->nbv->plainPairlist(fr->plainPairlistRange.value(), fr->shift_vec);
    const auto  pairEntries   = packPairEntries(plainPairlist);
    const int   excludedPairCount =
            static_cast<int>(std::count_if(pairEntries.begin(),
                                           pairEntries.end(),
                                           [](const ExactRespaGpuPairEntry& entry) { return entry.excluded != 0; }));
    const int pairCount = static_cast<int>(pairEntries.size()) - excludedPairCount;
    if (pairEntries.empty())
    {
        return;
    }

    auto* gpuNbv = fr->nbv->gpuNbv();
    GMX_RELEASE_ASSERT(gpuNbv->deviceContext_ != nullptr,
                       "CUDA exact r-RESPA GPU path requires an initialized device context");
    GMX_RELEASE_ASSERT(gpuNbv->deviceStreams[InteractionLocality::Local] != nullptr,
                       "CUDA exact r-RESPA GPU path requires a local nonbonded stream");
    const DeviceContext& deviceContext = *gpuNbv->deviceContext_;
    const DeviceStream&  deviceStream  = *gpuNbv->deviceStreams[InteractionLocality::Local];
    GMX_RELEASE_ASSERT(deviceStream.isValid(),
                       "CUDA exact r-RESPA GPU path requires a valid local nonbonded stream");
    deviceContext.activate();

    ExactRespaGpuRuntimeParams params;
    params.numAtoms          = static_cast<int>(coordinates.size());
    params.numPairs          = static_cast<int>(pairEntries.size());
    params.numTypes          = fr->ntype;
    params.ntype2            = 2 * fr->ntype;
    params.centralShiftIndex = c_centralShiftIndex;
    params.innerLevel        = exactRespaNonbondedInnerLevel(inputrec);
    params.outerLevel        = exactRespaNonbondedOuterLevel(inputrec);
    params.middleLevel =
            inputrec.exactRespa.forceLayout.hasMiddle() ? exactRespaNonbondedMiddleLevel(inputrec) : -1;
    params.hasMiddle       = inputrec.exactRespa.forceLayout.hasMiddle() ? 1 : 0;
    params.innerOff        = inputrec.exactRespa.forceLayout.innerOff;
    params.innerOn         = inputrec.exactRespa.forceLayout.innerOn;
    params.outerOn         = inputrec.exactRespa.forceLayout.outerOn;
    params.outerOff        = inputrec.exactRespa.forceLayout.outerOff;
    params.coulombCutoff2  = fr->ic->coulomb.cutoff * fr->ic->coulomb.cutoff;
    params.vdwCutoff2      = fr->ic->vdw.cutoff * fr->ic->vdw.cutoff;
    params.repulsionPower  = static_cast<float>(fr->ic->vdw.repulsionPower);
    params.invRepulsionPower =
            (params.repulsionPower != 0.0F) ? (1.0F / params.repulsionPower) : 0.0F;
    params.epsfac                 = fr->ic->coulomb.epsfac;
    params.ewaldShift             = fr->ic->coulomb.ewaldShift;
    params.coulombTableScale      = fr->ic->coulombEwaldTables->scale;
    params.coulombTableElementCount =
            static_cast<int>(fr->ic->coulombEwaldTables->tableFDV0.size());

    for (int level = 0; level < ExactRespaGpuOutputView::c_numLevels; ++level)
    {
        if (!outputView.levels[level].active)
        {
            continue;
        }
        params.activeLevelMask |= (1 << level);
        if (!outputView.levels[level].shift.empty())
        {
            params.shiftLevelMask |= (1 << level);
        }
        if (outputView.levels[level].directVirialOutput != nullptr)
        {
            params.directVirialLevelMask |= (1 << level);
        }
    }
    if (stepWork.computeEnergy && outputView.levels[params.outerLevel].active)
    {
        params.accumulateEnergyMask |= (1 << params.outerLevel);
    }

    DeviceBuffer<ExactRespaGpuPairEntry> d_pairEntries = nullptr;
    DeviceBuffer<Float3>                 d_levelForces = nullptr;
    DeviceBuffer<Float3>                 d_levelShiftForces = nullptr;
    DeviceBuffer<float>                  d_levelLjEnergies = nullptr;
    DeviceBuffer<float>                  d_levelCoulombEnergies = nullptr;
    DeviceBuffer<float>                  d_levelExcludedCoulombEnergies = nullptr;
    DeviceBuffer<float>                  d_levelVirials = nullptr;
    DeviceBuffer<Float3>                 d_coordinates = nullptr;
    DeviceBuffer<Float3>                 d_shiftVectors = nullptr;
    DeviceBuffer<int>                    d_atomTypes = nullptr;
    DeviceBuffer<float>                  d_atomCharges = nullptr;
    DeviceBuffer<float>                  d_nbfp = nullptr;
    DeviceBuffer<float>                  d_coulombTable = nullptr;

    allocateDeviceBuffer(&d_pairEntries, pairEntries.size(), deviceContext);
    allocateDeviceBuffer(&d_levelForces,
                         ExactRespaGpuOutputView::c_numLevels * static_cast<int>(coordinates.size()),
                         deviceContext);
    allocateDeviceBuffer(&d_levelShiftForces,
                         ExactRespaGpuOutputView::c_numLevels * c_numShiftVectors,
                         deviceContext);
    allocateDeviceBuffer(&d_levelLjEnergies, ExactRespaGpuOutputView::c_numLevels, deviceContext);
    allocateDeviceBuffer(&d_levelCoulombEnergies, ExactRespaGpuOutputView::c_numLevels, deviceContext);
    allocateDeviceBuffer(&d_levelExcludedCoulombEnergies,
                         ExactRespaGpuOutputView::c_numLevels,
                         deviceContext);
    allocateDeviceBuffer(&d_levelVirials, ExactRespaGpuOutputView::c_numLevels * DIM * DIM, deviceContext);
    allocateDeviceBuffer(&d_coordinates, coordinates.size(), deviceContext);
    allocateDeviceBuffer(&d_shiftVectors, fr->shift_vec.size(), deviceContext);
    allocateDeviceBuffer(&d_atomTypes, mdatoms.typeA.size(), deviceContext);
    allocateDeviceBuffer(&d_atomCharges, mdatoms.chargeA.size(), deviceContext);
    allocateDeviceBuffer(&d_nbfp, fr->nbfp.size(), deviceContext);
    allocateDeviceBuffer(&d_coulombTable, fr->ic->coulombEwaldTables->tableFDV0.size(), deviceContext);

    clearDeviceBufferAsync(&d_levelForces,
                           0,
                           ExactRespaGpuOutputView::c_numLevels * static_cast<int>(coordinates.size()),
                           deviceStream);
    clearDeviceBufferAsync(
            &d_levelShiftForces, 0, ExactRespaGpuOutputView::c_numLevels * c_numShiftVectors, deviceStream);
    clearDeviceBufferAsync(&d_levelLjEnergies, 0, ExactRespaGpuOutputView::c_numLevels, deviceStream);
    clearDeviceBufferAsync(&d_levelCoulombEnergies, 0, ExactRespaGpuOutputView::c_numLevels, deviceStream);
    clearDeviceBufferAsync(&d_levelExcludedCoulombEnergies,
                           0,
                           ExactRespaGpuOutputView::c_numLevels,
                           deviceStream);
    clearDeviceBufferAsync(&d_levelVirials,
                           0,
                           ExactRespaGpuOutputView::c_numLevels * DIM * DIM,
                           deviceStream);

    copyToDeviceBuffer(&d_pairEntries,
                       pairEntries.data(),
                       0,
                       pairEntries.size(),
                       deviceStream,
                       GpuApiCallBehavior::Sync,
                       nullptr);
    copyToDeviceBuffer(&d_coordinates,
                       asGenericFloat3Pointer(coordinates.data()),
                       0,
                       coordinates.size(),
                       deviceStream,
                       GpuApiCallBehavior::Sync,
                       nullptr);
    copyToDeviceBuffer(&d_shiftVectors,
                       asGenericFloat3Pointer(fr->shift_vec),
                       0,
                       fr->shift_vec.size(),
                       deviceStream,
                       GpuApiCallBehavior::Sync,
                       nullptr);
    copyToDeviceBuffer(&d_atomTypes,
                       mdatoms.typeA.data(),
                       0,
                       mdatoms.typeA.size(),
                       deviceStream,
                       GpuApiCallBehavior::Sync,
                       nullptr);
    copyToDeviceBuffer(&d_atomCharges,
                       mdatoms.chargeA.data(),
                       0,
                       mdatoms.chargeA.size(),
                       deviceStream,
                       GpuApiCallBehavior::Sync,
                       nullptr);
    copyToDeviceBuffer(&d_nbfp,
                       fr->nbfp.data(),
                       0,
                       fr->nbfp.size(),
                       deviceStream,
                       GpuApiCallBehavior::Sync,
                       nullptr);
    copyToDeviceBuffer(&d_coulombTable,
                       fr->ic->coulombEwaldTables->tableFDV0.data(),
                       0,
                       fr->ic->coulombEwaldTables->tableFDV0.size(),
                       deviceStream,
                       GpuApiCallBehavior::Sync,
                       nullptr);

    launchExactRespaNonbondedGpuKernel(params,
                                       d_pairEntries,
                                       d_coordinates,
                                       d_shiftVectors,
                                       d_atomTypes,
                                       d_atomCharges,
                                       d_nbfp,
                                       d_coulombTable,
                                       d_levelForces,
                                       d_levelShiftForces,
                                       d_levelLjEnergies,
                                       d_levelCoulombEnergies,
                                       d_levelExcludedCoulombEnergies,
                                       d_levelVirials,
                                       deviceStream);

    std::array<float, ExactRespaGpuOutputView::c_numLevels> h_levelLjEnergies = { 0.0F, 0.0F, 0.0F };
    std::array<float, ExactRespaGpuOutputView::c_numLevels> h_levelCoulombEnergies = { 0.0F, 0.0F, 0.0F };
    std::array<float, ExactRespaGpuOutputView::c_numLevels> h_levelExcludedCoulombEnergies = { 0.0F, 0.0F, 0.0F };
    std::array<float, ExactRespaGpuOutputView::c_numLevels * DIM * DIM> h_levelVirials = {};

    for (int level = 0; level < ExactRespaGpuOutputView::c_numLevels; ++level)
    {
        if (!outputView.levels[level].active)
        {
            continue;
        }

        copyFromDeviceBuffer(asGenericFloat3Pointer(outputView.levels[level].force.data()),
                             &d_levelForces,
                             level * params.numAtoms,
                             params.numAtoms,
                             deviceStream,
                             GpuApiCallBehavior::Sync,
                             nullptr);
        if (!outputView.levels[level].shift.empty())
        {
            copyFromDeviceBuffer(asGenericFloat3Pointer(outputView.levels[level].shift.data()),
                                 &d_levelShiftForces,
                                 level * c_numShiftVectors,
                                 c_numShiftVectors,
                                 deviceStream,
                                 GpuApiCallBehavior::Sync,
                                 nullptr);
        }
    }

    copyFromDeviceBuffer(h_levelLjEnergies.data(),
                         &d_levelLjEnergies,
                         0,
                         ExactRespaGpuOutputView::c_numLevels,
                         deviceStream,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    copyFromDeviceBuffer(h_levelCoulombEnergies.data(),
                         &d_levelCoulombEnergies,
                         0,
                         ExactRespaGpuOutputView::c_numLevels,
                         deviceStream,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    copyFromDeviceBuffer(h_levelExcludedCoulombEnergies.data(),
                         &d_levelExcludedCoulombEnergies,
                         0,
                         ExactRespaGpuOutputView::c_numLevels,
                         deviceStream,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    if (anyDirectVirialLevel(outputView))
    {
        copyFromDeviceBuffer(h_levelVirials.data(),
                             &d_levelVirials,
                             0,
                             ExactRespaGpuOutputView::c_numLevels * DIM * DIM,
                             deviceStream,
                             GpuApiCallBehavior::Sync,
                             nullptr);
    }

    for (int level = 0; level < ExactRespaGpuOutputView::c_numLevels; ++level)
    {
        if (outputView.levels[level].active && outputView.levels[level].directVirialOutput != nullptr)
        {
            addVirialContributionFromFlat(
                    outputView.levels[level].directVirialOutput, h_levelVirials.data(), level);
        }
    }

    if ((params.accumulateEnergyMask & (1 << params.outerLevel)) != 0)
    {
        enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJSR][0] += h_levelLjEnergies[params.outerLevel];
        enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR][0] +=
                h_levelCoulombEnergies[params.outerLevel];

        const float pmeSelfEnergy = computePmeSelfEnergy(*fr->ic);
        float       selfEnergyTotal = 0.0F;
        int         selfEnergyCount = 0;
        for (int atom = 0; atom < fr->natoms_force_constr; ++atom)
        {
            const float charge = mdatoms.chargeA[atom];
            if (charge == 0.0F)
            {
                continue;
            }
            const float selfEnergy = -fr->ic->coulomb.epsfac * charge * charge * pmeSelfEnergy;
            enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR][0] += selfEnergy;
            selfEnergyTotal += selfEnergy;
            ++selfEnergyCount;
        }
        if (shouldTraceCpuCorrectionEnergiesStep(step))
        {
            const double reciprocalEnergy =
                    static_cast<double>(enerd->term[InteractionFunction::CoulombReciprocalSpace]);
            const double excludedCorrectionEnergy =
                    static_cast<double>(h_levelExcludedCoulombEnergies[params.outerLevel]);
            const double shortRangePairEnergy =
                    static_cast<double>(h_levelCoulombEnergies[params.outerLevel]) - excludedCorrectionEnergy;
            const double shortRangeTotalEnergy =
                    static_cast<double>(h_levelCoulombEnergies[params.outerLevel]) + selfEnergyTotal;
            appendCpuCorrectionEnergyTrace(step,
                                           params.outerLevel,
                                           "gpu_offload_enabled",
                                           reciprocalEnergy,
                                           selfEnergyTotal,
                                           excludedCorrectionEnergy,
                                           shortRangePairEnergy,
                                           shortRangeTotalEnergy,
                                           1,
                                           selfEnergyCount,
                                           excludedPairCount,
                                           pairCount);
        }
    }

    freeDeviceBufferIfAllocated(&d_levelExcludedCoulombEnergies);
    freeDeviceBufferIfAllocated(&d_coulombTable);
    freeDeviceBufferIfAllocated(&d_nbfp);
    freeDeviceBufferIfAllocated(&d_atomCharges);
    freeDeviceBufferIfAllocated(&d_atomTypes);
    freeDeviceBufferIfAllocated(&d_shiftVectors);
    freeDeviceBufferIfAllocated(&d_coordinates);
    freeDeviceBufferIfAllocated(&d_levelVirials);
    freeDeviceBufferIfAllocated(&d_levelCoulombEnergies);
    freeDeviceBufferIfAllocated(&d_levelLjEnergies);
    freeDeviceBufferIfAllocated(&d_levelShiftForces);
    freeDeviceBufferIfAllocated(&d_levelForces);
    freeDeviceBufferIfAllocated(&d_pairEntries);
}

} // namespace gmx

#else

#include "gromacs/utility/gmxassert.h"

namespace gmx
{

bool exactRespaNonbondedGpuSupported(const t_inputrec&, const t_forcerec&)
{
    return false;
}

void computeExactRespaNonbondedGpu(const t_inputrec&,
                                   t_forcerec*,
                                   const t_mdatoms&,
                                   ArrayRef<const RVec>,
                                   const ExactRespaGpuOutputView&,
                                   gmx_enerdata_t*,
                                   const StepWorkload&,
                                   int64_t)
{
    GMX_RELEASE_ASSERT(false, "CUDA exact r-RESPA GPU path is only available in CUDA builds");
}

} // namespace gmx

#endif
