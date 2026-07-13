#include "config.h"
#include "gmxpre.h"

#include "exactrespa_nonbonded_gpu.h"

#if GMX_GPU_CUDA

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "exactrespa_nonbonded_gpu_internal.h"

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/exactrespaparameters.h"
#include "gromacs/mdtypes/exactrespaschedule.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/md_enums.h"
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

bool exactRespaEwaldRealOnlyRequested(const t_inputrec& inputrec)
{
    const char* env = std::getenv("GMX_PCFF_EWALD_REAL_ONLY");
    return env != nullptr && *env != '\0' && std::strcmp(env, "0") != 0 && useExactRespa(inputrec)
           && usingPmeOrEwald(inputrec.coulombtype);
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

int packPairEntries(const PlainPairlist& plainPairlist, HostVector<ExactRespaGpuPairEntry>* pairEntries)
{
    pairEntries->clear();
    pairEntries->reserve(plainPairlist.pairs.size() + plainPairlist.excludedPairs.size());

    for (const auto& entry : plainPairlist.pairs)
    {
        pairEntries->push_back({ entry.first.first, entry.first.second, entry.second, 0 });
    }
    for (const auto& entry : plainPairlist.excludedPairs)
    {
        pairEntries->push_back({ entry.first.first, entry.first.second, entry.second, 1 });
    }

    return static_cast<int>(plainPairlist.excludedPairs.size());
}

template<typename ValueType>
void freeDeviceBufferIfAllocated(DeviceBuffer<ValueType>* buffer)
{
    if (buffer != nullptr && *buffer != nullptr)
    {
        freeDeviceBuffer(buffer);
    }
}

template<typename ValueType>
struct CachedDeviceBuffer
{
    DeviceBuffer<ValueType> buffer       = nullptr;
    int                     numValues    = 0;
    int                     maxNumValues = -1;

    ~CachedDeviceBuffer() { freeDeviceBufferIfAllocated(&buffer); }

    void ensureCapacity(const size_t requestedValues, const DeviceContext& deviceContext)
    {
        reallocateDeviceBuffer(&buffer, requestedValues, &numValues, &maxNumValues, deviceContext);
    }
};

struct CachedHostUpload
{
    const void* hostPtr   = nullptr;
    size_t      numValues = 0;
    float       scale     = 0.0F;
    float       shift     = 0.0F;

    bool shouldUpload(const void* newHostPtr, const size_t newNumValues)
    {
        const bool upload = hostPtr != newHostPtr || numValues != newNumValues;
        hostPtr           = newHostPtr;
        numValues         = newNumValues;
        return upload;
    }

    bool shouldUpload(const void* newHostPtr, const size_t newNumValues, const float newScale, const float newShift)
    {
        const bool upload = hostPtr != newHostPtr || numValues != newNumValues || scale != newScale
                            || shift != newShift;
        hostPtr           = newHostPtr;
        numValues         = newNumValues;
        scale             = newScale;
        shift             = newShift;
        return upload;
    }
};

GpuApiCallBehavior copyKindForHostBuffer(const void* hostBuffer)
{
    return (hostBuffer != nullptr && isHostMemoryPinned(hostBuffer)) ? GpuApiCallBehavior::Async
                                                                     : GpuApiCallBehavior::Sync;
}

template<typename ValueType>
const ValueType* pinnedHostBufferForGpuCopy(const ValueType*       hostBuffer,
                                            const size_t           numValues,
                                            HostVector<ValueType>* stagedBuffer)
{
    GMX_ASSERT(stagedBuffer != nullptr, "Need a pinned staging buffer");
    if (numValues == 0 || hostBuffer == nullptr || isHostMemoryPinned(hostBuffer))
    {
        return hostBuffer;
    }

    stagedBuffer->assign(hostBuffer, hostBuffer + numValues);
    return stagedBuffer->data();
}

} // namespace

struct ExactRespaNonbondedGpuScratch
{
    HostVector<ExactRespaGpuPairEntry> pairEntriesHost = {
        {}, HostAllocationPolicy(PinningPolicy::PinnedIfSupported)
    };
    bool         pairEntriesHostValid          = false;
    const void* pairEntriesPairsPtr           = nullptr;
    size_t       pairEntriesPairsCount         = 0;
    const void* pairEntriesExcludedPairsPtr   = nullptr;
    size_t       pairEntriesExcludedPairsCount = 0;
    int          pairEntriesExcludedPairCount  = 0;
    CachedDeviceBuffer<ExactRespaGpuPairEntry> pairEntries;
    CachedDeviceBuffer<Float3>                 levelForces;
    CachedDeviceBuffer<Float3>                 levelShiftForces;
    CachedDeviceBuffer<float>                  levelEnergies;
    CachedDeviceBuffer<float>                  levelVirials;
    CachedDeviceBuffer<Float3>                 coordinates;
    CachedDeviceBuffer<Float3>                 shiftVectors;
    CachedDeviceBuffer<int>                    atomTypes;
    CachedDeviceBuffer<float>                  atomCharges;
    CachedDeviceBuffer<float>                  nbfp;
    CachedDeviceBuffer<float>                  coulombTable;
    HostVector<Float3>                         coordinatesHost = {
        {}, HostAllocationPolicy(PinningPolicy::PinnedIfSupported)
    };
    HostVector<Float3> shiftVectorsHost = { {}, HostAllocationPolicy(PinningPolicy::PinnedIfSupported) };
    HostVector<int>    atomTypesHost    = { {}, HostAllocationPolicy(PinningPolicy::PinnedIfSupported) };
    HostVector<float> atomChargesHost = { {}, HostAllocationPolicy(PinningPolicy::PinnedIfSupported) };
    HostVector<float> nbfpHost        = { {}, HostAllocationPolicy(PinningPolicy::PinnedIfSupported) };
    HostVector<float> coulombTableHost = { {}, HostAllocationPolicy(PinningPolicy::PinnedIfSupported) };
    HostVector<float>                          levelEnergiesHost = {
        {}, HostAllocationPolicy(PinningPolicy::PinnedIfSupported)
    };
    HostVector<float> levelVirialsHost = { {}, HostAllocationPolicy(PinningPolicy::PinnedIfSupported) };
    CachedHostUpload pairEntriesUpload;
    CachedHostUpload atomTypesUpload;
    CachedHostUpload atomChargesUpload;
    CachedHostUpload                           nbfpUpload;
    CachedHostUpload                           coulombTableUpload;
};

void ExactRespaNonbondedGpuScratchDeleter::operator()(ExactRespaNonbondedGpuScratch* scratch) const
{
    delete scratch;
}

bool exactRespaNonbondedGpuSupported(const t_inputrec& inputrec, const t_forcerec& fr)
{
    return (sizeof(real) == sizeof(float)) && useExactRespa(inputrec) && exactRespaHasPairSplitting(inputrec)
           && (usingPmeOrEwald(inputrec.coulombtype)
               || inputrec.coulombtype == CoulombInteractionType::Cut)
           && fr.completePairlistRange.has_value() && fr.nbv != nullptr;
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
    static_assert(c_exactRespaGpuNumEnergyLevels == ExactRespaGpuOutputView::c_numLevels,
                  "Exact r-RESPA GPU energy buffer layout must match the output level count");

    GMX_RELEASE_ASSERT(sizeof(real) == sizeof(float),
                       "CUDA exact r-RESPA GPU path currently supports mixed/single precision builds only");
    GMX_RELEASE_ASSERT(useExactRespa(inputrec),
                       "CUDA exact r-RESPA GPU path requires exact r-RESPA to be enabled");
    GMX_RELEASE_ASSERT(exactRespaHasPairSplitting(inputrec),
                       "CUDA exact r-RESPA GPU path requires standalone exact pair splitting");
    const bool usingEwaldCoulomb  = usingPmeOrEwald(fr->ic->coulomb.type);
    const bool usingCutoffCoulomb = (fr->ic->coulomb.type == CoulombInteractionType::Cut);
    const bool ewaldRealOnly      = exactRespaEwaldRealOnlyRequested(inputrec);
    GMX_RELEASE_ASSERT(usingEwaldCoulomb || usingCutoffCoulomb,
                       "CUDA exact r-RESPA GPU path requires PME/Ewald or Cut-off Coulomb");
    GMX_RELEASE_ASSERT(!usingEwaldCoulomb || fr->ic->coulombEwaldTables,
                       "CUDA exact r-RESPA GPU path requires PME/Ewald tables for PME/Ewald Coulomb");
    GMX_RELEASE_ASSERT(fr->completePairlistRange.has_value(),
                       "CUDA exact r-RESPA GPU path requires a complete pairlist");
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

    if (fr->exactRespaNonbondedGpuScratch == nullptr)
    {
        fr->exactRespaNonbondedGpuScratch.reset(new ExactRespaNonbondedGpuScratch);
    }
    auto& scratch = *fr->exactRespaNonbondedGpuScratch;

    const auto& plainPairlist = fr->nbv->plainPairlist(fr->completePairlistRange.value(), fr->shift_vec);
    const void* pairsPtr =
            plainPairlist.pairs.empty() ? nullptr : static_cast<const void*>(plainPairlist.pairs.data());
    const void* excludedPairsPtr = plainPairlist.excludedPairs.empty()
                                           ? nullptr
                                           : static_cast<const void*>(plainPairlist.excludedPairs.data());
    const bool rebuildPackedPairEntries =
            stepWork.doNeighborSearch || !scratch.pairEntriesHostValid
            || scratch.pairEntriesPairsPtr != pairsPtr
            || scratch.pairEntriesPairsCount != plainPairlist.pairs.size()
            || scratch.pairEntriesExcludedPairsPtr != excludedPairsPtr
            || scratch.pairEntriesExcludedPairsCount != plainPairlist.excludedPairs.size();
    if (rebuildPackedPairEntries)
    {
        scratch.pairEntriesExcludedPairCount = packPairEntries(plainPairlist, &scratch.pairEntriesHost);
        scratch.pairEntriesHostValid         = true;
        scratch.pairEntriesPairsPtr          = pairsPtr;
        scratch.pairEntriesPairsCount        = plainPairlist.pairs.size();
        scratch.pairEntriesExcludedPairsPtr  = excludedPairsPtr;
        scratch.pairEntriesExcludedPairsCount = plainPairlist.excludedPairs.size();
    }
    const int   excludedPairCount = scratch.pairEntriesExcludedPairCount;
    const auto& pairEntries       = scratch.pairEntriesHost;
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
    params.coulombUsesEwaldTable = usingEwaldCoulomb ? 1 : 0;
    params.suppressEwaldExcludedAndSelf = 0;
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
    if (usingEwaldCoulomb)
    {
        params.ewaldShift             = fr->ic->coulomb.ewaldShift;
        params.coulombTableScale      = fr->ic->coulombEwaldTables->scale;
        params.coulombTableElementCount =
                static_cast<int>(fr->ic->coulombEwaldTables->tableFDV0.size());
    }

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
    const bool accumulateAnyEnergy = params.accumulateEnergyMask != 0;
    const bool accumulateAnyShift  = params.shiftLevelMask != 0;
    const bool accumulateAnyDirectVirial = params.directVirialLevelMask != 0;

    scratch.pairEntries.ensureCapacity(pairEntries.size(), deviceContext);
    scratch.levelForces.ensureCapacity(
            ExactRespaGpuOutputView::c_numLevels * static_cast<int>(coordinates.size()), deviceContext);
    if (accumulateAnyShift)
    {
        scratch.levelShiftForces.ensureCapacity(ExactRespaGpuOutputView::c_numLevels * c_numShiftVectors,
                                                deviceContext);
    }
    if (accumulateAnyEnergy)
    {
        scratch.levelEnergies.ensureCapacity(c_exactRespaGpuNumEnergyValues, deviceContext);
    }
    if (accumulateAnyDirectVirial)
    {
        scratch.levelVirials.ensureCapacity(ExactRespaGpuOutputView::c_numLevels * DIM * DIM,
                                            deviceContext);
    }
    scratch.coordinates.ensureCapacity(coordinates.size(), deviceContext);
    scratch.shiftVectors.ensureCapacity(fr->shift_vec.size(), deviceContext);
    scratch.atomTypes.ensureCapacity(mdatoms.typeA.size(), deviceContext);
    scratch.atomCharges.ensureCapacity(mdatoms.chargeA.size(), deviceContext);
    scratch.nbfp.ensureCapacity(fr->nbfp.size(), deviceContext);
    if (usingEwaldCoulomb)
    {
        scratch.coulombTable.ensureCapacity(fr->ic->coulombEwaldTables->tableFDV0.size(),
                                            deviceContext);
    }

    auto& d_pairEntries                  = scratch.pairEntries.buffer;
    auto& d_levelForces                  = scratch.levelForces.buffer;
    auto& d_levelShiftForces             = scratch.levelShiftForces.buffer;
    auto& d_levelEnergies                = scratch.levelEnergies.buffer;
    auto& d_levelVirials                 = scratch.levelVirials.buffer;
    auto& d_coordinates                  = scratch.coordinates.buffer;
    auto& d_shiftVectors                 = scratch.shiftVectors.buffer;
    auto& d_atomTypes                    = scratch.atomTypes.buffer;
    auto& d_atomCharges                  = scratch.atomCharges.buffer;
    auto& d_nbfp                         = scratch.nbfp.buffer;
    auto& d_coulombTable                 = scratch.coulombTable.buffer;

    const int numAtoms = static_cast<int>(coordinates.size());
    for (int level = 0; level < ExactRespaGpuOutputView::c_numLevels; ++level)
    {
        if ((params.activeLevelMask & (1 << level)) == 0)
        {
            continue;
        }
        clearDeviceBufferAsync(&d_levelForces, level * numAtoms, numAtoms, deviceStream);
    }
    if (accumulateAnyShift)
    {
        for (int level = 0; level < ExactRespaGpuOutputView::c_numLevels; ++level)
        {
            if ((params.shiftLevelMask & (1 << level)) == 0)
            {
                continue;
            }
            clearDeviceBufferAsync(&d_levelShiftForces,
                                   level * c_numShiftVectors,
                                   c_numShiftVectors,
                                   deviceStream);
        }
    }
    if (accumulateAnyEnergy)
    {
        for (int level = 0; level < ExactRespaGpuOutputView::c_numLevels; ++level)
        {
            if ((params.accumulateEnergyMask & (1 << level)) == 0)
            {
                continue;
            }
            clearDeviceBufferAsync(&d_levelEnergies,
                                   c_exactRespaGpuLjEnergyOffset + level,
                                   1,
                                   deviceStream);
            clearDeviceBufferAsync(&d_levelEnergies,
                                   c_exactRespaGpuCoulombEnergyOffset + level,
                                   1,
                                   deviceStream);
            clearDeviceBufferAsync(&d_levelEnergies,
                                   c_exactRespaGpuExcludedCoulombEnergyOffset + level,
                                   1,
                                   deviceStream);
        }
    }
    if (accumulateAnyDirectVirial)
    {
        for (int level = 0; level < ExactRespaGpuOutputView::c_numLevels; ++level)
        {
            if ((params.directVirialLevelMask & (1 << level)) == 0)
            {
                continue;
            }
            clearDeviceBufferAsync(&d_levelVirials, level * DIM * DIM, DIM * DIM, deviceStream);
        }
    }

    bool pendingAsyncGpuWork = false;
    const Float3* h_coordinates =
            pinnedHostBufferForGpuCopy(asGenericFloat3Pointer(coordinates.data()),
                                       coordinates.size(),
                                       &scratch.coordinatesHost);
    const Float3* h_shiftVectors = pinnedHostBufferForGpuCopy(asGenericFloat3Pointer(fr->shift_vec.data()),
                                                              fr->shift_vec.size(),
                                                              &scratch.shiftVectorsHost);

    const bool pairEntriesPointerOrSizeChanged =
            scratch.pairEntriesUpload.shouldUpload(pairEntries.data(), pairEntries.size());
    if (stepWork.doNeighborSearch || pairEntriesPointerOrSizeChanged)
    {
        copyToDeviceBuffer(&d_pairEntries,
                           pairEntries.data(),
                           0,
                           pairEntries.size(),
                           deviceStream,
                           GpuApiCallBehavior::Async,
                           nullptr);
        pendingAsyncGpuWork = true;
    }
    copyToDeviceBuffer(&d_coordinates,
                       h_coordinates,
                       0,
                       coordinates.size(),
                       deviceStream,
                       GpuApiCallBehavior::Async,
                       nullptr);
    pendingAsyncGpuWork = true;
    copyToDeviceBuffer(&d_shiftVectors,
                       h_shiftVectors,
                       0,
                       fr->shift_vec.size(),
                       deviceStream,
                       GpuApiCallBehavior::Async,
                       nullptr);
    pendingAsyncGpuWork = true;
    const bool uploadAtomTypes = mdatoms.nTypePerturbed != 0
                                 || scratch.atomTypesUpload.shouldUpload(mdatoms.typeA.data(),
                                                                         mdatoms.typeA.size());
    if (uploadAtomTypes)
    {
        const int* h_atomTypes = pinnedHostBufferForGpuCopy(mdatoms.typeA.data(),
                                                            mdatoms.typeA.size(),
                                                            &scratch.atomTypesHost);
        copyToDeviceBuffer(&d_atomTypes,
                           h_atomTypes,
                           0,
                           mdatoms.typeA.size(),
                           deviceStream,
                           GpuApiCallBehavior::Async,
                           nullptr);
        pendingAsyncGpuWork = true;
    }
    const bool uploadAtomCharges = mdatoms.nChargePerturbed != 0
                                   || scratch.atomChargesUpload.shouldUpload(mdatoms.chargeA.data(),
                                                                             mdatoms.chargeA.size());
    if (uploadAtomCharges)
    {
        const float* h_atomCharges = pinnedHostBufferForGpuCopy(mdatoms.chargeA.data(),
                                                                mdatoms.chargeA.size(),
                                                                &scratch.atomChargesHost);
        copyToDeviceBuffer(&d_atomCharges,
                           h_atomCharges,
                           0,
                           mdatoms.chargeA.size(),
                           deviceStream,
                           GpuApiCallBehavior::Async,
                           nullptr);
        pendingAsyncGpuWork = true;
    }
    if (scratch.nbfpUpload.shouldUpload(fr->nbfp.data(), fr->nbfp.size()))
    {
        const float* h_nbfp = pinnedHostBufferForGpuCopy(fr->nbfp.data(), fr->nbfp.size(), &scratch.nbfpHost);
        copyToDeviceBuffer(&d_nbfp,
                           h_nbfp,
                           0,
                           fr->nbfp.size(),
                           deviceStream,
                           GpuApiCallBehavior::Async,
                           nullptr);
        pendingAsyncGpuWork = true;
    }
    if (usingEwaldCoulomb)
    {
        const auto& ewaldTable = fr->ic->coulombEwaldTables->tableFDV0;
        if (scratch.coulombTableUpload.shouldUpload(
                    ewaldTable.data(), ewaldTable.size(), params.coulombTableScale, params.ewaldShift))
        {
            const float* h_coulombTable = pinnedHostBufferForGpuCopy(
                    ewaldTable.data(), ewaldTable.size(), &scratch.coulombTableHost);
            copyToDeviceBuffer(&d_coulombTable,
                               h_coulombTable,
                               0,
                               ewaldTable.size(),
                               deviceStream,
                               GpuApiCallBehavior::Async,
                               nullptr);
            pendingAsyncGpuWork = true;
        }
    }

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
                                       d_levelEnergies,
                                       d_levelVirials,
                                       deviceStream);

    bool pendingAsyncReadback = false;
    scratch.levelEnergiesHost.resize(accumulateAnyEnergy ? c_exactRespaGpuNumEnergyValues : 0);
    scratch.levelVirialsHost.resize(accumulateAnyDirectVirial
                                            ? ExactRespaGpuOutputView::c_numLevels * DIM * DIM
                                            : 0);

    for (int level = 0; level < ExactRespaGpuOutputView::c_numLevels; ++level)
    {
        if (!outputView.levels[level].active)
        {
            continue;
        }

        const GpuApiCallBehavior forceCopyKind =
                copyKindForHostBuffer(outputView.levels[level].force.data());
        pendingAsyncReadback = pendingAsyncReadback || forceCopyKind == GpuApiCallBehavior::Async;
        copyFromDeviceBuffer(asGenericFloat3Pointer(outputView.levels[level].force.data()),
                             &d_levelForces,
                             level * params.numAtoms,
                             params.numAtoms,
                             deviceStream,
                             forceCopyKind,
                             nullptr);
        if (!outputView.levels[level].shift.empty())
        {
            const GpuApiCallBehavior shiftCopyKind =
                    copyKindForHostBuffer(outputView.levels[level].shift.data());
            pendingAsyncReadback = pendingAsyncReadback
                                   || shiftCopyKind == GpuApiCallBehavior::Async;
            copyFromDeviceBuffer(asGenericFloat3Pointer(outputView.levels[level].shift.data()),
                                 &d_levelShiftForces,
                                 level * c_numShiftVectors,
                                 c_numShiftVectors,
                                 deviceStream,
                                 shiftCopyKind,
                                 nullptr);
        }
    }

    if (accumulateAnyEnergy)
    {
        const GpuApiCallBehavior energyCopyKind =
                copyKindForHostBuffer(scratch.levelEnergiesHost.data());
        pendingAsyncReadback = pendingAsyncReadback || energyCopyKind == GpuApiCallBehavior::Async;
        copyFromDeviceBuffer(scratch.levelEnergiesHost.data(),
                             &d_levelEnergies,
                             0,
                             c_exactRespaGpuNumEnergyValues,
                             deviceStream,
                             energyCopyKind,
                             nullptr);
    }
    if (accumulateAnyDirectVirial)
    {
        const GpuApiCallBehavior virialCopyKind =
                copyKindForHostBuffer(scratch.levelVirialsHost.data());
        pendingAsyncReadback = pendingAsyncReadback || virialCopyKind == GpuApiCallBehavior::Async;
        copyFromDeviceBuffer(scratch.levelVirialsHost.data(),
                             &d_levelVirials,
                             0,
                             ExactRespaGpuOutputView::c_numLevels * DIM * DIM,
                             deviceStream,
                             virialCopyKind,
                             nullptr);
    }

    if (pendingAsyncGpuWork || pendingAsyncReadback)
    {
        deviceStream.synchronize();
    }

    for (int level = 0; level < ExactRespaGpuOutputView::c_numLevels; ++level)
    {
        if (outputView.levels[level].active && outputView.levels[level].directVirialOutput != nullptr)
        {
            addVirialContributionFromFlat(
                    outputView.levels[level].directVirialOutput,
                    scratch.levelVirialsHost.data(),
                    level);
        }
    }

    if ((params.accumulateEnergyMask & (1 << params.outerLevel)) != 0)
    {
        const float levelLjEnergy =
                scratch.levelEnergiesHost[c_exactRespaGpuLjEnergyOffset + params.outerLevel];
        const float levelCoulombEnergy =
                scratch.levelEnergiesHost[c_exactRespaGpuCoulombEnergyOffset + params.outerLevel];
        const float levelExcludedCoulombEnergy =
                scratch.levelEnergiesHost[c_exactRespaGpuExcludedCoulombEnergyOffset + params.outerLevel];

        enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJSR][0] += levelLjEnergy;
        enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR][0] +=
                levelCoulombEnergy;

        float       selfEnergyTotal = 0.0F;
        int         selfEnergyCount = 0;
        if (usingEwaldCoulomb && !ewaldRealOnly)
        {
            const float pmeSelfEnergy = computePmeSelfEnergy(*fr->ic);
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
        }
        if (shouldTraceCpuCorrectionEnergiesStep(step))
        {
            const double reciprocalEnergy =
                    static_cast<double>(enerd->term[InteractionFunction::CoulombReciprocalSpace]);
            const double excludedCorrectionEnergy =
                    static_cast<double>(levelExcludedCoulombEnergy);
            const double shortRangePairEnergy = static_cast<double>(levelCoulombEnergy)
                                                - excludedCorrectionEnergy;
            const double shortRangeTotalEnergy =
                    static_cast<double>(levelCoulombEnergy) + selfEnergyTotal;
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

}

} // namespace gmx

#else

#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

void ExactRespaNonbondedGpuScratchDeleter::operator()(ExactRespaNonbondedGpuScratch* scratch) const
{
    GMX_RELEASE_ASSERT(scratch == nullptr,
                       "Exact r-RESPA GPU scratch should not be allocated in non-CUDA builds");
}

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
