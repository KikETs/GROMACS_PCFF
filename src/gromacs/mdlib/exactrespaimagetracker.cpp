/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */

#include "gmxpre.h"

#include "exactrespaimagetracker.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/sysinfo.h"

namespace gmx
{

namespace
{

constexpr const char* c_sidecarMagic = "GMX_PCFF_EXACT_RESPA_IMAGE_SIDECAR";
constexpr int         c_sidecarVersion = 1;

ExactRespaImageBox doubleBox(const matrix box)
{
    ExactRespaImageBox result = {};
    for (int i = 0; i < DIM; ++i)
    {
        for (int j = 0; j < DIM; ++j)
        {
            result[i][j] = box[i][j];
        }
    }
    return result;
}

DVec doublePosition(const RVec& position)
{
    return { static_cast<double>(position[XX]),
             static_cast<double>(position[YY]),
             static_cast<double>(position[ZZ]) };
}

double representationTolerance(const double scale)
{
    // In a mixed-precision build state coordinates and boxes have already been
    // rounded to float. The tolerance only accepts a candidate after rounding
    // its lattice coefficient to an integer; it is not a displacement cutoff.
    return 128.0 * static_cast<double>(std::numeric_limits<real>::epsilon())
                   * std::max(1.0, scale)
           + 64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, scale);
}

bool equivalentDouble(const double lhs, const double rhs)
{
    const double scale = std::max({ 1.0, std::abs(lhs), std::abs(rhs) });
    return std::abs(lhs - rhs) <= representationTolerance(scale);
}

bool equivalentPosition(const DVec& lhs, const DVec& rhs)
{
    return equivalentDouble(lhs[XX], rhs[XX]) && equivalentDouble(lhs[YY], rhs[YY])
           && equivalentDouble(lhs[ZZ], rhs[ZZ]);
}

void validateBox(const ExactRespaImageBox& box, const char* context)
{
    for (int i = 0; i < DIM; ++i)
    {
        for (int j = 0; j < DIM; ++j)
        {
            if (!std::isfinite(box[i][j]))
            {
                GMX_THROW(InvalidInputError(
                        formatString("%s contains a non-finite box element", context)));
            }
        }
        if (!(box[i][i] > 0.0))
        {
            GMX_THROW(InvalidInputError(
                    formatString("%s requires a positive box diagonal", context)));
        }
        for (int j = i + 1; j < DIM; ++j)
        {
            if (!equivalentDouble(box[i][j], 0.0))
            {
                GMX_THROW(InvalidInputError(formatString(
                        "%s requires the lower-triangular GROMACS box representation", context)));
            }
        }
    }
}

void validateSameBox(const ExactRespaImageBox& expected,
                     const ExactRespaImageBox& actual,
                     const char*               context)
{
    validateBox(expected, context);
    validateBox(actual, context);
    for (int i = 0; i < DIM; ++i)
    {
        for (int j = 0; j < DIM; ++j)
        {
            if (!equivalentDouble(expected[i][j], actual[i][j]))
            {
                GMX_THROW(InvalidInputError(formatString(
                        "%s box mismatch at [%d][%d]: sidecar %.17g, state %.17g",
                        context,
                        i,
                        j,
                        expected[i][j],
                        actual[i][j])));
            }
        }
    }
}

void requireToken(std::istream& input,
                  const char*   expected,
                  const std::filesystem::path& path)
{
    std::string token;
    if (!(input >> token) || token != expected)
    {
        GMX_THROW(FileIOError(formatString("Invalid exact r-RESPA image sidecar '%s': "
                                           "expected token '%s'",
                                           path.string().c_str(),
                                           expected)));
    }
}

[[noreturn]] void throwSidecarIoError(const std::filesystem::path& path, const char* operation)
{
    GMX_THROW(FileIOError(formatString("Could not %s exact r-RESPA image sidecar '%s': %s",
                                       operation,
                                       path.string().c_str(),
                                       std::strerror(errno))));
}

} // namespace

DVec exactRespaContinuousPosition(const DVec&                statePosition,
                                  const ExactRespaAtomImage& image,
                                  const ExactRespaImageBox&  box)
{
    DVec result = statePosition;
    for (int imageDimension = 0; imageDimension < DIM; ++imageDimension)
    {
        for (int coordinateDimension = 0; coordinateDimension < DIM; ++coordinateDimension)
        {
            result[coordinateDimension] += static_cast<double>(image[imageDimension])
                                           * box[imageDimension][coordinateDimension];
        }
    }
    return result;
}

std::optional<ExactRespaAtomImage> exactRespaIntegerLatticeShift(
        const DVec& fromPosition, const DVec& toPosition, const ExactRespaImageBox& box)
{
    validateBox(box, "Exact r-RESPA image tracking");

    DVec difference = { fromPosition[XX] - toPosition[XX],
                        fromPosition[YY] - toPosition[YY],
                        fromPosition[ZZ] - toPosition[ZZ] };
    ExactRespaAtomImage image = { 0, 0, 0 };

    // GROMACS stores triclinic box vectors as rows in a lower-triangular
    // matrix. Solve from z to x so all off-diagonal contributions are known.
    for (int dimension = DIM - 1; dimension >= 0; --dimension)
    {
        double residual = difference[dimension];
        for (int higherDimension = dimension + 1; higherDimension < DIM; ++higherDimension)
        {
            residual -= static_cast<double>(image[higherDimension])
                        * box[higherDimension][dimension];
        }
        const double coefficient = residual / box[dimension][dimension];
        if (!std::isfinite(coefficient)
            || coefficient < static_cast<double>(std::numeric_limits<int64_t>::min())
            || coefficient > static_cast<double>(std::numeric_limits<int64_t>::max()))
        {
            return std::nullopt;
        }
        image[dimension] = static_cast<int64_t>(std::llround(coefficient));
    }

    const DVec reconstructed = exactRespaContinuousPosition(toPosition, image, box);
    if (!equivalentPosition(reconstructed, fromPosition))
    {
        return std::nullopt;
    }
    return image;
}

ExactRespaImageSidecar readExactRespaImageSidecar(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input)
    {
        GMX_THROW(FileIOError(formatString("Could not open exact r-RESPA image sidecar '%s'",
                                           path.string().c_str())));
    }

    requireToken(input, c_sidecarMagic, path);
    int version = 0;
    if (!(input >> version) || version != c_sidecarVersion)
    {
        GMX_THROW(FileIOError(formatString("Unsupported exact r-RESPA image sidecar version in '%s'",
                                           path.string().c_str())));
    }

    ExactRespaImageSidecar sidecar;
    requireToken(input, "step", path);
    if (!(input >> sidecar.step))
    {
        GMX_THROW(FileIOError(formatString("Invalid step in exact r-RESPA image sidecar '%s'",
                                           path.string().c_str())));
    }

    requireToken(input, "natoms", path);
    int64_t numAtoms = 0;
    if (!(input >> numAtoms) || numAtoms <= 0
        || numAtoms > static_cast<int64_t>(std::numeric_limits<int>::max()))
    {
        GMX_THROW(FileIOError(formatString("Invalid atom count in exact r-RESPA image sidecar '%s'",
                                           path.string().c_str())));
    }

    requireToken(input, "box", path);
    for (int i = 0; i < DIM; ++i)
    {
        for (int j = 0; j < DIM; ++j)
        {
            if (!(input >> sidecar.box[i][j]))
            {
                GMX_THROW(FileIOError(formatString("Invalid box in exact r-RESPA image sidecar '%s'",
                                                   path.string().c_str())));
            }
        }
    }
    validateBox(sidecar.box, "Exact r-RESPA image sidecar");

    requireToken(input, "atoms", path);
    sidecar.atoms.resize(numAtoms);
    for (int64_t atomIndex = 0; atomIndex < numAtoms; ++atomIndex)
    {
        auto& atom = sidecar.atoms[atomIndex];
        if (!(input >> atom.globalAtomIndex >> atom.image[XX] >> atom.image[YY]
              >> atom.image[ZZ] >> atom.statePosition[XX] >> atom.statePosition[YY]
              >> atom.statePosition[ZZ] >> atom.continuousPosition[XX]
              >> atom.continuousPosition[YY] >> atom.continuousPosition[ZZ]))
        {
            GMX_THROW(FileIOError(formatString("Invalid atom row %lld in exact r-RESPA image sidecar '%s'",
                                               static_cast<long long>(atomIndex),
                                               path.string().c_str())));
        }
        if (atom.globalAtomIndex != atomIndex)
        {
            GMX_THROW(InvalidInputError(formatString(
                    "Exact r-RESPA image sidecar '%s' atom order mismatch at row %lld",
                    path.string().c_str(),
                    static_cast<long long>(atomIndex))));
        }
        const DVec reconstructed =
                exactRespaContinuousPosition(atom.statePosition, atom.image, sidecar.box);
        if (!equivalentPosition(reconstructed, atom.continuousPosition))
        {
            GMX_THROW(InvalidInputError(formatString(
                    "Exact r-RESPA image sidecar '%s' has inconsistent image data for atom %lld",
                    path.string().c_str(),
                    static_cast<long long>(atomIndex))));
        }
    }
    requireToken(input, "end", path);
    std::string trailingToken;
    if (input >> trailingToken)
    {
        GMX_THROW(FileIOError(formatString("Unexpected trailing data in exact r-RESPA image sidecar '%s'",
                                           path.string().c_str())));
    }

    return sidecar;
}

void writeExactRespaImageSidecarAtomically(const std::filesystem::path& path,
                                           const ExactRespaImageSidecar& sidecar)
{
    if (path.empty())
    {
        GMX_THROW(InvalidInputError("Exact r-RESPA image sidecar output path is empty"));
    }
    if (sidecar.atoms.empty())
    {
        GMX_THROW(InvalidInputError("Exact r-RESPA image sidecar has no atoms"));
    }
    validateBox(sidecar.box, "Exact r-RESPA image sidecar output");

    std::error_code error;
    if (std::filesystem::exists(path, error))
    {
        GMX_THROW(FileIOError(formatString(
                "Refusing to overwrite existing exact r-RESPA image sidecar '%s'",
                path.string().c_str())));
    }
    if (error)
    {
        GMX_THROW(FileIOError(formatString("Could not inspect exact r-RESPA image sidecar path '%s': %s",
                                           path.string().c_str(),
                                           error.message().c_str())));
    }

    const std::filesystem::path temporaryPath =
            path.string() + formatString(".tmp.%d", gmx_getpid());
    if (std::filesystem::exists(temporaryPath, error))
    {
        GMX_THROW(FileIOError(formatString("Temporary exact r-RESPA image sidecar already exists: '%s'",
                                           temporaryPath.string().c_str())));
    }

    FILE* output = std::fopen(temporaryPath.string().c_str(), "w");
    if (output == nullptr)
    {
        throwSidecarIoError(temporaryPath, "open temporary");
    }

    bool writeSucceeded = true;
    writeSucceeded = writeSucceeded
                     && std::fprintf(output, "%s %d\n", c_sidecarMagic, c_sidecarVersion) >= 0;
    writeSucceeded = writeSucceeded
                     && std::fprintf(output, "step %lld\n", static_cast<long long>(sidecar.step)) >= 0;
    writeSucceeded = writeSucceeded
                     && std::fprintf(output, "natoms %zu\n", sidecar.atoms.size()) >= 0;
    writeSucceeded = writeSucceeded && std::fprintf(output, "box\n") >= 0;
    for (int i = 0; i < DIM && writeSucceeded; ++i)
    {
        writeSucceeded = std::fprintf(output,
                                      "%.17g %.17g %.17g\n",
                                      sidecar.box[i][XX],
                                      sidecar.box[i][YY],
                                      sidecar.box[i][ZZ])
                         >= 0;
    }
    writeSucceeded = writeSucceeded && std::fprintf(output, "atoms\n") >= 0;
    for (Index atomIndex = 0; atomIndex < ssize(sidecar.atoms) && writeSucceeded; ++atomIndex)
    {
        const auto& atom = sidecar.atoms[atomIndex];
        const DVec reconstructed =
                exactRespaContinuousPosition(atom.statePosition, atom.image, sidecar.box);
        if (atom.globalAtomIndex != atomIndex
            || !equivalentPosition(reconstructed, atom.continuousPosition))
        {
            writeSucceeded = false;
            break;
        }
        writeSucceeded = writeSucceeded
                         && std::fprintf(
                                    output,
                                    "%lld %lld %lld %lld %.17g %.17g %.17g %.17g %.17g %.17g\n",
                                    static_cast<long long>(atom.globalAtomIndex),
                                    static_cast<long long>(atom.image[XX]),
                                    static_cast<long long>(atom.image[YY]),
                                    static_cast<long long>(atom.image[ZZ]),
                                    atom.statePosition[XX],
                                    atom.statePosition[YY],
                                    atom.statePosition[ZZ],
                                    atom.continuousPosition[XX],
                                    atom.continuousPosition[YY],
                                    atom.continuousPosition[ZZ])
                                    >= 0;
    }
    writeSucceeded = writeSucceeded && std::fprintf(output, "end\n") >= 0;
    writeSucceeded = writeSucceeded && std::fflush(output) == 0;
    writeSucceeded = writeSucceeded && gmx_fsync(output) == 0;
    const int closeResult = std::fclose(output);
    writeSucceeded        = writeSucceeded && closeResult == 0;

    if (!writeSucceeded)
    {
        std::filesystem::remove(temporaryPath, error);
        GMX_THROW(FileIOError(formatString("Could not write exact r-RESPA image sidecar '%s'",
                                           temporaryPath.string().c_str())));
    }

    try
    {
        gmx_file_rename(temporaryPath, path);
    }
    catch (...)
    {
        std::filesystem::remove(temporaryPath, error);
        throw;
    }
}

ExactRespaImageTracker::ExactRespaImageTracker()
{
    const char* inputPath  = std::getenv("GMX_PCFF_EXACT_RESPA_IMAGE_SIDECAR_IN");
    const char* outputPath = std::getenv("GMX_PCFF_EXACT_RESPA_IMAGE_SIDECAR_OUT");
    const bool haveInput   = inputPath != nullptr && *inputPath != '\0';
    const bool haveOutput  = outputPath != nullptr && *outputPath != '\0';
    if (haveInput != haveOutput)
    {
        GMX_THROW(InvalidInputError(
                "GMX_PCFF_EXACT_RESPA_IMAGE_SIDECAR_IN and "
                "GMX_PCFF_EXACT_RESPA_IMAGE_SIDECAR_OUT must be set together"));
    }
    if (haveInput)
    {
        inputPath_  = inputPath;
        outputPath_ = outputPath;
        enabled_    = true;
        if (inputPath_ == outputPath_)
        {
            GMX_THROW(InvalidInputError(
                    "Exact r-RESPA image sidecar input and output paths must differ"));
        }
    }
}

ExactRespaImageTracker::ExactRespaImageTracker(std::filesystem::path inputPath,
                                               std::filesystem::path outputPath) :
    inputPath_(std::move(inputPath)), outputPath_(std::move(outputPath))
{
    if (inputPath_.empty() != outputPath_.empty())
    {
        GMX_THROW(InvalidInputError(
                "Exact r-RESPA image sidecar input and output paths must be set together"));
    }
    enabled_ = !inputPath_.empty();
    if (enabled_ && inputPath_ == outputPath_)
    {
        GMX_THROW(InvalidInputError(
                "Exact r-RESPA image sidecar input and output paths must differ"));
    }
}

void ExactRespaImageTracker::ensureInitialized(const int64_t              step,
                                               const matrix               box,
                                               const ArrayRef<const RVec> statePositions)
{
    if (!enabled_)
    {
        return;
    }
    if (initialized_)
    {
        if (statePositions.size() != numAtoms_)
        {
            GMX_THROW(InvalidInputError(
                    "Exact r-RESPA image tracker atom count changed; DD is not supported"));
        }
        if (step < inputStep_)
        {
            GMX_THROW(InvalidInputError("Exact r-RESPA image tracker step moved backwards"));
        }
        // A second EM coordinate buffer must explicitly inherit the counters
        // of the state from which its coordinates were derived. Silently using
        // one process-global vector here corrupts the accepted state whenever
        // a rejected line-search trial crosses PBC.
        imagesForCoordinateBuffer(statePositions);
        return;
    }

    std::error_code error;
    if (std::filesystem::exists(outputPath_, error))
    {
        GMX_THROW(FileIOError(formatString(
                "Refusing to start with an existing exact r-RESPA image sidecar output '%s'",
                outputPath_.string().c_str())));
    }
    if (error)
    {
        GMX_THROW(FileIOError(formatString("Could not inspect exact r-RESPA image sidecar output '%s': %s",
                                           outputPath_.string().c_str(),
                                           error.message().c_str())));
    }

    const ExactRespaImageSidecar sidecar = readExactRespaImageSidecar(inputPath_);
    if (sidecar.step != step)
    {
        GMX_THROW(InvalidInputError(formatString(
                "Exact r-RESPA image sidecar step mismatch: sidecar %lld, mdrun %lld. "
                "Checkpoint continuation requires a sidecar written at the same step.",
                static_cast<long long>(sidecar.step),
                static_cast<long long>(step))));
    }
    if (sidecar.atoms.size() != statePositions.size())
    {
        GMX_THROW(InvalidInputError(formatString(
                "Exact r-RESPA image sidecar atom count mismatch: sidecar %zu, state %zu",
                sidecar.atoms.size(),
                statePositions.size())));
    }
    const ExactRespaImageBox stateBox = doubleBox(box);
    validateSameBox(sidecar.box, stateBox, "Exact r-RESPA image sidecar initialization");

    std::vector<ExactRespaAtomImage> initialImages(statePositions.size());
    for (Index atomIndex = 0; atomIndex < statePositions.ssize(); ++atomIndex)
    {
        const auto recoveredImage = exactRespaIntegerLatticeShift(
                sidecar.atoms[atomIndex].continuousPosition,
                doublePosition(statePositions[atomIndex]),
                stateBox);
        if (!recoveredImage)
        {
            GMX_THROW(InvalidInputError(formatString(
                    "Exact r-RESPA image sidecar position mismatch for atom %td",
                    atomIndex)));
        }
        initialImages[atomIndex] = *recoveredImage;
    }

    numAtoms_ = statePositions.size();
    imagesByCoordinateBuffer_.emplace(statePositions.data(), std::move(initialImages));
    inputStep_   = step;
    initialized_ = true;
}

void ExactRespaImageTracker::putAtomsInBoxAndTrack(const int64_t           step,
                                                   const PbcType           pbcType,
                                                   const matrix            box,
                                                   const bool              haveBoxDeformation,
                                                   const matrix            boxDeformation,
                                                   const ArrayRef<RVec>    statePositions,
                                                   const ArrayRef<RVec>    velocities,
                                                   const int               numThreads)
{
    if (!enabled_)
    {
        put_atoms_in_box_omp(pbcType,
                             box,
                             haveBoxDeformation,
                             boxDeformation,
                             statePositions,
                             velocities,
                             numThreads);
        return;
    }

    ensureInitialized(step, box, statePositions);
    const ExactRespaImageBox currentBox = doubleBox(box);
    std::vector<DVec>        positionsBefore(statePositions.size());
    auto&                    images = imagesForCoordinateBuffer(statePositions);
    for (Index atomIndex = 0; atomIndex < statePositions.ssize(); ++atomIndex)
    {
        positionsBefore[atomIndex] = doublePosition(statePositions[atomIndex]);
    }

    put_atoms_in_box_omp(pbcType,
                         box,
                         haveBoxDeformation,
                         boxDeformation,
                         statePositions,
                         velocities,
                         numThreads);

    for (Index atomIndex = 0; atomIndex < statePositions.ssize(); ++atomIndex)
    {
        const DVec positionAfter = doublePosition(statePositions[atomIndex]);
        const auto appliedShift =
                exactRespaIntegerLatticeShift(positionsBefore[atomIndex], positionAfter, currentBox);
        if (!appliedShift)
        {
            GMX_THROW(InvalidInputError(formatString(
                    "Exact r-RESPA image tracker observed a non-lattice PBC change for atom %td",
                    atomIndex)));
        }

        const ExactRespaAtomImage oldImage = images[atomIndex];
        for (int dimension = 0; dimension < DIM; ++dimension)
        {
            if ((appliedShift->at(dimension) > 0
                 && images[atomIndex][dimension]
                            > std::numeric_limits<int64_t>::max() - appliedShift->at(dimension))
                || (appliedShift->at(dimension) < 0
                    && images[atomIndex][dimension]
                               < std::numeric_limits<int64_t>::min() - appliedShift->at(dimension)))
            {
                GMX_THROW(InvalidInputError("Exact r-RESPA image counter overflow"));
            }
            images[atomIndex][dimension] += appliedShift->at(dimension);
        }

        const DVec continuousBefore =
                exactRespaContinuousPosition(positionsBefore[atomIndex], oldImage, currentBox);
        const DVec continuousAfter =
                exactRespaContinuousPosition(positionAfter, images[atomIndex], currentBox);
        if (!equivalentPosition(continuousBefore, continuousAfter))
        {
            GMX_THROW(InvalidInputError(formatString(
                    "Exact r-RESPA image tracker continuity check failed for atom %td",
                    atomIndex)));
        }
    }
}

void ExactRespaImageTracker::inheritCoordinateBuffer(const ArrayRef<const RVec> source,
                                                     const ArrayRef<const RVec> destination)
{
    if (!enabled_)
    {
        return;
    }
    if (!initialized_)
    {
        GMX_THROW(InvalidInputError(
                "Exact r-RESPA image source buffer has not been initialized"));
    }
    if (source.size() != numAtoms_ || destination.size() != numAtoms_)
    {
        GMX_THROW(InvalidInputError(
                "Exact r-RESPA image buffer inheritance atom count mismatch"));
    }

    // Copy before insert_or_assign, since insertion can rehash the map and
    // invalidate the source iterator. Assignment is intentional even when the
    // destination already has counters from an earlier rejected trial.
    std::vector<ExactRespaAtomImage> inheritedImages = imagesForCoordinateBuffer(source);
    imagesByCoordinateBuffer_.insert_or_assign(destination.data(), std::move(inheritedImages));
}

void ExactRespaImageTracker::maybeWriteFinal(const int64_t              step,
                                             const int64_t              finalStep,
                                             const matrix               box,
                                             const ArrayRef<const RVec> statePositions)
{
    if (!enabled_)
    {
        return;
    }
    ensureInitialized(step, box, statePositions);
    if (finalStep < inputStep_)
    {
        GMX_THROW(InvalidInputError("Exact r-RESPA image sidecar requires a finite forward stage"));
    }
    if (outputWritten_)
    {
        return;
    }
    if (step > finalStep)
    {
        GMX_THROW(InvalidInputError("Exact r-RESPA image tracker passed its declared final step"));
    }
    if (step != finalStep)
    {
        return;
    }

    ExactRespaImageSidecar sidecar;
    sidecar.step = step;
    sidecar.box  = doubleBox(box);
    sidecar.atoms.resize(statePositions.size());
    const auto& images = imagesForCoordinateBuffer(statePositions);
    for (Index atomIndex = 0; atomIndex < statePositions.ssize(); ++atomIndex)
    {
        auto& atom              = sidecar.atoms[atomIndex];
        atom.globalAtomIndex    = atomIndex;
        atom.image              = images[atomIndex];
        atom.statePosition      = doublePosition(statePositions[atomIndex]);
        atom.continuousPosition =
                exactRespaContinuousPosition(atom.statePosition, atom.image, sidecar.box);
    }
    writeExactRespaImageSidecarAtomically(outputPath_, sidecar);
    outputWritten_ = true;
}

std::vector<ExactRespaAtomImage>& ExactRespaImageTracker::imagesForCoordinateBuffer(
        const ArrayRef<const RVec> statePositions)
{
    const auto found = imagesByCoordinateBuffer_.find(statePositions.data());
    if (found == imagesByCoordinateBuffer_.end())
    {
        GMX_THROW(InvalidInputError(
                "Exact r-RESPA image coordinate buffer has no inherited counter state"));
    }
    return found->second;
}

const std::vector<ExactRespaAtomImage>& ExactRespaImageTracker::imagesForCoordinateBuffer(
        const ArrayRef<const RVec> statePositions) const
{
    const auto found = imagesByCoordinateBuffer_.find(statePositions.data());
    if (found == imagesByCoordinateBuffer_.end())
    {
        GMX_THROW(InvalidInputError(
                "Exact r-RESPA image coordinate buffer has no inherited counter state"));
    }
    return found->second;
}

ArrayRef<const ExactRespaAtomImage> ExactRespaImageTracker::imagesForTesting(
        const ArrayRef<const RVec> statePositions) const
{
    return imagesForCoordinateBuffer(statePositions);
}

ExactRespaImageTracker& globalExactRespaImageTracker()
{
    static thread_local ExactRespaImageTracker tracker;
    return tracker;
}

bool exactRespaImageTrackerEnabled()
{
    return globalExactRespaImageTracker().enabled();
}

void ensureExactRespaImageTrackerInitialized(const int64_t              step,
                                             const matrix               box,
                                             const ArrayRef<const RVec> statePositions)
{
    globalExactRespaImageTracker().ensureInitialized(step, box, statePositions);
}

void putAtomsInBoxAndTrackExactRespaImages(const int64_t        step,
                                           const PbcType        pbcType,
                                           const matrix         box,
                                           const bool           haveBoxDeformation,
                                           const matrix         boxDeformation,
                                           const ArrayRef<RVec> statePositions,
                                           const ArrayRef<RVec> velocities,
                                           const int            numThreads)
{
    globalExactRespaImageTracker().putAtomsInBoxAndTrack(step,
                                                         pbcType,
                                                         box,
                                                         haveBoxDeformation,
                                                         boxDeformation,
                                                         statePositions,
                                                         velocities,
                                                         numThreads);
}

void inheritExactRespaImagesForCoordinateBuffer(const ArrayRef<const RVec> source,
                                                const ArrayRef<const RVec> destination)
{
    globalExactRespaImageTracker().inheritCoordinateBuffer(source, destination);
}

void maybeWriteFinalExactRespaImageSidecar(const int64_t              step,
                                           const int64_t              finalStep,
                                           const matrix               box,
                                           const ArrayRef<const RVec> statePositions)
{
    globalExactRespaImageTracker().maybeWriteFinal(step, finalStep, box, statePositions);
}

} // namespace gmx
