/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */
#ifndef GMX_MDLIB_EXACTRESPAIMAGETRACKER_H
#define GMX_MDLIB_EXACTRESPAIMAGETRACKER_H

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include <vector>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/vectypes.h"

enum class PbcType : int;

namespace gmx
{

/*! \brief Integer lattice image for one atom.
 *
 * Coordinates are reconstructed as
 * `stateX + image[0]*box[0] + image[1]*box[1] + image[2]*box[2]`.
 */
using ExactRespaAtomImage = std::array<int64_t, DIM>;

//! Double-precision copy of a simulation box used in image sidecars.
using ExactRespaImageBox = std::array<std::array<double, DIM>, DIM>;

struct ExactRespaImageSidecarAtom
{
    int64_t             globalAtomIndex     = -1;
    ExactRespaAtomImage image               = { 0, 0, 0 };
    DVec                statePosition       = { 0, 0, 0 };
    DVec                 continuousPosition = { 0, 0, 0 };
};

/*! \brief Exact image-tracking sidecar for one state endpoint.
 *
 * The text representation is deliberately versioned and stores both the
 * ordinary simulation representation and the continuous representation. This
 * lets a later mdrun validate atom count/order, box, and every image counter
 * before accepting the sidecar.
 */
struct ExactRespaImageSidecar
{
    int64_t                                step = 0;
    ExactRespaImageBox                     box  = {};
    std::vector<ExactRespaImageSidecarAtom> atoms;
};

//! Return the continuous coordinate represented by \p statePosition and \p image.
DVec exactRespaContinuousPosition(const DVec&                  statePosition,
                                  const ExactRespaAtomImage&   image,
                                  const ExactRespaImageBox&    box);

/*! \brief Recover an exact integer lattice shift between two equivalent positions.
 *
 * Returns an empty optional when the difference is not an integer lattice
 * vector within a precision-scaled tolerance. This is a representation check,
 * not a nearest-image displacement estimate.
 */
std::optional<ExactRespaAtomImage> exactRespaIntegerLatticeShift(
        const DVec& fromPosition, const DVec& toPosition, const ExactRespaImageBox& box);

ExactRespaImageSidecar readExactRespaImageSidecar(const std::filesystem::path& path);

//! Write \p sidecar through a temporary file followed by an atomic rename.
void writeExactRespaImageSidecarAtomically(const std::filesystem::path& path,
                                           const ExactRespaImageSidecar& sidecar);

/*! \brief Exact, no-DD per-atom image tracker used by the PCFF r-RESPA path.
 *
 * This tracker never modifies the simulation representation beyond calling the
 * normal put_atoms_in_box_omp() routine. It observes that routine's exact
 * pre/post lattice displacement and accumulates an integer image for each atom.
 * Global atom order must therefore remain fixed; callers must reject DD.
 */
class ExactRespaImageTracker
{
public:
    ExactRespaImageTracker();
    ExactRespaImageTracker(std::filesystem::path inputPath,
                           std::filesystem::path outputPath);

    bool enabled() const { return enabled_; }

    /*! \brief Initialize or validate tracker state at \p step.
     *
     * On first use, the sidecar step, atom count/order, box, and continuous
     * positions are validated against \p statePositions. A step mismatch
     * deliberately rejects checkpoint continuation without a matching sidecar.
     */
    void ensureInitialized(int64_t                 step,
                           const matrix            box,
                           ArrayRef<const RVec>    statePositions);

    /*! \brief Apply the normal PBC wrap and record its exact lattice shifts. */
    void putAtomsInBoxAndTrack(int64_t              step,
                               PbcType              pbcType,
                               const matrix         box,
                               bool                 haveBoxDeformation,
                               const matrix         boxDeformation,
                               ArrayRef<RVec>       statePositions,
                               ArrayRef<RVec>       velocities,
                               int                  numThreads);

    /*! \brief Copy image state when \p destination is derived from \p source.
     *
     * Coordinate trial buffers in energy minimization are reused after both
     * accepted and rejected trials. The destination state is deliberately
     * overwritten on every call so a rejected trial cannot leak image
     * crossings into the next trial derived from the accepted state.
     */
    void inheritCoordinateBuffer(ArrayRef<const RVec> source,
                                 ArrayRef<const RVec> destination);

    /*! \brief Write the final sidecar exactly once at the finite stage endpoint. */
    void maybeWriteFinal(int64_t                 step,
                         int64_t                 finalStep,
                         const matrix            box,
                         ArrayRef<const RVec>    statePositions);

    ArrayRef<const ExactRespaAtomImage> imagesForTesting(
            ArrayRef<const RVec> statePositions) const;

private:
    using CoordinateBufferIdentity = const RVec*;

    std::vector<ExactRespaAtomImage>& imagesForCoordinateBuffer(
            ArrayRef<const RVec> statePositions);
    const std::vector<ExactRespaAtomImage>& imagesForCoordinateBuffer(
            ArrayRef<const RVec> statePositions) const;

    std::filesystem::path inputPath_;
    std::filesystem::path outputPath_;
    bool                  enabled_       = false;
    bool                  initialized_   = false;
    bool                  outputWritten_ = false;
    int64_t               inputStep_      = 0;
    size_t                numAtoms_       = 0;
    std::unordered_map<CoordinateBufferIdentity, std::vector<ExactRespaAtomImage>>
            imagesByCoordinateBuffer_;
};

//! Process-local tracker configured by GMX_PCFF_EXACT_RESPA_IMAGE_SIDECAR_{IN,OUT}.
ExactRespaImageTracker& globalExactRespaImageTracker();

bool exactRespaImageTrackerEnabled();

void ensureExactRespaImageTrackerInitialized(int64_t              step,
                                             const matrix         box,
                                             ArrayRef<const RVec> statePositions);

void putAtomsInBoxAndTrackExactRespaImages(int64_t        step,
                                           PbcType        pbcType,
                                           const matrix   box,
                                           bool           haveBoxDeformation,
                                           const matrix   boxDeformation,
                                           ArrayRef<RVec> statePositions,
                                           ArrayRef<RVec> velocities,
                                           int            numThreads);

void inheritExactRespaImagesForCoordinateBuffer(ArrayRef<const RVec> source,
                                                ArrayRef<const RVec> destination);

void maybeWriteFinalExactRespaImageSidecar(int64_t              step,
                                           int64_t              finalStep,
                                           const matrix         box,
                                           ArrayRef<const RVec> statePositions);

} // namespace gmx

#endif
