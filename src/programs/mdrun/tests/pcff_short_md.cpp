/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
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
 */

/*! \internal \file
 * \brief
 * Short-MD CPU parity tests for frozen PCFF/Class2 M5 fixtures.
 */

#include "gmxpre.h"

#include <cstdlib>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/gpu_utils/capabilities.h"
#include "gromacs/hardware/device_management.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/math/units.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/trajectory/energyframe.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/message_string_collector.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textwriter.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/testfilemanager.h"
#include "testutils/trajectoryreader.h"

#include "energyreader.h"
#include "moduletest.h"

namespace gmx
{
namespace test
{
namespace
{

constexpr double c_kjToKcal = 1.0 / 4.184;
constexpr double c_barToAtm = 0.9869232667160128;
constexpr double c_m5FourierSpacingNm = 0.08;
constexpr int    c_respaEnergyInterval = 4;
constexpr double c_lammpsNvtTargetTemperatureK = 300.0;
constexpr double c_lammpsNvtTdampPs            = 0.05;
constexpr int    c_lammpsNvtTchain             = 3;
/* LAMMPS fix nvt uses a temperature compute that removes three translational
 * degrees of freedom by default. For the short 20-step M5 fixtures we want the
 * same thermostat mass, but we do not want actual COM velocity removal to perturb
 * the trajectory. Using Linear COM removal with nstcomm far beyond the fixture
 * length gives GROMACS the same 3N-3 thermostat DOF while keeping the run free of
 * explicit COM momentum zeroing.
 */
constexpr int    c_lammpsStyleNvtNstcomm       = 1000;

int getenvIntOrDefault(const char* name, const int defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return defaultValue;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max())
    {
        return defaultValue;
    }
    return static_cast<int>(parsed);
}

int respaOuterSteps()
{
    return getenvIntOrDefault("GMX_PCFF_RESPA_OUTER_STEPS", 5);
}

int respaPair14Level()
{
    return getenvIntOrDefault("GMX_PCFF_RESPA_PAIR14_LEVEL", 1);
}

double gromacsTauTFromLammpsTdamp(const double tdampPs)
{
    /* LAMMPS fix nvt uses Tdamp as an approximate relaxation time, whereas GROMACS
     * Nose-Hoover tau-t sets the equilibrium temperature-oscillation period.
     * Matching the thermostat mass therefore requires tau-t = 2*pi*Tdamp.
     *
     * Basis:
     * - docs.lammps.org fix nvt: Tdamp is a relaxation time in time units.
     * - manual.gromacs.org mdp options: tau-t controls the oscillation period.
     * - src/gromacs/modularsimulator/nosehooverchains.cpp: Q ~ (tau-t / 2*pi)^2.
     */
    return 2.0 * M_PI * tdampPs;
}

int readNumAtomsFromGro(const std::filesystem::path& groPath)
{
    std::ifstream input(groPath);
    GMX_RELEASE_ASSERT(input.is_open(), "Could not open GRO file to read atom count");

    std::string line;
    std::getline(input, line); // title
    std::getline(input, line); // atom count

    std::istringstream stream(line);
    int                numAtoms = 0;
    stream >> numAtoms;
    GMX_RELEASE_ASSERT(numAtoms > 1, "LAMMPS-style temperature requires at least two atoms");
    return numAtoms;
}

double lammpsStyleTemperatureFromKineticEnergy(const double kineticEnergyKjPerMol, const int numAtoms)
{
    /* LAMMPS thermo_style temp uses the default compute temp, which subtracts
     * three translational degrees of freedom from a periodic unconstrained system.
     * Reconstruct temperature from kinetic energy with that same DOF convention.
     */
    const int degreesOfFreedom = DIM * numAtoms - DIM;
    GMX_RELEASE_ASSERT(degreesOfFreedom > 0, "LAMMPS-style temperature requires positive degrees of freedom");
    return 2.0 * kineticEnergyKjPerMol / (degreesOfFreedom * gmx::c_boltz);
}

struct ReferenceContract
{
    std::string              systemId;
    std::map<std::string, double> reference;
    std::map<std::string, double> tolerance;
};

struct MetricComparison
{
    std::string name;
    std::string category;
    double      actual = 0;
    double      reference = 0;
    double      tolerance = 0;
    bool        pass = false;
};

struct StructuralMetrics
{
    double                polymerEndToEndNm = 0;
    double                polymerRgNm = 0;
    std::optional<double> ionDistanceNm;
};

std::filesystem::path repoRoot()
{
    std::filesystem::path root = TestFileManager::getInputDataDirectory();
    for (int i = 0; i < 4; ++i)
    {
        root = root.parent_path();
    }
    return root;
}

std::filesystem::path referenceRoot(const std::string& systemId)
{
    return repoRoot() / "tests" / "reference_results" / "m5" / systemId;
}

std::filesystem::path m4ReferenceRoot(const std::string& systemId)
{
    return repoRoot() / "tests" / "reference_results" / "m4" / systemId;
}

std::filesystem::path referenceResultPath(const std::string& milestone, const std::string& systemId, const std::string& fileName)
{
    return repoRoot() / "tests" / "reference_results" / milestone / systemId / fileName;
}

ReferenceContract loadReferenceContract(const std::filesystem::path& summaryPath)
{
    ReferenceContract contract;

    std::ifstream input(summaryPath);
    GMX_RELEASE_ASSERT(input.is_open(), "Could not open M5 reference summary");

    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream stream(line);
        std::string        kind;
        stream >> kind;
        if (kind == "system")
        {
            stream >> contract.systemId;
            continue;
        }

        std::string section;
        std::string key;
        double      value = 0;
        stream >> section >> key >> value;
        const std::string compositeKey = section + ":" + key;
        if (kind == "reference")
        {
            contract.reference[compositeKey] = value;
        }
        else if (kind == "tolerance")
        {
            contract.tolerance[compositeKey] = value;
        }
    }
    return contract;
}

ReferenceContract loadReferenceContract(const std::string& systemId)
{
    return loadReferenceContract(referenceRoot(systemId) / "reference_summary.tsv");
}

ReferenceContract loadRespaReferenceContract(const std::string& systemId)
{
    return loadReferenceContract(referenceResultPath("m6_respa", systemId, "reference_summary.tsv"));
}

std::map<std::string, double> loadReferenceFields(const std::filesystem::path& path)
{
    std::ifstream input(path);
    GMX_RELEASE_ASSERT(input.is_open(), "Could not open reference JSON fields file");

    std::map<std::string, double> fields;
    std::string                   line;
    bool                          inFields = false;
    while (std::getline(input, line))
    {
        if (!inFields)
        {
            inFields = (line.find("\"fields\"") != std::string::npos);
            continue;
        }
        if (line.find('}') != std::string::npos)
        {
            break;
        }

        const auto firstQuote = line.find('"');
        if (firstQuote == std::string::npos)
        {
            continue;
        }
        const auto secondQuote = line.find('"', firstQuote + 1);
        const auto colon       = line.find(':', secondQuote + 1);
        if (secondQuote == std::string::npos || colon == std::string::npos)
        {
            continue;
        }

        auto valueString = line.substr(colon + 1);
        valueString.erase(std::remove(valueString.begin(), valueString.end(), ','), valueString.end());
        std::istringstream valueStream(valueString);
        double             value = 0;
        valueStream >> value;
        fields.emplace(line.substr(firstQuote + 1, secondQuote - firstQuote - 1), value);
    }

    return fields;
}

std::optional<std::pair<std::string, double>> parseJsonNumericFieldLine(const std::string& line)
{
    const auto firstQuote = line.find('"');
    if (firstQuote == std::string::npos)
    {
        return std::nullopt;
    }
    const auto secondQuote = line.find('"', firstQuote + 1);
    const auto colon       = line.find(':', secondQuote + 1);
    if (secondQuote == std::string::npos || colon == std::string::npos)
    {
        return std::nullopt;
    }

    auto valueString = line.substr(colon + 1);
    valueString.erase(std::remove(valueString.begin(), valueString.end(), ','), valueString.end());
    std::istringstream valueStream(valueString);
    double             value = 0;
    if (!(valueStream >> value))
    {
        return std::nullopt;
    }

    return std::make_pair(line.substr(firstQuote + 1, secondQuote - firstQuote - 1), value);
}

std::vector<RVec> loadReferenceForces(const std::filesystem::path& path)
{
    std::ifstream input(path);
    GMX_RELEASE_ASSERT(input.is_open(), "Could not open reference JSON forces file");

    std::vector<RVec> forces;
    std::string       line;
    bool              inAtoms = false;
    bool              inAtom  = false;
    int               atomId  = -1;
    RVec              currentForce = { 0, 0, 0 };
    bool              haveFx       = false;
    bool              haveFy       = false;
    bool              haveFz       = false;

    while (std::getline(input, line))
    {
        if (!inAtoms)
        {
            if (line.find("\"atoms\"") != std::string::npos)
            {
                inAtoms = true;
            }
            continue;
        }

        if (!inAtom)
        {
            if (line.find(']') != std::string::npos)
            {
                break;
            }
            if (line.find('{') != std::string::npos)
            {
                inAtom      = true;
                atomId      = -1;
                currentForce = { 0, 0, 0 };
                haveFx      = false;
                haveFy      = false;
                haveFz      = false;
            }
            continue;
        }

        if (const auto field = parseJsonNumericFieldLine(line))
        {
            const auto& [name, value] = *field;
            if (name == "id")
            {
                atomId = static_cast<int>(value);
            }
            else if (name == "fx")
            {
                currentForce[XX] = value;
                haveFx           = true;
            }
            else if (name == "fy")
            {
                currentForce[YY] = value;
                haveFy           = true;
            }
            else if (name == "fz")
            {
                currentForce[ZZ] = value;
                haveFz           = true;
            }
        }

        if (line.find('}') != std::string::npos)
        {
            GMX_RELEASE_ASSERT(atomId > 0, "Reference force atom object is missing an id field");
            GMX_RELEASE_ASSERT(haveFx && haveFy && haveFz, "Reference force atom object is missing a force component");
            if (gmx::ssize(forces) < atomId)
            {
                forces.resize(atomId);
            }
            forces[atomId - 1] = currentForce;
            inAtom             = false;
        }
    }

    GMX_RELEASE_ASSERT(!forces.empty(), "Reference force JSON did not contain any atoms");
    return forces;
}

double minimumImage(const double delta, const double boxLength)
{
    if (boxLength <= 0)
    {
        return delta;
    }
    return delta - boxLength * std::round(delta / boxLength);
}

RVec minimumImageVector(const RVec& from, const RVec& to, const BoxMatrix& box)
{
    return RVec{ static_cast<real>(minimumImage(to[XX] - from[XX], box[XX][XX])),
                 static_cast<real>(minimumImage(to[YY] - from[YY], box[YY][YY])),
                 static_cast<real>(minimumImage(to[ZZ] - from[ZZ], box[ZZ][ZZ])) };
}

double norm(const RVec& vector)
{
    return std::sqrt(static_cast<double>(vector[XX] * vector[XX] + vector[YY] * vector[YY] + vector[ZZ] * vector[ZZ]));
}

std::vector<RVec> unwrapPolymerCoordinates(ArrayRef<const RVec> coordinates, const BoxMatrix& box, const int numPolymerAtoms)
{
    std::vector<RVec> result;
    result.reserve(numPolymerAtoms);
    result.push_back(coordinates[0]);
    for (int atom = 1; atom < numPolymerAtoms; ++atom)
    {
        const RVec delta = minimumImageVector(coordinates[atom - 1], coordinates[atom], box);
        result.push_back(
                RVec{ result.back()[XX] + delta[XX], result.back()[YY] + delta[YY], result.back()[ZZ] + delta[ZZ] });
    }
    return result;
}

StructuralMetrics computeStructuralMetrics(const std::string& systemId, const TrajectoryFrame& frame)
{
    const auto coordinates = frame.x();
    GMX_RELEASE_ASSERT(frame.hasBox(), "M5 trajectories require periodic box information");

    const int   numPolymerAtoms = (systemId == "small_oligomer") ? 6 : 8;
    const auto  unwrapped = unwrapPolymerCoordinates(coordinates, frame.box(), numPolymerAtoms);
    const RVec  endToEnd = minimumImageVector(coordinates[0], coordinates[numPolymerAtoms - 1], frame.box());
    StructuralMetrics metrics;
    metrics.polymerEndToEndNm = norm(endToEnd);

    RVec centerOfMass = { 0, 0, 0 };
    for (const auto& coordinate : unwrapped)
    {
        centerOfMass[XX] += coordinate[XX];
        centerOfMass[YY] += coordinate[YY];
        centerOfMass[ZZ] += coordinate[ZZ];
    }
    centerOfMass[XX] /= numPolymerAtoms;
    centerOfMass[YY] /= numPolymerAtoms;
    centerOfMass[ZZ] /= numPolymerAtoms;

    double sumSquared = 0;
    for (const auto& coordinate : unwrapped)
    {
        const double dx = coordinate[XX] - centerOfMass[XX];
        const double dy = coordinate[YY] - centerOfMass[YY];
        const double dz = coordinate[ZZ] - centerOfMass[ZZ];
        sumSquared += dx * dx + dy * dy + dz * dz;
    }
    metrics.polymerRgNm = std::sqrt(sumSquared / numPolymerAtoms);

    if (systemId == "small_salt_polymer_box")
    {
        const RVec ionVector = minimumImageVector(coordinates[8], coordinates[9], frame.box());
        metrics.ionDistanceNm = norm(ionVector);
    }

    return metrics;
}

std::vector<EnergyFrame> readEnergyFrames(const std::string& filename, const std::vector<std::string>& terms)
{
    auto reader = openEnergyFileToReadTerms(filename, terms);
    std::vector<EnergyFrame> frames;
    while (reader->readNextFrame())
    {
        frames.emplace_back(reader->frame());
    }
    return frames;
}

void appendRespaObservableTraceLine(const char* fileName, const std::string& line)
{
    const char* traceDir = std::getenv("GMX_PCFF_RESPA_M2P_TRACE_DIR");
    if (traceDir == nullptr || *traceDir == '\0')
    {
        return;
    }

    std::filesystem::create_directories(traceDir);
    const auto outputPath = std::filesystem::path(traceDir) / fileName;
    std::ofstream output(outputPath, std::ios::app);
    GMX_RELEASE_ASSERT(output.is_open(), "Could not open PCFF observable trace file");
    output << line << "\n";
}

void appendStep0EdrFrameVirialVolumeObservableTrace(const std::string& runKind,
                                                    const EnergyFrame&  frame,
                                                    const BoxMatrix&    box)
{
    const char* caseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2P_CASE_LABEL");
    const std::string caseLabel =
            (caseLabelEnv != nullptr && *caseLabelEnv != '\0') ? caseLabelEnv : "unknown";
    const double volumeNm3 = static_cast<double>(box[XX][XX] * box[YY][YY] * box[ZZ][ZZ]);
    appendRespaObservableTraceLine(
            "step0_edr_frame_virial_volume_observable_trace.txt",
            "stage=EDR_FRAME_OBSERVABLE_VIRIAL_VOLUME code_location=src/programs/mdrun/tests/pcff_short_md.cpp:appendStep0EdrFrameVirialVolumeObservableTrace"
                    + std::string(" case_label=") + caseLabel
                    + " execution_path=readEnergyFrames_front_frame run_kind=" + runKind + " step=0"
                    + " volume_nm3=" + formatString("%.15f", volumeNm3)
                    + " vir_xx=" + formatString("%.15f", frame.at("Vir-XX"))
                    + " vir_xy=" + formatString("%.15f", frame.at("Vir-XY"))
                    + " vir_xz=" + formatString("%.15f", frame.at("Vir-XZ"))
                    + " vir_yx=" + formatString("%.15f", frame.at("Vir-YX"))
                    + " vir_yy=" + formatString("%.15f", frame.at("Vir-YY"))
                    + " vir_yz=" + formatString("%.15f", frame.at("Vir-YZ"))
                    + " vir_zx=" + formatString("%.15f", frame.at("Vir-ZX"))
                    + " vir_zy=" + formatString("%.15f", frame.at("Vir-ZY"))
                    + " vir_zz=" + formatString("%.15f", frame.at("Vir-ZZ")));
}

void appendStep0DerivedVirialPressureObservableTrace(const std::string&                    runKind,
                                                     const std::map<std::string, double>& pressureRows)
{
    const char* caseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2P_CASE_LABEL");
    const std::string caseLabel =
            (caseLabelEnv != nullptr && *caseLabelEnv != '\0') ? caseLabelEnv : "unknown";
    appendRespaObservableTraceLine(
            "step0_derived_virial_pressure_observable_trace.txt",
            "stage=DERIVED_VIRIAL_PRESSURE_OBSERVABLE code_location=src/programs/mdrun/tests/pcff_short_md.cpp:step0VirialPressureTensorAtm"
                    + std::string(" case_label=") + caseLabel
                    + " execution_path=step0VirialPressureTensorAtm run_kind=" + runKind + " step=0"
                    + " step0_virial_pressure_xx_atm="
                    + formatString("%.15f", pressureRows.at("step0_virial_pressure_xx_atm"))
                    + " step0_virial_pressure_yy_atm="
                    + formatString("%.15f", pressureRows.at("step0_virial_pressure_yy_atm"))
                    + " step0_virial_pressure_zz_atm="
                    + formatString("%.15f", pressureRows.at("step0_virial_pressure_zz_atm"))
                    + " step0_virial_pressure_xy_atm="
                    + formatString("%.15f", pressureRows.at("step0_virial_pressure_xy_atm"))
                    + " step0_virial_pressure_xz_atm="
                    + formatString("%.15f", pressureRows.at("step0_virial_pressure_xz_atm"))
                    + " step0_virial_pressure_yz_atm="
                    + formatString("%.15f", pressureRows.at("step0_virial_pressure_yz_atm")));
}

void appendStep0ScalarEnergyObservableTrace(const std::string& runKind, const EnergyFrame& frame)
{
    const char* caseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2P_CASE_LABEL");
    const std::string caseLabel =
            (caseLabelEnv != nullptr && *caseLabelEnv != '\0') ? caseLabelEnv : "unknown";
    const double potentialKcalMol =
            static_cast<double>(frame.at(interaction_function[InteractionFunction::PotentialEnergy].longname))
            * c_kjToKcal;
    const double initialTotalKcalMol =
            static_cast<double>(frame.at(interaction_function[InteractionFunction::TotalEnergy].longname))
            * c_kjToKcal;
    const double kineticGapKcalMol = initialTotalKcalMol - potentialKcalMol;
    appendRespaObservableTraceLine(
            "step0_scalar_energy_observable_trace.txt",
            "stage=SCALAR_ENERGY_OBSERVABLE code_location=src/programs/mdrun/tests/pcff_short_md.cpp:appendStep0ScalarEnergyObservableTrace"
                    + std::string(" case_label=") + caseLabel
                    + " execution_path=readEnergyFrames_front_frame run_kind=" + runKind + " step=0"
                    + " step0_potential_kcal_mol=" + formatString("%.15f", potentialKcalMol)
                    + " initial_total_kcal_mol=" + formatString("%.15f", initialTotalKcalMol)
                    + " kinetic_gap_kcal_mol=" + formatString("%.15f", kineticGapKcalMol));
}

void appendScalarTotalObservableTrace(const std::string& runKind,
                                      const EnergyFrame& firstFrame,
                                      const EnergyFrame& lastFrame,
                                      const double       totalEnergySpanKcalMol)
{
    const char* caseLabelEnv = std::getenv("GMX_PCFF_RESPA_M2P_CASE_LABEL");
    const std::string caseLabel =
            (caseLabelEnv != nullptr && *caseLabelEnv != '\0') ? caseLabelEnv : "unknown";
    const double initialTotalKcalMol =
            static_cast<double>(firstFrame.at(interaction_function[InteractionFunction::TotalEnergy].longname))
            * c_kjToKcal;
    const double finalTotalKcalMol =
            static_cast<double>(lastFrame.at(interaction_function[InteractionFunction::TotalEnergy].longname))
            * c_kjToKcal;
    const double totalEnergyDriftAbsKcalMol = std::abs(finalTotalKcalMol - initialTotalKcalMol);
    appendRespaObservableTraceLine(
            "scalar_total_observable_trace.txt",
            "stage=SCALAR_TOTAL_OBSERVABLE code_location=src/programs/mdrun/tests/pcff_short_md.cpp:appendScalarTotalObservableTrace"
                    + std::string(" case_label=") + caseLabel
                    + " execution_path=readEnergyFrames_front_back_frame run_kind=" + runKind
                    + " initial_total_kcal_mol=" + formatString("%.15f", initialTotalKcalMol)
                    + " final_total_kcal_mol=" + formatString("%.15f", finalTotalKcalMol)
                    + " total_energy_drift_abs_kcal_mol="
                    + formatString("%.15f", totalEnergyDriftAbsKcalMol)
                    + " total_energy_span_kcal_mol=" + formatString("%.15f", totalEnergySpanKcalMol));
}

StructuralMetrics readLastStructuralMetrics(const std::string& systemId, const std::string& filename)
{
    TrajectoryFrameReader reader(filename);
    StructuralMetrics     lastMetrics = computeStructuralMetrics(systemId, reader.frame());
    while (reader.readNextFrame())
    {
        lastMetrics = computeStructuralMetrics(systemId, reader.frame());
    }
    return lastMetrics;
}

double kjToKcal(const double value)
{
    return value * c_kjToKcal;
}

double kjNmToKcalA(const double value)
{
    return value * c_kjToKcal / 10.0;
}

double barToAtm(const double value)
{
    return value * c_barToAtm;
}

double boxVolumeNm3(const BoxMatrix& box)
{
    return static_cast<double>(box[XX][XX] * box[YY][YY] * box[ZZ][ZZ]);
}

double virialEnergyToVirialPressureAtm(const double virialKjPerMol, const double volumeNm3)
{
    GMX_RELEASE_ASSERT(volumeNm3 > 0, "Virial-pressure comparison requires a positive box volume");
    return barToAtm((-virialKjPerMol) * (2.0 * gmx::c_presfac) / volumeNm3);
}

std::map<std::string, double> step0VirialPressureTensorAtm(const EnergyFrame& frame, const BoxMatrix& box)
{
    const double volumeNm3 = boxVolumeNm3(box);

    const double pxx = virialEnergyToVirialPressureAtm(frame.at("Vir-XX"), volumeNm3);
    const double pyy = virialEnergyToVirialPressureAtm(frame.at("Vir-YY"), volumeNm3);
    const double pzz = virialEnergyToVirialPressureAtm(frame.at("Vir-ZZ"), volumeNm3);
    const double pxy = 0.5 * (virialEnergyToVirialPressureAtm(frame.at("Vir-XY"), volumeNm3)
                              + virialEnergyToVirialPressureAtm(frame.at("Vir-YX"), volumeNm3));
    const double pxz = 0.5 * (virialEnergyToVirialPressureAtm(frame.at("Vir-XZ"), volumeNm3)
                              + virialEnergyToVirialPressureAtm(frame.at("Vir-ZX"), volumeNm3));
    const double pyz = 0.5 * (virialEnergyToVirialPressureAtm(frame.at("Vir-YZ"), volumeNm3)
                              + virialEnergyToVirialPressureAtm(frame.at("Vir-ZY"), volumeNm3));

    return {
        { "step0_virial_pressure_xx_atm", pxx },
        { "step0_virial_pressure_yy_atm", pyy },
        { "step0_virial_pressure_zz_atm", pzz },
        { "step0_virial_pressure_xy_atm", pxy },
        { "step0_virial_pressure_xz_atm", pxz },
        { "step0_virial_pressure_yz_atm", pyz },
    };
}

void assignRunnerOutputs(SimulationRunner* runner, TestFileManager* fileManager, const std::string& stem)
{
    runner->logFileName_                     = fileManager->getTemporaryFilePath(stem + ".log").string();
    runner->edrFileName_                     = fileManager->getTemporaryFilePath(stem + ".edr").string();
    runner->mtxFileName_                     = fileManager->getTemporaryFilePath(stem + ".mtx").string();
    runner->fullPrecisionTrajectoryFileName_ = fileManager->getTemporaryFilePath(stem + ".trr").string();
    runner->reducedPrecisionTrajectoryFileName_ = fileManager->getTemporaryFilePath(stem + ".xtc").string();
    runner->groOutputFileName_               = fileManager->getTemporaryFilePath(stem + ".gro").string();
    runner->cptOutputFileName_               = fileManager->getTemporaryFilePath(stem + ".cpt").string();
    runner->mdpOutputFileName_               = fileManager->getTemporaryFilePath(stem + ".mdout.mdp").string();
}

struct FinalTrajectorySnapshot
{
    int64_t           step = -1;
    double            time = 0;
    BoxMatrix         box = {};
    std::vector<RVec> coordinates;
    std::vector<RVec> velocities;
};

FinalTrajectorySnapshot readFinalTrajectorySnapshot(const std::string& trajectoryFileName)
{
    TrajectoryFrameReader  reader(trajectoryFileName);
    FinalTrajectorySnapshot snapshot;

    auto copyFrame = [&snapshot](const TrajectoryFrame& frame)
    {
        snapshot.step = frame.step();
        snapshot.time = frame.time();
        if (frame.hasBox())
        {
            snapshot.box = frame.box();
        }
        snapshot.coordinates.assign(frame.x().begin(), frame.x().end());
        snapshot.velocities.assign(frame.v().begin(), frame.v().end());
    };

    copyFrame(reader.frame());
    while (reader.readNextFrame())
    {
        copyFrame(reader.frame());
    }

    GMX_RELEASE_ASSERT(!snapshot.coordinates.empty(), "Expected at least one trajectory frame with coordinates");
    GMX_RELEASE_ASSERT(!snapshot.velocities.empty(), "Expected at least one trajectory frame with velocities");
    return snapshot;
}

std::string metricCategory(const std::string& section, const std::string& key)
{
    if (key.find("step0_") == 0 || key.find("initial_") == 0)
    {
        return "physics";
    }
    if (section == "nve" || section == "nvt")
    {
        return "numerics";
    }
    return "physics";
}

std::string makeSinglePointMdp()
{
    std::ostringstream mdp;
    mdp << "title                   = pcff single point parity\n"
        << "integrator              = md-vv\n"
        << "dt                      = 0.0005\n"
        << "nsteps                  = 0\n"
        << "continuation            = yes\n"
        << "constraints             = none\n"
        << "cutoff-scheme           = Verlet\n"
        << "nstlist                 = 1\n"
        << "rlist                   = 0.9\n"
        << "rvdw                    = 0.9\n"
        << "rcoulomb                = 0.9\n"
        << "vdwtype                 = Cut-off\n"
        << "coulombtype             = PME\n"
        << "vdw-modifier            = none\n"
        << "coulomb-modifier        = none\n"
        << "ewald-rtol              = 1e-6\n"
        << "pme-order               = 4\n"
        << "fourierspacing          = " << c_m5FourierSpacingNm << "\n"
        << "epsilon-r               = 1\n"
        << "pbc                     = xyz\n"
        << "tcoupl                  = no\n"
        << "pcoupl                  = no\n"
        << "comm-mode               = none\n"
        << "verlet-buffer-tolerance = -1\n"
        << "gen-vel                 = no\n"
        << "nstcalcenergy           = 1\n"
        << "nstenergy               = 1\n"
        << "nstlog                  = 1\n"
        << "nstxout                 = 1\n"
        << "nstvout                 = 1\n"
        << "nstfout                 = 1\n"
        << "nstxout-compressed      = 0\n";
    return mdp.str();
}

std::string makeMdp(const std::string& ensemble)
{
    std::ostringstream mdp;
    mdp << "title                   = pcff short md parity\n"
        << "integrator              = md-vv\n"
        << "dt                      = 0.0005\n"
        << "nsteps                  = 20\n"
        << "constraints             = none\n"
        << "cutoff-scheme           = Verlet\n"
        << "nstlist                 = 1\n"
        << "rlist                   = 0.9\n"
        << "rvdw                    = 0.9\n"
        << "rcoulomb                = 0.9\n"
        << "vdwtype                 = Cut-off\n"
        << "coulombtype             = PME\n"
        << "ewald-rtol              = 1e-6\n"
        << "pme-order               = 4\n"
        << "fourierspacing          = " << c_m5FourierSpacingNm << "\n"
        << "epsilon-r               = 1\n"
        << "pbc                     = xyz\n"
        << "tcoupl                  = " << (ensemble == "nvt" ? "nose-hoover" : "no") << "\n"
        << "pcoupl                  = no\n"
        << "comm-mode               = " << (ensemble == "nvt" ? "Linear" : "none") << "\n"
        << "nstcomm                 = " << (ensemble == "nvt" ? c_lammpsStyleNvtNstcomm : 0) << "\n"
        << "verlet-buffer-tolerance = -1\n"
        << "gen-vel                 = no\n"
        << "nstcalcenergy           = 1\n"
        << "nstenergy               = 1\n"
        << "nstlog                  = 1\n"
        << "nstxout                 = 1\n"
        << "nstvout                 = 1\n"
        << "nstfout                 = 1\n"
        << "nstxout-compressed      = 0\n";
    if (ensemble == "nvt")
    {
        mdp << "tc-grps                 = System\n"
            << "tau-t                   = " << gromacsTauTFromLammpsTdamp(c_lammpsNvtTdampPs) << "\n"
            << "ref-t                   = " << c_lammpsNvtTargetTemperatureK << "\n"
            << "nsttcouple              = 1\n"
            << "nh-chain-length         = " << c_lammpsNvtTchain << "\n";
    }
    return mdp.str();
}

std::string makeRespaNveMdp()
{
    const int outerSteps = respaOuterSteps();
    const int pair14Level = respaPair14Level();
    std::ostringstream mdp;
    mdp << "title                   = pcff exact respa parity\n"
        << "integrator              = md-vv\n"
        << "dt                      = 0.0005\n"
        << "nsteps                  = " << (outerSteps * c_respaEnergyInterval) << "\n"
        << "constraints             = none\n"
        << "cutoff-scheme           = Verlet\n"
        << "nstlist                 = " << c_respaEnergyInterval << "\n"
        << "rlist                   = 0.99\n"
        << "rvdw                    = 0.9\n"
        << "rcoulomb                = 0.9\n"
        << "vdwtype                 = Cut-off\n"
        << "vdw-modifier            = none\n"
        << "coulombtype             = PME\n"
        << "coulomb-modifier        = none\n"
        << "ewald-rtol              = 1e-6\n"
        << "pme-order               = 4\n"
        << "fourierspacing          = " << c_m5FourierSpacingNm << "\n"
        << "epsilon-r               = 1\n"
        << "pbc                     = xyz\n"
        << "tcoupl                  = no\n"
        << "pcoupl                  = no\n"
        << "comm-mode               = none\n"
        << "verlet-buffer-tolerance = -1\n"
        << "gen-vel                 = no\n"
        << "exact-respa             = yes\n"
        << "exact-respa-levels      = 3\n"
        << "exact-respa-level2-factor = 2\n"
        << "exact-respa-level3-factor = 4\n"
        << "exact-respa-bond-level  = 1\n"
        << "exact-respa-angle-level = 1\n"
        << "exact-respa-dihedral-level = 1\n"
        << "exact-respa-improper-level = 1\n"
        << "exact-respa-pair14-level = " << pair14Level << "\n"
        << "exact-respa-pair-level  = 3\n"
        << "exact-respa-kspace-level = 3\n"
        << "exact-respa-inner-level = 1\n"
        << "exact-respa-middle-level = 2\n"
        << "exact-respa-outer-level = 3\n"
        << "exact-respa-inner-off   = 0.30\n"
        << "exact-respa-inner-on    = 0.45\n"
        << "exact-respa-outer-on    = 0.60\n"
        << "exact-respa-outer-off   = 0.80\n"
        << "nstcalcenergy           = " << c_respaEnergyInterval << "\n"
        << "nstenergy               = " << c_respaEnergyInterval << "\n"
        << "nstlog                  = " << c_respaEnergyInterval << "\n"
        << "nstxout                 = " << c_respaEnergyInterval << "\n"
        << "nstvout                 = " << c_respaEnergyInterval << "\n"
        << "nstfout                 = 0\n"
        << "nstxout-compressed      = 0\n";
    return mdp.str();
}

std::vector<MetricComparison> compareMetrics(const ReferenceContract& contract,
                                             const std::string&       section,
                                             const std::map<std::string, double>& actualValues)
{
    std::vector<MetricComparison> comparisons;
    const std::string prefix = section + ":";
    for (const auto& [key, actual] : actualValues)
    {
        const std::string compositeKey = prefix + key;
        const auto referenceIt = contract.reference.find(compositeKey);
        const auto toleranceIt = contract.tolerance.find(compositeKey);
        if (referenceIt == contract.reference.end() || toleranceIt == contract.tolerance.end())
        {
            continue;
        }
        const double reference = referenceIt->second;
        const double tolerance = toleranceIt->second;
        comparisons.push_back(
                MetricComparison{ compositeKey, metricCategory(section, key), actual, reference, tolerance, std::abs(actual - reference) <= tolerance });
    }
    return comparisons;
}

std::vector<std::string> sortedUniqueStrings(const std::vector<std::string>& values)
{
    std::vector<std::string> result = values;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::string> observedFailureCategories(const std::vector<MetricComparison>& comparisons)
{
    std::vector<std::string> categories;
    for (const auto& comparison : comparisons)
    {
        if (!comparison.pass)
        {
            categories.push_back(comparison.category);
        }
    }
    return sortedUniqueStrings(categories);
}

void writeJsonStringArray(std::ostringstream* output, const std::string& name, const std::vector<std::string>& values, const bool trailingComma)
{
    *output << "  \"" << name << "\": [\n";
    for (gmx::Index index = 0; index < gmx::ssize(values); ++index)
    {
        *output << "    \"" << values[index] << "\"" << (index + 1 == gmx::ssize(values) ? "\n" : ",\n");
    }
    *output << "  ]" << (trailingComma ? ",\n" : "\n");
}

void writeCaseSummary(const std::string&                    systemId,
                      const std::string&                    ensemble,
                      const std::vector<MetricComparison>&  comparisons,
                      const std::vector<std::string>&       notes,
                      const std::vector<std::string>&       harnessNotes)
{
    const char* summaryDir = std::getenv("GMX_PCFF_M5_SUMMARY_DIR");
    if (summaryDir == nullptr || std::string(summaryDir).empty())
    {
        return;
    }

    const std::filesystem::path outputPath =
            std::filesystem::path(summaryDir) / formatString("%s_%s.json", systemId.c_str(), ensemble.c_str());
    std::filesystem::create_directories(outputPath.parent_path());

    const std::vector<std::string> supportedCategories = { "physics", "numerics", "harness" };
    const auto                     failedCategories    = observedFailureCategories(comparisons);

    std::ostringstream output;
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"system_id\": \"" << systemId << "\",\n"
           << "  \"ensemble\": \"" << ensemble << "\",\n"
           << "  \"status\": \"" << (std::all_of(comparisons.begin(),
                                                 comparisons.end(),
                                                 [](const MetricComparison& comparison) { return comparison.pass; })
                                                  ? "pass"
                                                  : "fail")
           << "\",\n";
    writeJsonStringArray(&output, "supported_failure_categories", supportedCategories, true);
    writeJsonStringArray(&output, "observed_failure_categories", failedCategories, true);
    writeJsonStringArray(&output, "harness_notes", harnessNotes, true);
    output << "  \"metrics\": [\n";

    for (gmx::Index index = 0; index < gmx::ssize(comparisons); ++index)
    {
        const auto& comparison = comparisons[index];
        output << "    {\n"
               << "      \"name\": \"" << comparison.name << "\",\n"
               << "      \"category\": \"" << comparison.category << "\",\n"
               << "      \"actual\": " << std::setprecision(12) << comparison.actual << ",\n"
               << "      \"reference\": " << comparison.reference << ",\n"
               << "      \"tolerance\": " << comparison.tolerance << ",\n"
               << "      \"pass\": " << (comparison.pass ? "true" : "false") << "\n"
               << "    }" << (index + 1 == gmx::ssize(comparisons) ? "\n" : ",\n");
    }
    output << "  ],\n";
    writeJsonStringArray(&output, "notes", notes, false);
    output << "}\n";

    TextWriter::writeFileFromString(outputPath.string(), output.str());
}

void writeRespaActualSummary(const std::string&               systemId,
                             const std::map<std::string, double>& actualValues,
                             const std::vector<std::string>&  notes,
                             const std::map<std::string, std::map<std::string, double>>& diagnostics = {})
{
    const char* summaryDir = std::getenv("GMX_PCFF_RESPA_ACTUAL_DIR");
    if (summaryDir == nullptr || std::string(summaryDir).empty())
    {
        return;
    }

    const std::filesystem::path outputPath =
            std::filesystem::path(summaryDir) / formatString("%s_nve.json", systemId.c_str());
    std::filesystem::create_directories(outputPath.parent_path());

    std::ostringstream output;
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"system_id\": \"" << systemId << "\",\n"
           << "  \"engine\": \"gromacs\",\n"
           << "  \"mode\": \"lammps-respa\",\n"
           << "  \"schedule\": {\n"
           << "    \"outer_steps\": " << respaOuterSteps() << ",\n"
           << "    \"pair14_level\": " << respaPair14Level() << "\n"
           << "  },\n"
           << "  \"observables\": {\n"
           << "    \"nve\": {\n";

    gmx::Index metricIndex = 0;
    for (const auto& [name, value] : actualValues)
    {
        output << "      \"" << name << "\": " << std::setprecision(12) << value;
        output << (++metricIndex == gmx::ssize(actualValues) ? "\n" : ",\n");
    }

    output << "    }\n"
           << "  }";

    if (!diagnostics.empty())
    {
        output << ",\n"
               << "  \"diagnostics\": {\n";

        gmx::Index sectionIndex = 0;
        for (const auto& [sectionName, values] : diagnostics)
        {
            output << "    \"" << sectionName << "\": {\n";

            gmx::Index valueIndex = 0;
            for (const auto& [name, value] : values)
            {
                output << "      \"" << name << "\": " << std::setprecision(12) << value;
                output << (++valueIndex == gmx::ssize(values) ? "\n" : ",\n");
            }

            output << "    }" << (++sectionIndex == gmx::ssize(diagnostics) ? "\n" : ",\n");
        }
        output << "  },\n";
    }
    else
    {
        output << ",\n";
    }

    output
           << "  \"notes\": [\n";
    for (gmx::Index index = 0; index < gmx::ssize(notes); ++index)
    {
        output << "    \"" << notes[index] << "\"" << (index + 1 == gmx::ssize(notes) ? "\n" : ",\n");
    }
    output << "  ]\n"
           << "}\n";

    TextWriter::writeFileFromString(outputPath.string(), output.str());
}

class PcffShortMdParityTest :
    public MdrunTestFixture,
    public ::testing::WithParamInterface<std::tuple<const char*, const char*>>
{
};

class PcffSinglePointParityTest : public MdrunTestFixture, public ::testing::WithParamInterface<const char*>
{
};

#if GMX_GPU_CUDA
class PcffGpuSinglePointParityTest :
    public MdrunTestFixture,
    public ::testing::WithParamInterface<const char*>
{
};

class PcffGpuPerfSmokeTest : public MdrunTestFixture
{
};

class PcffGpuResidentParityTest :
    public MdrunTestFixture,
    public ::testing::WithParamInterface<const char*>
{
};
#endif

class PcffRespaObservableDumpTest : public MdrunTestFixture, public ::testing::WithParamInterface<const char*>
{
};

class PcffRespaRestartParityTest : public MdrunTestFixture, public ::testing::WithParamInterface<const char*>
{
};

std::map<std::string, double> readStep0EnergyBreakdown(const std::string& energyFileName)
{
    const std::vector<std::string> energyTerms = {
        "Class2 Bond",
        "Class2 Angle",
        "Class2 Dih.",
        "LJ-14",
        "LJ (SR)",
        "Coulomb-14",
        "Coulomb (SR)",
        "Coul. recip.",
        interaction_function[InteractionFunction::PotentialEnergy].longname,
    };
    const auto energyFrames = readEnergyFrames(energyFileName, energyTerms);
    GMX_RELEASE_ASSERT(!energyFrames.empty(), "Expected at least one energy frame");

    const auto& frame = energyFrames.front();
    std::map<std::string, double> result;
    result["bond_kcal_mol"] = kjToKcal(frame.at("Class2 Bond"));
    result["angle_kcal_mol"] = kjToKcal(frame.at("Class2 Angle"));
    result["dihedral_kcal_mol"] = kjToKcal(frame.at("Class2 Dih."));
    result["lj14_kcal_mol"] = kjToKcal(frame.at("LJ-14"));
    result["ljsr_kcal_mol"] = kjToKcal(frame.at("LJ (SR)"));
    result["coul14_kcal_mol"] = kjToKcal(frame.at("Coulomb-14"));
    result["coulsr_kcal_mol"] = kjToKcal(frame.at("Coulomb (SR)"));
    result["coul_recip_kcal_mol"] = kjToKcal(frame.at("Coul. recip."));
    result["vdw_total_kcal_mol"] = result["lj14_kcal_mol"] + result["ljsr_kcal_mol"];
    result["electro_total_kcal_mol"] =
            result["coul14_kcal_mol"] + result["coulsr_kcal_mol"] + result["coul_recip_kcal_mol"];
    result["potential_total_kcal_mol"] =
            kjToKcal(frame.at(interaction_function[InteractionFunction::PotentialEnergy].longname));
    return result;
}

std::map<std::string, double> subtractMaps(const std::map<std::string, double>& left,
                                           const std::map<std::string, double>& right)
{
    std::map<std::string, double> difference;
    for (const auto& [name, leftValue] : left)
    {
        const auto it = right.find(name);
        if (it != right.end())
        {
            difference[name] = leftValue - it->second;
        }
    }
    return difference;
}

TEST_P(PcffShortMdParityTest, MatchesFrozenLammpsObservables)
{
    const auto [systemIdRaw, ensembleRaw] = GetParam();
    const std::string systemId(systemIdRaw);
    const std::string ensemble(ensembleRaw);
    const auto        fixtureRoot = referenceRoot(systemId);

    runner_.topFileName_ = (fixtureRoot / "topol.top").string();
    runner_.groFileName_ = (fixtureRoot / (ensemble == "nve" ? "initial_nve.gro" : "initial_nvt.gro")).string();
    runner_.useStringAsMdpFile(makeMdp(ensemble));
    runner_.setMaxWarn(1);

    ASSERT_EQ(0, runner_.callGrompp()) << "grompp failed for " << systemId << " " << ensemble;
    ASSERT_EQ(0, runner_.callMdrun()) << "mdrun failed for " << systemId << " " << ensemble;

    const std::vector<std::string> energyTerms = {
        interaction_function[InteractionFunction::PotentialEnergy].longname,
        interaction_function[InteractionFunction::TotalEnergy].longname,
        interaction_function[InteractionFunction::KineticEnergy].longname,
        interaction_function[InteractionFunction::Temperature].longname,
        interaction_function[InteractionFunction::Pressure].longname,
    };
    const auto energyFrames = readEnergyFrames(runner_.edrFileName_, energyTerms);
    ASSERT_FALSE(energyFrames.empty());
    const auto& firstFrame = energyFrames.front();
    const auto& lastFrame  = energyFrames.back();

    std::map<std::string, double> actualValues;
    actualValues["step0_potential_kcal_mol"] = kjToKcal(firstFrame.at(interaction_function[InteractionFunction::PotentialEnergy].longname));

    if (ensemble == "nve")
    {
        const auto totalEnergyName = interaction_function[InteractionFunction::TotalEnergy].longname;
        const auto totalEnergyToKcal = [&totalEnergyName](const EnergyFrame& frame) { return kjToKcal(frame.at(totalEnergyName)); };
        double minTotal = totalEnergyToKcal(firstFrame);
        double maxTotal = minTotal;
        for (const auto& frame : energyFrames)
        {
            const double total = totalEnergyToKcal(frame);
            minTotal = std::min(minTotal, total);
            maxTotal = std::max(maxTotal, total);
        }
        actualValues["initial_total_kcal_mol"] = totalEnergyToKcal(firstFrame);
        actualValues["final_total_kcal_mol"] = totalEnergyToKcal(lastFrame);
        actualValues["total_energy_drift_abs_kcal_mol"] = std::abs(actualValues["final_total_kcal_mol"] - actualValues["initial_total_kcal_mol"]);
        actualValues["total_energy_span_kcal_mol"] = maxTotal - minTotal;
    }
    else
    {
        const int numAtoms = readNumAtomsFromGro(runner_.groFileName_);
        actualValues["final_potential_kcal_mol"] = kjToKcal(lastFrame.at(interaction_function[InteractionFunction::PotentialEnergy].longname));
        actualValues["final_total_kcal_mol"] = kjToKcal(lastFrame.at(interaction_function[InteractionFunction::TotalEnergy].longname));
        actualValues["final_temperature_K"] =
                lammpsStyleTemperatureFromKineticEnergy(lastFrame.at(interaction_function[InteractionFunction::KineticEnergy].longname),
                                                       numAtoms);
        actualValues["final_pressure_atm"] = barToAtm(lastFrame.at(interaction_function[InteractionFunction::Pressure].longname));
    }

    const auto structural = readLastStructuralMetrics(systemId, runner_.fullPrecisionTrajectoryFileName_);
    actualValues["polymer_end_to_end_nm"] = structural.polymerEndToEndNm;
    actualValues["polymer_rg_nm"] = structural.polymerRgNm;
    if (structural.ionDistanceNm.has_value())
    {
        actualValues["ion_distance_nm"] = structural.ionDistanceNm.value();
    }

    const auto contract = loadReferenceContract(systemId);
    const auto comparisons = compareMetrics(contract, ensemble, actualValues);
    ASSERT_FALSE(comparisons.empty());

    std::vector<std::string> notes;
    std::vector<std::string> harnessNotes;
    if (ensemble == "nvt")
    {
        harnessNotes.emplace_back("NVT parity is evaluated on final observables, not exact trajectory identity, because Nose-Hoover implementations are not expected to match stepwise across engines.");
        harnessNotes.emplace_back("Final temperature is reconstructed with the default LAMMPS compute temp convention (3 translational degrees of freedom removed) before comparison.");
        notes = harnessNotes;
    }
    writeCaseSummary(systemId, ensemble, comparisons, notes, harnessNotes);

    for (const auto& comparison : comparisons)
    {
        EXPECT_NEAR(comparison.actual, comparison.reference, comparison.tolerance)
                << "Mismatch for " << comparison.name << " (" << comparison.category << ")";
    }
}

TEST_P(PcffSinglePointParityTest, MatchesFrozenLammpsSinglePointBreakdown)
{
    const std::string systemId(GetParam());
    const auto        fixtureRoot = m4ReferenceRoot(systemId);

    runner_.topFileName_ = (fixtureRoot / "topol.top").string();
    runner_.groFileName_ = (fixtureRoot / "initial_nve.gro").string();
    runner_.useStringAsMdpFile(makeSinglePointMdp());
    runner_.setMaxWarn(1);

    ASSERT_EQ(0, runner_.callGrompp()) << "grompp failed for " << systemId;
    ASSERT_EQ(0, runner_.callMdrun()) << "mdrun failed for " << systemId;

    const std::vector<std::string> energyTerms = {
        "Class2 Bond",
        "Class2 Angle",
        "Class2 Dih.",
        "LJ-14",
        "LJ (SR)",
        "Coulomb-14",
        "Coulomb (SR)",
        "Coul. recip.",
        interaction_function[InteractionFunction::PotentialEnergy].longname,
    };
    const auto energyFrames = readEnergyFrames(runner_.edrFileName_, energyTerms);
    ASSERT_EQ(energyFrames.size(), 1);
    const auto& frame = energyFrames.front();

    const auto referenceFields = loadReferenceFields(referenceResultPath("m4", systemId, "single_point.json"));

    const double bondedTolerance = 5e-4;
    const double vdwTolerance    = 2e-2;
    const double electroTolerance = 7e-2;
    const double totalTolerance   = 6e-2;

    const double bondKcal = kjToKcal(frame.at("Class2 Bond"));
    const double angleKcal = kjToKcal(frame.at("Class2 Angle"));
    const double dihedralKcal = kjToKcal(frame.at("Class2 Dih."));
    const double vdwKcal = kjToKcal(frame.at("LJ-14") + frame.at("LJ (SR)"));
    const double electroKcal =
            kjToKcal(frame.at("Coulomb-14") + frame.at("Coulomb (SR)") + frame.at("Coul. recip."));
    const double totalKcal = kjToKcal(frame.at(interaction_function[InteractionFunction::PotentialEnergy].longname));

    EXPECT_NEAR(bondKcal, referenceFields.at("ebond"), bondedTolerance) << systemId << " bond";
    EXPECT_NEAR(angleKcal, referenceFields.at("eangle"), bondedTolerance) << systemId << " angle";
    EXPECT_NEAR(dihedralKcal, referenceFields.at("edihed"), bondedTolerance) << systemId << " dihedral";
    EXPECT_NEAR(vdwKcal, referenceFields.at("evdwl"), vdwTolerance) << systemId << " vdw";
    EXPECT_NEAR(electroKcal, referenceFields.at("ecoul") + referenceFields.at("elong"), electroTolerance)
            << systemId << " electro";
    EXPECT_NEAR(totalKcal, referenceFields.at("pe"), totalTolerance) << systemId << " total";
}

TEST_P(PcffSinglePointParityTest, MatchesFrozenLammpsSinglePointForces)
{
    const std::string systemId(GetParam());
    const auto        fixtureRoot = m4ReferenceRoot(systemId);

    runner_.topFileName_ = (fixtureRoot / "topol.top").string();
    runner_.groFileName_ = (fixtureRoot / "initial_nve.gro").string();
    runner_.useStringAsMdpFile(makeSinglePointMdp());
    runner_.setMaxWarn(1);

    ASSERT_EQ(0, runner_.callGrompp()) << "grompp failed for " << systemId;
    ASSERT_EQ(0, runner_.callMdrun()) << "mdrun failed for " << systemId;

    const auto referenceForces = loadReferenceForces(referenceResultPath("m4", systemId, "forces.json"));

    TrajectoryFrameReader reader(runner_.fullPrecisionTrajectoryFileName_);
    const auto            frame  = reader.frame();
    const auto            forces = frame.f();

    ASSERT_EQ(gmx::ssize(forces), gmx::ssize(referenceForces)) << systemId << " atom count";

    const double componentTolerance = 9e-2;
    double       maxAbsDelta        = 0;

    for (Index atom = 0; atom < ssize(forces); ++atom)
    {
        for (int d = 0; d < DIM; ++d)
        {
            const double actual    = kjNmToKcalA(forces[atom][d]);
            const double reference = referenceForces[atom][d];
            const double delta     = std::abs(actual - reference);
            maxAbsDelta            = std::max(maxAbsDelta, delta);
            EXPECT_NEAR(actual, reference, componentTolerance)
                    << systemId << " atom=" << atom << " dim=" << d;
        }
    }

    EXPECT_LE(maxAbsDelta, componentTolerance) << systemId << " maximum force component delta";
}

#if GMX_GPU_CUDA
MessageStringCollector getGpuSinglePointSkipMessages()
{
    MessageStringCollector messages;
    messages.startContext("Test is being skipped because:");
    messages.appendIf(!GpuConfigurationCapabilities::Nonbonded,
                      "this build does not support nonbonded GPU execution");
    messages.appendIf(!GpuConfigurationCapabilities::Pme, "this build does not support PME GPU execution");
    messages.appendIf(!GpuConfigurationCapabilities::Fft, "this build does not support GPU FFT execution");
    messages.appendIf(getCompatibleDevices(MdrunTestFixtureBase::s_hwinfo->deviceInfoList).empty(),
                      "no compatible GPU devices were detected");
    return messages;
}

CommandLine makeSinglePointMdrunCaller(const bool        useGpuNonbonded,
                                       const bool        useGpuPme,
                                       const std::string& bondedTarget = "cpu",
                                       const std::string& updateTarget = "cpu")
{
    CommandLine caller;
    caller.addOption("-notunepme");
    caller.addOption("-nb", useGpuNonbonded ? "gpu" : "cpu");
    caller.addOption("-pme", useGpuPme ? "gpu" : "cpu");
    if (useGpuPme)
    {
        caller.addOption("-pmefft", "gpu");
    }
    caller.addOption("-bonded", bondedTarget);
    caller.addOption("-update", updateTarget);
    return caller;
}

void setRunnerOutputPrefix(SimulationRunner* runner, TestFileManager* fileManager, const std::string& prefix)
{
    runner->fullPrecisionTrajectoryFileName_ = fileManager->getTemporaryFilePath(prefix + ".trr").string();
    runner->reducedPrecisionTrajectoryFileName_ = fileManager->getTemporaryFilePath(prefix + ".xtc").string();
    runner->groOutputFileName_ = fileManager->getTemporaryFilePath(prefix + ".gro").string();
    runner->cptOutputFileName_ = fileManager->getTemporaryFilePath(prefix + ".cpt").string();
    runner->logFileName_ = fileManager->getTemporaryFilePath(prefix + ".log").string();
    runner->edrFileName_ = fileManager->getTemporaryFilePath(prefix + ".edr").string();
    runner->mtxFileName_ = fileManager->getTemporaryFilePath(prefix + ".mtx").string();
}

struct RuntimeSinglePointResult
{
    std::map<std::string, double> breakdownKcalMol;
    std::vector<RVec>             forces;
    double                        elapsedMilliseconds = 0;
};

RuntimeSinglePointResult runSinglePointSimulation(SimulationRunner* runner, const CommandLine& caller)
{
    const auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(0, runner->callMdrun(caller));
    const auto stop = std::chrono::steady_clock::now();

    RuntimeSinglePointResult result;
    result.breakdownKcalMol = readStep0EnergyBreakdown(runner->edrFileName_);
    TrajectoryFrameReader reader(runner->fullPrecisionTrajectoryFileName_);
    const auto frameForces = reader.frame().f();
    result.forces.assign(frameForces.begin(), frameForces.end());
    result.elapsedMilliseconds = std::chrono::duration<double, std::milli>(stop - start).count();
    return result;
}

MessageStringCollector getGpuResidentSkipMessages()
{
    MessageStringCollector messages = getGpuSinglePointSkipMessages();
    messages.appendIf(!GpuConfigurationCapabilities::Update, "this build does not support GPU update");
    messages.appendIf(!GpuConfigurationCapabilities::BufferOps,
                      "this build does not support GPU X/F buffer operations");
    return messages;
}

std::string makeGpuResidentNveMdp()
{
    /* Keep energy/virial work off most steps so the run actually exercises GPU
     * F-buffer ops and GPU PME force reduction on the intermediate steps.
     * nstcalcenergy = 1 would force virial every step and route force reduction
     * back through the CPU path, which is the wrong dataflow for M8. */
    std::ostringstream mdp;
    mdp << "title                   = pcff gpu resident parity\n"
        << "integrator              = md\n"
        << "dt                      = 0.0005\n"
        << "nsteps                  = 20\n"
        << "constraints             = none\n"
        << "cutoff-scheme           = Verlet\n"
        << "nstlist                 = 1\n"
        << "rlist                   = 0.9\n"
        << "rvdw                    = 0.9\n"
        << "rcoulomb                = 0.9\n"
        << "vdwtype                 = Cut-off\n"
        << "coulombtype             = PME\n"
        << "ewald-rtol              = 1e-6\n"
        << "pme-order               = 4\n"
        << "fourierspacing          = " << c_m5FourierSpacingNm << "\n"
        << "epsilon-r               = 1\n"
        << "pbc                     = xyz\n"
        << "tcoupl                  = no\n"
        << "pcoupl                  = no\n"
        << "comm-mode               = none\n"
        << "verlet-buffer-tolerance = -1\n"
        << "gen-vel                 = no\n"
        << "nstcalcenergy           = 20\n"
        << "nstenergy               = 20\n"
        << "nstlog                  = 20\n"
        << "nstxout                 = 20\n"
        << "nstvout                 = 20\n"
        << "nstfout                 = 0\n"
        << "nstxout-compressed      = 0\n";
    return mdp.str();
}

CommandLine makeGpuResidentShortMdCaller(const bool        useGpuPath,
                                         const std::string& bondedTarget = "cpu")
{
    CommandLine caller;
    caller.addOption("-notunepme");
    caller.addOption("-nb", useGpuPath ? "gpu" : "cpu");
    caller.addOption("-pme", useGpuPath ? "gpu" : "cpu");
    if (useGpuPath)
    {
        caller.addOption("-pmefft", "gpu");
        caller.addOption("-update", "gpu");
    }
    else
    {
        caller.addOption("-update", "cpu");
    }
    caller.addOption("-bonded", bondedTarget);
    return caller;
}

std::string readWholeFile(const std::string& path)
{
    std::ifstream input(path);
    GMX_RELEASE_ASSERT(input.is_open(), "Could not open file for reading");
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

struct GroAtomRecord
{
    std::string residueName;
    std::string atomName;
    RVec        coordinate = { 0, 0, 0 };
    RVec        velocity   = { 0, 0, 0 };
    bool        hasVelocity = false;
};

struct ReplicatedSaltBoxFixture
{
    std::filesystem::path topPath;
    std::filesystem::path groPath;
    int                   numAtoms    = 0;
    int                   numReplicas = 0;
};

struct BondedCpuBenchmarkResult
{
    int    numAtoms                       = 0;
    int    numReplicas                    = 0;
    double forceSeconds                   = 0;
    double oneFourPercentOfTotalFlops     = 0;
    double bondsPercentOfTotalFlops       = 0;
    double anglesPercentOfTotalFlops      = 0;
    double propersPercentOfTotalFlops     = 0;
    double impropersPercentOfTotalFlops   = 0;
    double class2BondedPercentOfTotalFlops = 0;
    double listedRelatedPercentOfTotalFlops = 0;
    double class2OverallSpeedupUpperBound   = 1.0;
    double listedOverallSpeedupUpperBound   = 1.0;
};

std::string trimCopy(const std::string& value)
{
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c) != 0; });
    const auto last =
            std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (first >= last)
    {
        return "";
    }
    return std::string(first, last);
}

std::vector<std::string> splitWhitespace(const std::string& line)
{
    std::istringstream        stream(line);
    std::vector<std::string>  tokens;
    std::string               token;
    while (stream >> token)
    {
        tokens.push_back(token);
    }
    return tokens;
}

bool startsWithWallcycleLabel(const std::string& line, const std::string& label)
{
    const std::string trimmed = trimCopy(line);
    return trimmed.rfind(label, 0) == 0
           && (trimmed.size() == label.size() || std::isspace(static_cast<unsigned char>(trimmed[label.size()])) != 0);
}

std::optional<double> extractWallcycleSeconds(const std::string& logContents, const std::string& label)
{
    std::istringstream input(logContents);
    std::string        line;
    while (std::getline(input, line))
    {
        if (!startsWithWallcycleLabel(line, label))
        {
            continue;
        }

        const auto tokens = splitWhitespace(line);
        if (tokens.size() < 4)
        {
            continue;
        }
        try
        {
            return std::stod(tokens[tokens.size() - 3]);
        }
        catch (const std::exception&)
        {
            continue;
        }
    }
    return std::nullopt;
}

std::optional<double> extractMegaFlopsPercent(const std::string& logContents, const std::string& label)
{
    std::istringstream input(logContents);
    std::string        line;
    while (std::getline(input, line))
    {
        if (!startsWithWallcycleLabel(line, label))
        {
            continue;
        }

        const auto tokens = splitWhitespace(line);
        if (tokens.empty())
        {
            continue;
        }
        try
        {
            return std::stod(tokens.back());
        }
        catch (const std::exception&)
        {
            continue;
        }
    }
    return std::nullopt;
}

GroAtomRecord parseGroAtomLine(const std::string& line)
{
    GMX_RELEASE_ASSERT(line.size() >= 44, "GRO atom line is unexpectedly short");
    const auto parseRealField = [&line](std::size_t offset, std::size_t width) {
        return std::stod(line.substr(offset, width));
    };

    GroAtomRecord atom;
    atom.residueName         = trimCopy(line.substr(5, 5));
    atom.atomName            = trimCopy(line.substr(10, 5));
    atom.coordinate[XX]      = parseRealField(20, 12);
    atom.coordinate[YY]      = parseRealField(32, 12);
    atom.coordinate[ZZ]      = parseRealField(44, 12);
    atom.hasVelocity         = (line.size() >= 80);
    if (atom.hasVelocity)
    {
        atom.velocity[XX] = parseRealField(56, 12);
        atom.velocity[YY] = parseRealField(68, 12);
        atom.velocity[ZZ] = parseRealField(80, 12);
    }
    return atom;
}

ReplicatedSaltBoxFixture writeReplicatedSaltBoxFixture(TestFileManager* fileManager, const int nx, const int ny, const int nz)
{
    GMX_RELEASE_ASSERT(nx > 0 && ny > 0 && nz > 0, "Replication factors must be positive");

    const auto templateRoot = m4ReferenceRoot("small_salt_polymer_box");
    const auto groContents  = readWholeFile((templateRoot / "initial_nve.gro").string());
    const auto topContents  = readWholeFile((templateRoot / "topol.top").string());

    std::istringstream groStream(groContents);
    std::string        groTitle;
    std::string        atomCountLine;
    std::getline(groStream, groTitle);
    std::getline(groStream, atomCountLine);
    const int templateAtomCount = std::stoi(trimCopy(atomCountLine));

    std::vector<GroAtomRecord> templateAtoms;
    templateAtoms.reserve(templateAtomCount);
    std::string line;
    for (int atom = 0; atom < templateAtomCount; ++atom)
    {
        std::getline(groStream, line);
        templateAtoms.push_back(parseGroAtomLine(line));
    }

    std::getline(groStream, line);
    const auto boxTokens = splitWhitespace(line);
    GMX_RELEASE_ASSERT(boxTokens.size() == 3, "Expected orthorhombic GRO box for replicated salt-box benchmark");
    const double boxX = std::stod(boxTokens[0]);
    const double boxY = std::stod(boxTokens[1]);
    const double boxZ = std::stod(boxTokens[2]);

    const int polymerAtoms = 8;
    const int cationAtoms  = 1;
    const int anionAtoms   = 1;
    GMX_RELEASE_ASSERT(templateAtomCount == polymerAtoms + cationAtoms + anionAtoms,
                       "Unexpected template atom count for replicated salt-box benchmark");

    ReplicatedSaltBoxFixture fixture;
    fixture.numReplicas = nx * ny * nz;
    fixture.numAtoms    = fixture.numReplicas * templateAtomCount;
    fixture.topPath     = fileManager->getTemporaryFilePath(formatString("pcff-m9-%dx%dx%d.top", nx, ny, nz));
    fixture.groPath     = fileManager->getTemporaryFilePath(formatString("pcff-m9-%dx%dx%d.gro", nx, ny, nz));

    std::ostringstream groOutput;
    groOutput << "replicated small_salt_polymer_box for M9 bonded benchmark\n";
    groOutput << fixture.numAtoms << "\n";

    int atomNumber    = 1;
    int residueNumber = 1;
    for (int iz = 0; iz < nz; ++iz)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            for (int ix = 0; ix < nx; ++ix)
            {
                const RVec shift = {
                    static_cast<real>(ix * boxX),
                    static_cast<real>(iy * boxY),
                    static_cast<real>(iz * boxZ),
                };

                for (int localAtom = 0; localAtom < templateAtomCount; ++localAtom)
                {
                    const auto& templateAtom = templateAtoms[localAtom];
                    const int currentResidue = residueNumber + (localAtom < polymerAtoms ? 0 : (localAtom == polymerAtoms ? 1 : 2));
                    const auto& coordinate   = templateAtom.coordinate;
                    const auto& velocity     = templateAtom.velocity;
                    groOutput << formatString("%5d%-5.5s%5.5s%5d%12.8f%12.8f%12.8f%12.8f%12.8f%12.8f\n",
                                              currentResidue % 100000,
                                              templateAtom.residueName.c_str(),
                                              templateAtom.atomName.c_str(),
                                              atomNumber % 100000,
                                              coordinate[XX] + shift[XX],
                                              coordinate[YY] + shift[YY],
                                              coordinate[ZZ] + shift[ZZ],
                                              velocity[XX],
                                              velocity[YY],
                                              velocity[ZZ]);
                    atomNumber++;
                }
                residueNumber += 3;
            }
        }
    }
    groOutput << formatString("%12.8f%12.8f%12.8f\n", nx * boxX, ny * boxY, nz * boxZ);
    TextWriter::writeFileFromString(fixture.groPath.string(), groOutput.str());

    std::istringstream topStream(topContents);
    std::ostringstream topOutput;
    bool               wroteMoleculeSection = false;
    while (std::getline(topStream, line))
    {
        if (trimCopy(line) == "[ molecules ]")
        {
            topOutput << "[ molecules ]\n";
            topOutput << "; Name number\n";
            for (int replica = 0; replica < fixture.numReplicas; ++replica)
            {
                topOutput << "POL 1\n";
                topOutput << "CAT 1\n";
                topOutput << "ANI 1\n";
            }
            wroteMoleculeSection = true;
            break;
        }
        topOutput << line << "\n";
    }
    GMX_RELEASE_ASSERT(wroteMoleculeSection, "Could not locate [ molecules ] section in benchmark topology");
    TextWriter::writeFileFromString(fixture.topPath.string(), topOutput.str());

    return fixture;
}

std::string makeGpuBondedBenchmarkMdp()
{
    std::ostringstream mdp;
    mdp << "title                   = pcff gpu bonded benchmark\n"
        << "integrator              = md\n"
        << "dt                      = 0.0005\n"
        << "nsteps                  = 200\n"
        << "constraints             = none\n"
        << "cutoff-scheme           = Verlet\n"
        << "nstlist                 = 20\n"
        << "rlist                   = 0.9\n"
        << "rvdw                    = 0.9\n"
        << "rcoulomb                = 0.9\n"
        << "vdwtype                 = Cut-off\n"
        << "coulombtype             = PME\n"
        << "ewald-rtol              = 1e-6\n"
        << "pme-order               = 4\n"
        << "fourierspacing          = " << c_m5FourierSpacingNm << "\n"
        << "epsilon-r               = 1\n"
        << "pbc                     = xyz\n"
        << "tcoupl                  = no\n"
        << "pcoupl                  = no\n"
        << "comm-mode               = none\n"
        << "verlet-buffer-tolerance = -1\n"
        << "gen-vel                 = no\n"
        << "nstcalcenergy           = 200\n"
        << "nstenergy               = 200\n"
        << "nstlog                  = 200\n"
        << "nstxout                 = 0\n"
        << "nstvout                 = 0\n"
        << "nstfout                 = 0\n"
        << "nstxout-compressed      = 0\n";
    return mdp.str();
}

BondedCpuBenchmarkResult benchmarkCpuBondedOnReplicatedSaltBox(SimulationRunner* runner,
                                                               TestFileManager*  fileManager)
{
    const auto fixture = writeReplicatedSaltBoxFixture(fileManager, 8, 8, 8);

    runner->topFileName_ = fixture.topPath.string();
    runner->groFileName_ = fixture.groPath.string();
    runner->useStringAsMdpFile(makeGpuBondedBenchmarkMdp());
    runner->setMaxWarn(1);
    assignRunnerOutputs(runner, fileManager, "pcff-m9-benchmark");

    CommandLine caller;
    caller.addOption("-notunepme");
    caller.addOption("-nb", "gpu");
    caller.addOption("-pme", "gpu");
    caller.addOption("-pmefft", "gpu");
    caller.addOption("-bonded", "cpu");
    caller.addOption("-update", "gpu");

    EXPECT_EQ(0, runner->callGrompp()) << "grompp failed for replicated M9 benchmark fixture";
    EXPECT_EQ(0, runner->callMdrun(caller)) << "mdrun failed for replicated M9 benchmark fixture";

    const std::string logContents        = readWholeFile(runner->logFileName_);
    const auto        forceSeconds       = extractWallcycleSeconds(logContents, "Force");
    const auto        oneFourPercent     = extractMegaFlopsPercent(logContents, "1,4 nonbonded interactions");
    const auto        bondsPercent       = extractMegaFlopsPercent(logContents, "Bonds");
    const auto        anglesPercent      = extractMegaFlopsPercent(logContents, "Angles");
    const auto        propersPercent     = extractMegaFlopsPercent(logContents, "Propers");
    const auto        impropersPercent   = extractMegaFlopsPercent(logContents, "Impropers");

    GMX_RELEASE_ASSERT(forceSeconds.has_value(), "Could not parse Force wallcycle time from M9 benchmark log");
    GMX_RELEASE_ASSERT(oneFourPercent.has_value(),
                       "Could not parse 1,4 nonbonded M-Flops percentage from M9 benchmark log");
    GMX_RELEASE_ASSERT(bondsPercent.has_value(), "Could not parse Bonds M-Flops percentage from M9 benchmark log");
    GMX_RELEASE_ASSERT(anglesPercent.has_value(), "Could not parse Angles M-Flops percentage from M9 benchmark log");
    GMX_RELEASE_ASSERT(propersPercent.has_value(),
                       "Could not parse Propers M-Flops percentage from M9 benchmark log");

    BondedCpuBenchmarkResult result;
    result.numAtoms                       = fixture.numAtoms;
    result.numReplicas                    = fixture.numReplicas;
    result.forceSeconds                   = forceSeconds.value();
    result.oneFourPercentOfTotalFlops     = oneFourPercent.value();
    result.bondsPercentOfTotalFlops       = bondsPercent.value();
    result.anglesPercentOfTotalFlops      = anglesPercent.value();
    result.propersPercentOfTotalFlops     = propersPercent.value();
    result.impropersPercentOfTotalFlops   = impropersPercent.value_or(0.0);
    result.class2BondedPercentOfTotalFlops =
            result.bondsPercentOfTotalFlops + result.anglesPercentOfTotalFlops + result.propersPercentOfTotalFlops
            + result.impropersPercentOfTotalFlops;
    result.listedRelatedPercentOfTotalFlops =
            result.class2BondedPercentOfTotalFlops + result.oneFourPercentOfTotalFlops;
    result.class2OverallSpeedupUpperBound =
            1.0 / std::max(1e-12, 1.0 - result.class2BondedPercentOfTotalFlops / 100.0);
    result.listedOverallSpeedupUpperBound =
            1.0 / std::max(1e-12, 1.0 - result.listedRelatedPercentOfTotalFlops / 100.0);
    return result;
}

struct RuntimeShortMdResult
{
    std::map<std::string, double> observables;
    StructuralMetrics             structural;
    FinalTrajectorySnapshot       finalSnapshot;
    std::string                   logContents;
};

RuntimeShortMdResult runShortMdSimulation(SimulationRunner* runner,
                                          const CommandLine& caller,
                                          const std::string& systemId)
{
    EXPECT_EQ(0, runner->callMdrun(caller));

    const std::vector<std::string> energyTerms = {
        interaction_function[InteractionFunction::PotentialEnergy].longname,
        interaction_function[InteractionFunction::TotalEnergy].longname,
    };
    const auto energyFrames = readEnergyFrames(runner->edrFileName_, energyTerms);
    GMX_RELEASE_ASSERT(energyFrames.size() >= 2, "Expected at least initial and final energy frames");

    RuntimeShortMdResult result;
    const auto& firstFrame = energyFrames.front();
    const auto& lastFrame  = energyFrames.back();

    const auto totalEnergyName = interaction_function[InteractionFunction::TotalEnergy].longname;
    const auto totalEnergyToKcal = [&totalEnergyName](const EnergyFrame& frame) { return kjToKcal(frame.at(totalEnergyName)); };
    double minTotal = totalEnergyToKcal(firstFrame);
    double maxTotal = minTotal;
    for (const auto& frame : energyFrames)
    {
        const double total = totalEnergyToKcal(frame);
        minTotal = std::min(minTotal, total);
        maxTotal = std::max(maxTotal, total);
    }

    result.observables["step0_potential_kcal_mol"] =
            kjToKcal(firstFrame.at(interaction_function[InteractionFunction::PotentialEnergy].longname));
    result.observables["initial_total_kcal_mol"] = totalEnergyToKcal(firstFrame);
    result.observables["final_total_kcal_mol"]   = totalEnergyToKcal(lastFrame);
    result.observables["total_energy_drift_abs_kcal_mol"] =
            std::abs(result.observables["final_total_kcal_mol"] - result.observables["initial_total_kcal_mol"]);
    result.observables["total_energy_span_kcal_mol"] = maxTotal - minTotal;

    result.structural    = readLastStructuralMetrics(systemId, runner->fullPrecisionTrajectoryFileName_);
    result.finalSnapshot = readFinalTrajectorySnapshot(runner->fullPrecisionTrajectoryFileName_);
    result.logContents   = readWholeFile(runner->logFileName_);
    return result;
}

double coordinateMaxDifferenceNm(const FinalTrajectorySnapshot& reference, const FinalTrajectorySnapshot& actual)
{
    GMX_RELEASE_ASSERT(reference.coordinates.size() == actual.coordinates.size(), "Coordinate sizes must match");
    double maxDelta = 0;
    for (gmx::Index atom = 0; atom < gmx::ssize(reference.coordinates); ++atom)
    {
        const RVec delta = minimumImageVector(reference.coordinates[atom], actual.coordinates[atom], reference.box);
        maxDelta         = std::max(maxDelta, norm(delta));
    }
    return maxDelta;
}

double coordinateRmsDifferenceNm(const FinalTrajectorySnapshot& reference, const FinalTrajectorySnapshot& actual)
{
    GMX_RELEASE_ASSERT(reference.coordinates.size() == actual.coordinates.size(), "Coordinate sizes must match");
    double sumSquared = 0;
    for (gmx::Index atom = 0; atom < gmx::ssize(reference.coordinates); ++atom)
    {
        const RVec delta = minimumImageVector(reference.coordinates[atom], actual.coordinates[atom], reference.box);
        const double deltaNorm = norm(delta);
        sumSquared += deltaNorm * deltaNorm;
    }
    return std::sqrt(sumSquared / std::max<gmx::Index>(1, gmx::ssize(reference.coordinates)));
}

double velocityMaxDifferenceNmPerPs(const FinalTrajectorySnapshot& reference, const FinalTrajectorySnapshot& actual)
{
    GMX_RELEASE_ASSERT(reference.velocities.size() == actual.velocities.size(), "Velocity sizes must match");
    double maxDelta = 0;
    for (gmx::Index atom = 0; atom < gmx::ssize(reference.velocities); ++atom)
    {
        const double dx = actual.velocities[atom][XX] - reference.velocities[atom][XX];
        const double dy = actual.velocities[atom][YY] - reference.velocities[atom][YY];
        const double dz = actual.velocities[atom][ZZ] - reference.velocities[atom][ZZ];
        maxDelta = std::max(maxDelta, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return maxDelta;
}

double velocityRmsDifferenceNmPerPs(const FinalTrajectorySnapshot& reference, const FinalTrajectorySnapshot& actual)
{
    GMX_RELEASE_ASSERT(reference.velocities.size() == actual.velocities.size(), "Velocity sizes must match");
    double sumSquared = 0;
    for (gmx::Index atom = 0; atom < gmx::ssize(reference.velocities); ++atom)
    {
        const double dx = actual.velocities[atom][XX] - reference.velocities[atom][XX];
        const double dy = actual.velocities[atom][YY] - reference.velocities[atom][YY];
        const double dz = actual.velocities[atom][ZZ] - reference.velocities[atom][ZZ];
        sumSquared += dx * dx + dy * dy + dz * dz;
    }
    return std::sqrt(sumSquared / std::max<gmx::Index>(1, gmx::ssize(reference.velocities)));
}

TEST_P(PcffGpuSinglePointParityTest, GpuNonbondedAndPmeMatchCpuSinglePointReference)
{
    const auto skipMessages = getGpuSinglePointSkipMessages();
    if (!skipMessages.isEmpty())
    {
        GTEST_SKIP() << skipMessages.toString();
    }

    const std::string systemId(GetParam());
    const auto        fixtureRoot = m4ReferenceRoot(systemId);

    runner_.topFileName_ = (fixtureRoot / "topol.top").string();
    runner_.groFileName_ = (fixtureRoot / "initial_nve.gro").string();
    runner_.useStringAsMdpFile(makeSinglePointMdp());
    runner_.setMaxWarn(1);

    ASSERT_EQ(0, runner_.callGrompp()) << "grompp failed for " << systemId;

    SimulationRunner gpuRunner(&fileManager_);
    gpuRunner.tprFileName_ = runner_.tprFileName_;
    setRunnerOutputPrefix(&runner_, &fileManager_, systemId + "_cpu_single_point");
    setRunnerOutputPrefix(&gpuRunner, &fileManager_, systemId + "_gpu_single_point");

    const auto cpuResult = runSinglePointSimulation(&runner_, makeSinglePointMdrunCaller(false, false));
    const auto gpuResult = runSinglePointSimulation(&gpuRunner, makeSinglePointMdrunCaller(true, true));

    const std::array<std::string, 9> terms = { "bond_kcal_mol",
                                               "angle_kcal_mol",
                                               "dihedral_kcal_mol",
                                               "lj14_kcal_mol",
                                               "ljsr_kcal_mol",
                                               "coul14_kcal_mol",
                                               "coulsr_kcal_mol",
                                               "coul_recip_kcal_mol",
                                               "potential_total_kcal_mol" };

    const double energyTolerance = 8e-3;
    for (const auto& term : terms)
    {
        ASSERT_TRUE(cpuResult.breakdownKcalMol.count(term) > 0) << term;
        ASSERT_TRUE(gpuResult.breakdownKcalMol.count(term) > 0) << term;
        EXPECT_NEAR(gpuResult.breakdownKcalMol.at(term), cpuResult.breakdownKcalMol.at(term), energyTolerance)
                << systemId << " term=" << term;
    }

    ASSERT_EQ(gmx::ssize(cpuResult.forces), gmx::ssize(gpuResult.forces)) << systemId << " atom count";
    const double forceTolerance = 6e-2;
    for (Index atom = 0; atom < ssize(cpuResult.forces); ++atom)
    {
        for (int d = 0; d < DIM; ++d)
        {
            EXPECT_NEAR(gpuResult.forces[atom][d], cpuResult.forces[atom][d], forceTolerance)
                    << systemId << " atom=" << atom << " dim=" << d;
        }
    }

}

TEST_F(PcffGpuPerfSmokeTest, GpuSinglePointRunsProduceFiniteWallclock)
{
    const auto skipMessages = getGpuSinglePointSkipMessages();
    if (!skipMessages.isEmpty())
    {
        GTEST_SKIP() << skipMessages.toString();
    }

    const auto fixtureRoot = m4ReferenceRoot("small_oligomer");
    runner_.topFileName_ = (fixtureRoot / "topol.top").string();
    runner_.groFileName_ = (fixtureRoot / "initial_nve.gro").string();
    runner_.useStringAsMdpFile(makeSinglePointMdp());
    runner_.setMaxWarn(1);

    ASSERT_EQ(0, runner_.callGrompp()) << "grompp failed for small_oligomer";

    double accumulatedGpuMilliseconds = 0.0;
    for (int iteration = 0; iteration < 3; ++iteration)
    {
        SimulationRunner gpuRunner(&fileManager_);
        gpuRunner.tprFileName_ = runner_.tprFileName_;
        const auto gpuResult = runSinglePointSimulation(&gpuRunner, makeSinglePointMdrunCaller(true, true));
        EXPECT_TRUE(std::isfinite(gpuResult.elapsedMilliseconds));
        EXPECT_GT(gpuResult.elapsedMilliseconds, 0.0);
        accumulatedGpuMilliseconds += gpuResult.elapsedMilliseconds;
    }

    EXPECT_TRUE(std::isfinite(accumulatedGpuMilliseconds));
    EXPECT_GT(accumulatedGpuMilliseconds, 0.0);

}

TEST_F(PcffGpuPerfSmokeTest, ReplicatedSaltBoxBenchmarkReportsCpuBondedShare)
{
    const auto skipMessages = getGpuResidentSkipMessages();
    if (!skipMessages.isEmpty())
    {
        GTEST_SKIP() << skipMessages.toString();
    }

    const auto result = benchmarkCpuBondedOnReplicatedSaltBox(&runner_, &fileManager_);

    EXPECT_GT(result.numAtoms, 0);
    EXPECT_GT(result.numReplicas, 0);
    EXPECT_TRUE(std::isfinite(result.forceSeconds));
    EXPECT_GT(result.forceSeconds, 0.0);
    EXPECT_TRUE(std::isfinite(result.oneFourPercentOfTotalFlops));
    EXPECT_TRUE(std::isfinite(result.bondsPercentOfTotalFlops));
    EXPECT_TRUE(std::isfinite(result.anglesPercentOfTotalFlops));
    EXPECT_TRUE(std::isfinite(result.propersPercentOfTotalFlops));
    EXPECT_TRUE(std::isfinite(result.impropersPercentOfTotalFlops));
    EXPECT_TRUE(std::isfinite(result.class2BondedPercentOfTotalFlops));
    EXPECT_TRUE(std::isfinite(result.listedRelatedPercentOfTotalFlops));
    EXPECT_TRUE(std::isfinite(result.class2OverallSpeedupUpperBound));
    EXPECT_TRUE(std::isfinite(result.listedOverallSpeedupUpperBound));
    EXPECT_GT(result.class2BondedPercentOfTotalFlops, 0.0);
    EXPECT_GT(result.listedRelatedPercentOfTotalFlops, result.class2BondedPercentOfTotalFlops);
    EXPECT_LT(result.listedRelatedPercentOfTotalFlops, 100.0);
    EXPECT_GT(result.class2OverallSpeedupUpperBound, 1.0);
    EXPECT_GT(result.listedOverallSpeedupUpperBound, result.class2OverallSpeedupUpperBound);

    std::cout << "[M9 perf] replicated_salt_polymer_box atoms=" << result.numAtoms
              << " replicas=" << result.numReplicas << " force_s=" << std::setprecision(6)
              << result.forceSeconds << " bonds_pct=" << result.bondsPercentOfTotalFlops
              << " angles_pct=" << result.anglesPercentOfTotalFlops
              << " propers_pct=" << result.propersPercentOfTotalFlops
              << " impropers_pct=" << result.impropersPercentOfTotalFlops
              << " pair14_pct=" << result.oneFourPercentOfTotalFlops
              << " class2_bonded_pct=" << result.class2BondedPercentOfTotalFlops
              << " listed_related_pct=" << result.listedRelatedPercentOfTotalFlops
              << " class2_speedup_upper_bound=" << result.class2OverallSpeedupUpperBound
              << " listed_speedup_upper_bound=" << result.listedOverallSpeedupUpperBound << std::endl;
}

TEST_P(PcffGpuResidentParityTest, GpuUpdateAndBufferOpsPreserveShortNveObservables)
{
    const auto skipMessages = getGpuResidentSkipMessages();
    if (!skipMessages.isEmpty())
    {
        GTEST_SKIP() << skipMessages.toString();
    }

    const std::string systemId(GetParam());
    const auto        fixtureRoot = referenceRoot(systemId);

    runner_.topFileName_ = (fixtureRoot / "topol.top").string();
    runner_.groFileName_ = (fixtureRoot / "initial_nve.gro").string();
    runner_.useStringAsMdpFile(makeGpuResidentNveMdp());
    runner_.setMaxWarn(1);

    ASSERT_EQ(0, runner_.callGrompp()) << "grompp failed for " << systemId;

    SimulationRunner gpuRunner(&fileManager_);
    gpuRunner.tprFileName_ = runner_.tprFileName_;
    assignRunnerOutputs(&runner_, &fileManager_, systemId + "-m8-cpu");
    assignRunnerOutputs(&gpuRunner, &fileManager_, systemId + "-m8-gpu");

    const auto cpuResult = runShortMdSimulation(&runner_, makeGpuResidentShortMdCaller(false), systemId);
    const auto gpuResult = runShortMdSimulation(&gpuRunner, makeGpuResidentShortMdCaller(true), systemId);

    EXPECT_NE(gpuResult.logContents.find("PP task will update and constrain coordinates on the GPU"), std::string::npos)
            << systemId;
    EXPECT_NE(gpuResult.logContents.find("PME tasks will do all aspects on the GPU"), std::string::npos)
            << systemId;

    const std::array<std::string, 5> scalarObservables = {
        "step0_potential_kcal_mol",
        "initial_total_kcal_mol",
        "final_total_kcal_mol",
        "total_energy_drift_abs_kcal_mol",
        "total_energy_span_kcal_mol",
    };
    const double energyTolerance = 3e-2;
    for (const auto& name : scalarObservables)
    {
        ASSERT_TRUE(cpuResult.observables.count(name) > 0) << name;
        ASSERT_TRUE(gpuResult.observables.count(name) > 0) << name;
        EXPECT_NEAR(gpuResult.observables.at(name), cpuResult.observables.at(name), energyTolerance)
                << systemId << " observable=" << name;
    }

    const double structureTolerance = 2e-4;
    EXPECT_NEAR(gpuResult.structural.polymerEndToEndNm, cpuResult.structural.polymerEndToEndNm, structureTolerance)
            << systemId << " polymer_end_to_end_nm";
    EXPECT_NEAR(gpuResult.structural.polymerRgNm, cpuResult.structural.polymerRgNm, structureTolerance)
            << systemId << " polymer_rg_nm";
    if (cpuResult.structural.ionDistanceNm.has_value())
    {
        ASSERT_TRUE(gpuResult.structural.ionDistanceNm.has_value()) << systemId;
        EXPECT_NEAR(gpuResult.structural.ionDistanceNm.value(),
                    cpuResult.structural.ionDistanceNm.value(),
                    structureTolerance)
                << systemId << " ion_distance_nm";
    }

    const double coordinateRmsTolerance = 2e-4;
    const double coordinateMaxTolerance = 6e-4;
    const double velocityRmsTolerance   = 1.5e-3;
    const double velocityMaxTolerance   = 5e-3;

    const double coordinateRms = coordinateRmsDifferenceNm(cpuResult.finalSnapshot, gpuResult.finalSnapshot);
    const double coordinateMax = coordinateMaxDifferenceNm(cpuResult.finalSnapshot, gpuResult.finalSnapshot);
    const double velocityRms =
            velocityRmsDifferenceNmPerPs(cpuResult.finalSnapshot, gpuResult.finalSnapshot);
    const double velocityMax =
            velocityMaxDifferenceNmPerPs(cpuResult.finalSnapshot, gpuResult.finalSnapshot);

    EXPECT_LE(coordinateRms, coordinateRmsTolerance) << systemId << " coordinate rms";
    EXPECT_LE(coordinateMax, coordinateMaxTolerance) << systemId << " coordinate max";
    EXPECT_LE(velocityRms, velocityRmsTolerance) << systemId << " velocity rms";
    EXPECT_LE(velocityMax, velocityMaxTolerance) << systemId << " velocity max";
}
#endif

TEST_P(PcffRespaObservableDumpTest, DumpsExactRespaNveObservables)
{
    const std::string systemId(GetParam());
    const auto        fixtureRoot = repoRoot() / "tests" / "reference_results" / "m6_respa" / systemId;
    const std::vector<std::string> energyTerms = {
        interaction_function[InteractionFunction::PotentialEnergy].longname,
        interaction_function[InteractionFunction::TotalEnergy].longname,
        "Vir-XX",
        "Vir-XY",
        "Vir-XZ",
        "Vir-YX",
        "Vir-YY",
        "Vir-YZ",
        "Vir-ZX",
        "Vir-ZY",
        "Vir-ZZ",
    };
    const auto totalEnergyName = interaction_function[InteractionFunction::TotalEnergy].longname;
    const auto potentialName   = interaction_function[InteractionFunction::PotentialEnergy].longname;
    const auto totalEnergyToKcal =
            [&totalEnergyName](const EnergyFrame& frame) { return kjToKcal(frame.at(totalEnergyName)); };
    const auto computeTotalEnergySpanKcal = [&totalEnergyToKcal](const std::vector<EnergyFrame>& frames)
    {
        double minTotal = totalEnergyToKcal(frames.front());
        double maxTotal = minTotal;
        for (const auto& frame : frames)
        {
            const double total = totalEnergyToKcal(frame);
            minTotal = std::min(minTotal, total);
            maxTotal = std::max(maxTotal, total);
        }
        return maxTotal - minTotal;
    };

    runner_.topFileName_ = (fixtureRoot / "topol.top").string();
    runner_.groFileName_ = (fixtureRoot / "initial_nve.gro").string();
    runner_.useStringAsMdpFile(makeSinglePointMdp());
    runner_.setMaxWarn(1);

    ASSERT_EQ(0, runner_.callGrompp()) << "grompp failed for " << systemId << " single-time step reference";
    ASSERT_EQ(0, runner_.callMdrun()) << "mdrun failed for " << systemId << " single-time step reference";
    const auto unsplitEnergyFrames = readEnergyFrames(runner_.edrFileName_, energyTerms);
    ASSERT_FALSE(unsplitEnergyFrames.empty());
    const double unsplitTotalEnergySpanKcalMol = computeTotalEnergySpanKcal(unsplitEnergyFrames);
    TrajectoryFrameReader unsplitTrajectoryReader(runner_.fullPrecisionTrajectoryFileName_);
    const auto unsplitBox = unsplitTrajectoryReader.frame().box();
    appendStep0ScalarEnergyObservableTrace("single_step", unsplitEnergyFrames.front());
    appendScalarTotalObservableTrace(
            "single_step",
            unsplitEnergyFrames.front(),
            unsplitEnergyFrames.back(),
            unsplitTotalEnergySpanKcalMol);
    appendStep0EdrFrameVirialVolumeObservableTrace("single_step", unsplitEnergyFrames.front(), unsplitBox);
    const auto unsplitPressureRows = step0VirialPressureTensorAtm(unsplitEnergyFrames.front(), unsplitBox);
    appendStep0DerivedVirialPressureObservableTrace("single_step", unsplitPressureRows);
    const auto unsplitBreakdown = readStep0EnergyBreakdown(runner_.edrFileName_);

    runner_.topFileName_ = (fixtureRoot / "topol.top").string();
    runner_.groFileName_ = (fixtureRoot / "initial_nve.gro").string();
    runner_.useStringAsMdpFile(makeRespaNveMdp());
    runner_.setMaxWarn(1);

    ASSERT_EQ(0, runner_.callGrompp()) << "grompp failed for " << systemId << " exact respa";
    ASSERT_EQ(0, runner_.callMdrun()) << "mdrun failed for " << systemId << " exact respa";
    const auto energyFrames = readEnergyFrames(runner_.edrFileName_, energyTerms);
    ASSERT_FALSE(energyFrames.empty());
    const double exactTotalEnergySpanKcalMol = computeTotalEnergySpanKcal(energyFrames);
    TrajectoryFrameReader exactTrajectoryReader(runner_.fullPrecisionTrajectoryFileName_);
    const auto exactBox = exactTrajectoryReader.frame().box();
    appendScalarTotalObservableTrace(
            "exact_respa", energyFrames.front(), energyFrames.back(), exactTotalEnergySpanKcalMol);
    appendStep0EdrFrameVirialVolumeObservableTrace("exact_respa", energyFrames.front(), exactBox);
    const auto exactPressureRows = step0VirialPressureTensorAtm(energyFrames.front(), exactBox);
    appendStep0DerivedVirialPressureObservableTrace("exact_respa", exactPressureRows);

    const auto exactBreakdown = readStep0EnergyBreakdown(runner_.edrFileName_);

    std::map<std::string, double> actualValues;
    actualValues["step0_potential_kcal_mol"] = kjToKcal(energyFrames.front().at(potentialName));
    actualValues["initial_total_kcal_mol"] = totalEnergyToKcal(energyFrames.front());
    actualValues["final_total_kcal_mol"] = totalEnergyToKcal(energyFrames.back());
    actualValues["total_energy_drift_abs_kcal_mol"] =
            std::abs(actualValues["final_total_kcal_mol"] - actualValues["initial_total_kcal_mol"]);
    actualValues["total_energy_span_kcal_mol"] = exactTotalEnergySpanKcalMol;

    for (const auto& [name, value] : exactPressureRows)
    {
        actualValues[name] = value;
    }

    const auto structural = readLastStructuralMetrics(systemId, runner_.fullPrecisionTrajectoryFileName_);
    actualValues["polymer_end_to_end_nm"] = structural.polymerEndToEndNm;
    actualValues["polymer_rg_nm"] = structural.polymerRgNm;
    if (structural.ionDistanceNm.has_value())
    {
        actualValues["ion_distance_nm"] = structural.ionDistanceNm.value();
    }

    const auto referenceContract = loadRespaReferenceContract(systemId);
    const auto comparisons       = compareMetrics(referenceContract, "nve", actualValues);
    ASSERT_FALSE(comparisons.empty()) << "Expected frozen M6 comparisons for " << systemId;

    const auto breakdownDelta = subtractMaps(exactBreakdown, unsplitBreakdown);

    writeRespaActualSummary(
            systemId,
            actualValues,
            { "Exact r-RESPA observables are dumped as machine-readable JSON and also checked against the frozen LAMMPS M6 tolerance contract in this test.",
              "The diagnostics block compares exact r-RESPA step-0 energy terms against the same GROMACS topology/state evaluated without MTS.",
              "GROMACS exact r-RESPA now targets a dedicated integrator = md-vv path instead of reusing the leap-frog md update path.",
              "The current pair14 level is pinned to level 1 for this harness, but LAMMPS special_bonds 1-4 terms are embedded in pair-style respa splitting rather than exposed as a separate schedule keyword." },
            { { "exact_step0_breakdown", exactBreakdown },
              { "single_time_step_breakdown", unsplitBreakdown },
              { "exact_minus_single_time_step", breakdownDelta } });

    for (const auto& [name, value] : actualValues)
    {
        EXPECT_TRUE(std::isfinite(value)) << "Non-finite exact r-RESPA observable for " << systemId << ": " << name;
    }

    for (const auto& [name, exactValue] : exactBreakdown)
    {
        const auto unsplitIt = unsplitBreakdown.find(name);
        ASSERT_NE(unsplitIt, unsplitBreakdown.end()) << "Missing single-time-step diagnostic for " << name;
        EXPECT_NEAR(exactValue, unsplitIt->second, 5e-3) << systemId << " step-0 breakdown mismatch for " << name;
    }

    for (const auto& comparison : comparisons)
    {
        EXPECT_NEAR(comparison.actual, comparison.reference, comparison.tolerance)
                << systemId << " exact r-RESPA mismatch for " << comparison.name;
    }
}

TEST_P(PcffRespaRestartParityTest, RestartFromCheckpointMatchesFullExactRun)
{
    const std::string systemId(GetParam());
    const auto        fixtureRoot = repoRoot() / "tests" / "reference_results" / "m6_respa" / systemId;
    const int         halfOuterSteps = std::max(1, respaOuterSteps() / 2);
    const int         halfSteps      = halfOuterSteps * c_respaEnergyInterval;
    ASSERT_EQ(0, halfSteps % c_respaEnergyInterval)
            << "exact r-RESPA restart smoke should split on an outer-step boundary";

    runner_.topFileName_ = (fixtureRoot / "topol.top").string();
    runner_.groFileName_ = (fixtureRoot / "initial_nve.gro").string();
    runner_.useStringAsMdpFile(makeRespaNveMdp());
    runner_.setMaxWarn(1);

    runner_.tprFileName_ = fileManager_.getTemporaryFilePath(systemId + "-exact-restart.tpr").string();
    ASSERT_EQ(0, runner_.callGrompp()) << "grompp failed for " << systemId << " exact restart smoke";

    assignRunnerOutputs(&runner_, &fileManager_, systemId + "-exact-full");
    const std::string fullEdrFileName = runner_.edrFileName_;
    const std::string fullTrrFileName = runner_.fullPrecisionTrajectoryFileName_;

    CommandLine fullRunCaller;
    fullRunCaller.append("mdrun");
    fullRunCaller.append("-reprod");
    ASSERT_EQ(0, runner_.callMdrun(fullRunCaller)) << "full exact run failed for " << systemId;

    assignRunnerOutputs(&runner_, &fileManager_, systemId + "-exact-split");
    const std::string splitEdrFileName = runner_.edrFileName_;
    const std::string splitTrrFileName = runner_.fullPrecisionTrajectoryFileName_;
    const std::string splitCheckpointFileName = runner_.cptOutputFileName_;

    CommandLine firstPartCaller;
    firstPartCaller.append("mdrun");
    firstPartCaller.append("-reprod");
    firstPartCaller.addOption("-nsteps", halfSteps);
    ASSERT_EQ(0, runner_.callMdrun(firstPartCaller)) << "first exact restart segment failed for " << systemId;
    ASSERT_TRUE(std::filesystem::exists(splitCheckpointFileName)) << "missing checkpoint after first segment";

    CommandLine secondPartCaller;
    secondPartCaller.append("mdrun");
    secondPartCaller.append("-reprod");
    secondPartCaller.addOption("-cpi", splitCheckpointFileName);
    ASSERT_EQ(0, runner_.callMdrun(secondPartCaller)) << "restart segment failed for " << systemId;

    const std::vector<std::string> energyTerms = {
        interaction_function[InteractionFunction::PotentialEnergy].longname,
        interaction_function[InteractionFunction::TotalEnergy].longname,
    };
    const auto fullEnergyFrames  = readEnergyFrames(fullEdrFileName, energyTerms);
    const auto splitEnergyFrames = readEnergyFrames(splitEdrFileName, energyTerms);
    ASSERT_FALSE(fullEnergyFrames.empty());
    ASSERT_FALSE(splitEnergyFrames.empty());

    const auto& fullFinalEnergy  = fullEnergyFrames.back();
    const auto& splitFinalEnergy = splitEnergyFrames.back();
    const double energyTolerance = 1e-6;

    EXPECT_NEAR(fullFinalEnergy.at(interaction_function[InteractionFunction::PotentialEnergy].longname),
                splitFinalEnergy.at(interaction_function[InteractionFunction::PotentialEnergy].longname),
                energyTolerance)
            << systemId << " final potential after checkpoint restart";
    EXPECT_NEAR(fullFinalEnergy.at(interaction_function[InteractionFunction::TotalEnergy].longname),
                splitFinalEnergy.at(interaction_function[InteractionFunction::TotalEnergy].longname),
                energyTolerance)
            << systemId << " final total energy after checkpoint restart";

    const auto fullSnapshot  = readFinalTrajectorySnapshot(fullTrrFileName);
    const auto splitSnapshot = readFinalTrajectorySnapshot(splitTrrFileName);
    ASSERT_EQ(fullSnapshot.step, splitSnapshot.step) << systemId << " final step mismatch";
    EXPECT_NEAR(fullSnapshot.time, splitSnapshot.time, 1e-12) << systemId << " final time mismatch";
    ASSERT_EQ(gmx::ssize(fullSnapshot.coordinates), gmx::ssize(splitSnapshot.coordinates));
    ASSERT_EQ(gmx::ssize(fullSnapshot.velocities), gmx::ssize(splitSnapshot.velocities));

    const double coordinateTolerance = 1e-7;
    const double velocityTolerance   = 1e-7;
    for (Index atom = 0; atom < ssize(fullSnapshot.coordinates); ++atom)
    {
        for (int d = 0; d < DIM; ++d)
        {
            EXPECT_NEAR(fullSnapshot.coordinates[atom][d], splitSnapshot.coordinates[atom][d], coordinateTolerance)
                    << systemId << " atom=" << atom << " dim=" << d << " coordinate restart mismatch";
            EXPECT_NEAR(fullSnapshot.velocities[atom][d], splitSnapshot.velocities[atom][d], velocityTolerance)
                    << systemId << " atom=" << atom << " dim=" << d << " velocity restart mismatch";
        }
    }
}

std::string shortMdCaseName(const testing::TestParamInfo<std::tuple<const char*, const char*>>& info)
{
    return formatString("%s_%s", std::get<0>(info.param), std::get<1>(info.param));
}

INSTANTIATE_TEST_SUITE_P(PcffShortMdParity,
                         PcffShortMdParityTest,
                         ::testing::Combine(::testing::Values("small_oligomer", "small_salt_polymer_box"),
                                            ::testing::Values("nve", "nvt")),
                         shortMdCaseName);

INSTANTIATE_TEST_SUITE_P(PcffSinglePointParity,
                         PcffSinglePointParityTest,
                         ::testing::Values("small_oligomer", "small_salt_polymer_box"));

#if GMX_GPU_CUDA
INSTANTIATE_TEST_SUITE_P(PcffGpuSinglePointParity,
                         PcffGpuSinglePointParityTest,
                         ::testing::Values("small_oligomer", "small_salt_polymer_box"));

INSTANTIATE_TEST_SUITE_P(PcffGpuResidentParity,
                         PcffGpuResidentParityTest,
                         ::testing::Values("small_oligomer", "small_salt_polymer_box"));
#endif

INSTANTIATE_TEST_SUITE_P(PcffRespaObservableDump,
                         PcffRespaObservableDumpTest,
                         ::testing::Values("small_oligomer", "small_salt_polymer_box"));

INSTANTIATE_TEST_SUITE_P(PcffRespaRestartParity,
                         PcffRespaRestartParityTest,
                         ::testing::Values("small_oligomer", "small_salt_polymer_box"));

} // namespace
} // namespace test
} // namespace gmx
