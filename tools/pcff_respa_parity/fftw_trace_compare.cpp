#include <fftw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace
{

struct ComplexRow
{
    int   callIndex;
    int   ix;
    int   iy;
    int   iz;
    float re;
    float im;
};

struct RealRow
{
    int   callIndex;
    int   ix;
    int   iy;
    int   iz;
    float value;
};

struct Summary
{
    double maxAbs  = 0.0;
    double rmse    = 0.0;
    double meanAbs = 0.0;
    int    ix      = -1;
    int    iy      = -1;
    int    iz      = -1;
    double a       = 0.0;
    double b       = 0.0;
    double diff    = 0.0;
};

std::unordered_map<std::string, std::string> parseTokens(const std::string& line)
{
    std::unordered_map<std::string, std::string> result;
    std::stringstream                             stream(line);
    std::string                                   token;
    while (stream >> token)
    {
        const auto pos = token.find('=');
        if (pos == std::string::npos)
        {
            continue;
        }
        result.emplace(token.substr(0, pos), token.substr(pos + 1));
    }
    return result;
}

std::vector<ComplexRow> loadComplexRows(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("Failed to open complex trace: " + path);
    }

    std::vector<ComplexRow> rows;
    std::string             line;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        const auto tokens = parseTokens(line);
        rows.push_back(ComplexRow{ tokens.count("call_index") ? std::stoi(tokens.at("call_index")) : 0,
                                   std::stoi(tokens.at("ix")),
                                   std::stoi(tokens.at("iy")),
                                   std::stoi(tokens.at("iz")),
                                   std::stof(tokens.at("re")),
                                   std::stof(tokens.at("im")) });
    }
    return rows;
}

std::vector<RealRow> loadRealRows(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("Failed to open real trace: " + path);
    }

    std::vector<RealRow> rows;
    std::string          line;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        const auto tokens = parseTokens(line);
        rows.push_back(RealRow{ std::stoi(tokens.at("call_index")),
                                std::stoi(tokens.at("ix")),
                                std::stoi(tokens.at("iy")),
                                std::stoi(tokens.at("iz")),
                                std::stof(tokens.at("value")) });
    }
    return rows;
}

std::size_t linearIndex(int ix, int iy, int iz, int ny, int nz)
{
    return static_cast<std::size_t>((ix * ny + iy) * nz + iz);
}

using ComplexValue = std::array<float, 2>;

std::vector<ComplexValue> reconstructCpuComplex(const std::vector<ComplexRow>& rows, int nx, int ny, int nzHalf)
{
    std::vector<ComplexRow> filtered;
    filtered.reserve(rows.size());
    for (const auto& row : rows)
    {
        if (row.callIndex == 0)
        {
            filtered.push_back(row);
        }
    }
    const auto expectedCount = static_cast<std::size_t>(nx * ny * nzHalf);
    if (filtered.size() == 2 * expectedCount)
    {
        filtered.resize(expectedCount);
    }
    if (filtered.size() != expectedCount)
    {
        throw std::runtime_error("Unexpected CPU complex row count");
    }

    std::vector<ComplexValue> data(filtered.size());
    for (std::size_t n = 0; n < filtered.size(); ++n)
    {
        const int x = static_cast<int>(n % nx);
        const int tmp = static_cast<int>(n / nx);
        const int z = tmp % nzHalf;
        const int y = tmp / nzHalf;
        const auto& row = filtered[n];
        auto&       out = data[linearIndex(x, y, z, ny, nzHalf)];
        out[0]          = row.re;
        out[1]          = row.im;
    }
    return data;
}

std::vector<ComplexValue> reconstructGpuComplex(const std::vector<ComplexRow>& rows, int nx, int ny, int nzHalf)
{
    const auto expectedCount = static_cast<std::size_t>(nx * ny * nzHalf);
    std::vector<ComplexRow> filtered;
    filtered.reserve(rows.size());
    for (const auto& row : rows)
    {
        if (row.callIndex == 0)
        {
            filtered.push_back(row);
        }
    }
    if (filtered.size() == 2 * expectedCount)
    {
        filtered.resize(expectedCount);
    }
    if (filtered.size() != expectedCount)
    {
        throw std::runtime_error("Unexpected GPU complex row count");
    }

    std::vector<ComplexValue> data(expectedCount);
    for (const auto& row : filtered)
    {
        auto& out = data[linearIndex(row.ix, row.iy, row.iz, ny, nzHalf)];
        out[0]    = row.re;
        out[1]    = row.im;
    }
    return data;
}

std::vector<float> reconstructReal(const std::vector<RealRow>& rows, int nx, int ny, int nz)
{
    std::vector<float> data(static_cast<std::size_t>(nx * ny * nz));
    for (const auto& row : rows)
    {
        if (row.callIndex != 0)
        {
            continue;
        }
        data[linearIndex(row.ix, row.iy, row.iz, ny, nz)] = row.value;
    }
    return data;
}

std::vector<float> inverseFft3d(const std::vector<ComplexValue>& complexGrid, int nx, int ny, int nz)
{
    const int nzHalf = nz / 2 + 1;
    if (complexGrid.size() != static_cast<std::size_t>(nx * ny * nzHalf))
    {
        throw std::runtime_error("Unexpected complex grid size");
    }

    std::vector<ComplexValue> in = complexGrid;
    std::vector<float>         out(static_cast<std::size_t>(nx * ny * nz));
    fftwf_plan plan =
            fftwf_plan_dft_c2r_3d(nx,
                                  ny,
                                  nz,
                                  reinterpret_cast<fftwf_complex*>(in.data()),
                                  out.data(),
                                  FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    if (plan == nullptr)
    {
        throw std::runtime_error("Failed to create FFTW plan");
    }
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    return out;
}

Summary summarize(const std::vector<float>& a, const std::vector<float>& b, int nx, int ny, int nz)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error("Size mismatch in summarize()");
    }
    Summary summary;
    double  sumSq   = 0.0;
    double  sumAbs  = 0.0;
    auto    count   = static_cast<double>(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        const double absd = std::abs(diff);
        if (absd > summary.maxAbs)
        {
            summary.maxAbs = absd;
            const int ix   = static_cast<int>(i / (ny * nz));
            const int rem  = static_cast<int>(i % (ny * nz));
            const int iy   = rem / nz;
            const int iz   = rem % nz;
            summary.ix     = ix;
            summary.iy     = iy;
            summary.iz     = iz;
            summary.a      = a[i];
            summary.b      = b[i];
            summary.diff   = diff;
        }
        sumSq += diff * diff;
        sumAbs += absd;
    }
    summary.rmse    = std::sqrt(sumSq / count);
    summary.meanAbs = sumAbs / count;
    return summary;
}

void printSummary(const std::string& label, const Summary& s)
{
    std::cout << label << " max_abs=" << std::setprecision(17) << s.maxAbs << " worst=(" << s.ix << ","
              << s.iy << "," << s.iz << ") a=" << s.a << " b=" << s.b << " diff=" << s.diff
              << " rmse=" << s.rmse << " mean_abs=" << s.meanAbs << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::cerr << "usage: " << argv[0]
                  << " CPU_COMPLEX_TSV GPU_COMPLEX_TSV CPU_REAL_TSV GPU_REAL_TSV\n";
        return 2;
    }

    constexpr int nx     = 32;
    constexpr int ny     = 32;
    constexpr int nz     = 32;
    constexpr int nzHalf = nz / 2 + 1;

    const auto cpuComplexRows = loadComplexRows(argv[1]);
    const auto gpuComplexRows = loadComplexRows(argv[2]);
    const auto cpuRealRows    = loadRealRows(argv[3]);
    const auto gpuRealRows    = loadRealRows(argv[4]);

    const auto cpuComplex = reconstructCpuComplex(cpuComplexRows, nx, ny, nzHalf);
    const auto gpuComplex = reconstructGpuComplex(gpuComplexRows, nx, ny, nzHalf);
    const auto cpuReal    = reconstructReal(cpuRealRows, nx, ny, nz);
    const auto gpuReal    = reconstructReal(gpuRealRows, nx, ny, nz);

    const auto cpuFftw = inverseFft3d(cpuComplex, nx, ny, nz);
    const auto gpuFftw = inverseFft3d(gpuComplex, nx, ny, nz);

    printSummary("fftw(cpu_complex) vs cpu_trace", summarize(cpuFftw, cpuReal, nx, ny, nz));
    printSummary("fftw(gpu_complex) vs gpu_trace", summarize(gpuFftw, gpuReal, nx, ny, nz));
    printSummary("fftw(cpu_complex) vs fftw(gpu_complex)", summarize(cpuFftw, gpuFftw, nx, ny, nz));
    printSummary("cpu_trace vs gpu_trace", summarize(cpuReal, gpuReal, nx, ny, nz));

    const std::vector<std::tuple<int, int, int>> keys = {
        { 16, 21, 20 }, { 16, 21, 21 }, { 16, 21, 22 }, { 16, 21, 23 }, { 17, 21, 20 }, { 17, 22, 20 }
    };
    for (const auto& [ix, iy, iz] : keys)
    {
        const auto index = linearIndex(ix, iy, iz, ny, nz);
        std::cout << "key=(" << ix << "," << iy << "," << iz << ") "
                  << "fftw_cpu_minus_gpu=" << static_cast<double>(cpuFftw[index]) - static_cast<double>(gpuFftw[index])
                  << " trace_cpu_minus_gpu=" << static_cast<double>(cpuReal[index]) - static_cast<double>(gpuReal[index])
                  << "\n";
    }

    return 0;
}
