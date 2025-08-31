#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numbers>
#include <span>
#include <vector>

void fft(float* data, int n, int isign)
{
    int nn, mmax, m, istep, i;
    float wtemp, wr, wpr, wpi, wi, theta;

    nn = n << 1;

    {
        auto j = 1;
        for (i = 1; i < nn; i += 2) {
            if (j > i) {
                std::swap(data[j - 1], data[i - 1]);
                std::swap(data[j], data[i]);
            }
            m = n;
            while (m >= 2 && j > m) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }
    }

    mmax = 2;
    while (nn > mmax) {
        istep = mmax << 1;
        theta = isign * (6.28318530717959F / mmax);
        wtemp = std::sin(0.5F * theta);
        wpr   = -2.0F * wtemp * wtemp;
        wpi   = std::sin(theta);
        wr    = 1.0F;
        wi    = 0.0F;
        for (m = 1; m < mmax; m += 2) {
            for (i = m; i <= nn; i += istep) {
                auto const j   = i + mmax;
                auto const jn1 = j - 1;
                auto const in1 = i - 1;

                auto const tempr = wr * data[jn1] - wi * data[j];
                auto const tempi = wr * data[j] + wi * data[jn1];

                data[jn1] = data[in1] - tempr;
                data[j]   = data[i] - tempi;

                data[in1] += tempr;
                data[i] += tempi;
            }

            wr = (wtemp = wr) * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }
        mmax = istep;
    }
}

void fft(std::vector<float>& data, int isign) { fft(&data[0], data.size() / 2, isign); }

void rfft(std::vector<float>& data, int isign)
{
    auto const n  = data.size();
    auto const c1 = 0.5F;
    auto const c2 = isign == 1 ? -0.5F : 0.5F;

    float h1r;
    float h1i;
    float h2r;
    float h2i;
    float wr;
    float wi;
    float wtemp;
    float theta = float(std::numbers::pi) / static_cast<float>(n >> 1);

    if (isign == 1) {
        fft(data, 1);
    } else {
        theta = -theta;
    }

    wtemp          = std::sin(0.5F * theta);
    auto const wpr = -2.0F * wtemp * wtemp;
    auto const wpi = std::sin(theta);
    wr             = 1.0F + wpr;
    wi             = wpi;

    for (auto i = 1; i < (n >> 2); i++) {
        auto const i1 = i + i;
        auto const i2 = 1 + i1;
        auto const i3 = n - i1;
        auto const i4 = 1 + i3;

        h1r = c1 * (data[i1] + data[i3]);
        h1i = c1 * (data[i2] - data[i4]);
        h2r = -c2 * (data[i2] + data[i4]);
        h2i = c2 * (data[i1] - data[i3]);

        data[i1] = h1r + wr * h2r - wi * h2i;
        data[i2] = h1i + wr * h2i + wi * h2r;
        data[i3] = h1r - wr * h2r + wi * h2i;
        data[i4] = -h1i + wr * h2i + wi * h2r;

        wr = (wtemp = wr) * wpr - wi * wpi + wr;
        wi = wi * wpr + wtemp * wpi + wi;
    }
    if (isign == 1) {
        data[0] = (h1r = data[0]) + data[1];
        data[1] = h1r - data[1];
    } else {
        data[0] = c1 * ((h1r = data[0]) + data[1]);
        data[1] = c1 * (h1r - data[1]);
        fft(data, -1);
    }
}

auto main(int argc, char** argv) -> int
{
    auto N = 1024zu;
    if (argc == 2) {
        N = std::stoul(argv[1]);
    }

    auto runs = std::vector(10000zu, std::pair{0, 0.0});

    auto buffer = std::vector(N, 0.0F);
    buffer[0]   = 1.0F;

    for (auto i{0zu}; i < runs.size(); ++i) {

        auto const start = std::chrono::system_clock::now();
        rfft(buffer, 1);
        rfft(buffer, -1);
        auto const stop    = std::chrono::system_clock::now();
        auto const elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(stop - start);
        auto const size    = static_cast<double>(buffer.size());
        auto const mflops  = static_cast<int>(std::lround(5.0 * size * std::log2(size) / elapsed.count()));

        runs[i] = {mflops, buffer[0] * (2.0 / static_cast<double>(buffer.size()))};
    }

    std::printf(
        "MFLOPS: %d min, %d max\n",
        std::min_element(runs.begin(), runs.end(), [](auto l, auto r) { return l.first < r.first; })->first,
        std::max_element(runs.begin(), runs.end(), [](auto l, auto r) { return l.first < r.first; })->first
    );

    auto buf = std::vector(16zu, 0.0F);
    buf[0]   = 1.0F;

    // rfft
    rfft(buf, 1);
    for (auto val :
         std::span<std::complex<float>>{reinterpret_cast<std::complex<float>*>(buf.data()), buf.size() / 2U}) {
        std::printf("(%.1f, %.1f) ", val.real(), val.imag());
    }
    std::puts("\n");

    // irfft
    rfft(buf, -1);
    for (auto val : buf) {
        std::printf("%.1f ", val * (2.0F / static_cast<float>(buf.size())));
    }
    std::puts("\n");

    return 0;
}
