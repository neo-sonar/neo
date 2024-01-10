#include <complex>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

enum direction
{
    forward,
    backward,
};

// https://www.cs.waikato.ac.nz/~ihw/PhD_theses/Anthony_Blake.pdf
template<std::floating_point Float>
struct split_radix_fft_plan
{
    explicit split_radix_fft_plan(int n)
        : _size{static_cast<size_t>(n)}
        , _lut1_f(static_cast<size_t>(n / 4))
        , _lut3_f(static_cast<size_t>(n / 4))
        , _lut1_b(static_cast<size_t>(n / 4))
        , _lut3_b(static_cast<size_t>(n / 4))
    {
        auto const w = [n](auto i) {
            auto const sign   = Float(-1);
            auto const two_pi = static_cast<Float>(std::numbers::pi * 2.0);
            auto const angle  = sign * two_pi * Float(i) / Float(n);
            return std::polar(Float(1), angle);
        };

        for (auto i{0}; i < n / 4; i++) {
            _lut1_f[i] = w(i);
            _lut3_f[i] = w(3 * i);
            _lut1_b[i] = std::conj(_lut1_f[i]);
            _lut3_b[i] = std::conj(_lut3_f[i]);
        }
    }

    auto operator()(std::complex<Float> const* in, std::complex<Float>* out, direction dir) -> void
    {
        if (dir == direction::forward) {
            splitfft(in, out, 0, 1, static_cast<int>(_size), _lut1_f.data(), _lut3_f.data());
        } else {
            splitfft(in, out, 0, 1, static_cast<int>(_size), _lut1_b.data(), _lut3_b.data());
        }
    }

private:
    auto splitfft(
        std::complex<Float> const* in,
        std::complex<Float>* out,
        int log2stride,
        int stride,
        int N,
        std::complex<Float> const* lut1,
        std::complex<Float> const* lut3
    ) -> void
    {
        if (N == 1) {
            out[0] = in[0];
            return;
        }

        if (N == 2) {
            out[0]     = in[0] + in[stride];
            out[N / 2] = in[0] - in[stride];
            return;
        }

        splitfft(in, out, log2stride + 1, stride << 1, N >> 1, lut1, lut3);
        splitfft(in + stride, out + N / 2, log2stride + 2, stride << 2, N >> 2, lut1, lut3);
        splitfft(in + 3 * stride, out + 3 * N / 4, log2stride + 2, stride << 2, N >> 2, lut1, lut3);

        auto const I = std::complex<Float>{Float(0), Float(1)};

        {
            auto const Uk  = out[0];
            auto const Zk  = out[0 + N / 2];
            auto const Uk2 = out[0 + N / 4];
            auto const Zdk = out[0 + 3 * N / 4];

            out[0]             = Uk + (Zk + Zdk);
            out[0 + N / 2]     = Uk - (Zk + Zdk);
            out[0 + N / 4]     = Uk2 - I * (Zk - Zdk);
            out[0 + 3 * N / 4] = Uk2 + I * (Zk - Zdk);
        }

        for (auto k = 1; k < N / 4; k++) {
            auto const Uk  = out[k];
            auto const Zk  = out[k + N / 2];
            auto const Uk2 = out[k + N / 4];
            auto const Zdk = out[k + 3 * N / 4];
            auto const w1  = lut1[k << log2stride];
            auto const w3  = lut3[k << log2stride];

            out[k]             = Uk + (w1 * Zk + w3 * Zdk);
            out[k + N / 2]     = Uk - (w1 * Zk + w3 * Zdk);
            out[k + N / 4]     = Uk2 - I * (w1 * Zk - w3 * Zdk);
            out[k + 3 * N / 4] = Uk2 + I * (w1 * Zk - w3 * Zdk);
        }
    }

    std::size_t _size;
    std::vector<std::complex<Float>> _lut1_f;
    std::vector<std::complex<Float>> _lut3_f;
    std::vector<std::complex<Float>> _lut1_b;
    std::vector<std::complex<Float>> _lut3_b;
};

auto main() -> int
{
    auto plan = split_radix_fft_plan<double>{4};
    auto in   = std::vector<std::complex<double>>(4);
    auto out  = std::vector<std::complex<double>>(4);
    in[0]     = 1.0F;

    std::printf("forward\n");
    plan(in.data(), out.data(), direction::forward);
    for (auto z : out) {
        std::printf("%f,%f\n", z.real(), z.imag());
    }

    std::printf("\nbackward\n");
    plan(out.data(), in.data(), direction::backward);
    for (auto z : in) {
        std::printf("%f,%f\n", z.real(), z.imag());
    }
    return 0;
}
