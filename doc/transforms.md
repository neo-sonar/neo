# Transforms

- [FFT](#fft)
- [DFT](#dft)
- [RFFT](#rfft)
- [Resources](#resources)
  - [DCT](#dct)
  - [DSP](#dsp)

```cpp
#include <neo/fft.hpp>

namespace neo::fft {
    enum struct direction : int
    {
        forward  = -1,
        backward = 1,
    };

    enum struct order : std::size_t
    {
    };

    template<std::integral Int>
    auto next_order(Int size) noexcept -> order;
}
```

## FFT

```cpp
namespace neo::fft {
    template<typename Complex>
    struct fft_plan
    {
        using value_type = Complex;
        using size_type  = std::size_t;

        explicit fft_plan(fft::order order);

        static constexpr auto max_order() noexcept -> fft::order;
        static constexpr auto max_size() noexcept -> size_type;

        auto order() const noexcept -> fft::order;
        auto size() const noexcept -> size_type;

        template<inout_vector_of<Complex> InOutVec>
        auto operator()(InOutVec x, direction dir) -> void;

        template<in_vector_of<Complex> InVec, out_vector_of<Complex> OutVec>
        auto operator()(InVec in, OutVec out, direction dir) -> void;
    };

    template<typename Plan, inout_vector InOutVec>
    auto fft(Plan& plan, InOutVec x);

    template<typename Plan, in_vector InVec, out_vector OutVec>
    auto fft(Plan& plan, InVec in, OutVec out);

    template<typename Plan, inout_vector InOutVec>
    auto ifft(Plan& plan, InOutVec x);

    template<typename Plan, in_vector InVec, out_vector OutVec>
    auto ifft(Plan& plan, InVec in, OutVec out);
}
```

## DFT

```cpp
namespace neo::fft {
    template<typename Complex>
    struct dft_plan
    {
        using value_type = Complex;
        using size_type  = std::size_t;

        explicit dft_plan(size_type size);

        static constexpr auto max_size() noexcept -> size_type;
        auto size() const noexcept -> size_type;

        template<inout_vector_of<Complex> InOutVec>
        auto operator()(InOutVec x, direction dir) -> void;

        template<in_vector_of<Complex> InVec, out_vector_of<Complex> OutVec>
        auto operator()(InVec in, OutVec out, direction dir) -> void;
    };

    template<typename Plan, inout_vector InOutVec>
    auto dft(Plan& plan, InOutVec x);

    template<typename Plan, in_vector InVec, out_vector OutVec>
    auto dft(Plan& plan, InVec in, OutVec out);

    template<typename Plan, inout_vector InOutVec>
    auto idft(Plan& plan, InOutVec x);

    template<typename Plan, in_vector InVec, out_vector OutVec>
    auto idft(Plan& plan, InVec in, OutVec out);
}
```

## RFFT

```cpp
namespace neo::fft {
    template<std::floating_point Float>
    struct rfft_plan
    {
        using real_type    = Float;
        using complex_type = std::complex<Float>;
        using size_type    = std::size_t;

        explicit rfft_plan(fft::order order);

        static constexpr auto max_order() noexcept -> fft::order;
        static constexpr auto max_size() noexcept -> size_type;

        auto order() const noexcept -> fft::order;
        auto size() const noexcept -> size_type;

        template<in_vector_of<Float> InVec, out_vector_of<complex_type> OutVec>
        auto operator()(InVec in, OutVec out) -> void;

        template<in_vector_of<complex_type> InVec, out_vector_of<Float> OutVec>
        auto operator()(InVec in, OutVec out) -> void;
    };

    template<typename Plan, in_vector InVec, out_vector OutVec>
        requires(std::floating_point<value_type_t<InVec>> and neo::complex<value_type_t<OutVec>>)
    auto rfft(Plan& plan, InVec input, OutVec output);

    template<typename Plan, in_vector InVec, out_vector OutVec>
        requires(neo::complex<value_type_t<InVec>> and std::floating_point<value_type_t<OutVec>>)
    auto irfft(Plan& plan, InVec input, OutVec output);
}
```

## Resources

- [COMPUTING THE FAST FOURIER TRANSFORMON SIMD MICROPROCESSOR](https://www.cs.waikato.ac.nz/~ihw/PhD_theses/Anthony_Blake.pdf)
- [Notes on FFTs: for implementers](https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/)
- [OTFFT Library](http://wwwa.pikara.ne.jp/okojisan/otfft-en/stockham1.html)
- [Discrete fourier transform and convolution -- scaling factor?](https://mathematica.stackexchange.com/questions/206788/discrete-fourier-transform-and-convolution-scaling-factor)
- [FXT Library](https://www.jjj.de/fxt/demo/arith/index.html)
- [FXT Library: Algorithms for programmers ideas and source code](http://dsp-book.narod.ru/fxtbook.pdf)

### DCT

- [Fast Cosine Transform via FFT](https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft)
- [Fast discrete cosine transform algorithms](https://www.nayuki.io/page/fast-discrete-cosine-transform-algorithms)

### DSP

- [musicinformationretrieval.com](https://musicinformationretrieval.com/index.html)
- [Audio Signal Processing for Machine Learning](https://youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&si=51JQNk_IuZSZITxX)
- [AudioLabsErlangen](https://www.youtube.com/@AudioLabsErlangen/videos)
- [CMUL via FMA](https://stackoverflow.com/questions/30089859/using-fma-fused-multiply-instructions-for-complex-multiplication)
