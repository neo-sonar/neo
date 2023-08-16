#pragma once

#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>
#include <neo/fft/convolution/multiply_elements_add_columns.hpp>
#include <neo/fft/convolution/overlap_add.hpp>
#include <neo/fft/convolution/overlap_save.hpp>
#include <neo/fft/convolution/uniform_partitioned_convolver.hpp>
#include <neo/math/complex.hpp>

namespace neo::fft {

template<typename Float>
struct dense_filter
{
    using value_type = Float;

    dense_filter() = default;

    auto filter(in_matrix auto filter) -> void { _filter = filter; }

    auto operator()(in_matrix auto fdl, out_vector auto accumulator) -> void
    {
        multiply_elements_add_columns(fdl, _filter, accumulator);
    }

private:
    stdex::mdspan<std::complex<Float> const, stdex::dextents<size_t, 2>> _filter;
};

template<typename Float>
using upols_convolver = uniform_partitioned_convolver<dense_filter<Float>, overlap_save<Float>>;

template<typename Float>
using upola_convolver = uniform_partitioned_convolver<dense_filter<Float>, overlap_add<Float>>;

}  // namespace neo::fft
