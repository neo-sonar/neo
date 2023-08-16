#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>
#include <neo/fft/convolution/multiply_elements_add_columns.hpp>
#include <neo/fft/convolution/overlap_add.hpp>
#include <neo/fft/convolution/overlap_save.hpp>
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

template<typename Filter, typename Overlap>
struct uniform_partitioned_convolver
{
    using filter_type  = Filter;
    using value_type   = typename Filter::value_type;
    using overlap_type = Overlap;

    uniform_partitioned_convolver() = default;

    auto filter(in_matrix auto filter, auto... args) -> void;
    auto operator()(in_vector auto block) -> void;

private:
    Overlap _overlap{1, 1};
    Filter _filter;
    stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 1>> _accumulator;
    stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 2>> _fdl;
};

template<typename Filter, typename Overlap>
auto uniform_partitioned_convolver<Filter, Overlap>::filter(in_matrix auto filter, auto... args) -> void
{
    _overlap     = Overlap{filter.extent(1) - 1, filter.extent(1) - 1};
    _fdl         = stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 2>>{filter.extents()};
    _accumulator = stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 1>>{filter.extent(1)};
    _filter.filter(filter, args...);
}

template<typename Filter, typename Overlap>
auto uniform_partitioned_convolver<Filter, Overlap>::operator()(in_vector auto block) -> void
{
    _overlap(block, [this](inout_vector auto inout) {
        auto const fdl         = _fdl.to_mdspan();
        auto const accumulator = _accumulator.to_mdspan();

        shift_rows_up(fdl);
        copy(inout, stdex::submdspan(fdl, 0, stdex::full_extent));
        _filter(fdl, accumulator);
        copy(accumulator, inout);
    });
}

template<typename Float>
using upols_convolver = uniform_partitioned_convolver<dense_filter<Float>, overlap_save<Float>>;

template<typename Float>
using upola_convolver = uniform_partitioned_convolver<dense_filter<Float>, overlap_add<Float>>;

}  // namespace neo::fft
