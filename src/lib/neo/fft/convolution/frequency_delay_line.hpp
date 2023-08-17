#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>

namespace neo::fft {

template<typename Complex>
struct frequency_delay_line
{
    using value_type = Complex;

    frequency_delay_line() noexcept = default;

    explicit frequency_delay_line(stdex::dextents<std::size_t, 2> extents) : _delay_lines{extents} {}

    auto operator()(in_vector auto in, auto callback) -> void
    {
        copy(in, stdex::submdspan(_delay_lines.to_mdspan(), _write_pos, stdex::full_extent));

        auto const num_lines = _delay_lines.extent(0);

        for (auto r{0UL}; r < num_lines; ++r) {
            auto const line         = stdex::submdspan(_delay_lines.to_mdspan(), r, stdex::full_extent);
            auto const filter_index = (_write_pos + num_lines - r) % num_lines;
            callback(line, filter_index);
        }

        if (++_write_pos; _write_pos >= num_lines) {
            _write_pos = 0;
        }
    }

private:
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 2>> _delay_lines{0, 0};
    std::size_t _write_pos{0};
};

}  // namespace neo::fft
