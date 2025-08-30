// SPDX-License-Identifier: MIT

#pragma once

#include <neo/bit/bit_log2.hpp>
#include <neo/complex/complex.hpp>
#include <neo/complex/split_complex.hpp>
#include <neo/container/mdspan.hpp>

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace neo::fft {

/// Reorder input using bit reversal permutation.
/// \ingroup neo-fft
struct bitrevorder_plan
{
    explicit bitrevorder_plan(std::size_t order) : _table{make(std::size_t(1) << order)} {}

    template<inout_vector Vec>
        requires complex<value_type_t<Vec>>
    auto operator()(Vec x) -> void
    {
        for (auto i{0zu}; i < _table.size(); ++i) {
            if (i < _table[i]) {
                std::swap(x[i], x[_table[i]]);
            }
        }
    }

    template<inout_vector Vec>
        requires std::floating_point<value_type_t<Vec>>
    auto operator()(Vec x) -> void
    {
        for (auto i{0zu}; i < _table.size(); ++i) {
            if (i < _table[i]) {
                auto const src_re = i * 2zu;
                auto const src_im = src_re + 1zu;

                auto const dest_re = _table[i] * 2zu;
                auto const dest_im = dest_re + 1zu;

                std::swap(x[src_re], x[dest_re]);
                std::swap(x[src_im], x[dest_im]);
            }
        }
    }

    template<inout_vector Vec>
    auto operator()(split_complex<Vec> x) -> void
    {
        for (auto i{0zu}; i < _table.size(); ++i) {
            auto const other_idx = _table[i];
            if (i < other_idx) {
                std::swap(x.real[i], x.real[other_idx]);
                std::swap(x.imag[i], x.imag[other_idx]);
            }
        }
    }

private:
    [[nodiscard]] static auto make(std::size_t size) -> std::vector<std::uint32_t>
    {
        auto const order = bit_log2(size);
        auto table       = std::vector<std::uint32_t>(size, 0);
        for (auto i{0zu}; i < size; ++i) {
            for (auto j{0zu}; j < order; ++j) {
                table[i] |= ((i >> j) & 1) << (order - 1 - j);
            }
        }
        return table;
    }

    std::vector<std::uint32_t> _table;
};

template<inout_vector Vec>
    requires complex<typename Vec::value_type>
constexpr auto bitrevorder(Vec x) -> void
{
    std::size_t j = 0;
    for (std::size_t i = 0; i < x.size() - 1zu; ++i) {
        if (i < j) {
            std::swap(x[i], x[j]);
        }
        std::size_t k = x.size() / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
}

template<inout_vector Vec>
    requires std::floating_point<typename Vec::value_type>
constexpr auto bitrevorder(Vec x) -> void
{
    auto const nn = static_cast<int>(x.extent(0));
    auto const n  = nn / 2;

    auto j = 1;
    for (auto i{1}; i < nn; i += 2) {
        if (j > i) {
            std::swap(x[j - 1], x[i - 1]);
            std::swap(x[j], x[i]);
        }
        auto m = n;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

}  // namespace neo::fft
