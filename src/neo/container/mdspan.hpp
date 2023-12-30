// SPDX-License-Identifier: MIT

#pragma once

#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wextra-semi"
    #pragma clang diagnostic ignored "-Wshadow"
#elif defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wextra-semi"
    #pragma GCC diagnostic ignored "-Wshadow"
#endif

#include <mdspan/mdarray.hpp>
#include <mdspan/mdspan.hpp>

#if defined(__clang__)
    #pragma clang diagnostic pop
#elif defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

#include <concepts>
#include <type_traits>

namespace stdex {
using Kokkos::default_accessor;
using Kokkos::dextents;
using Kokkos::extents;
using Kokkos::full_extent;
using Kokkos::layout_left;
using Kokkos::layout_right;
using Kokkos::layout_stride;
using Kokkos::mdspan;
using Kokkos::submdspan;
using Kokkos::Experimental::mdarray;
}  // namespace stdex

namespace neo {

template<typename T>
inline constexpr auto const is_mdspan = false;

template<typename T, typename Extents, typename Layout, typename Accessor>
inline constexpr auto const is_mdspan<stdex::mdspan<T, Extents, Layout, Accessor>> = true;

template<typename T>
concept in_vector = is_mdspan<T> && T::rank() == 1;

template<typename T>
concept out_vector =                                                                          //
    is_mdspan<T>                                                                              //
    && T::rank() == 1                                                                         //
    && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type>  //
    && T::is_always_unique();                                                                 //

template<typename T>
concept inout_vector =                                                                        //
    is_mdspan<T>                                                                              //
    && T::rank() == 1                                                                         //
    && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type>  //
    && T::is_always_unique();                                                                 //

template<typename T>
concept in_matrix = is_mdspan<T> && T::rank() == 2;

template<typename T>
concept out_matrix =                                                                          //
    is_mdspan<T>                                                                              //
    && T::rank() == 2                                                                         //
    && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type>  //
    && T::is_always_unique();                                                                 //

template<typename T>
concept inout_matrix =                                                                        //
    is_mdspan<T>                                                                              //
    && T::rank() == 2                                                                         //
    && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type>  //
    && T::is_always_unique();                                                                 //

template<typename T>
concept in_object = is_mdspan<T> && (T::rank() == 1 || T::rank() == 2);

template<typename T>
concept out_object =                                                                          //
    is_mdspan<T>                                                                              //
    && (T::rank() == 1 || T::rank() == 2)                                                     //
    && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type>  //
    && T::is_always_unique();                                                                 //

template<typename T>
concept inout_object =                                                                        //
    is_mdspan<T>                                                                              //
    && (T::rank() == 1 || T::rank() == 2)                                                     //
    && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type>  //
    && T::is_always_unique();                                                                 //

template<typename Vec, typename Value>
concept in_vector_of = in_vector<Vec> and std::same_as<Value, typename Vec::value_type>;

template<typename Vec, typename Value>
concept out_vector_of = out_vector<Vec> and std::same_as<Value, typename Vec::value_type>;

template<typename Vec, typename Value>
concept inout_vector_of = inout_vector<Vec> and std::same_as<Value, typename Vec::value_type>;

template<typename Mat, typename Value>
concept in_matrix_of = in_matrix<Mat> and std::same_as<Value, typename Mat::value_type>;

template<typename Mat, typename Value>
concept out_matrix_of = out_matrix<Mat> and std::same_as<Value, typename Mat::value_type>;

template<typename Mat, typename Value>
concept inout_matrix_of = inout_matrix<Mat> and std::same_as<Value, typename Mat::value_type>;

template<typename Obj, typename Value>
concept in_object_of = in_object<Obj> and std::same_as<Value, typename Obj::value_type>;

template<typename Obj, typename Value>
concept out_object_of = out_object<Obj> and std::same_as<Value, typename Obj::value_type>;

template<typename Obj, typename Value>
concept inout_object_of = inout_object<Obj> and std::same_as<Value, typename Obj::value_type>;

namespace detail {

[[nodiscard]] constexpr auto extents_equal(in_object auto ref, in_object auto... others) noexcept -> bool
{
    return ((ref.extents() == others.extents()) and ...);
}

template<auto Stride>
[[nodiscard]] constexpr auto strides_equal_to(in_vector auto... vecs) noexcept -> bool
{
    return ((vecs.stride(0) == Stride) and ...);
}

template<in_object First, in_object... Objs>
inline constexpr auto all_same_value_type_v
    = (std::same_as<typename First::value_type, typename Objs::value_type> and ...);

}  // namespace detail

template<typename... Objs>
inline constexpr auto has_default_accessor
    = (std::same_as<typename Objs::accessor_type, stdex::default_accessor<typename Objs::element_type>> and ...);

template<typename... Objs>
inline constexpr auto has_layout_left = (std::same_as<typename Objs::layout_type, stdex::layout_left> and ...);

template<typename... Objs>
inline constexpr auto has_layout_right = (std::same_as<typename Objs::layout_type, stdex::layout_right> and ...);

template<typename... Objs>
inline constexpr auto has_layout_left_or_right = has_layout_left<Objs...> or has_layout_right<Objs...>;

}  // namespace neo
