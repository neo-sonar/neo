// SPDX-License-Identifier: MIT

#pragma once

namespace neo::convolution {

enum struct convolution_method
{
    automatic,
    direct,
    fft,
    ola,
    ols,
    upola,
    upols,
};

}  // namespace neo::convolution
