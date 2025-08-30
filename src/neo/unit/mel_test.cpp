// SPDX-License-Identifier: MIT

#include "mel.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEMPLATE_TEST_CASE("neo/unit: mel_to_hertz(hertz_to_mel)", "", float, double)
{
    using Float = TestType;

    static constexpr auto const margin = [] {
        if constexpr (std::same_as<Float, float>) {
            return 0.0005;
        } else {
            return 0.0000001;
        }
    }();

    static constexpr auto const roundtrip = [](Float hertz) {
        auto const mel = neo::hertz_to_mel(hertz);
        return neo::mel_to_hertz(mel);
    };

    REQUIRE_THAT(roundtrip(Float(55)), Catch::Matchers::WithinAbs(55.0, margin));
    REQUIRE_THAT(roundtrip(Float(110)), Catch::Matchers::WithinAbs(110.0, margin));
    REQUIRE_THAT(roundtrip(Float(220)), Catch::Matchers::WithinAbs(220.0, margin));
    REQUIRE_THAT(roundtrip(Float(440)), Catch::Matchers::WithinAbs(440.0, margin));
    REQUIRE_THAT(roundtrip(Float(880)), Catch::Matchers::WithinAbs(880.0, margin));
}

TEMPLATE_TEST_CASE("neo/unit: mel_frequencies ", "", float, double)
{
    using Float = TestType;

    SECTION("empty")
    {
        auto buffer = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{0};
        neo::mel_frequencies(buffer.to_mdspan(), Float(0), Float(11'025));
    }

    auto const fmin   = GENERATE(as<Float>{}, 0, 55, 110);
    auto const fmax   = GENERATE(as<Float>{}, 11'025, 22'050, 44'100);
    auto const n_mels = GENERATE(as<std::size_t>{}, 1, 2, 33, 128);

    CAPTURE(n_mels);
    CAPTURE(fmin);
    CAPTURE(fmax);

    auto buffer = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{n_mels};
    neo::mel_frequencies(buffer.to_mdspan(), fmin, fmax);

    REQUIRE(buffer(0) == Catch::Approx(fmin));

    if (n_mels > 1zu) {
        REQUIRE(buffer(n_mels - 1) == Catch::Approx(fmax));
    }
}

TEMPLATE_TEST_CASE("neo/unit: mel_frequencies vs. librosa", "", float, double)
{
    using Float = TestType;

    // import librosa
    //
    // print("{")
    // for x in librosa.mel_frequencies(n_mels=40, htk=True):
    //     print(f"Float({x}),")
    // print("}")

    static constexpr auto expected = std::array{
        Float(0.0),
        Float(52.45933632029549),
        Float(108.85007545082802),
        Float(169.46684422335863),
        Float(234.6263493668718),
        Float(304.6690322172243),
        Float(379.9608474338946),
        Float(460.89517501716034),
        Float(547.894875615494),
        Float(641.4144998616176),
        Float(741.94266328042),
        Float(850.0045991770061),
        Float(966.164902843051),
        Float(1091.0304814192168),
        Float(1225.2537248258907),
        Float(1369.5359143295234),
        Float(1524.6308865534415),
        Float(1691.3489720766192),
        Float(1870.5612291985624),
        Float(2063.203994990619),
        Float(2270.283777411758),
        Float(2492.882514048846),
        Float(2732.1632249569593),
        Float(2989.3760891343404),
        Float(3265.864976379991),
        Float(3563.074468661131),
        Float(3882.55740767536),
        Float(4225.983008041506),
        Float(4595.145578508522),
        Float(4991.9738967483845),
        Float(5418.541286713908),
        Float(5877.076451212964),
        Float(6369.975116296503),
        Float(6899.8125482992455),
        Float(7469.35700893129),
        Float(8081.584218719991),
        Float(8739.69290436975),
        Float(9447.121511270665),
        Float(10207.566168474454),
        Float(11024.999999999998),
    };

    auto buffer = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{expected.size()};
    neo::mel_frequencies(buffer.to_mdspan(), Float(0), Float(11'025));

    REQUIRE(buffer.extent(0) == 40UL);
    for (auto i{0U}; i < expected.size(); ++i) {
        CAPTURE(i);
        REQUIRE(buffer(i) == Catch::Approx(expected[i]));
    }
}
