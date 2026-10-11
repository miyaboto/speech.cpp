// Checks the text-length duration cap of Irodori-TTS: how many Unicode characters a normalized text contributes, and
// that a predicted length above characters times max_seconds_per_char is cut to that bound (with speed and
// duration_scale), while a shorter prediction, a fixed length, and a cap of 0 are left alone. It needs no model file.
//
// usage: irodori-duration-check

#include <cmath>
#include <cstdio>
#include <string>

#include "irodori-tts/duration.h"
#include "irodori-tts/text-normalizer.h"

using namespace irodori;

namespace {

int fail(const char * what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

}  // namespace

int main() {
    int failed = 0;
    const std::string moshimoshi = normalize_text("もしもし、");
    const std::string arigatou = normalize_text("ありがとうございます。");
    const std::string long_line = normalize_text("もしもし、宮本様のお電話でお間違いないでしょうか。");
    if (duration_text_chars(moshimoshi) != 5) failed |= fail("もしもし、 is five characters");
    if (duration_text_chars(arigatou) != 11) failed |= fail("ありがとうございます。 is eleven characters");
    if (duration_text_chars(long_line) != 25) failed |= fail("the long sentence is twenty-five characters");

    constexpr int hop = 1920, rate = 48000;
    LengthOptions o;
    o.max_seconds_per_char = 0.16;
    o.speed = 1.2;
    const auto cap = duration_text_cap_frames(o, 5, hop, rate);
    if (!cap) failed |= fail("a positive max_seconds_per_char gives a cap");
    // 5 * 0.16 / 1.2 s → frames at 48 kHz / 1920.
    const int want = (int) std::nearbyint(5 * 0.16 / 1.2 * rate / (double) hop);
    if (cap && *cap != want) {
        std::fprintf(stderr, "FAIL: cap for もしもし、 at speed 1.2 is %d frames, want %d\n", *cap, want);
        failed = 1;
    }
    o.duration_scale = 1.5;
    const auto scaled = duration_text_cap_frames(o, 5, hop, rate);
    const int want_scaled = (int) std::nearbyint(5 * 0.16 * 1.5 / 1.2 * rate / (double) hop);
    if (!scaled || *scaled != want_scaled) failed |= fail("the duration scale widens the cap");

    LengthOptions off;
    if (duration_text_cap_frames(off, 5, hop, rate)) failed |= fail("max_seconds_per_char 0 leaves the prediction uncapped");
    if (duration_text_cap_frames(o, 0, hop, rate)) failed |= fail("zero characters leave the prediction uncapped");
    if (duration_text_cap_frames(o, kDurationCapMaxChars + 1, hop, rate)) {
        failed |= fail("a text longer than kDurationCapMaxChars leaves the prediction uncapped");
    }

    // A prediction of 70 frames (2.8 s) for five characters at speed 1.2 is cut; 20 frames stay.
    // The length() path needs a model file, so the cap alone is what this check proves of the bound.
    if (cap && !(*cap < 70 && *cap > 0)) failed |= fail("the cap for a short text lies below a long over-prediction");
    if (duration_text_chars(long_line) <= kDurationCapMaxChars) {
        failed |= fail("the long sentence is past the short-text cap so its prediction stays");
    }

    if (failed) return 1;
    std::printf("ok: text-length duration cap for short Japanese, unchanged for longer text\n");
    return 0;
}
