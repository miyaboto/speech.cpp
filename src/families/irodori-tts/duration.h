#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "graph.h"
#include "model-file.h"

namespace irodori {

/**
 * What a request asks of the length of its speech. The runtime takes `seconds` and `duration_scale`;
 * `speed` is Irodori-TTS-Server's, which divides both by it to serve OpenAI's speech API.
 * `max_seconds_per_char` is speech.cpp's: an upper bound on a predicted length from the text's characters.
 */
struct LengthOptions {
    /** The length in seconds, or 0 for the length the duration predictor gives. */
    double seconds = 0;
    /** The factor of the predicted length, above 0. It cannot be given with `seconds`. */
    double duration_scale = 1;
    /** The speaking rate, which divides the length, fixed or predicted. */
    double speed = 1;
    /**
     * The most seconds of predicted speech per Unicode character of the normalized text, at speed 1, or 0 for no
     * text-length bound. Applied only while the text is at most `kDurationCapMaxChars` characters. Fixed `seconds`
     * ignore it. The C API's default for Irodori-TTS is 0.16.
     */
    double max_seconds_per_char = 0;

    /** Whether the length is fixed, so that the duration predictor does not run. */
    bool fixed() const { return seconds > 0; }
};

/** How long a synthesis is: the latent's frames, and the samples its audio is cut to before the cut where the latent goes flat. */
struct Length {
    int frames = 0;
    int64_t samples = 0;
};

/**
 * The Unicode scalar values in normalized text that the text-length duration cap counts: every well-formed code
 * point, so that punctuation and emoji that steer the speech lengthen the bound with the rest of the text.
 */
size_t duration_text_chars(const std::string & normalized);

/**
 * The most characters for which `max_seconds_per_char` still bounds the prediction. Longer text keeps the predictor's
 * length, so an ordinary sentence is not shortened when the per-character bound is tight enough for short chunks.
 */
constexpr size_t kDurationCapMaxChars = 16;

/**
 * The most frames `max_seconds_per_char` allows for `chars` characters at the options' scale and speed, or nothing
 * when the cap is off or the text is longer than `kDurationCapMaxChars`. The bound is at least one frame.
 */
std::optional<int> duration_text_cap_frames(const LengthOptions & options, size_t chars, int hop, int sample_rate);

/**
 * The duration predictor of v4.1: a stack of SwiGLU blocks over the text condition's tokens, each modulated by the
 * speaker's summary and by the mean of the caption condition's tokens or, without a caption, the learned null caption.
 * Each token contributes softplus(output) frames.
 */
class DurationPredictor {
public:
    /** The predictor of the model file `m`, which outlives it, with the bounds of the length and the speed it gives. */
    explicit DurationPredictor(const ModelFile & m);

    /**
     * The predicted frames summed over the tokens, [1], for a text condition [text_dim, n], a speaker summary
     * [speaker_dim] and a caption condition [caption_dim, tokens], or the null caption where it is null.
     */
    ggml_tensor * build(Graph & g, ggml_tensor * text_state, ggml_tensor * speaker_summary, ggml_tensor * caption_state = nullptr) const;

    /**
     * Throws, naming the option at fault, unless the options are ones the runtime takes: a speed within the model's
     * bounds, a duration scale above 0, a max_seconds_per_char of 0 or more, and seconds within the model's bounds of
     * the length that, divided by the speed, still lie within them. The runtime clamps such seconds into the bounds
     * and ignores a duration scale given with seconds; both are refused here instead.
     */
    void check(const LengthOptions & options) const;

    /**
     * The length the runtime synthesizes for checked options and, when they fix no length, the predicted sum.
     * Fixed seconds give int(seconds * sample rate) samples in the frames that hold them. A prediction goes
     * through the runtime's float32 log1p and expm1, the scale in double precision and its rounding to the
     * nearest frame with half to even, and gives whole frames of samples. When `text_chars` is set, at most
     * `kDurationCapMaxChars`, and `max_seconds_per_char` is above 0, a longer prediction is cut to that many characters
     * times the option, times the duration scale and over the speed, and not below the shortest length: short Japanese
     * with a reference voice is often predicted far longer than the text, and the extra frames are filled with speech
     * that is not in the text (v4.1-Small-MF, 2026-10-11). A prediction past the model's longest length throws, naming
     * the text, unless the scale and the speed bring it within the bounds; the runtime bounds it instead. At a scale
     * and a speed of 1 a prediction under the shortest length is raised to it, as the runtime raises it; one that a
     * scale or a speed takes outside the bounds throws, naming the scale, or the speed when the scale is 1.
     */
    Length length(const LengthOptions & options, float predicted_sum, size_t text_chars = 0) const;

private:
    const ModelFile & m_;
    int layers_, sample_rate_, hop_, min_frames_, max_frames_;
    double min_seconds_, max_seconds_, min_speed_, max_speed_;
    float eps_;
};

}  // namespace irodori
