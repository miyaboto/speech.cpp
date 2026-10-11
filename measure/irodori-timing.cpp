// Times Irodori-TTS on the CPU: for each text, the median over the runs of the first audio, the total time, the text
// and duration encoder, the sampler and the codec, and the real-time factor. One synthesis of the first text goes
// first, untimed, so that the times leave out the compilation of a GPU's kernels. It prints one JSON object per line.
//
// usage: irodori-timing <model.gguf> <gpu|cpu|device name> <runs> [--threads N] [--steps N] [--speed F] [--seed N]
//                       [--voice none|PATH] [--] <text>...

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "irodori-tts/synthesizer.h"

using namespace irodori;

namespace {

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

std::string json_escape(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out += '\\', out += (char) c;
        else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            out += buf;
        } else out += (char) c;
    }
    return out + "\"";
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 5) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <gpu|cpu|device name> <runs> [--threads N] [--steps N] [--speed F] "
                     "[--seed N] [--voice none|PATH] [--] <text>...\n",
                     args[0].c_str());
        return 2;
    }
    try {
        int threads = 0, steps = 0, runs = std::stoi(args[3]);
        double speed = 1.0;
        uint64_t seed = 7;
        std::string voice_path = "none";
        size_t i = 4;
        while (i < args.size() && args[i].rfind("--", 0) == 0) {
            if (args[i] == "--") {
                i++;
                break;
            }
            if (i + 1 >= args.size()) throw std::runtime_error("missing value for " + args[i]);
            if (args[i] == "--threads") threads = std::stoi(args[++i]);
            else if (args[i] == "--steps") steps = std::stoi(args[++i]);
            else if (args[i] == "--speed") speed = std::stod(args[++i]);
            else if (args[i] == "--seed") seed = (uint64_t) std::stoull(args[++i]);
            else if (args[i] == "--voice") voice_path = args[++i];
            else throw std::runtime_error("unknown flag " + args[i]);
            i++;
        }
        if (i >= args.size()) throw std::runtime_error("give at least one text");
        if (runs < 1) throw std::runtime_error("runs must be 1 or more");
        ggml_backend_dev_t device = find_device(args[2]);
        const int use_threads = threads > 0 ? threads : default_threads();
        ggml_backend_t backend = start_device(device, use_threads);
        Synthesizer synth(args[1], backend);
        Voice voice;
        if (voice_path != "none") voice = synth.load_voice(voice_path);
        else if (!synth.has_null_speaker()) throw std::runtime_error("this file has no voice none; give --voice PATH");

        auto speak = [&](const std::string & text, Stats * stats) {
            Request r;
            r.text = text;
            r.seed = seed;
            r.steps = steps;
            r.length.speed = (float) speed;
            size_t samples = 0;
            synth.synthesize(
                r, voice,
                [&](const float * /*s*/, size_t n) {
                    samples += n;
                    return true;
                },
                stats);
            return samples;
        };

        speak(args[i], nullptr);
        for (; i < args.size(); i++) {
            const std::string & text = args[i];
            std::vector<double> first, total, text_t, sampling, codec, rtf;
            Stats last;
            size_t samples = 0;
            for (int r = 0; r < runs; r++) {
                Stats st;
                const auto start = std::chrono::steady_clock::now();
                samples = speak(text, &st);
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                const double audio = (double) samples / synth.sample_rate();
                first.push_back(st.first_audio);
                total.push_back(elapsed);
                text_t.push_back(st.text);
                sampling.push_back(st.sampling);
                codec.push_back(st.codec);
                rtf.push_back(elapsed / audio);
                last = st;
            }
            std::printf(
                "{\"sentence\":%s,\"device\":%s,\"threads\":%d,\"steps\":%d,\"speed\":%g,\"seed\":%llu,\"runs\":%d,"
                "\"samples\":%zu,\"audio_s\":%.3f,\"frames\":%d,\"tokens\":%d,"
                "\"first_audio\":%.3f,\"total\":%.3f,\"text_s\":%.3f,\"sampling_s\":%.3f,\"codec_s\":%.3f,\"rtf\":%.3f}\n",
                json_escape(text).c_str(), json_escape(ggml_backend_name(backend)).c_str(), use_threads,
                steps > 0 ? steps : (int) synth.model().u32("irodori-tts.sampler.default_steps"), speed,
                (unsigned long long) seed, runs, samples, (double) samples / synth.sample_rate(), last.frames, last.tokens,
                median(first), median(total), median(text_t), median(sampling), median(codec), median(rtf));
            std::fflush(stdout);
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
