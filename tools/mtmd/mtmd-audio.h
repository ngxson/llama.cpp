#pragma once

#include "ggml.h"
#include "clip-model.h"

#include <cstdint>
#include <vector>
#include <string>

#define MTMD_INTERNAL_HEADER

struct mtmd_audio_mel {
    int64_t n_len;
    int64_t n_len_org;
    int64_t n_mel;

    std::vector<float> data;
};

struct mtmd_audio_mel_filters {
    int64_t n_mel;
    int64_t n_fft;

    std::vector<float> data;
};

// cache for audio processing, each processor instance owns its own cache
struct mtmd_audio_cache {
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;

    std::vector<float> hann_window;

    mtmd_audio_mel_filters filters;

    void fill_sin_cos_table(uint32_t n);

    void fill_hann_window(uint32_t length, bool periodic);

    // Build mel filterbank matrix [n_mel × n_fft_bins] at runtime.
    // n_fft_bins must be (N_fft / 2 + 1). Example: if N_fft=512 -> n_fft_bins=257.
    void fill_mel_filterbank_matrix(int64_t n_mel,
                                    int64_t n_fft,
                                    int   sample_rate,               // e.g. 16000
                                    float fmin             = 0.0f,   // e.g. 0.0
                                    float fmax             = -1.0f,  // e.g. sr/2; pass -1 for auto
                                    bool  slaney_area_norm = true,
                                    float scale            = 1.0f,
                                    bool  use_htk          = false
    );
};

// whisper style log-mel used by the chatterbox s3 tokenizer front-end:
// hann 400 periodic, hop 160, centered frames with reflect padding, power
// spectrum, caller-supplied mel filters [n_mel x (n_fft / 2 + 1)], log10
// clamped to 1e-10, global max - 8 dynamic range, (x + 4) / 4 scaling.
// output layout matches the audio batch entries: out[m * n_frames + t].
bool mtmd_audio_s3tok_log_mel(const float * samples, size_t n_samples,
                              const float * filters, int n_mel,
                              std::vector<float> & out, int & n_frames);

// rational 3/2 upsampler (16 kHz -> 24 kHz), windowed sinc polyphase.
// output length is exactly n_samples * 3 / 2 for even n_samples.
void mtmd_audio_upsample_3_2(const float * samples, size_t n_samples, std::vector<float> & out);

// matcha style log-mel of the chatterbox s3gen prompt features (utils/mel.py):
// 24 kHz, n_fft 1920, hop 480, hann 1920 periodic, (n_fft - hop) / 2 reflect
// padding with center false, magnitude spectrum, slaney mel 80 bins fmin 0
// fmax 8000, natural log clamped to 1e-5.
// output layout is frame major: out[t * n_mel + m], as the prompt features
// are consumed row by row at mel rate.
bool mtmd_audio_matcha_log_mel(const float * samples, size_t n_samples,
                               std::vector<float> & out, int & n_frames);

// librosa.effects.trim replica: rms over centered 2048/512 windows with zero
// padding, threshold top_db below the loudest window, returns the sample
// span [start, end) of the non-silent region (start == end when all silent).
void mtmd_audio_trim_silence(const float * samples, size_t n_samples, float top_db,
                             size_t & start, size_t & end);

// power mel of the chatterbox voice encoder (voice_encoder/melspec.py):
// 16 kHz, hann 400 periodic, hop 160, centered frames with reflect padding,
// squared magnitude against slaney mel 40 bins fmin 0 fmax 8000, no log.
// output layout is frame major: out[t * 40 + m].
bool mtmd_audio_ve_mel(const float * samples, size_t n_samples,
                       std::vector<float> & out, int & n_frames);

// ITU-R BS.1770 integrated loudness (pyloudnorm replica, mono): K-weighting
// (RBJ high shelf 1681.97 Hz +4 dB then high pass 38.14 Hz), 400 ms blocks
// with 75% overlap, absolute -70 then relative -10 gating.
// returns the loudness in LUFS, or -HUGE_VALF when everything is gated out.
float mtmd_audio_lufs(const float * samples, size_t n_samples, int sample_rate);

struct mtmd_audio_preprocessor {
    const clip_hparams & hparams;

    mtmd_audio_preprocessor(const clip_ctx * ctx): hparams(*clip_get_hparams(ctx)) {}

    virtual ~mtmd_audio_preprocessor() = default;
    virtual void initialize() = 0; // NOT thread-safe
    virtual bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) = 0;
};

struct mtmd_audio_preprocessor_whisper : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_whisper(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_conformer : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_conformer(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_granite_speech : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_granite_speech(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_gemma4a : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_gemma4a(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_gemma4ua : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_gemma4ua(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;
};

struct mtmd_audio_preprocessor_qwen3a : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_qwen3a(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_mimo_audio : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_mimo_audio(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_qwen3tts_spk : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_qwen3tts_spk(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_chatterbox_spk : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_chatterbox_spk(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    std::vector<float> window;  // povey window, frame_length points
    std::vector<float> filters; // kaldi mel filterbank, n_mel x (n_fft / 2) dense
};

struct mtmd_audio_preprocessor_parakeet : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_parakeet(clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) { }
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;

    static void worker_thread(int ith, const float * window_func, int window_size,
                              const std::vector<float> & samples, int n_samples,
                              int frame_size, int frame_step, int n_threads,
                              int n_fft_bins,
                              const mtmd_audio_cache & cache, mtmd_audio_mel & mel);
};

//
// streaming ISTFT - converts spectrogram frames back to audio one frame at a time
//
struct mtmd_audio_streaming_istft {
    mtmd_audio_streaming_istft(int n_fft, int hop_length);

    // reset streaming state
    void reset();

    // process a single STFT frame (streaming)
    // frame_spectrum: [n_fft_bins x 2] interleaved real/imag
    // returns: up to hop_length samples
    std::vector<float> process_frame(const float * frame_spectrum);

    // flush remaining samples at end of stream
    std::vector<float> flush();

  private:
    int n_fft;
    int hop_length;
    int n_fft_bins;

    // Own cache for output processing
    mtmd_audio_cache cache;

    // Streaming state
    std::vector<float> overlap_buffer;
    std::vector<float> window_sum_buffer;
    int                padding_to_remove;

    // Working buffers for IFFT
    std::vector<float> ifft_in;
    std::vector<float> ifft_out;
};
