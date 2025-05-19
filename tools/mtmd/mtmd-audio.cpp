#include "mtmd-audio.h"

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio.h"

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#define _USE_MATH_DEFINES // for M_PI
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>
#include <fstream>
#include <algorithm>

// most of the code here is copied from whisper.cpp

namespace whisper_preprocessor {

#define SIN_COS_N_COUNT WHISPER_N_FFT
namespace {
struct whisper_global_cache {
    // In FFT, we frequently use sine and cosine operations with the same values.
    // We can use precalculated values to speed up the process.
    float sin_vals[SIN_COS_N_COUNT];
    float cos_vals[SIN_COS_N_COUNT];

    // Hann window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    float hann_window[WHISPER_N_FFT];

    whisper_global_cache() {
        fill_sin_cos_table();
        fill_hann_window(sizeof(hann_window)/sizeof(hann_window[0]), true, hann_window);
    }

    void fill_sin_cos_table() {
        for (int i = 0; i < SIN_COS_N_COUNT; i++) {
            double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
            sin_vals[i] = sinf(theta);
            cos_vals[i] = cosf(theta);
        }
    }

    void fill_hann_window(int length, bool periodic, float * output) {
        int offset = -1;
        if (periodic) {
            offset = 0;
        }
        for (int i = 0; i < length; i++) {
            output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
        }
    }
} global_cache;
}

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const float* in, int N, float* out) {
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT); // t = 2*M_PI*k*n/N
            re += in[n]*global_cache.cos_vals[idx]; // cos(t)
            im -= in[n]*global_cache.sin_vals[idx]; // sin(t)
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
static void fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    const int half_N = N / 2;
    if (N - half_N*2 == 1) {
        dft(in, N, out);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i]= in[2*i];
    }
    float* even_fft = out + 2 * N;
    fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2*i + 1];
    }
    float* odd_fft = even_fft + N;
    fft(odd, half_N, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < half_N; k++) {
        int idx = k * sin_cos_step; // t = 2*M_PI*k/N
        float re = global_cache.cos_vals[idx]; // cos(t)
        float im = -global_cache.sin_vals[idx]; // sin(t)

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + half_N) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + half_N) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}

static void log_mel_spectrogram_worker_thread(int ith, const float * hann, const std::vector<float> & samples,
                                              int n_samples, int frame_size, int frame_step, int n_threads,
                                              const whisper_filters & filters, whisper_mel & mel) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fft = filters.n_fft;
    int i = ith;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    WHISPER_ASSERT(n_fft == 1 + (frame_size / 2));

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in.data(), frame_size, fft_out.data());

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum +=
                        fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                        fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                        fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                        fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }
            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }
            sum = log10(std::max(sum, 1e-10));
            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
static bool log_mel_spectrogram(
        const float * samples,
        const int   n_samples,
        const int   /*sample_rate*/,
        const int   frame_size,
        const int   frame_step,
        const int   n_mel,
        const int   n_threads,
        const whisper_filters & filters,
        const bool   debug,
        whisper_mel & mel) {
    //const int64_t t_start_us = ggml_time_us();

    // Hann window
    WHISPER_ASSERT(frame_size == WHISPER_N_FFT && "Unsupported frame_size");
    const float * hann = global_cache.hann_window;

    // Calculate the length of padding
    int64_t stage_1_pad = WHISPER_SAMPLE_RATE * 30;
    int64_t stage_2_pad = frame_size / 2;

    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

    mel.n_mel     = n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len     = (samples_padded.size() - frame_size) / frame_step;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
    mel.data.resize(mel.n_mel * mel.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(
                    log_mel_spectrogram_worker_thread, iw + 1, hann, std::cref(samples_padded),
                    n_samples + stage_2_pad, frame_size, frame_step, n_threads,
                    std::cref(filters), std::ref(mel));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples + stage_2_pad, frame_size, frame_step, n_threads, filters, mel);

        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0)/4.0;
    }

    // Dump log_mel_spectrogram
    if (debug) {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < mel.data.size() - 1; i++) {
            outFile << mel.data[i] << ", ";
        }
        outFile << mel.data[mel.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}

bool preprocess_audio(
        const float * samples,
        size_t n_samples,
        whisper_filters & filters,
        whisper_mel & output) {
    return log_mel_spectrogram(
                samples,
                n_samples,
                COMMON_SAMPLE_RATE,
                WHISPER_N_FFT,
                WHISPER_HOP_LENGTH,
                filters.n_mel,
                4, // n_threads
                filters,
                false, // debug
                output);
}

} // namespace whisper_preprocessor


namespace wav_utils {

// Sinc function: sin(pi*x) / (pi*x)
static double calculate_sinc(double x) {
    if (x == 0.0) return 1.0;
    double pi_x = M_PI * x;
    return std::sin(pi_x) / pi_x;
}

// Hann window function
static double calculate_hann_window(double x, double half_width) {
    if (half_width == 0.0) return 1.0;
    if (std::abs(x) >= half_width) return 0.0;
    return 0.5 * (1.0 + std::cos(M_PI * x / half_width));
}

/**
 * @brief Resamples audio data using windowed sinc interpolation.
 * @param kernel_half_width_input_samples Number of input samples on each side of the
 *        interpolation point for the sinc kernel. Larger values improve quality but cost performance.
 */
static std::vector<float> resample_sinc(const std::vector<float>& samples,
                                 int new_rate,
                                 int old_rate,
                                 int kernel_half_width_input_samples = 16) {
    if (old_rate <= 0 || new_rate <= 0) {
        throw std::invalid_argument("Sample rates must be positive.");
    }
    if (samples.empty()) return {};
    if (new_rate == old_rate) return samples;
    if (kernel_half_width_input_samples <= 0) {
        throw std::invalid_argument("Kernel half width must be positive.");
    }

    double ratio = static_cast<double>(new_rate) / old_rate;
    size_t new_num_samples = static_cast<size_t>(std::round(static_cast<double>(samples.size()) * ratio));

    if (new_num_samples == 0) return {};

    std::vector<float> resampled_samples(new_num_samples);

    // Sinc argument scaling for anti-aliasing/anti-imaging:
    // adjusts filter cutoff to the lower of the two Nyquist frequencies.
    double sinc_argument_scale_factor = std::min(1.0, ratio);

    for (size_t i = 0; i < new_num_samples; ++i) {
        double t_new_sample_time = static_cast<double>(i) / new_rate;
        double center_input_idx_float = t_new_sample_time * old_rate; // Fractional index in original samples

        double current_output_value = 0.0;
        double current_kernel_sum = 0.0; // For normalizing filter gain

        int first_input_idx_to_consider = static_cast<int>(std::floor(center_input_idx_float)) - kernel_half_width_input_samples + 1;
        int last_input_idx_to_consider  = static_cast<int>(std::floor(center_input_idx_float)) + kernel_half_width_input_samples;

        for (int k = first_input_idx_to_consider; k <= last_input_idx_to_consider; ++k) {
            if (k < 0 || k >= static_cast<int>(samples.size())) {
                continue; // Effectively zero-padding
            }

            // Distance (in original sample intervals) from original sample 'k' to new sample's ideal position
            double time_diff_in_old_samples = center_input_idx_float - static_cast<double>(k);
            double sinc_kernel_arg = time_diff_in_old_samples * sinc_argument_scale_factor;
            
            double sinc_value = calculate_sinc(sinc_kernel_arg);
            double window_value = calculate_hann_window(time_diff_in_old_samples, static_cast<double>(kernel_half_width_input_samples));
            
            double tap_weight = sinc_value * window_value;
            
            current_output_value += samples[k] * tap_weight;
            current_kernel_sum += tap_weight;
        }

        if (current_kernel_sum != 0.0) {
            resampled_samples[i] = static_cast<float>(current_output_value / current_kernel_sum);
        } else {
            resampled_samples[i] = 0.0f; // If kernel sum is zero (e.g., outside original signal range)
        }
    }
    return resampled_samples;
}

bool is_wav_buffer(const std::string buf) {
    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    if (buf.size() < 12 || buf.substr(0, 4) != "RIFF" || buf.substr(8, 4) != "WAVE") {
        return false;
    }

    // uint32_t chunk_size = *reinterpret_cast<const uint32_t*>(buf.data() + 4);
    // if (chunk_size + 8 != buf.size()) {
    //     return false;
    // }

    return true;
}

// returns mono PCM data
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <vector>
// #include <iostream> // For debugging, can be removed

bool read_wav_from_buf(const unsigned char * buf_in, size_t len, int target_sampler_rate, std::vector<float> & pcmf32_mono) {
    ma_result result;
    // Request f32 output from the decoder. Channel count and sample rate are determined from the file.
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, 0, 0);
    ma_decoder decoder;

    result = ma_decoder_init_memory(buf_in, len, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        fprintf(stderr, "Unable to initialize decoder\n");
        return false;
    }

    // Decoder will output ma_format_f32.
    // We need to use the data converter if:
    // 1. The sample rate needs to be changed.
    // 2. The audio is not already mono (decoder.outputChannels != 1).
    bool needs_resampling = (decoder.outputSampleRate != (ma_uint32)target_sampler_rate);
    bool needs_channel_mixing = (decoder.outputChannels != 1);

    if (!needs_resampling && !needs_channel_mixing) {
        // Already target sample rate, already mono, and decoder is outputting f32. Direct read.
        ma_uint64 frame_count_total;
        result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count_total);
        if (result != MA_SUCCESS) {
            ma_decoder_uninit(&decoder);
            return false;
        }

        pcmf32_mono.resize(frame_count_total); // Mono, so frames == samples
        ma_uint64 frames_read = 0;
        result = ma_decoder_read_pcm_frames(&decoder, pcmf32_mono.data(), frame_count_total, &frames_read);
        if (result != MA_SUCCESS || frames_read != frame_count_total) {
            ma_decoder_uninit(&decoder);
            return false;
        }
    } else {
        // Resampling and/or channel mixing is needed.
        ma_data_converter_config data_converter_config = ma_data_converter_config_init_default();
        data_converter_config.formatIn = decoder.outputFormat; // This will be ma_format_f32
        data_converter_config.formatOut = ma_format_f32;       // Output is also f32
        data_converter_config.channelsIn = decoder.outputChannels;
        data_converter_config.channelsOut = 1; // MONO output
        data_converter_config.sampleRateIn = decoder.outputSampleRate;
        data_converter_config.sampleRateOut = (ma_uint32)target_sampler_rate;
        data_converter_config.resampling.algorithm = ma_resample_algorithm_linear; // Or other algorithm

        ma_data_converter data_converter;
        result = ma_data_converter_init(&data_converter_config, NULL, &data_converter);
        if (result != MA_SUCCESS) {
            ma_decoder_uninit(&decoder);
            return false;
        }

        ma_uint64 total_frames_expected_from_decoder;
        result = ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames_expected_from_decoder);
        if (result != MA_SUCCESS) {
            ma_data_converter_uninit(&data_converter, NULL);
            ma_decoder_uninit(&decoder);
            return false;
        }
        
        double resample_ratio = (double)target_sampler_rate / decoder.outputSampleRate;
        // Reserve for mono output
        pcmf32_mono.reserve(static_cast<size_t>(total_frames_expected_from_decoder * resample_ratio * 1.1) + 1);

        // Buffer to hold data read from the decoder (multi-channel, original sample rate, f32 format)
        const ma_uint64 DECODE_BUFFER_SIZE_FRAMES = 1024;
        std::vector<float> temp_decode_buffer(DECODE_BUFFER_SIZE_FRAMES * decoder.outputChannels);

        while (true) {
            ma_uint64 frames_decoded_this_iteration = 0;
            result = ma_decoder_read_pcm_frames(&decoder, temp_decode_buffer.data(), DECODE_BUFFER_SIZE_FRAMES, &frames_decoded_this_iteration);

            if (result != MA_SUCCESS && result != MA_AT_END) {
                ma_data_converter_uninit(&data_converter, NULL);
                ma_decoder_uninit(&decoder);
                return false;
            }

            if (frames_decoded_this_iteration == 0 && result == MA_AT_END) { // Ensure we process the last bit if MA_AT_END was from previous read
                break; 
            }
            
            ma_uint64 frame_count_in = frames_decoded_this_iteration;
            ma_uint64 frame_count_out_capacity;

            result = ma_data_converter_get_expected_output_frame_count(&data_converter, frame_count_in, &frame_count_out_capacity);
            if (result != MA_SUCCESS) {
                ma_data_converter_uninit(&data_converter, NULL);
                ma_decoder_uninit(&decoder);
                return false;
            }
            
            size_t current_pcmf32_sample_offset = pcmf32_mono.size();
            // Resize for mono output (channelsOut is 1)
            pcmf32_mono.resize(current_pcmf32_sample_offset + frame_count_out_capacity * data_converter.channelsOut);

            ma_uint64 frames_actually_output = frame_count_out_capacity;

            result = ma_data_converter_process_pcm_frames(
                &data_converter,
                temp_decode_buffer.data(),
                &frame_count_in, 
                pcmf32_mono.data() + current_pcmf32_sample_offset,
                &frames_actually_output
            );

            if (result != MA_SUCCESS) {
                ma_data_converter_uninit(&data_converter, NULL);
                ma_decoder_uninit(&decoder);
                return false;
            }
            
            // Adjust size to actual frames output (mono)
            pcmf32_mono.resize(current_pcmf32_sample_offset + frames_actually_output * data_converter.channelsOut);

            if (result == MA_AT_END) {
                if (frames_decoded_this_iteration == 0 || frame_count_in == 0) break; // No more input frames processed or decoded
            }
        }
        ma_data_converter_uninit(&data_converter, NULL);
    }

    ma_decoder_uninit(&decoder);
    return true;
}

} // namespace wav_utils


// precalculated mel filter banks
//
// generated from python code:
//
// from numpy import load
// data = load('mel_filters.npz')
// lst = data.files
// for item in lst:
//   print(item)
//   print(data[item].shape)
//   n_mel = data[item].shape[0]
//   n_fft = data[item].shape[1]
//   for i, row in enumerate(data[item]):
//     for j, val in enumerate(row):
//       if val != 0:
//         print(f"data[{i*n_fft + j}] = {val:.6f};")

namespace whisper_precalc_filters {

whisper_preprocessor::whisper_filters get_128_bins() {
    whisper_preprocessor::whisper_filters filters;
    filters.n_mel = 128;
    filters.n_fft = 201;
    std::vector data(filters.n_mel * filters.n_fft, 0.0f);

    data[1] = 0.01237399;
    data[202] = 0.03039256;
    data[404] = 0.02474797;
    data[605] = 0.01801858;
    data[807] = 0.03712196;
    data[1008] = 0.00564459;
    data[1009] = 0.00672939;
    data[1210] = 0.03603716;
    data[1412] = 0.01910338;
    data[1613] = 0.02366317;
    data[1815] = 0.03147737;
    data[2016] = 0.01128918;
    data[2017] = 0.00108480;
    data[2218] = 0.04168175;
    data[2420] = 0.01345879;
    data[2621] = 0.02930776;
    data[2823] = 0.02583277;
    data[3024] = 0.01693378;
    data[3226] = 0.03820676;
    data[3427] = 0.00455979;
    data[3428] = 0.00781420;
    data[3629] = 0.03495236;
    data[3831] = 0.02018818;
    data[4032] = 0.02257837;
    data[4234] = 0.03256217;
    data[4435] = 0.01020438;
    data[4436] = 0.00216960;
    data[4637] = 0.04059695;
    data[4839] = 0.01454359;
    data[5040] = 0.02822296;
    data[5242] = 0.02691758;
    data[5443] = 0.01584898;
    data[5645] = 0.03929156;
    data[5846] = 0.00347499;
    data[5847] = 0.00889900;
    data[6048] = 0.03386755;
    data[6250] = 0.02127299;
    data[6451] = 0.02149357;
    data[6653] = 0.03364697;
    data[6854] = 0.00911958;
    data[6855] = 0.00325441;
    data[7056] = 0.03951215;
    data[7258] = 0.01562839;
    data[7459] = 0.02713816;
    data[7661] = 0.02800238;
    data[7862] = 0.01476417;
    data[8064] = 0.04037637;
    data[8265] = 0.00238069;
    data[8266] = 0.01020264;
    data[8467] = 0.03161146;
    data[8669] = 0.02454700;
    data[8870] = 0.01532919;
    data[8871] = 0.00166584;
    data[9072] = 0.03672905;
    data[9274] = 0.02009710;
    data[9475] = 0.01693103;
    data[9476] = 0.00290266;
    data[9677] = 0.03284499;
    data[9879] = 0.02352005;
    data[10080] = 0.01103894;
    data[10081] = 0.01072583;
    data[10282] = 0.02271829;
    data[10484] = 0.03227873;
    data[10685] = 0.00011627;
    data[10686] = 0.02285348;
    data[10887] = 0.00856344;
    data[10888] = 0.01497979;
    data[11089] = 0.01551398;
    data[11090] = 0.00851491;
    data[11291] = 0.02110680;
    data[11292] = 0.00332652;
    data[11493] = 0.02547065;
    data[11695] = 0.02735908;
    data[11896] = 0.00065854;
    data[11897] = 0.02383813;
    data[12098] = 0.00344359;
    data[12099] = 0.02122455;
    data[12300] = 0.00535842;
    data[12301] = 0.01942556;
    data[12502] = 0.00649325;
    data[12503] = 0.01835542;
    data[12704] = 0.00693138;
    data[12705] = 0.01793505;
    data[12906] = 0.00674968;
    data[12907] = 0.01809152;
    data[13108] = 0.00601899;
    data[13109] = 0.01875767;
    data[13310] = 0.00480453;
    data[13311] = 0.01987173;
    data[13512] = 0.00316628;
    data[13513] = 0.02137691;
    data[13514] = 0.00125317;
    data[13714] = 0.00115934;
    data[13715] = 0.02080362;
    data[13716] = 0.00404487;
    data[13917] = 0.01755363;
    data[13918] = 0.00708320;
    data[14119] = 0.01407539;
    data[14120] = 0.01032655;
    data[14321] = 0.01040921;
    data[14322] = 0.01373696;
    data[14523] = 0.00659188;
    data[14524] = 0.01727988;
    data[14525] = 0.00146804;
    data[14725] = 0.00265682;
    data[14726] = 0.01809193;
    data[14727] = 0.00585656;
    data[14928] = 0.01334278;
    data[14929] = 0.01028268;
    data[15130] = 0.00856800;
    data[15131] = 0.01472231;
    data[15132] = 0.00104040;
    data[15332] = 0.00379086;
    data[15333] = 0.01714678;
    data[15334] = 0.00611609;
    data[15535] = 0.01175929;
    data[15536] = 0.01113394;
    data[15737] = 0.00643858;
    data[15738] = 0.01607806;
    data[15739] = 0.00423917;
    data[15939] = 0.00119989;
    data[15940] = 0.01275672;
    data[15941] = 0.00965299;
    data[16142] = 0.00706935;
    data[16143] = 0.01494055;
    data[16144] = 0.00419025;
    data[16344] = 0.00151483;
    data[16345] = 0.01200900;
    data[16346] = 0.00984823;
    data[16547] = 0.00610224;
    data[16548] = 0.01533857;
    data[16549] = 0.00557677;
    data[16749] = 0.00036827;
    data[16750] = 0.00989749;
    data[16751] = 0.01135340;
    data[16752] = 0.00205122;
    data[16952] = 0.00389297;
    data[16953] = 0.01297352;
    data[16954] = 0.00806632;
    data[17155] = 0.00674493;
    data[17156] = 0.01385875;
    data[17157] = 0.00541191;
    data[17357] = 0.00074220;
    data[17358] = 0.00898779;
    data[17359] = 0.01137871;
    data[17360] = 0.00332958;
    data[17560] = 0.00282314;
    data[17561] = 0.01068049;
    data[17562] = 0.00943341;
    data[17563] = 0.00176326;
    data[17763] = 0.00439019;
    data[17764] = 0.01187759;
    data[17765] = 0.00797006;
    data[17766] = 0.00066105;
    data[17966] = 0.00549467;
    data[17967] = 0.01262954;
    data[17968] = 0.00693988;
    data[18169] = 0.00618402;
    data[18170] = 0.01293473;
    data[18171] = 0.00629779;
    data[18371] = 0.00002325;
    data[18372] = 0.00650207;
    data[18373] = 0.01232662;
    data[18374] = 0.00600217;
    data[18574] = 0.00031549;
    data[18575] = 0.00648926;
    data[18576] = 0.01204130;
    data[18577] = 0.00601463;
    data[18777] = 0.00029980;
    data[18778] = 0.00618288;
    data[18779] = 0.01204273;
    data[18780] = 0.00629981;
    data[18781] = 0.00055690;
    data[18980] = 0.00001120;
    data[18981] = 0.00561729;
    data[18982] = 0.01122338;
    data[18983] = 0.00682516;
    data[18984] = 0.00135264;
    data[19184] = 0.00482410;
    data[19185] = 0.01016623;
    data[19186] = 0.00756076;
    data[19187] = 0.00234590;
    data[19387] = 0.00383236;
    data[19388] = 0.00892296;
    data[19389] = 0.00847910;
    data[19390] = 0.00350979;
    data[19590] = 0.00266873;
    data[19591] = 0.00751965;
    data[19592] = 0.00955501;
    data[19593] = 0.00481966;
    data[19594] = 0.00008432;
    data[19793] = 0.00135767;
    data[19794] = 0.00598020;
    data[19795] = 0.01060272;
    data[19796] = 0.00625298;
    data[19797] = 0.00174060;
    data[19997] = 0.00432644;
    data[19998] = 0.00873132;
    data[19999] = 0.00778917;
    data[20000] = 0.00348924;
    data[20200] = 0.00257835;
    data[20201] = 0.00677583;
    data[20202] = 0.00940942;
    data[20203] = 0.00531195;
    data[20204] = 0.00121448;
    data[20403] = 0.00075411;
    data[20404] = 0.00475396;
    data[20405] = 0.00875380;
    data[20406] = 0.00719209;
    data[20407] = 0.00328754;
    data[20607] = 0.00268180;
    data[20608] = 0.00649331;
    data[20609] = 0.00911458;
    data[20610] = 0.00539387;
    data[20611] = 0.00167317;
    data[20810] = 0.00057394;
    data[20811] = 0.00420600;
    data[20812] = 0.00783806;
    data[20813] = 0.00752023;
    data[20814] = 0.00397471;
    data[20815] = 0.00042919;
    data[21014] = 0.00190464;
    data[21015] = 0.00536569;
    data[21016] = 0.00882674;
    data[21017] = 0.00627609;
    data[21018] = 0.00289751;
    data[21218] = 0.00289885;
    data[21219] = 0.00619694;
    data[21220] = 0.00856699;
    data[21221] = 0.00534748;
    data[21222] = 0.00212797;
    data[21421] = 0.00044750;
    data[21422] = 0.00359030;
    data[21423] = 0.00673311;
    data[21424] = 0.00777024;
    data[21425] = 0.00470231;
    data[21426] = 0.00163439;
    data[21625] = 0.00101536;
    data[21626] = 0.00401019;
    data[21627] = 0.00700501;
    data[21628] = 0.00723443;
    data[21629] = 0.00431096;
    data[21630] = 0.00138748;
    data[21829] = 0.00133349;
    data[21830] = 0.00418731;
    data[21831] = 0.00704113;
    data[21832] = 0.00693188;
    data[21833] = 0.00414606;
    data[21834] = 0.00136023;
    data[22033] = 0.00142880;
    data[22034] = 0.00414825;
    data[22035] = 0.00686770;
    data[22036] = 0.00683705;
    data[22037] = 0.00418239;
    data[22038] = 0.00152774;
    data[22237] = 0.00132610;
    data[22238] = 0.00391751;
    data[22239] = 0.00650892;
    data[22240] = 0.00692640;
    data[22241] = 0.00439673;
    data[22242] = 0.00186706;
    data[22441] = 0.00104828;
    data[22442] = 0.00351767;
    data[22443] = 0.00598707;
    data[22444] = 0.00717824;
    data[22445] = 0.00476768;
    data[22446] = 0.00235712;
    data[22645] = 0.00061636;
    data[22646] = 0.00296949;
    data[22647] = 0.00532262;
    data[22648] = 0.00757265;
    data[22649] = 0.00527559;
    data[22650] = 0.00297852;
    data[22651] = 0.00068146;
    data[22849] = 0.00004971;
    data[22850] = 0.00229205;
    data[22851] = 0.00453438;
    data[22852] = 0.00677672;
    data[22853] = 0.00590241;
    data[22854] = 0.00371350;
    data[22855] = 0.00152459;
    data[23054] = 0.00150285;
    data[23055] = 0.00363961;
    data[23056] = 0.00577637;
    data[23057] = 0.00663159;
    data[23058] = 0.00454574;
    data[23059] = 0.00245990;
    data[23060] = 0.00037405;
    data[23258] = 0.00061796;
    data[23259] = 0.00265411;
    data[23260] = 0.00469026;
    data[23261] = 0.00672641;
    data[23262] = 0.00546035;
    data[23263] = 0.00347271;
    data[23264] = 0.00148507;
    data[23463] = 0.00159234;
    data[23464] = 0.00353262;
    data[23465] = 0.00547290;
    data[23466] = 0.00644368;
    data[23467] = 0.00454963;
    data[23468] = 0.00265558;
    data[23469] = 0.00076153;
    data[23667] = 0.00046749;
    data[23668] = 0.00231642;
    data[23669] = 0.00416534;
    data[23670] = 0.00601427;
    data[23671] = 0.00567845;
    data[23672] = 0.00387357;
    data[23673] = 0.00206870;
    data[23674] = 0.00026383;
    data[23872] = 0.00105349;
    data[23873] = 0.00281536;
    data[23874] = 0.00457723;
    data[23875] = 0.00633910;
    data[23876] = 0.00512816;
    data[23877] = 0.00340826;
    data[23878] = 0.00168837;
    data[24077] = 0.00143350;
    data[24078] = 0.00311242;
    data[24079] = 0.00479133;
    data[24080] = 0.00640944;
    data[24081] = 0.00477052;
    data[24082] = 0.00313161;
    data[24083] = 0.00149269;
    data[24281] = 0.00002932;
    data[24282] = 0.00162919;
    data[24283] = 0.00322906;
    data[24284] = 0.00482892;
    data[24285] = 0.00614671;
    data[24286] = 0.00458497;
    data[24287] = 0.00302322;
    data[24288] = 0.00146147;
    data[24486] = 0.00013602;
    data[24487] = 0.00166056;
    data[24488] = 0.00318509;
    data[24489] = 0.00470963;
    data[24490] = 0.00604072;
    data[24491] = 0.00455251;
    data[24492] = 0.00306429;
    data[24493] = 0.00157608;
    data[24494] = 0.00008786;
    data[24691] = 0.00009328;
    data[24692] = 0.00154604;
    data[24693] = 0.00299880;
    data[24694] = 0.00445155;
    data[24695] = 0.00590431;
    data[24696] = 0.00465566;
    data[24697] = 0.00323752;
    data[24698] = 0.00181937;
    data[24699] = 0.00040123;
    data[24897] = 0.00130263;
    data[24898] = 0.00268698;
    data[24899] = 0.00407134;
    data[24900] = 0.00545570;
    data[24901] = 0.00487832;
    data[24902] = 0.00352695;
    data[24903] = 0.00217558;
    data[24904] = 0.00082420;
    data[25102] = 0.00094595;
    data[25103] = 0.00226513;
    data[25104] = 0.00358430;
    data[25105] = 0.00490348;
    data[25106] = 0.00520570;
    data[25107] = 0.00391795;
    data[25108] = 0.00263021;
    data[25109] = 0.00134246;
    data[25110] = 0.00005471;
    data[25307] = 0.00049038;
    data[25308] = 0.00174744;
    data[25309] = 0.00300451;
    data[25310] = 0.00426157;
    data[25311] = 0.00551864;
    data[25312] = 0.00439707;
    data[25313] = 0.00316996;
    data[25314] = 0.00194284;
    data[25315] = 0.00071573;
    data[25513] = 0.00114698;
    data[25514] = 0.00234486;
    data[25515] = 0.00354273;
    data[25516] = 0.00474061;
    data[25517] = 0.00495198;
    data[25518] = 0.00378265;
    data[25519] = 0.00261331;
    data[25520] = 0.00144397;
    data[25521] = 0.00027464;
    data[25718] = 0.00047570;
    data[25719] = 0.00161717;
    data[25720] = 0.00275865;
    data[25721] = 0.00390013;
    data[25722] = 0.00504160;
    data[25723] = 0.00445712;
    data[25724] = 0.00334284;
    data[25725] = 0.00222856;
    data[25726] = 0.00111428;

    filters.data = std::move(data);
    return filters;
}

} // namespace whisper_precalc_filters
