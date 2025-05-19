#pragma once

#include "ggml.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>
#include <fstream>
#include <algorithm>

// most of the code here is copied from whisper.cpp

#define WHISPER_ASSERT GGML_ASSERT

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30

namespace whisper_preprocessor {

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

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

} // namespace whisper_preprocessor


namespace wav_utils {

#define COMMON_SAMPLE_RATE 16000

static bool is_wav_buffer(const std::string buf) {
    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    if (buf.size() < 12 || buf.substr(0, 4) != "RIFF" || buf.substr(8, 4) != "WAVE") {
        return false;
    }

    uint32_t chunk_size = *reinterpret_cast<const uint32_t*>(buf.data() + 4);
    if (chunk_size + 8 != buf.size()) {
        return false;
    }

    return true;
}

// returns mono PCM data
static bool read_wav_from_buf(const unsigned char * buf_in, size_t len, std::vector<float> & pcmf32) {
    drwav wav;
    
    if (drwav_init_memory(&wav, buf_in, len, nullptr) == false) {
        fprintf(stderr, "error: failed to open WAV file from buffer\n");
        return false;
    }

    if (wav.channels != 1 && wav.channels != 2) {
        fprintf(stderr, "%s: WAV input must be mono or stereo\n", __func__);
        drwav_uninit(&wav);
        return false;
    }

    if (wav.sampleRate != COMMON_SAMPLE_RATE) {
        fprintf(stderr, "%s: WAV input must be %i kHz\n", __func__, COMMON_SAMPLE_RATE/1000);
        drwav_uninit(&wav);
        return false;
    }

    if (wav.bitsPerSample != 16) {
        fprintf(stderr, "%s: WAV input must be 16-bit\n", __func__);
        drwav_uninit(&wav);
        return false;
    }

    const uint64_t n = wav.totalPCMFrameCount;

    std::vector<int16_t> pcm16;
    pcm16.resize(n*wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    drwav_uninit(&wav);

    // convert to mono, float
    pcmf32.resize(n);
    if (wav.channels == 1) {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[i])/32768.0f;
        }
    } else {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[2*i] + pcm16[2*i + 1])/65536.0f;
        }
    }

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

static whisper_preprocessor::whisper_filters get_80_bins() {
    whisper_preprocessor::whisper_filters filters;
    filters.n_mel = 80;
    filters.n_fft = 201;
    std::vector data(filters.n_mel * filters.n_fft, 0.0f);

    data[1] = 0.02486259;
    data[202] = 0.00199082;
    data[203] = 0.02287177;
    data[404] = 0.00398164;
    data[405] = 0.02088095;
    data[606] = 0.00597247;
    data[607] = 0.01889013;
    data[808] = 0.00796329;
    data[809] = 0.01689931;
    data[1010] = 0.00995411;
    data[1011] = 0.01490848;
    data[1212] = 0.01194493;
    data[1213] = 0.01291766;
    data[1414] = 0.01393575;
    data[1415] = 0.01092684;
    data[1616] = 0.01592658;
    data[1617] = 0.00893602;
    data[1818] = 0.01791740;
    data[1819] = 0.00694520;
    data[2020] = 0.01990822;
    data[2021] = 0.00495437;
    data[2222] = 0.02189904;
    data[2223] = 0.00296355;
    data[2424] = 0.02388986;
    data[2425] = 0.00097273;
    data[2626] = 0.02588068;
    data[2828] = 0.02583532;
    data[3029] = 0.00101809;
    data[3030] = 0.02384450;
    data[3231] = 0.00300891;
    data[3232] = 0.02185368;
    data[3433] = 0.00499973;
    data[3434] = 0.01986286;
    data[3635] = 0.00699056;
    data[3636] = 0.01787204;
    data[3837] = 0.00898138;
    data[3838] = 0.01588122;
    data[4039] = 0.01097220;
    data[4040] = 0.01389039;
    data[4241] = 0.01296302;
    data[4242] = 0.01189957;
    data[4443] = 0.01495384;
    data[4444] = 0.00990875;
    data[4645] = 0.01694467;
    data[4646] = 0.00791793;
    data[4847] = 0.01893549;
    data[4848] = 0.00592711;
    data[5049] = 0.02087401;
    data[5050] = 0.00404043;
    data[5251] = 0.02211422;
    data[5252] = 0.00331861;
    data[5453] = 0.02173672;
    data[5454] = 0.00361097;
    data[5655] = 0.02049770;
    data[5656] = 0.00476219;
    data[5857] = 0.01848666;
    data[5858] = 0.00659262;
    data[6059] = 0.01585604;
    data[6060] = 0.00896277;
    data[6261] = 0.01273877;
    data[6262] = 0.01175133;
    data[6463] = 0.00925037;
    data[6464] = 0.01485314;
    data[6665] = 0.00549084;
    data[6666] = 0.01817747;
    data[6667] = 0.00281555;
    data[6867] = 0.00154637;
    data[6868] = 0.01632952;
    data[6869] = 0.00742019;
    data[7070] = 0.01118105;
    data[7071] = 0.01201886;
    data[7272] = 0.00606535;
    data[7273] = 0.01656128;
    data[7274] = 0.00436088;
    data[7474] = 0.00102980;
    data[7475] = 0.01277054;
    data[7476] = 0.00970719;
    data[7677] = 0.00698640;
    data[7678] = 0.01485430;
    data[7679] = 0.00439122;
    data[7879] = 0.00141805;
    data[7880] = 0.01148692;
    data[7881] = 0.01008974;
    data[7882] = 0.00040022;
    data[8082] = 0.00541110;
    data[8083] = 0.01473557;
    data[8084] = 0.00651819;
    data[8285] = 0.00827841;
    data[8286] = 0.01227756;
    data[8287] = 0.00396781;
    data[8487] = 0.00218781;
    data[8488] = 0.01018448;
    data[8489] = 0.00998188;
    data[8490] = 0.00228649;
    data[8690] = 0.00386943;
    data[8691] = 0.01127489;
    data[8692] = 0.00846622;
    data[8693] = 0.00133977;
    data[8893] = 0.00482029;
    data[8894] = 0.01167825;
    data[8895] = 0.00760868;
    data[8896] = 0.00100910;
    data[9096] = 0.00515696;
    data[9097] = 0.01150789;
    data[9098] = 0.00730182;
    data[9099] = 0.00119017;
    data[9299] = 0.00498210;
    data[9300] = 0.01086350;
    data[9301] = 0.00745119;
    data[9302] = 0.00179138;
    data[9502] = 0.00438592;
    data[9503] = 0.00983249;
    data[9504] = 0.00797396;
    data[9505] = 0.00273259;
    data[9705] = 0.00344745;
    data[9706] = 0.00849135;
    data[9707] = 0.00879769;
    data[9708] = 0.00394383;
    data[9908] = 0.00223576;
    data[9909] = 0.00690675;
    data[9910] = 0.00985924;
    data[9911] = 0.00536424;
    data[9912] = 0.00086923;
    data[10111] = 0.00081100;
    data[10112] = 0.00513665;
    data[10113] = 0.00946230;
    data[10114] = 0.00694108;
    data[10115] = 0.00277840;
    data[10315] = 0.00323120;
    data[10316] = 0.00723705;
    data[10317] = 0.00862883;
    data[10318] = 0.00477391;
    data[10319] = 0.00091899;
    data[10518] = 0.00123364;
    data[10519] = 0.00494332;
    data[10520] = 0.00865301;
    data[10521] = 0.00681850;
    data[10522] = 0.00324858;
    data[10722] = 0.00261644;
    data[10723] = 0.00605185;
    data[10724] = 0.00888047;
    data[10725] = 0.00557448;
    data[10726] = 0.00226849;
    data[10925] = 0.00028637;
    data[10926] = 0.00346780;
    data[10927] = 0.00664923;
    data[10928] = 0.00787146;
    data[10929] = 0.00480990;
    data[10930] = 0.00174833;
    data[11129] = 0.00092459;
    data[11130] = 0.00387081;
    data[11131] = 0.00681703;
    data[11132] = 0.00728334;
    data[11133] = 0.00444812;
    data[11134] = 0.00161290;
    data[11333] = 0.00117033;
    data[11334] = 0.00389873;
    data[11335] = 0.00662713;
    data[11336] = 0.00704732;
    data[11337] = 0.00442171;
    data[11338] = 0.00179611;
    data[11537] = 0.00108930;
    data[11538] = 0.00361598;
    data[11539] = 0.00614267;
    data[11540] = 0.00710294;
    data[11541] = 0.00467145;
    data[11542] = 0.00223996;
    data[11741] = 0.00073923;
    data[11742] = 0.00307911;
    data[11743] = 0.00541899;
    data[11744] = 0.00739719;
    data[11745] = 0.00514546;
    data[11746] = 0.00289374;
    data[11747] = 0.00064202;
    data[11945] = 0.00017069;
    data[11946] = 0.00233757;
    data[11947] = 0.00450446;
    data[11948] = 0.00667135;
    data[11949] = 0.00579848;
    data[11950] = 0.00371323;
    data[11951] = 0.00162798;
    data[12150] = 0.00143453;
    data[12151] = 0.00344122;
    data[12152] = 0.00544790;
    data[12153] = 0.00659109;
    data[12154] = 0.00466001;
    data[12155] = 0.00272893;
    data[12156] = 0.00079785;
    data[12354] = 0.00040750;
    data[12355] = 0.00226583;
    data[12356] = 0.00412416;
    data[12357] = 0.00598248;
    data[12358] = 0.00570082;
    data[12359] = 0.00391251;
    data[12360] = 0.00212420;
    data[12361] = 0.00033589;
    data[12559] = 0.00100991;
    data[12560] = 0.00273085;
    data[12561] = 0.00445178;
    data[12562] = 0.00617272;
    data[12563] = 0.00515091;
    data[12564] = 0.00349481;
    data[12565] = 0.00183871;
    data[12566] = 0.00018261;
    data[12764] = 0.00129437;
    data[12765] = 0.00288807;
    data[12766] = 0.00448178;
    data[12767] = 0.00607548;
    data[12768] = 0.00488666;
    data[12769] = 0.00335300;
    data[12770] = 0.00181934;
    data[12771] = 0.00028568;
    data[12969] = 0.00131314;
    data[12970] = 0.00278902;
    data[12971] = 0.00426489;
    data[12972] = 0.00574077;
    data[12973] = 0.00485998;
    data[12974] = 0.00343971;
    data[12975] = 0.00201943;
    data[12976] = 0.00059916;
    data[13174] = 0.00111217;
    data[13175] = 0.00247893;
    data[13176] = 0.00384569;
    data[13177] = 0.00521246;
    data[13178] = 0.00502864;
    data[13179] = 0.00371337;
    data[13180] = 0.00239810;
    data[13181] = 0.00108283;
    data[13379] = 0.00073175;
    data[13380] = 0.00199747;
    data[13381] = 0.00326318;
    data[13382] = 0.00452890;
    data[13383] = 0.00535569;
    data[13384] = 0.00413766;
    data[13385] = 0.00291963;
    data[13386] = 0.00170160;
    data[13387] = 0.00048358;
    data[13584] = 0.00020714;
    data[13585] = 0.00137928;
    data[13586] = 0.00255141;
    data[13587] = 0.00372355;
    data[13588] = 0.00489569;
    data[13589] = 0.00468090;
    data[13590] = 0.00355292;
    data[13591] = 0.00242494;
    data[13592] = 0.00129697;
    data[13593] = 0.00016899;
    data[13790] = 0.00065453;
    data[13791] = 0.00174001;
    data[13792] = 0.00282548;
    data[13793] = 0.00391096;
    data[13794] = 0.00499644;
    data[13795] = 0.00427098;
    data[13796] = 0.00322640;
    data[13797] = 0.00218181;
    data[13798] = 0.00113723;
    data[13799] = 0.00009265;
    data[13996] = 0.00085463;
    data[13997] = 0.00185985;
    data[13998] = 0.00286508;
    data[13999] = 0.00387031;
    data[14000] = 0.00487553;
    data[14001] = 0.00408314;
    data[14002] = 0.00311578;
    data[14003] = 0.00214843;
    data[14004] = 0.00118108;
    data[14005] = 0.00021372;
    data[14202] = 0.00084834;
    data[14203] = 0.00177925;
    data[14204] = 0.00271016;
    data[14205] = 0.00364107;
    data[14206] = 0.00457197;
    data[14207] = 0.00407973;
    data[14208] = 0.00318389;
    data[14209] = 0.00228806;
    data[14210] = 0.00139222;
    data[14211] = 0.00049639;
    data[14408] = 0.00067162;
    data[14409] = 0.00153370;
    data[14410] = 0.00239579;
    data[14411] = 0.00325787;
    data[14412] = 0.00411996;
    data[14413] = 0.00422773;
    data[14414] = 0.00339812;
    data[14415] = 0.00256852;
    data[14416] = 0.00173891;
    data[14417] = 0.00090931;
    data[14418] = 0.00007970;
    data[14614] = 0.00035598;
    data[14615] = 0.00115433;
    data[14616] = 0.00195268;
    data[14617] = 0.00275102;
    data[14618] = 0.00354937;
    data[14619] = 0.00434772;
    data[14620] = 0.00372996;
    data[14621] = 0.00296169;
    data[14622] = 0.00219342;
    data[14623] = 0.00142515;
    data[14624] = 0.00065688;
    data[14821] = 0.00066829;
    data[14822] = 0.00140762;
    data[14823] = 0.00214694;
    data[14824] = 0.00288627;
    data[14825] = 0.00362559;
    data[14826] = 0.00415458;
    data[14827] = 0.00344311;
    data[14828] = 0.00273164;
    data[14829] = 0.00202017;
    data[14830] = 0.00130870;
    data[14831] = 0.00059723;
    data[15027] = 0.00009927;
    data[15028] = 0.00078393;
    data[15029] = 0.00146859;
    data[15030] = 0.00215326;
    data[15031] = 0.00283792;
    data[15032] = 0.00352259;
    data[15033] = 0.00399152;
    data[15034] = 0.00333265;
    data[15035] = 0.00267378;
    data[15036] = 0.00201491;
    data[15037] = 0.00135604;
    data[15038] = 0.00069717;
    data[15039] = 0.00003830;
    data[15234] = 0.00010181;
    data[15235] = 0.00073586;
    data[15236] = 0.00136990;
    data[15237] = 0.00200395;
    data[15238] = 0.00263799;
    data[15239] = 0.00327204;
    data[15240] = 0.00390609;
    data[15241] = 0.00336826;
    data[15242] = 0.00275810;
    data[15243] = 0.00214794;
    data[15244] = 0.00153778;
    data[15245] = 0.00092763;
    data[15246] = 0.00031747;
    data[15442] = 0.00055304;
    data[15443] = 0.00114021;
    data[15444] = 0.00172738;
    data[15445] = 0.00231454;
    data[15446] = 0.00290171;
    data[15447] = 0.00348888;
    data[15448] = 0.00352334;
    data[15449] = 0.00295829;
    data[15450] = 0.00239325;
    data[15451] = 0.00182820;
    data[15452] = 0.00126315;
    data[15453] = 0.00069810;
    data[15454] = 0.00013306;
    data[15649] = 0.00026084;
    data[15650] = 0.00080460;
    data[15651] = 0.00134836;
    data[15652] = 0.00189211;
    data[15653] = 0.00243587;
    data[15654] = 0.00297963;
    data[15655] = 0.00352339;
    data[15656] = 0.00325138;
    data[15657] = 0.00272811;
    data[15658] = 0.00220484;
    data[15659] = 0.00168156;
    data[15660] = 0.00115829;
    data[15661] = 0.00063502;
    data[15662] = 0.00011175;
    data[15857] = 0.00038498;
    data[15858] = 0.00088854;
    data[15859] = 0.00139210;
    data[15860] = 0.00189565;
    data[15861] = 0.00239921;
    data[15862] = 0.00290277;
    data[15863] = 0.00340633;
    data[15864] = 0.00313276;
    data[15865] = 0.00264818;
    data[15866] = 0.00216359;
    data[15867] = 0.00167901;
    data[15868] = 0.00119442;
    data[15869] = 0.00070984;
    data[15870] = 0.00022525;
    data[16065] = 0.00036674;
    data[16066] = 0.00083307;
    data[16067] = 0.00129940;
    data[16068] = 0.00176573;
    data[16069] = 0.00223205;
    data[16070] = 0.00269838;
    data[16071] = 0.00316471;
    data[16072] = 0.00314131;
    data[16073] = 0.00269255;
    data[16074] = 0.00224380;
    data[16075] = 0.00179504;
    data[16076] = 0.00134628;
    data[16077] = 0.00089752;
    data[16078] = 0.00044876;

    filters.data = std::move(data);
    return filters;
}

} // namespace whisper_precalc_filters
