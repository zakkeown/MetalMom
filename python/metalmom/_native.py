"""Native library loader (cffi dlopen)."""

import os
import cffi

ffi = cffi.FFI()

# C declarations matching metalmom.h
ffi.cdef("""
    /* Status codes */
    #define MM_OK 0
    #define MM_ERR_INVALID_INPUT -1
    #define MM_ERR_METAL_UNAVAILABLE -2
    #define MM_ERR_ALLOC_FAILED -3

    /* Buffer type: holds data + shape for NumPy interop */
    typedef struct {
        float* data;
        int64_t shape[8];   /* max 8 dimensions */
        int32_t ndim;
        int32_t dtype;      /* 0=float32 */
        int64_t count;      /* total element count */
    } MMBuffer;

    /* STFT parameters */
    typedef struct {
        int32_t n_fft;
        int32_t hop_length;
        int32_t win_length;
        int32_t center;     /* bool: 1=true, 0=false */
    } MMSTFTParams;

    /* Opaque context handle */
    typedef void* mm_context;

    /* Lifecycle */
    mm_context mm_init(void);
    void mm_destroy(mm_context ctx);

    /* STFT */
    int32_t mm_stft(mm_context ctx, const float* signal_data, int64_t signal_length,
                    int32_t sample_rate, const MMSTFTParams* params, MMBuffer* out);

    /* iSTFT */
    int32_t mm_istft(mm_context ctx, const float* stft_data, int64_t stft_count,
                     int32_t n_freqs, int32_t n_frames, int32_t sample_rate,
                     int32_t hop_length, int32_t win_length, int32_t center,
                     int64_t output_length, MMBuffer* out);

    /* dB Scaling */
    int32_t mm_amplitude_to_db(mm_context ctx, const float* data, int64_t count,
                               float ref, float amin, float top_db,
                               MMBuffer* out);
    int32_t mm_power_to_db(mm_context ctx, const float* data, int64_t count,
                           float ref, float amin, float top_db,
                           MMBuffer* out);

    /* Mel Spectrogram */
    int32_t mm_mel_spectrogram(mm_context ctx, const float* signal_data, int64_t signal_length,
                               int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                               int32_t win_length, int32_t center, float power,
                               int32_t n_mels, float f_min, float f_max,
                               MMBuffer* out);

    /* MFCC */
    int32_t mm_mfcc(mm_context ctx, const float* signal_data, int64_t signal_length,
                    int32_t sample_rate, int32_t n_mfcc, int32_t n_fft,
                    int32_t hop_length, int32_t win_length,
                    int32_t n_mels, float f_min, float f_max,
                    int32_t center, MMBuffer* out);

    /* Chroma STFT */
    int32_t mm_chroma_stft(mm_context ctx, const float* signal_data, int64_t signal_length,
                           int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                           int32_t win_length, int32_t n_chroma, int32_t center,
                           MMBuffer* out);

    /* Spectral Descriptors */
    int32_t mm_spectral_centroid(mm_context ctx, const float* signal_data, int64_t signal_length,
                                 int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                                 int32_t win_length, int32_t center, MMBuffer* out);
    int32_t mm_spectral_bandwidth(mm_context ctx, const float* signal_data, int64_t signal_length,
                                  int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                                  int32_t win_length, int32_t center, float p, MMBuffer* out);
    int32_t mm_spectral_contrast(mm_context ctx, const float* signal_data, int64_t signal_length,
                                 int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                                 int32_t win_length, int32_t center, int32_t n_bands, float f_min, MMBuffer* out);
    int32_t mm_spectral_rolloff(mm_context ctx, const float* signal_data, int64_t signal_length,
                                int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                                int32_t win_length, int32_t center, float roll_percent, MMBuffer* out);
    int32_t mm_spectral_flatness(mm_context ctx, const float* signal_data, int64_t signal_length,
                                 int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                                 int32_t win_length, int32_t center, MMBuffer* out);

    /* RMS Energy */
    int32_t mm_rms(mm_context ctx, const float* signal_data, int64_t signal_length,
                   int32_t sample_rate, int32_t frame_length, int32_t hop_length,
                   int32_t center, MMBuffer* out);

    /* Zero-Crossing Rate */
    int32_t mm_zero_crossing_rate(mm_context ctx, const float* signal_data, int64_t signal_length,
                                  int32_t sample_rate, int32_t frame_length, int32_t hop_length,
                                  int32_t center, MMBuffer* out);

    /* Memory */
    void mm_buffer_free(MMBuffer* buf);
""")

# Load the dynamic library
_lib_dir = os.path.join(os.path.dirname(__file__), "_lib")
_dylib_path = os.path.join(_lib_dir, "libmetalmom.dylib")

if not os.path.exists(_dylib_path):
    raise RuntimeError(
        f"MetalMom dylib not found at {_dylib_path}. "
        "Run ./scripts/build_dylib.sh first."
    )

lib = ffi.dlopen(_dylib_path)
