#ifndef METALMOM_H
#define METALMOM_H

#include <stdint.h>

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

/* Opaque context handle — NOT thread-safe. Create one per thread. */
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

/* Tonnetz */
int32_t mm_tonnetz(mm_context ctx, const float* signal_data, int64_t signal_length,
                   int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                   int32_t win_length, int32_t n_chroma, int32_t center,
                   MMBuffer* out);

/* Delta Features */
int32_t mm_delta(mm_context ctx, const float* data, int64_t count,
                 int32_t n_features, int32_t n_frames,
                 int32_t width, int32_t order, MMBuffer* out);

/* Stack Memory */
int32_t mm_stack_memory(mm_context ctx, const float* data, int64_t count,
                        int32_t n_features, int32_t n_frames,
                        int32_t n_steps, int32_t delay, MMBuffer* out);

/* Poly Features */
int32_t mm_poly_features(mm_context ctx, const float* data, int64_t count,
                          int32_t n_features, int32_t n_frames,
                          int32_t order, int32_t sr, int32_t n_fft,
                          MMBuffer* out);

/* Audio Info */
int32_t mm_get_duration(mm_context ctx, const char* path, float* out_duration);
int32_t mm_get_sample_rate(mm_context ctx, const char* path, int32_t* out_sr);

/* Audio Loading */
int32_t mm_load(mm_context ctx, const char* path, int32_t sr,
                int32_t mono, float offset, float duration,
                MMBuffer* out, int32_t* out_sr);

/* Resampling */
int32_t mm_resample(mm_context ctx, const float* signal_data, int64_t signal_length,
                    int32_t source_sr, int32_t target_sr, MMBuffer* out);

/* Signal Generation */
int32_t mm_tone(mm_context ctx, float frequency, int32_t sr,
                int64_t length, float phi, MMBuffer* out);
int32_t mm_chirp(mm_context ctx, float fmin, float fmax, int32_t sr,
                 int64_t length, int32_t linear, MMBuffer* out);
int32_t mm_clicks(mm_context ctx, const float* times, int32_t n_times,
                  int32_t sr, int64_t length, float click_freq,
                  float click_duration, MMBuffer* out);

/* Onset Evaluation */
int32_t mm_onset_evaluate(mm_context ctx,
                          const float* reference, int32_t n_ref,
                          const float* estimated, int32_t n_est,
                          float window,
                          float* out_precision, float* out_recall, float* out_fmeasure);

/* Beat Evaluation */
int32_t mm_beat_evaluate(mm_context ctx,
                         const float* reference, int32_t n_ref,
                         const float* estimated, int32_t n_est,
                         float fmeasure_window,
                         float* out_fmeasure, float* out_cemgil, float* out_pscore,
                         float* out_cmlc, float* out_cmlt, float* out_amlc, float* out_amlt);

/* Tempo Evaluation */
int32_t mm_tempo_evaluate(mm_context ctx, float ref_tempo, float est_tempo,
                          float tolerance, float* out_pscore);

/* Chord Evaluation */
int32_t mm_chord_accuracy(mm_context ctx, const int32_t* reference, const int32_t* estimated,
                          int32_t n, float* out_accuracy);

/* Onset Detection */
int32_t mm_onset_strength(mm_context ctx, const float* signal_data, int64_t signal_length,
                          int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                          int32_t n_mels, float f_min, float f_max,
                          int32_t center, int32_t aggregate, MMBuffer* out);

int32_t mm_onset_detect(mm_context ctx, const float* signal_data, int64_t signal_length,
                        int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                        int32_t n_mels, float f_min, float f_max,
                        int32_t center,
                        int32_t pre_max, int32_t post_max,
                        int32_t pre_avg, int32_t post_avg,
                        float delta, int32_t wait,
                        int32_t backtrack,
                        MMBuffer* out);

/* Tempo Estimation */
int32_t mm_tempo(mm_context ctx, const float* signal_data, int64_t signal_length,
                 int32_t sample_rate, int32_t hop_length, int32_t n_fft,
                 int32_t n_mels, float f_min, float f_max,
                 float start_bpm, int32_t center,
                 float* out_tempo);

/* Beat Tracking */
int32_t mm_beat_track(mm_context ctx, const float* signal_data, int64_t signal_length,
                      int32_t sample_rate, int32_t hop_length, int32_t n_fft,
                      int32_t n_mels, float f_min, float f_max,
                      float start_bpm, int32_t trim,
                      float* out_tempo, MMBuffer* out_beats);

/* Tempogram */
int32_t mm_tempogram(mm_context ctx, const float* signal_data, int64_t signal_length,
                     int32_t sample_rate, int32_t hop_length, int32_t n_fft,
                     int32_t n_mels, float f_min, float f_max,
                     int32_t center, int32_t win_length,
                     MMBuffer* out);

int32_t mm_fourier_tempogram(mm_context ctx, const float* signal_data, int64_t signal_length,
                             int32_t sample_rate, int32_t hop_length, int32_t n_fft,
                             int32_t n_mels, float f_min, float f_max,
                             int32_t center, int32_t win_length,
                             MMBuffer* out);

/* Predominant Local Pulse (PLP) */
int32_t mm_plp(mm_context ctx, const float* signal_data, int64_t signal_length,
               int32_t sample_rate, int32_t hop_length, int32_t n_fft,
               int32_t n_mels, float f_min, float f_max,
               int32_t center, int32_t win_length,
               float tempo_min, float tempo_max,
               MMBuffer* out);

/* YIN Pitch Estimation */
int32_t mm_yin(mm_context ctx, const float* signal_data, int64_t signal_length,
               int32_t sample_rate, float f_min, float f_max,
               int32_t frame_length, int32_t hop_length,
               float trough_threshold, int32_t center,
               MMBuffer* out);

/* pYIN Probabilistic Pitch Estimation */
int32_t mm_pyin(mm_context ctx, const float* signal_data, int64_t signal_length,
                int32_t sample_rate, float f_min, float f_max,
                int32_t frame_length, int32_t hop_length,
                int32_t n_thresholds, float beta_alpha, float beta_beta,
                float resolution, float switch_prob, int32_t center,
                MMBuffer* out);

/* Piptrack (Parabolic Interpolation Pitch Tracking) */
int32_t mm_piptrack(mm_context ctx, const float* signal_data, int64_t signal_length,
                    int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                    int32_t win_length, float f_min, float f_max,
                    float threshold, int32_t center,
                    MMBuffer* out);

/* Tuning Estimation */
int32_t mm_estimate_tuning(mm_context ctx, const float* signal_data, int64_t signal_length,
                           int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                           int32_t win_length, float resolution,
                           int32_t bins_per_octave, int32_t center,
                           float* out_tuning);

/* HPSS (Harmonic-Percussive Source Separation) */
int32_t mm_hpss(mm_context ctx, const float* signal_data, int64_t signal_length,
                int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                int32_t win_length, int32_t center,
                int32_t kernel_size, float power, float margin,
                MMBuffer* out);

int32_t mm_harmonic(mm_context ctx, const float* signal_data, int64_t signal_length,
                    int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                    int32_t win_length, int32_t center,
                    int32_t kernel_size, float power, float margin,
                    MMBuffer* out);

int32_t mm_percussive(mm_context ctx, const float* signal_data, int64_t signal_length,
                      int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                      int32_t win_length, int32_t center,
                      int32_t kernel_size, float power, float margin,
                      MMBuffer* out);

/* Time Stretching */
int32_t mm_time_stretch(mm_context ctx, const float* signal_data, int64_t signal_length,
                        int32_t sample_rate, float rate,
                        int32_t n_fft, int32_t hop_length,
                        MMBuffer* out);

/* Pitch Shifting */
int32_t mm_pitch_shift(mm_context ctx, const float* signal_data, int64_t signal_length,
                       int32_t sample_rate, float n_steps,
                       int32_t bins_per_octave, int32_t n_fft, int32_t hop_length,
                       MMBuffer* out);

/* Trim (Silence Trimming) */
int32_t mm_trim(mm_context ctx, const float* signal_data, int64_t signal_length,
                int32_t sample_rate, float top_db,
                int32_t frame_length, int32_t hop_length,
                MMBuffer* out, int64_t* out_start, int64_t* out_end);

/* Split (Non-Silent Interval Detection) */
int32_t mm_split(mm_context ctx, const float* signal_data, int64_t signal_length,
                 int32_t sample_rate, float top_db,
                 int32_t frame_length, int32_t hop_length,
                 MMBuffer* out);

/* Preemphasis */
int32_t mm_preemphasis(mm_context ctx, const float* signal_data, int64_t signal_length,
                       int32_t sample_rate, float coef, MMBuffer* out);

/* Deemphasis */
int32_t mm_deemphasis(mm_context ctx, const float* signal_data, int64_t signal_length,
                      int32_t sample_rate, float coef, MMBuffer* out);

/* Neural Beat Decode */
int32_t mm_neural_beat_decode(mm_context ctx,
                              const float* activations, int32_t n_frames,
                              float fps, float min_bpm, float max_bpm,
                              float transition_lambda, float threshold,
                              int32_t trim,
                              float* out_tempo, MMBuffer* out_beats);

/* Neural Onset Detect */
int32_t mm_neural_onset_detect(mm_context ctx,
                               const float* activations, int32_t n_frames,
                               float fps, float threshold,
                               int32_t pre_max, int32_t post_max,
                               int32_t pre_avg, int32_t post_avg,
                               int32_t combine_method, int32_t wait,
                               MMBuffer* out);

/* Downbeat Detection */
int32_t mm_downbeat_detect(mm_context ctx,
                            const float* activations, int32_t n_frames,
                            float fps, int32_t beats_per_bar,
                            float min_bpm, float max_bpm,
                            float transition_lambda,
                            MMBuffer* out_beats, MMBuffer* out_downbeats);

/* Key Detection */
int32_t mm_key_detect(mm_context ctx,
                       const float* activations, int32_t n_frames,
                       int32_t* out_key_index, float* out_confidence,
                       MMBuffer* out_probabilities);

/* Memory */
void mm_buffer_free(MMBuffer* buf);

#endif /* METALMOM_H */
