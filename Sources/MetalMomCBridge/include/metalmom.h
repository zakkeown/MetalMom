#ifndef METALMOM_H
#define METALMOM_H

#include <stdint.h>

/* Status codes */
#define MM_OK 0
#define MM_ERR_INVALID_INPUT -1
#define MM_ERR_METAL_UNAVAILABLE -2
#define MM_ERR_ALLOC_FAILED -3
#define MM_ERR_INTERNAL -4

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

/* Chord Recognition */
int32_t mm_chord_detect(mm_context ctx,
                         const float* activations, int32_t n_frames,
                         int32_t n_classes, const float* transition_scores,
                         float self_transition_bias,
                         MMBuffer* out);

/* Piano Transcription */
int32_t mm_piano_transcribe(mm_context ctx,
                             const float* activations, int32_t n_frames,
                             float threshold, int32_t min_duration,
                             int32_t use_hmm,
                             MMBuffer* out);

/* CQT (Constant-Q Transform) */
int32_t mm_cqt(mm_context ctx, const float* signal_data, int64_t signal_length,
               int32_t sample_rate, int32_t hop_length,
               float f_min, float f_max, int32_t bins_per_octave,
               int32_t n_fft, int32_t center, MMBuffer* out);

/* VQT (Variable-Q Transform) */
int32_t mm_vqt(mm_context ctx, const float* signal_data, int64_t signal_length,
               int32_t sample_rate, int32_t hop_length,
               float f_min, float f_max, int32_t bins_per_octave,
               float gamma, int32_t n_fft, int32_t center, MMBuffer* out);

/* Hybrid CQT */
int32_t mm_hybrid_cqt(mm_context ctx, const float* signal_data, int64_t signal_length,
                       int32_t sample_rate, int32_t hop_length,
                       float f_min, float f_max, int32_t bins_per_octave,
                       int32_t n_fft, int32_t center, MMBuffer* out);

/* Reassigned Spectrogram */
int32_t mm_reassigned_spectrogram(mm_context ctx, const float* signal_data, int64_t signal_length,
                                   int32_t sample_rate, int32_t n_fft, int32_t hop_length,
                                   int32_t win_length, int32_t center, MMBuffer* out);

/* Phase Vocoder */
int32_t mm_phase_vocoder(mm_context ctx, const float* stft_data, int64_t stft_count,
                          int32_t n_freqs, int32_t n_frames, int32_t sample_rate,
                          float rate, int32_t hop_length, MMBuffer* out);

/* Griffin-Lim */
int32_t mm_griffinlim(mm_context ctx, const float* mag_data, int64_t mag_count,
                       int32_t n_freqs, int32_t n_frames, int32_t sample_rate,
                       int32_t n_iter, int32_t hop_length, int32_t win_length,
                       int32_t center, int64_t output_length, MMBuffer* out);

/* Griffin-Lim CQT */
int32_t mm_griffinlim_cqt(mm_context ctx, const float* mag_data, int64_t mag_count,
                           int32_t n_bins, int32_t n_frames, int32_t sr,
                           int32_t n_iter, int32_t hop_length,
                           float fmin, int32_t bins_per_octave,
                           MMBuffer* out);

/* PCEN (Per-Channel Energy Normalization) */
int32_t mm_pcen(mm_context ctx, const float* data, int64_t count,
                int32_t n_bands, int32_t n_frames, int32_t sample_rate,
                int32_t hop_length, float gain, float bias,
                float power, float time_constant, float eps,
                MMBuffer* out);

/* A-Weighting */
int32_t mm_a_weighting(mm_context ctx, const float* frequencies, int32_t n_freqs,
                       MMBuffer* out);

/* Chroma CQT */
int32_t mm_chroma_cqt(mm_context ctx, const float* signal_data, int64_t signal_length,
                       int32_t sample_rate, int32_t hop_length,
                       float f_min, int32_t bins_per_octave, int32_t n_octaves,
                       int32_t n_chroma, float norm, MMBuffer* out);

/* Chroma VQT */
int32_t mm_chroma_vqt(mm_context ctx, const float* signal_data, int64_t signal_length,
                       int32_t sample_rate, int32_t hop_length,
                       float f_min, int32_t bins_per_octave, int32_t n_octaves,
                       float gamma, int32_t n_chroma, float norm, MMBuffer* out);

/* Chroma CENS */
int32_t mm_chroma_cens(mm_context ctx, const float* signal_data, int64_t signal_length,
                        int32_t sample_rate, int32_t hop_length,
                        float f_min, int32_t bins_per_octave, int32_t n_octaves,
                        int32_t n_chroma, int32_t win_len_smooth, MMBuffer* out);

/* Feature Inversion: Mel to Audio */
int32_t mm_mel_to_audio(mm_context ctx, const float* mel_data, int64_t mel_count,
                         int32_t n_mels, int32_t n_frames, int32_t sample_rate,
                         int32_t n_fft, int32_t hop_length, int32_t win_length,
                         int32_t center, int32_t n_iter, float power,
                         float f_min, float f_max, int64_t output_length,
                         MMBuffer* out);

/* Feature Inversion: MFCC to Mel */
int32_t mm_mfcc_to_mel(mm_context ctx, const float* mfcc_data, int64_t mfcc_count,
                        int32_t n_mfcc, int32_t n_frames, int32_t sample_rate,
                        int32_t n_mels, MMBuffer* out);

/* NMF (Non-negative Matrix Factorization) */
int32_t mm_nmf(mm_context ctx, const float* data, int32_t n_features, int32_t n_samples,
               int32_t sample_rate, int32_t n_components, int32_t n_iter,
               int32_t objective, MMBuffer* out_w, MMBuffer* out_h);

/* Nearest-Neighbor Filter */
int32_t mm_nn_filter(mm_context ctx, const float* data, int32_t n_features, int32_t n_frames,
                     int32_t sample_rate, int32_t k, int32_t metric, int32_t aggregate,
                     int32_t exclude_self, MMBuffer* out);

/* Recurrence Matrix */
int32_t mm_recurrence_matrix(mm_context ctx, const float* data, int64_t count,
                              int32_t n_features, int32_t n_frames,
                              int32_t mode, float mode_param,
                              int32_t metric, int32_t symmetric,
                              MMBuffer* out);

/* Cross-Similarity Matrix */
int32_t mm_cross_similarity(mm_context ctx, const float* data_a, int64_t count_a,
                             const float* data_b, int64_t count_b,
                             int32_t n_features, int32_t n_frames_a, int32_t n_frames_b,
                             int32_t metric, MMBuffer* out);

/* Dynamic Time Warping */
int32_t mm_dtw(mm_context ctx, const float* data, int64_t count,
               int32_t rows, int32_t cols,
               int32_t step_pattern, int32_t band_width,
               MMBuffer* out);

/* Agglomerative Segmentation */
int32_t mm_agglomerative(mm_context ctx, const float* data, int64_t count,
                          int32_t n_features, int32_t n_frames,
                          int32_t n_segments, MMBuffer* out);

/* Viterbi Decoding (HMM) */
int32_t mm_viterbi(mm_context ctx,
                   const float* log_obs_data, int32_t n_frames, int32_t n_states,
                   const float* log_initial, const float* log_transition,
                   MMBuffer* out);

/* Viterbi Decoding (Discriminative / CRF) */
int32_t mm_viterbi_discriminative(mm_context ctx,
                                   const float* unary_data, int32_t n_frames, int32_t n_states,
                                   const float* pairwise_data,
                                   MMBuffer* out);

/* Unit Conversions */
int32_t mm_hz_to_midi(mm_context ctx, const float* data, int64_t count, MMBuffer* out);
int32_t mm_midi_to_hz(mm_context ctx, const float* data, int64_t count, MMBuffer* out);
int32_t mm_times_to_frames(mm_context ctx, const float* data, int64_t count,
                            int32_t sr, int32_t hop_length, MMBuffer* out);
int32_t mm_frames_to_time(mm_context ctx, const float* data, int64_t count,
                           int32_t sr, int32_t hop_length, MMBuffer* out);
int32_t mm_fft_frequencies(mm_context ctx, int32_t sr, int32_t n_fft, MMBuffer* out);
int32_t mm_mel_frequencies(mm_context ctx, int32_t n_mels, float f_min, float f_max, MMBuffer* out);

/* Semitone Bandpass Filterbank */
int32_t mm_semitone_filterbank(mm_context ctx, const float* data, int64_t count,
                                int32_t sr, int32_t midi_low, int32_t midi_high,
                                int32_t order, MMBuffer* out);
int32_t mm_semitone_frequencies(mm_context ctx, int32_t midi_low, int32_t midi_high,
                                 MMBuffer* out);

/* Model Prediction (CoreML Inference) */
int32_t mm_model_predict(mm_context ctx, const char* model_path,
                          const float* input_data, const int32_t* input_shape,
                          int32_t input_shape_len, int32_t input_count,
                          MMBuffer* out);

/* Memory */
void mm_buffer_free(MMBuffer* buf);

#endif /* METALMOM_H */
