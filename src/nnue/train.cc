#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>

#include <math.h>

#include "nnue.hh"

struct label {
    int16_t eval;
    uint16_t game_ply;
    int8_t result;
    int8_t turn;
};

#ifdef USE_NEON
const float32x4_t v_zero = vdupq_n_f32(0.0);
const float32x4_t v_one = vdupq_n_f32(1.0);
#endif

int get_output_bucket(uint64_t occupancy) {
    return (__builtin_popcountll(occupancy) - 2) * OUTPUT_BUCKETS / 32;
}

// add vector v to u, store in u
static inline void add_f32(float * u, float * v, int n) {
#ifdef USE_NEON
    for (int i = 0; i < n; i += 4) {
        float32x4_t u_vals = vld1q_f32(u + i);
        float32x4_t v_vals = vld1q_f32(v + i);
        float32x4_t sum = vaddq_f32(u_vals, v_vals);
        vst1q_f32(u + i, sum);
    }
#else
    for (int i = 0; i < n; i++) {
        u[i] += v[i];
    }
#endif
}

// sigmoid activation
static inline float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}


float * hidden_weights = (float *) malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
float hidden_biases[HIDDEN_SIZE];
float hidden_pre_activation_stm[HIDDEN_SIZE];
float hidden_pre_activation_opp[HIDDEN_SIZE];
float hidden_layer_stm[HIDDEN_SIZE];
float hidden_layer_opp[HIDDEN_SIZE];
float output_weights_stm[OUTPUT_BUCKETS][HIDDEN_SIZE];
float output_weights_opp[OUTPUT_BUCKETS][HIDDEN_SIZE];
float output_bias[OUTPUT_BUCKETS];

float infer(uint8_t * input_stm, uint8_t * input_opp, int output_bucket) {
    std::memcpy(hidden_pre_activation_stm, hidden_biases, HIDDEN_SIZE * sizeof(float));
    std::memcpy(hidden_pre_activation_opp, hidden_biases, HIDDEN_SIZE * sizeof(float));

    for (int i = 0; i < INPUT_SIZE; i++) {
        if (input_stm[i]) {
            add_f32(hidden_pre_activation_stm, hidden_weights + HIDDEN_SIZE * i, HIDDEN_SIZE);
        }
        if (input_opp[i]) {
            add_f32(hidden_pre_activation_opp, hidden_weights + HIDDEN_SIZE * i, HIDDEN_SIZE);
        }
    }

    // hidden layer activation: clipped ReLU = clamp(x, 0, 1)
#ifdef USE_NEON
    float32x4_t sums_stm = vdupq_n_f32(0.0);
    float32x4_t sums_opp = vdupq_n_f32(0.0);
    for (int i = 0; i < HIDDEN_SIZE; i += 4) {
        float32x4_t h_stm = vld1q_f32(hidden_pre_activation_stm + i);
        float32x4_t weights_stm = vld1q_f32(output_weights_stm[output_bucket] + i);

        h_stm = vmaxq_f32(h_stm, v_zero);
        h_stm = vminq_f32(h_stm, v_one);

        sums_stm = vfmaq_f32(sums_stm, h_stm, weights_stm);
        vst1q_f32(hidden_layer_stm + i, h_stm);

        float32x4_t h_opp = vld1q_f32(hidden_pre_activation_opp + i);
        float32x4_t weights_opp = vld1q_f32(output_weights_opp[output_bucket] + i);

        h_opp = vmaxq_f32(h_opp, v_zero);
        h_opp = vminq_f32(h_opp, v_one);
        
        sums_opp = vfmaq_f32(sums_opp, h_opp, weights_opp);
        vst1q_f32(hidden_layer_opp + i, h_opp);
    }

    float output = output_bias[output_bucket] + vaddvq_f32(sums_stm) + vaddvq_f32(sums_opp);
#else
    std::memcpy(hidden_layer_stm, hidden_pre_activation_stm, HIDDEN_SIZE * sizeof(float));
    std::memcpy(hidden_layer_opp, hidden_pre_activation_opp, HIDDEN_SIZE * sizeof(float));

    float output = output_bias[output_bucket];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        if (hidden_layer_stm[i] > 0) {
            if (hidden_layer_stm[i] > 1) {
                output += output_weights_stm[output_bucket][i];
                hidden_layer_stm[i] = 1.0;
            } else {
                output += hidden_layer_stm[i] * output_weights_stm[output_bucket][i];
            }
        } else {
            hidden_layer_stm[i] = 0.0;
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        if (hidden_layer_opp[i] > 0) {
            if (hidden_layer_opp[i] > 1) {
                output += output_weights_opp[output_bucket][i];
                hidden_layer_opp[i] = 1.0;
            } else {
                output += hidden_layer_opp[i] * output_weights_opp[output_bucket][i];
            }
        } else {
            hidden_layer_opp[i] = 0.0;
        }
    }
#endif

    // output activation
    return sigmoid(output);
}


void dump_weights(std::string out_file) {
    int16_t quantized_hidden_weights[INPUT_SIZE][HIDDEN_SIZE];
    int16_t quantized_hidden_biases[HIDDEN_SIZE];
    int16_t quantized_output_weights_stm[OUTPUT_BUCKETS][HIDDEN_SIZE];
    int16_t quantized_output_weights_opp[OUTPUT_BUCKETS][HIDDEN_SIZE];
    int16_t quantized_output_bias[OUTPUT_BUCKETS];

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            quantized_hidden_weights[j][i] = static_cast<int>(std::round(hidden_weights[j * HIDDEN_SIZE + i] * QA));
        }

        quantized_hidden_biases[i] = static_cast<int>(std::round(hidden_biases[i] * QA));
    }

    for (int i = 0; i < OUTPUT_BUCKETS; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            quantized_output_weights_stm[i][j] = static_cast<int>(std::round(output_weights_stm[i][j] * QB));
            quantized_output_weights_opp[i][j] = static_cast<int>(std::round(output_weights_opp[i][j] * QB));
        }
        quantized_output_bias[i] = static_cast<int>(std::round(output_bias[i] * QA * QB));
    }

    std::ofstream out(out_file, std::ios::binary);
	if (!out) {
        perror("error writing weights file");
    }
	out.write(reinterpret_cast<char*>(quantized_hidden_weights), sizeof(quantized_hidden_weights));
	out.write(reinterpret_cast<char*>(quantized_hidden_biases), sizeof(quantized_hidden_biases));
	out.write(reinterpret_cast<char*>(quantized_output_weights_stm), sizeof(quantized_output_weights_stm));
    out.write(reinterpret_cast<char*>(quantized_output_weights_opp), sizeof(quantized_output_weights_opp));
	out.write(reinterpret_cast<char*>(quantized_output_bias), sizeof(quantized_output_bias));
}


/*
x: input, y: label, h: hidden layer, w1: hidden weights, b1: hidden biases,
a1: CReLU activation, out: output, w2: output layer weights,
b2: output bias, a2: sigmoid activation, loss: squared error

h = w1 * x + b1
a1 = CReLU(h)
out = w2 * a1 + b2
a2 = sigmoid(out)
loss = 0.5 * (a2 - y)^2

d_loss / d_a2 = a2 - y
d_loss / d_out = (d_loss / d_a2) * (d_a2 / d_sigmoid) = (a2 - y) * (a2 * (1 - a2)) -> dL_dO
d_loss / d_w2 = (d_loss / d_out) * (d_out / d_w2) = dL_dO * a1
d_loss / d_b2 = (d_loss / d_out) * (d_out / d_b2) = dL_dO * 1 = dL_dO
d_loss / d_a1 = (d_loss / d_out) * (d_out / d_a1) = dL_dO * w2
d_loss / d_h = (d_loss / d_a1) * (d_a1 / d_h) = dL_dO * w2 * (h[i] between 0, 1 ? 1 : 0) -> dL_dH
d_loss / d_w1 = (d_loss / d_h) * (d_h / d_w1) = dL_dH * x
d_loss / d_b1 = (d_loss / d_h) * (d_h / d_b1) = dL_dH * 1 = dL_dH
*/
void train(const char * positions_file, const char * labels_file, long num_positions, int epochs, int batch_size, std::string out_file) {
    // init random weights
    std::mt19937_64 gen(123);
    std::normal_distribution<float> d1(0.0, std::sqrt(2.0 / INPUT_SIZE));
    std::normal_distribution<float> d2(0.0, std::sqrt(2.0 / HIDDEN_SIZE));

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden_weights[i * HIDDEN_SIZE + j] = d1(gen);
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_biases[i] = 0.0;
    }
    for (int i = 0; i < OUTPUT_BUCKETS; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output_weights_stm[i][j] = d2(gen);
            output_weights_opp[i][j] = d2(gen);
        }
        output_bias[i] = 0.0;
    }

    float lr_min = 0.00001;
    float lr_max = 0.001;
    int num_lr_steps = 50;
    float learn_rate = lr_max;

    // adam
    const float beta1 = 0.9;
    const float beta2 = 0.999;
    const float epsilon = 1e-8;

    float * m_hidden_weights = (float *) malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float * v_hidden_weights = (float *) malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            m_hidden_weights[i * HIDDEN_SIZE + j] = 0.0;
            v_hidden_weights[i * HIDDEN_SIZE + j] = 0.0;
        }
    }
    float m_hidden_biases[HIDDEN_SIZE] = {};
    float v_hidden_biases[HIDDEN_SIZE] = {};

    float m_output_weights_stm[OUTPUT_BUCKETS][HIDDEN_SIZE] = {{}};
    float v_output_weights_stm[OUTPUT_BUCKETS][HIDDEN_SIZE] = {{}};
    float m_output_weights_opp[OUTPUT_BUCKETS][HIDDEN_SIZE] = {{}};
    float v_output_weights_opp[OUTPUT_BUCKETS][HIDDEN_SIZE] = {{}};
    float m_output_bias[OUTPUT_BUCKETS] = {}; //0.0;
    float v_output_bias[OUTPUT_BUCKETS] = {}; //0.0;

#ifdef USE_NEON
    // vector constants for fast gradient updates
    float inv_batch = 1.0 / (float) batch_size;
    const float32x4_t v_inv_batch = vdupq_n_f32(inv_batch);
	const float32x4_t v_beta1 = vdupq_n_f32(beta1);
	const float32x4_t v_beta2 = vdupq_n_f32(beta2);
	const float32x4_t v_1_minus_beta1 = vdupq_n_f32(1.0 - beta1);
	const float32x4_t v_1_minus_beta2 = vdupq_n_f32(1.0 - beta2);
	const float32x4_t v_epsilon = vdupq_n_f32(epsilon);
#endif
    
    FILE * f_positions = fopen(positions_file, "rb");   
    FILE * f_labels = fopen(labels_file, "rb");

    uint64_t * positions = (uint64_t *) malloc(batch_size * 8 * sizeof(uint64_t));
    label * labels = (label *) malloc(batch_size * sizeof(label));

    for (int e = 1; e <= epochs; e++) {
        printf("\nepoch: %d\n\n", e);

        // shuffle batch order for each epoch
        long num_batches = num_positions / batch_size;
        std::vector<int> indices(num_batches);
		for (int i = 0; i < num_batches; i++) {
			indices[i] = i;
		}
		for (int i = num_batches - 1; i > 0; i--) {
			int j = rand() % (i + 1);
			int tmp = indices[i];
			indices[i] = indices[j];
			indices[j] = tmp;
		}

        float loss = 0.0;

        for (int batch = 0; batch < num_batches; batch++) {
            // cosine annealing
            long total_batches = num_positions * epochs / batch_size;
            int current_batch = (e - 1) * num_batches + batch + 1;
            if ((current_batch % (total_batches / num_lr_steps)) == 0) {
                int lr_step = current_batch / (total_batches / num_lr_steps);
                learn_rate = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos((lr_step * 3.141592) / num_lr_steps));
                printf("\nstep %d/%d: learn rate dropped to %f\n\n", lr_step, num_lr_steps, learn_rate);
            }

            int index = batch_size * indices[batch];

            fseek(f_labels, index * sizeof(label), SEEK_SET);
            fread(labels, sizeof(label), batch_size, f_labels);

            off_t offset = (off_t) index * 8 * sizeof(uint64_t);
            fseek(f_positions, offset, SEEK_SET);
            fread(positions, sizeof(uint64_t), batch_size * 8, f_positions);

            float * hidden_weight_gradient = (float *) malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
            for (int i = 0; i < INPUT_SIZE; i++) {
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    hidden_weight_gradient[i * HIDDEN_SIZE + j] = 0.0;
                }
            }
            float hidden_bias_gradient[HIDDEN_SIZE] = {};
            float output_weight_stm_gradient[OUTPUT_BUCKETS][HIDDEN_SIZE] = {{}};
            float output_weight_opp_gradient[OUTPUT_BUCKETS][HIDDEN_SIZE] = {{}};
            float output_bias_gradient[OUTPUT_BUCKETS] = {};

            int bucket_counts[OUTPUT_BUCKETS] = {};

            for (int b = 0; b < batch_size; b++) {
                int eval = labels[b].eval;
                
                // normalize evaluations
                float y = sigmoid((float) eval / SCALE);

                uint64_t * pos = positions + 8 * b;
                uint64_t white_pieces = pos[0];
                uint64_t black_pieces = pos[1];

                int output_bucket = get_output_bucket(white_pieces | black_pieces);
                bucket_counts[output_bucket]++;

                // inputs
                uint8_t white_perspective[INPUT_SIZE] = {};
                uint8_t black_perspective[INPUT_SIZE] = {};

                for (int i = 2; i < 8; i++) {
                    uint64_t pieces = pos[i];

                    while (pieces) {
                        int square = __builtin_ctzll(pieces);
                        pieces &= pieces - 1;

                        int piece;
                        if (white_pieces & (1ULL << square)) {
                            piece = i - 2;
                        } else {
                            piece = i + 4;
                        }

                        int row = square / 8;
                        int col = square % 8;
                        int flipped_square = (7 - row) * 8 + col;
                        int flipped_piece = piece <= 5 ? piece + 6 : piece - 6;

                        white_perspective[square * 12 + piece] = 1;
                        black_perspective[flipped_square * 12 + flipped_piece] = 1;
                    }
                }

                uint8_t * x_stm; // side to move perspective
                uint8_t * x_opp;
                if (labels[b].turn == 0) {
                    x_stm = white_perspective;
                    x_opp = black_perspective;
                } else {
                    x_stm = black_perspective;
                    x_opp = white_perspective;
                }

                float a2 = infer(x_stm, x_opp, output_bucket);

                float diff = a2 - y;
                loss += 0.5 * diff * diff;

                int pos_num = batch * batch_size + b;
                if (batch && pos_num % 100000 == 0) {
                    printf("position: %d\n", pos_num);
                    printf("average loss: %f\n", loss / 100000);

                    loss = 0.0;

                    if (batch && pos_num % 1000000 == 0) {
                        dump_weights(out_file);
                    }
                }

                float dL_dO = (a2 - y) * a2 * (1.0 - a2);

                output_bias_gradient[output_bucket] += dL_dO;

                float dL_dH_stm[HIDDEN_SIZE];
                float dL_dH_opp[HIDDEN_SIZE];

                for (int i = 0; i < HIDDEN_SIZE; i++) {
                    output_weight_stm_gradient[output_bucket][i] += dL_dO * hidden_layer_stm[i];
                    dL_dH_stm[i] = 0.0;
                    if (hidden_pre_activation_stm[i] > 0 && hidden_pre_activation_stm[i] < 1) {
                        dL_dH_stm[i] = dL_dO * output_weights_stm[output_bucket][i];
                    }
                    hidden_bias_gradient[i] += dL_dH_stm[i];

                    output_weight_opp_gradient[output_bucket][i] += dL_dO * hidden_layer_opp[i];
                    dL_dH_opp[i] = 0.0;
                    if (hidden_pre_activation_opp[i] > 0 && hidden_pre_activation_opp[i] < 1) {
                        dL_dH_opp[i] = dL_dO * output_weights_opp[output_bucket][i];
                    }
                    hidden_bias_gradient[i] += dL_dH_opp[i];
                }

                for (int i = 0; i < INPUT_SIZE; i++) {
                    if (x_stm[i]) {
                        add_f32(hidden_weight_gradient + i * HIDDEN_SIZE, dL_dH_stm, HIDDEN_SIZE);
                    }
                    if (x_opp[i]) {
                        add_f32(hidden_weight_gradient + i * HIDDEN_SIZE, dL_dH_opp, HIDDEN_SIZE);
                    }
                }
            }

            // update
            int step = (e - 1) * num_batches + batch + 1;

            for (int i = 0; i < HIDDEN_SIZE; i++) {
                #ifndef USE_NEON
                for (int j = 0; j < INPUT_SIZE; j++) {
                    int idx = j * HIDDEN_SIZE + i;
                    float g = hidden_weight_gradient[idx] / batch_size;

                    m_hidden_weights[idx] = beta1 * m_hidden_weights[idx] + (1 - beta1) * g;
                    v_hidden_weights[idx] = beta2 * v_hidden_weights[idx] + (1 - beta2) * g * g;

                    float m = m_hidden_weights[idx] / (1 - pow(beta1, step));
                    float v = v_hidden_weights[idx] / (1 - pow(beta2, step));

                    hidden_weights[idx] -= (learn_rate * m) / (sqrt(v) + epsilon);
                }
                #endif

                float g_bias = hidden_bias_gradient[i] / batch_size;
                m_hidden_biases[i] = beta1 * m_hidden_biases[i] + (1 - beta1) * g_bias;
                v_hidden_biases[i] = beta2 * v_hidden_biases[i] + (1 - beta2) * g_bias * g_bias;
                float m = m_hidden_biases[i] / (1 - pow(beta1, step));
                float v = v_hidden_biases[i] / (1 - pow(beta2, step));
                hidden_biases[i] -= (learn_rate * m) / (sqrt(v) + epsilon);
            }

            #ifdef USE_NEON // ~2x speed
            float32x4_t v_lr = vdupq_n_f32(learn_rate);
            float32x4_t m_bias_correct = vdupq_n_f32(1 - pow(beta1, step));
            float32x4_t v_bias_correct = vdupq_n_f32(1 - pow(beta2, step));
            for (int i = 0; i < INPUT_SIZE; i++) {
                int base = i * HIDDEN_SIZE;
                for (int j = 0; j < HIDDEN_SIZE; j += 4) {
                    int idx = base + j;

                    // load gradients, divide by batch_size
                    float32x4_t grads = vld1q_f32(hidden_weight_gradient + idx);
                    grads = vmulq_f32(grads, v_inv_batch);

                    // g^2
                    float32x4_t grads_sq = vmulq_f32(grads, grads);

                    // load
                    float32x4_t m = vld1q_f32(m_hidden_weights + idx);
                    float32x4_t v = vld1q_f32(v_hidden_weights + idx);
                    float32x4_t w = vld1q_f32(hidden_weights + idx);

                    // m = beta1 * m + (1 - beta1) * g
                    float32x4_t m_new = vfmaq_f32(vmulq_f32(v_beta1, m), v_1_minus_beta1, grads);

                    // v = beta2 * v + (1 - beta2) * g * g
                    float32x4_t v_new = vfmaq_f32(vmulq_f32(v_beta2, v), v_1_minus_beta2, grads_sq);

                    // bias correction
                    float32x4_t m_hat = vdivq_f32(m_new, m_bias_correct);
                    float32x4_t v_hat = vdivq_f32(v_new, v_bias_correct);

                    // denom = sqrt(v_hat) + epsilon
                    float32x4_t v_sqrt = vsqrtq_f32(v_hat);
                    float32x4_t denom = vaddq_f32(v_sqrt, v_epsilon);

                    // delta = learn_rate * m_hat / denom
                    float32x4_t ratio = vdivq_f32(m_hat, denom);
                    float32x4_t delta = vmulq_f32(v_lr, ratio);

                    // w -= delta
                    float32x4_t w_new = vsubq_f32(w, delta);

                    // store results
                    vst1q_f32(m_hidden_weights + idx, m_new);
                    vst1q_f32(v_hidden_weights + idx, v_new);
                    vst1q_f32(hidden_weights + idx, w_new);
                }
            }
            #endif

            for (int bucket = 0; bucket < OUTPUT_BUCKETS; bucket++) {
                if (bucket_counts[bucket] == 0) {
                    continue;
                }

                float g_out_bias = output_bias_gradient[bucket] / bucket_counts[bucket];
                m_output_bias[bucket] = beta1 * m_output_bias[bucket] + (1 - beta1) * g_out_bias;
                v_output_bias[bucket] = beta2 * v_output_bias[bucket] + (1 - beta2) * g_out_bias * g_out_bias;
                float m = m_output_bias[bucket] / (1 - pow(beta1, step));
                float v = v_output_bias[bucket] / (1 - pow(beta2, step));
                output_bias[bucket] -= (learn_rate * m) / (sqrt(v) + epsilon);


                for (int i = 0; i < HIDDEN_SIZE; i++) {
                    double g = output_weight_stm_gradient[bucket][i] / bucket_counts[bucket];

                    m_output_weights_stm[bucket][i] = beta1 * m_output_weights_stm[bucket][i] + (1 - beta1) * g;
                    v_output_weights_stm[bucket][i] = beta2 * v_output_weights_stm[bucket][i] + (1 - beta2) * g * g;
                    float m = m_output_weights_stm[bucket][i] / (1 - pow(beta1, step));
                    float v = v_output_weights_stm[bucket][i] / (1 - pow(beta2, step));
                    output_weights_stm[bucket][i] -= (learn_rate * m) / (sqrt(v) + epsilon);


                    g = output_weight_opp_gradient[bucket][i] / bucket_counts[bucket];

                    m_output_weights_opp[bucket][i] = beta1 * m_output_weights_opp[bucket][i] + (1 - beta1) * g;
                    v_output_weights_opp[bucket][i] = beta2 * v_output_weights_opp[bucket][i] + (1 - beta2) * g * g;
                    m = m_output_weights_opp[bucket][i] / (1 - pow(beta1, step));
                    v = v_output_weights_opp[bucket][i] / (1 - pow(beta2, step));
                    output_weights_opp[bucket][i] -= (learn_rate * m) / (sqrt(v) + epsilon);

                    // prevent overflow for quantized inference
                    if (output_weights_stm[bucket][i] > 1.98) output_weights_stm[bucket][i] = 1.98;
                    if (output_weights_stm[bucket][i] < -1.98) output_weights_stm[bucket][i] = -1.98;
                    if (output_weights_opp[bucket][i] > 1.98) output_weights_opp[bucket][i] = 1.98;
                    if (output_weights_opp[bucket][i] < -1.98) output_weights_opp[bucket][i] = -1.98;
                }
            }
            
            free(hidden_weight_gradient);
        }
    }

    fclose(f_labels);
    fclose(f_positions);

    free(m_hidden_weights);
    free(v_hidden_weights);
    free(positions);
    free(labels);
}


int main() {
    std::string out_file = "new_nnue.bin";
    train("positions.bin", "labels.bin", 1560000000, 3, 1024, out_file);

    dump_weights(out_file);

    free(hidden_weights);

    return 0;
}