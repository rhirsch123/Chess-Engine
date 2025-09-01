#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <vector>

#include <math.h>

#include "nnue.hh"

// SIMD optimizations
#if __ARM_NEON && __aarch64__
    #define USE_SIMD
    #include "arm_neon.h"
#endif

#define LEARN_RATE 0.001


// vectorized clipped ReLU activation = clamp(x, 0, 1)
static inline void CReLU_f32(float * v, int size) {
#ifdef USE_SIMD
    float32x4_t zero = vdupq_n_f32(0.0);
    float32x4_t one = vdupq_n_f32(1.0);

    for (int i = 0; i < size; i += 4) {
        // load 4 floats
        float32x4_t clamped = vld1q_f32(v + i);

        clamped = vmaxq_f32(clamped, zero);
        clamped = vminq_f32(clamped, one);

        // copy back to vector
        vst1q_f32(v + i, clamped);
    }
#else
    for (int i = 0; i < size; i++) {
        if (v[i] > 1) {
            v[i] = 1;
        } else if (v[i] < 0) {
            v[i] = 0;
        }
    }
#endif
}

// dot product
static inline float dot_f32(float * u, float * v, int n) {
#ifdef USE_SIMD
    float32x4_t sums = vdupq_n_f32(0);
    for (int i = 0; i < n; i += 4) {
        float32x4_t u_vals = vld1q_f32(u + i);
        float32x4_t v_vals = vld1q_f32(v + i);
        sums = vfmaq_f32(sums, u_vals, v_vals);
    }
    return vaddvq_f32(sums);
#else
    float output = 0.0;
    for (int i = 0; i < n; i++) {
        output += u[i] * v[i];
    }
    return output;
#endif
}

// add vector v to u, store in u
static inline void add_f32(float * u, float * v, int n) {
#ifdef USE_SIMD
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
float hidden_pre_activation[HIDDEN_SIZE];
float hidden_layer[HIDDEN_SIZE];
float output_weights[HIDDEN_SIZE];
float output_bias;

float infer(uint8_t * input) {
    std::memcpy(hidden_pre_activation, hidden_biases, HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < INPUT_SIZE; i++) {
        if (input[i]) {
            add_f32(hidden_pre_activation, hidden_weights + HIDDEN_SIZE * i, HIDDEN_SIZE);
        }
    }

    std::memcpy(hidden_layer, hidden_pre_activation, HIDDEN_SIZE * sizeof(float));
    // hidden layer activation
    CReLU_f32(hidden_layer, HIDDEN_SIZE);

    float output = output_bias + dot_f32(hidden_layer, output_weights, HIDDEN_SIZE);

    // output activation
    return sigmoid(output);
}


void dump_weights() {
    int16_t quantized_hidden_weights[INPUT_SIZE][HIDDEN_SIZE];
    int16_t quantized_hidden_biases[HIDDEN_SIZE];
    int16_t quantized_output_weights[HIDDEN_SIZE];
    int16_t quantized_output_bias;

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            quantized_hidden_weights[j][i] = static_cast<int>(std::round(hidden_weights[j * HIDDEN_SIZE + i] * QA));
        }

        quantized_hidden_biases[i] = static_cast<int>(std::round(hidden_biases[i] * QA));
        quantized_output_weights[i] = static_cast<int>(std::round(output_weights[i] * QB));
    }
    quantized_output_bias = static_cast<int>(std::round(output_bias * QB));

    std::ofstream out("new_nnue.bin", std::ios::binary);
	if (!out) {
        perror("error writing weights file");
    }
	out.write(reinterpret_cast<char*>(quantized_hidden_weights), sizeof(quantized_hidden_weights));
	out.write(reinterpret_cast<char*>(quantized_hidden_biases), sizeof(quantized_hidden_biases));
	out.write(reinterpret_cast<char*>(quantized_output_weights), sizeof(quantized_output_weights));
	out.write(reinterpret_cast<char*>(&quantized_output_bias), sizeof(quantized_output_bias));
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
void train(const char * positions_file, const char * evaluations_file, int num_positions, int epochs, int batch_size) {
    // init random weights
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double limit = sqrt(6.0 / INPUT_SIZE);
			hidden_weights[i * HIDDEN_SIZE + j] = -limit + 2 * limit * (rand() / (double) RAND_MAX);
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double limit = sqrt(6.0 / HIDDEN_SIZE);
		output_weights[i] = -limit + 2 * limit * (rand() / (double) RAND_MAX);

        hidden_biases[i] = 0.0;
    }
    output_bias = 0.0;

    // adam
    const float beta1 = 0.9;
    const float beta2 = 0.999;
    const float eps = 1e-8;

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

    float m_output_weights[HIDDEN_SIZE] = {};
    float v_output_weights[HIDDEN_SIZE] = {};
    float m_output_bias = 0.0;
    float v_output_bias = 0.0;

    
    FILE * f_positions = fopen(positions_file, "rb");   
    FILE * f_evals = fopen(evaluations_file, "rb");

    uint8_t * positions = (uint8_t *) malloc(batch_size * INPUT_SIZE * sizeof(uint8_t));
	int16_t * evals = (int16_t *) malloc(batch_size * sizeof(int16_t));

    for (int e = 1; e <= epochs; e++) {
        printf("\nepoch: %d\n\n", e);

        // shuffle batch order for each epoch
        int num_batches = num_positions / batch_size;
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
            int index = batch_size * indices[batch];

            fseek(f_evals, index * sizeof(int16_t), SEEK_SET);
            fread(evals, sizeof(int16_t), batch_size, f_evals);

            off_t offset = (off_t) index * INPUT_SIZE * sizeof(uint8_t);
            fseek(f_positions, offset, SEEK_SET);
            fread(positions, sizeof(uint8_t), batch_size * INPUT_SIZE, f_positions);

            float * hidden_weight_gradient = (float *) malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
            for (int i = 0; i < INPUT_SIZE; i++) {
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    hidden_weight_gradient[i * HIDDEN_SIZE + j] = 0.0;
                }
            }
            float hidden_bias_gradient[HIDDEN_SIZE] = {};
            float output_weight_gradient[HIDDEN_SIZE] = {};
            float output_bias_gradient = 0;

            for (int b = 0; b < batch_size; b++) {
                // normalize evaluations
                float y = sigmoid((float) evals[b] / SCALE);

                // output with sigmoid activation
                uint8_t * x = positions + INPUT_SIZE * b;
                float a2 = infer(x);

                float diff = a2 - y;
                loss += 0.5 * diff * diff;

                int pos_num = batch * batch_size + b;
                if (batch && pos_num % 100000 == 0) {
                    printf("position: %d\n", pos_num);
                    printf("average loss: %f\n", loss / 100000);

                    loss = 0.0;

                    if (batch && pos_num % 1000000 == 0) {
                        dump_weights();
                    }
                }

                float dL_dO = (a2 - y) * a2 * (1.0 - a2);

                output_bias_gradient += dL_dO;

                float dL_dH[HIDDEN_SIZE];

                for (int i = 0; i < HIDDEN_SIZE; i++) {
                    output_weight_gradient[i] += dL_dO * hidden_layer[i];

                    dL_dH[i] = 0.0;
                    if (hidden_pre_activation[i] > 0 && hidden_pre_activation[i] < 1) {
                        dL_dH[i] = dL_dO * output_weights[i];
                    }

                    hidden_bias_gradient[i] += dL_dH[i];
                }

                for (int i = 0; i < INPUT_SIZE; i++) {
                    if (x[i]) {
                        add_f32(hidden_weight_gradient + i * HIDDEN_SIZE, dL_dH, HIDDEN_SIZE);
                    }
                }
            }

            // update
            int step = batch + 1;
            double step_learn_rate = LEARN_RATE * sqrt(1.0 - pow(beta2, step)) / (1.0 - pow(beta1, step));

            for (int i = 0; i < HIDDEN_SIZE; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) {
                    int idx = j * HIDDEN_SIZE + i;
                    double g = hidden_weight_gradient[idx] / batch_size;

                    if (g > 1.0) g = 1.0;
                    if (g < -1.0) g = -1.0;

                    m_hidden_weights[idx] = beta1 * m_hidden_weights[idx] + (1 - beta1) * g;
                    v_hidden_weights[idx] = beta2 * v_hidden_weights[idx] + (1 - beta2) * g * g;

                    hidden_weights[idx] -= step_learn_rate * m_hidden_weights[idx] / (sqrt(v_hidden_weights[idx]) + eps);
                }

                double g_bias = hidden_bias_gradient[i] / batch_size;
                m_hidden_biases[i] = beta1 * m_hidden_biases[i] + (1 - beta1) * g_bias;
                v_hidden_biases[i] = beta2 * v_hidden_biases[i] + (1 - beta2) * g_bias * g_bias;
                hidden_biases[i] -= step_learn_rate * m_hidden_biases[i] / (sqrt(v_hidden_biases[i]) + eps);
            }

            float g_out_bias = output_bias_gradient / batch_size;
            m_output_bias = beta1 * m_output_bias + (1 - beta1) * g_out_bias;
            v_output_bias = beta2 * v_output_bias + (1 - beta2) * g_out_bias * g_out_bias;
            output_bias -= step_learn_rate * m_output_bias / (sqrt(v_output_bias) + eps);
            
            free(hidden_weight_gradient);
        }
    }

    fclose(f_evals);
    fclose(f_positions);

    free(m_hidden_weights);
    free(v_hidden_weights);
    free(positions);
    free(evals);
}


int main() {
    train("positions.bin", "evals.bin", 90000000, 2, 1024);
    dump_weights();

    free(hidden_weights);

    return 0;
}
