#include "nnue.hh"

namespace NNUE {
    int16_t hidden_weights[INPUT_SIZE][HIDDEN_SIZE];
    int16_t hidden_biases[HIDDEN_SIZE];
    int16_t output_weights_stm[OUTPUT_BUCKETS][HIDDEN_SIZE];
    int16_t output_weights_opp[OUTPUT_BUCKETS][HIDDEN_SIZE];
    int16_t output_bias[OUTPUT_BUCKETS];
    int16_t accumulators[2][HIDDEN_SIZE];

    void init(std::string weights_file) {
        std::ifstream in(weights_file, std::ios::binary);
        if (!in) {
            perror("cannot access weights file");
            return;
        }

        in.read(reinterpret_cast<char *>(hidden_weights), sizeof(hidden_weights));
        in.read(reinterpret_cast<char *>(hidden_biases), sizeof(hidden_biases));
        in.read(reinterpret_cast<char *>(output_weights_stm), sizeof(output_weights_stm));
        in.read(reinterpret_cast<char *>(output_weights_opp), sizeof(output_weights_opp));
        in.read(reinterpret_cast<char *>(output_bias), sizeof(output_bias));

        in.close();
    }


    int black_index(int square, int piece) {
        // flip square vertically, flip piece color
        return (square ^ 0b111000) * 12 + (piece <= 6 ? piece + 5 : piece - 7);
    }

    int get_output_bucket(uint64_t occupancy) {
        return (__builtin_popcountll(occupancy) - 2) * OUTPUT_BUCKETS / 32;
    }


    int evaluate(int board[8][8], int turn, int output_bucket) {
        uint8_t white_input[768] = {};
        uint8_t black_input[768] = {};
        for (int square = 0; square < 64; square++) {
            int row = square / 8;
            int col = square % 8;
            int piece = board[row][col];
            if (piece) {
                white_input[square * 12 + piece - 1] = 1;
                black_input[black_index(square, piece)] = 1;
            }
        }

        std::memcpy(accumulators[0], hidden_biases, HIDDEN_SIZE * sizeof(int16_t));
        std::memcpy(accumulators[1], hidden_biases, HIDDEN_SIZE * sizeof(int16_t));
        for (int i = 0; i < INPUT_SIZE; i++) {
            // can optimize matrix multiplication because input is sparse and only 0 or 1
            if (white_input[i]) {
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    accumulators[0][j] += hidden_weights[i][j];
                }
            }
            if (black_input[i]) {
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    accumulators[1][j] += hidden_weights[i][j];
                }
            }
        }

        int16_t * acc_stm = accumulators[turn];
        int16_t * acc_opp = accumulators[!turn];

        int output = output_bias[output_bucket];

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            // hidden layer activation: clipped relu
            if (acc_stm[i] > 0) {
                if (acc_stm[i] > QA) {
                    output += QA * output_weights_stm[output_bucket][i];
                } else {
                    output += acc_stm[i] * output_weights_stm[output_bucket][i];
                }
            }
            if (acc_opp[i] > 0) {
                if (acc_opp[i] > QA) {
                    output += QA * output_weights_opp[output_bucket][i];
                } else {
                    output += acc_opp[i] * output_weights_opp[output_bucket][i];
                }
            }
        }

        return output * SCALE / (QA * QB);
    }

    
    // assumes hidden layer has been efficiently updated in make_move()/unmake_move()
    int evaluate_incremental(int turn, int output_bucket) {
        int16_t * acc_stm = accumulators[turn];
        int16_t * acc_opp = accumulators[!turn];
    
    #ifdef USE_NEON

        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        // unroll x2
        for (int i = 0; i < HIDDEN_SIZE; i += 16) {
            int16x8_t layer_vals_stm0 = vld1q_s16(acc_stm + i);
            int16x8_t weights_stm0 = vld1q_s16(output_weights_stm[output_bucket] + i);
            int16x8_t layer_vals_stm1 = vld1q_s16(acc_stm + i + 8);
            int16x8_t weights_stm1 = vld1q_s16(output_weights_stm[output_bucket] + i + 8);

            // clamp
            layer_vals_stm0 = vmaxq_s16(layer_vals_stm0, zero);
            layer_vals_stm0 = vminq_s16(layer_vals_stm0, max);
            layer_vals_stm1 = vmaxq_s16(layer_vals_stm1, zero);
            layer_vals_stm1 = vminq_s16(layer_vals_stm1, max);
            
            // multiply
            int16x8_t prod_stm0 = vmulq_s16(layer_vals_stm0, weights_stm0);
            int16x8_t prod_stm1 = vmulq_s16(layer_vals_stm1, weights_stm1);

            int16x8_t layer_vals_opp0 = vld1q_s16(acc_opp + i);
            int16x8_t weights_opp0 = vld1q_s16(output_weights_opp[output_bucket] + i);
            int16x8_t layer_vals_opp1 = vld1q_s16(acc_opp + i + 8);
            int16x8_t weights_opp1 = vld1q_s16(output_weights_opp[output_bucket] + i + 8);

            // clamp
            layer_vals_opp0 = vmaxq_s16(layer_vals_opp0, zero);
            layer_vals_opp0 = vminq_s16(layer_vals_opp0, max);
            layer_vals_opp1 = vmaxq_s16(layer_vals_opp1, zero);
            layer_vals_opp1 = vminq_s16(layer_vals_opp1, max);

            // multiply
            int16x8_t prod_opp0 = vmulq_s16(layer_vals_opp0, weights_opp0);
            int16x8_t prod_opp1 = vmulq_s16(layer_vals_opp1, weights_opp1);
            
            // accumulate
            acc0 = vpadalq_s16(acc0, prod_stm0);
            acc1 = vpadalq_s16(acc1, prod_stm1);
            acc2 = vpadalq_s16(acc2, prod_opp0);
            acc3 = vpadalq_s16(acc3, prod_opp1);
        }

        int32x4_t acc = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
        int output = output_bias[output_bucket] + vaddvq_s32(acc);
        return output * SCALE / (QA * QB);

    #else

        int output = output_bias[output_bucket];

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            if (acc_stm[i] > 0) {
                if (acc_stm[i] > QA) {
                    output += QA * output_weights_stm[output_bucket][i];
                } else {
                    output += acc_stm[i] * output_weights_stm[output_bucket][i];
                }
            }
            if (acc_opp[i] > 0) {
                if (acc_opp[i] > QA) {
                    output += QA * output_weights_opp[output_bucket][i];
                } else {
                    output += acc_opp[i] * output_weights_opp[output_bucket][i];
                }
            }
        }

        return output * SCALE / (QA * QB);
    #endif
    }
}