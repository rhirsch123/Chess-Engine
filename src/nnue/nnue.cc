#include "nnue.hh"

namespace NNUE {
    int16_t hidden_weights[INPUT_SIZE][HIDDEN_SIZE];
    int16_t hidden_biases[HIDDEN_SIZE];
    int16_t hidden_layer[HIDDEN_SIZE];
    int16_t output_weights[HIDDEN_SIZE];
    int16_t output_bias;

    void init(std::string weights_file) {
        std::ifstream in(weights_file, std::ios::binary);
        if (!in) {
            perror("cannot access weights file");
            return;
        }

        in.read(reinterpret_cast<char *>(hidden_weights), sizeof(hidden_weights));
        in.read(reinterpret_cast<char *>(hidden_biases), sizeof(hidden_biases));
        in.read(reinterpret_cast<char *>(output_weights), sizeof(output_weights));
        in.read(reinterpret_cast<char *>(&output_bias), sizeof(output_bias));
    }


    int evaluate(Position& position) {
        uint8_t input[768] = {};
        uint64_t pieces = position.white_pieces | position.black_pieces;
        while (pieces) {
            int square = least_set_bit(pieces);
            pieces &= pieces - 1;

            int piece = position.board[square / 8][square % 8];
            input[square * 12 + piece - 1] = 1;
        }

        std::memcpy(hidden_layer, hidden_biases, HIDDEN_SIZE * sizeof(int16_t));
        for (int i = 0; i < INPUT_SIZE; i++) {
            // can optimize matrix multiplication because input is sparse and only 0 or 1
            if (input[i]) {
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    hidden_layer[j] += hidden_weights[i][j];
                }
            }
        }

        int output = output_bias;

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            // hidden layer activation: clipped relu
            if (hidden_layer[i] > 0) {
                if (hidden_layer[i] > QA) {
                    output += QA * output_weights[i];
                } else {
                    output += hidden_layer[i] * output_weights[i];
                }
            }
        }

        return output * SCALE / (QA * QB);
    }

    // assumes hidden layer has been efficiently updated in make_move()/unmake_move()
    int evaluate_incremental() {
    #ifdef USE_SIMD
        const int16x8_t zero = vdupq_n_s16(0);
        const int16x8_t max  = vdupq_n_s16(QA);

        int32x4_t acc_low0 = vdupq_n_s32(0);
        int32x4_t acc_high0 = vdupq_n_s32(0);
        int32x4_t acc_low1 = vdupq_n_s32(0);
        int32x4_t acc_high1 = vdupq_n_s32(0);

        for (int i = 0; i < HIDDEN_SIZE; i += 16) {
            int16x8_t layer_vals0 = vld1q_s16(hidden_layer + i);
            int16x8_t weights0 = vld1q_s16(output_weights + i);
            int16x8_t layer_vals1 = vld1q_s16(hidden_layer + i + 8);
            int16x8_t weights1 = vld1q_s16(output_weights + i + 8);

            // clamp
            layer_vals0 = vmaxq_s16(layer_vals0, zero);
            layer_vals0 = vminq_s16(layer_vals0, max);
            layer_vals1 = vmaxq_s16(layer_vals1, zero);
            layer_vals1 = vminq_s16(layer_vals1, max);

            // multiply-accumulate, widen for overflow
            acc_low0 = vmlal_s16(acc_low0, vget_low_s16(layer_vals0), vget_low_s16(weights0));
            acc_high0 = vmlal_s16(acc_high0, vget_high_s16(layer_vals0), vget_high_s16(weights0));
            acc_low1 = vmlal_s16(acc_low1, vget_low_s16(layer_vals1), vget_low_s16(weights1));
            acc_high1 = vmlal_s16(acc_high1, vget_high_s16(layer_vals1), vget_high_s16(weights1));
        }

        int32x4_t acc = vaddq_s32(vaddq_s32(acc_low0, acc_high0), vaddq_s32(acc_low1, acc_high1));
        int output = output_bias + vaddvq_s32(acc);
        return output * SCALE / (QA * QB);
    #else
        int output = output_bias;

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            if (hidden_layer[i] > 0) {
                if (hidden_layer[i] > QA) {
                    output += QA * output_weights[i];
                } else {
                    output += hidden_layer[i] * output_weights[i];
                }
            }
        }

        return output * SCALE / (QA * QB);
    #endif
    }
}