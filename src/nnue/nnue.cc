#include "nnue.hh"

namespace NNUE {
    alignas(64) int16_t accumulators[1024][2][L1_SIZE];
    alignas(64) AccInfo acc_info[1024];

    alignas(64) int16_t l1_weights[INPUT_SIZE][L1_SIZE];
    alignas(64) int16_t l1_biases[L1_SIZE];
    alignas(64) int8_t l2_weights[L1_SIZE][L2_SIZE * 2];
    alignas(64) float l2_biases[L2_SIZE];
    alignas(64) float l3_weights[L2_SIZE][L3_SIZE];
    alignas(64) float l3_biases[L3_SIZE];
    alignas(64) float output_weights[L3_SIZE];
                float output_bias;

    static constexpr int jump32 = REGISTER_WIDTH / 32;
    static constexpr int l2_chunks = L2_SIZE / jump32;
    static constexpr int l3_chunks = L3_SIZE / jump32;

    alignas(64) vec_i32 l2_acc[l2_chunks];
    alignas(64) float l2_buff[L2_SIZE];
    alignas(64) vec_f32 l3_acc[l3_chunks];

    alignas(64) uint16_t active_indices[L1_SIZE];
    int num_active = 0;

    #if USE_NEON || USE_AVX2
        // indexed by every possible uint16 mask of active indices
        // holds corresponding contiguous array of indices
        alignas(64) uint16_t active_table[1 << 16][16];
    #endif


    static std::filesystem::path get_executable_dir() {
        #if defined(__APPLE__)
            uint32_t size = 0;
            _NSGetExecutablePath(NULL, &size);
            std::vector<char> buf(size);
            _NSGetExecutablePath(buf.data(), &size);
            return std::filesystem::canonical(buf.data()).parent_path();
        #elif defined(__linux__)
            return std::filesystem::canonical("/proc/self/exe").parent_path();
        #elif defined(_WIN32)
            static constexpr int MAX_PATH = 1024;
            char path[MAX_PATH];
            GetModuleFileNameA(NULL, path, MAX_PATH);
            return std::filesystem::canonical(path).parent_path();
        #endif
    }

    void init() {
        auto dir = get_executable_dir();
        std::string nnue_path = dir.string() + "/nnue/" + nnue_file;

        std::ifstream in(nnue_path, std::ios::binary);
        if (!in) {
            fprintf(stderr, "Error: cannot access weights file\n");
            return;
        }

        float * fl1_weights = (float *) malloc(INPUT_SIZE * L1_SIZE * sizeof(float));
        float fl1_biases[L1_SIZE];
        float * fl2_weights_stm = (float *) malloc(L1_SIZE * L2_SIZE * sizeof(float));
        float * fl2_weights_opp = (float *) malloc(L1_SIZE * L2_SIZE * sizeof(float));

        in.read(reinterpret_cast<char *>(fl1_weights), INPUT_SIZE * L1_SIZE * sizeof(float));
        in.read(reinterpret_cast<char *>(fl1_biases), sizeof(fl1_biases));

        in.read(reinterpret_cast<char *>(fl2_weights_stm), L1_SIZE * L2_SIZE * sizeof(float));
        in.read(reinterpret_cast<char *>(fl2_weights_opp), L1_SIZE * L2_SIZE * sizeof(float));
        in.read(reinterpret_cast<char *>(l2_biases), sizeof(l2_biases));

        in.read(reinterpret_cast<char *>(l3_weights), sizeof(l3_weights));
        in.read(reinterpret_cast<char *>(l3_biases), sizeof(l3_biases));

        in.read(reinterpret_cast<char *>(output_weights), sizeof(output_weights));
        in.read(reinterpret_cast<char *>(&output_bias), sizeof(output_bias));

        in.close();

        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < L1_SIZE; j++) {
                l1_weights[i][j] = static_cast<int>(std::round(fl1_weights[i * L1_SIZE + j] * QA));
            }
        }
        for (int i = 0; i < L1_SIZE; i++) {
            l1_biases[i] = static_cast<int>(std::round(fl1_biases[i] * QA));
        }

        for (int i = 0; i < L1_SIZE; i++) {
            for (int j = 0; j < L2_SIZE; j++) {
                // l2 weights are interleaved side to move and opponent
                l2_weights[i][j * 2] = static_cast<int>(std::round(fl2_weights_stm[i * L2_SIZE + j] * QB));
                l2_weights[i][j * 2 + 1] = static_cast<int>(std::round(fl2_weights_opp[i * L2_SIZE + j] * QB));
            }
        }
        for (int i = 0; i < L2_SIZE; i++) {
            l2_biases[i] *= QA * QB;
        }

        for (int i = 0; i < L3_SIZE; i++) {
            l3_biases[i] *= QA * QB;
        }

        output_bias *= QA * QB;

        free(fl1_weights);
        free(fl2_weights_stm);
        free(fl2_weights_opp);


        #if USE_NEON || USE_AVX2
        std::memset(active_table, 0, sizeof(active_table));
        for (int i = 0; i < (1 << 16); i++) {
            uint32_t mask = i;
            int n = 0;
            while (mask) {
                uint16_t idx = __builtin_ctz(mask);
                mask &= mask - 1;

                active_table[i][n++] = idx;
            }
        }
        #endif
    }


    void reset_accumulators(Position& position) {
        int ply = position.half_moves;

        std::memcpy(accumulators[ply][WHITE], l1_biases, L1_SIZE * sizeof(int16_t));
        std::memcpy(accumulators[ply][BLACK], l1_biases, L1_SIZE * sizeof(int16_t));

        int white_king = lsb(position.piece_maps[KING][WHITE]);
        int black_king = lsb(position.piece_maps[KING][BLACK]);
        bool white_mirror = is_mirrored(white_king);
        bool black_mirror = is_mirrored(black_king);

        for (int square = 0; square < 64; square++) {
            int piece = position.board[square];
            if (piece) {
                int white_idx = white_index(square, piece, white_mirror);
                int black_idx = black_index(square, piece, black_mirror);

                for (int j = 0; j < L1_SIZE; j++) {
                    accumulators[ply][WHITE][j] += l1_weights[white_idx][j];
                }
                for (int j = 0; j < L1_SIZE; j++) {
                    accumulators[ply][BLACK][j] += l1_weights[black_idx][j];
                }
            }
        }

        acc_info[ply].clean = true;
    }


    // not optimized - used for debugging
    int evaluate(Position& position) {
        reset_accumulators(position);

        int ply = position.half_moves;
        int turn = position.turn;

        int16_t* acc_stm = accumulators[ply][turn];
        int16_t* acc_opp = accumulators[ply][!turn];

        float l2_layer[L2_SIZE];
        for (int i = 0; i < L2_SIZE; i++) {
            l2_layer[i] = l2_biases[i];
        }
        for (int i = 0; i < L1_SIZE; i++) {
            int16_t a_stm = acc_stm[i] < 0 ? 0 : (acc_stm[i] > QA ? QA : acc_stm[i]);
            int16_t a_opp = acc_opp[i] < 0 ? 0 : (acc_opp[i] > QA ? QA : acc_opp[i]);
            for (int j = 0; j < L2_SIZE; j++) {
                l2_layer[j] += a_stm * l2_weights[i][j * 2];
                l2_layer[j] += a_opp * l2_weights[i][j * 2 + 1];
            }
        }

        // CReLU
        for (int i = 0; i < L2_SIZE; i++) {
            if (l2_layer[i] < 0) {
                l2_layer[i] = 0;
            } else if (l2_layer[i] > QA * QB) {
                l2_layer[i] = QA * QB;
            }
        }

        float l3_layer[L3_SIZE];
        for (int i = 0; i < L3_SIZE; i++) {
            l3_layer[i] = l3_biases[i];
        }
        for (int i = 0; i < L2_SIZE; i++) {
            for (int j = 0; j < L3_SIZE; j++) {
                l3_layer[j] += l2_layer[i] * l3_weights[i][j];
            }
        }

        // CReLU
        for (int i = 0; i < L3_SIZE; i++) {
            if (l3_layer[i] < 0) {
                l3_layer[i] = 0;
            } else if (l3_layer[i] > QA * QB) {
                l3_layer[i] = QA * QB;
            }
        }

        float output = output_bias;
        for (int i = 0; i < L3_SIZE; i++) {
            output += l3_layer[i] * output_weights[i];
        }

        return output * (float) SCALE / (QA * QB);
    }


    // populate active_indices where either side's accumulator value will be nonzero after crelu
    void set_active(int ply) {
        num_active = 0;

        const vec_i16* acc_white = reinterpret_cast<const vec_i16*>(accumulators[ply][WHITE]);
        const vec_i16* acc_black = reinterpret_cast<const vec_i16*>(accumulators[ply][BLACK]);

    #if USE_NEON
    
        static constexpr uint16_t nnz_mask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
        static constexpr int num_chunks = L1_SIZE / 16;

        uint16x8_t base = vdupq_n_u16(0);
        const uint16x8_t increment = vdupq_n_u16(16);

        for (int i = 0; i < num_chunks; i++) {
            const int16x8_t white0 = acc_white[i * 2];
            const int16x8_t black0 = acc_black[i * 2];
            const int16x8_t white1 = acc_white[i * 2 + 1];
            const int16x8_t black1 = acc_black[i * 2 + 1];

            const uint16x8_t cmp0 = vorrq_u16(vcgtq_s16(white0, v_zero_i16), vcgtq_s16(black0, v_zero_i16));
            const uint16x8_t cmp1 = vorrq_u16(vcgtq_s16(white1, v_zero_i16), vcgtq_s16(black1, v_zero_i16));

            const uint8_t m0 = vaddvq_u16(vandq_u16(cmp0, vld1q_u16(nnz_mask)));
            const uint8_t m1 = vaddvq_u16(vandq_u16(cmp1, vld1q_u16(nnz_mask)));
            const uint16_t idx = m0 | (m1 << 8);

            const uint16x8_t idxs0 = vld1q_u16(active_table[idx]);
            const uint16x8_t idxs1 = vld1q_u16(active_table[idx] + 8);

            vst1q_u16(active_indices + num_active, vaddq_u16(idxs0, base));
            vst1q_u16(active_indices + num_active + 8, vaddq_u16(idxs1, base));
            base = vaddq_u16(base, increment);
            num_active += __builtin_popcount(idx);
        }
    
    #elif USE_AVX512_VBMI2

        static constexpr int num_chunks = L1_SIZE / 32;

        __m512i base = _mm512_set_epi16(
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0
        );
        const __m512i increment = _mm512_set1_epi16(32);

        for (int i = 0; i < num_chunks; i++) {
            const uint32_t mask = _mm512_cmpgt_epi16_mask(acc_white[i], v_zero_i16)
                                | _mm512_cmpgt_epi16_mask(acc_black[i], v_zero_i16);
            
            const __m512i indices = _mm512_maskz_compress_epi16(mask, base);
            _mm512_storeu_epi16(active_indices + num_active, indices);
            base = _mm512_add_epi16(base, increment);
            num_active += __builtin_popcount(mask);
        }

    #elif USE_AVX512

        static constexpr int num_chunks = L1_SIZE / 32;

        __m512i base = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0);
        const __m512i increment = _mm512_set1_epi32(16);

        for (int i = 0; i < num_chunks; i++) {
            const uint32_t mask = _mm512_cmpgt_epi16_mask(acc_white[i], v_zero_i16)
                                | _mm512_cmpgt_epi16_mask(acc_black[i], v_zero_i16);

            uint16_t m0 = mask & 0xFFFF;
            uint16_t m1 = mask >> 16;

            __m512i indices = _mm512_maskz_compress_epi32(m0, base);
            _mm512_mask_cvtepi32_storeu_epi16(active_indices + num_active, 0xFFFF, indices);
            base = _mm512_add_epi32(base, increment);
            num_active += __builtin_popcount(m0);

            indices = _mm512_maskz_compress_epi32(m1, base);
            _mm512_mask_cvtepi32_storeu_epi16(active_indices + num_active, 0xFFFF, indices);
            base = _mm512_add_epi32(base, increment);
            num_active += __builtin_popcount(m1);
        }

    #elif USE_AVX2

        static constexpr int num_chunks = L1_SIZE / 32;

        __m256i base = _mm256_set1_epi16(0);
        const __m256i increment = _mm256_set1_epi16(16);

        for (int i = 0; i < num_chunks; i++) {
            const __m256i white0 = acc_white[i * 2];
            const __m256i black0 = acc_black[i * 2];
            const __m256i white1 = acc_white[i * 2 + 1];
            const __m256i black1 = acc_black[i * 2 + 1];

            const __m256i cmp0 = _mm256_or_si256(_mm256_cmpgt_epi16(white0, v_zero_i16), _mm256_cmpgt_epi16(black0, v_zero_i16));
            const __m256i cmp1 = _mm256_or_si256(_mm256_cmpgt_epi16(white1, v_zero_i16), _mm256_cmpgt_epi16(black1, v_zero_i16));

            // order: first 8 of cmp0, first 8 of cmp1, second 8 of cmp0, second 8 of cmp1
            uint32_t mask = _mm256_movemask_epi8(_mm256_packs_epi16(cmp0, cmp1));

            #if defined(__BMI2__)
            uint16_t m0 = _pext_u32(mask, 0x00FF00FF);
            uint16_t m1 = _pext_u32(mask, 0xFF00FF00);
            #else
            uint16_t m0 = (mask & 0xFF) | ((mask & 0xFF0000) >> 8);
            uint16_t m1 = ((mask & 0xFF00) >> 8) | ((mask & 0xFF000000) >> 16);
            #endif

            __m256i indices = _mm256_load_si256(reinterpret_cast<const __m256i *>(active_table[m0]));
            indices = _mm256_add_epi16(indices, base);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(active_indices + num_active), indices);
            base = _mm256_add_epi16(base, increment);
            num_active += __builtin_popcount(m0);

            indices = _mm256_load_si256(reinterpret_cast<const __m256i *>(active_table[m1]));
            indices = _mm256_add_epi16(indices, base);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(active_indices + num_active), indices);
            base = _mm256_add_epi16(base, increment);
            num_active += __builtin_popcount(m1);
        }

    #endif
    }


    void set_dirty(int ply, DirtyPieces& dps) {
        acc_info[ply].dps = dps;
        acc_info[ply].clean = false;
    }


    // efficiently update accumulator - only have to worry about the few indices that changed this move
    void update_accumulators(int ply) {
        acc_info[ply].clean = true;
        DirtyPieces dps = acc_info[ply].dps;

        int16_t* acc_white = accumulators[ply][WHITE];
        int16_t* acc_black = accumulators[ply][BLACK];
        int16_t* prev_white = accumulators[ply - 1][WHITE];
        int16_t* prev_black = accumulators[ply - 1][BLACK];

        // compilers should vectorize this automatically
        if (dps.type == QUIET || dps.type == PROMOTION) {
            // add sub
            for (int i = 0; i < L1_SIZE; i++) {
                acc_white[i] = prev_white[i] + l1_weights[dps.white_add0][i] - l1_weights[dps.white_sub0][i];
                acc_black[i] = prev_black[i] + l1_weights[dps.black_add0][i] - l1_weights[dps.black_sub0][i];
            }
        } else if (dps.type == CAPTURE || dps.type == CAP_PROMO || dps.type == EN_PASSANT) {
            // add sub sub
            for (int i = 0; i < L1_SIZE; i++) {
                acc_white[i] = prev_white[i]
                  + l1_weights[dps.white_add0][i] - l1_weights[dps.white_sub0][i] - l1_weights[dps.white_sub1][i];
                acc_black[i] = prev_black[i]
                  + l1_weights[dps.black_add0][i] - l1_weights[dps.black_sub0][i] - l1_weights[dps.black_sub1][i];
            }
        } else if (dps.type == CASTLE) {
            // add add sub sub
            for (int i = 0; i < L1_SIZE; i++) {
                acc_white[i] = prev_white[i]
                  + l1_weights[dps.white_add0][i] + l1_weights[dps.white_add1][i]
                  - l1_weights[dps.white_sub0][i] - l1_weights[dps.white_sub1][i];
                acc_black[i] = prev_black[i]
                  + l1_weights[dps.black_add0][i] + l1_weights[dps.black_add1][i]
                  - l1_weights[dps.black_sub0][i] - l1_weights[dps.black_sub1][i];
            }
        } else {
            // null move
            std::memcpy(acc_white, prev_white, L1_SIZE * sizeof(int16_t));
            std::memcpy(acc_black, prev_black, L1_SIZE * sizeof(int16_t));
        }
    }


    void clean_accumulators(int ply) {
        int last_clean = ply;
        while (!acc_info[last_clean].clean) {
            last_clean--;
        }

        for (int i = last_clean + 1; i <= ply; i++) {
            update_accumulators(i);
        }
    }


    int evaluate_incremental(int ply, int turn) {
        clean_accumulators(ply);
        set_active(ply);

        const int16_t* acc_stm = accumulators[ply][turn];
        const int16_t* acc_opp = accumulators[ply][!turn];

        for (int i = 0; i < l2_chunks; i++) {
            l2_acc[i] = v_zero_i32;
        }

        // usually only ~10% of the accumulator is active after crelu
        // inference is sped up by looping through non-zero indices

        // to take advantage of vector operations that add adjacent elements,
        // two accumulator values are interleaved and handled at the same time

    #ifdef USE_NEON

        for (int i = 0; i < num_active; i++) {
            int idx = active_indices[i];
            uint16_t a_stm = acc_stm[idx] < 0 ? 0 : (acc_stm[idx] > QA ? QA : acc_stm[idx]);
            uint16_t a_opp = acc_opp[idx] < 0 ? 0 : (acc_opp[idx] > QA ? QA : acc_opp[idx]);

            uint32_t concat = (uint32_t) a_stm | ((uint32_t) a_opp << 16);
            int16x8_t interleaved = vreinterpretq_s16_s32(vdupq_n_s32(concat));

            int8x16_t w0 = vld1q_s8(l2_weights[idx]);
            int8x16_t w1 = vld1q_s8(l2_weights[idx] + 16);
            int16x8_t weights0 = vmovl_s8(vget_low_s8(w0));
            int16x8_t weights1 = vmovl_high_s8(w0);
            int16x8_t weights2 = vmovl_s8(vget_low_s8(w1));
            int16x8_t weights3 = vmovl_high_s8(w1);

            int16x8_t prod0 = vmulq_s16(weights0, interleaved);
            int16x8_t prod1 = vmulq_s16(weights1, interleaved);
            int16x8_t prod2 = vmulq_s16(weights2, interleaved);
            int16x8_t prod3 = vmulq_s16(weights3, interleaved);
            
            l2_acc[0] = vpadalq_s16(l2_acc[0], prod0);
            l2_acc[1] = vpadalq_s16(l2_acc[1], prod1);
            l2_acc[2] = vpadalq_s16(l2_acc[2], prod2);
            l2_acc[3] = vpadalq_s16(l2_acc[3], prod3);
        }

    #elif USE_AVX2 || USE_AVX512

        for (int i = 0; i < num_active; i++) {
            int idx = active_indices[i];
            uint8_t a_stm = acc_stm[idx] < 0 ? 0 : (acc_stm[idx] > QA ? QA : acc_stm[idx]);
            uint8_t a_opp = acc_opp[idx] < 0 ? 0 : (acc_opp[idx] > QA ? QA : acc_opp[idx]);

            uint16_t concat = (uint16_t) a_stm | ((uint16_t) a_opp << 8);
            __m256i interleaved = _mm256_set1_epi16(concat);

            __m256i weights = _mm256_load_si256(reinterpret_cast<const __m256i *>(l2_weights[idx]));
            __m256i acc = _mm256_maddubs_epi16(interleaved, weights);

            #if USE_AVX512
            l2_acc[0] = _mm512_add_epi32(l2_acc[0], _mm512_cvtepi16_epi32(acc));
            #else
            __m256i low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc));
            __m256i high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc, 1));

            l2_acc[0] = _mm256_add_epi32(l2_acc[0], low);
            l2_acc[1] = _mm256_add_epi32(l2_acc[1], high);
            #endif
        }

    #endif

        const vec_f32 v_qaqb_f32 = vec_dup_f32(QA * QB);

        for (int i = 0; i < l2_chunks; i++) {
            vec_f32 v = vec_add_f32(vec_i32_to_f32(l2_acc[i]), vec_load_f32(l2_biases + jump32 * i));
            v = vec_max_f32(v, v_zero_f32);
            v = vec_min_f32(v, v_qaqb_f32);
            vec_store_f32(l2_buff + jump32 * i, v);
        }

        for (int i = 0; i < l3_chunks; i++) {
            l3_acc[i] = vec_load_f32(l3_biases + jump32 * i);
        }

        for (int i = 0; i < L2_SIZE; i++) {
            vec_f32 l2 = vec_dup_f32(l2_buff[i]);
            for (int j = 0; j < l3_chunks; j++) {
                l3_acc[j] = vec_mla_f32(l3_acc[j], l2, vec_load_f32(l3_weights[i] + jump32 * j));
            }
        }

        vec_f32 acc = v_zero_f32;
        for (int i = 0; i < l3_chunks; i++) {
            vec_f32 l3 = vec_max_f32(l3_acc[i], v_zero_f32);
            l3 = vec_min_f32(l3, v_qaqb_f32);
            vec_f32 w = vec_load_f32(output_weights + jump32 * i);
            acc = vec_mla_f32(acc, l3, w);
        }

        int output = output_bias + vec_sum_f32(acc);
        return output * (float) SCALE / (QA * QB);
    }
}