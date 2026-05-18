#include "nnue.hh"

namespace NNUE {
    alignas(64) int16_t accumulators[1024][2][L1_SIZE];
    alignas(64) AccInfo acc_info[1024];
    alignas(64) uint8_t activated_accumulators[L1_SIZE * 2];

    alignas(64) int16_t l1_weights[INPUT_SIZE][L1_SIZE];
    alignas(64) int16_t l1_biases[L1_SIZE];
    alignas(64) int8_t l2_weights[L1_SIZE / 2][L2_SIZE * 4];
    alignas(64) float l2_biases[L2_SIZE];
    alignas(64) float l3_weights[L2_SIZE][L3_SIZE];
    alignas(64) float l3_biases[L3_SIZE];
    alignas(64) float output_weights[L3_SIZE];
                float output_bias;

    static constexpr int jump32 = REGISTER_WIDTH / 32;
    static constexpr int jump16 = REGISTER_WIDTH / 16;
    static constexpr int jump8 = REGISTER_WIDTH / 8;
    static constexpr int l2_chunks = L2_SIZE / jump32;
    static constexpr int l3_chunks = L3_SIZE / jump32;

    alignas(64) vec_i32 l2_acc[l2_chunks];
    alignas(64) float l2_buff[L2_SIZE];
    alignas(64) vec_f32 l3_acc[l3_chunks];

    alignas(64) uint16_t active_indices[L1_SIZE];
    int num_active = 0;

    #if USE_NEON || USE_AVX2
        // indexed by every possible uint8 mask of active indices
        // holds corresponding contiguous array of indices
        alignas(64) uint16_t active_table[1 << 8][8];
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
                l1_weights[i][j] = static_cast<int16_t>(std::round(fl1_weights[i * L1_SIZE + j] * QA));
            }
        }
        for (int i = 0; i < L1_SIZE; i++) {
            l1_biases[i] = static_cast<int16_t>(std::round(fl1_biases[i] * QA));
        }

        // l2_weights are grouped by four and side-to-move and opponent weights are concatenated
        for (int i = 0; i < L1_SIZE / 4; i++) {
            for (int j = 0; j < L2_SIZE; j++) {
                for (int k = 0; k < 4; k++) {
                    l2_weights[i][j * 4 + k] =
                      static_cast<int8_t>(std::round(fl2_weights_stm[(i * 4 + k) * L2_SIZE + j] * QB));
                    l2_weights[i + L1_SIZE / 4][j * 4 + k] =
                      static_cast<int8_t>(std::round(fl2_weights_opp[(i * 4 + k) * L2_SIZE + j] * QB));
                }
            }
        }

        free(fl1_weights);
        free(fl2_weights_stm);
        free(fl2_weights_opp);


        #if USE_NEON || USE_AVX2
            std::memset(active_table, 0, sizeof(active_table));
            for (int i = 0; i < (1 << 8); i++) {
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

        uint64_t pieces = position.pieces();
        while (pieces) {
            int square = pop_lsb(pieces);
            int piece = position.board[square];

            int white_idx = make_index<WHITE>(square, piece, white_mirror);
            int black_idx = make_index<BLACK>(square, piece, black_mirror);

            for (int i = 0; i < L1_SIZE; i++) {
                accumulators[ply][WHITE][i] += l1_weights[white_idx][i];
                accumulators[ply][BLACK][i] += l1_weights[black_idx][i];
            }
        }

        acc_info[ply].clean = true;
    }

    static inline float crelu(float x, float max) {
        return x < 0 ? 0 : (x > max ? max : x);
    }

    // not optimized - used for debugging
    int evaluate(Position& position) {
        reset_accumulators(position);

        int ply = position.half_moves;
        int turn = position.turn;

        int16_t* acc_stm = accumulators[ply][turn];
        int16_t* acc_opp = accumulators[ply][!turn];

        float l2_layer[L2_SIZE] = {};
        for (int i = 0; i < L1_SIZE / 4; i++) {
            for (int k = 0; k < 4; k++) {
                // screlu
                float a_stm = crelu(acc_stm[i * 4 + k], QA);
                float a_opp = crelu(acc_opp[i * 4 + k], QA);
                a_stm *= a_stm;
                a_opp *= a_opp;

                for (int j = 0; j < L2_SIZE; j++) {
                    l2_layer[j] += a_stm * l2_weights[i][j * 4 + k];
                    l2_layer[j] += a_opp * l2_weights[i + L1_SIZE / 4][j * 4 + k];
                }
            }
        }

        for (int i = 0; i < L2_SIZE; i++) {
            float f = l2_biases[i] + l2_layer[i] / float(QA * QA * QB);
            l2_layer[i] = crelu(f, 1.0);
        }

        float l3_layer[L3_SIZE];
        std::memcpy(l3_layer, l3_biases, L3_SIZE * sizeof(float));
        for (int i = 0; i < L2_SIZE; i++) {
            for (int j = 0; j < L3_SIZE; j++) {
                l3_layer[j] += l2_layer[i] * l3_weights[i][j];
            }
        }

        for (int i = 0; i < L3_SIZE; i++) {
            l3_layer[i] = crelu(l3_layer[i], 1.0);
        }

        float output = output_bias;
        for (int i = 0; i < L3_SIZE; i++) {
            output += l3_layer[i] * output_weights[i];
        }

        return output * float(SCALE);
    }


    void set_dirty(int ply, DirtyPieces& dps) {
        acc_info[ply].dps = dps;
        acc_info[ply].clean = false;
    }


    // efficiently update accumulator - only have to worry about the few indices that changed this move
    void update_accumulators(int ply) {
        acc_info[ply].clean = true;
        DirtyPieces &dps = acc_info[ply].dps;

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


    // apply screlu activation to the accumulator and populate active_indices
    // with the indices of nonzero groups of four adjacent values
    void activate_accumulators(int ply, int turn) {
        num_active = 0;

        #if USE_NEON
            uint16x8_t base = vdupq_n_u16(0);
            const uint16x8_t increment = vdupq_n_u16(8);
        #elif USE_AVX512_VBMI2
            __m512i base = _mm512_set_epi16(
                31, 30, 29, 28, 15, 14, 13, 12, 27, 26, 25, 24, 11, 10, 9, 8,
                23, 22, 21, 20, 7,  6,  5,  4,  19, 18, 17, 16, 3,  2,  1, 0
            );
            const __m512i increment = _mm512_set1_epi16(32);
        #elif USE_AVX512
            __m512i base = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0);
            const __m512i increment = _mm512_set1_epi32(16);
        #elif USE_AVX2
            __m128i base = _mm_setzero_si128();
            const __m128i increment = _mm_set1_epi16(8);
        #endif

        const vec_i16 v_qa_i16 = vec_dup_i16(QA);

        for (int i = 0; i < 2; i++) {
            // side to move first, then opponent
            int16_t* acc = accumulators[ply][turn ^ i];

            for (int j = 0; j < L1_SIZE; j += jump16 * 4) {
                vec_i16 v0 = vec_load_i16(acc + j);
                vec_i16 v1 = vec_load_i16(acc + j + jump16);
                vec_i16 v2 = vec_load_i16(acc + j + jump16 * 2);
                vec_i16 v3 = vec_load_i16(acc + j + jump16 * 3);

                v0 = vec_max_i16(v0, v_zero_i16);
                v0 = vec_min_i16(v0, v_qa_i16);
                v1 = vec_max_i16(v1, v_zero_i16);
                v1 = vec_min_i16(v1, v_qa_i16);
                v2 = vec_max_i16(v2, v_zero_i16);
                v2 = vec_min_i16(v2, v_qa_i16);
                v3 = vec_max_i16(v3, v_zero_i16);
                v3 = vec_min_i16(v3, v_qa_i16);

                // the neon mulhi instruction doubles the result
                #if USE_NEON
                    static constexpr int SHIFT = 7;
                #else
                    static constexpr int SHIFT = 8;
                #endif

                vec_i16 screlu0 = vec_mulhi_16(v0, vec_shl_i16(v0, SHIFT));
                vec_i16 screlu1 = vec_mulhi_16(v1, vec_shl_i16(v1, SHIFT));
                vec_i16 screlu2 = vec_mulhi_16(v2, vec_shl_i16(v2, SHIFT));
                vec_i16 screlu3 = vec_mulhi_16(v3, vec_shl_i16(v3, SHIFT));

                vec_u8 activated0 = vec_packus_ordered_i16(screlu0, screlu1);
                vec_u8 activated1 = vec_packus_ordered_i16(screlu2, screlu3);

                vec_store_u8(activated_accumulators + i * L1_SIZE + j, activated0);
                vec_store_u8(activated_accumulators + i * L1_SIZE + j + jump8, activated1);

                // set active indices

                vec_u32 grouped0 = vec_u32(activated0);
                vec_u32 grouped1 = vec_u32(activated1);

            #if USE_NEON
                static constexpr uint16_t nnz_mask[8] = {1, 2, 4, 8, 16, 32, 64, 128};

                uint16x8_t combined = vcombine_u16(
                    vqmovn_u32(vtstq_u32(grouped0, grouped0)),
                    vqmovn_u32(vtstq_u32(grouped1, grouped1))
                );
                
                uint8_t mask = vaddvq_u16(vandq_u16(combined, vld1q_u16(nnz_mask)));
                uint16x8_t indices = vld1q_u16(active_table[mask]);

                vst1q_u16(active_indices + num_active, vaddq_u16(indices, base));
                base = vaddq_u16(base, increment);
                num_active += __builtin_popcount(mask);

            #elif USE_AVX512_VBMI2

                __m512i combined = _mm512_packs_epi32(grouped0, grouped1);
                uint32_t mask = _mm512_test_epi16_mask(combined, combined);
                __m512i indices = _mm512_maskz_compress_epi16(mask, base);
                _mm512_storeu_si512(active_indices + num_active, indices);
                base = _mm512_add_epi16(base, increment);
                num_active += __builtin_popcount(mask);

            #elif USE_AVX512

                uint16_t mask = _mm512_test_epi32_mask(grouped0, grouped0);
                __m512i indices = _mm512_maskz_compress_epi32(mask, base);
                _mm512_mask_cvtepi32_storeu_epi16(active_indices + num_active, 0xFFFF, indices);
                base = _mm512_add_epi32(base, increment);
                num_active += __builtin_popcount(mask);

                mask = _mm512_test_epi32_mask(grouped1, grouped1);
                indices = _mm512_maskz_compress_epi32(mask, base);
                _mm512_mask_cvtepi32_storeu_epi16(active_indices + num_active, 0xFFFF, indices);
                base = _mm512_add_epi32(base, increment);
                num_active += __builtin_popcount(mask);

            #elif USE_AVX2

                __m256i nonzero = _mm256_xor_si256(_mm256_cmpeq_epi32(grouped0, v_zero_i32), _mm256_set1_epi32(-1));
                int mask = _mm256_movemask_ps(_mm256_castsi256_ps(nonzero));
                __m128i indices = _mm_load_si128(reinterpret_cast<const __m128i*>(active_table[mask]));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(active_indices + num_active), _mm_add_epi16(base, indices));
                num_active += __builtin_popcount(mask);
                base = _mm_add_epi16(base, increment);

                nonzero = _mm256_xor_si256(_mm256_cmpeq_epi32(grouped1, v_zero_i32), _mm256_set1_epi32(-1));
                mask = _mm256_movemask_ps(_mm256_castsi256_ps(nonzero));
                indices = _mm_load_si128(reinterpret_cast<const __m128i*>(active_table[mask]));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(active_indices + num_active), _mm_add_epi16(base, indices));
                num_active += __builtin_popcount(mask);
                base = _mm_add_epi16(base, increment);

            #endif
            }
        }
    }


    int evaluate_incremental(int ply, int turn) {
        clean_accumulators(ply);
        activate_accumulators(ply, turn);

        for (int i = 0; i < l2_chunks; i++) {
            l2_acc[i] = v_zero_i32;
        }

        // usually only ~10% of the accumulator is active after crelu
        // inference is sped up by looping through non-zero indices

        // to take advantage of vector operations that add adjacent elements,
        // four adjacent accumulator values are handled at the same time

        const uint32_t* grouped_activations = reinterpret_cast<const uint32_t*>(activated_accumulators);
        
        for (int i = 0; i < num_active; i++) {
            int idx = active_indices[i];
            vec_u8 vals = vec_dup_u32(grouped_activations[idx]);
            for (int j = 0; j < l2_chunks; j++) {
                vec_i8 weights = vec_load_i8(l2_weights[idx] + j * jump8);
                l2_acc[j] = vec_dpbusd_i32(l2_acc[j], vals, weights);
            }
        }

        const vec_f32 v_L2_norm = vec_dup_f32(float(1 << 8) / float(QA * QA * QB));

        for (int i = 0; i < l2_chunks; i++) {
            // convert to floats and normalize
            vec_f32 v = vec_i32_to_f32(l2_acc[i]);
            v = vec_mul_f32(v, v_L2_norm);
            v = vec_add_f32(v, vec_load_f32(l2_biases + jump32 * i));

            // crelu
            v = vec_max_f32(v, v_zero_f32);
            v = vec_min_f32(v, v_one_f32);

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
            l3 = vec_min_f32(l3, v_one_f32);
            vec_f32 w = vec_load_f32(output_weights + jump32 * i);
            acc = vec_mla_f32(acc, l3, w);
        }

        float output = output_bias + vec_sum_f32(acc);
        return output * float(SCALE);
    }
}