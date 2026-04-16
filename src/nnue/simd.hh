#ifndef simd_hh
#define simd_hh

#if defined(__ARM_NEON) && defined(__aarch64__)
    #define USE_NEON 1
    #include <arm_neon.h>
#elif defined(__AVX512VBMI2__)
    #define USE_AVX512_VBMI2 1
    #define USE_AVX512 1
    #include <immintrin.h>
#elif defined(__AVX512F__) && defined(__AVX512BW__)
    #define USE_AVX512 1
    #include <immintrin.h>
#elif defined(__AVX2__)
    #define USE_AVX2 1
    #include <immintrin.h>
#endif

#if USE_NEON

    #define REGISTER_WIDTH 128

    using vec_i8 = int8x16_t;
    using vec_u8 = uint8x16_t;
    using vec_i16 = int16x8_t;
    using vec_i32 = int32x4_t;
    using vec_u32 = uint32x4_t;
    using vec_f32 = float32x4_t;

    #define vec_load_i16(a) vld1q_s16(a)
    #define vec_store_i16(a, b) vst1q_s16(a, b)
    #define vec_load_i8(a) vld1q_s8(a)
    #define vec_store_u8(a, b) vst1q_u8(a, b)
    #define vec_dup_u32(a) vdupq_n_u32(a)
    #define vec_load_f32(a) vld1q_f32(a)
    #define vec_store_f32(a, b) vst1q_f32(a, b)
    #define vec_dup_f32(a) vdupq_n_f32(a)
    #define vec_add_f32(a, b) vaddq_f32(a, b)
    #define vec_mla_f32(a, b, c) vmlaq_f32(a, b, c) // multiply-accumulate: result[i] = a[i] + b[i] * c[i]
    #define vec_max_f32(a, b) vmaxq_f32(a, b)
    #define vec_min_f32(a, b) vminq_f32(a, b)
    #define vec_i32_to_f32(a) vcvtq_f32_s32(a)
    #define vec_sum_f32(a) vaddvq_f32(a) // horizontally add elements

    // combine two int16x8 into one uint8x16, clamping elements to [0, 255]
    #define vec_packus_i16(a, b) vcombine_u8(vqmovun_s16(a), vqmovun_s16(b))
    #define vec_packus_ordered_i16(a, b) vec_packus_i16(a, b)

    // for each group of 4 adjacent uint8s in b and int8s in c, accumulate their dot product in a
    #if defined(__ARM_FEATURE_MATMUL_INT8)
        #define vec_dpbusd_i32(a, b, c) vusdotq_s32(a, b, c)
    #else
        inline int32x4_t vec_dpbusd_i32(int32x4_t a, uint8x16_t b, int8x16_t c) {
            int16x8_t prod0 = vmulq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b))),
                vmovl_s8(vget_low_s8(c))
            );
            int16x8_t prod1 = vmulq_s16(
                vreinterpretq_s16_u16(vmovl_high_u8(b)),
                vmovl_high_s8(c)
            );
            return vpadalq_s16(a, vpaddq_s16(prod0, prod1));
        }
    #endif

    static const vec_i16 v_zero_i16 = vdupq_n_s16(0);
    static const vec_i32 v_zero_i32 = vdupq_n_s32(0);
    static const vec_f32 v_zero_f32 = vdupq_n_f32(0);

#elif USE_AVX512

    #define REGISTER_WIDTH 512

    using vec_i8 = __m512i;
    using vec_u8 = __m512i;
    using vec_i16 = __m512i;
    using vec_i32 = __m512i;
    using vec_u32 = __m512i;
    using vec_f32 = __m512;

    #define vec_load_i16(a) _mm512_load_si512(reinterpret_cast<const __m512i *>(a))
    #define vec_store_i16(a, b) _mm512_store_si512(reinterpret_cast<__m512i *>(a), b)
    #define vec_load_i8(a) _mm512_load_si512(reinterpret_cast<const __m512i *>(a))
    #define vec_store_u8(a, b) _mm512_store_si512(reinterpret_cast<__m512i *>(a), b)
    #define vec_dup_u32(a) _mm512_set1_epi32(a)
    #define vec_load_f32(a) _mm512_load_ps(a)
    #define vec_store_f32(a, b) _mm512_store_ps(a, b)
    #define vec_dup_f32(a) _mm512_set1_ps(a)
    #define vec_add_f32(a, b) _mm512_add_ps(a, b)
    #define vec_mla_f32(a, b, c) _mm512_fmadd_ps(b, c, a)
    #define vec_max_f32(a, b) _mm512_max_ps(a, b)
    #define vec_min_f32(a, b) _mm512_min_ps(a, b)
    #define vec_i32_to_f32(a) _mm512_cvtepi32_ps(a)
    #define vec_sum_f32(a) _mm512_reduce_add_ps(a)
    #define vec_packus_i16(a, b) _mm512_packus_epi16(a, b)
    
    // saturated vectors are concatenated instead of shuffled
    inline __m512i vec_packus_ordered_i16(__m512i a, __m512i b) {
        const __m512i order = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
        __m512i packed = _mm512_packus_epi16(a, b);
        return _mm512_permutexvar_epi64(order, packed);
    }

    inline __m512i vec_dpbusd_i32(__m512i a, __m512i b, __m512i c) {
        __m512i dot = _mm512_madd_epi16(_mm512_maddubs_epi16(b, c), _mm512_set1_epi16(1));
        return _mm512_add_epi32(a, dot);
    }

    static const vec_i16 v_zero_i16 = _mm512_setzero_si512();
    static const vec_i32 v_zero_i32 = _mm512_setzero_si512();
    static const vec_f32 v_zero_f32 = _mm512_setzero_ps();

    static const __m512i vec_squares = _mm512_set_epi8(
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
        47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0
    );

#elif USE_AVX2

    #define REGISTER_WIDTH 256

    using vec_i8 = __m256i;
    using vec_u8 = __m256i;
    using vec_i16 = __m256i;
    using vec_i32 = __m256i;
    using vec_u32 = __m256i;
    using vec_f32 = __m256;

    #define vec_load_i16(a) _mm256_load_si256(reinterpret_cast<const __m256i *>(a))
    #define vec_store_i16(a, b) _mm256_store_si256(reinterpret_cast<__m256i *>(a), b)
    #define vec_load_i8(a) _mm256_load_si256(reinterpret_cast<const __m256i *>(a))
    #define vec_store_u8(a, b) _mm256_store_si256(reinterpret_cast<__m256i *>(a), b)
    #define vec_dup_u32(a) _mm256_set1_epi32(a)
    #define vec_load_f32(a) _mm256_load_ps(a)
    #define vec_store_f32(a, b) _mm256_store_ps(a, b)
    #define vec_dup_f32(a) _mm256_set1_ps(a)
    #define vec_add_f32(a, b) _mm256_add_ps(a, b)
    #define vec_mla_f32(a, b, c) _mm256_fmadd_ps(b, c, a)
    #define vec_max_f32(a, b) _mm256_max_ps(a, b)
    #define vec_min_f32(a, b) _mm256_min_ps(a, b)
    #define vec_i32_to_f32(a) _mm256_cvtepi32_ps(a)
    #define vec_packus_i16(a, b) _mm256_packus_epi16(a, b)

    inline __m256i vec_packus_ordered_i16(__m256i a, __m256i b) {
        __m256i packed = _mm256_packus_epi16(a, b);
        return _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
    }

    inline __m256i vec_dpbusd_i32(__m256i a, __m256i b, __m256i c) {
        __m256i dot = _mm256_madd_epi16(_mm256_maddubs_epi16(b, c), _mm256_set1_epi16(1));
        return _mm256_add_epi32(a, dot);
    }

    inline float vec_sum_f32(__m256 vec) {
        __m256 v1 = _mm256_hadd_ps(vec, vec);
        __m256 v2 = _mm256_hadd_ps(v1, v1);

        __m128 low = _mm256_castps256_ps128(v2);
        __m128 high = _mm256_extractf128_ps(v2, 1);

        return _mm_cvtss_f32(_mm_add_ps(low, high));
    }

    static const vec_i16 v_zero_i16 = _mm256_setzero_si256();
    static const vec_i32 v_zero_i32 = _mm256_setzero_si256();
    static const vec_f32 v_zero_f32 = _mm256_setzero_ps();

#endif


#endif