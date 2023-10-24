#ifndef VECMA_TYPES_H
#define VECMA_TYPES_H 1

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum e_vecma_max_index {
    kVECMA_MAX_I = 4,
    kVECMA_MAX_O = 2,
    kVECMA_MAX_C = 4,
};

typedef uint16_t vecma_half;

enum e_vecma_function {
    kVECMA_FUNC_NOP = 0,
    kVECMA_FUNC_POW = 1,
};

enum e_vecma_precision {
    kVECMA_PREC_NOT = 0,

    kVECMA_PREC_H = 1,
    kVECMA_PREC_S = 2,
    kVECMA_PREC_D = 3,
    kVECMA_PREC_C = 4,
    kVECMA_PREC_Z = 5,

    kVECMA_PREC_STATUS = 0xEE,

    kVECMA_PREC_INT32 = 0xC0,
    kVECMA_PREC_INT64 = 0xC1
};

enum e_vecma_status {
    kVECMA_SUCCESS = 0x0,

    kVECMA_DOMAIN_ERROR = 0x1,
    kVECMA_SINGULARITY = 0x2,
    kVECMA_UNDERFLOW = 0x4,
    kVECMA_OVERFLOW = 0x8,

    kVECMA_RUNTIME_ERROR = 0xE0000000
};

struct s_vecma_complex_float {
    float re;
    float im;
};

struct s_vecma_complex_double {
    double re;
    double im;
};

struct s_vecma_slice {
    size_t start;
    size_t size;
    int64_t stride;
};


static const size_t kVECMA_NO_SIZE = UINT64_C(0xEEEEEEEEEEEEEEEE);

typedef struct s_vecma_complex_float vecma_complex_float;
typedef struct s_vecma_complex_double vecma_complex_double;
typedef e_vecma_status vecma_status;


struct s_vecma_scalar {
    union {
        int32_t op32;
        int64_t op64;

        vecma_half h;
        float s;
        double d;

        vecma_complex_float c;
        vecma_complex_double z;
    } value;

    e_vecma_precision prec;
};

struct s_vecma_vector {
    union {
        const void* v;

        const vecma_half* h;
        const float* s;
        const double* d;

        const vecma_complex_float* c;
        const vecma_complex_double* z;

        const int32_t *op32;
        const int64_t *op64;

        const vecma_status* status;
    } cptr;

    union {
        void *v;

        vecma_half* h;
        float* s;
        double* d;

        vecma_complex_float* c;
        vecma_complex_double* z;

        int32_t *op32;
        int64_t *op64;

        vecma_status* status;
    } ptr;

    e_vecma_precision prec;

    struct s_vecma_slice index;
    size_t size;
};

struct s_vecma {
    struct s_vecma_vector arg_i[kVECMA_MAX_I + 1];
    struct s_vecma_vector arg_o[kVECMA_MAX_O + 1];
    struct s_vecma_scalar arg_c[kVECMA_MAX_C + 1];

    struct s_vecma_vector arg_status;
};

#ifdef __cplusplus
}
#endif


#endif // #ifndef VECMA_TYPES_H

