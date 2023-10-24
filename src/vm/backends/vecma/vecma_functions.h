#ifndef VECMA_FUNCTIONS_H
#define VECMA_FUNCTIONS_H 1

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "vecma_types.h"

#ifdef __cplusplus
extern "C" {
#endif


static inline s_vecma_vector vecma_make_input_vector_linear(e_vecma_precision prec, size_t n, size_t s, const void* v) {
    s_vecma_vector ret = {
        .cptr = { .v = v },
        .ptr = { .v = NULL },
        .prec = kVECMA_PREC_NOT,
        .index = { .start = 0, .size = n, .stride = +1 },
        .size = s
    };

    switch (prec) {
        case kVECMA_PREC_H: ret.prec = kVECMA_PREC_H; ret.cptr.h = (const vecma_half*)v; break;
        case kVECMA_PREC_S: ret.prec = kVECMA_PREC_S; ret.cptr.s = (const float*)v; break;
        case kVECMA_PREC_D: ret.prec = kVECMA_PREC_D; ret.cptr.d = (const double*)v; break;
        case kVECMA_PREC_C: ret.prec = kVECMA_PREC_C; ret.cptr.c = (const vecma_complex_float*)v; break;
        case kVECMA_PREC_Z: ret.prec = kVECMA_PREC_Z; ret.cptr.z = (const vecma_complex_double*)v; break;

        case kVECMA_PREC_INT32: ret.prec = kVECMA_PREC_INT32; ret.cptr.op32 = (const int32_t*)v; break; 
        case kVECMA_PREC_INT64: ret.prec = kVECMA_PREC_INT64; ret.cptr.op64 = (const int64_t*)v; break; 
        case kVECMA_PREC_STATUS: ret.prec = kVECMA_PREC_STATUS; ret.cptr.status = (const vecma_status*)v; break;

        case kVECMA_PREC_NOT: break;
    }

    return ret;
}

static inline s_vecma_vector vecma_make_input_vector(e_vecma_precision prec, s_vecma_slice sl, size_t s, void* v) {
    s_vecma_vector ret = {
        .cptr = { .v = v },
        .ptr = { .v = NULL },
        .prec = kVECMA_PREC_NOT,
        .index = sl,
        .size = s
    };

    switch (prec) {
        case kVECMA_PREC_H: ret.prec = kVECMA_PREC_H; ret.ptr.h = (vecma_half*)v; break;
        case kVECMA_PREC_S: ret.prec = kVECMA_PREC_S; ret.ptr.s = (float*)v; break;
        case kVECMA_PREC_D: ret.prec = kVECMA_PREC_D; ret.ptr.d = (double*)v; break;
        case kVECMA_PREC_C: ret.prec = kVECMA_PREC_C; ret.ptr.c = (vecma_complex_float*)v; break;
        case kVECMA_PREC_Z: ret.prec = kVECMA_PREC_Z; ret.ptr.z = (vecma_complex_double*)v; break;

        case kVECMA_PREC_INT32: ret.prec = kVECMA_PREC_INT32; ret.ptr.op32 = (int32_t*)v; break; 
        case kVECMA_PREC_INT64: ret.prec = kVECMA_PREC_INT64; ret.ptr.op64 = (int64_t*)v; break; 
        case kVECMA_PREC_STATUS: ret.prec = kVECMA_PREC_STATUS; ret.ptr.status = (e_vecma_status*)v; break;

        case kVECMA_PREC_NOT: break;
    }
    return ret;
}


static inline s_vecma_vector vecma_make_output_vector_linear(e_vecma_precision prec, size_t n, size_t s, void* v) {
    s_vecma_vector ret = {
        .cptr = { .v = NULL },
        .ptr = { .v = NULL },
        .prec = kVECMA_PREC_NOT,
        .index = { .start = 0, .size = n, .stride = +1 },
        .size = s
    };

    switch (prec) {
        case kVECMA_PREC_H: ret.prec = kVECMA_PREC_H; ret.ptr.h = (vecma_half*)v; break;
        case kVECMA_PREC_S: ret.prec = kVECMA_PREC_S; ret.ptr.s = (float*)v; break;
        case kVECMA_PREC_D: ret.prec = kVECMA_PREC_D; ret.ptr.d = (double*)v; break;
        case kVECMA_PREC_C: ret.prec = kVECMA_PREC_C; ret.ptr.c = (vecma_complex_float*)v; break;
        case kVECMA_PREC_Z: ret.prec = kVECMA_PREC_Z; ret.ptr.z = (vecma_complex_double*)v; break;

        case kVECMA_PREC_INT32: ret.prec = kVECMA_PREC_INT32; ret.ptr.op32 = (int32_t*)v; break; 
        case kVECMA_PREC_INT64: ret.prec = kVECMA_PREC_INT64; ret.ptr.op64 = (int64_t*)v; break; 
        case kVECMA_PREC_STATUS: ret.prec = kVECMA_PREC_STATUS; ret.ptr.status = (vecma_status*)v; break;

        case kVECMA_PREC_NOT: break;
    }

    return ret;
}

static inline s_vecma_vector vecma_make_output_vector(e_vecma_precision prec, s_vecma_slice sl, size_t s, void* v) {
    s_vecma_vector ret = {
        .cptr = { .v = NULL },
        .ptr = { .v = NULL },
        .prec = kVECMA_PREC_NOT,
        .index = sl,
        .size = s
    };

    switch (prec) {
        case kVECMA_PREC_H: ret.prec = kVECMA_PREC_H; ret.ptr.h = (vecma_half*)v; break;
        case kVECMA_PREC_S: ret.prec = kVECMA_PREC_S; ret.ptr.s = (float*)v; break;
        case kVECMA_PREC_D: ret.prec = kVECMA_PREC_D; ret.ptr.d = (double*)v; break;
        case kVECMA_PREC_C: ret.prec = kVECMA_PREC_C; ret.ptr.c = (vecma_complex_float*)v; break;
        case kVECMA_PREC_Z: ret.prec = kVECMA_PREC_Z; ret.ptr.z = (vecma_complex_double*)v; break;

        case kVECMA_PREC_INT32: ret.prec = kVECMA_PREC_INT32; ret.ptr.op32 = (int32_t*)v; break; 
        case kVECMA_PREC_INT64: ret.prec = kVECMA_PREC_INT64; ret.ptr.op64 = (int64_t*)v; break; 
        case kVECMA_PREC_STATUS: ret.prec = kVECMA_PREC_STATUS; ret.ptr.status = (e_vecma_status*)v; break;

        case kVECMA_PREC_NOT: break;
    }
    return ret;
}

static inline void vecma_init(struct s_vecma* ekr) { memset(ekr, 0, sizeof(s_vecma)); }

static inline int vecma_set_input_vector(struct s_vecma* ekr, int i, s_vecma_vector arg) {
    if (kVECMA_PREC_NOT == arg.prec) { return -1; }
    if (i < 0 || i >= kVECMA_MAX_I) { return -2; }
    ekr->arg_i[i] = arg;
    return 0;
}

static inline int vecma_set_output_vector(struct s_vecma* ekr, int i, s_vecma_vector arg) {
    if (kVECMA_PREC_NOT == arg.prec) { return -1; }
    if (i < 0 || i >= kVECMA_MAX_O) { return -2; }
    ekr->arg_o[i] = arg;
    return 0;
}

static inline int vecma_set_constant(struct s_vecma* ekr, int i, s_vecma_scalar arg) {
    if (kVECMA_PREC_NOT == arg.prec) { return -1; }
    if (i < 0 || i >= kVECMA_MAX_C) { return -2; }
    ekr->arg_c[i] = arg;
    return 0;
}

static inline int vecma_set_status(struct s_vecma* ekr, int i, s_vecma_vector arg) {
    ekr->arg_status = arg;
    return 0;
}

e_vecma_status
vecma_evaluate(e_vecma_function func, struct s_vecma* ekr);


#ifdef __cplusplus
}
#endif



#endif // #ifndef VECMA_FUNCTIONS_H

