#ifndef VECMA_H
#define VECMA_H 1

#include <stddef.h>

#include "vecma_types.h"
#include "vecma_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline void vecma_init(struct s_vecma* ekr);

static inline int vecma_set_input_vector(struct s_vecma* ekr, int i, s_vecma_vector arg);
static inline int vecma_set_output_vector(struct s_vecma* ekr, int i, s_vecma_vector arg);
static inline int vecma_set_constant(struct s_vecma* ekr, int i, s_vecma_scalar arg);
static inline int vecma_set_status(struct s_vecma* ekr, int i, s_vecma_vector arg);

static inline s_vecma_vector vecma_make_input_vector_linear(e_vecma_precision prec, size_t n, size_t s, const void* v);
static inline s_vecma_vector vecma_make_output_vector_linear(e_vecma_precision prec, size_t n, size_t s, void* v);

static inline s_vecma_vector vecma_make_input_vector(e_vecma_precision prec, s_vecma_slice sl, size_t s, void* v);
static inline s_vecma_vector vecma_make_output_vector(e_vecma_precision prec, s_vecma_slice sl, size_t s, void* v);

e_vecma_status
vecma_evaluate(e_vecma_function func, struct s_vecma* ekr);


#ifdef __cplusplus
}
#endif

#endif // #ifndef VECMA_H

