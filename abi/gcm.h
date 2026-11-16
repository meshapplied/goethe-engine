/* gcm.h: the C ABI of a media encoder library (bucket kind "encode" with io.media; body "ggml-pinned-lib").
 *
 * The bucket names the library (bucket "lib"), the media kind (io.media: "audio") and its input/output (io.in, io.out); the
 * engine knows only these fields, never a model. A library turns a prompt with media markers into chunks, encodes each media
 * chunk into embedding rows, and (optional, gcm_decode) runs the whole backbone itself; a pack that serves the backbone with
 * compiled decoder buckets feeds the rows to their supplied-rows prefill instead.
 *
 * Conventions as gc.h: plain C99, ints return GCM_OK (0) or a negative code, gcm_last_error(h) the text (gcm_last_error(NULL)
 * the last gcm_open error of the calling thread), one handle is not thread safe. Version MAJOR << 16 | MINOR; a minor only
 * appends. Pointers a library returns inside chunks stay valid until the next gcm_tokenize on that handle. */
#ifndef GCM_H
#define GCM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GCM_ABI_MAJOR 1
#define GCM_ABI_MINOR 0

typedef struct gcm_handle gcm_handle;

enum { GCM_OK = 0, GCM_E_ARG = -1, GCM_E_WEIGHTS = -3, GCM_E_DEVICE = -4, GCM_E_UNSUPPORTED = -5, GCM_E_CAPACITY = -7 };
enum { GCM_CHUNK_TEXT = 0, GCM_CHUNK_AUDIO = 1 };

typedef struct {
  uint32_t kind;        /* GCM_CHUNK_TEXT | GCM_CHUNK_AUDIO */
  uint32_t n;           /* text: token ids; audio: embedding rows (0 from gcm_tokenize: known after gcm_encode_audio) */
  const int32_t *ids;   /* text: n ids */
  const float *rows;    /* audio: n x n_embd float32 rows (set by the caller from gcm_encode_audio before gcm_decode) */
} gcm_chunk;

uint32_t gcm_abi_version(void);
/* weights: the media projector file and the text model file (by path; the pack holds no weights); device: CUDA ordinal */
int gcm_open(const char *mmproj, const char *text_model, int device, gcm_handle **out);
int gcm_n_embd(gcm_handle *h);
/* the media marker string gcm_tokenize splits on (static storage) */
const char *gcm_marker(gcm_handle *h);
/* prompt: UTF-8 text, already chat-templated, one marker per media item; chunks in order; GCM_E_CAPACITY sets *n_chunks */
int gcm_tokenize(gcm_handle *h, const char *prompt, gcm_chunk *chunks, uint32_t cap, uint32_t *n_chunks);
/* pcm: float32 mono at io.in.rate; rows_out: cap_rows x n_embd; GCM_E_CAPACITY sets *n_rows to the rows needed */
int gcm_encode_audio(gcm_handle *h, const float *pcm, uint32_t n, float *rows_out, uint32_t cap_rows, uint32_t *n_rows);
/* optional (bucket io.decode true): prefill every chunk, then up to max_new tokens (greedy != 0: argmax), stop at an end-of-
 * generation token of the text model's vocabulary (not written); out_ids holds max_new ids, *n_out the count */
int gcm_decode(gcm_handle *h, const gcm_chunk *chunks, uint32_t n_chunks, int greedy, uint32_t max_new, int32_t *out_ids,
               uint32_t *n_out);
/* the versions this process bound, as JSON {"cublas": "M.m.p", "cublasLt": "M.m.p", "pdl": "off"|"on", "lib": "<pin>"}, for the
 * engine's check against the bucket's provenance; GCM_E_CAPACITY when cap is too small */
int gcm_provenance(gcm_handle *h, char *json, uint32_t cap);
const char *gcm_last_error(gcm_handle *h);
void gcm_close(gcm_handle *h);

#ifdef __cplusplus
}
#endif
#endif
