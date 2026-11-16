/* Mock media library (abi/gcm.h 1.0) for CPU tests: text ids are the prompt bytes, the marker "<m>" is one audio chunk, a clip of n
 * samples encodes to ceil(n / 4) rows [sum of its 4 samples, row index, n, 1], decode emits per chunk one id (text: sum of ids mod 97;
 * audio: rows * 10 + int(first row sum)) then counts up to max_new. Provenance cuBLAS version from GCM_MOCK_CUBLAS (default 13.4.1). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../abi/gcm.h"

struct gcm_handle { int32_t ids[4096]; char err[256]; };
static char open_err[256];

uint32_t gcm_abi_version(void) { return (GCM_ABI_MAJOR << 16) | GCM_ABI_MINOR; }

int gcm_open(const char *mmproj, const char *text_model, int device, gcm_handle **out) {
  if (!mmproj || !text_model || !out) { snprintf(open_err, sizeof open_err, "null argument"); return GCM_E_ARG; }
  (void) device;
  *out = calloc(1, sizeof(gcm_handle));
  return GCM_OK;
}

int gcm_n_embd(gcm_handle *h) { (void) h; return 4; }
const char *gcm_marker(gcm_handle *h) { (void) h; return "<m>"; }

int gcm_tokenize(gcm_handle *h, const char *prompt, gcm_chunk *chunks, uint32_t cap, uint32_t *n_chunks) {
  uint32_t n = 0, k = 0;
  const char *p = prompt;
  while (*p) {
    const char *m = strstr(p, "<m>");
    size_t len = m ? (size_t) (m - p) : strlen(p);
    if (len) {
      if (n < cap) { chunks[n].kind = GCM_CHUNK_TEXT; chunks[n].n = (uint32_t) len; chunks[n].ids = h->ids + k; chunks[n].rows = 0; }
      for (size_t i = 0; i < len; i++) h->ids[k++] = (unsigned char) p[i];
      n++;
    }
    if (!m) break;
    if (n < cap) { chunks[n].kind = GCM_CHUNK_AUDIO; chunks[n].n = 0; chunks[n].ids = 0; chunks[n].rows = 0; }
    n++;
    p = m + 3;
  }
  *n_chunks = n;
  return n > cap ? GCM_E_CAPACITY : GCM_OK;
}

int gcm_encode_audio(gcm_handle *h, const float *pcm, uint32_t n, float *rows_out, uint32_t cap_rows, uint32_t *n_rows) {
  (void) h;
  uint32_t r = (n + 3) / 4;
  *n_rows = r;
  if (r > cap_rows) return GCM_E_CAPACITY;
  for (uint32_t i = 0; i < r; i++) {
    float s = 0;
    for (uint32_t j = 4 * i; j < 4 * i + 4 && j < n; j++) s += pcm[j];
    rows_out[4 * i] = s; rows_out[4 * i + 1] = (float) i; rows_out[4 * i + 2] = (float) n; rows_out[4 * i + 3] = 1;
  }
  return GCM_OK;
}

int gcm_decode(gcm_handle *h, const gcm_chunk *chunks, uint32_t n_chunks, int greedy, uint32_t max_new, int32_t *out_ids,
               uint32_t *n_out) {
  uint32_t k = 0;
  if (!greedy) { snprintf(h->err, sizeof h->err, "mock: greedy only"); return GCM_E_UNSUPPORTED; }
  for (uint32_t c = 0; c < n_chunks && k < max_new; c++) {
    int32_t v = 0;
    if (chunks[c].kind == GCM_CHUNK_TEXT) { for (uint32_t i = 0; i < chunks[c].n; i++) v += chunks[c].ids[i]; v %= 97; }
    else { if (!chunks[c].rows) { snprintf(h->err, sizeof h->err, "audio chunk without rows"); return GCM_E_ARG; }
           v = (int32_t) chunks[c].n * 10 + (int32_t) chunks[c].rows[0]; }
    out_ids[k++] = v;
  }
  for (int32_t i = 0; k < max_new && i < 3; i++) out_ids[k++] = 1000 + i;
  *n_out = k;
  return GCM_OK;
}

int gcm_provenance(gcm_handle *h, char *json, uint32_t cap) {
  (void) h;
  const char *v = getenv("GCM_MOCK_CUBLAS");
  int n = snprintf(json, cap, "{\"cublas\": \"%s\", \"cublasLt\": \"13.4.1\", \"pdl\": \"off\", \"lib\": \"mock\"}", v ? v : "13.4.1");
  return n < 0 || (uint32_t) n >= cap ? GCM_E_CAPACITY : GCM_OK;
}

const char *gcm_last_error(gcm_handle *h) { return h ? h->err : open_err; }
void gcm_close(gcm_handle *h) { free(h); }
