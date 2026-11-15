// Head-input capture for the graft fit (Nov 15 2026): decode each token sequence causally in chunks (cache kept between
// chunks), embeddings on + pooling none = the final normed hidden state ("result_norm"), i.e. exactly the vector the
// output head multiplies.  usage: capt MODEL CHUNK in1.i32 out1.f16 [in2.i32 out2.f16 ...]   (raw int32 ids / raw fp16 rows)
#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: capt MODEL CHUNK in.i32 out.f16 ...\n"); return 1; }
    int chunk = atoi(argv[2]);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 40960; cp.n_batch = chunk; cp.n_ubatch = chunk; cp.n_seq_max = 1;
    cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_NONE; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    const int d = llama_model_n_embd(model);
    llama_batch b = llama_batch_init(chunk, 0, 1);
    std::vector<ggml_fp16_t> row(d);
    for (int a = 3; a + 1 < argc; a += 2) {
        FILE * fi = fopen(argv[a], "rb"); std::vector<llama_token> ids;
        llama_token t; while (fread(&t, sizeof t, 1, fi) == 1) ids.push_back(t); fclose(fi);
        if ((int) ids.size() > (int) cp.n_ctx) { fprintf(stderr, "%s: too long (%zu)\n", argv[a], ids.size()); return 1; }
        llama_memory_clear(llama_get_memory(ctx), true);
        FILE * fo = fopen(argv[a + 1], "wb");
        for (size_t s = 0; s < ids.size(); s += chunk) {
            b.n_tokens = 0;
            for (size_t i = s; i < ids.size() && i < s + chunk; i++) {
                int k = b.n_tokens++; b.token[k] = ids[i]; b.pos[k] = (llama_pos) i; b.n_seq_id[k] = 1; b.seq_id[k][0] = 0; b.logits[k] = 1;
            }
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at %zu\n", s); return 1; }
            for (int k = 0; k < b.n_tokens; k++) {
                const float * e = llama_get_embeddings_ith(ctx, k);
                ggml_fp32_to_fp16_row(e, row.data(), d); fwrite(row.data(), sizeof(ggml_fp16_t), d, fo);
            }
        }
        fclose(fo); fprintf(stderr, "%s: %zu rows\n", argv[a + 1], ids.size());
    }
    llama_batch_free(b); llama_free(ctx); llama_model_free(model); return 0;
}
