// Capture the final head input and enough logits to recompute stop probabilities
// under the actual penalties -> top-k -> top-p -> temperature sampler.
// Arguments: model chunk close_id input.i32 output.f16 stats.f32 ...
// stats rows: penalized close logit, top 20 penalized non-close logits (descending).
#include "llama.h"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 7 || (argc - 4) % 3) return 2;
    const int chunk = std::atoi(argv[2]), close = std::atoi(argv[3]);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 40960; cp.n_batch = chunk; cp.n_ubatch = chunk; cp.n_seq_max = 1;
    cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    const int dim = llama_model_n_embd(model);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto b = llama_batch_init(chunk, 0, 1);
    for (int a = 4; a < argc; a += 3) {
        FILE *fi = std::fopen(argv[a], "rb");
        if (!fi) return 5;
        std::vector<llama_token> ids; llama_token token;
        while (std::fread(&token, sizeof token, 1, fi)) ids.push_back(token);
        std::fclose(fi);
        if (ids.empty() || ids.size() > cp.n_ctx) return 6;
        // Each fit request used an output-budget+1024 penalty window, covering its entire prefix.
        std::vector<int> first(vocab, std::numeric_limits<int>::max());
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] < 0 || ids[i] >= vocab) return 7;
            first[ids[i]] = std::min(first[ids[i]], (int)i);
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        FILE *fo = std::fopen(argv[a+1], "wb"), *fs = std::fopen(argv[a+2], "wb");
        if (!fo || !fs) return 8;
        for (size_t s = 0; s < ids.size(); s += chunk) {
            b.n_tokens = std::min((size_t)chunk, ids.size()-s);
            for (int k = 0; k < b.n_tokens; ++k) {
                b.token[k] = ids[s+k]; b.pos[k] = s+k;
                b.n_seq_id[k] = 1; b.seq_id[k][0] = 0; b.logits[k] = 1;
            }
            if (llama_decode(ctx, b)) return 9;
            std::vector<ggml_fp16_t> emb((size_t)b.n_tokens*dim);
            std::vector<std::array<float,21>> stats(b.n_tokens);
            // Resolve API pointers before entering the parallel region (output reorder mutates state).
            std::vector<const float *> ee(b.n_tokens), ll(b.n_tokens);
            for (int k=0; k<b.n_tokens; ++k) {
                ee[k] = llama_get_embeddings_ith(ctx,k); ll[k] = llama_get_logits_ith(ctx,k);
                if (!ee[k] || !ll[k]) return 10;
            }
            #pragma omp parallel for num_threads(4)
            for (int k = 0; k < b.n_tokens; ++k) {
                ggml_fp32_to_fp16_row(ee[k], emb.data()+(size_t)k*dim,dim);
                auto &out = stats[k];
                out[0] = ll[k][close] - (first[close] <= (int)(s+k) ? 1.5f : 0.f);
                std::array<float,20> top; top.fill(-std::numeric_limits<float>::infinity());
                // A min heap retains exactly the twenty strongest alternatives.
                for (int v=0; v<vocab; ++v) {
                    if (v == close) continue;
                    const float x = ll[k][v] - (first[v] <= (int)(s+k) ? 1.5f : 0.f);
                    if (x <= top[0]) continue;
                    std::pop_heap(top.begin(),top.end(),std::greater<float>());
                    top.back()=x;
                    std::push_heap(top.begin(),top.end(),std::greater<float>());
                }
                std::sort(top.begin(),top.end(),std::greater<float>());
                std::copy(top.begin(),top.end(),out.begin()+1);
            }
            if (std::fwrite(emb.data(),sizeof(ggml_fp16_t),emb.size(),fo)!=emb.size()) return 11;
            if (std::fwrite(stats.data(),sizeof(stats[0]),stats.size(),fs)!=stats.size()) return 12;
        }
        std::fclose(fo); std::fclose(fs);
        std::string done = std::string(argv[a+1]) + ".done";
        FILE *fd = std::fopen(done.c_str(), "w");
        if (!fd) return 13;
        std::fprintf(fd,"%zu\n",ids.size()); std::fclose(fd);
        std::fprintf(stderr,"%s: %zu rows captured\n",argv[a+1],ids.size());
    }
    llama_batch_free(b); llama_free(ctx); llama_model_free(model);
}
