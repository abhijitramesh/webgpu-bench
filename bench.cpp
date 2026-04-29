// Minimal llama.cpp WASM benchmark wrapper.
// Exports C functions for JavaScript to call: init, load, run, exit.
// Based on llama.cpp/examples/simple/simple.cpp

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

// Global state
static llama_model   * g_model   = nullptr;
static llama_context * g_ctx     = nullptr;
static llama_sampler * g_sampler = nullptr;
static const llama_vocab * g_vocab = nullptr;
static int g_n_ctx = 2048;

// Static buffer for returning JSON strings to JS
// Sized for bench_run output: ~200 fixed fields + 4096 text + 900 token_ids
static char g_result_buf[16384];

// Parse a comma-separated list of integers into token IDs
static std::vector<llama_token> parse_token_ids(const char* csv) {
    std::vector<llama_token> ids;
    const char* p = csv;
    while (*p) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        ids.push_back((llama_token)atoi(p));
        while (*p && *p != ',') p++;
    }
    return ids;
}

extern "C" {

int bench_init() {
    ggml_backend_load_all();
    return 0;
}

int bench_load(const char * model_path, int n_ctx, int n_gpu_layers, int use_mmap) {
    // Clean up previous state if any
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_ctx)     { llama_free(g_ctx);             g_ctx = nullptr; }
    if (g_model)   { llama_model_free(g_model);     g_model = nullptr; }

    g_n_ctx = n_ctx;

    // Load model. use_mmap=0 forces fread-based loading via the C library —
    // required for OPFS-backed models, where the file is exposed via
    // patched MEMFS stream_ops that route reads to a FileSystemSyncAccessHandle.
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.use_mmap = use_mmap != 0;

    g_model = llama_model_load_from_file(model_path, model_params);
    if (!g_model) {
        fprintf(stderr, "bench_load: failed to load model from %s\n", model_path);
        return -1;
    }

    g_vocab = llama_model_get_vocab(g_model);

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = n_ctx;
    ctx_params.n_batch  = n_ctx;
    ctx_params.no_perf  = false; // enable performance counters

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        fprintf(stderr, "bench_load: failed to create context\n");
        llama_model_free(g_model);
        g_model = nullptr;
        return -2;
    }

    // Create greedy sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_greedy());

    fprintf(stderr, "bench_load: model loaded, ctx=%d, gpu_layers=%d\n", n_ctx, n_gpu_layers);
    return 0;
}

const char * bench_run(const char * prompt, int n_predict) {
    if (!g_model || !g_ctx || !g_sampler) {
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"error\":\"model not loaded\"}");
        return g_result_buf;
    }

    // Tokenize
    const int n_prompt_max = g_n_ctx;
    const int n_prompt = -llama_tokenize(g_vocab, prompt, strlen(prompt), NULL, 0, true, true);
    if (n_prompt <= 0 || n_prompt > n_prompt_max) {
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"error\":\"tokenization failed, n_prompt=%d\"}", n_prompt);
        return g_result_buf;
    }

    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(g_vocab, prompt, strlen(prompt),
                       prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"error\":\"tokenization failed\"}");
        return g_result_buf;
    }

    // Reset perf counters
    llama_perf_context_reset(g_ctx);

    // Prepare batch for prompt (prefill)
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // Handle encoder-decoder models
    if (llama_model_has_encoder(g_model)) {
        if (llama_encode(g_ctx, batch)) {
            snprintf(g_result_buf, sizeof(g_result_buf),
                "{\"error\":\"encode failed\"}");
            return g_result_buf;
        }
        llama_token decoder_start = llama_model_decoder_start_token(g_model);
        if (decoder_start == LLAMA_TOKEN_NULL) {
            decoder_start = llama_vocab_bos(g_vocab);
        }
        batch = llama_batch_get_one(&decoder_start, 1);
    }

    // Decode prompt (prefill)
    if (llama_decode(g_ctx, batch)) {
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"error\":\"prefill decode failed\"}");
        return g_result_buf;
    }

    // Generate tokens
    std::string output_text;
    std::vector<llama_token> generated_ids;
    int n_decoded = 0;
    int n_pos = batch.n_tokens;

    for (int i = 0; i < n_predict; i++) {
        llama_token new_token = llama_sampler_sample(g_sampler, g_ctx, -1);

        if (llama_vocab_is_eog(g_vocab, new_token)) {
            break;
        }

        generated_ids.push_back(new_token);

        // Convert token to text
        char piece_buf[256];
        int piece_len = llama_token_to_piece(g_vocab, new_token, piece_buf, sizeof(piece_buf), 0, true);
        if (piece_len > 0) {
            output_text.append(piece_buf, piece_len);
        }

        // Prepare next batch
        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(g_ctx, batch)) {
            snprintf(g_result_buf, sizeof(g_result_buf),
                "{\"error\":\"decode failed at token %d\"}", i);
            return g_result_buf;
        }

        n_decoded++;
        n_pos++;
    }

    // Get performance metrics
    llama_perf_context_data perf = llama_perf_context(g_ctx);

    // Print perf to stderr for debug
    llama_perf_sampler_print(g_sampler);
    llama_perf_context_print(g_ctx);

    // Escape output text for JSON (basic: replace " and \ and newlines)
    std::string escaped_output;
    for (char c : output_text) {
        if (c == '"')       escaped_output += "\\\"";
        else if (c == '\\') escaped_output += "\\\\";
        else if (c == '\n') escaped_output += "\\n";
        else if (c == '\r') escaped_output += "\\r";
        else if (c == '\t') escaped_output += "\\t";
        else                escaped_output += c;
    }

    // Truncate output for JSON if too long
    if (escaped_output.size() > 4096) {
        escaped_output.resize(4096);
        escaped_output += "...(truncated)";
    }

    // Serialize generated token IDs as a compact JSON int array
    std::string token_ids_str;
    token_ids_str.reserve(generated_ids.size() * 6 + 2);
    token_ids_str = "[";
    for (size_t i = 0; i < generated_ids.size(); i++) {
        if (i > 0) token_ids_str += ",";
        token_ids_str += std::to_string(generated_ids[i]);
    }
    token_ids_str += "]";

    snprintf(g_result_buf, sizeof(g_result_buf),
        "{"
        "\"success\":true,"
        "\"n_prompt_tokens\":%d,"
        "\"n_generated\":%d,"
        "\"t_p_eval_ms\":%.2f,"
        "\"t_eval_ms\":%.2f,"
        "\"n_p_eval\":%d,"
        "\"n_eval\":%d,"
        "\"output\":\"%s\","
        "\"token_ids\":%s"
        "}",
        n_prompt,
        n_decoded,
        perf.t_p_eval_ms,
        perf.t_eval_ms,
        perf.n_p_eval,
        perf.n_eval,
        escaped_output.c_str(),
        token_ids_str.c_str()
    );

    return g_result_buf;
}

// Forced-decoding consistency check against a CPU reference token sequence.
// Feeds each reference token into the (already loaded) model one at a time and
// checks whether this backend independently agrees on the same top-1 token.
// This is the same "same top-1" metric used by llama.cpp's perplexity tool.
// ref_ids_csv: comma-separated token IDs from the CPU baseline run.
const char* bench_eval_tokens(const char* prompt, const char* ref_ids_csv) {
    if (!g_model || !g_ctx) {
        snprintf(g_result_buf, sizeof(g_result_buf), "{\"error\":\"model not loaded\"}");
        return g_result_buf;
    }

    std::vector<llama_token> ref_ids = parse_token_ids(ref_ids_csv);
    if (ref_ids.empty()) {
        snprintf(g_result_buf, sizeof(g_result_buf), "{\"error\":\"no reference token ids\"}");
        return g_result_buf;
    }

    // Start fresh — clear the memory (KV cache) from the bench_run that just completed
    llama_memory_clear(llama_get_memory(g_ctx), false);

    // Tokenize and prefill prompt (same as bench_run)
    const int n_prompt = -llama_tokenize(g_vocab, prompt, strlen(prompt), NULL, 0, true, true);
    if (n_prompt <= 0) {
        snprintf(g_result_buf, sizeof(g_result_buf), "{\"error\":\"tokenization failed\"}");
        return g_result_buf;
    }
    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(g_vocab, prompt, strlen(prompt), prompt_tokens.data(), n_prompt, true, true);

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(g_ctx, batch)) {
        snprintf(g_result_buf, sizeof(g_result_buf), "{\"error\":\"prefill failed\"}");
        return g_result_buf;
    }

    // Forced decoding: at each position, check what this backend would pick,
    // then advance context with the reference token (not our prediction).
    const int n_vocab = llama_vocab_n_tokens(g_vocab);
    const int n_tokens = (int)ref_ids.size();
    int n_agree = 0;
    int first_disagreement = -1;

    // matches: compact "0"/"1" array for per-token agreement
    std::string matches_str;
    matches_str.reserve(n_tokens * 2 + 2);
    matches_str = "[";

    for (int i = 0; i < n_tokens; i++) {
        float* logits = llama_get_logits(g_ctx);

        // Argmax over vocabulary
        llama_token top1 = 0;
        float max_logit = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                top1 = v;
            }
        }

        const bool match = (top1 == ref_ids[i]);
        if (match) n_agree++;
        if (!match && first_disagreement < 0) first_disagreement = i;

        if (i > 0) matches_str += ",";
        matches_str += match ? "1" : "0";

        // Feed reference token (forced) to build the correct context for next position
        batch = llama_batch_get_one(&ref_ids[i], 1);
        if (llama_decode(g_ctx, batch)) break;
    }

    matches_str += "]";

    const float agreement_rate = n_tokens > 0 ? (float)n_agree / n_tokens : 0.0f;

    snprintf(g_result_buf, sizeof(g_result_buf),
        "{"
        "\"agreement_rate\":%.4f,"
        "\"n_agree\":%d,"
        "\"n_tokens\":%d,"
        "\"first_disagreement\":%d,"
        "\"matches\":%s"
        "}",
        agreement_rate,
        n_agree,
        n_tokens,
        first_disagreement,
        matches_str.c_str()
    );

    return g_result_buf;
}

// llama-bench-style synthetic-token prefill test. Mirrors test_prompt() in
// tools/llama-bench/llama-bench.cpp: BOS as the first token (when the vocab
// expects one) and uniformly-random fillers thereafter, batched up to n_ctx.
// Caller times the call; we just clear the KV, run the work, synchronize, and
// return. Unlike bench_run there is no sampler involvement.
const char * bench_pp(int n_prompt) {
    if (!g_model || !g_ctx) {
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"error\":\"model not loaded\"}");
        return g_result_buf;
    }
    if (n_prompt <= 0) {
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"error\":\"n_prompt must be > 0\"}");
        return g_result_buf;
    }

    llama_memory_clear(llama_get_memory(g_ctx), false);

    const int32_t n_vocab = llama_vocab_n_tokens(g_vocab);
    const int n_batch = g_n_ctx;  // ctx_params.n_batch is set to n_ctx in bench_load
    std::vector<llama_token> tokens(n_batch);

    int n_processed = 0;
    while (n_processed < n_prompt) {
        const int n_tokens = std::min(n_prompt - n_processed, n_batch);
        tokens[0] = (n_processed == 0 && llama_vocab_get_add_bos(g_vocab))
            ? llama_vocab_bos(g_vocab)
            : std::rand() % n_vocab;
        for (int i = 1; i < n_tokens; i++) {
            tokens[i] = std::rand() % n_vocab;
        }
        if (llama_decode(g_ctx, llama_batch_get_one(tokens.data(), n_tokens))) {
            snprintf(g_result_buf, sizeof(g_result_buf),
                "{\"error\":\"prefill decode failed at processed=%d\"}", n_processed);
            return g_result_buf;
        }
        n_processed += n_tokens;
    }
    llama_synchronize(g_ctx);

    snprintf(g_result_buf, sizeof(g_result_buf),
        "{\"success\":true,\"n_prompt\":%d}", n_prompt);
    return g_result_buf;
}

// llama-bench-style decode test. Mirrors test_gen() in
// tools/llama-bench/llama-bench.cpp: prime with BOS (or a random token if the
// vocab doesn't expect BOS), then loop n_gen single-token decodes with a
// synchronize after each one — the synchronize is what makes wall time
// reflect per-token GPU latency rather than dispatch queue depth.
const char * bench_tg(int n_gen) {
    if (!g_model || !g_ctx) {
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"error\":\"model not loaded\"}");
        return g_result_buf;
    }
    if (n_gen <= 0) {
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"error\":\"n_gen must be > 0\"}");
        return g_result_buf;
    }

    llama_memory_clear(llama_get_memory(g_ctx), false);

    const int32_t n_vocab = llama_vocab_n_tokens(g_vocab);
    llama_token token = llama_vocab_get_add_bos(g_vocab)
        ? llama_vocab_bos(g_vocab)
        : std::rand() % n_vocab;

    for (int i = 0; i < n_gen; i++) {
        if (llama_decode(g_ctx, llama_batch_get_one(&token, 1))) {
            snprintf(g_result_buf, sizeof(g_result_buf),
                "{\"error\":\"decode failed at token %d\"}", i);
            return g_result_buf;
        }
        llama_synchronize(g_ctx);
        token = std::rand() % n_vocab;
    }

    snprintf(g_result_buf, sizeof(g_result_buf),
        "{\"success\":true,\"n_gen\":%d}", n_gen);
    return g_result_buf;
}

void bench_exit() {
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_ctx)     { llama_free(g_ctx);             g_ctx = nullptr; }
    if (g_model)   { llama_model_free(g_model);     g_model = nullptr; }
    g_vocab = nullptr;
    llama_backend_free();
}

} // extern "C"

// Dummy main (--no-entry is set, but some linkers want it)
int main() { return 0; }
