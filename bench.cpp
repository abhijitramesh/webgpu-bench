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

// Snapshot of the KV cache after a successful bench_set_depth call. Lets
// repeated reps at the same depth skip the prefill cost — same trick
// llama-bench's `cstate` plays in tools/llama-bench/llama-bench.cpp.
// Reset whenever the model/context is (re)created.
static int                  g_depth_cached = -1;
static std::vector<uint8_t> g_depth_state;

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
    fprintf(stderr, "bench_load: begin path=%s ctx=%d gpu_layers=%d mmap=%d\n",
        model_path ? model_path : "(null)", n_ctx, n_gpu_layers, use_mmap);
    fflush(stderr);

    // Clean up previous state if any
    fprintf(stderr, "bench_load: cleanup previous state\n");
    fflush(stderr);
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_ctx)     { llama_free(g_ctx);             g_ctx = nullptr; }
    if (g_model)   { llama_model_free(g_model);     g_model = nullptr; }

    // Old depth snapshot is bound to the freed context — invalidate it so
    // the next bench_set_depth doesn't try to restore stale bytes.
    g_depth_cached = -1;
    g_depth_state.clear();

    g_n_ctx = n_ctx;

    // Load model. use_mmap=0 forces fread-based loading via the C library —
    // required for OPFS-backed models, where the file is exposed via
    // patched MEMFS stream_ops that route reads to a FileSystemSyncAccessHandle.
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.use_mmap = use_mmap != 0;

    fprintf(stderr, "bench_load: llama_model_load_from_file start\n");
    fflush(stderr);
    g_model = llama_model_load_from_file(model_path, model_params);
    if (!g_model) {
        fprintf(stderr, "bench_load: failed to load model from %s\n", model_path);
        fflush(stderr);
        return -1;
    }
    fprintf(stderr, "bench_load: llama_model_load_from_file done\n");
    fflush(stderr);

    fprintf(stderr, "bench_load: fetch vocab\n");
    fflush(stderr);
    g_vocab = llama_model_get_vocab(g_model);

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = n_ctx;
    ctx_params.n_batch  = n_ctx;
    ctx_params.no_perf  = false; // enable performance counters

    fprintf(stderr, "bench_load: llama_init_from_model start\n");
    fflush(stderr);
    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        fprintf(stderr, "bench_load: failed to create context\n");
        fflush(stderr);
        llama_model_free(g_model);
        g_model = nullptr;
        return -2;
    }
    fprintf(stderr, "bench_load: llama_init_from_model done\n");
    fflush(stderr);

    // Create greedy sampler
    fprintf(stderr, "bench_load: create sampler chain\n");
    fflush(stderr);
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_greedy());

    fprintf(stderr, "bench_load: model loaded, ctx=%d, gpu_layers=%d\n", n_ctx, n_gpu_layers);
    fflush(stderr);
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

// Random-token prefill loop shared by bench_pp and bench_set_depth. Uses BOS
// as the first token if the vocab expects one (only when n_pos_start == 0,
// i.e. the cache is empty), random fillers otherwise. Returns 0 on success,
// or the position at which llama_decode failed.
static int prefill_random_tokens(int n_tokens_total, int n_pos_start) {
    const int32_t n_vocab = llama_vocab_n_tokens(g_vocab);
    const int n_batch = g_n_ctx;
    std::vector<llama_token> tokens(n_batch);

    int n_processed = 0;
    while (n_processed < n_tokens_total) {
        const int n_tokens = std::min(n_tokens_total - n_processed, n_batch);
        const bool first_in_ctx = (n_pos_start == 0 && n_processed == 0);
        tokens[0] = (first_in_ctx && llama_vocab_get_add_bos(g_vocab))
            ? llama_vocab_bos(g_vocab)
            : std::rand() % n_vocab;
        for (int i = 1; i < n_tokens; i++) {
            tokens[i] = std::rand() % n_vocab;
        }
        if (llama_decode(g_ctx, llama_batch_get_one(tokens.data(), n_tokens))) {
            return n_processed > 0 ? n_processed : 1;
        }
        n_processed += n_tokens;
    }
    return 0;
}

// Prefill the KV cache to a given depth so the subsequent bench_pp/bench_tg
// runs at non-zero context. Mirrors the depth setup in llama-bench's main
// loop (tools/llama-bench/llama-bench.cpp lines 2320-2358): clear the cache,
// random-token prefill, then snapshot via llama_state_seq_get_data so a
// follow-up call at the same depth restores the snapshot instead of
// recomputing. Caller times bench_pp/bench_tg only — this call is untimed
// setup, the same way llama-bench's depth fill is excluded from t_start.
//
// n_depth == 0 just clears the cache (no prefill, no snapshot reuse).
const char * bench_set_depth(int n_depth) {
    if (!g_model || !g_ctx) {
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"error\":\"model not loaded\"}");
        return g_result_buf;
    }
    if (n_depth < 0) {
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"error\":\"n_depth must be >= 0\"}");
        return g_result_buf;
    }

    llama_memory_clear(llama_get_memory(g_ctx), false);

    if (n_depth == 0) {
        llama_synchronize(g_ctx);
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"success\":true,\"n_depth\":0,\"cached\":false}");
        return g_result_buf;
    }

    // Cache hit: restore the snapshot we took on a previous call. set_data
    // returns the number of bytes consumed; 0 means the snapshot is
    // incompatible with the current context (e.g. n_ctx changed) and we
    // fall through to re-prefill.
    if (n_depth == g_depth_cached && !g_depth_state.empty()) {
        const size_t ret = llama_state_seq_set_data(
            g_ctx, g_depth_state.data(), g_depth_state.size(), 0);
        if (ret > 0) {
            llama_synchronize(g_ctx);
            snprintf(g_result_buf, sizeof(g_result_buf),
                "{\"success\":true,\"n_depth\":%d,\"cached\":true}", n_depth);
            return g_result_buf;
        }
        g_depth_cached = -1;
        g_depth_state.clear();
    }

    const int err_pos = prefill_random_tokens(n_depth, 0);
    if (err_pos != 0) {
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"error\":\"depth decode failed near processed=%d\"}", err_pos);
        return g_result_buf;
    }
    llama_synchronize(g_ctx);

    g_depth_cached = n_depth;
    g_depth_state.resize(llama_state_seq_get_size(g_ctx, 0));
    llama_state_seq_get_data(g_ctx, g_depth_state.data(), g_depth_state.size(), 0);

    snprintf(g_result_buf, sizeof(g_result_buf),
        "{\"success\":true,\"n_depth\":%d,\"cached\":false}", n_depth);
    return g_result_buf;
}

// llama-bench-style synthetic-token prefill test. Mirrors test_prompt() in
// tools/llama-bench/llama-bench.cpp: BOS as the first token (when the cache
// is empty AND the vocab expects one) and uniformly-random fillers
// thereafter, batched up to n_ctx. Caller times the call; we just run the
// work, synchronize, and return. Unlike bench_run there is no sampler.
//
// Cache state is owned by the caller — bench_set_depth() must run first to
// either clear (d=0) or pre-fill the KV. We don't clear here so depth runs
// stay intact for the timed measurement.
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

    const int n_pos_start = (int) llama_memory_seq_pos_max(llama_get_memory(g_ctx), 0) + 1;
    const int err_pos = prefill_random_tokens(n_prompt, n_pos_start);
    if (err_pos != 0) {
        snprintf(g_result_buf, sizeof(g_result_buf),
            "{\"error\":\"prefill decode failed at processed=%d\"}", err_pos);
        return g_result_buf;
    }
    llama_synchronize(g_ctx);

    snprintf(g_result_buf, sizeof(g_result_buf),
        "{\"success\":true,\"n_prompt\":%d}", n_prompt);
    return g_result_buf;
}

// llama-bench-style decode test. Mirrors test_gen() in
// tools/llama-bench/llama-bench.cpp: prime the first decode with BOS (only
// when the cache is empty AND the vocab expects BOS), random token
// otherwise, then loop n_gen single-token decodes with a synchronize after
// each — the synchronize is what makes wall time reflect per-token GPU
// latency rather than dispatch queue depth.
//
// Cache state is owned by the caller — bench_set_depth() must run first.
// When depth > 0 the cache already holds a BOS at position 0, so we never
// re-prime with BOS here.
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

    const int32_t n_vocab = llama_vocab_n_tokens(g_vocab);
    const int n_pos_start = (int) llama_memory_seq_pos_max(llama_get_memory(g_ctx), 0) + 1;
    llama_token token = (n_pos_start == 0 && llama_vocab_get_add_bos(g_vocab))
        ? llama_vocab_bos(g_vocab)
        : (llama_token)(std::rand() % n_vocab);

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

// Memory snapshot from llama.cpp's perspective: loaded model size, current
// state-buffer size, and per-device free/total from every registered ggml
// backend. Intended to be called after bench_load so model_size reflects the
// loaded model and the per-device free counters reflect what's left after
// allocation. Safe to call before bench_load too (model_size + state_size
// will be 0). Returns the same g_result_buf — caller must consume before the
// next bench_* call overwrites it.
const char* bench_memory_info() {
    const uint64_t model_size = g_model ? llama_model_size(g_model) : 0;
    const size_t   state_size = g_ctx   ? llama_state_get_size(g_ctx) : 0;

    std::string devices_json = "[";
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char* name = ggml_backend_dev_name(dev);
        // The function name `ggml_backend_dev_type` shadows the enum tag, so
        // we have to spell it `enum ggml_backend_dev_type` to refer to the type.
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        const char* type_str = (t == GGML_BACKEND_DEVICE_TYPE_GPU)   ? "GPU"
                             : (t == GGML_BACKEND_DEVICE_TYPE_ACCEL) ? "ACCEL"
                             : "CPU";
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        if (i > 0) devices_json += ",";
        char buf[512];
        snprintf(buf, sizeof(buf),
            "{\"name\":\"%s\",\"type\":\"%s\",\"free\":%zu,\"total\":%zu}",
            name ? name : "", type_str, free_mem, total_mem);
        devices_json += buf;
    }
    devices_json += "]";

    snprintf(g_result_buf, sizeof(g_result_buf),
        "{"
        "\"model_size\":%llu,"
        "\"state_size\":%zu,"
        "\"devices\":%s"
        "}",
        (unsigned long long)model_size,
        state_size,
        devices_json.c_str()
    );
    return g_result_buf;
}

void bench_exit() {
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_ctx)     { llama_free(g_ctx);             g_ctx = nullptr; }
    if (g_model)   { llama_model_free(g_model);     g_model = nullptr; }
    g_vocab = nullptr;
    g_depth_cached = -1;
    g_depth_state.clear();
    llama_backend_free();
}

} // extern "C"

// Dummy main (--no-entry is set, but some linkers want it)
int main() { return 0; }
