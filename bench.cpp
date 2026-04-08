// Minimal llama.cpp WASM benchmark wrapper.
// Exports C functions for JavaScript to call: init, load, run, exit.
// Based on llama.cpp/examples/simple/simple.cpp

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
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
static char g_result_buf[8192];

extern "C" {

int bench_init() {
    ggml_backend_load_all();
    return 0;
}

int bench_load(const char * model_path, int n_ctx, int n_gpu_layers) {
    // Clean up previous state if any
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_ctx)     { llama_free(g_ctx);             g_ctx = nullptr; }
    if (g_model)   { llama_model_free(g_model);     g_model = nullptr; }

    g_n_ctx = n_ctx;

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

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
    int n_decoded = 0;
    int n_pos = batch.n_tokens;

    for (int i = 0; i < n_predict; i++) {
        llama_token new_token = llama_sampler_sample(g_sampler, g_ctx, -1);

        if (llama_vocab_is_eog(g_vocab, new_token)) {
            break;
        }

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

    snprintf(g_result_buf, sizeof(g_result_buf),
        "{"
        "\"success\":true,"
        "\"n_prompt_tokens\":%d,"
        "\"n_generated\":%d,"
        "\"t_p_eval_ms\":%.2f,"
        "\"t_eval_ms\":%.2f,"
        "\"n_p_eval\":%d,"
        "\"n_eval\":%d,"
        "\"output\":\"%s\""
        "}",
        n_prompt,
        n_decoded,
        perf.t_p_eval_ms,
        perf.t_eval_ms,
        perf.n_p_eval,
        perf.n_eval,
        escaped_output.c_str()
    );

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
