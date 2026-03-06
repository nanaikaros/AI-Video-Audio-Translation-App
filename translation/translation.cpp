#include "translation.h"

#include <thread>
#include <atomic>
#include <chrono>

std::shared_ptr<spdlog::logger> logger_trans;

static void cb_log_disable(enum ggml_log_level , const char * , void * ) { }

static std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

// todo: 多种语言翻译
static std::string translate_to_zh(
    llama_model * model,
    llama_context * ctx,
    llama_sampler * smpl,
    const llama_vocab * vocab,
    const std::string & sentances,
    int n_predict
) {
    llama_memory_clear(llama_get_memory(ctx), true);

    std::string prompt =
        "You are a subtitle translation engine.\n"
        "Automatically detect the source language of each sentence and translate everything into Simplified Chinese.\n"
        "Output only translated subtitle text.\n"
        "No explanation, no extra notes, no original text.\n"
        "If one subtitle line is long, insert natural line breaks for reading.\n"
        "Keep meaning accurate, concise, and subtitle-friendly.\n"
        "Input:\n" + sentances + "\n"
        "Output:\n";

    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        return "";
    }

    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        return "";
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            return "";
        }
        llama_token decoder_start = llama_model_decoder_start_token(model);
        if (decoder_start == LLAMA_TOKEN_NULL) {
            decoder_start = llama_vocab_bos(vocab);
        }
        batch = llama_batch_get_one(&decoder_start, 1);
    }

    std::string out;
    int n_decode = 0;

    while (n_decode < n_predict) {
        if (llama_decode(ctx, batch)) {
            break;
        }

        llama_token token = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            out.append(buf, n);
        }

        batch = llama_batch_get_one(&token, 1);
        n_decode += 1;
    }

    return trim_copy(out);
}

/**
 * translation entrance
 * 
 * @param atp
 * @param buffer
 */
int translation_start(ai_translation_parmas& atp, pipeline_buffer& buffer) {
    logger_trans = spdlog::get("app_logger");

    // model path
    std::string model_path = atp.translation_model_path;
    std::vector<SubtitlesEntry> in_srt = buffer.asr_entries;
    std::vector<SubtitlesEntry>& out_srt = buffer.subtitles_entries;
    int workers = atp.thread_num? atp.thread_num : 
        std::max(1u, std::thread::hardware_concurrency() / 2);
    int n_predict = 64;
    int ngl = 999;
    bool print_log = false;
    bool show_progress = true;

    if (model_path.empty() || in_srt.empty()) {
        logger_trans->error("params error");
        return -1;
    }

    // disable log
    if(!print_log) {
        llama_log_set(cb_log_disable, NULL);
    }

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        logger_trans->error("failed to load model");
        // std::cerr << "failed to load model\n";
        return -1;
    }

    auto entries = in_srt;

    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};
    const size_t total = entries.size();

    std::atomic<bool> stop_progress{false};
    std::thread progress_thread([&]() {
        int last_percent = -1;
        while (!stop_progress.load()) {
            const size_t d = done.load(std::memory_order_relaxed);
            const int p = total ? (int)((100.0 * d) / total) : 100;
            if (p != last_percent) {
                progress_ipc_send("translation", p);
                last_percent = p;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        }
    });

    std::vector<std::thread> pool;
    pool.reserve(workers);

    for (int t = 0; t < workers; ++t) {
        pool.emplace_back([&]() {
            llama_context_params lc = llama_context_default_params();
            lc.n_ctx = 1024;
            lc.n_batch = 256;
            lc.n_threads = (ngl > 0 ? 1 : 4);

            llama_context * tctx = llama_init_from_model(model, lc);
            if (!tctx) return;

            auto sp = llama_sampler_chain_default_params();
            llama_sampler * tsmpl = llama_sampler_chain_init(sp);
            llama_sampler_chain_add(tsmpl, llama_sampler_init_greedy());

            const llama_vocab * tvocab = llama_model_get_vocab(model);

            while (true) {
                size_t i = next.fetch_add(1);
                if (i >= entries.size()) break;
                if (!entries[i].text.empty()) {
                    std::string temp = translate_to_zh(model, tctx, tsmpl, tvocab, entries[i].text, n_predict);
                    entries[i].text = temp;
                }
                done.fetch_add(1, std::memory_order_relaxed);
            }

            llama_sampler_free(tsmpl);
            llama_free(tctx);
        });
    }

    for (auto & th : pool) th.join();
    stop_progress.store(true);
    if (progress_thread.joinable()) progress_thread.join();

    out_srt = entries;

    llama_model_free(model);

    logger_trans->info("translation done!");

    return 0;
}