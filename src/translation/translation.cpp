#include "translation.h"

#include <thread>
#include <atomic>
#include <chrono>
#include <stack>

static void cb_log_disable(enum ggml_log_level , const char * , void * ) { }

static std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

static std::string clean_translation(std::string out) {
    // 去掉换行，强制单行
    for (char &c : out) {
        if (c == '\r' || c == '\n' || c == '\t') c = ' ';
    }

    out = trim_copy(out);

    // 只保留 Input 之前的部分，Input 及之后全部丢弃
    size_t pos = out.find("Input");
    if (pos != std::string::npos) {
        out = trim_copy(out.substr(0, pos));
    }

    pos = out.find("Terminology");
    if (pos != std::string::npos) {
        out = trim_copy(out.substr(0, pos));
    }

    return out;
}

static std::string send_to_model(
    llama_model * model,
    llama_context * ctx,
    llama_sampler * smpl,
    const llama_vocab * vocab,
    const std::string & sentances,
    bool is_background,
    int n_predict,
    const std::vector<glossary_pair> * glossary_hits
) {
    // llama_memory_clear(llama_get_memory(ctx), true);

    // todo: rag
    std::string prompt;

    if(is_background){
        prompt = 
            "You are a subtitle translation engine.\n"
            "Automatically detect the source language of each sentence and translate everything into Simplified Chinese.\n"
            "Output only translated subtitle text.\n"
            "No explanation, no extra notes, no original text.\n"
            "If one subtitle line is long, insert natural line breaks for reading.\n"
            "Keep meaning accurate, concise, and subtitle-friendly.\n"
            "Do not add labels, explanations, prefixes, or notes.\n";
    } else {
        std::string glossary_block;
        if (glossary_hits && !glossary_hits->empty()) {
            glossary_block += "Terminology constraints (must follow):\n";
            for (const auto & p : *glossary_hits) {
                glossary_block += "- " + p.src + " => " + p.dst + "\n";
            }
            glossary_block += "\n";
        }

        prompt =
            glossary_block +
            "Input:\n" + sentances + "\n"
            "Output:\n";
    }

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

    if(is_background || n_predict <= 0){
        if (!llama_decode(ctx, batch)) return "";
        else return "";
    }

    std::string out;
    int n_decode = 0;

    while (n_decode < n_predict) {
        // forward
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

static std::vector<glossary_pair> collect_glossary_hits(
    const std::vector<glossary_pair> & glossary,
    const std::string & text,
    size_t max_hits = 12
) {
    std::vector<glossary_pair> hits;
    hits.reserve(std::min(max_hits, glossary.size()));

    for (const auto & g : glossary) {
        if (g.src.empty()) continue;
        if (text.find(g.src) != std::string::npos) {
            hits.push_back(g);
            if (hits.size() >= max_hits) break;
        }
    }
    return hits;
}

static bool split_tsv_2cols(const std::string& line, std::string& a, std::string& b) {
    const size_t p = line.find(';');
    if (p == std::string::npos) return false;
    a = trim_copy(line.substr(0, p));
    b = trim_copy(line.substr(p + 1));
    return !a.empty() && !b.empty();
}

static std::vector<glossary_pair> load_glossary_file(const std::string & path) {
    std::vector<glossary_pair> out;
    std::ifstream fin(path);
    if (!fin.is_open()) return out;

    std::string line;
    while (std::getline(fin, line)) {
        line = trim_copy(line);
        if (line.empty()) continue;
        if (!line.empty() && line[0] == '#') continue;

        std::string src, dst;
        if (!split_tsv_2cols(line, src, dst)) {
            continue;
        }

        out.push_back({src, dst});
    }
    
    std::sort(out.begin(), out.end(), [](const glossary_pair & x, const glossary_pair & y) {
        return x.src.size() > y.src.size();
    });

    return out;
}

/**
 * translation entrance
 * 
 * @param atp
 * @param buffer
 */
int translation_start(ai_translation_parmas& atp, pipeline_buffer& buffer) {
    // model path
    std::string model_path = atp.translation_model_path;
    std::vector<SubtitlesEntry> in_srt = buffer.asr_entries;
    std::vector<SubtitlesEntry>& out_srt = buffer.subtitles_entries;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int workers = atp.thread_num? atp.thread_num : 1;
    int n_predict = 64;
    int ngl = -1;
    bool print_log = false;
    bool show_progress = true;

    if (model_path.empty() || in_srt.empty()) {
        std::cerr << "model path is empty" << std::endl;
        return -1;
    }

    // disable log
    if(!print_log) {
        llama_log_set(cb_log_disable, NULL);
    }

    const auto glossary = load_glossary_file("/Users/wang/code/test/src/rag/qwer");

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "failed to load model" << std::endl;
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
            lc.n_ctx = 2048;
            lc.n_batch = 512;
            lc.n_threads = std::max(2u, hw / 2);

            llama_context * tctx = llama_init_from_model(model, lc);
            if (!tctx) return;

            auto sp = llama_sampler_chain_default_params();
            llama_sampler * tsmpl = llama_sampler_chain_init(sp);
            llama_sampler_chain_add(tsmpl, llama_sampler_init_greedy());

            const llama_vocab * tvocab = llama_model_get_vocab(model);

            // video background
            send_to_model(model, tctx, tsmpl, tvocab, "", true, n_predict, {});

            while (true) {
                size_t i = next.fetch_add(1);
                if (i >= entries.size()) break;
                if (!entries[i].text.empty()) {
                    auto hits = collect_glossary_hits(glossary, entries[i].text, 12);
                    std::string trans = send_to_model(model, tctx, tsmpl, tvocab, entries[i].text, false, n_predict, &hits);
                    entries[i].text = clean_translation(trans);
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

    return 0;
}