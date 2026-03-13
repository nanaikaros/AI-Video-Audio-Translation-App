#pragma once
#include "llama.h"

#include "params.h"
#include "progress_ipc.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string translate_ko_to_zh(
    llama_model * model,
    llama_context * ctx,
    llama_sampler * smpl,
    const llama_vocab * vocab,
    const std::string & korean,
    int n_predict
);

int translation_start(ai_translation_parmas&, pipeline_buffer&);