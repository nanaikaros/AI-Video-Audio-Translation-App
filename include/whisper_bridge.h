#pragma once

#include "params.h"
#include "progress_ipc.h"

#include <string>
#include <vector>

struct whisper_context;

int whisper_start(ai_translation_parmas&, output_params&, pipeline_buffer&);