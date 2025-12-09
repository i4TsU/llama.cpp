/**
 * qwen3vl-inference: Run inference with pre-computed image embeddings
 *
 * This tool loads pre-computed image embeddings and runs inference using
 * only the language model (no mmproj required). This allows running the
 * expensive vision encoding once and reusing the embeddings.
 *
 * Usage:
 *   qwen3vl-inference -m model.gguf -p "Describe this image"
 *
 * The embeddings file (image.embd) must be in the current directory.
 */

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

// Must match the header in qwen3vl-preprocess.cpp
struct img_embd_file_header {
    char magic[4];      // "IMGE"
    int32_t version;    // 1
    int32_t n_tokens;   // Number of image tokens
    int32_t n_embd;     // Embedding dimension
    int32_t nx;         // Grid width (for M-RoPE)
    int32_t ny;         // Grid height (for M-RoPE)
    int32_t n_pos;      // Number of positions for M-RoPE
    int32_t reserved;   // Reserved
};

// Helper struct for managing embedding batches with M-RoPE support
// Adapted from mtmd-helper.cpp's decode_embd_batch
struct embd_batch {
    int n_pos_per_embd;
    int n_embd_dim;
    std::vector<llama_pos> pos;
    std::vector<llama_pos> pos_view;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t> logits;
    llama_batch batch;

    embd_batch(float * embd, int32_t n_tokens, int n_pos_per_embd, int n_embd_dim)
        : n_pos_per_embd(n_pos_per_embd), n_embd_dim(n_embd_dim) {
        pos.resize(n_tokens * n_pos_per_embd);
        n_seq_id.resize(n_tokens);
        seq_ids.resize(n_tokens + 1);
        logits.resize(n_tokens);
        seq_id_0.resize(1);
        seq_ids[n_tokens] = nullptr;

        batch = {
            /*n_tokens   =*/ n_tokens,
            /*tokens     =*/ nullptr,
            /*embd       =*/ embd,
            /*pos        =*/ pos.data(),
            /*n_seq_id   =*/ n_seq_id.data(),
            /*seq_id     =*/ seq_ids.data(),
            /*logits     =*/ logits.data(),
        };
    }

    // Set up M-RoPE 2D positions for image grid
    void set_position_mrope_2d(llama_pos pos_0, int nx, int ny, llama_seq_id seq_id) {
        seq_id_0[0] = seq_id;
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                int i = y * nx + x;
                pos[i]                      = pos_0;     // temporal
                pos[i + batch.n_tokens]     = pos_0 + y; // height
                pos[i + batch.n_tokens * 2] = pos_0 + x; // width
                pos[i + batch.n_tokens * 3] = 0;         // unused
            }
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = seq_id_0.data();
            batch.logits[i] = false;
        }
    }

    // Set normal (non-M-RoPE) positions
    void set_position_normal(llama_pos pos_0, llama_seq_id seq_id) {
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.pos[i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = seq_id_0.data();
            batch.logits[i] = false;
        }
    }

    // Get a view of a subset of the batch for chunked processing
    llama_batch get_view(int offset, int n_tokens_view) {
        llama_pos * pos_ptr;
        pos_view.clear();

        if (n_pos_per_embd > 1) {
            // M-RoPE: positions are stored in planar format
            pos_view.reserve(n_tokens_view * n_pos_per_embd);
            for (int d = 0; d < n_pos_per_embd; d++) {
                size_t src_idx = d * batch.n_tokens + offset;
                pos_view.insert(pos_view.end(),
                    pos.data() + src_idx,
                    pos.data() + src_idx + n_tokens_view);
            }
            pos_ptr = pos_view.data();
        } else {
            pos_ptr = pos.data() + offset;
        }

        return {
            /*n_tokens   =*/ n_tokens_view,
            /*tokens     =*/ nullptr,
            /*embd       =*/ batch.embd + offset * n_embd_dim,
            /*pos        =*/ pos_ptr,
            /*n_seq_id   =*/ batch.n_seq_id + offset,
            /*seq_id     =*/ batch.seq_id + offset,
            /*logits     =*/ batch.logits + offset,
        };
    }
};

static void print_usage(int argc, char ** argv) {
    fprintf(stderr, "\nUsage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "Run inference with pre-computed image embeddings.\n\n");
    fprintf(stderr, "Required options:\n");
    fprintf(stderr, "  -m, --model FILE      Path to the language model\n");
    fprintf(stderr, "\nOptional:\n");
    fprintf(stderr, "  -p, --prompt TEXT     Prompt to use (default: 'Describe this image.')\n");
    fprintf(stderr, "  -n, --predict N       Number of tokens to predict (default: 512)\n");
    fprintf(stderr, "\nThe embeddings file 'image.embd' must be in the current directory.\n");
    fprintf(stderr, "\n");
}

static bool load_embeddings(const std::string & path, img_embd_file_header & header,
                            std::vector<float> & embeddings) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "error: failed to open embeddings file '%s'\n", path.c_str());
        return false;
    }

    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in.good()) {
        fprintf(stderr, "error: failed to read header from '%s'\n", path.c_str());
        return false;
    }

    if (strncmp(header.magic, "IMGE", 4) != 0) {
        fprintf(stderr, "error: invalid magic in '%s' (expected 'IMGE')\n", path.c_str());
        return false;
    }

    if (header.version != 1) {
        fprintf(stderr, "error: unsupported version %d in '%s'\n", header.version, path.c_str());
        return false;
    }

    size_t expected_size = (size_t)header.n_tokens * header.n_embd;
    embeddings.resize(expected_size);

    in.read(reinterpret_cast<char*>(embeddings.data()), expected_size * sizeof(float));
    if (!in.good()) {
        fprintf(stderr, "error: failed to read embeddings from '%s'\n", path.c_str());
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 512;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    if (params.model.path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        print_usage(argc, argv);
        return 1;
    }

    std::string embd_file = "image.embd";

    if (params.prompt.empty()) {
        params.prompt = "Describe this image.";
    }

    // Initialize backend
    llama_backend_init();

    // Load the language model
    fprintf(stderr, "Loading language model...\n");
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (!model || !ctx) {
        fprintf(stderr, "error: failed to load model or create context\n");
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_batch = llama_n_batch(ctx);

    // Load pre-computed embeddings
    fprintf(stderr, "Loading embeddings from '%s'...\n", embd_file.c_str());
    img_embd_file_header header;
    std::vector<float> embeddings;

    if (!load_embeddings(embd_file, header, embeddings)) {
        llama_backend_free();
        return 1;
    }

    fprintf(stderr, "Loaded %d image tokens (grid=%dx%d, dim=%d, n_pos=%d)\n",
            header.n_tokens, header.nx, header.ny, header.n_embd, header.n_pos);

    // Check embedding dimensions - Qwen3-VL uses deep stack layers where n_embd_inp = n_embd * 4
    int model_n_embd_inp = llama_model_n_embd_inp(model);

    if (model_n_embd_inp == header.n_embd * 4) {
        // Apply 2x2 patch merging (pixel shuffle) for Qwen3-VL deep stack
        fprintf(stderr, "Applying 2x2 patch merging for deep stack layers...\n");

        int new_nx = header.nx / 2;
        int new_ny = header.ny / 2;
        int new_n_embd = header.n_embd * 4;
        std::vector<float> merged(new_nx * new_ny * new_n_embd);

        for (int y = 0; y < new_ny; y++) {
            for (int x = 0; x < new_nx; x++) {
                int src_x = x * 2;
                int src_y = y * 2;

                int dst_idx = (y * new_nx + x) * new_n_embd;

                // Source indices for 2x2 block (TL, TR, BL, BR)
                int src_tl = (src_y * header.nx + src_x) * header.n_embd;
                int src_tr = (src_y * header.nx + (src_x + 1)) * header.n_embd;
                int src_bl = ((src_y + 1) * header.nx + src_x) * header.n_embd;
                int src_br = ((src_y + 1) * header.nx + (src_x + 1)) * header.n_embd;

                // Concatenate: TL, TR, BL, BR
                memcpy(merged.data() + dst_idx, embeddings.data() + src_tl, header.n_embd * sizeof(float));
                memcpy(merged.data() + dst_idx + header.n_embd, embeddings.data() + src_tr, header.n_embd * sizeof(float));
                memcpy(merged.data() + dst_idx + header.n_embd * 2, embeddings.data() + src_bl, header.n_embd * sizeof(float));
                memcpy(merged.data() + dst_idx + header.n_embd * 3, embeddings.data() + src_br, header.n_embd * sizeof(float));
            }
        }

        header.nx = new_nx;
        header.ny = new_ny;
        header.n_embd = new_n_embd;
        header.n_tokens = new_nx * new_ny;
        embeddings = std::move(merged);

        fprintf(stderr, "Merged to %d tokens (grid=%dx%d, dim=%d)\n",
                header.n_tokens, header.nx, header.ny, header.n_embd);
    } else if (header.n_embd != model_n_embd_inp) {
        fprintf(stderr, "error: embedding dimension mismatch (file: %d, model expects: %d)\n",
                header.n_embd, model_n_embd_inp);
        llama_backend_free();
        return 1;
    }

    // Initialize sampler
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Create batch for text tokens
    llama_batch text_batch = llama_batch_init(n_batch, 0, 1);

    // Helper to decode text tokens
    auto decode_text = [&](const std::string & text, llama_pos & pos, bool logits_last) -> bool {
        std::vector<llama_token> tokens = common_tokenize(ctx, text, false, true);
        if (tokens.empty()) return true;

        for (size_t i = 0; i < tokens.size(); ) {
            text_batch.n_tokens = 0;

            for (; i < tokens.size() && text_batch.n_tokens < n_batch; i++) {
                int j = text_batch.n_tokens;
                text_batch.token[j] = tokens[i];
                text_batch.pos[j] = pos++;
                text_batch.n_seq_id[j] = 1;
                text_batch.seq_id[j][0] = 0;
                text_batch.logits[j] = (logits_last && i == tokens.size() - 1);
                text_batch.n_tokens++;
            }

            if (llama_decode(ctx, text_batch) != 0) {
                fprintf(stderr, "error: failed to decode text\n");
                return false;
            }
        }
        return true;
    };

    llama_pos n_past = 0;

    // Build the full prompt with proper chat template structure
    // For Qwen3-VL the format is:
    // <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    // <|im_start|>user\n<|vision_start|>[IMAGE]<|vision_end|>PROMPT<|im_end|>\n
    // <|im_start|>assistant\n

    fprintf(stderr, "Processing prompt...\n");

    // 1. System prompt
    std::string system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    if (!decode_text(system_prompt, n_past, false)) {
        llama_sampler_free(smpl);
        llama_batch_free(text_batch);
        llama_backend_free();
        return 1;
    }

    // 2. User turn start with vision_start token
    std::string user_start = "<|im_start|>user\n<|vision_start|>";
    if (!decode_text(user_start, n_past, false)) {
        llama_sampler_free(smpl);
        llama_batch_free(text_batch);
        llama_backend_free();
        return 1;
    }

    // 3. Decode image embeddings with M-RoPE
    fprintf(stderr, "Decoding %d image tokens...\n", header.n_tokens);

    // Determine if we need M-RoPE (4 position dimensions) or standard positions
    // For Qwen3-VL, we use M-RoPE with 2D positions
    bool use_mrope = (header.nx > 0 && header.ny > 0);
    int n_pos_per_embd = use_mrope ? 4 : 1;

    embd_batch img_batch(embeddings.data(), header.n_tokens, n_pos_per_embd, header.n_embd);

    if (use_mrope) {
        img_batch.set_position_mrope_2d(n_past, header.nx, header.ny, 0);
        // For non-causal attention during image processing
        llama_set_causal_attn(ctx, false);
    } else {
        img_batch.set_position_normal(n_past, 0);
    }

    // Process image embeddings in batches
    int n_img_tokens = header.n_tokens;
    for (int offset = 0; offset < n_img_tokens; ) {
        int chunk_size = std::min(n_batch, n_img_tokens - offset);
        llama_batch view = img_batch.get_view(offset, chunk_size);

        if (llama_decode(ctx, view) != 0) {
            fprintf(stderr, "error: failed to decode image embeddings at offset %d\n", offset);
            llama_set_causal_attn(ctx, true);
            llama_sampler_free(smpl);
            llama_batch_free(text_batch);
            llama_backend_free();
            return 1;
        }

        offset += chunk_size;
        fprintf(stderr, "  processed %d/%d image tokens\n", offset, n_img_tokens);
    }

    if (use_mrope) {
        llama_set_causal_attn(ctx, true);
    }

    // Advance position by n_pos (for M-RoPE this is max(t,h,w), otherwise n_tokens)
    n_past += (header.n_pos > 0) ? header.n_pos : header.n_tokens;

    // 4. Vision end, user prompt, and assistant start
    std::string user_end = "<|vision_end|>" + params.prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (!decode_text(user_end, n_past, true)) {
        llama_sampler_free(smpl);
        llama_batch_free(text_batch);
        llama_backend_free();
        return 1;
    }

    // 5. Generate response
    fprintf(stderr, "Generating response...\n\n");

    int n_predict = params.n_predict;
    for (int i = 0; i < n_predict; i++) {
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        std::string piece = common_token_to_piece(ctx, new_token);
        printf("%s", piece.c_str());
        fflush(stdout);

        // Prepare batch for next token
        text_batch.n_tokens = 1;
        text_batch.token[0] = new_token;
        text_batch.pos[0] = n_past++;
        text_batch.n_seq_id[0] = 1;
        text_batch.seq_id[0][0] = 0;
        text_batch.logits[0] = true;

        if (llama_decode(ctx, text_batch) != 0) {
            fprintf(stderr, "\nerror: failed to decode generated token\n");
            break;
        }
    }

    printf("\n");

    // Cleanup
    llama_sampler_free(smpl);
    llama_batch_free(text_batch);
    llama_backend_free();

    return 0;
}
