#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>

struct img_embd_file_header {
    char magic[4];     // "Q3VL"
    int32_t version;   // 1
    int32_t n_tokens;
    int32_t n_embd;
    int32_t nx;
    int32_t ny;
};

static void print_usage(int argc, char ** argv) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "Check qwen3vl-inference.cpp for options\n");
}

int main(int argc, char ** argv) {
    common_params params;
    // Set some defaults
    params.n_gpu_layers = 999;
    params.n_ubatch = 2048; // Ensure ubatch is large enough for image tokens
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // Force disable for verification
    
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }
    
    if (params.model.path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        return 1;
    }

    // Initialize backend
    llama_backend_init();
    
    // Load Model (LLM ONLY - No mmproj loaded!)
    common_init_result llama_init = common_init_from_params(params);
    
    // Access unique_ptrs
    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();
    
    if (model == NULL || ctx == NULL) {
        fprintf(stderr, "%s: error: failed to load model/context\n", __func__);
        return 1;
    }
    
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Initialize Sampler
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Load Image Embeddings
    std::string img_file = "image.img_embd"; // Fixed name for now matching preprocess
    std::ifstream in(img_file, std::ios::binary);
    if (!in) {
        fprintf(stderr, "error: failed to open '%s'\n", img_file.c_str());
        return 1;
    }
    
    img_embd_file_header header;
    in.read((char*)&header, sizeof(header));
    
    if (strncmp(header.magic, "Q3VL", 4) != 0) {
        fprintf(stderr, "error: invalid magic in '%s'\n", img_file.c_str());
        return 1;
    }
    
    std::vector<float> embeddings(header.n_tokens * header.n_embd);
    in.read((char*)embeddings.data(), embeddings.size() * sizeof(float));
    in.close();
    
    fprintf(stderr, "Loaded %d image tokens (nx=%d, ny=%d)\n", header.n_tokens, header.nx, header.ny);
    fprintf(stderr, "header.n_embd: %d\n", header.n_embd);
    fprintf(stderr, "model n_embd_inp: %d\n", llama_model_n_embd_inp(model));
    
    if (llama_model_n_embd_inp(model) == header.n_embd * 4) {
        fprintf(stderr, "Applying 2x2 patch merging (Pixel Shuffle)...\n");
        int new_nx = header.nx / 2;
        int new_ny = header.ny / 2;
        int new_n_embd = header.n_embd * 4;
        std::vector<float> merged_embeddings(new_nx * new_ny * new_n_embd);

        for (int y = 0; y < new_ny; y++) {
            for (int x = 0; x < new_nx; x++) {
                int src_x = x * 2;
                int src_y = y * 2;
                
                // Destination index for the merged token
                int dst_idx = (y * new_nx + x) * new_n_embd;
                
                // Source indices for 2x2 block
                int src_idx_tl = (src_y * header.nx + src_x) * header.n_embd;                 // Top-Left
                int src_idx_tr = (src_y * header.nx + (src_x + 1)) * header.n_embd;           // Top-Right
                int src_idx_bl = ((src_y + 1) * header.nx + src_x) * header.n_embd;           // Bottom-Left
                int src_idx_br = ((src_y + 1) * header.nx + (src_x + 1)) * header.n_embd;     // Bottom-Right
                
                // Copy data: TL, TR, BL, BR
                memcpy(merged_embeddings.data() + dst_idx, 
                       embeddings.data() + src_idx_tl, header.n_embd * sizeof(float));
                memcpy(merged_embeddings.data() + dst_idx + header.n_embd, 
                       embeddings.data() + src_idx_tr, header.n_embd * sizeof(float));
                memcpy(merged_embeddings.data() + dst_idx + header.n_embd * 2, 
                       embeddings.data() + src_idx_bl, header.n_embd * sizeof(float));
                memcpy(merged_embeddings.data() + dst_idx + header.n_embd * 3, 
                       embeddings.data() + src_idx_br, header.n_embd * sizeof(float));
            }
        }
        
        // Update header info and embeddings
        header.nx = new_nx;
        header.ny = new_ny;
        header.n_embd = new_n_embd;
        header.n_tokens = new_nx * new_ny;
        embeddings = std::move(merged_embeddings);
        fprintf(stderr, "Merged: %d tokens (nx=%d, ny=%d, dim=%d)\n", header.n_tokens, header.nx, header.ny, header.n_embd);
    } else if (header.n_embd != llama_model_n_embd_inp(model)) {
        fprintf(stderr, "error: Embedding dimension mismatch! File: %d, Model: %d. No automatic fix available.\n", header.n_embd, llama_model_n_embd_inp(model));
        return 1;
    }

    // Prepare Batch
    llama_batch batch = llama_batch_init(8192, 0, 1); // Ensure enough capacity

    // Helper to decode text
    auto decode_text = [&](const std::string & text, int32_t & pos) {
        std::vector<llama_token> toks = common_tokenize(ctx, text, false, true);
        if (toks.empty()) return;
        
        batch.n_tokens = toks.size();
        for (size_t i = 0; i < toks.size(); i++) {
            batch.token[i] = toks[i];
            batch.pos[i] = pos + (int32_t)i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; 
            batch.logits[i] = (i == toks.size() - 1); // Enable logits for the last token
        }
        batch.embd = nullptr;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "failed to decode text: %s\n", text.c_str());
            exit(1);
        }
        pos += (int32_t)toks.size();
    };

    // Construct M-RoPE Positions
    int32_t n_past = 0; 
    
    // 1. Decode "<|vision_start|>"
    decode_text("<|vision_start|>", n_past);

    // 2. Decode Image Embeddings with M-RoPE
    int32_t n_total = header.n_tokens;
    // For non-causal attention, we must not exceed n_ubatch per call
    int32_t n_batch_size = 512; // default safe
    if (llama_n_ubatch(ctx) > 0) {
        n_batch_size = llama_n_ubatch(ctx);
    }
    
    // Allocate M-RoPE positions (4 dimensions per token)
    std::vector<llama_pos> mrope_pos(n_total * 4);
    
    int32_t base_pos = n_past; // The temporal position for the image
    
    for (int y = 0; y < header.ny; y++) {
        for (int x = 0; x < header.nx; x++) {
            long long i = (long long)y * header.nx + x;
            if (i >= n_total) break; 
            
            mrope_pos[i]               = base_pos;     // dim 0: time
            mrope_pos[i + n_total]     = base_pos + y; // dim 1: height
            mrope_pos[i + n_total * 2] = base_pos + x; // dim 2: width
            mrope_pos[i + n_total * 3] = 0;            // dim 3: unused
        }
    }
    
    llama_set_causal_attn(ctx, false);
    
    // Save original batch pointers to restore later
    llama_token * original_token = batch.token;
    llama_pos   * original_pos   = batch.pos;

    int32_t offset = 0;
    while (offset < n_total) {
        int32_t chunk_size = std::min(n_batch_size, n_total - offset);
        
        batch.n_tokens = chunk_size;
        batch.embd = embeddings.data() + (offset * header.n_embd);
        batch.token = nullptr;
        
        // Construct Pos View
        static std::vector<llama_pos> pos_view;
        pos_view.clear();
        pos_view.reserve(chunk_size * 4);
        
        for (int d = 0; d < 4; d++) {
             for (int i = 0; i < chunk_size; i++) {
                 // mrope_pos is PLANAR: [Time 0..N, Height 0..N, Width 0..N, Unused 0..N]
                 // We want Dim d for token 'offset + i'.
                 // Index = d * n_total + (offset + i);
                 int32_t dim_index = d * n_total + (offset + i);
                 pos_view.push_back(mrope_pos[dim_index]);
             }
        }
        batch.pos = pos_view.data(); // Override pos pointer
        
        for (int i = 0; i < chunk_size; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = false; 
        }
        
        if (batch.embd == nullptr || batch.pos == nullptr || pos_view.empty()) {
            fprintf(stderr, "error: invalid batch pointers or empty pos_view\n");
            return 1;
        }
        
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "error: llama_decode failed at offset %d\n", offset);
            return 1;
        }
        
        offset += chunk_size;
        fprintf(stderr, "Processed %d/%d image tokens\n", offset, n_total);
        fflush(stderr);
    }
    
    // Restore original batch pointers
    batch.token = original_token;
    batch.pos   = original_pos;

    llama_set_causal_attn(ctx, true);
    
    // Increment pos by 1 (M-RoPE logic for Qwen3-VL: image takes 1 temporal unit)
    n_past += 1;

    // 3. Decode "<|vision_end|>"
    decode_text("<|vision_end|>", n_past);
    
    // 4. Decode Prompt (Chat Template)
    std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>";
    // Image tokens are inserted here conceptually, but we already processed them.
    // The model expects <|vision_start|>...<|vision_end|> inside the user content.
    // We already processed <|vision_start|>...image...<|vision_end|>.
    // So we just need to append the text part of the user message and then start the assistant turn.
    
    // Actually, looking at how we processed things:
    // 1. Loaded image
    // 2. Decoded "<|vision_start|>" -> image tokens -> "<|vision_end|>"
    // So we are currently at the state where <|vision_end|> was just processed.
    
    // We need to verify where <|vision_start|> was decoded.
    // In code: 
    // decode_text("<|vision_start|>", n_past);
    // process image...
    // decode_text("<|vision_end|>", n_past);
    
    // So we need to prepend the system prompt and the user start tag *before* the vision start?
    // Or just wrap the "Describe this image" part?
    
    // Correct order:
    // <|im_start|>system...<|im_end|>
    // <|im_start|>user
    // <|vision_start|> [IMAGE] <|vision_end|> Describe this image. <|im_end|>
    // <|im_start|>assistant
    
    // We have ALREADY processed <|vision_start|>...<|vision_end|>.
    // This is awkward. We should have output the system preamble FIRST.
    // But since that's water under the bridge for the `main` flow structure (unless I change order),
    // I will try to just add the rest of the user prompt and the assistant start.
    // BUT the <|vision_start|>...<|vision_end|> block is floating with no <|im_start|>user before it?
    // That confuses the model.
    
    // I MUST move the system prompt and user start to BEFORE the image processing.
    // But I can't easily jump back in lines with this tool.
    
    // Alternative: Just fix the text being decoded NOW.
    // "Describe this image.<|im_end|>\n<|im_start|>assistant\n"
    // This assumes the model can tolerate the missing <|im_start|>user and system prompt at the very beginning.
    // Qwen models are usually sensitive to template.
    
    // Let's scroll up and see where vision_start is.
    // It is at line 168. 
    
    // I will modify this specific block to close the user turn and start the assistant turn.
    // Maybe that's enough to kick it out of "EOS mode".
    decode_text("Describe this image.<|im_end|>\n<|im_start|>assistant\n", n_past);
    int n_predict = 512;
    for (int i = 0; i < n_predict; i++) {
        // Sample
        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        
        // Check EOS
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }
        
        // Print
        std::string piece = common_token_to_piece(ctx, new_token_id);
        printf("%s", piece.c_str());
        fflush(stdout);
        
        // Decode new token
        batch.n_tokens = 1;
        batch.token = &new_token_id; 
        batch.embd = nullptr;
        
        // Restore/Reset pos pointer to valid memory! 
        // We overrode batch.pos with pos_view.data() previously.
        // We need to point it back to something. 
        // Since llama_batch_init allocated memory for pos, but we lost the pointer to it (it's in the struct but we overwrote it).
        // Wait, `llama_batch` structure: `llama_pos * pos`.
        // `llama_batch_init` allocates the array and sets `pos` to point to it.
        // If we overwrote `pos`, we lost the original buffer pointer.
        // BUT `llama_batch_free` handles freeing. It relies on `batch.token` or `batch.embd`?
        // Actually `llama_batch_init` does `detail::batch_init`. It creates a single buffer and points distinct pointers into it.
        // If we lose `pos`, we might cause issues if we try to use it again without setting it validly, BUT `llama_batch_free` frees the `tokens` pointer (which starts the block). 
        // So memory leak is unlikely if we don't restore, BUT we need a valid pointer to Write to for this decode!
        
        llama_pos pos_single = n_past;
        batch.pos = &pos_single; // Point to stack variable
        
        batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = true;
        
        if (llama_decode(ctx, batch) != 0) {
             fprintf(stderr, "failed to decode generated token\n");
             break;
        }
        n_past++;
    }
    printf("\n");

    llama_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
