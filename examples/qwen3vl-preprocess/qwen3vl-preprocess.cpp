#include "arg.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

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
    fprintf(stderr, "Check qwen3vl-preprocess.cpp for options\n");
}

int main(int argc, char ** argv) {
    mtmd_context_params ctx_params = mtmd_context_params_default();
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, print_usage)) {
        return 1;
    }

    if (params.mmproj.path.empty() || params.image.empty()) {
        fprintf(stderr, "error: --mmproj and --image are required\n");
        return 1;
    }
    
    // Hardcode some defaults if not set, to ensure functionality
    if (params.model.path.empty()) {
        fprintf(stderr, "error: --model is required for vocabulary loading\n");
        return 1;
    }

    llama_backend_init();
    
    // Load LLM (we just need it for vocab and mtmd_context)
    common_init_result llama_init = common_init_from_params(params);
    
    // Convert shared_ptr/unique_ptr to raw pointer if needed
    // common_init_result uses llama_model_ptr which is smart pointer
    llama_model * model = llama_init.model.get(); 
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }

    // Initialize MTMD context
    mtmd_context * mctx = mtmd_init_from_file(params.mmproj.path.c_str(), model, ctx_params);
    if (!mctx) {
        fprintf(stderr, "%s: error: failed to init mtmd from '%s'\n", __func__, params.mmproj.path.c_str());
        return 1;
    }

    // Load Image
    std::vector<mtmd_bitmap *> bitmaps;
    for (const auto & img_path : params.image) {
        mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(mctx, img_path.c_str());
        if (!bmp) {
            fprintf(stderr, "error: failed to load image '%s'\n", img_path.c_str());
            return 1;
        }
        bitmaps.push_back(bmp);
    }
    
    // Use const mtmd_bitmap ** for tokenize
    std::vector<const mtmd_bitmap *> bitmaps_ptrs;
    for (auto bmp : bitmaps) bitmaps_ptrs.push_back(bmp);

    // Prepare Tokenization Input (Just the image)
    mtmd_input_text input_text;
    input_text.text = "<__media__>"; 
    input_text.add_special = false;
    input_text.parse_special = false;

    // Tokenize
    // Provide chunks object
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    
    if (mtmd_tokenize(mctx, chunks, &input_text, bitmaps_ptrs.data(), bitmaps_ptrs.size()) != 0) {
         fprintf(stderr, "error: mtmd_tokenize failed\n");
         return 1;
    }

    // Process Chunks
    int32_t nx = 0, ny = 0;
    std::vector<float> final_embeddings;
    
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        enum mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);
        
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            // Encode
            if (mtmd_encode_chunk(mctx, chunk) != 0) {
                fprintf(stderr, "error: mtmd_encode_chunk failed\n");
                return 1;
            }
            
            // Extract Basics
            float * embd = mtmd_get_output_embd(mctx);
            size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            // n_embd? mtmd_context doesn't seem to expose n_embd directly in header?
            // Actually it relies on llama_model_n_embd_inp(model) typically?
            // But mtmd.h says: "mtmd_get_output_embd... reading size ... llama_model_n_embd(model) * n_tokens"
            // So we use model n_embd.
            int n_embd = llama_model_n_embd(model); // or n_embd_inp? 
            // `mtmd.h` comment: "llama_model_n_embd(model)"
            // Let's assume standard embedding size.
            
            // Extract M-RoPE Metadata (CRITICAL)
            const mtmd_image_tokens * img_toks = mtmd_input_chunk_get_tokens_image(chunk);
            nx = (int32_t)mtmd_image_tokens_get_nx(img_toks);
            ny = (int32_t)mtmd_image_tokens_get_ny(img_toks);
            
            if (final_embeddings.size() > 0) {
                fprintf(stderr, "warning: multiple image chunks detected, appending... (might not be supported by simple inference tool)\n");
            }
            
            final_embeddings.insert(final_embeddings.end(), embd, embd + (n_tokens * n_embd));
        }
    }
    
    if (final_embeddings.empty()) {
        fprintf(stderr, "error: no image embeddings generated\n");
        return 1;
    }

    // Save to File
    std::string out_file = "image.img_embd";
    
    std::ofstream out(out_file, std::ios::binary);
    if (!out) {
         fprintf(stderr, "error: failed to open output file '%s'\n", out_file.c_str());
         return 1;
    }
    
    img_embd_file_header header;
    memcpy(header.magic, "Q3VL", 4);
    header.version = 1;
    int n_embd = llama_model_n_embd(model);
    header.n_tokens = (int32_t)(final_embeddings.size() / n_embd);
    header.n_embd = n_embd;
    header.nx = nx;
    header.ny = ny;
    
    out.write((char*)&header, sizeof(header));
    out.write((char*)final_embeddings.data(), final_embeddings.size() * sizeof(float));
    out.close();
    
    fprintf(stderr, "Saved %d tokens (dim %d, nx=%d, ny=%d) to %s\n", header.n_tokens, header.n_embd, header.nx, header.ny, out_file.c_str());

    // Clean up
    mtmd_input_chunks_free(chunks);
    for (auto bmp : bitmaps) mtmd_bitmap_free(bmp);
    mtmd_free(mctx);
    
    llama_backend_free();
    return 0;
}
