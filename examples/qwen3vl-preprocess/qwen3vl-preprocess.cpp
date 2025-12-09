/**
 * qwen3vl-preprocess: Pre-compute image embeddings for Qwen3-VL models
 *
 * This tool processes images through the vision encoder (mmproj) and saves
 * the embeddings to a file, allowing the inference tool to run with just
 * the language model loaded (no mmproj required at inference time).
 *
 * Usage:
 *   qwen3vl-preprocess -m model.gguf --mmproj mmproj.gguf --image photo.jpg -o output.embd
 */

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

// File format for pre-computed image embeddings
struct img_embd_file_header {
    char magic[4];      // "IMGE" (IMaGe Embedding)
    int32_t version;    // Format version (1)
    int32_t n_tokens;   // Number of image tokens
    int32_t n_embd;     // Embedding dimension
    int32_t nx;         // Grid width (for M-RoPE)
    int32_t ny;         // Grid height (for M-RoPE)
    int32_t n_pos;      // Number of positions for M-RoPE (max of temporal, height, width)
    int32_t reserved;   // Reserved for future use
};

static void print_usage(int argc, char ** argv) {
    fprintf(stderr, "\nUsage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "Pre-compute image embeddings for Qwen3-VL models.\n\n");
    fprintf(stderr, "Required options:\n");
    fprintf(stderr, "  -m, --model FILE      Path to the language model (for vocabulary)\n");
    fprintf(stderr, "  --mmproj FILE         Path to the vision encoder model\n");
    fprintf(stderr, "  --image FILE          Path to the image file to process\n");
    fprintf(stderr, "\nOptional:\n");
    fprintf(stderr, "  -o, --output FILE     Output file path (default: image.embd)\n");
    fprintf(stderr, "  --image-min-tokens N  Minimum number of image tokens (default: from model metadata)\n");
    fprintf(stderr, "  --image-max-tokens N  Maximum number of image tokens (default: from model metadata)\n");
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, print_usage)) {
        return 1;
    }

    if (params.model.path.empty()) {
        fprintf(stderr, "error: --model is required (needed for vocabulary)\n");
        print_usage(argc, argv);
        return 1;
    }

    if (params.mmproj.path.empty()) {
        fprintf(stderr, "error: --mmproj is required\n");
        print_usage(argc, argv);
        return 1;
    }

    if (params.image.empty()) {
        fprintf(stderr, "error: --image is required\n");
        print_usage(argc, argv);
        return 1;
    }

    // Determine output file path
    std::string out_file = params.out_file.empty() ? "image.embd" : params.out_file;

    llama_backend_init();

    // Load the language model (needed for vocabulary and embedding dimensions)
    fprintf(stderr, "Loading language model for vocabulary...\n");
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    if (model == nullptr) {
        fprintf(stderr, "error: failed to load model '%s'\n", params.model.path.c_str());
        llama_backend_free();
        return 1;
    }

    // Initialize the vision encoder context
    fprintf(stderr, "Loading vision encoder...\n");
    mtmd_context_params ctx_params = mtmd_context_params_default();
    ctx_params.use_gpu = params.mmproj_use_gpu;
    ctx_params.n_threads = params.cpuparams.n_threads;
    ctx_params.image_min_tokens = params.image_min_tokens;
    ctx_params.image_max_tokens = params.image_max_tokens;

    mtmd::context_ptr mctx(mtmd_init_from_file(params.mmproj.path.c_str(), model, ctx_params));
    if (!mctx) {
        fprintf(stderr, "error: failed to init vision encoder from '%s'\n", params.mmproj.path.c_str());
        llama_backend_free();
        return 1;
    }

    if (!mtmd_support_vision(mctx.get())) {
        fprintf(stderr, "error: the mmproj model does not support vision input\n");
        llama_backend_free();
        return 1;
    }

    // Load and process each image
    mtmd::bitmaps bitmaps;
    for (const auto & img_path : params.image) {
        fprintf(stderr, "Loading image: %s\n", img_path.c_str());
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(mctx.get(), img_path.c_str()));
        if (!bmp.ptr) {
            fprintf(stderr, "error: failed to load image '%s'\n", img_path.c_str());
            llama_backend_free();
            return 1;
        }
        bitmaps.entries.push_back(std::move(bmp));
    }

    // Build input text with media markers
    // The marker tells mtmd_tokenize where to insert the image
    std::string input_str;
    for (size_t i = 0; i < bitmaps.entries.size(); i++) {
        input_str += mtmd_default_marker();
    }

    mtmd_input_text input_text;
    input_text.text = input_str.c_str();
    input_text.add_special = false;
    input_text.parse_special = true;  // Required to parse the <__media__> marker

    // Tokenize the input (this processes the image through the vision encoder preprocessing)
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();

    fprintf(stderr, "Tokenizing input...\n");
    int32_t ret = mtmd_tokenize(mctx.get(), chunks.ptr.get(), &input_text,
                                 bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    if (ret != 0) {
        fprintf(stderr, "error: mtmd_tokenize failed (code %d)\n", ret);
        if (ret == 1) {
            fprintf(stderr, "  hint: number of images doesn't match number of markers\n");
        } else if (ret == 2) {
            fprintf(stderr, "  hint: image preprocessing failed\n");
        }
        llama_backend_free();
        return 1;
    }

    // Process chunks and extract image embeddings
    std::vector<float> final_embeddings;
    int32_t nx = 0, ny = 0, n_pos = 0;
    int n_embd = llama_model_n_embd(model);

    size_t n_chunks = chunks.size();
    fprintf(stderr, "Processing %zu chunk(s)...\n", n_chunks);

    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = chunks[i];
        enum mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);

        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            fprintf(stderr, "Encoding image chunk %zu...\n", i);

            // Run the vision encoder
            if (mtmd_encode_chunk(mctx.get(), chunk) != 0) {
                fprintf(stderr, "error: mtmd_encode_chunk failed for chunk %zu\n", i);
                llama_backend_free();
                return 1;
            }

            // Get the output embeddings
            float * embd = mtmd_get_output_embd(mctx.get());
            size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);

            // Get M-RoPE grid dimensions
            const mtmd_image_tokens * img_toks = mtmd_input_chunk_get_tokens_image(chunk);
            if (img_toks) {
                nx = (int32_t)mtmd_image_tokens_get_nx(img_toks);
                ny = (int32_t)mtmd_image_tokens_get_ny(img_toks);
                n_pos = (int32_t)mtmd_input_chunk_get_n_pos(chunk);
            }

            if (!final_embeddings.empty()) {
                fprintf(stderr, "warning: multiple image chunks - only the last one's grid dimensions will be saved\n");
            }

            final_embeddings.insert(final_embeddings.end(), embd, embd + (n_tokens * n_embd));
            fprintf(stderr, "  encoded %zu tokens (nx=%d, ny=%d, n_pos=%d)\n", n_tokens, nx, ny, n_pos);
        } else if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            // Text chunks around the image are not needed for embedding-only output
            fprintf(stderr, "Skipping text chunk %zu\n", i);
        }
    }

    if (final_embeddings.empty()) {
        fprintf(stderr, "error: no image embeddings were generated\n");
        llama_backend_free();
        return 1;
    }

    // Write the embeddings to file
    fprintf(stderr, "Writing embeddings to '%s'...\n", out_file.c_str());

    std::ofstream out(out_file, std::ios::binary);
    if (!out) {
        fprintf(stderr, "error: failed to open output file '%s'\n", out_file.c_str());
        llama_backend_free();
        return 1;
    }

    img_embd_file_header header = {};
    memcpy(header.magic, "IMGE", 4);
    header.version = 1;
    header.n_tokens = (int32_t)(final_embeddings.size() / n_embd);
    header.n_embd = n_embd;
    header.nx = nx;
    header.ny = ny;
    header.n_pos = n_pos;
    header.reserved = 0;

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(final_embeddings.data()),
              final_embeddings.size() * sizeof(float));

    if (!out.good()) {
        fprintf(stderr, "error: failed to write embeddings to file\n");
        llama_backend_free();
        return 1;
    }

    out.close();

    fprintf(stderr, "\nSuccess! Saved %d tokens (dim=%d, grid=%dx%d, n_pos=%d) to %s\n",
            header.n_tokens, header.n_embd, header.nx, header.ny, header.n_pos, out_file.c_str());
    fprintf(stderr, "File size: %.2f MB\n",
            (sizeof(header) + final_embeddings.size() * sizeof(float)) / (1024.0 * 1024.0));

    llama_backend_free();
    return 0;
}
