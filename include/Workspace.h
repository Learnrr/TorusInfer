#pragma once
#include "define.h"
#include "cuda_runtime.h"
#include "error.h"

struct TransformerLayerWorkspace {
    void* base;
    size_t qkv_offset;
    size_t attn_norm_offset;
    size_t attn_out_offset;
    size_t context_offset;
    size_t mlp_offset;

};

struct WorkspaceLayout {

    size_t hidden_offset;
    size_t hidden2_offset;

    TransformerLayerWorkspace layer_workspace;
    size_t temp_offset;

    size_t logits_offset;

    size_t total_size;
};
inline size_t align_size(size_t size, size_t alignment = 256) {
    return (size + alignment - 1) & ~(alignment - 1);
}
class Workspace {
    public:
        Workspace() : workspace(nullptr) {}
        Workspace(const Workspace&) = delete;
        Workspace& operator=(const Workspace&) = delete;

        ErrorCode init(){
            // Re-init should not leak the previous device buffer.
            free();

            size_t hidden_size = MAX_SEQ_LEN * HIDDEN_SIZE * DTYPE;
            size_t qkv_size = MAX_SEQ_LEN * 3 * HIDDEN_SIZE * DTYPE;
            size_t attn_out_size = MAX_SEQ_LEN * HIDDEN_SIZE * DTYPE;
            size_t context_size = MAX_SEQ_LEN * HIDDEN_SIZE * DTYPE;
            size_t mlp_size = MAX_SEQ_LEN * INTERMEDIATE_SIZE * DTYPE;
            size_t logits_size = MAX_SEQ_LEN * VOCAB_SIZE * DTYPE;

            size_t offset = 0;
            layout.hidden_offset = offset;
            offset += align_size(hidden_size, 256);
            layout.hidden2_offset = offset;
            offset += align_size(hidden_size, 256);
            layout.layer_workspace.qkv_offset = offset;
            offset += align_size(qkv_size, 256);
            layout.layer_workspace.attn_out_offset = offset;
            offset += align_size(attn_out_size, 256);
            layout.layer_workspace.attn_norm_offset = offset;
            offset += align_size(attn_out_size, 256);
            layout.layer_workspace.context_offset = offset;
            offset += align_size(context_size, 256);
            layout.layer_workspace.mlp_offset = offset;
            offset += align_size(mlp_size, 256);
            layout.temp_offset = offset;
            offset += align_size(mlp_size, 256);
            layout.logits_offset = offset;
            offset += align_size(logits_size, 256);
            layout.total_size = offset;

            cudaError_t cuda_error = cudaMalloc(&workspace, layout.total_size);
            if (cuda_error != cudaSuccess) {
                workspace = nullptr;
                layout.layer_workspace.base = nullptr;
                return ErrorCode::CUDA_FAILURE;
            }

            layout.layer_workspace.base = workspace;
            return ErrorCode::SUCCESS;
        }



        ~Workspace(){
            free();
        }

        void* get_workspace() {
            return workspace;
        }
        void* get_embedding_workspace() {
            return (void*)((char*)workspace + layout.hidden_offset);
        }
        void* get_qkv_workspace() {
            return (void*)((char*)workspace + layout.layer_workspace.qkv_offset);
        }
        void* get_hidden2_workspace() {
            return (void*)((char*)workspace + layout.hidden2_offset);
        }
        void* get_logits_workspace() {
            return (void*)((char*)workspace + layout.logits_offset);
        }
        void* get_mlp_workspace() {
            return (void*)((char*)workspace + layout.layer_workspace.mlp_offset);
        }
        void* get_temp_workspace() {
            return (void*)((char*)workspace + layout.temp_offset);
        }
        void* get_attn_norm_workspace() {
            return (void*)((char*)workspace + layout.layer_workspace.attn_norm_offset);
        }
        void* get_attn_context_workspace() {
            return (void*)((char*)workspace + layout.layer_workspace.context_offset);
        }
        void* get_attn_output_workspace() {
            return (void*)((char*)workspace + layout.layer_workspace.attn_out_offset);
        }
        WorkspaceLayout get_layout() {
            return layout;
        }

    private:
        void* workspace;
        WorkspaceLayout layout;

        void free() {
            if (workspace != nullptr) {
                cudaFree(workspace);
                workspace = nullptr;
            }
            layout.layer_workspace.base = nullptr;
        }


};