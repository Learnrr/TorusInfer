#pragma once
#include "define.h"
#include "cuda_runtime.h"
#include "error.h"
#include "llm_engine_config.h"

struct AttentionLayerWorkspace {
    // after qkv projection
    size_t qkv_offset;
    //after softmax and PV
    size_t context_offset;
    //after attention output projection
    size_t attn_out_offset;
    

};

struct MLPWorkspace {
    //between two linear layers
    size_t mlp_offset;
    //after mlp output projection
    size_t mlp_out_offset;
};

struct TransformerLayerWorkspace {
    // rmsnorm before qkv projection
    size_t attn_norm_offset;
    //attention
    AttentionLayerWorkspace attention_workspace;
    //rmsnorm before mlp
    size_t mlp_norm_offset;
    //mlp
    MLPWorkspace mlp_workspace;

};

struct WorkspaceLayout {

    //for each transfermer layer output -- ping pong buffer
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

        ErrorCode init(const LLMEngineConfig& engine_config);


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
            return (void*)((char*)workspace + layout.layer_workspace.attention_workspace.qkv_offset);
        }
        void* get_hidden_workspace() {
            return (void*)((char*)workspace + layout.hidden_offset);
        }
        void* get_hidden2_workspace() {
            return (void*)((char*)workspace + layout.hidden2_offset);
        }
        void* get_attn_norm_workspace() {
            return (void*)((char*)workspace + layout.layer_workspace.attn_norm_offset);
        }        
        void* get_attn_context_workspace() {
            return (void*)((char*)workspace + layout.layer_workspace.attention_workspace.context_offset);
        }
        void* get_attn_output_workspace() {
            return (void*)((char*)workspace + layout.layer_workspace.attention_workspace.attn_out_offset);
        }    
        void* get_mlp_norm_workspace() {
            return (void*)((char*)workspace + layout.layer_workspace.mlp_norm_offset);
        }    
        void* get_mlp_workspace() {
            return (void*)((char*)workspace + layout.layer_workspace.mlp_workspace.mlp_offset);
        }      
        void* get_mlp_out_workspace() {
            return (void*)((char*)workspace + layout.layer_workspace.mlp_workspace.mlp_out_offset);
        }          
        void* get_logits_workspace() {
            return (void*)((char*)workspace + layout.logits_offset);
        }
        void* get_temp_workspace() {
            return (void*)((char*)workspace + layout.temp_offset);
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
        }


};