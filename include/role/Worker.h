#pragma once
#include "llm_engine_config.h"
#include "role/Role.h"
#include "executor/Executor.h"
#include "executor/SingleCardExecutor.h"
#include "executor/PipelineExecutor.h"
#include "model/IModel.h"
#include "Workspace.h"
#include "KVCacheManager.h"
#include "error.h"
#include "channel/Channel.h"
#include "channel/ChannelMessage.h"
#include "SequencePool.h"
#include "PrefixCacheManager.h"
#include <unordered_map>
#include <atomic>
class Worker: public Role {
    public:
        Worker(
            KVCacheManager* cache_manager,
            IModel* model,
            Workspace* workspace,
            const LLMEngineConfig& engine_config
        ){
                this->engine_config = engine_config;
                this->seq_pool = std::make_unique<SequencePool>();
                //prefix caching
                if(engine_config.enable_prefix_cache){
                    prefix_cache_manager = std::make_unique<PrefixCacheManager>(engine_config);
                } else {
                    prefix_cache_manager = nullptr;
                }
                      
                if(engine_config.enable_pipeline_parallel){
                    model_executor = std::make_unique<PipelineExecutor>(
                        model, 
                        workspace, 
                        engine_config.stage_start_layer, 
                        engine_config.stage_end_layer,
                        seq_pool.get(),
                        cache_manager,
                        prefix_cache_manager.get(),
                        &retained_outgoing_events
                    );
                } else {
                    model_executor = std::make_unique<SingleCardExecutor>(
                        model, 
                        workspace,
                        seq_pool.get(),
                        cache_manager,
                        prefix_cache_manager.get(),
                        &retained_outgoing_events
                    );
                }
                this->cache_manager = cache_manager;
                this->workspace = workspace;

                // Allocate tmp buffer for one decode batch worth of KV payload:
                // all seqs in batch * max sequence length (rounded to blocks).
                const size_t max_decode_batch =
                    engine_config.max_decode_batch_size > 0 ? engine_config.max_decode_batch_size : 1;
                const size_t max_blocks_per_seq =
                    (engine_config.max_sequence_length + engine_config.block_size - 1) /
                    engine_config.block_size;
                const size_t tmp_kv_capacity_bytes =
                    max_decode_batch *
                    max_blocks_per_seq *
                    engine_config.block_size *
                    engine_config.model_config.num_hidden_layers *
                    engine_config.model_config.head_dim *
                    engine_config.model_config.num_kv_heads *
                    DataTypeBytes(engine_config.model_config.data_type);
                cudaMalloc(
                    &tmpKeyCache,
                    tmp_kv_capacity_bytes
                );
                cudaMalloc(
                    &tmpValueCache,
                    tmp_kv_capacity_bytes
                );

            } 


        void run() override;
        void work();
        void set_channels();

    private:
        KVCacheManager* cache_manager;
        std::unique_ptr<SequencePool> seq_pool;
        std::unique_ptr<Executor> model_executor;
        LLMEngineConfig engine_config;
        Workspace* workspace;
        std::unique_ptr<PrefixCacheManager> prefix_cache_manager;

        // Communication channels
        Channel* from_scheduler = nullptr;
        Channel* to_scheduler = nullptr;
        Channel* from_prev_worker = nullptr;
        Channel* to_next_worker = nullptr;
        Channel* from_peer_transfer = nullptr;
        Channel* to_peer_transfer = nullptr;

        std::unordered_map<size_t, cudaEvent_t> retained_outgoing_events;
        std::atomic<bool> stop_requested{false};

        void cleanup_retained_events();
        void setdevice();
        ErrorCode receive(ForwardMessage& message);
        ErrorCode send(const ForwardMessage& message);
        ErrorCode allocate_blocks(ForwardMessage& message);
        ErrorCode handle_remote_forward(ForwardMessage& message, void** external_hidden_out);
        ErrorCode handle_local_forward(ForwardMessage& message);
        ErrorCode build_response_and_send(ForwardMessage& message, void* external_hidden_out, size_t produced_hidden_tokens);
        ErrorCode bind_cacheblocks_for_batch(const Batch& batch);
        ErrorCode trim_prefill_batch_after_prefix_bind(Batch& batch);
        
        bool is_pd_prefiller_worker() const;
        bool is_pd_decoder_worker() const;

        // Ensure required KV blocks for decode are ready before running decode forward, 
        //by checking local state for decoder worker and sending transfer request to peer prefiller worker if needed.
        ErrorCode ensure_decode_kv_ready(const ForwardMessage& message);
        std::unordered_map<size_t, size_t> build_required_blocks_map(const Batch& batch, bool is_prefill, bool is_decode) const;
        ErrorCode receive_pull_req();
        ErrorCode handle_kv_pull_req(const TransferMessage& message);

        void pull_required_kv_blocks_from_peer(const std::unordered_map<size_t, size_t>& seq_required_blocks);
        
        void* tmpKeyCache = nullptr;
        void* tmpValueCache = nullptr;
};