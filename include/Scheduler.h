#pragma once

#include"Sequence.h"
#include"define.h"
#include "KVCacheManager.h"
#include "model/IModel.h"
#include "Batch.h"
#include "Cacheblock.h"
#include <thread>
#include <vector>
#include <atomic>
#include "error.h"
#include <variant>
#include "llm_engine_config.h"
#include <mutex>
#include <condition_variable>

class Scheduler{
    public:
        Scheduler(
            KVCacheManager* cache_manager, 
            IModel* model, 
            Workspace* workspace, 
            const LLMEngineConfig& engine_config
        )
            : cache_manager(cache_manager),
              model(model),
              engine_config(engine_config),
              workspace(workspace),
              eos_token_id(engine_config.model_config.eos_token_id) {}


        void schedule();
        void request_stop();
    
        ErrorCode addSequence(size_t seq_id, std::vector<size_t> token_ids);

        ErrorCode getSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq);

        ErrorCode getFinishedSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq);

        ErrorCode returnSequenceOutput();

        ErrorCode removeFinishedSequenceById(size_t seq_id);

    private:
        std::vector<std::shared_ptr<Sequence>> prepared_queue;
        std::vector<std::shared_ptr<Sequence>> waiting_queue;
        std::vector<std::shared_ptr<Sequence>> decoding_queue;
        std::vector<std::shared_ptr<Sequence>> prefilling_queue;
        std::vector<std::shared_ptr<Sequence>> finished_queue;

        std::mutex queue_mutex;
        std::condition_variable queue_cv;

        KVCacheManager* cache_manager;
        IModel* model;
        LLMEngineConfig engine_config;
        Workspace* workspace;
        size_t eos_token_id;
        std::atomic<bool> stop_requested{false};

        ErrorCode movePrefilledToDecoding(const Batch& prefill_batch);
        ErrorCode moveDecodingToFinished(const Batch& decode_batch);
        std::variant<Batch, ErrorCode> buildDecodeBatch();
        std::variant<Batch, ErrorCode> buildPrefillBatch();
        ErrorCode launchSequence();
        ErrorCode handleFinishedSequence();
        void appendDecodedTokens(Batch& decode_batch);
        bool hasPendingWorkLocked() const;
};