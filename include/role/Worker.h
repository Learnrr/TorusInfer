#pragma once
#include "llm_engine_config.h"
#include "role/Role.h"
#include "executor/Executor.h"
#include "model/IModel.h"
#include "Workspace.h"
#include "KVCacheManager.h"
#include "error.h"
#include "channel/Channel.h"
#include "channel/ChannelMessage.h"
#include "SequencePool.h"
class Worker: public Role {
    public:
        Worker(
            KVCacheManager* cache_manager,
            IModel* model,
            Workspace* workspace,
            const LLMEngineConfig& engine_config
        ): cache_manager(cache_manager),
            seq_pool(std::make_unique<SequencePool>()),
            model_executor(std::make_unique<SingleCardExecutor>(model, workspace, seq_pool.get())),
            engine_config(engine_config),
            eos_token_id(engine_config.model_config.eos_token_id) {}


        void run() override;
        void work();
        ErrorCode receive(ForwardMessage& message);
        ErrorCode send(const ForwardMessage& message);

        void set_channels();

    private:
        KVCacheManager* cache_manager;
        std::unique_ptr<Workspace> workspace;
        std::unique_ptr<SequencePool> seq_pool;
        std::unique_ptr<Executor> model_executor;
        LLMEngineConfig engine_config;
        size_t eos_token_id;

        Channel* from_scheduler = nullptr;
        Channel* to_scheduler = nullptr;
        Channel* from_prev_worker = nullptr;
        Channel* to_next_worker = nullptr;

        void freeFinishedSequencesOnWorkers(const std::vector<size_t>& sequence_ids);

};