#pragma once
#include <vector>
#include <deque>
#include <unordered_map>
#include <memory>
#include "Sequence.h"
#include "Role.h"
#include "llm_engine_config.h"
#include "error.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include "channel/Channel.h"

struct RouteMeta {
    RouteType route_type;
};

class Router: public Role {
    public:
        Router(LLMEngineConfig engine_config): engine_config(engine_config) {};

        void run() override;
        void route();
        void set_channels();
        
        ErrorCode add_sequence(
            size_t seq_id, 
            std::vector<size_t> token_ids, 
            const SequenceConfig& sequence_config = SequenceConfig()
        );
        ErrorCode getSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq);
        ErrorCode removeFinishedSequenceById(size_t seq_id);
        ErrorCode wait_until_finished(size_t seq_id);

    private:
        //functions extracted for better readability
        void add_to_prefiller(size_t seq_id);
        void add_to_decoder(size_t seq_id);
        void from_prefiller_handler();
        void from_decoder_handler();
        ErrorCode send_free_seq_to_schedulers(size_t seq_id);

        LLMEngineConfig engine_config;

        //queues and state tracking for routing logic
        std::deque<size_t> prefill_ready_queue;
        std::deque<size_t> decode_ready_queue;
        std::unordered_map<size_t, RouteMeta> prefill_inflight;
        std::unordered_map<size_t, RouteMeta> decode_inflight;
        std::unordered_map<size_t, RouteType> route_states;
        std::unordered_map<size_t, std::shared_ptr<Sequence>> sequence_store;
        // protect the route states, queues and sequence store
        std::mutex queue_mutex;
        std::condition_variable route_cv;

        std::atomic<bool> stop_requested{false};

        //communication channels with schedulers
        Channel* to_prefiller_channel = nullptr;
        Channel* to_decoder_channel = nullptr;
        Channel* from_prefiller_channel = nullptr;
        Channel* from_decoder_channel = nullptr;

};