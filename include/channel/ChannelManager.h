#pragma once
#include "Channel.h"
#include "error.h"
#include "utils/logger.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "channel/IpcChannel.h"
#include "llm_engine_config.h"
class ChannelManager {
    public:
        static ChannelManager* get_instance() {
            static ChannelManager instance;
            return &instance;
        }


        ErrorCode build_channels(
        /*
            scheduler_to_worker_r：scheduler create，worker_r open
            worker_r_to_scheduler：worker_r create，scheduler open
            worker_i_to_worker_i+1：smaller rank (i) create，larger rank (i+1) open
        */
            const LLMEngineConfig& engine_config);

        ErrorCode get_channel(const std::string& channel_name, Channel*& channel);

        std::unique_ptr<Channel> create_ipc_channel(const std::string& channel_name);

        void clear();
    private:
        ChannelManager() = default;
        LLMEngineConfig engine_config_;
        std::unordered_map<std::string, std::unique_ptr<Channel>> channels_;
        std::vector<std::string> channel_names_;
        std::mutex mutex_;
};