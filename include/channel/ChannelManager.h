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
            const std::string& role, 
            int world_size, 
            int pipeline_rank){
            std::lock_guard<std::mutex> lock(mutex_);
            channels_.clear();
            channel_names_.clear();

            auto add_channel = [this](const std::string& channel_name) {
                channels_[channel_name] = create_ipc_channel(channel_name);
                channel_names_.push_back(channel_name);
            };

            if (role == "scheduler") {
                for (int i = 0; i < world_size; ++i) {
                    add_channel("scheduler_to_worker_" + std::to_string(i));
                    add_channel("worker_" + std::to_string(i) + "_to_scheduler");
                }
                return ErrorCode::SUCCESS;
            }

            if (role == "worker") {
                const int r = pipeline_rank;
                add_channel("scheduler_to_worker_" + std::to_string(r));
                add_channel("worker_" + std::to_string(r) + "_to_scheduler");

                if (r > 0) {
                    add_channel("worker_" + std::to_string(r - 1) + "_to_worker_" + std::to_string(r));
                }
                if (r + 1 < world_size) {
                    add_channel("worker_" + std::to_string(r) + "_to_worker_" + std::to_string(r + 1));
                }
                return ErrorCode::SUCCESS;
            }

            LOG_ERROR("Invalid role specified in engine config: " + role);
            return ErrorCode::FAILED_TO_OPEN_CONFIG_FILE;
        }

        ErrorCode get_channel(const std::string& channel_name, Channel*& channel) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = channels_.find(channel_name);
            if(it != channels_.end()){
                channel = it->second.get();
                return ErrorCode::SUCCESS;
            } else {
                LOG_ERROR("Channel not found: " + channel_name);
                return ErrorCode::UNKNOWN_ERROR;
            }
        }

        std::unique_ptr<Channel> create_ipc_channel(const std::string& channel_name) {
            return std::make_unique<IpcChannel>(channel_name);
        }

        void clear() {
            std::lock_guard<std::mutex> lock(mutex_);
            channels_.clear();
            channel_names_.clear();
        }
    private:
        ChannelManager() = default;
        std::unordered_map<std::string, std::unique_ptr<Channel>> channels_;
        std::vector<std::string> channel_names_;
        std::mutex mutex_;
};