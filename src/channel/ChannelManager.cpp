#include "channel/ChannelManager.h"

ErrorCode ChannelManager::build_channels(
/*
    scheduler_to_worker_r：scheduler create，worker_r open
    worker_r_to_scheduler：worker_r create，scheduler open
    worker_i_to_worker_i+1：smaller rank (i) create，larger rank (i+1) open
*/
    const LLMEngineConfig& engine_config){
    std::lock_guard<std::mutex> lock(mutex_);
    engine_config_ = engine_config;

    const std::string& role = engine_config_.role;
    const int world_size = engine_config_.world_size;
    const int pipeline_rank = engine_config_.pipeline_rank;
    const bool enable_pd_disaggregation = engine_config_.enable_pd_disaggregation;

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

            if (enable_pd_disaggregation) {
                // PD scheduler-worker channels with explicit role prefixes.
                add_channel("prefill_scheduler_to_worker_" + std::to_string(i));
                add_channel("prefill_worker_" + std::to_string(i) + "_to_scheduler");
                add_channel("decode_scheduler_to_worker_" + std::to_string(i));
                add_channel("decode_worker_" + std::to_string(i) + "_to_scheduler");
            }
        }
        if (enable_pd_disaggregation) {
            // PD-disaggregation router channels.
            add_channel("router_to_prefill_scheduler");
            add_channel("prefill_scheduler_to_router");
            add_channel("router_to_decode_scheduler");
            add_channel("decode_scheduler_to_router");
        }
        return ErrorCode::SUCCESS;
    }

    if (role == "router") {
        if (enable_pd_disaggregation) {
            add_channel("router_to_prefill_scheduler");
            add_channel("prefill_scheduler_to_router");
            add_channel("router_to_decode_scheduler");
            add_channel("decode_scheduler_to_router");
        }
        return ErrorCode::SUCCESS;
    }

    if (role == "worker") {
        const int r = pipeline_rank;

        // only first stage receives requests directly from scheduler
        if (r == 0) {
            add_channel("scheduler_to_worker_" + std::to_string(r));
        }

        // inbound from previous stage (if any)
        if (r > 0) {
            add_channel("worker_" + std::to_string(r - 1) + "_to_worker_" + std::to_string(r));
        }

        // only last stage sends final responses back to scheduler
        if (r == world_size - 1) {
            add_channel("worker_" + std::to_string(r) + "_to_scheduler");
        }

        // outbound to next stage only
        if (r + 1 < world_size) {
            add_channel("worker_" + std::to_string(r) + "_to_worker_" + std::to_string(r + 1));
        }

        if (enable_pd_disaggregation) {
            const std::string& pd_role = engine_config_.pd_role;
            auto add_pd_pipeline_channels = [&](const std::string& role_prefix) {
                if (r == 0) {
                    add_channel(role_prefix + "_scheduler_to_worker_" + std::to_string(r));
                }
                if (r > 0) {
                    add_channel(
                        role_prefix + "_worker_" + std::to_string(r - 1) +
                        "_to_" + role_prefix + "_worker_" + std::to_string(r)
                    );
                }
                if (r == world_size - 1) {
                    add_channel(role_prefix + "_worker_" + std::to_string(r) + "_to_scheduler");
                }
                if (r + 1 < world_size) {
                    add_channel(
                        role_prefix + "_worker_" + std::to_string(r) +
                        "_to_" + role_prefix + "_worker_" + std::to_string(r + 1)
                    );
                }
            };

            if (pd_role == "prefiller") {
                add_pd_pipeline_channels("prefill");
            } else if (pd_role == "decoder") {
                add_pd_pipeline_channels("decode");
            } else {
                LOG_ERROR("Invalid pd_role specified in engine config: " + pd_role);
                return ErrorCode::INVALID_INPUT;
            }

            // PD transfer control channel pair (decoder <-> prefiller) for the same pipeline rank.
            add_channel("decoder_" + std::to_string(r) + "_to_prefiller_" + std::to_string(r) + "_transfer");
            add_channel("prefiller_" + std::to_string(r) + "_to_decoder_" + std::to_string(r) + "_transfer");
        }

        return ErrorCode::SUCCESS;
    }

    LOG_ERROR("Invalid role specified in engine config: " + role);
    return ErrorCode::FAILED_TO_OPEN_CONFIG_FILE;
}

ErrorCode ChannelManager::get_channel(const std::string& channel_name, Channel*& channel) {
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

std::unique_ptr<Channel> ChannelManager::create_ipc_channel(const std::string& channel_name) {
    return std::make_unique<IpcChannel>(channel_name);
}

void ChannelManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    channels_.clear();
    channel_names_.clear();
}