#include "role/Worker.h"
#include "executor/PiplineExecutor.h"
#include "channel/ChannelManager.h"
#include "utils/logger.h"
#include "Sequence.h"
#include <algorithm>
#include <numeric>

void Worker::set_channels() {
    ChannelManager* manager = ChannelManager::get_instance();
    auto get_or_null = [manager](const std::string& name) -> Channel* {
        Channel* channel = nullptr;
        ErrorCode err = manager->get_channel(name, channel);
        if (err != ErrorCode::SUCCESS) {
            return nullptr;
        }
        return channel;
    };

    const int rank = engine_config.pipeline_rank;
    from_scheduler = get_or_null("scheduler_to_worker_" + std::to_string(rank));
    to_scheduler = get_or_null("worker_" + std::to_string(rank) + "_to_scheduler");

    from_prev_worker = nullptr;
    to_next_worker = nullptr;
    if (rank > 0) {
        from_prev_worker = get_or_null("worker_" + std::to_string(rank - 1) + "_to_worker_" + std::to_string(rank));
    }
    if (rank + 1 < engine_config.world_size) {
        to_next_worker = get_or_null("worker_" + std::to_string(rank) + "_to_worker_" + std::to_string(rank + 1));
    }
}

void Worker::run(){
    LOG_INFO("Worker started running.");
    work();
    LOG_INFO("Worker stopped running.");
}

void Worker::work() {
    while (true) {
        ForwardMessage message;
        ErrorCode recv_error = receive(message);
        if (recv_error != ErrorCode::SUCCESS) {
            LOG_ERROR("Worker failed to receive forward message.");
            continue;
        }

        if (message.op_type == ForwardOp::FREE_SEQ) {
            freeFinishedSequencesOnWorkers(message.batch.sequence_ids);
            continue;
        }
           
        

        Batch batch = message.batch;
        batch.num_tokens = batch.num_tokens > 0 ? batch.num_tokens : batch.token_ids.size();
        if (batch.sequence_ids.size() < batch.num_tokens || batch.token_positions.size() < batch.num_tokens) {
            LOG_ERROR("Worker received malformed batch: sequence_ids/token_positions size mismatch");
            continue;
        }

        const bool is_prefill = (message.op_type == ForwardOp::PREFILL);
        const bool is_decode = (message.op_type == ForwardOp::DECODE);
        if (!is_prefill && !is_decode) {
            LOG_ERROR("Worker received unknown op_type");
            continue;
        }

        // Allocate KV blocks based on local token positions
        std::unordered_map<size_t, size_t> seq_required_blocks;
        for (size_t i = 0; i < batch.num_tokens; ++i) {
            const size_t seq_id = batch.sequence_ids[i];
            const size_t pos = batch.token_positions[i];
            const size_t required = (pos / engine_config.block_size) + 1;
            auto it = seq_required_blocks.find(seq_id);
            if (it == seq_required_blocks.end()) {
                seq_required_blocks[seq_id] = required;
            } else {
                it->second = std::max(it->second, required);
            }
        }

        bool alloc_failed = false;
        for (const auto& [seq_id, required] : seq_required_blocks) {
            auto seq = seq_pool->get(seq_id);
            if (!seq) {
                seq = seq_pool->create(seq_id);
            }

            auto& blocks = seq->blocks;
            while (blocks.size() < required) {
                auto result = cache_manager->allocate_cache_block();
                if (std::holds_alternative<std::shared_ptr<CacheBlock>>(result)) {
                    blocks.push_back(std::get<std::shared_ptr<CacheBlock>>(result));
                } else {
                    LOG_ERROR("Worker failed to allocate KV block for sequence " + std::to_string(seq_id));
                    alloc_failed = true;
                    break;
                }
            }
            if (alloc_failed) {
                break;
            }
        }
        if (alloc_failed) {
            continue;
        }

        PiplineExecutor* pipeline_executor = dynamic_cast<PiplineExecutor*>(model_executor.get());
        auto run_forward = [&](void* external_hidden) -> bool {
            if (is_prefill) {
                if (external_hidden != nullptr) {
                    if (pipeline_executor == nullptr) {
                        LOG_ERROR("External hidden input requires PiplineExecutor in prefill path.");
                        return false;
                    }
                    pipeline_executor->run_prefill(batch, external_hidden);
                } else {
                    model_executor->run_prefill(batch);
                }
            } else {
                if (external_hidden != nullptr) {
                    if (pipeline_executor == nullptr) {
                        LOG_ERROR("External hidden input requires PiplineExecutor in decode path.");
                        return false;
                    }
                    pipeline_executor->run_decode(batch, external_hidden);
                } else {
                    model_executor->run_decode(batch);
                }
            }
            return true;
        };

        if (message.has_cuda_ipc_handle) {
            cudaIpcMemHandle_t handle = message.cuda_ipc_handle();
            void* external_hidden = nullptr;
            cudaError_t err = cudaIpcOpenMemHandle(&external_hidden, handle, cudaIpcMemLazyEnablePeerAccess);
            if (err != cudaSuccess) {
                LOG_ERROR("Worker failed to open CUDA IPC handle: " + std::string(cudaGetErrorString(err)));
                continue;
            }

            const bool ok = run_forward(external_hidden);
            cudaError_t close_err = cudaIpcCloseMemHandle(external_hidden);
            if (close_err != cudaSuccess) {
                LOG_ERROR("Worker failed to close CUDA IPC handle: " + std::string(cudaGetErrorString(close_err)));
            }
            if (!ok) {
                continue;
            }
        } else if (!run_forward(nullptr)) {
            continue;
        }

        //build message for scheduler
        ForwardMessage response;
        if(engine_config.is_last_stage()){
            response.op_type = ForwardOp::DECODE;
            response.batch = batch;
            // for the last stage worker, already append sampled token ids in model execution
            response.batch.token_ids = batch.token_ids;
            response.batch.sampled_token_ids = batch.sampled_token_ids;
            response.batch.num_tokens = response.batch.token_ids.size();
        } else { //build message for next stage worker
            response.op_type = ForwardOp::PREFILL;
            response.batch = batch;
        }
        ErrorCode send_error = send(response);
        if (send_error != ErrorCode::SUCCESS) {
            LOG_ERROR("Worker failed to send forward response.");
        }
    }
}

ErrorCode Worker::receive(ForwardMessage& message) {
    Channel* input = engine_config.is_first_stage() ? from_scheduler : from_prev_worker;
    if (input == nullptr) {
        LOG_ERROR("Worker input channel is null.");
        return ErrorCode::UNKNOWN_ERROR;
    }

    input->receive(message);
    return ErrorCode::SUCCESS;
}

ErrorCode Worker::send(const ForwardMessage& message) {
    Channel* output = engine_config.is_last_stage() ? to_scheduler : to_next_worker;
    if (output == nullptr) {
        LOG_ERROR("Worker output channel is null.");
        return ErrorCode::UNKNOWN_ERROR;
    }

    output->send(message);
    return ErrorCode::SUCCESS;
}


void Worker::freeFinishedSequencesOnWorkers(const std::vector<size_t>& sequence_ids) {
    for (size_t seq_id : sequence_ids) {
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }
        for (const auto& block : seq->blocks) {
            if (block) {
                cache_manager->free_cache_block(block->block_id);
            }
        }
        seq->blocks.clear();
        seq_pool->erase(seq_id);
    }

    ForwardMessage cleanup_response;
    cleanup_response.batch.sequence_ids = sequence_ids;
    if (engine_config.is_last_stage()) {
        cleanup_response.op_type = ForwardOp::DONE;
    } else {
        cleanup_response.op_type = ForwardOp::FREE_SEQ;
    }
    ErrorCode send_error = send(cleanup_response);
    if (send_error != ErrorCode::SUCCESS) {
        LOG_ERROR("Worker failed to forward FREE_SEQ cleanup response.");
    }    
}