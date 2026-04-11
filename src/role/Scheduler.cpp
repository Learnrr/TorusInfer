#include "role/Scheduler.h"
#include "executor/Executor.h"
#include "executor/CoordinatorExecutor.h"
#include "executor/PipelineCoordinatorExecutor.h"
#include "channel/ChannelManager.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "model/ModelForwardContext.h"

#include <unordered_set>

Scheduler::Scheduler(
    const LLMEngineConfig& engine_config
)
    : seq_pool(std::make_unique<SequencePool>()),
    // initialize coordinator executor based on whether pipeline parallel is enabled
      coordinator(
          engine_config.enable_pipeline_parallel
              ? std::unique_ptr<Executor>(std::make_unique<PipelineCoordinatorExecutor>())
              : std::unique_ptr<Executor>(std::make_unique<CoordinatorExecutor>())
      ),
      engine_config(engine_config),
      eos_token_id(engine_config.model_config.eos_token_id) {}

void Scheduler::set_channels() {
    ChannelManager* manager = ChannelManager::get_instance();
    auto get_or_null = [manager](const std::string& name) -> Channel* {
        Channel* channel = nullptr;
        ErrorCode err = manager->get_channel(name, channel);
        if (err != ErrorCode::SUCCESS) {
            return nullptr;
        }
        return channel;
    };

    const int last_rank = engine_config.world_size - 1;
    const bool pd_scheduler_mode =
        engine_config.enable_pd_disaggregation && engine_config.role == "scheduler";

    Channel* to_worker0 = nullptr;
    Channel* from_worker_last = nullptr;
    if (pd_scheduler_mode && engine_config.scheduler_mode == "prefiller") {
        to_worker0 = get_or_null("prefill_scheduler_to_worker_0");
        from_worker_last = get_or_null("prefill_worker_" + std::to_string(last_rank) + "_to_scheduler");
    } else if (pd_scheduler_mode && engine_config.scheduler_mode == "decoder") {
        to_worker0 = get_or_null("decode_scheduler_to_worker_0");
        from_worker_last = get_or_null("decode_worker_" + std::to_string(last_rank) + "_to_scheduler");
    } else {
        to_worker0 = get_or_null("scheduler_to_worker_0");
        from_worker_last = get_or_null("worker_" + std::to_string(last_rank) + "_to_scheduler");
    }

    coordinator->set_channels(to_worker0, from_worker_last);

    if (engine_config.enable_pd_disaggregation && engine_config.role == "scheduler") {
        if (engine_config.scheduler_mode == "prefiller") {
            from_router_channel = get_or_null("router_to_prefill_scheduler");
            to_router_channel = get_or_null("prefill_scheduler_to_router");
        } else if (engine_config.scheduler_mode == "decoder") {
            from_router_channel = get_or_null("router_to_decode_scheduler");
            to_router_channel = get_or_null("decode_scheduler_to_router");
        }
    }
}

void Scheduler::run() {
    LOG_INFO("Scheduler started running.");
    schedule();
    LOG_INFO("Scheduler stopped running.");
}

bool Scheduler::hasPendingWorkLocked() const {
    return !prepared_queue.empty() ||
           !waiting_queue.empty() ||
           !prefilling_queue.empty() ||
           !decoding_queue.empty() ||
           !decode_inflight_batches.empty() ||
           !prefill_inflight_batches.empty();
}

bool Scheduler::hasRunnableDecodeWork() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (size_t seq_id : decoding_queue) {
        if (sequences_in_decoding.find(seq_id) != sequences_in_decoding.end()) {
            continue;
        }
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }
        if (seq->state == SequenceState::DECODING) {
            return true;
        }
    }

    for (size_t seq_id : prefilling_queue) {
        if (sequences_in_decoding.find(seq_id) != sequences_in_decoding.end()) {
            continue;
        }
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }
        if (seq->state == SequenceState::PREFILLED) {
            return true;
        }
    }

    return false;
}

bool Scheduler::canRunDecode() const {
    const bool pd_scheduler_mode =
        engine_config.enable_pd_disaggregation && engine_config.role == "scheduler";
    if (!pd_scheduler_mode) {
        return true;
    }
    return engine_config.scheduler_mode == "decoder";
}

bool Scheduler::canRunPrefill() const {
    const bool pd_scheduler_mode =
        engine_config.enable_pd_disaggregation && engine_config.role == "scheduler";
    if (!pd_scheduler_mode) {
        return true;
    }
    return engine_config.scheduler_mode == "prefiller";
}

void Scheduler::phaseAReceiveRouterCommands() {
    if (!(engine_config.enable_pd_disaggregation && engine_config.role == "scheduler")) {
        return;
    }

    if (from_router_channel == nullptr) {
        return;
    }

    // Stage A remains non-blocking via try_receive; poll once per loop to avoid
    // starving router commands when local work stays busy.

    constexpr size_t kMaxRouterMessagesPerLoop = 16;

    if (canRunPrefill() && !canRunDecode()) {
        size_t received = 0;
        while (received < kMaxRouterMessagesPerLoop) {
            RouteMessage msg;
            if (!from_router_channel->try_receive(msg)) {
                break;
            }
            if (msg.route_type != RouteType::PREFILL) {
                LOG_ERROR("Prefiller scheduler received non-prefill RouteMessage.");
                continue;
            }
            addSequence(msg.seq_id, msg.token_ids, msg.sequence_config);
            ++received;
        }
        return;
    }
    if (canRunDecode() && !canRunPrefill()) {
        size_t received = 0;
        while (received < kMaxRouterMessagesPerLoop) {
            RouteMessage msg;
            if (!from_router_channel->try_receive(msg)) {
                break;
            }
            if (msg.route_type != RouteType::DECODE) {
                LOG_ERROR("Decoder scheduler received non-decode RouteMessage.");
                continue;
            }
            auto seq = seq_pool->create(msg.seq_id, msg.sequence_config);
            seq->token_ids = msg.token_ids;
            seq->seq_len = msg.token_ids.size();
            seq->state = SequenceState::DECODING;
            seq->seq_config = msg.sequence_config;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                decoding_queue.push_back(msg.seq_id);
            }
            ++received;
        }
        return;
    }
}

void Scheduler::drainCompletionRecords() {
    CompletionRecord record;
    while (coordinator->poll_completion(record)) {
        if (record.status == CompletionStatus::DONE) {
            Batch completed_batch;
            auto decode_it = decode_inflight_batches.find(record.batch_id);
            auto prefill_it = prefill_inflight_batches.find(record.batch_id);
            bool is_decode_completion = false;

            if (decode_it != decode_inflight_batches.end()) {
                completed_batch = decode_it->second.batch;
                completed_batch.sampled_token_ids = record.sampled_token_ids;
                is_decode_completion = true;
                decode_inflight_batches.erase(record.batch_id);
                for (size_t seq_id : record.sequence_ids) {
                    sequences_in_decoding.erase(seq_id);
                }
            } else if (prefill_it != prefill_inflight_batches.end()) {
                completed_batch = prefill_it->second.batch;
                prefill_inflight_batches.erase(record.batch_id);
            } else {
                LOG_ERROR("Received completion record for unknown batch id " + std::to_string(record.batch_id));
                continue;
            }

            if (is_decode_completion) {
                appendDecodedTokens(completed_batch);
                moveDecodingToFinished(completed_batch);
            } else {
                for (size_t seq_id : completed_batch.sequence_ids) {
                    auto seq = seq_pool->get(seq_id);
                    if (seq && seq->state == SequenceState::PREFILLING) {
                        seq->state = SequenceState::PREFILLED;
                        prefill_report_pending_queue.push_back(seq_id);
                    }
                }
            }
        } else {
            LOG_ERROR("Received failed completion record for batch " +
                std::to_string(record.batch_id) + " with status " +
                std::to_string(static_cast<int>(record.status)));
            auto decode_it = decode_inflight_batches.find(record.batch_id);
            auto prefill_it = prefill_inflight_batches.find(record.batch_id);
            if (decode_it != decode_inflight_batches.end()) {
                Batch failed_batch = decode_it->second.batch;
                recoverFromDecodeFailure(failed_batch);
            } else if (prefill_it != prefill_inflight_batches.end()) {
                Batch failed_batch = prefill_it->second.batch;
                recoverFromPrefillFailure(failed_batch);
            } else {
                LOG_ERROR("Received failure completion record for unknown batch id " + std::to_string(record.batch_id));
            }
        }
    }
}

void Scheduler::submitDecodePath() {
    size_t decode_flight_vacancy = engine_config.max_decode_batch_flight - decode_inflight_batches.size();
    bool has_decode_work = hasRunnableDecodeWork();

    if (decode_flight_vacancy > 0 && has_decode_work) {
        auto result = buildDecodeBatch();
        if (std::holds_alternative<ErrorCode>(result)) {
            LOG_ERROR("Failed to build decode batch.");
            return;
        }
        Batch decode_batch = std::get<Batch>(result);
        if (decode_batch.batch_size > 0 && decode_batch.num_tokens > 0) {
            ModelForwardContext decode_context;
            ErrorCode err_code = coordinator->run_decode(decode_batch, decode_context);

            if (err_code != ErrorCode::SUCCESS) {
                LOG_ERROR("Scheduler decode forward failed; skipping state transition for this batch.");
                recoverFromDecodeFailure(decode_batch);
                return;
            }

            InflightEntry entry;
            entry.batch = decode_batch;
            entry.op_type = InflightOp::DECODE;
            entry.sequence_ids = decode_batch.sequence_ids;
            decode_inflight_batches[decode_batch.batch_id] = std::move(entry);
            for (size_t seq_id : decode_batch.sequence_ids) {
                sequences_in_decoding.insert(seq_id);
            }

            LOG_DEBUG(
                "INFLIGHT SUBMIT DECODE: batch_id=" + std::to_string(decode_batch.batch_id) +
                ", decode_inflight=" + std::to_string(decode_inflight_batches.size()) +
                ", prefill_inflight=" + std::to_string(prefill_inflight_batches.size()) +
                ", total_inflight=" + std::to_string(decode_inflight_batches.size() + prefill_inflight_batches.size())
            );
        }
    }
}

void Scheduler::submitPrefillPath() {
    size_t prefill_flight_vacancy = engine_config.max_prefill_batch_flight - prefill_inflight_batches.size();
    bool has_waiting_work = false;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        has_waiting_work = !waiting_queue.empty();
    }

    if (prefill_flight_vacancy > 0 && has_waiting_work) {
        auto result = buildPrefillBatch();
        if (std::holds_alternative<ErrorCode>(result)) {
            LOG_ERROR("Failed to build prefill batch.");
            return;
        }
        Batch prefill_batch = std::get<Batch>(result);

        if (engine_config.enable_prefix_cache) {
            ErrorCode probe_error = coordinator->run_prefix_probe(prefill_batch);
            if (probe_error != ErrorCode::SUCCESS) {
                LOG_ERROR(
                    "Scheduler prefix probe failed for batch_id=" +
                    std::to_string(prefill_batch.batch_id) +
                    ", skip this batch and retry in next loop."
                );
                recoverFromPrefillFailure(prefill_batch);
                return;
            }
            applyPrefixProbeToPrefillBatch(prefill_batch);
        }

        if (prefill_batch.batch_size > 0 && prefill_batch.num_tokens > 0) {
            ModelForwardContext prefill_context;
            ErrorCode err_code = coordinator->run_prefill(prefill_batch, prefill_context);
            if (err_code != ErrorCode::SUCCESS) {
                LOG_ERROR("Scheduler prefill forward failed; recovering affected sequences to WAITING.");
                recoverFromPrefillFailure(prefill_batch);
                return;
            }

            if (engine_config.enable_prefix_cache) {
                if (!prefill_batch.prefill_full_hit_sequence_ids.empty()) {
                    std::unordered_set<size_t> full_hit_unique(
                        prefill_batch.prefill_full_hit_sequence_ids.begin(),
                        prefill_batch.prefill_full_hit_sequence_ids.end()
                    );
                    for (size_t seq_id : full_hit_unique) {
                        auto seq = seq_pool->get(seq_id);
                        if (seq && seq->state == SequenceState::PREFILLING) {
                            seq->state = SequenceState::PREFILLED;
                            prefill_report_pending_queue.push_back(seq_id);
                        }
                    }
                }
            }

            InflightEntry entry;
            entry.batch = prefill_batch;
            entry.op_type = InflightOp::PREFILL;
            entry.sequence_ids = prefill_batch.sequence_ids;
            prefill_inflight_batches[prefill_batch.batch_id] = std::move(entry);

            LOG_DEBUG(
                "INFLIGHT SUBMIT PREFILL: batch_id=" + std::to_string(prefill_batch.batch_id) +
                ", decode_inflight=" + std::to_string(decode_inflight_batches.size()) +
                ", prefill_inflight=" + std::to_string(prefill_inflight_batches.size()) +
                ", total_inflight=" + std::to_string(decode_inflight_batches.size() + prefill_inflight_batches.size())
            );
        }
    }
}

void Scheduler::handleFinishedAndReport() {
    const bool pd_scheduler_mode =
        engine_config.enable_pd_disaggregation && engine_config.role == "scheduler";

    // Prefiller has no decode-completion ownership.
    if (!pd_scheduler_mode || canRunDecode()) {
        handleFinishedSequence();
    }

    if (pd_scheduler_mode && to_router_channel != nullptr) {
        if (canRunPrefill()) {
            send_finished_prefill_to_router();
        }
        if (canRunDecode()) {
            send_finished_decode_to_router();
        }
    }

    // Prefiller does not own final output. Keep output/worker-free on non-PD or decoder.
    if (!pd_scheduler_mode || canRunDecode()) {
        returnSequenceOutput();
    }
}

void Scheduler::schedule() {
    while(!stop_requested.load()) {
        const bool pd_scheduler_mode =
            engine_config.enable_pd_disaggregation && engine_config.role == "scheduler";

        if (!pd_scheduler_mode) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this]() {
                return stop_requested.load() || hasPendingWorkLocked();
            });
            if (stop_requested.load()) {
                stopWorkers();
                break;
            }
        } else {
            if (stop_requested.load()) {
                stopWorkers();
                break;
            }
        }

        // Stage A: receive router commands only in PD mode.
        if (pd_scheduler_mode) {
            phaseAReceiveRouterCommands();
        }

        // move prepared sequences to waiting queue
        launchSequence();

        // Stage B: drain completion records and update sequence states
        drainCompletionRecords();

        // Stage C: submit decode path (gated by mode)
        if (canRunDecode()) {
            submitDecodePath();
        }

        // Stage D: submit prefill path (gated by mode)
        if (canRunPrefill()) {
            submitPrefillPath();
        }

        // Stage E: completion handling and output reporting
        handleFinishedAndReport();
    }
}

void Scheduler::request_stop() {
    stop_requested.store(true);
    queue_cv.notify_all();
}

void Scheduler::stopWorkers() {
    ErrorCode stop_error = coordinator->run_stop();
    if (stop_error != ErrorCode::SUCCESS) {
        LOG_ERROR("Scheduler failed to send STOP to workers.");
    }
}

void Scheduler::recoverFromPrefillFailure(const Batch& prefill_batch) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    prefill_inflight_batches.erase(prefill_batch.batch_id);
    std::unordered_set<size_t> recovered;
    for (size_t seq_id : prefill_batch.sequence_ids) {
        if (!recovered.insert(seq_id).second) {
            continue;
        }

        auto seq = seq_pool->get(seq_id);
        if (seq == nullptr) {
            continue;
        }
        //failed prefill should be moved back to waiting queue for next entry
        if (seq->state != SequenceState::PREFILLING) {
            continue;
        }
        seq->state = SequenceState::WAITING;
        waiting_queue.push_back(seq_id);

        for (auto it = prefilling_queue.begin(); it != prefilling_queue.end();) {
            if (*it == seq_id) {
                it = prefilling_queue.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void Scheduler::recoverFromDecodeFailure(const Batch& decode_batch) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    decode_inflight_batches.erase(decode_batch.batch_id);
    for (size_t seq_id : decode_batch.sequence_ids) {
        sequences_in_decoding.erase(seq_id);
    }

    // Keep decode sequences in-place for retry on the next scheduling loop.
    // no ops here
}

ErrorCode Scheduler::moveDecodingToFinished(const Batch& decode_batch) {
    for (size_t seq_id : decode_batch.sequence_ids) {
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }
        if (seq->state == SequenceState::DECODING) {
            if (seq->token_ids.size() >= engine_config.max_sequence_length
                || seq->token_ids.size() >= seq->seq_config.max_tokens
                || seq->token_ids.back() == engine_config.model_config.eos_token_id) {
                seq->state = SequenceState::FINISHED;
                LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to FINISHED state.");
            }
        }
    }
    return ErrorCode::SUCCESS;
}

void Scheduler::appendDecodedTokens(Batch& decode_batch) {
    if (decode_batch.sampled_token_ids.size() != decode_batch.sequence_ids.size()) {
        return;
    }

    for (size_t i = 0; i < decode_batch.sequence_ids.size(); ++i) {
        auto seq = seq_pool->get(decode_batch.sequence_ids[i]);
        if (!seq) {
            continue;
        }
        const size_t current_time = current_time_ns();

        if (seq->generated_token_count == 0) {
            seq->first_token_time = current_time;
        } else {
            seq->itl_sum += (current_time - seq->last_token_time);
            seq->itl_count += 1;
        }

        seq->last_token_time = current_time;
        seq->generated_token_count += 1;
        seq->add_token(decode_batch.sampled_token_ids[i]);
    }
}

ErrorCode Scheduler::movePrefilledToDecoding(const Batch& prefill_batch) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    std::unordered_set<size_t> moved;
    for (size_t seq_id : prefill_batch.sequence_ids) {
        if (!moved.insert(seq_id).second) {
            continue;
        }
        std::shared_ptr<Sequence> seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }
        if (seq->state == SequenceState::PREFILLED) {
            seq->state = SequenceState::DECODING;
            LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to DECODING state.");
            decoding_queue.push_back(seq_id);

            for (auto qit = prefilling_queue.begin(); qit != prefilling_queue.end(); ++qit) {
                if (*qit == seq_id) {
                    prefilling_queue.erase(qit);
                    break;
                }
            }
        }
    }
    return ErrorCode::SUCCESS;
}

std::variant<Batch, ErrorCode> Scheduler::buildDecodeBatch() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    Batch batch;
    batch.batch_id = next_batch_id.fetch_add(1);
    batch.batch_size = 0;
    batch.num_tokens = 0;

    for (size_t seq_id : decoding_queue) {
        if (sequences_in_decoding.find(seq_id) != sequences_in_decoding.end()) {
            continue;
        }
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }
        if (seq->state == SequenceState::FINISHED || seq->token_ids.empty()) {
            continue;
        }

        size_t last_pos = seq->token_ids.size() - 1;
        batch.token_ids.push_back(seq->token_ids[last_pos]);
        batch.token_positions.push_back(last_pos);
        batch.sequence_ids.push_back(seq_id);
        batch.num_tokens += 1;
        batch.batch_size++;

        if (batch.batch_size >= engine_config.max_decode_batch_size) {
            break;
        }
    }

    for (auto it = prefilling_queue.begin();
         it != prefilling_queue.end() && batch.batch_size < engine_config.max_decode_batch_size;) {
        size_t seq_id = *it;
        auto seq = seq_pool->get(seq_id);

        if (!seq) {
            it = prefilling_queue.erase(it);
            continue;
        }

        if (sequences_in_decoding.find(seq_id) != sequences_in_decoding.end()) {
            ++it;
            continue;
        }

        if (seq->state != SequenceState::PREFILLED) {
            ++it;
            continue;
        }

        seq->state = SequenceState::DECODING;
        decoding_queue.push_back(seq_id);
        LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved from PREFILLED to DECODING state.");
        it = prefilling_queue.erase(it);

        if (seq->token_ids.empty()) {
            continue;
        }

        size_t last_pos = seq->token_ids.size() - 1;
        batch.token_ids.push_back(seq->token_ids[last_pos]);
        batch.token_positions.push_back(last_pos);
        batch.sequence_ids.push_back(seq_id);
        batch.num_tokens += 1;
        batch.batch_size++;
    }

    if (batch.batch_size > 0) {
        LOG_DEBUG("BUILD DECODE BATCH: batch_size=" + std::to_string(batch.batch_size) + ", num_tokens=" + std::to_string(batch.num_tokens));
    }
    return batch;
}

std::variant<Batch, ErrorCode> Scheduler::buildPrefillBatch() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    Batch batch;
    batch.batch_id = next_batch_id.fetch_add(1);
    batch.batch_size = 0;
    batch.num_tokens = 0;

    for (auto it = waiting_queue.begin(); it != waiting_queue.end();) {
        size_t seq_id = *it;
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            it = waiting_queue.erase(it);
            continue;
        }
        if (seq->state != SequenceState::WAITING) {
            ++it;
            continue;
        }

        it = waiting_queue.erase(it);
        seq->state = SequenceState::PREFILLING;
        prefilling_queue.push_back(seq_id);
        LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to PREFILLING state.");

        batch.token_ids.insert(batch.token_ids.end(), seq->token_ids.begin(), seq->token_ids.end());
        size_t seq_len = seq->token_ids.size();
        for (size_t i = 0; i < seq_len; ++i) {
            batch.token_positions.push_back(i);
            batch.sequence_ids.push_back(seq_id);
        }
        batch.max_token_positions.push_back(seq_len - 1);
        batch.num_tokens += seq_len;
        batch.batch_size++;

        LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " added to prefill batch with " + std::to_string(seq_len) + " tokens.");

        if (batch.batch_size >= engine_config.max_prefill_batch_size) {
            break;
        }
    }
    if(batch.batch_size > 0){
        LOG_DEBUG("BUILD PREFILL BATCH: batch_size=" + std::to_string(batch.batch_size) + ", num_tokens=" + std::to_string(batch.num_tokens));
    }
    return batch;
}

ErrorCode Scheduler::addSequence(
    size_t seq_id,
    std::vector<size_t> token_ids,
    const SequenceConfig& sequence_config
) {
    auto new_seq = seq_pool->create(seq_id, sequence_config);
    new_seq->token_ids = token_ids;
    new_seq->seq_len = token_ids.size();
    new_seq->state = SequenceState::PREPARED;
    new_seq->blocks.clear();

    std::lock_guard<std::mutex> lock(queue_mutex);
    prepared_queue.push_back(seq_id);
    new_seq->submitted_time = current_time_ns();

    LOG_DEBUG("Sequence added to prepared queue: " + std::to_string(new_seq->seq_id));
    queue_cv.notify_one();
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::launchSequence() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    while (!prepared_queue.empty()) {
        size_t seq_id = prepared_queue.front();
        prepared_queue.erase(prepared_queue.begin());
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }
        seq->state = SequenceState::WAITING;
        waiting_queue.push_back(seq_id);
        LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to WAITING state.");
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::handleFinishedSequence() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto it = decoding_queue.begin(); it != decoding_queue.end();) {
        auto seq = seq_pool->get(*it);
        if (seq && seq->state == SequenceState::FINISHED) {
            finished_queue.push_back(*it);
            decode_report_pending_queue.push_back(*it);
            LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to FINISHED queue.");
            it = decoding_queue.erase(it);
        } else {
            ++it;
        }
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::returnSequenceOutput() {
    std::vector<size_t> finished_to_free;
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex);
        for (size_t seq_id : finished_queue) {
            auto seq = seq_pool->get(seq_id);
            if (!seq) {
                continue;
            }
            //notify waiting get_request_output to return output and free sequence
            if (seq->state == SequenceState::FINISHED && !seq->finish_handled) {
                std::lock_guard<std::mutex> lock(seq->mtx);
                seq->cv.notify_one();
                seq->finish_handled = true;
                finished_to_free.push_back(seq_id);
            }
        }
    }
    //notify workers to free cache for finished sequences
    if (!finished_to_free.empty()) {
        freeFinishedSequencesOnWorkers(finished_to_free);
    }
    return ErrorCode::SUCCESS;
}

void Scheduler::freeFinishedSequencesOnWorkers(const std::vector<size_t>& sequence_ids) {
    if (sequence_ids.empty()) {
        return;
    }
    //build a lightwight control batch with sequence ids 
    // to notify workers to free cache for these finished sequences
    Batch control_batch;
    control_batch.sequence_ids = sequence_ids;

    ErrorCode free_error = coordinator->run_free(control_batch);
    if (free_error != ErrorCode::SUCCESS) {
        LOG_ERROR("Scheduler failed to send FREE_SEQ control batch to workers.");
    }
}

ErrorCode Scheduler::getSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq) {
    seq = seq_pool->get(seq_id);
    if (!seq) {
        return ErrorCode::SEQUENCE_NOT_FOUND;
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::getFinishedSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq) {
    seq = nullptr;
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (size_t finished_id : finished_queue) {
        if (finished_id == seq_id) {
            seq = seq_pool->get(seq_id);
            return seq ? ErrorCode::SUCCESS : ErrorCode::SEQUENCE_NOT_FOUND;
        }
    }
    return ErrorCode::SEQUENCE_NOT_FOUND;
}

ErrorCode Scheduler::removeFinishedSequenceById(size_t seq_id) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto it = finished_queue.begin(); it != finished_queue.end(); ++it) {
        if (*it == seq_id) {
            finished_queue.erase(it);
            seq_pool->erase(seq_id);
            LOG_DEBUG("Sequence " + std::to_string(seq_id) + " removed from finished queue and seq pool.");
            return ErrorCode::SUCCESS;
        }
    }
    return ErrorCode::SEQUENCE_NOT_FOUND;
}

void Scheduler::applyPrefixProbeToPrefillBatch(Batch& prefill_batch) {
    prefill_batch.prefill_full_hit_sequence_ids.clear();
    bool prefix_probe_success = true;
    if(prefill_batch.prefix_hit_tokens_per_seq.empty()){
        LOG_ERROR("Prefix probe failed for batch " + std::to_string(prefill_batch.batch_id));
        prefix_probe_success = false;
    }
    if(prefill_batch.prefix_hit_tokens_per_seq.size() != prefill_batch.batch_size){
        LOG_ERROR("Prefix probe returned mismatched prefix hit tokens info for batch " + std::to_string(prefill_batch.batch_id));
        prefix_probe_success = false;
    }
    if(prefill_batch.max_token_positions.size() != prefill_batch.batch_size){
        LOG_ERROR("Prefill batch has mismatched max_token_positions for batch " + std::to_string(prefill_batch.batch_id));
        prefix_probe_success = false;
    }
    if(!prefix_probe_success){
        prefill_batch.prefix_hit_tokens_per_seq.clear();
    }else{
        LOG_DEBUG("Prefix probe succeeded for batch " + std::to_string(prefill_batch.batch_id) + 
            ", prefix hit seq num: " + 
            std::to_string(prefill_batch.prefix_hit_tokens_per_seq.size())
        );
        std::vector<size_t> new_prefix_hit_tokens;
        new_prefix_hit_tokens.reserve(prefill_batch.batch_size);

        size_t cursor = 0;
        for(size_t seq_idx = 0; seq_idx < prefill_batch.batch_size; ++seq_idx) {
            if (seq_idx >= prefill_batch.max_token_positions.size()) {
                break;
            }

            const size_t seq_len = prefill_batch.max_token_positions[seq_idx] + 1;
            if (cursor + seq_len > prefill_batch.token_ids.size() ||
                cursor + seq_len > prefill_batch.token_positions.size() ||
                cursor + seq_len > prefill_batch.sequence_ids.size()) {
                LOG_ERROR("Prefill batch layout invalid while applying prefix probe for batch " + std::to_string(prefill_batch.batch_id));
                break;
            }

            const size_t seq_id = prefill_batch.sequence_ids[cursor];
            size_t hit_tokens = prefill_batch.prefix_hit_tokens_per_seq[seq_idx];
            if (hit_tokens > seq_len) {
                hit_tokens = seq_len;
            }

            if (hit_tokens >= seq_len) {
                prefill_batch.prefill_full_hit_sequence_ids.push_back(seq_id);
            }

            // Keep full sequence payload in batch and only pass normalized hit tokens
            // to workers. Worker side is responsible for binding cache blocks and
            // deciding whether to skip prefill compute for fully hit sequences.
            new_prefix_hit_tokens.push_back(hit_tokens);

            cursor += seq_len;
        }
        prefill_batch.prefix_hit_tokens_per_seq.swap(new_prefix_hit_tokens);

        LOG_DEBUG(
            "PREFIX APPLY: batch_id=" + std::to_string(prefill_batch.batch_id) +
            ", batch_size=" + std::to_string(prefill_batch.batch_size) +
            ", num_tokens=" + std::to_string(prefill_batch.num_tokens)
        );
    }
}

void Scheduler::send_finished_prefill_to_router() {
    if (to_router_channel == nullptr) {
        return;
    }

    std::vector<size_t> prefill_snapshot;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        prefill_snapshot.swap(prefill_report_pending_queue);
    }

    for (size_t seq_id : prefill_snapshot) {
        auto seq = seq_pool->get(seq_id);
        if (seq && seq->state == SequenceState::PREFILLED) {
            RouteMessage msg;
            msg.seq_id = seq_id;
            msg.route_type = RouteType::PREFILL;
            msg.sequence_config = seq->seq_config;
            msg.token_ids = seq->token_ids;
            to_router_channel->send(msg);
        }
    }
}
void Scheduler::send_finished_decode_to_router() {
    if (to_router_channel == nullptr) {
        return;
    }

    std::vector<size_t> finished_snapshot;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        finished_snapshot.swap(decode_report_pending_queue);
    }

    for (size_t seq_id : finished_snapshot) {
        auto seq = seq_pool->get(seq_id);
        if (seq && seq->state == SequenceState::FINISHED) {
            RouteMessage msg;
            msg.seq_id = seq_id;
            msg.route_type = RouteType::DECODE;
            msg.sequence_config = seq->seq_config;
            msg.token_ids = seq->token_ids;
            to_router_channel->send(msg);
        }
    }
    
}