#pragma once

#include "Sequence.h"

#include <memory>
#include <mutex>
#include <unordered_map>

class SequencePool {
public:
    std::shared_ptr<Sequence> create(
        size_t seq_id,
        const SequenceConfig& config = SequenceConfig()
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto seq = std::make_shared<Sequence>(seq_id, config);
        pool_[seq_id] = seq;
        return seq;
    }

    std::shared_ptr<Sequence> get(size_t seq_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pool_.find(seq_id);
        if (it == pool_.end()) {
            return nullptr;
        }
        return it->second;
    }

    void upsert(const std::shared_ptr<Sequence>& seq) {
        if (!seq) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        pool_[seq->seq_id] = seq;
    }

    bool erase(size_t seq_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.erase(seq_id) > 0;
    }

    bool contains(size_t seq_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.find(seq_id) != pool_.end();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::shared_ptr<Sequence>> pool_;
};
