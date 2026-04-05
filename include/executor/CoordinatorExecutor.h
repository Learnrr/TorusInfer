#pragma once

#include "executor/Executor.h"
#include "channel/Channel.h"
#include "channel/ChannelMessage.h"
#include "model/IModel.h"

class CoordinatorExecutor : public Executor {
public:
    explicit CoordinatorExecutor(IModel* model = nullptr) : model(model) {}

    void set_channels(
        Channel* to_worker0,
        Channel* from_worker_last
    ) {
        this->to_worker0 = to_worker0;
        this->from_worker_last = from_worker_last;
    }

    void run_prefill(Batch& batch, ModelForwardContext& context) override;
    void run_decode(Batch& batch, ModelForwardContext& context) override;
    void run_free(Batch& batch);
    void run_release_events(Batch& batch);
    void run_stop();
    bool consume_last_forward_ok();

private:
    IModel* model = nullptr;
    Channel* to_worker0 = nullptr;
    Channel* from_worker_last = nullptr;
    bool last_forward_ok = true;
};