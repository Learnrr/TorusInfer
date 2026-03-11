#include "Engine.h"

void Engine::init(){
    cache_manager = make_unique<KVCacheManager>();
    workspace = make_unique<Workspace>();
    model = make_unique<Model>(*workspace);
    scheduler = make_unique<Scheduler>(cache_manager, model);
}

void Engine::run() {
    scheduler->schedule();
}   

void Engine::create_sequence(size_t seq_id, vector<size_t> token_ids) {
    scheduler->addSequence(seq_id, token_ids);
}