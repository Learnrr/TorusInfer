#pragma once
#include <vector>


struct SequenceOutput {
    size_t seq_id;
    std::vector<size_t> token_ids;
};

