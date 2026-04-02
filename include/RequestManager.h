#pragma once
#include <algorithm>
#include <deque>
#include <unordered_map>
#include <vector>
#include "error.h"
#include "Request.h"
#include <mutex>

class RequestManager {
public:
    RequestManager() = default;
    ~RequestManager() = default;
    // Create a new request with the given token IDs and return the assigned request ID
    // request created will be in pending state, and waiting to be submitted to scheduler
    ErrorCode create_request(const std::vector<size_t>& token_ids, size_t& request_id);
    // Submit a request for processing by its request ID
    ErrorCode submit_request(size_t request_id);
    // Cancel a request by its request ID
    // cancel request will be removed from pending queue;
    // only pending request can be cancelled, in-progress request cannot be cancelled
    ErrorCode cancel_request(size_t request_id);

    ErrorCode get_request_sequence_id(size_t request_id, size_t& sequence_id) const;

    ErrorCode get_request_token_ids(size_t request_id, std::vector<size_t>& token_ids) const;

    ErrorCode get_request_status(size_t request_id, RequestStatus& status) const;

    ErrorCode set_request_status(size_t request_id, RequestStatus status);

private:
    std::unordered_map<size_t, Request> id_requests;
    std::deque<size_t> pending_requests;
    size_t next_request_id = 1;
    size_t next_sequence_id = 1;
    mutable std::mutex requests_mutex;

};