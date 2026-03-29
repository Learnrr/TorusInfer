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

    ErrorCode create_request(const std::vector<size_t>& token_ids, size_t& request_id) {
        std::lock_guard<std::mutex> lock(requests_mutex);
        if (token_ids.empty()) {
            return ErrorCode::INVALID_INPUT;
        }

        request_id = next_request_id++;
        Request new_request(request_id, token_ids, RequestStatus::PENDING);
        new_request.set_sequence_id(next_sequence_id++);
        id_requests.emplace(request_id, new_request);
        pending_requests.push_back(request_id);

        return ErrorCode::SUCCESS;
    }

    ErrorCode submit_request(size_t request_id) {
        std::lock_guard<std::mutex> lock(requests_mutex);
        auto it = id_requests.find(request_id);
        if (it == id_requests.end()) {
            return ErrorCode::SEQUENCE_NOT_FOUND;
        }

        Request& req = it->second;
        
        pending_requests.erase(
            std::remove(
                pending_requests.begin(),
                pending_requests.end(),
                request_id
            ),
            pending_requests.end()
        );
        req.set_status(RequestStatus::IN_PROGRESS);
        return ErrorCode::SUCCESS;
    }

    ErrorCode cancel_request(size_t request_id) {
        std::lock_guard<std::mutex> lock(requests_mutex);
        auto it = id_requests.find(request_id);
        if (it == id_requests.end()) {
            return ErrorCode::SEQUENCE_NOT_FOUND;
        }

        Request& req = it->second;
        req.set_status(RequestStatus::CANCELLED);
        pending_requests.erase(
            std::remove(
                pending_requests.begin(),
                pending_requests.end(),
                request_id
            ),
            pending_requests.end()
        );

        return ErrorCode::SUCCESS;
    }

    ErrorCode get_request_sequence_id(size_t request_id, size_t& sequence_id) const {
        std::lock_guard<std::mutex> lock(requests_mutex);
        auto it = id_requests.find(request_id);
        if (it == id_requests.end()) {
            sequence_id = 0;
            return ErrorCode::SEQUENCE_NOT_FOUND;
        }

        sequence_id = it->second.get_sequence_id();
        return ErrorCode::SUCCESS;
    }

    ErrorCode get_request_token_ids(size_t request_id, std::vector<size_t>& token_ids) const {
        std::lock_guard<std::mutex> lock(requests_mutex);
        auto it = id_requests.find(request_id);
        if (it == id_requests.end()) {
            token_ids.clear();
            return ErrorCode::SEQUENCE_NOT_FOUND;
        }

        token_ids = it->second.get_token_ids();
        return ErrorCode::SUCCESS;
    }

    ErrorCode get_request_status(size_t request_id, RequestStatus& status) const {
        std::lock_guard<std::mutex> lock(requests_mutex);
        auto it = id_requests.find(request_id);
        if (it == id_requests.end()) {
            status = RequestStatus::FAILED;
            return ErrorCode::SEQUENCE_NOT_FOUND;
        }

        status = it->second.get_status();
        return ErrorCode::SUCCESS;
    }

    ErrorCode set_request_status(size_t request_id, RequestStatus status) {
        std::lock_guard<std::mutex> lock(requests_mutex);
        auto it = id_requests.find(request_id);
        if (it == id_requests.end()) {
            return ErrorCode::SEQUENCE_NOT_FOUND;
        }

        it->second.set_status(status);
        return ErrorCode::SUCCESS;
    }

private:
    std::unordered_map<size_t, Request> id_requests;
    std::deque<size_t> pending_requests;
    size_t next_request_id = 1;
    size_t next_sequence_id = 1;
    mutable std::mutex requests_mutex;

};