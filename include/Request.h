#pragma once
#include <vector>

enum class RequestStatus {
    PENDING,
    IN_PROGRESS,
    COMPLETED,
    CANCELLED,
    FAILED
};

class Request{
public:
    Request() = delete;
    Request(
        size_t request_id, 
        const std::vector<size_t>& token_ids, 
        RequestStatus status = RequestStatus::PENDING
    )
        : status(status), 
        request_id(request_id), 
        sequence_id(0), 
        token_ids(token_ids) {}
        
    ~Request() = default;

    void set_sequence_id(size_t seq_id) {
        sequence_id = seq_id;
    }

    void set_status(RequestStatus status) {
        this->status = status;
    }

    RequestStatus get_status() const {
        return status;
    }

    size_t get_request_id() const {
        return request_id;
    }

    size_t get_sequence_id() const {
        return sequence_id;
    }

    const std::vector<size_t>& get_token_ids() const {
        return token_ids;
    }

private:
    RequestStatus status;
    size_t request_id;
    size_t sequence_id;
    std::vector<size_t> token_ids;
};
