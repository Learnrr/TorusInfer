#include "channel/IpcChannel.h"

#include "utils/logger.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {
std::string fifo_path_from_name(const std::string& name) {
    return "/tmp/infer2_" + name + ".fifo";
}
}

IpcChannel::IpcChannel(std::string name, int read_fd, int write_fd)
    : name(std::move(name)), read_fd(read_fd), write_fd(write_fd) {
    ensure_open();
}

IpcChannel::~IpcChannel() {
    if (read_fd >= 0) {
        close(read_fd);
        read_fd = -1;
    }
    if (write_fd >= 0 && write_fd != read_fd) {
        close(write_fd);
        write_fd = -1;
    }
}

bool IpcChannel::ensure_open() {
    if (read_fd >= 0 && write_fd >= 0) {
        return true;
    }

    const std::string fifo_path = fifo_path_from_name(name);
    if (mkfifo(fifo_path.c_str(), 0666) != 0 && errno != EEXIST) {
        std::ostringstream oss;
        oss << "mkfifo failed for " << fifo_path << ": " << std::strerror(errno);
        LOG_ERROR(oss.str());
        return false;
    }

    // Open as RDWR to avoid startup ordering deadlocks (peer not opened yet).
    const int fd = open(fifo_path.c_str(), O_RDWR);
    if (fd < 0) {
        std::ostringstream oss;
        oss << "open failed for " << fifo_path << ": " << std::strerror(errno);
        LOG_ERROR(oss.str());
        return false;
    }

    read_fd = fd;
    write_fd = fd;
    return true;
}

bool IpcChannel::write_all(const void* buf, size_t len) {
    const char* p = static_cast<const char*>(buf);
    size_t written = 0;
    while (written < len) {
        const ssize_t n = write(write_fd, p + written, len - written);
        if (n <= 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        written += static_cast<size_t>(n);
    }
    return true;
}

bool IpcChannel::read_all(void* buf, size_t len) {
    char* p = static_cast<char*>(buf);
    size_t read_n = 0;
    while (read_n < len) {
        const ssize_t n = read(read_fd, p + read_n, len - read_n);
        if (n <= 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        read_n += static_cast<size_t>(n);
    }
    return true;
}

void IpcChannel::send(const ChannelMessage& message) {
    if (!ensure_open()) {
        LOG_ERROR("IpcChannel send failed: fifo open not ready");
        return;
    }

    const std::vector<char> payload = message.serialize();
    const uint64_t payload_size = static_cast<uint64_t>(payload.size());

    if (!write_all(&payload_size, sizeof(payload_size))) {
        LOG_ERROR("IpcChannel send failed: write size prefix failed");
        return;
    }

    if (payload_size == 0) {
        return;
    }

    if (!write_all(payload.data(), payload.size())) {
        LOG_ERROR("IpcChannel send failed: write payload failed");
    }
}

void IpcChannel::receive(ChannelMessage& message) {
    if (!ensure_open()) {
        LOG_ERROR("IpcChannel receive failed: fifo open not ready");
        return;
    }

    uint64_t payload_size = 0;
    if (!read_all(&payload_size, sizeof(payload_size))) {
        LOG_ERROR("IpcChannel receive failed: read size prefix failed");
        return;
    }

    std::vector<char> payload(static_cast<size_t>(payload_size));
    if (payload_size > 0 && !read_all(payload.data(), payload.size())) {
        LOG_ERROR("IpcChannel receive failed: read payload failed");
        return;
    }

    message.deserialize(payload);
}
