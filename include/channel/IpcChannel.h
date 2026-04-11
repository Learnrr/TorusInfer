#pragma once

#include "channel/Channel.h"
#include "channel/ChannelMessage.h"
#include <string>
#include <vector>

class IpcChannel : public Channel {
    public:
        IpcChannel(std::string name, int read_fd = -1, int write_fd = -1);
        ~IpcChannel();

        void send(const ChannelMessage& message) override;
        void receive(ChannelMessage& message) override;
        bool try_receive(ChannelMessage& message) override;
    private:
        bool ensure_open();
        bool write_all(const void* buf, size_t len);
        bool read_all(void* buf, size_t len);

        std::string name;
        int read_fd;
        int write_fd;
        std::vector<char> recv_buffer;
};
