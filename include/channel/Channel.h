#pragma once
#include "channel/ChannelMessage.h"

class Channel {
    public:
        virtual ~Channel() = default;
        virtual void send(const ChannelMessage& message) = 0;
        virtual void receive(ChannelMessage& message) = 0;
    virtual bool try_receive(ChannelMessage& message) = 0;
};