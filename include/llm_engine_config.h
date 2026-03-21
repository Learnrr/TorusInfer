#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include "ModelConfig.h"
#include "utils/logger.h"
#include "error.h"

class LLMEngineConfig {
public:
    size_t max_batch_size;
    size_t max_sequence_length;
    size_t total_cache_size;
    size_t block_size;
    ModelConfig model_config;
    std::string model_config_path;

    LLMEngineConfig(
        size_t max_batch_size, 
        size_t max_sequence_length,
        size_t total_cache_size,
        size_t block_size
    )
        : max_batch_size(max_batch_size), 
        max_sequence_length(max_sequence_length), 
        total_cache_size(total_cache_size), 
        block_size(block_size) {}

    ErrorCode build_from_file(const char* config_path) {
        std::ifstream file(config_path);
        if(!file.is_open()) {
            {
                std::ostringstream oss;
                oss << "Failed to open engine config file: " << config_path;
                LOG_ERROR(oss.str());
            }
            return ErrorCode::FAILED_TO_OPEN_CONFIG_FILE;
        }
        nlohmann::json config;
        file >> config;

        max_batch_size = config.value("max_batch_size", static_cast<size_t>(16));
        max_sequence_length = config.value("max_sequence_length", static_cast<size_t>(1024));
        total_cache_size = config.value("total_cache_size", static_cast<size_t>(1024ULL * 1024ULL * 1024ULL)); // 1GB default
        block_size = config.value("block_size", static_cast<size_t>(16)); // 1KB default
        model_config_path = config.value("model_config_path", "");
        // Load model config from the specified path
        if(model_config_path.empty()) {
            {
                std::ostringstream oss;
                oss << "model_config_path is required in engine config JSON";
                LOG_ERROR(oss.str());
            }
            return ErrorCode::FAILED_TO_OPEN_CONFIG_FILE;
        }
        ErrorCode model_config_result = model_config.build_from_file(model_config_path.c_str());
        if (model_config_result != ErrorCode::SUCCESS) {
            return model_config_result;
        }
        return ErrorCode::SUCCESS;
    }
};