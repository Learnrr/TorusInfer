#include "model/ModelWeights.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include "utils/tensor_debug.h"

std::vector<std::string> ResolveSafetensorShardsFromIndex(
    const std::filesystem::path& index_path) {

    std::vector<std::string> shards;
    if (!std::filesystem::exists(index_path)) {
        return shards;
    }

    std::ifstream infile(index_path);
    if (!infile.is_open()) {
        return shards;
    }

    json index_json;
    try {
        infile >> index_json;
    } catch (...) {
        return {};
    }

    if (!index_json.contains("weight_map") || !index_json["weight_map"].is_object()) {
        return {};
    }

    std::unordered_set<std::string> seen;
    for (const auto& item : index_json["weight_map"].items()) {
        const std::string rel_name = item.value().get<std::string>();
        const std::filesystem::path shard_path = index_path.parent_path() / rel_name;
        const std::string shard = shard_path.string();
        if (seen.insert(shard).second) {
            if (!std::filesystem::exists(shard_path)) {
                return {};
            }
            shards.push_back(shard);
        }
    }

    return shards;
}

std::vector<std::string> ResolveSafetensorShards(const std::string& model_path) {
    std::filesystem::path p(model_path);
    if (!std::filesystem::exists(p)) {
        return {};
    }

    const std::filesystem::path base_dir = std::filesystem::is_directory(p) ? p : p.parent_path();
    const std::filesystem::path index_path = base_dir / "model.safetensors.index.json";
    return ResolveSafetensorShardsFromIndex(index_path);
}

ErrorCode WeightLayout::build_weight_layout(
    const ModelConfig& config
) {
    size_t offset = 0;
    const DataType layout_dtype = config.data_type;
    if (weights == nullptr) {
        LOG_ERROR("build_weight_layout called with null weights pointer");
        return ErrorCode::INVALID_INPUT;
    }
    {
        std::ostringstream oss;
        oss << "build_weight_layout begin, layer_config_count=" << config.layer_configs.size()
            << ", weights_ptr=" << weights;
        LOG_DEBUG(oss.str());
    }
    layer_weights.clear();
    layer_weights.reserve(config.layer_configs.size());

    if (!engine_config->enable_pipeline_parallel || engine_config->is_first_stage()) {
        //embedding weights
        embedding_weights = Tensor(
            config.vocab_size * config.hidden_size,
            static_cast<void*>(static_cast<char*>(weights) + offset),
            {config.vocab_size, config.hidden_size},
            layout_dtype
        );
        offset += embedding_weights.size;
        {
            std::ostringstream oss;
            oss << "layout embedding done, bytes=" << embedding_weights.size << ", offset=" << offset;
            LOG_DEBUG(oss.str());
        }
    }
    
    //transformer layers and layernorm and lm head
    size_t cfg_idx = 0;
    for (const auto& layer_cfg_base : config.layer_configs) {
        if (engine_config->enable_pipeline_parallel) {
            bool keep_cfg = true;
            const size_t num_layers = config.num_hidden_layers;
            if (cfg_idx < num_layers) {
                keep_cfg = (cfg_idx >= engine_config->stage_start_layer && cfg_idx < engine_config->stage_end_layer);
            } else if (cfg_idx == num_layers || cfg_idx == num_layers + 1) {
                keep_cfg = engine_config->is_last_stage();
            }
            if (!keep_cfg) {
                layer_weights.push_back(nullptr);
                ++cfg_idx;
                continue;
            }
        }
        if (!layer_cfg_base) {
            std::ostringstream oss;
            oss << "null layer config at index=" << cfg_idx;
            LOG_ERROR(oss.str());
            return ErrorCode::INVALID_INPUT;
        }
        if (auto transformer_cfg = std::dynamic_pointer_cast<TransformerLayerConfig>(layer_cfg_base)) {
            {
                std::ostringstream oss;
                oss << "layout cfg_idx=" << cfg_idx
                    << " type=Transformer"
                    << " q_heads=" << transformer_cfg->attention_config.num_attention_heads
                    << " kv_heads=" << transformer_cfg->attention_config.num_kv_heads
                    << " head_dim=" << transformer_cfg->attention_config.head_dim
                    << " mlp_linears=" << transformer_cfg->mlp_config.mlp_linears.size();
                LOG_DEBUG(oss.str());
            }
            auto transformer_layout = std::make_shared<TransformerLayerWeightLayout>();

            //attention qkv weights
            size_t q_hidden;
            q_hidden = transformer_cfg->attention_config.num_attention_heads 
                    * transformer_cfg->attention_config.head_dim;
            size_t kv_hidden;
            kv_hidden = transformer_cfg->attention_config.num_kv_heads 
                    * transformer_cfg->attention_config.head_dim;

            transformer_layout->attention_weights.qkv_proj_weight = Tensor(
                config.hidden_size * (q_hidden + 2 * kv_hidden),
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {config.hidden_size, q_hidden + 2 * kv_hidden},
                layout_dtype
            );
            offset += transformer_layout->attention_weights.qkv_proj_weight.size;
            {
                std::ostringstream oss;
                oss << "  qkv_proj bytes=" << transformer_layout->attention_weights.qkv_proj_weight.size
                    << ", offset=" << offset;
                LOG_DEBUG(oss.str());
            }
            transformer_layout->attention_weights.qkv_proj_bias = Tensor(
                q_hidden + 2 * kv_hidden,
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {q_hidden + 2 * kv_hidden},
                layout_dtype
            );
            offset += transformer_layout->attention_weights.qkv_proj_bias.size;
            {
                std::ostringstream oss;
                oss << "  qkv_proj bias bytes=" << transformer_layout->attention_weights.qkv_proj_bias.size
                    << ", offset=" << offset;
                LOG_DEBUG(oss.str());
            }
            //attention output projection weights
            transformer_layout->attention_weights.o_proj_weight = Tensor(
                q_hidden * config.hidden_size,
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {q_hidden, config.hidden_size},
                layout_dtype
            );
            offset += transformer_layout->attention_weights.o_proj_weight.size;
            {
                std::ostringstream oss;
                oss << "  o_proj bytes=" << transformer_layout->attention_weights.o_proj_weight.size
                    << ", offset=" << offset;
                LOG_DEBUG(oss.str());
            }
            //layer norm weights
            transformer_layout->norm_weights.resize(2);
            transformer_layout->norm_weights[0] = LayerNormLayerWeightLayout();
            transformer_layout->norm_weights[1] = LayerNormLayerWeightLayout();
            transformer_layout->norm_weights[0].norm_weight = Tensor(
                config.hidden_size,
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {config.hidden_size},
                layout_dtype
            );
            transformer_layout->norm_weights[0].gamma = transformer_layout->norm_weights[0].norm_weight.data;
            offset += transformer_layout->norm_weights[0].norm_weight.size;
            transformer_layout->norm_weights[1].norm_weight = Tensor(
                config.hidden_size,
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {config.hidden_size},
                layout_dtype
            );
            transformer_layout->norm_weights[1].gamma = transformer_layout->norm_weights[1].norm_weight.data;
            offset += transformer_layout->norm_weights[1].norm_weight.size;
            {
                std::ostringstream oss;
                oss << "  norms bytes="
                    << (transformer_layout->norm_weights[0].norm_weight.size + transformer_layout->norm_weights[1].norm_weight.size)
                    << ", offset=" << offset;
                LOG_DEBUG(oss.str());
            }


            //mlp weights
            size_t intermediate_size = transformer_cfg->mlp_config.intermediate_size;
            if (intermediate_size == 0 && !transformer_cfg->mlp_config.mlp_linears.empty()) {
                intermediate_size = transformer_cfg->mlp_config.mlp_linears[0].out_features;
            }
            transformer_layout->mlp_weights.mlp_linears_weight.clear();
            transformer_layout->mlp_weights.mlp_linears_weight.reserve(
                transformer_cfg->mlp_config.mlp_linears.size()
            );
            for(const auto& linear_cfg : transformer_cfg->mlp_config.mlp_linears){
                transformer_layout->mlp_weights.mlp_linears_weight.emplace_back();
                auto& linear_layout = transformer_layout->mlp_weights.mlp_linears_weight.back();
                linear_layout.linear_weight = Tensor(
                    linear_cfg.in_features * linear_cfg.out_features,
                    static_cast<void*>(static_cast<char*>(weights) + offset),
                    {linear_cfg.in_features, linear_cfg.out_features},
                    layout_dtype
                );
                offset += linear_layout.linear_weight.size;
                {
                    std::ostringstream oss;
                    oss << "  mlp linear weight bytes=" << linear_layout.linear_weight.size
                        << ", in=" << linear_cfg.in_features << ", out=" << linear_cfg.out_features
                        << ", offset=" << offset;
                    LOG_DEBUG(oss.str());
                }
                if (transformer_cfg->mlp_config.has_bias) {
                    linear_layout.linear_bias = Tensor(
                        linear_cfg.out_features,
                        static_cast<void*>(static_cast<char*>(weights) + offset),
                        {linear_cfg.out_features},
                        layout_dtype
                    );
                    offset += linear_layout.linear_bias.size;
                    {
                        std::ostringstream oss;
                        oss << "  mlp linear bias bytes=" << linear_layout.linear_bias.size
                            << ", offset=" << offset;
                        LOG_DEBUG(oss.str());
                    }
                }
            }

            layer_weights.push_back(transformer_layout);
            ++cfg_idx;
            continue;
        }

        if (auto linear_cfg = std::dynamic_pointer_cast<LinearLayerConfig>(layer_cfg_base)) {
            {
                std::ostringstream oss;
                oss << "layout cfg_idx=" << cfg_idx << " type=Linear"
                    << " in=" << linear_cfg->linear_config.in_features
                    << " out=" << linear_cfg->linear_config.out_features;
                LOG_DEBUG(oss.str());
            }
            auto linear_layout = std::make_shared<LinearLayerWeightLayout>();
            const size_t in_features = linear_cfg->linear_config.in_features;
            const size_t out_features = linear_cfg->linear_config.out_features;
            linear_layout->linear_weight = Tensor(
                in_features * out_features, 
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {in_features, out_features}, 
                layout_dtype
            );
            offset += linear_layout->linear_weight.size;
            layer_weights.push_back(linear_layout);
            ++cfg_idx;
            continue;
        }

        if (auto norm_cfg = std::dynamic_pointer_cast<LayerNormLayerConfig>(layer_cfg_base)) {
            {
                std::ostringstream oss;
                oss << "layout cfg_idx=" << cfg_idx << " type=LayerNorm"
                    << " size=" << (norm_cfg->norm_size == 0 ? config.hidden_size : norm_cfg->norm_size);
                LOG_DEBUG(oss.str());
            }
            auto norm_layout = std::make_shared<LayerNormLayerWeightLayout>();
            const size_t norm_size = norm_cfg->norm_size == 0 ? config.hidden_size : norm_cfg->norm_size;
            norm_layout->norm_weight = Tensor(
                norm_size, 
                static_cast<void*>(static_cast<char*>(weights) + offset), 
                {norm_size}, 
                layout_dtype
            );
            norm_layout->gamma = norm_layout->norm_weight.data;
            offset += norm_layout->norm_weight.size;
            layer_weights.push_back(norm_layout);
            ++cfg_idx;
            continue;
        }

        {
            std::ostringstream oss;
            oss << "Unknown layer config type at index=" << cfg_idx;
            LOG_ERROR(oss.str());
        }
        return ErrorCode::INVALID_INPUT;
    }
    {
        std::ostringstream oss;
        oss << "build_weight_layout finished, total offset=" << offset
            << ", layer_weights=" << layer_weights.size();
        LOG_DEBUG(oss.str());
    }
    total_size = offset;
    return ErrorCode::SUCCESS;
}

std::variant<ErrorCode, size_t> ModelWeights::read_total_size(const char* model_safetensors_index_json) const {
    size_t total_size = 0;
    std::ifstream infile(model_safetensors_index_json, std::ios::binary);
    if (!infile.is_open()) {
        return ErrorCode::LOAD_ERROR;
    }
    json index_json;
    try {
        infile >> index_json;
    } catch (...) {
        return ErrorCode::LOAD_ERROR;
    }
    if (!index_json.contains("metadata") || !index_json["metadata"].is_object()) {
        return ErrorCode::LOAD_ERROR;
    }
    const auto& metadata = index_json["metadata"];
    if (metadata.contains("total_size")) {
        return metadata["total_size"].get<size_t>();
    }
    return ErrorCode::LOAD_ERROR;
}

std::variant<ErrorCode, size_t> ModelWeights::get_actual_size() const {
    size_t actual_size = 0;
    if(!engine_config.enable_pipeline_parallel){
        //read total size from safetensors index json
        auto total_size_or_error = read_total_size(
            engine_config.model_config.model_safetensors_index_json.c_str()
        );
        if (std::holds_alternative<ErrorCode>(total_size_or_error)) {
            LOG_ERROR("Failed to read total size from safetensors index");
            return std::get<ErrorCode>(total_size_or_error);
        }
        {
            std::ostringstream oss;
            oss << "read_total_size success, bytes=" << std::get<size_t>(total_size_or_error);
            LOG_DEBUG(oss.str());
        }
        actual_size = std::get<size_t>(total_size_or_error);
    }else{
        //if pipeline parallel enabled, each rank only loads part of weights
        for (const auto& header_pair : headers) {
            if (std::find(weight_names.begin(), weight_names.end(), header_pair.first) == weight_names.end()) {
                continue;
            }
            const WeightHeader& header = header_pair.second;
            size_t weight_size = header.offset_end - header.offset_start;
            actual_size += weight_size;
        }
        {
            std::ostringstream oss;
            oss << "calculated actual size from headers, bytes=" << actual_size;
            LOG_DEBUG(oss.str());
        }
    }
    return actual_size;
}
    
ErrorCode ModelWeights::init(){
    LOG_DEBUG("ModelWeights::init begin");
    //build weight names list in sequence
    ErrorCode error = build_weight_names(engine_config.model_config.weight_names_path.c_str());
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to build weight names list");
        return error;
    }
    {
        std::ostringstream oss;
        oss << "build_weight_names success, weight_names=" << weight_names.size();
        LOG_DEBUG(oss.str());
    }
    //parse header
    error = parse_header(engine_config.model_config.model_path.c_str());
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to parse model weight header");
        return error;
    }
    {
        std::ostringstream oss;
        oss << "parse_header success, headers=" << headers.size();
        LOG_DEBUG(oss.str());
    }

    auto actual_size_or_error = get_actual_size();
    if (std::holds_alternative<ErrorCode>(actual_size_or_error)) {
        LOG_ERROR("Failed to get actual size");
        return std::get<ErrorCode>(actual_size_or_error);
    }
    size_t allocated_bytes = std::get<size_t>(actual_size_or_error);

    //allocate gpu memory for weights
    cudaError_t cuda_err = cudaMalloc(&weights, allocated_bytes);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate GPU memory for model weights ");
        return ErrorCode::CUDA_FAILURE;
    }
    cuda_err = cudaMemset(weights, 0, allocated_bytes);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to zero-initialize GPU memory for model weights");
        return ErrorCode::CUDA_FAILURE;
    }
    {
        std::ostringstream oss;
        oss << "cudaMalloc success, ptr=" << weights;
        LOG_DEBUG(oss.str());
    }
    //build weight layout
    LOG_DEBUG("calling build_weight_layout");
    layout.weights = weights;

    ModelConfig effective_config = engine_config.model_config;
    DataType effective_dtype = effective_config.data_type;

    auto embed_it = headers.find("model.embed_tokens.weight");
    if (embed_it != headers.end()) {
        effective_dtype = embed_it->second.dtype;
    } else if (!headers.empty()) {
        effective_dtype = headers.begin()->second.dtype;
    }

    if (effective_dtype != engine_config.model_config.data_type) {
        std::ostringstream oss;
        oss << "Model config dtype differs from safetensors dtype, config="
            << static_cast<int>(engine_config.model_config.data_type)
            << ", effective=" << static_cast<int>(effective_dtype)
            << ". Using effective dtype for layout.";
        LOG_INFO(oss.str());
    }

    effective_config.data_type = effective_dtype;
    ErrorCode build_weight_layout_error = layout.build_weight_layout(effective_config);
    if (build_weight_layout_error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to build weight layout");
        return build_weight_layout_error;
    }
    {
        std::ostringstream oss;
        oss << "layout.total_size=" << layout.total_size
            << ", allocated_bytes=" << allocated_bytes;
        LOG_DEBUG(oss.str());
    }
    if (layout.total_size > allocated_bytes) {
        LOG_ERROR("Weight layout exceeds allocated GPU memory; model config likely mismatches real weights");
        return ErrorCode::INVALID_INPUT;
    }
    LOG_DEBUG("build_weight_layout success");

    return ErrorCode::SUCCESS;
}
        
ErrorCode ModelWeights::parse_header(const char* file_name){
    headers.clear();

    const std::vector<std::string> shards = ResolveSafetensorShards(file_name);
    if (shards.empty()) {
        LOG_ERROR("Failed to resolve safetensors shards from path:");
        return ErrorCode::LOAD_ERROR;
    }
    {
        std::ostringstream oss;
        oss << "parse_header shard_count=" << shards.size();
        LOG_DEBUG(oss.str());
    }

    size_t layer_idx = 0;

    for (const auto& shard : shards) {
        std::ifstream infile(shard, std::ios::binary);
        if (!infile.is_open()) {
            LOG_ERROR("Failed to open model weight file shard: ");
            return ErrorCode::LOAD_ERROR;
        }

        uint64_t header_size = 0;
        infile.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));
        if (!infile || header_size == 0) {
            LOG_ERROR("Failed to read safetensors header size from shard");
            return ErrorCode::LOAD_ERROR;
        }
        {
            std::ostringstream oss;
            oss << "parse_header shard=" << shard << ", header_size=" << header_size;
            LOG_DEBUG(oss.str());
        }

        std::vector<char> header_data(header_size);
        infile.read(header_data.data(), static_cast<std::streamsize>(header_size));
        if (!infile) {
            LOG_ERROR("Failed to read safetensors header data from shard");
            return ErrorCode::LOAD_ERROR;
        }

        json header_json = json::parse(header_data.data(), header_data.data() + header_size);
        for (const auto& item : header_json.items()) {
            std::string name = item.key();
            const auto& value = item.value();

            if (name == "__metadata__") {
                continue;
            }
            if (!value.is_object() || !value.contains("dtype") || !value.contains("shape") ||
                !value.contains("data_offsets") || !value["data_offsets"].is_array() ||
                value["data_offsets"].size() != 2) {
                LOG_ERROR("Malformed tensor entry in safetensors header");
                return ErrorCode::LOAD_ERROR;
            }

            std::string dtype = value["dtype"].get<std::string>();
            std::vector<int> shape = value["shape"].get<std::vector<int>>();

            size_t offset_start = value["data_offsets"][0];
            size_t offset_end = value["data_offsets"][1];

            DataType parsed_dtype;
            if (dtype == "fp16" || dtype == "F16") {
                parsed_dtype = DataType::FLOAT16;
            } else if (dtype == "bf16" || dtype == "BF16") {
                parsed_dtype = DataType::BF16;
            } else if (dtype == "fp32" || dtype == "F32") {
                parsed_dtype = DataType::FLOAT32;
            } else {
                LOG_ERROR("Unsupported tensor dtype in safetensors header");
                return ErrorCode::LOAD_ERROR;
            }

            WeightHeader header = {
                layer_idx,
                shape,
                name,
                shard,
                offset_start + 8 + header_size,
                offset_end + 8 + header_size,
                parsed_dtype,
                false
            };
            headers[name] = header;
            layer_idx++;
        }
    }

    {
        std::ostringstream oss;
        oss << "parse_header finished, total headers=" << headers.size();
        LOG_DEBUG(oss.str());
    }

    return ErrorCode::SUCCESS;
}

//load to cpu
Tensor ModelWeights::load_layer(std::ifstream& file, const std::string& name) {
    if(headers.find(name) == headers.end()){
        LOG_ERROR("Weight name not found in header");
        return Tensor();
    }
    const WeightHeader& header = headers[name];
    size_t weight_size = (header.offset_end - header.offset_start);
    std::vector<size_t> shape;
    shape.reserve(header.shape.size());
    for (int dim : header.shape) {
        shape.push_back(static_cast<size_t>(dim));
    }
    Tensor layer_tensor(
        weight_size / Tensor::element_size_bytes(header.dtype),
        nullptr,
        shape,
        header.dtype,
        "cpu"
    );
    layer_tensor.data = malloc(weight_size);
    if (layer_tensor.data == nullptr) {
        LOG_ERROR("Failed to allocate host buffer for layer");
        return Tensor();
    }
    file.seekg(header.offset_start);
    file.read((char*)layer_tensor.data, weight_size);
    if (!file) {
        LOG_ERROR("Failed to read layer tensor bytes");
        free(layer_tensor.data);
        layer_tensor.data = nullptr;
        return Tensor();
    }

    return layer_tensor;

}
ErrorCode ModelWeights::build_weight_names(const char* file_name){
    std::ifstream infile(file_name);
    if(!infile.is_open()){
        LOG_ERROR("Failed to open model weight file");
        return ErrorCode::LOAD_ERROR;
    }

    weight_names.clear();
    size_t total_weight_names = 0;
    size_t filtered_out_weight_names = 0;
    std::string line;
    while (std::getline(infile, line)) {
        const size_t begin = line.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos) {
            continue;
        }
        const size_t end = line.find_last_not_of(" \t\r\n");
        const std::string name = line.substr(begin, end - begin + 1);
        if (name.empty()) {
            continue;
        }

        total_weight_names++;
        // if pipeline parallel, filter out weights that do not belong to current stage
        if (engine_config.enable_pipeline_parallel) {
            const size_t start_layer = engine_config.stage_start_layer;
            const size_t end_layer = engine_config.stage_end_layer;
            const size_t num_layers = engine_config.model_config.num_hidden_layers;

            bool keep = false;
            if (name == "model.embed_tokens.weight") {
                keep = engine_config.is_first_stage();
            } else if (name == "model.norm.weight" || name == "lm_head.weight") {
                keep = engine_config.is_last_stage();
            } else {
                const std::string marker = "model.layers.";
                const size_t marker_pos = name.find(marker);
                if (marker_pos != std::string::npos) {
                    const size_t id_begin = marker_pos + marker.size();
                    const size_t id_end = name.find('.', id_begin);
                    if (id_end != std::string::npos && id_end > id_begin) {
                        const std::string layer_str = name.substr(id_begin, id_end - id_begin);
                        if (!layer_str.empty()) {
                            const size_t layer_id = static_cast<size_t>(std::stoull(layer_str));
                            keep = (layer_id >= start_layer && layer_id < end_layer);
                        }
                    }
                }
            }

            if (!keep) {
                filtered_out_weight_names++;
                continue;
            }
        }

        weight_names.push_back(name);
    }

    {
        std::ostringstream oss;
        oss << "build_weight_names loaded=" << weight_names.size()
            << ", total=" << total_weight_names
            << ", filtered_out=" << filtered_out_weight_names
            << ", pipeline=" << (engine_config.enable_pipeline_parallel ? "true" : "false")
            << ", stage=[" << engine_config.stage_start_layer << ", " << engine_config.stage_end_layer << ")";
        LOG_DEBUG(oss.str());
    }

    return ErrorCode::SUCCESS;
}
Tensor ModelWeights::concat_qkv_bias(const Tensor& q_bias, const Tensor& k_bias, const Tensor& v_bias){
    if (q_bias.shape.size() != 1 || k_bias.shape.size() != 1 || v_bias.shape.size() != 1) {
        LOG_ERROR("concat_qkv_bias expects 1D tensors");
        return Tensor();
    }
    if (q_bias.dtype != k_bias.dtype || q_bias.dtype != v_bias.dtype) {
        LOG_ERROR("concat_qkv_bias dtype mismatch");
        return Tensor();
    }

    const size_t q_size = q_bias.shape[0];
    const size_t k_size = k_bias.shape[0];
    const size_t v_size = v_bias.shape[0];

    const size_t elem_size = Tensor::element_size_bytes(q_bias.dtype);
    const size_t total_size = (q_size + k_size + v_size) * elem_size;

    Tensor out(
        q_size + k_size + v_size,
        nullptr, 
        {q_size + k_size + v_size}, 
        q_bias.dtype,
        "cpu"
    );

    char* data = new char[out.size];
    out.data = data;

    const char* q = static_cast<const char*>(q_bias.data);
    const char* k = static_cast<const char*>(k_bias.data);
    const char* v = static_cast<const char*>(v_bias.data);

    std::memcpy(data, q, q_size * elem_size);
    std::memcpy(data + q_size * elem_size, k, k_size * elem_size);
    std::memcpy(data + (q_size + k_size) * elem_size, v, v_size * elem_size);

    return out;
}
//concat qkv on cpu
Tensor ModelWeights::concat_qkv(const Tensor& Wq, const Tensor& Wk, const Tensor& Wv){
    if (Wq.shape.size() != 2 || Wk.shape.size() != 2 || Wv.shape.size() != 2) {
        LOG_ERROR("concat_qkv expects 2D tensors");
        return Tensor();
    }
    if (Wq.dtype != Wk.dtype || Wq.dtype != Wv.dtype) {
        LOG_ERROR("concat_qkv dtype mismatch");
        return Tensor();
    }

    // HF/Qwen linear weights are [out_features, in_features].
    // Q/K/V must share in_features and be concatenated on out_features (row dimension),
    // then transposes to [in_features, out_features_total].
    const size_t q_rows = Wq.shape[0];
    const size_t k_rows = Wk.shape[0];
    const size_t v_rows = Wv.shape[0];
    const size_t in_cols = Wq.shape[1];
    if (Wk.shape[1] != in_cols || Wv.shape[1] != in_cols) {
        LOG_ERROR("concat_qkv shape mismatch on input feature dimension");
        return Tensor();
    }

    const size_t elem_size = Tensor::element_size_bytes(Wq.dtype);
    const size_t out_rows = q_rows + k_rows + v_rows;

    Tensor out(
        out_rows * in_cols,
        nullptr, 
        {out_rows, in_cols}, 
        Wq.dtype,
        "cpu"
    );

    char* data = new char[out.size];
    out.data = data;

    const char* q = static_cast<const char*>(Wq.data);
    const char* k = static_cast<const char*>(Wk.data);
    const char* v = static_cast<const char*>(Wv.data);

    const size_t q_bytes = q_rows * in_cols * elem_size;
    const size_t k_bytes = k_rows * in_cols * elem_size;
    const size_t v_bytes = v_rows * in_cols * elem_size;

    std::memcpy(data, q, q_bytes);
    std::memcpy(data + q_bytes, k, k_bytes);
    std::memcpy(data + q_bytes + k_bytes, v, v_bytes);

    return out;
}

//copy from cpu to gpu
ErrorCode ModelWeights::load_weights(const char* weight_path) {
    (void)weight_path;
    LOG_DEBUG("load_weights begin");

    Tensor tmp_layer_tensor_q;
    Tensor tmp_layer_tensor_k;
    Tensor tmp_layer_tensor_v;
    Tensor tmp_layer_tensor_qbias;
    Tensor tmp_layer_tensor_kbias;
    Tensor tmp_layer_tensor_vbias;

    std::unordered_map<std::string, std::unique_ptr<std::ifstream>> shard_streams;

    auto get_stream_for_name = [&](const std::string& name) -> std::ifstream* {
        const auto hit = headers.find(name);
        if (hit == headers.end()) {
            return nullptr;
        }
        const std::string& shard = hit->second.shard_file;
        auto stream_it = shard_streams.find(shard);
        if (stream_it == shard_streams.end()) {
            auto stream = std::make_unique<std::ifstream>(shard, std::ios::binary);
            if (!stream->is_open()) {
                LOG_ERROR("Failed to open shard file for tensor ");
                return nullptr;
            }
            stream_it = shard_streams.emplace(shard, std::move(stream)).first;
        }
        return stream_it->second.get();
    };
    
    bool has_q = false;
    bool has_k = false;
    bool has_v = false;
    bool has_q_bias = false;
    bool has_k_bias = false;
    bool has_v_bias = false;
    size_t idx = 0;
    for(const auto& name : weight_names){
        {
            std::ostringstream oss;
            oss << "load_weights idx=" << idx << ", name=" << name;
            LOG_DEBUG(oss.str());
        }
        ++idx;

        std::ifstream* stream = get_stream_for_name(name);
        if (stream == nullptr) {
            const bool is_bias_name = name.find(".bias") != std::string::npos;
            if (is_bias_name) {
                LOG_INFO("Skipping missing optional bias tensor from weight list: " + name);
                continue;
            }
            LOG_ERROR("load_weights stream resolve failed");
            return ErrorCode::LOAD_ERROR;
        }

        if(name.find("embed_tokens") != std::string::npos){
            Tensor tmp_layer_tensor = load_layer(*stream, name);
            if(tmp_layer_tensor.data == nullptr){
                return ErrorCode::LOAD_ERROR;
            }
            if (tmp_layer_tensor.size != layout.embedding_weights.size) {
                std::ostringstream oss;
                oss << "embed_tokens tensor size mismatch, src=" << tmp_layer_tensor.size
                    << ", dst=" << layout.embedding_weights.size;
                LOG_ERROR(oss.str());
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
                return ErrorCode::INVALID_INPUT;
            }
            cudaError_t copy_err = cudaMemcpy(
                layout.embedding_weights.data,
                tmp_layer_tensor.data, 
                layout.embedding_weights.size, 
                cudaMemcpyHostToDevice
            );
            if (copy_err != cudaSuccess) {
                LOG_ERROR("cudaMemcpy embed_tokens failed");
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
                return ErrorCode::CUDA_FAILURE;
            }
            free(tmp_layer_tensor.data);
            tmp_layer_tensor.data = nullptr;
            log_tensor_anomaly(layout.embedding_weights, name);
        } else if(name.find("layers.") != std::string::npos){
            size_t pos1 = name.find("layers.") + 7;
            size_t pos2 = name.find(".", pos1);
            size_t layer_id = std::stoi(name.substr(pos1, pos2 - pos1));

            auto layer_layout = layout.get_layer_layout<TransformerLayerWeightLayout>(layer_id);
            if (!layer_layout) {
                LOG_ERROR("Transformer layer layout missing");
                continue;
            }
            if(name.find("q_proj.weight") != std::string::npos){
                if (tmp_layer_tensor_q.data != nullptr) {
                    free(tmp_layer_tensor_q.data);
                    tmp_layer_tensor_q.data = nullptr;
                }
                Tensor loaded_q = load_layer(*stream, name);
                if(loaded_q.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                tmp_layer_tensor_q = std::move(loaded_q);
                has_q = true;
            } else if(name.find("q_proj.bias") != std::string::npos){
                if(tmp_layer_tensor_qbias.data != nullptr){
                    free(tmp_layer_tensor_qbias.data);
                    tmp_layer_tensor_qbias.data = nullptr;
                }
                Tensor loaded_qbias = load_layer(*stream, name);
                if(loaded_qbias.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                tmp_layer_tensor_qbias = std::move(loaded_qbias);
                has_q_bias = true;
            } else if(name.find("k_proj.weight") != std::string::npos){
                if (tmp_layer_tensor_k.data != nullptr) {
                    free(tmp_layer_tensor_k.data);
                    tmp_layer_tensor_k.data = nullptr;
                }
                Tensor loaded_k = load_layer(*stream, name);
                if(loaded_k.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                tmp_layer_tensor_k = std::move(loaded_k);
                has_k = true;
            } else if(name.find("k_proj.bias") != std::string::npos){
                if(tmp_layer_tensor_kbias.data != nullptr){
                    free(tmp_layer_tensor_kbias.data);
                    tmp_layer_tensor_kbias.data = nullptr;
                }
                Tensor loaded_kbias = load_layer(*stream, name);
                if(loaded_kbias.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                tmp_layer_tensor_kbias = std::move(loaded_kbias);
                has_k_bias = true;
            } else if(name.find("v_proj.weight") != std::string::npos){
                if (tmp_layer_tensor_v.data != nullptr) {
                    free(tmp_layer_tensor_v.data);
                    tmp_layer_tensor_v.data = nullptr;
                }
                Tensor loaded_v = load_layer(*stream, name);
                if(loaded_v.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                tmp_layer_tensor_v = std::move(loaded_v);
                has_v = true;
            } else if(name.find("v_proj.bias") != std::string::npos){
                if(tmp_layer_tensor_vbias.data != nullptr){
                    free(tmp_layer_tensor_vbias.data);
                    tmp_layer_tensor_vbias.data = nullptr;
                }
                Tensor loaded_vbias = load_layer(*stream, name);
                if(loaded_vbias.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                tmp_layer_tensor_vbias = std::move(loaded_vbias);
                has_v_bias = true;
            } else if(name.find("o_proj.weight") != std::string::npos){
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                if (tmp_layer_tensor.size != layer_layout->attention_weights.o_proj_weight.size) {
                    std::ostringstream oss;
                    oss << "o_proj tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << layer_layout->attention_weights.o_proj_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                Tensor Wo_proj_tans = tmp_layer_tensor.transpose();
                if (Wo_proj_tans.data == nullptr) {
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::LOAD_ERROR;
                }
                cudaError_t copy_err = cudaMemcpy(
                    layer_layout->attention_weights.o_proj_weight.data,
                    Wo_proj_tans.data,
                    Wo_proj_tans.size,
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy o_proj failed");
                    delete[] static_cast<char*>(Wo_proj_tans.data);
                    Wo_proj_tans.data = nullptr;
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                delete[] static_cast<char*>(Wo_proj_tans.data);
                Wo_proj_tans.data = nullptr;
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
                log_tensor_anomaly(layer_layout->attention_weights.o_proj_weight, name);

            } else if(name.find("gate_proj") != std::string::npos){
                if (layer_layout->mlp_weights.mlp_linears_weight.size() < 3) {
                    LOG_ERROR("MLP linear layouts are insufficient");
                    return ErrorCode::LOAD_ERROR;
                }
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                if (tmp_layer_tensor.size != layer_layout->mlp_weights.mlp_linears_weight[0].linear_weight.size) {
                    std::ostringstream oss;
                    oss << "gate_proj tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << layer_layout->mlp_weights.mlp_linears_weight[0].linear_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                Tensor Wgate_proj_tans = tmp_layer_tensor.transpose();
                if (Wgate_proj_tans.data == nullptr) {
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::LOAD_ERROR;
                }
                cudaError_t copy_err = cudaMemcpy(
                    layer_layout->mlp_weights.mlp_linears_weight[0].linear_weight.data,
                    Wgate_proj_tans.data,
                    Wgate_proj_tans.size,
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy gate_proj failed");
                    delete[] static_cast<char*>(Wgate_proj_tans.data);
                    Wgate_proj_tans.data = nullptr;
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                delete[] static_cast<char*>(Wgate_proj_tans.data);
                Wgate_proj_tans.data = nullptr;
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
                log_tensor_anomaly(layer_layout->mlp_weights.mlp_linears_weight[0].linear_weight, name);
            } else if(name.find("up_proj") != std::string::npos){
                if (layer_layout->mlp_weights.mlp_linears_weight.size() < 3) {
                    LOG_ERROR("MLP linear layouts are insufficient");
                    return ErrorCode::LOAD_ERROR;
                }
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                if (tmp_layer_tensor.size != layer_layout->mlp_weights.mlp_linears_weight[1].linear_weight.size) {
                    std::ostringstream oss;
                    oss << "up_proj tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << layer_layout->mlp_weights.mlp_linears_weight[1].linear_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                Tensor Wup_proj_tans = tmp_layer_tensor.transpose();
                if (Wup_proj_tans.data == nullptr) {
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::LOAD_ERROR;
                }
                cudaError_t copy_err = cudaMemcpy(
                    layer_layout->mlp_weights.mlp_linears_weight[1].linear_weight.data,
                    Wup_proj_tans.data,
                    Wup_proj_tans.size,
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy up_proj failed");
                    delete[] static_cast<char*>(Wup_proj_tans.data);
                    Wup_proj_tans.data = nullptr;
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                delete[] static_cast<char*>(Wup_proj_tans.data);
                Wup_proj_tans.data = nullptr;
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
                log_tensor_anomaly(layer_layout->mlp_weights.mlp_linears_weight[1].linear_weight, name);
            } else if(name.find("down_proj") != std::string::npos){
                if (layer_layout->mlp_weights.mlp_linears_weight.size() < 3) {
                    LOG_ERROR("MLP linear layouts are insufficient");
                    return ErrorCode::LOAD_ERROR;
                }
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                if (tmp_layer_tensor.size != layer_layout->mlp_weights.mlp_linears_weight[2].linear_weight.size) {
                    std::ostringstream oss;
                    oss << "down_proj tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << layer_layout->mlp_weights.mlp_linears_weight[2].linear_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                Tensor Wdown_proj_tans = tmp_layer_tensor.transpose();
                if (Wdown_proj_tans.data == nullptr) {
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::LOAD_ERROR;
                }
                cudaError_t copy_err = cudaMemcpy(
                    layer_layout->mlp_weights.mlp_linears_weight[2].linear_weight.data,
                    Wdown_proj_tans.data,
                    Wdown_proj_tans.size,
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy down_proj failed");
                    delete[] static_cast<char*>(Wdown_proj_tans.data);
                    Wdown_proj_tans.data = nullptr;
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                delete[] static_cast<char*>(Wdown_proj_tans.data);
                Wdown_proj_tans.data = nullptr;
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
                log_tensor_anomaly(layer_layout->mlp_weights.mlp_linears_weight[2].linear_weight, name);
            } else if(name.find("input_layernorm") != std::string::npos){
                if (layer_layout->norm_weights.size() < 2) {
                    LOG_ERROR("Norm layouts are insufficient");
                    return ErrorCode::LOAD_ERROR;
                }
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                if (tmp_layer_tensor.size != layer_layout->norm_weights[0].norm_weight.size) {
                    std::ostringstream oss;
                    oss << "input_layernorm tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << layer_layout->norm_weights[0].norm_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                cudaError_t copy_err = cudaMemcpy(
                    layer_layout->norm_weights[0].norm_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy input_layernorm failed");
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
                log_tensor_anomaly(layer_layout->norm_weights[0].norm_weight, name);
            } else if(name.find("post_attention_layernorm") != std::string::npos){
                if (layer_layout->norm_weights.size() < 2) {
                    LOG_ERROR("Norm layouts are insufficient");
                    return ErrorCode::LOAD_ERROR;
                }
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                if (tmp_layer_tensor.size != layer_layout->norm_weights[1].norm_weight.size) {
                    std::ostringstream oss;
                    oss << "post_attention_layernorm tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << layer_layout->norm_weights[1].norm_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                cudaError_t copy_err = cudaMemcpy(
                    layer_layout->norm_weights[1].norm_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy post_attention_layernorm failed");
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
                log_tensor_anomaly(layer_layout->norm_weights[1].norm_weight, name);
            }else{
                LOG_ERROR("Unrecognized weight name, weights may be incomplete");
                continue;
            }
            if (has_q_bias && has_k_bias && has_v_bias) {
                Tensor bias_qkv = concat_qkv_bias(tmp_layer_tensor_qbias, tmp_layer_tensor_kbias, tmp_layer_tensor_vbias);
                if (bias_qkv.data == nullptr) {
                    LOG_ERROR("concat_qkv_bias failed");
                    free(tmp_layer_tensor_qbias.data);
                    free(tmp_layer_tensor_kbias.data);
                    free(tmp_layer_tensor_vbias.data);
                    tmp_layer_tensor_qbias.data = nullptr;
                    tmp_layer_tensor_kbias.data = nullptr;
                    tmp_layer_tensor_vbias.data = nullptr;
                    return ErrorCode::LOAD_ERROR;
                }
                cudaError_t copy_bias_err = cudaMemcpy(
                    layer_layout->attention_weights.qkv_proj_bias.data,
                    bias_qkv.data,
                    bias_qkv.size,
                    cudaMemcpyHostToDevice
                );
                if (copy_bias_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy qkv_proj_bias failed");
                    delete[] static_cast<char*>(bias_qkv.data);
                    bias_qkv.data = nullptr;
                    free(tmp_layer_tensor_qbias.data);
                    free(tmp_layer_tensor_kbias.data);
                    free(tmp_layer_tensor_vbias.data);
                    tmp_layer_tensor_qbias.data = nullptr;
                    tmp_layer_tensor_kbias.data = nullptr;
                    tmp_layer_tensor_vbias.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                delete[] static_cast<char*>(bias_qkv.data);
                bias_qkv.data = nullptr;
                free(tmp_layer_tensor_qbias.data);
                free(tmp_layer_tensor_kbias.data);
                free(tmp_layer_tensor_vbias.data);
                tmp_layer_tensor_qbias.data = nullptr;
                tmp_layer_tensor_kbias.data = nullptr;
                tmp_layer_tensor_vbias.data = nullptr;
                has_q_bias = false;
                has_k_bias = false;
                has_v_bias = false;
                log_tensor_anomaly(layer_layout->attention_weights.qkv_proj_bias, std::string("qkv_proj_bias"));
            }

            if (!(has_q && has_k && has_v)) {
                continue;
            }

            //concat Wq, Wk, Wv
            Tensor Wqkv = concat_qkv(tmp_layer_tensor_q, tmp_layer_tensor_k, tmp_layer_tensor_v);
            if (Wqkv.data == nullptr) {
                LOG_ERROR("concat_qkv failed");
                free(tmp_layer_tensor_q.data);
                free(tmp_layer_tensor_k.data);
                free(tmp_layer_tensor_v.data);
                tmp_layer_tensor_q.data = nullptr;
                tmp_layer_tensor_k.data = nullptr;
                tmp_layer_tensor_v.data = nullptr;
                return ErrorCode::LOAD_ERROR;
            }
            //transpose
            Tensor Wqkv_trans = Wqkv.transpose();
            if (Wqkv_trans.data == nullptr) {
                LOG_ERROR("transpose qkv failed");
                delete[] static_cast<char*>(Wqkv.data);
                Wqkv.data = nullptr;
                free(tmp_layer_tensor_q.data);
                free(tmp_layer_tensor_k.data);
                free(tmp_layer_tensor_v.data);
                tmp_layer_tensor_q.data = nullptr;
                tmp_layer_tensor_k.data = nullptr;
                tmp_layer_tensor_v.data = nullptr;
                return ErrorCode::LOAD_ERROR;
            }
            if (Wqkv_trans.size != layer_layout->attention_weights.qkv_proj_weight.size) {
                std::ostringstream oss;
                oss << "qkv_proj tensor size mismatch after concat/transpose, src=" << Wqkv_trans.size
                    << ", dst=" << layer_layout->attention_weights.qkv_proj_weight.size
                    << ", q_shape=[" << tmp_layer_tensor_q.shape[0] << "," << tmp_layer_tensor_q.shape[1] << "]"
                    << ", k_shape=[" << tmp_layer_tensor_k.shape[0] << "," << tmp_layer_tensor_k.shape[1] << "]"
                    << ", v_shape=[" << tmp_layer_tensor_v.shape[0] << "," << tmp_layer_tensor_v.shape[1] << "]"
                    << ", cfg_layer_id=" << layer_id;
                LOG_ERROR(oss.str());
                delete[] static_cast<char*>(Wqkv_trans.data);
                Wqkv_trans.data = nullptr;
                delete[] static_cast<char*>(Wqkv.data);
                Wqkv.data = nullptr;
                free(tmp_layer_tensor_q.data);
                free(tmp_layer_tensor_k.data);
                free(tmp_layer_tensor_v.data);
                tmp_layer_tensor_q.data = nullptr;
                tmp_layer_tensor_k.data = nullptr;
                tmp_layer_tensor_v.data = nullptr;
                return ErrorCode::INVALID_INPUT;
            }
            //copy from cpu to gpu
            cudaError_t copy_err = cudaMemcpy(
                layer_layout->attention_weights.qkv_proj_weight.data,
                Wqkv_trans.data,
                Wqkv_trans.size,
                cudaMemcpyHostToDevice
            );
            if (copy_err != cudaSuccess) {
                LOG_ERROR("cudaMemcpy qkv_proj_weight failed");
                delete[] static_cast<char*>(Wqkv_trans.data);
                Wqkv_trans.data = nullptr;
                delete[] static_cast<char*>(Wqkv.data);
                Wqkv.data = nullptr;
                free(tmp_layer_tensor_q.data);
                free(tmp_layer_tensor_k.data);
                free(tmp_layer_tensor_v.data);
                tmp_layer_tensor_q.data = nullptr;
                tmp_layer_tensor_k.data = nullptr;
                tmp_layer_tensor_v.data = nullptr;
                return ErrorCode::CUDA_FAILURE;
            }
            delete[] static_cast<char*>(Wqkv_trans.data);
            Wqkv_trans.data = nullptr;
            delete[] static_cast<char*>(Wqkv.data);
            Wqkv.data = nullptr;

            free(tmp_layer_tensor_q.data);
            free(tmp_layer_tensor_k.data);
            free(tmp_layer_tensor_v.data);
            tmp_layer_tensor_q.data = nullptr;
            tmp_layer_tensor_k.data = nullptr;
            tmp_layer_tensor_v.data = nullptr;

            has_k = false;
            has_q = false;  
            has_v = false;
            log_tensor_anomaly(layer_layout->attention_weights.qkv_proj_weight, std::string("qkv_proj_weight"));

        } else if(name.find("model.norm") != std::string::npos){
            Tensor tmp_layer_tensor = load_layer(*stream, name);
            if(tmp_layer_tensor.data == nullptr){
                return ErrorCode::LOAD_ERROR;
            }
            std::shared_ptr<LayerNormLayerWeightLayout> final_norm_layout;
            for (size_t i = 0; i < layout.layer_weights.size(); ++i) {
                auto candidate = layout.get_layer_layout<LayerNormLayerWeightLayout>(i);
                if (candidate) {
                    final_norm_layout = candidate;
                }
            }
            if (final_norm_layout) {
                if (tmp_layer_tensor.size != final_norm_layout->norm_weight.size) {
                    std::ostringstream oss;
                    oss << "model.norm tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << final_norm_layout->norm_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                cudaError_t copy_err = cudaMemcpy(
                    final_norm_layout->norm_weight.data,
                    tmp_layer_tensor.data,
                    final_norm_layout->norm_weight.size,
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy model.norm failed");
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
            }
            free(tmp_layer_tensor.data);
            tmp_layer_tensor.data = nullptr;
            log_tensor_anomaly(final_norm_layout->norm_weight, name);
        } else if(name.find("lm_head") != std::string::npos){
            Tensor tmp_layer_tensor = load_layer(*stream, name);
            if(tmp_layer_tensor.data == nullptr){
                return ErrorCode::LOAD_ERROR;
            }
            std::shared_ptr<LinearLayerWeightLayout> lm_head_layout;
            for (size_t i = 0; i < layout.layer_weights.size(); ++i) {
                auto candidate = layout.get_layer_layout<LinearLayerWeightLayout>(i);
                if (candidate) {
                    lm_head_layout = candidate;
                }
            }
            if(lm_head_layout){
                if (tmp_layer_tensor.size != lm_head_layout->linear_weight.size) {
                    std::ostringstream oss;
                    oss << "lm_head tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << lm_head_layout->linear_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                Tensor lm_head_tans = tmp_layer_tensor.transpose();
                if (lm_head_tans.data == nullptr) {
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::LOAD_ERROR;
                }
                cudaError_t copy_err = cudaMemcpy(
                    lm_head_layout->linear_weight.data,
                    lm_head_tans.data,
                    lm_head_tans.size,
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy lm_head failed");
                    delete[] static_cast<char*>(lm_head_tans.data);
                    lm_head_tans.data = nullptr;
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                delete[] static_cast<char*>(lm_head_tans.data);
                lm_head_tans.data = nullptr;
            }
            free(tmp_layer_tensor.data);
            tmp_layer_tensor.data = nullptr;
            log_tensor_anomaly(lm_head_layout->linear_weight, name);
        } else {
            LOG_ERROR("Unrecognized weight name, weights may be incomplete");
            continue;
        }
        
    }

    if (tmp_layer_tensor_q.data) {
        free(tmp_layer_tensor_q.data);
    }
    if (tmp_layer_tensor_k.data) {
        free(tmp_layer_tensor_k.data);
    }
    if (tmp_layer_tensor_v.data) {
        free(tmp_layer_tensor_v.data);
    }

    LOG_DEBUG("load_weights finished");
    
    return ErrorCode::SUCCESS;
}