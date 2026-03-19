#include "ModelWeights.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <unordered_set>
#include <unordered_map>
#include "include/error.h"
#include "include/utils/logger.h"

namespace {

std::vector<std::string> ResolveSafetensorShardsFromIndex(
    const std::filesystem::path& index_path) {

    namespace fs = std::filesystem;
    std::vector<std::string> shards;
    if (!fs::exists(index_path)) {
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
        const fs::path shard_path = index_path.parent_path() / rel_name;
        const std::string shard = shard_path.string();
        if (seen.insert(shard).second) {
            if (!fs::exists(shard_path)) {
                return {};
            }
            shards.push_back(shard);
        }
    }

    return shards;
}

std::vector<std::string> ResolveSafetensorShards(const std::string& model_path) {
    namespace fs = std::filesystem;
    fs::path p(model_path);
    if (!fs::exists(p)) {
        return {};
    }

    const fs::path base_dir = fs::is_directory(p) ? p : p.parent_path();
    const fs::path index_path = base_dir / "model.safetensors.index.json";
    return ResolveSafetensorShardsFromIndex(index_path);
}

} // namespace


ErrorCode WeightLayout::build_weight_layout(const ModelConfig& config) {
    size_t offset = 0;
    layer_weights.clear();
    layer_weights.reserve(config.layer_configs.size());
    //embedding weights
    embedding_weights = Tensor(
        config.vocab_size * config.hidden_size,
        static_cast<void*>(static_cast<char*>(weights) + offset),
        {config.vocab_size, config.hidden_size},
        DTYPE
    );
    offset += embedding_weights.size;
    //transformer layers
    for (const auto& layer_cfg_base : config.layer_configs) {
        if (auto transformer_cfg = std::dynamic_pointer_cast<TransformerLayerConfig>(layer_cfg_base)) {
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
                DTYPE
            );
            offset += transformer_layout->attention_weights.qkv_proj_weight.size;
            //attention output projection weights
            transformer_layout->attention_weights.o_proj_weight = Tensor(
                q_hidden * config.hidden_size,
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {q_hidden, config.hidden_size},
                DTYPE
            );
            offset += transformer_layout->attention_weights.o_proj_weight.size;
            //layer norm weights
            transformer_layout->norm_weights.resize(2);
            transformer_layout->norm_weights[0] = LayerNormLayerWeightLayout();
            transformer_layout->norm_weights[1] = LayerNormLayerWeightLayout();
            transformer_layout->norm_weights[0].norm_weight = Tensor(
                config.hidden_size,
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {config.hidden_size},
                DTYPE
            );
            offset += transformer_layout->norm_weights[0].norm_weight.size;
            transformer_layout->norm_weights[1].norm_weight = Tensor(
                config.hidden_size,
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {config.hidden_size},
                DTYPE
            );
            offset += transformer_layout->norm_weights[1].norm_weight.size;


            //mlp weights
            size_t intermediate_size = transformer_cfg->mlp_config.intermediate_size;
            if (intermediate_size == 0 && !transformer_cfg->mlp_config.mlp_linears.empty()) {
                intermediate_size = transformer_cfg->mlp_config.mlp_linears[0].out_features;
            }
            for(const auto& linear_cfg : transformer_cfg->mlp_config.mlp_linears){
                LinearLayerWeightLayout linear_layout;
                linear_layout.linear_weight = Tensor(
                    linear_cfg.in_features * linear_cfg.out_features,
                    static_cast<void*>(static_cast<char*>(weights) + offset),
                    {linear_cfg.in_features, linear_cfg.out_features},
                    DTYPE
                );
                offset += linear_layout.linear_weight.size;
                if (transformer_cfg->mlp_config.has_bias) {
                    linear_layout.linear_bias = Tensor(
                        linear_cfg.out_features,
                        static_cast<void*>(static_cast<char*>(weights) + offset),
                        {linear_cfg.out_features},
                        DTYPE
                    );
                    offset += linear_layout.linear_bias.size;
                }
                transformer_layout->mlp_weights.mlp_linears_weight.push_back(linear_layout);
            }

            layer_weights.push_back(transformer_layout);
            continue;
        }

        if (auto linear_cfg = std::dynamic_pointer_cast<LinearLayerConfig>(layer_cfg_base)) {
            auto linear_layout = std::make_shared<LinearLayerWeightLayout>();
            const size_t in_features = linear_cfg->linear_config.in_features;
            const size_t out_features = linear_cfg->linear_config.out_features;
            linear_layout->linear_weight = Tensor(
                in_features * out_features, 
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {in_features, out_features}, 
                DTYPE
            );
            offset += linear_layout->linear_weight.size;
            layer_weights.push_back(linear_layout);
            continue;
        }

        if (auto norm_cfg = std::dynamic_pointer_cast<LayerNormLayerConfig>(layer_cfg_base)) {
            auto norm_layout = std::make_shared<LayerNormLayerWeightLayout>();
            const size_t norm_size = norm_cfg->norm_size == 0 ? config.hidden_size : norm_cfg->norm_size;
            norm_layout->norm_weight = Tensor(
                norm_size, 
                static_cast<void*>(static_cast<char*>(weights) + offset), 
                {norm_size}, 
                DTYPE
            );
            offset += norm_layout->norm_weight.size;
            layer_weights.push_back(norm_layout);
            continue;
        }
    }
    return ErrorCode::SUCCESS;
}

std::variant<ErrorCode, size_t> ModelWeights::read_total_size(const char* model_safetensors_index_json) {
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
    
ErrorCode ModelWeights::init(const ModelConfig& config){

    //parse header
    ErrorCode error = parse_header(config.model_path.c_str());
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to parse model weight header");
        return error;
    }
    //build weight names list in sequence
    error = build_weight_names(config.weight_names_path.c_str());
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to build weight names list");
        return error;
    }
    //read total size from safetensors index json
    auto total_size_or_error = read_total_size(
        config.model_safetensors_index_json.c_str()
    );
    if (std::holds_alternative<ErrorCode>(total_size_or_error)) {
        LOG_ERROR("Failed to read total size from safetensors index");
        return std::get<ErrorCode>(total_size_or_error);
    }
    //allocate gpu memory for weights
    cudaError_t cuda_err = cudaMalloc(&weights, std::get<size_t>(total_size_or_error));
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate GPU memory for model weights: %s", cudaGetErrorString(cuda_err));
        return ErrorCode::CUDA_FAILURE;
    }
    //build weight layout
    layout.weights = weights;
    ErrorCode build_weight_layout_error = layout.build_weight_layout(config);
    if (build_weight_layout_error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to build weight layout");
        return build_weight_layout_error;
    }

    return ErrorCode::SUCCESS;
}
        
ErrorCode ModelWeights::parse_header(const char* file_name){
    headers.clear();

    const std::vector<std::string> shards = ResolveSafetensorShards(file_name);
    if (shards.empty()) {
        LOG_ERROR("Failed to resolve safetensors shards from path: %s", file_name);
        return ErrorCode::LOAD_ERROR;
    }

    size_t layer_idx = 0;

    for (const auto& shard : shards) {
        std::ifstream infile(shard, std::ios::binary);
        if (!infile.is_open()) {
            LOG_ERROR("Failed to open model weight file shard: %s", shard.c_str());
            return ErrorCode::LOAD_ERROR;
        }

        uint64_t header_size = 0;
        infile.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));
        if (!infile || header_size == 0) {
            LOG_ERROR("Failed to read safetensors header size from shard: %s", shard.c_str());
            return ErrorCode::LOAD_ERROR;
        }

        std::vector<char> header_data(header_size);
        infile.read(header_data.data(), static_cast<std::streamsize>(header_size));
        if (!infile) {
            LOG_ERROR("Failed to read safetensors header data from shard: %s", shard.c_str());
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
                LOG_ERROR("Malformed tensor entry in safetensors header: %s", name.c_str());
                return ErrorCode::LOAD_ERROR;
            }

            std::string dtype = value["dtype"].get<std::string>();
            std::vector<int> shape = value["shape"].get<std::vector<int>>();

            size_t offset_start = value["data_offsets"][0];
            size_t offset_end = value["data_offsets"][1];

            DataType parsed_dtype;
            if (dtype == "fp16" || dtype == "F16" || dtype == "BF16") {
                parsed_dtype = DataType::FLOAT16;
            } else if (dtype == "fp32" || dtype == "F32") {
                parsed_dtype = DataType::FLOAT32;
            } else {
                LOG_ERROR("Unsupported tensor dtype in safetensors header: %s (%s)", name.c_str(), dtype.c_str());
                return ErrorCode::LOAD_ERROR;
            }

            WeightHeader header = {
                layer_idx,
                shape,
                name,
                shard,
                offset_start + 8 + header_size,
                offset_end + 8 + header_size,
                parsed_dtype
            };
            headers[name] = header;
            layer_idx++;
        }
    }

    return ErrorCode::SUCCESS;
}

//load to cpu
Tensor ModelWeights::load_layer(std::ifstream& file, const std::string& name) {
    if(headers.find(name) == headers.end()){
        LOG_ERROR("Weight name not found in header: %s", name.c_str());
        return Tensor();
    }
    const WeightHeader& header = headers[name];
    size_t weight_size = (header.offset_end - header.offset_start);
    std::vector<size_t> shape;
    shape.reserve(header.shape.size());
    for (int dim : header.shape) {
        shape.push_back(static_cast<size_t>(dim));
    }
    Tensor layer_tensor(weight_size / Tensor::element_size_bytes(header.dtype), nullptr, shape, header.dtype);
    layer_tensor.data = malloc(weight_size);
    if (layer_tensor.data == nullptr) {
        LOG_ERROR("Failed to allocate host buffer for layer: %s", name.c_str());
        return Tensor();
    }
    file.seekg(header.offset_start);
    file.read((char*)layer_tensor.data, weight_size);
    if (!file) {
        LOG_ERROR("Failed to read layer tensor bytes for: %s", name.c_str());
        free(layer_tensor.data);
        layer_tensor.data = nullptr;
        return Tensor();
    }

    return layer_tensor;

}
ErrorCode ModelWeights::build_weight_names(const char* file_name){
    std::ifstream infile(file_name);
    if(!infile.is_open()){
        LOG_ERROR("Failed to open model weight file: %s", file_name);
        return ErrorCode::LOAD_ERROR;
    }

    weight_names.clear();
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
        weight_names.push_back(name);
    }

    return ErrorCode::SUCCESS;
}
//concat qkv on cpu
Tensor ModelWeights::concat_qkv(const Tensor& Wq, const Tensor& Wk, const Tensor& Wv){
    size_t H = Wq.shape[0];
    const size_t elem_size = Tensor::element_size_bytes(Wq.dtype);
    const size_t q_cols = Wq.shape[1];
    const size_t k_cols = Wk.shape[1];
    const size_t v_cols = Wv.shape[1];
    const size_t out_cols = q_cols + k_cols + v_cols;

    Tensor out(
        Wq.shape[0] * out_cols,
        nullptr, 
        {H, out_cols}, 
        Wq.dtype,
        "cpu"
    );

    char* data = new char[out.size];
    out.data = data;

    const char* q = static_cast<const char*>(Wq.data);
    const char* k = static_cast<const char*>(Wk.data);
    const char* v = static_cast<const char*>(Wv.data);

    for (size_t i = 0; i < H; i++) {
        const size_t row_out_elems = out_cols;
        const size_t q_row_bytes = q_cols * elem_size;
        const size_t k_row_bytes = k_cols * elem_size;
        const size_t v_row_bytes = v_cols * elem_size;
        char* out_row = data + i * row_out_elems * elem_size;

        memcpy(
            out_row,
            q + i * q_row_bytes,
            q_row_bytes
        );

        memcpy(
            out_row + q_row_bytes,
            k + i * k_row_bytes,
            k_row_bytes
        );

        memcpy(
            out_row + q_row_bytes + k_row_bytes,
            v + i * v_row_bytes,
            v_row_bytes
        );
    }

    return out;
}

//copy from cpu to gpu
ErrorCode ModelWeights::load_weights(const char* weight_path) {
    (void)weight_path;

    Tensor tmp_layer_tensor_q;
    Tensor tmp_layer_tensor_k;
    Tensor tmp_layer_tensor_v;

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
                LOG_ERROR("Failed to open shard file for tensor %s: %s", name.c_str(), shard.c_str());
                return nullptr;
            }
            stream_it = shard_streams.emplace(shard, std::move(stream)).first;
        }
        return stream_it->second.get();
    };
    
    bool has_q = false;
    bool has_k = false;
    bool has_v = false;
    for(const auto& name : weight_names){

        std::ifstream* stream = get_stream_for_name(name);
        if (stream == nullptr) {
            return ErrorCode::LOAD_ERROR;
        }

        if(name.find("embed_tokens") != std::string::npos){
            Tensor tmp_layer_tensor = load_layer(*stream, name);
            if(tmp_layer_tensor.data == nullptr){
                return ErrorCode::LOAD_ERROR;
            }
            cudaMemcpy(
                layout.embedding_weights.data,
                tmp_layer_tensor.data, 
                layout.embedding_weights.size, 
                cudaMemcpyHostToDevice
            );
            free(tmp_layer_tensor.data);
            tmp_layer_tensor.data = nullptr;
        } else if(name.find("layers.") != std::string::npos){
            size_t pos1 = name.find("layers.") + 7;
            size_t pos2 = name.find(".", pos1);
            size_t layer_id = std::stoi(name.substr(pos1, pos2 - pos1));

            std::shared_ptr<TransformerLayerWeightLayout> layer_layout = layout.get_layer_layout<TransformerLayerWeightLayout>(layer_id);
            if (!layer_layout) {
                continue;
            }
            if(name.find("q_proj") != std::string::npos){
                tmp_layer_tensor_q = load_layer(*stream, name);
                if(tmp_layer_tensor_q.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                has_q = true;

            } else if(name.find("k_proj") != std::string::npos){
                tmp_layer_tensor_k = load_layer(*stream, name);
                if(tmp_layer_tensor_k.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                has_k = true;

            } else if(name.find("v_proj") != std::string::npos){
                tmp_layer_tensor_v = load_layer(*stream, name);
                if(tmp_layer_tensor_v.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                has_v = true;

            } else if(name.find("o_proj") != std::string::npos){
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                cudaMemcpy(
                    layer_layout->attention_weights.o_proj_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
            } else if(name.find("gate_proj") != std::string::npos){
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                cudaMemcpy(
                    layer_layout->mlp_weights.mlp_linears_weight[0].linear_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
            } else if(name.find("up_proj") != std::string::npos){
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                cudaMemcpy(
                    layer_layout->mlp_weights.mlp_linears_weight[1].linear_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
            } else if(name.find("down_proj") != std::string::npos){
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                cudaMemcpy(
                    layer_layout->mlp_weights.mlp_linears_weight[2].linear_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
            } else if(name.find("input_layernorm") != std::string::npos){
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                cudaMemcpy(
                    layer_layout->norm_weights[0].norm_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
            } else if(name.find("post_attention_layernorm") != std::string::npos){
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                cudaMemcpy(
                    layer_layout->norm_weights[1].norm_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
            }else{
                LOG_ERROR("Unrecognized weight name: %s, weights may be incomplete", name.c_str());
                continue;
            }

            if(!(has_q && has_k && has_v)){
                continue;
            }
            //concat Wq, Wk, Wv
            Tensor Wqkv = concat_qkv(tmp_layer_tensor_q, tmp_layer_tensor_k, tmp_layer_tensor_v);
            //transpose
            Tensor Wqkv_trans = Wqkv.transpose();
            //copy from cpu to gpu
            cudaMemcpy(
                layer_layout->attention_weights.qkv_proj_weight.data,
                Wqkv_trans.data,
                Wqkv_trans.size,
                cudaMemcpyHostToDevice
            );
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
                cudaMemcpy(
                    final_norm_layout->norm_weight.data,
                    tmp_layer_tensor.data,
                    final_norm_layout->norm_weight.size,
                    cudaMemcpyHostToDevice
                );
            }
            free(tmp_layer_tensor.data);
            tmp_layer_tensor.data = nullptr;
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
            if (lm_head_layout) {
                cudaMemcpy(
                    lm_head_layout->linear_weight.data,
                    tmp_layer_tensor.data,
                    lm_head_layout->linear_weight.size,
                    cudaMemcpyHostToDevice
                );
            }
            free(tmp_layer_tensor.data);
            tmp_layer_tensor.data = nullptr;
        } else {
            LOG_ERROR("Unrecognized weight name: %s, weights may be incomplete", name.c_str());
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
    
    return ErrorCode::SUCCESS;
}