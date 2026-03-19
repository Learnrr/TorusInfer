#include "ModelWeights.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <regex>
#include <unordered_map>
#include "include/error.h"
#include "include/utils/logger.h"

namespace {

std::vector<std::string> ResolveSafetensorShards(const std::string& model_path) {
    namespace fs = std::filesystem;
    std::vector<std::string> shards;

    fs::path p(model_path);
    if (!fs::exists(p)) {
        return shards;
    }

    const std::string fname = p.filename().string();
    const std::regex shard_pat(R"((.*)-(\d{5})-of-(\d{5})\.safetensors$)");
    std::smatch m;
    if (!std::regex_match(fname, m, shard_pat)) {
        shards.push_back(p.string());
        return shards;
    }

    const std::string prefix = m[1].str();
    const int total = std::stoi(m[3].str());
    for (int i = 1; i <= total; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s-%05d-of-%05d.safetensors", prefix.c_str(), i, total);
        fs::path part = p.parent_path() / buf;
        if (fs::exists(part)) {
            shards.push_back(part.string());
        }
    }

    return shards;
}

} // namespace

void WeightLayout::build_config(const ModelConfig& config){
    size_t offset = 0;
    layer_weights.clear();
    layer_weights.reserve(config.layer_configs.size());

    embedding_weights = Tensor(
        config.vocab_size * config.hidden_size,
        nullptr,
        {config.vocab_size, config.hidden_size},
        DTYPE
    );
    offset += embedding_weights.size;

    for (const auto& layer_cfg_base : config.layer_configs) {
        if (auto transformer_cfg = std::dynamic_pointer_cast<TransformerLayerConfig>(layer_cfg_base)) {
            auto transformer_layout = std::make_shared<TransformerLayerWeightLayout>();

            size_t q_hidden;
            q_hidden = transformer_cfg->attention_config.num_attention_heads * transformer_cfg->attention_config.head_dim;
            size_t kv_hidden;
            kv_hidden = transformer_cfg->attention_config.num_kv_heads * transformer_cfg->attention_config.head_dim;

            transformer_layout->attention_weights.qkv_proj_weight = Tensor(
                config.hidden_size * (q_hidden + 2 * kv_hidden),
                nullptr,
                {config.hidden_size, q_hidden + 2 * kv_hidden},
                DTYPE
            );
            offset += transformer_layout->attention_weights.qkv_proj_weight.size;

            transformer_layout->attention_weights.o_proj_weight = Tensor(
                q_hidden * config.hidden_size,
                nullptr,
                {q_hidden, config.hidden_size},
                DTYPE
            );
            offset += transformer_layout->attention_weights.o_proj_weight.size;

            size_t intermediate_size = transformer_cfg->mlp_config.intermediate_size;
            if (intermediate_size == 0 && !transformer_cfg->mlp_config.mlp_linears.empty()) {
                intermediate_size = transformer_cfg->mlp_config.mlp_linears[0].out_features;
            }
            if (intermediate_size == 0) {
                intermediate_size = config.hidden_size;
            }

            if (transformer_cfg->mlp_config.mlp_linears.size() >= 3) {
                const auto& gate_cfg = transformer_cfg->mlp_config.mlp_linears[0];
                const auto& up_cfg = transformer_cfg->mlp_config.mlp_linears[1];
                const auto& down_cfg = transformer_cfg->mlp_config.mlp_linears[2];

                transformer_layout->mlp_weights.gate_proj_weight = Tensor(gate_cfg.in_features * gate_cfg.out_features, nullptr, {gate_cfg.in_features, gate_cfg.out_features}, DTYPE);
                transformer_layout->mlp_weights.up_proj_weight = Tensor(up_cfg.in_features * up_cfg.out_features, nullptr, {up_cfg.in_features, up_cfg.out_features}, DTYPE);
                transformer_layout->mlp_weights.down_proj_weight = Tensor(down_cfg.in_features * down_cfg.out_features, nullptr, {down_cfg.in_features, down_cfg.out_features}, DTYPE);
            } else {
                transformer_layout->mlp_weights.gate_proj_weight = Tensor(config.hidden_size * intermediate_size, nullptr, {config.hidden_size, intermediate_size}, DTYPE);
                transformer_layout->mlp_weights.up_proj_weight = Tensor(config.hidden_size * intermediate_size, nullptr, {config.hidden_size, intermediate_size}, DTYPE);
                transformer_layout->mlp_weights.down_proj_weight = Tensor(intermediate_size * config.hidden_size, nullptr, {intermediate_size, config.hidden_size}, DTYPE);
            }

            offset += transformer_layout->mlp_weights.gate_proj_weight.size;
            offset += transformer_layout->mlp_weights.up_proj_weight.size;
            offset += transformer_layout->mlp_weights.down_proj_weight.size;

            transformer_layout->attn_norm_weight = Tensor(config.hidden_size, nullptr, {config.hidden_size}, DTYPE);
            transformer_layout->ffn_norm_weight = Tensor(config.hidden_size, nullptr, {config.hidden_size}, DTYPE);
            offset += transformer_layout->attn_norm_weight.size;
            offset += transformer_layout->ffn_norm_weight.size;

            layer_weights.push_back(transformer_layout);
            continue;
        }

        if (auto linear_cfg = std::dynamic_pointer_cast<LinearLayerConfig>(layer_cfg_base)) {
            auto linear_layout = std::make_shared<LinearLayerWeightLayout>();
            const size_t in_features = linear_cfg->linear_config.in_features;
            const size_t out_features = linear_cfg->linear_config.out_features;
            linear_layout->linear_weight = Tensor(in_features * out_features, nullptr, {in_features, out_features}, DTYPE);
            offset += linear_layout->linear_weight.size;
            layer_weights.push_back(linear_layout);
            continue;
        }

        if (auto norm_cfg = std::dynamic_pointer_cast<LayerNormLayerConfig>(layer_cfg_base)) {
            auto norm_layout = std::make_shared<LayerNormLayerWeightLayout>();
            const size_t norm_size = norm_cfg->norm_size == 0 ? config.hidden_size : norm_cfg->norm_size;
            norm_layout->norm_weight = Tensor(norm_size, nullptr, {norm_size}, DTYPE);
            offset += norm_layout->norm_weight.size;
            layer_weights.push_back(norm_layout);
            continue;
        }
    }

    total_size = offset;

}

void WeightLayout::build(){
    size_t offset = 0;
    embedding_weights.data = static_cast<void*>(static_cast<char*>(weights) + offset);
    offset += embedding_weights.size;

    for (size_t i = 0; i < layer_weights.size(); ++i) {
        if (auto transformer = get_layer_layout<TransformerLayerWeightLayout>(i)) {
            transformer->attention_weights.qkv_proj_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->attention_weights.qkv_proj_weight.size;

            transformer->attention_weights.o_proj_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->attention_weights.o_proj_weight.size;

            transformer->mlp_weights.gate_proj_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->mlp_weights.gate_proj_weight.size;

            transformer->mlp_weights.up_proj_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->mlp_weights.up_proj_weight.size;

            transformer->mlp_weights.down_proj_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->mlp_weights.down_proj_weight.size;

            transformer->attn_norm_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->attn_norm_weight.size;

            transformer->ffn_norm_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->ffn_norm_weight.size;
            continue;
        }

        if (auto linear = get_layer_layout<LinearLayerWeightLayout>(i)) {
            linear->linear_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += linear->linear_weight.size;
            continue;
        }

        if (auto norm = get_layer_layout<LayerNormLayerWeightLayout>(i)) {
            norm->norm_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += norm->norm_weight.size;
            continue;
        }
    }
}
    
ErrorCode ModelWeights::init(const ModelConfig& config){
    layout.build_config(config);
    cudaError_t cuda_err = cudaMalloc(&weights, layout.total_size);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate GPU memory for model weights: %s", cudaGetErrorString(cuda_err));
        return ErrorCode::CUDA_FAILURE;
    }
    layout.weights = weights;
    layout.build();
    ErrorCode error = parse_header(config.model_path.c_str());
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to parse model weight header");
        return error;
    }
    error = build_weight_names(config.weight_names_path.c_str());
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to build weight names list");
        return error;
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

            std::string dtype = value["dtype"].get<std::string>();
            std::vector<int> shape = value["shape"].get<std::vector<int>>();

            size_t offset_start = value["data_offsets"][0];
            size_t offset_end = value["data_offsets"][1];

            WeightHeader header = {
                layer_idx,
                shape,
                name,
                shard,
                offset_start + 8 + header_size,
                offset_end + 8 + header_size,
                dtype == "fp16" ? DataType::FLOAT16 : DataType::FLOAT32
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

    Tensor out(
        Wq.shape[0] * (Wq.shape[1] + 2 * Wk.shape[1]), //q_ROW*(q_COL + 2 * k_COL)
        nullptr, 
        {H, Wq.shape[1] + 2 * Wk.shape[1]}, 
        Wq.dtype,
        "cpu"
    );

    float* data = new float[Wq.shape[0] * (Wq.shape[1] + 2 * Wk.shape[1])];
    out.data = data;

    float* q = (float*)Wq.data;
    float* k = (float*)Wk.data;
    float* v = (float*)Wv.data;

    for (size_t i = 0; i < H; i++) {

        memcpy(
            data + i * (Wq.shape[1] + 2 * Wk.shape[1]),
            q + i * Wq.shape[1],
            Wq.shape[1] * sizeof(float)
        );

        memcpy(
            data + i * (Wq.shape[1] + 2 * Wk.shape[1]) + Wq.shape[1],
            k + i * Wk.shape[1],
            Wk.shape[1] * sizeof(float)
        );

        memcpy(
            data + i * (Wq.shape[1] + 2 * Wk.shape[1]) + Wq.shape[1] + Wk.shape[1],
            v + i * Wv.shape[1],
            Wv.shape[1] * sizeof(float)
        );
    }

    return out;
}

//copy from cpu to gpu
ErrorCode ModelWeights::load_weights(const char* weight_path) {
    (void)weight_path;

    Tensor tmp_layer_tensor;
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
            tmp_layer_tensor = load_layer(*stream, name);
            if(tmp_layer_tensor.data == nullptr){
                return ErrorCode::LOAD_ERROR;
            }
            cudaMemcpy(
                layout.embedding_weights.data,
                tmp_layer_tensor.data, 
                layout.embedding_weights.size, 
                cudaMemcpyHostToDevice
            );
        } else if(name.find("layers.") != std::string::npos){
            size_t pos1 = name.find("layers.") + 7;
            size_t pos2 = name.find(".", pos1);
            size_t layer_id = std::stoi(name.substr(pos1, pos2 - pos1));

                        std::shared_ptr<TransformerLayerWeightLayout> layer_layout = layout.get_layer_layout<TransformerLayerWeightLayout>(layer_id);
                        if (!layer_layout) {
                            continue;
                        }
            if(name.find("q_proj") != std::string::npos){
                tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                has_q = true;
                /*
                cudaMemcpy(
                    (char*)weights + layer_layout.q_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                */
            } else if(name.find("k_proj") != std::string::npos){
                tmp_layer_tensor_k = load_layer(*stream, name);
                if(tmp_layer_tensor_k.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                has_k = true;
                /*
                cudaMemcpy(
                    (char*)weights + layer_layout.k_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                */
            } else if(name.find("v_proj") != std::string::npos){
                tmp_layer_tensor_v = load_layer(*stream, name);
                if(tmp_layer_tensor_v.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                has_v = true;
                /*
                cudaMemcpy(
                    (char*)weights + layer_layout.v_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                */
            } else if(name.find("o_proj") != std::string::npos){
                tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                cudaMemcpy(
                    layer_layout->attention_weights.o_proj_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("gate_proj") != std::string::npos){
                tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                cudaMemcpy(
                    layer_layout->mlp_weights.gate_proj_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("up_proj") != std::string::npos){
                tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                cudaMemcpy(
                    layer_layout->mlp_weights.up_proj_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("down_proj") != std::string::npos){
                tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                cudaMemcpy(
                    layer_layout->mlp_weights.down_proj_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("attn_norm") != std::string::npos ||
                      name.find("input_layernorm") != std::string::npos){
                tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                cudaMemcpy(
                    layer_layout->attn_norm_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("ffn_norm") != std::string::npos ||
                      name.find("post_attention_layernorm") != std::string::npos){
                tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                cudaMemcpy(
                    layer_layout->ffn_norm_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            }else{
                LOG_ERROR("Unrecognized weight name: %s, weights may be incomplete", name.c_str());
                continue;
            }

            if(!(has_q && has_k && has_v)){
                continue;
            }
            //concat Wq, Wk, Wv
            Tensor Wqkv = concat_qkv(tmp_layer_tensor, tmp_layer_tensor_k, tmp_layer_tensor_v);
            //transpose
            Tensor Wqkv_trans = Wqkv.transpose();
            //copy from cpu to gpu
            cudaMemcpy(
                layer_layout->attention_weights.qkv_proj_weight.data,
                Wqkv_trans.data,
                Wqkv_trans.size,
                cudaMemcpyHostToDevice
            );
            delete[] Wqkv_trans.data;
            Wqkv_trans.data = nullptr;
            delete[] Wqkv.data;
            Wqkv.data = nullptr;

            has_k = false;
            has_q = false;  
            has_v = false;

        } else if(name.find("model.norm") != std::string::npos){
            tmp_layer_tensor = load_layer(*stream, name);
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
        } else if(name.find("lm_head") != std::string::npos){
            tmp_layer_tensor = load_layer(*stream, name);
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
        } else {
            LOG_ERROR("Unrecognized weight name: %s, weights may be incomplete", name.c_str());
            continue;
        }

        free(tmp_layer_tensor.data);
        free(tmp_layer_tensor_k.data);
        free(tmp_layer_tensor_v.data);

        tmp_layer_tensor.data = nullptr;
        tmp_layer_tensor_k.data = nullptr;
        tmp_layer_tensor_v.data = nullptr;        
        
    }
    
    return ErrorCode::SUCCESS;
}