#include "ModelWeights.h"
#include <cstring>

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
    offset += embedding_weights.size * DTYPE;

    for (const auto& layer_cfg_base : config.layer_configs) {
        if (auto transformer_cfg = std::dynamic_pointer_cast<TransformerLayerConfig>(layer_cfg_base)) {
            auto transformer_layout = std::make_shared<TransformerLayerWeightLayout>();

            size_t qkv_hidden = config.hidden_size;
            if (transformer_cfg->attention_config.num_attention_heads > 0 && transformer_cfg->attention_config.head_dim > 0) {
                qkv_hidden = transformer_cfg->attention_config.num_attention_heads * transformer_cfg->attention_config.head_dim;
            }

            transformer_layout->attention_weights.qkv_proj_weight = Tensor(
                config.hidden_size * qkv_hidden * 3,
                nullptr,
                {config.hidden_size, qkv_hidden * 3},
                DTYPE
            );
            offset += transformer_layout->attention_weights.qkv_proj_weight.size * DTYPE;

            transformer_layout->attention_weights.o_proj_weight = Tensor(
                qkv_hidden * config.hidden_size,
                nullptr,
                {qkv_hidden, config.hidden_size},
                DTYPE
            );
            offset += transformer_layout->attention_weights.o_proj_weight.size * DTYPE;

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

            offset += transformer_layout->mlp_weights.gate_proj_weight.size * DTYPE;
            offset += transformer_layout->mlp_weights.up_proj_weight.size * DTYPE;
            offset += transformer_layout->mlp_weights.down_proj_weight.size * DTYPE;

            transformer_layout->attn_norm_weight = Tensor(config.hidden_size, nullptr, {config.hidden_size}, DTYPE);
            transformer_layout->ffn_norm_weight = Tensor(config.hidden_size, nullptr, {config.hidden_size}, DTYPE);
            offset += transformer_layout->attn_norm_weight.size * DTYPE;
            offset += transformer_layout->ffn_norm_weight.size * DTYPE;

            layer_weights.push_back(transformer_layout);
            continue;
        }

        if (auto linear_cfg = std::dynamic_pointer_cast<LinearLayerConfig>(layer_cfg_base)) {
            auto linear_layout = std::make_shared<LinearLayerWeightLayout>();
            const size_t in_features = linear_cfg->linear_config.in_features;
            const size_t out_features = linear_cfg->linear_config.out_features;
            linear_layout->linear_weight = Tensor(in_features * out_features, nullptr, {in_features, out_features}, DTYPE);
            offset += linear_layout->linear_weight.size * DTYPE;
            layer_weights.push_back(linear_layout);
            continue;
        }

        if (auto norm_cfg = std::dynamic_pointer_cast<LayerNormLayerConfig>(layer_cfg_base)) {
            auto norm_layout = std::make_shared<LayerNormLayerWeightLayout>();
            const size_t norm_size = norm_cfg->norm_size == 0 ? config.hidden_size : norm_cfg->norm_size;
            norm_layout->norm_weight = Tensor(norm_size, nullptr, {norm_size}, DTYPE);
            offset += norm_layout->norm_weight.size * DTYPE;
            layer_weights.push_back(norm_layout);
            continue;
        }
    }

    total_size = offset;

}

void WeightLayout::build(){
    size_t offset = 0;
    embedding_weights.data = static_cast<void*>(static_cast<char*>(weights) + offset);
    offset += embedding_weights.size * DTYPE;

    for (size_t i = 0; i < layer_weights.size(); ++i) {
        if (auto transformer = get_layer_layout<TransformerLayerWeightLayout>(i)) {
            transformer->attention_weights.qkv_proj_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->attention_weights.qkv_proj_weight.size * DTYPE;

            transformer->attention_weights.o_proj_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->attention_weights.o_proj_weight.size * DTYPE;

            transformer->mlp_weights.gate_proj_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->mlp_weights.gate_proj_weight.size * DTYPE;

            transformer->mlp_weights.up_proj_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->mlp_weights.up_proj_weight.size * DTYPE;

            transformer->mlp_weights.down_proj_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->mlp_weights.down_proj_weight.size * DTYPE;

            transformer->attn_norm_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->attn_norm_weight.size * DTYPE;

            transformer->ffn_norm_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += transformer->ffn_norm_weight.size * DTYPE;
            continue;
        }

        if (auto linear = get_layer_layout<LinearLayerWeightLayout>(i)) {
            linear->linear_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += linear->linear_weight.size * DTYPE;
            continue;
        }

        if (auto norm = get_layer_layout<LayerNormLayerWeightLayout>(i)) {
            norm->norm_weight.data = static_cast<void*>(static_cast<char*>(weights) + offset);
            offset += norm->norm_weight.size * DTYPE;
            continue;
        }
    }
}
    
void ModelWeights::init(const ModelConfig& config){
    layout.build_config(config);
    cudaMalloc(&weights, layout.total_size);
    layout.weights = weights;
    layout.build();
}
        
void ModelWeights::parse_header(const char* file_name){
    std::ifstream infile(file_name, std::ios::binary);
    uint64_t header_size;
    infile.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));

    char* header_data = new char[header_size];
    infile.read(header_data, header_size);

    json header_json = json::parse(header_data);
    size_t layer_idx = 0;
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
            offset_start + 8 + header_size, 
            offset_end + 8 + header_size, 
            dtype == "fp16" ? DataType::FLOAT16 : DataType::FLOAT32
        };
        headers[name] = header;
        layer_idx++;
    }
    delete[] header_data;


}

//load to cpu
Tensor ModelWeights::load_layer(std::ifstream& file, const std::string& name) {
    WeightHeader header = headers[name];
    size_t weight_size = (header.offset_end - header.offset_start);
    std::vector<size_t> shape;
    shape.reserve(header.shape.size());
    for (int dim : header.shape) {
        shape.push_back(static_cast<size_t>(dim));
    }
    Tensor layer_tensor(weight_size, nullptr, shape, header.dtype);
    layer_tensor.data = malloc(weight_size);
    file.seekg(header.offset_start);
    file.read((char*)layer_tensor.data, weight_size);

    return layer_tensor;

}
//concat qkv on cpu
Tensor ModelWeights::concat_qkv(const Tensor& Wq, const Tensor& Wk, const Tensor& Wv){
    size_t H = Wq.shape[0];

    Tensor out(H * 3 * H, nullptr, {H, 3 * H}, Wq.dtype);
    float* data = new float[out.size];
    out.data = data;

    float* q = (float*)Wq.data;
    float* k = (float*)Wk.data;
    float* v = (float*)Wv.data;

    for (size_t i = 0; i < H; i++) {

        memcpy(data + i * 3 * H,
            q + i * H,
            H * sizeof(float));

        memcpy(data + i * 3 * H + H,
            k + i * H,
            H * sizeof(float));

        memcpy(data + i * 3 * H + 2 * H,
            v + i * H,
            H * sizeof(float));
    }

    return out;
}

//copy from cpu to gpu
void ModelWeights::load_weights(const char* weight_path) {
    // Load model weights logic, e.g., read weights from file
    std::ifstream infile(weight_path, std::ios::binary);


    Tensor tmp_layer_tensor;
    Tensor tmp_layer_tensor_k;
    Tensor tmp_layer_tensor_v;            
    for(auto& item : headers){
        const std::string& name = item.first;
        if(name.find("embed_tokens") != std::string::npos){
            tmp_layer_tensor = load_layer(infile, name);
            cudaMemcpy(
                layout.embedding_weights.data,
                tmp_layer_tensor.data, 
                layout.embedding_weights.size, 
                cudaMemcpyHostToDevice
            );
        } else if(name.find("layer.") != std::string::npos){
            size_t pos1 = name.find("layers.") + 7;
            size_t pos2 = name.find(".", pos1);
            size_t layer_id = std::stoi(name.substr(pos1, pos2 - pos1));

                        std::shared_ptr<TransformerLayerWeightLayout> layer_layout = layout.get_layer_layout<TransformerLayerWeightLayout>(layer_id);
                        if (!layer_layout) {
                            continue;
                        }
            if(name.find("q_proj") != std::string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                /*
                cudaMemcpy(
                    (char*)weights + layer_layout.q_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                */
            } else if(name.find("k_proj") != std::string::npos){
                tmp_layer_tensor_k = load_layer(infile, name);
                /*
                cudaMemcpy(
                    (char*)weights + layer_layout.k_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                */
            } else if(name.find("v_proj") != std::string::npos){
                tmp_layer_tensor_v = load_layer(infile, name);
                /*
                cudaMemcpy(
                    (char*)weights + layer_layout.v_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                */
            } else if(name.find("o_proj") != std::string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                cudaMemcpy(
                    layer_layout->attention_weights.o_proj_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("gate_proj") != std::string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                cudaMemcpy(
                    layer_layout->mlp_weights.gate_proj_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("up_proj") != std::string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                cudaMemcpy(
                    layer_layout->mlp_weights.up_proj_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("down_proj") != std::string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                cudaMemcpy(
                    layer_layout->mlp_weights.down_proj_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("attn_norm") != std::string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                cudaMemcpy(
                    layer_layout->attn_norm_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("ffn_norm") != std::string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                cudaMemcpy(
                    layer_layout->ffn_norm_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
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

        } else if(name.find("lm_head") != std::string::npos){
            tmp_layer_tensor = load_layer(infile, name);
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
        }
        free(tmp_layer_tensor.data);
        
    }
    
    
}