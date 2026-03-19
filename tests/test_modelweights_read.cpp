#include "ModelWeights.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>

namespace {

std::string ResolveWeightEntryPath(const std::string& input_path) {
    namespace fs = std::filesystem;

    fs::path p(input_path);
    if (fs::is_directory(p)) {
        for (const auto& entry : fs::directory_iterator(p)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string fname = entry.path().filename().string();
            if (std::regex_match(fname, std::regex(R"(.*-00001-of-\d{5}\.safetensors$)"))) {
                return entry.path().string();
            }
        }

        fs::path single = p / "qwen-7b.safetensors";
        if (fs::exists(single)) {
            return single.string();
        }
        return input_path;
    }

    if (fs::exists(p)) {
        const std::string fname = p.filename().string();
        std::smatch m;
        const std::regex shard_pat(R"((.*)-(\d{5})-of-(\d{5})\.safetensors$)");
        if (std::regex_match(fname, m, shard_pat)) {
            fs::path entry = p.parent_path() /
                (m[1].str() + "-00001-of-" + m[3].str() + ".safetensors");
            if (fs::exists(entry)) {
                return entry.string();
            }
        }
        return input_path;
    }

    return input_path;
}

void TestParseHeaderAndLoadLayerFromRealWeights(
    const std::string& weight_path,
    const std::string& names_path
) {
    assert(std::filesystem::exists(weight_path) && "Real safetensors file not found");
    assert(std::filesystem::exists(names_path) && "Real weight_names file not found");

    ModelWeights model_weights;

    const ErrorCode parse_err = model_weights.parse_header(weight_path.c_str());
    assert(parse_err == ErrorCode::SUCCESS && "parse_header should succeed");

    const ErrorCode names_err = model_weights.build_weight_names(names_path.c_str());
    assert(names_err == ErrorCode::SUCCESS && "build_weight_names should succeed");

    assert(!model_weights.headers.empty() && "headers should not be empty");
    assert(!model_weights.weight_names.empty() && "weight_names should not be empty");
    assert(model_weights.headers.count("model.embed_tokens.weight") == 1);

    const WeightHeader& embed_header = model_weights.headers.at("model.embed_tokens.weight");
    assert(!embed_header.shard_file.empty() && "embed tensor shard file should not be empty");
    assert(std::filesystem::exists(embed_header.shard_file) && "embed tensor shard file does not exist");

    std::ifstream in(embed_header.shard_file, std::ios::binary);
    assert(in.is_open() && "Failed to open shard safetensors file for embed tensor");

    Tensor embed = model_weights.load_layer(in, "model.embed_tokens.weight");
    assert(embed.data != nullptr && "embed tensor data should not be null");
    assert(embed.size > 0 && "embed tensor size should be > 0");
    assert(!embed.shape.empty() && "embed tensor shape should not be empty");
    assert(embed.shape[0] > 0 && "embed tensor first dimension should be > 0");

    std::free(embed.data);
}

} // namespace

int main(int argc, char** argv) {
    std::string weight_path = "weights";
    std::string names_path = "weights/qwen-7b_weight_names.txt";

    if (argc >= 2) {
        weight_path = argv[1];
    }
    if (argc >= 3) {
        names_path = argv[2];
    }

    weight_path = ResolveWeightEntryPath(weight_path);

    std::cout << "Using weight file: " << weight_path << "\n";
    std::cout << "Using names file: " << names_path << "\n";

    TestParseHeaderAndLoadLayerFromRealWeights(weight_path, names_path);
    std::cout << "test_modelweights_read passed\n";
    return 0;
}
