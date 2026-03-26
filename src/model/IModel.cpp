#include "model/IModel.h"
#include "model/QWEN_Model.h"

std::unique_ptr<IModel> ModelFactory::create_model(const std::string& model_name) {
    if (model_name == "QWEN") {
        return std::make_unique<QWEN_Model>();
    }
    return nullptr;
}
