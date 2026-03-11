#include "Batch.h"
#include "Workspace.h"
struct ForwardContext {
    size_t layer_id;
    Batch* batch;
    Workspace* workspace;

};