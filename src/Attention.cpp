#include "Attention.h"


void Attention::prefill_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) override{

    Tensor qkv;
    qkv.data = context.workspace->get_qkv_workspace();
    qkv_projection(input, layer_layout->qkv_proj_weight, qkv);

    Tensor q;
    Tensor k;
    Tensor v;

    split_qkv(qkv, q, k, v);

    
    write_cache(context, k.data, v.data);

    Tensor P;
    P.data = context.workspace->get_attn_score_workspace();
    attention_qk(q, k, P);

    softmax_mask(P);
    Tensor attn_context;
    attn_context.data = context.workspace->get_attn_context_workspace();
    attention_Pv(P, v, attn_context);

    output_projection(attn_context, layer_layout->o_proj_weight, output);


};
void Attention::decode_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) override{
    Tensor qkv;
    qkv.data = context.workspace->get_qkv_workspace();
    qkv_projection(input, layer_layout->qkv_proj_weight, qkv);

    Tensor q;
    Tensor k;
    Tensor v;

    split_qkv(qkv, q, k, v);

    write_cache(context, k.data, v.data);

    Tensor P;
    P.data = context.workspace->get_attn_score_workspace();
    attention_qk(q, k, P);

    softmax_mask(P, false);
    Tensor attn_context;
    attn_context.data = context.workspace->get_attn_context_workspace();
    attention_Pv(P, v, attn_context);

    output_projection(attn_context, layer_layout->o_proj_weight, output);

};

void Attention::write_cache(ForwardContext& context, void* key_data, void* value_data) {
    size_t* block_ids = new size_t[context.batch->num_tokens];
    size_t* block_offsets = new size_t[context.batch->num_tokens];

    for(size_t i = 0; i < context.batch->num_tokens; ++i) {
        Batch* batch = context.batch;
        Sequence* seq = batch->sequences[i];
        size_t pos = batch->token_positions[i];

        size_t block_idx = pos / BLOCK_SIZE;
        size_t offset = pos % BLOCK_SIZE;

        block_ids[i] = seq->blocks[block_idx];
        block_offsets[i] = offset;
        /*
        void* k_dst = (char*)block_id->key_cache_ptr 
                    + context.layer_id * HEAD_DIM * NUM_HEADS * DTYPE 
                    + offset * HEAD_DIM * NUM_HEADS * DTYPE;

        void* v_dst = (char*)block_id->value_cache_ptr 
                    + context.layer_id * HEAD_DIM * NUM_HEADS * DTYPE 
                    + offset * HEAD_DIM * NUM_HEADS * DTYPE;

        cudaMemcpy(
            k_dst, 
            key_data + i * HEAD_DIM * NUM_HEADS * DTYPE, HEAD_DIM * NUM_HEADS * DTYPE, 
            cudaMemcpyDeviceToDevice
        );
        cudaMemcpy(
            v_dst, 
            value_data + i * HEAD_DIM * NUM_HEADS * DTYPE, HEAD_DIM * NUM_HEADS * DTYPE, 
            cudaMemcpyDeviceToDevice
        );
        */

    }

}

void Attention::read_cache(ForwardContext& context, void* key_data, void* value_data) {

    size_t* block_ids = new int[contact.batch->num_tokens];
    size_t* block_offsets = new size_t[contect->batch->num_tokens];
    for(size_t i = 0; i < context.batch->num_tokens; ++i) {
        Batch* batch = context.batch;
        Sequence* seq = batch->sequences[i];
        size_t pos = batch->token_positions[i];

        size_t block_idx = pos / BLOCK_SIZE;
        size_t offset = pos % BLOCK_SIZE;

        block_ids[i] = seq->blocks[block_idx];
        block_offsets[i] = offset;

        /*
        void* k_src = (char*)block_id->key_cache_ptr 
                    + context.layer_id * HEAD_DIM * NUM_HEADS * DTYPE 
                    + offset * HEAD_DIM * NUM_HEADS * DTYPE;

        void* v_src = (char*)block_id->value_cache_ptr 
                    + context.layer_id * HEAD_DIM * NUM_HEADS * DTYPE 
                    + offset * HEAD_DIM * NUM_HEADS * DTYPE;

        cudaMemcpy(
            key_data + i * HEAD_DIM * NUM_HEADS * DTYPE, 
            k_src, 
            HEAD_DIM * NUM_HEADS * DTYPE, 
            cudaMemcpyDeviceToDevice
        );
        cudaMemcpy(
            value_data + i * HEAD_DIM * NUM_HEADS * DTYPE, 
            v_src, 
            HEAD_DIM * NUM_HEADS * DTYPE, 
            cudaMemcpyDeviceToDevice
        );
        */

    }

}
void Attention::split_qkv(const Tensor& qkv, Tensor& q, Tensor& k, Tensor& v) {
    // Implement logic to split the combined qkv tensor into separate q, k, v tensors
    q.data = qkv.data;
    q.size = qkv.size / 3; 
    q.shape = {qkv.shape[0], qkv.shape[1], qkv.shape[2] / 3}; 
    q.dtype = qkv.dtype;

    k.data = qkv.data + (qkv.size / 3); 
    k.size = qkv.size / 3; 
    k.shape = {qkv.shape[0], qkv.shape[1], qkv.shape[2] / 3}; 
    k.dtype = qkv.dtype;

    v.data = qkv.data + (2 * qkv.size / 3);
    v.size = qkv.size / 3; 
    v.shape = {qkv.shape[0], qkv.shape[1], qkv.shape[2] / 3}; 
    v.dtype = qkv.dtype;
}
void Attention::qkv_projection(const Tensor& input, const Tensor& weight, Tensor& qkv) {
    // Implement logic to project the input tensor into a combined qkv tensor
    qkv.size = input.shape[0] * weight.shape[1] * 3;
    qkv.shape = {input.shape[0], weight.shape[1] * 3;
    qkv.dtype = input.dtype;
    qkv_projection_kernel(input, weight, qkv);
    

}

void Attention::attention_qk(const Tensor& q, const Tensor& k, Tensor& P) {
    // Implement logic to compute attention scores
    P.size = q.shape[0] * q.shape[1] * k.shape[2];
    P.shape = {q.shape[0], q.shape[1], k.shape[2]};
    P.dtype = q.dtype;

    attention_qk_kernel(q, k, P);
}

void Attention::softmax_mask(Tensor& scores, bool causal_mask = true) {
    // Implement softmax operation with masking on the attention scores
    softmax_mask_kernel(scores);
}
void Attention::attention_Pv(const Tensor& P, const Tensor& v, Tensor& context) {
    // Implement logic to compute attention context
    context.size = P.shape[0] * P.shape[1] * v.shape[2];
    context.shape = {P.shape[0], P.shape[1], v.shape[2]};
    context.dtype = P.dtype;

    attention_Pv_kernel(P, v, context);


}

void Attention::output_projection(const Tensor& input, const Tensor& weight, Tensor& output) {

}