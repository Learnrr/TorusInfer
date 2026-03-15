#include "Attention.h"


void Attention::prefill_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) override{

    Tensor qkv;
    qkv.data = context.workspace->get_qkv_workspace();

    size_t batch_seq_len = input.shape[0];

    qkv_projection(
        input, 
        layer_layout->qkv_proj_weight, 
        qkv, 
        batch_seq_len, 
        context.config->num_attention_heads, 
        context.config->head_dim
    );

    Tensor q;
    Tensor k;
    Tensor v;

    split_qkv(
        qkv, q, k, v, 
        batch_seq_len, 
        context.config->num_attention_heads, 
        context.config->head_dim
    );
    
    //write k and v to blocked cache
    write_cache(context, k, v);

    Tensor attn_output;
    attn_output.data = context.workspace->get_attn_context_workspace();

    // block_ids， block_offsets
    size_t num_tokens = context.batch->num_tokens;
    size_t* h_block_ids = new size_t[num_tokens];
    size_t* h_block_offsets = new size_t[num_tokens];
    float** h_kcache_block_ptrs = new float*[num_tokens];
    float** h_vcache_block_ptrs = new float*[num_tokens];
    build_read_cache(context, h_block_ids, h_block_offsets, (void**)h_kcache_block_ptrs, (void**)h_vcache_block_ptrs);

    size_t* d_block_ids;
    size_t* d_block_offsets;
    float** d_kcache_block_ptrs;
    float** d_vcache_block_ptrs;
    cudaMalloc(&d_block_ids, num_tokens * sizeof(size_t));
    cudaMalloc(&d_block_offsets, num_tokens * sizeof(size_t));
    cudaMalloc(&d_kcache_block_ptrs, num_tokens * sizeof(float*));
    cudaMalloc(&d_vcache_block_ptrs, num_tokens * sizeof(float*));
    cudaMemcpy(d_block_ids, h_block_ids, num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_offsets, h_block_offsets, num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kcache_block_ptrs, h_kcache_block_ptrs, num_tokens * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vcache_block_ptrs, h_vcache_block_ptrs, num_tokens * sizeof(float*), cudaMemcpyHostToDevice);

    size_t layer_id = context.layer_id;
    dim3 grid(batch_seq_len, context.config->num_attention_heads);
    dim3 block(context.config->head_dim);
    attention_qk_softmax_Pv_kernel<<<grid, block, context.config->max_seq_len * sizeof(float)>>>(
        (float*)q.data,
        d_kcache_block_ptrs,
        d_vcache_block_ptrs,
        d_block_ids,
        d_block_offsets,
        (float*)attn_output.data,
        batch_seq_len,
        context.config->num_layers,
        context.config->num_attention_heads,
        context.config->head_dim,
        context.config->block_size,
        context.config->max_seq_len,
        layer_id
    );

    cudaFree(d_block_ids);
    cudaFree(d_block_offsets);
    cudaFree(d_kcache_block_ptrs);
    cudaFree(d_vcache_block_ptrs);
    delete[] h_block_ids;
    delete[] h_block_offsets;
    delete[] h_kcache_block_ptrs;
    delete[] h_vcache_block_ptrs;

    output_projection(attn_output, layer_layout->o_proj_weight, output);

};

void Attention::decode_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) override{

    Tensor qkv;
    qkv.data = context.workspace->get_qkv_workspace();
    
    size_t batch_seq_len = input.shape[0];
    qkv_projection(
        input,                          //input tensor
        layer_layout->qkv_proj_weight,  //projection weight
        qkv,                            //output qkv tensor           
        batch_seq_len,                  // batch_seq_len
        context.config->num_attention_heads,
        context.config->head_dim
    );

    Tensor q;
    Tensor k;
    Tensor v;

    split_qkv(
        qkv, q, k, v, 
        batch_seq_len, 
        context.config->num_attention_heads, 
        context.config->head_dim
    );

    //write k and v to blocked cache
    write_cache(context, k, v);

    size_t num_tokens = context.batch->num_tokens;
    size_t* h_block_ids = new size_t[num_tokens];
    size_t* h_block_offsets = new size_t[num_tokens];
    float** h_kcache_block_ptrs = new float*[num_tokens];
    float** h_vcache_block_ptrs = new float*[num_tokens];
    build_read_cache(context, h_block_ids, h_block_offsets, (void**)h_kcache_block_ptrs, (void**)h_vcache_block_ptrs);

    size_t* d_block_ids;
    size_t* d_block_offsets;
    float** d_kcache_block_ptrs;
    float** d_vcache_block_ptrs;
    cudaMalloc(&d_block_ids, num_tokens * sizeof(size_t));
    cudaMalloc(&d_block_offsets, num_tokens * sizeof(size_t));
    cudaMalloc(&d_kcache_block_ptrs, num_tokens * sizeof(float*));
    cudaMalloc(&d_vcache_block_ptrs, num_tokens * sizeof(float*));
    cudaMemcpy(d_block_ids, h_block_ids, num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_offsets, h_block_offsets, num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kcache_block_ptrs, h_kcache_block_ptrs, num_tokens * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vcache_block_ptrs, h_vcache_block_ptrs, num_tokens * sizeof(float*), cudaMemcpyHostToDevice);

    Tensor attn_output;
    attn_output.data = context.workspace->get_attn_context_workspace();
    size_t layer_id = context.layer_id;
    dim3 grid(batch_seq_len, context.config->num_attention_heads);
    dim3 block(context.config->head_dim);

    attention_qk_softmax_Pv_kernel<<<grid, block, context.config->max_seq_len * sizeof(float)>>>(
        (float*)q.data,
        d_kcache_block_ptrs,
        d_vcache_block_ptrs,
        d_block_ids,
        d_block_offsets,
        (float*)attn_output.data,
        batch_seq_len,
        context.config->num_layers,
        context.config->num_attention_heads,
        context.config->head_dim,
        context.config->block_size,
        context.config->max_seq_len,
        layer_id
    );

    cudaFree(d_block_ids);
    cudaFree(d_block_offsets);
    cudaFree(d_kcache_block_ptrs);
    cudaFree(d_vcache_block_ptrs);
    delete[] h_block_ids;
    delete[] h_block_offsets;
    delete[] h_kcache_block_ptrs;
    delete[] h_vcache_block_ptrs;

    output_projection(attn_output, layer_layout->o_proj_weight, output);

};

void Attention::write_cache(ForwardContext& context, const Tensor& key, const Tensor& value) {

    size_t num_tokens = context.batch->num_tokens;
    size_t* h_block_ids = new size_t[num_tokens];
    size_t* h_block_offsets = new size_t[num_tokens];
    void** h_kcache_block_ptrs = new void*[num_tokens];
    void** h_vcache_block_ptrs = new void*[num_tokens];

    size_t* d_block_ids;
    size_t* d_block_offsets;
    float** d_kcache_block_ptrs;
    float** d_vcache_block_ptrs;
    cudaMalloc(&d_block_ids, num_tokens * sizeof(size_t));
    cudaMalloc(&d_block_offsets, num_tokens * sizeof(size_t));
    cudaMalloc(&d_kcache_block_ptrs, num_tokens * sizeof(void*));
    cudaMalloc(&d_vcache_block_ptrs, num_tokens * sizeof(void*));
    cudaMemcpy(d_block_ids, h_block_ids, num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_offsets, h_block_offsets, num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kcache_block_ptrs, h_kcache_block_ptrs, num_tokens * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vcache_block_ptrs, h_vcache_block_ptrs, num_tokens * sizeof(void*), cudaMemcpyHostToDevice);


    for(size_t i = 0; i < num_tokens; ++i) {
        Batch* batch = context.batch;
        auto seq = batch->sequences[i];
        size_t pos = batch->token_positions[i];

        size_t block_idx = pos / BLOCK_SIZE;
        size_t offset = pos % BLOCK_SIZE;

        h_block_ids[i] = seq->blocks[block_idx];
        h_block_offsets[i] = offset;
        h_kcache_block_ptrs[i] = seq->blocks[block_idx].key_cache_ptr;
        h_vcache_block_ptrs[i] = seq->blocks[block_idx].value_cache_ptr;
    }

    write_kvcache_kernel(
        d_kcache_block_ptrs, 
        d_vcache_block_ptrs, 
        d_block_ids,
        d_block_offsets, 
        key.data, 
        value.data,
        num_tokens,
        context.config->num_layers,
        context.config->num_attention_heads,
        context.config->head_dim,
        context.config->block_size
        context.layer_id
    );

    delete[] block_ids;
    delete[] block_offsets;
    delete[] kcache_block_ptrs;
    delete[] vcache_block_ptrs;

}

void Attention::build_read_cache(
    ForwardContext& context, 
    size_t* block_ids, 
    size_t* block_offsets,
    void** kcache_block_ptrs,
    void** vcache_block_ptrs

) {
      
    for(size_t i = 0; i < context.batch->num_tokens; ++i) {
        Batch* batch = context.batch;
        auto seq = batch->sequences[i];
        size_t pos = batch->token_positions[i];

        size_t block_idx = pos / BLOCK_SIZE;
        size_t offset = pos % BLOCK_SIZE;

        block_ids[i] = seq->blocks[block_idx].block_id;
        block_offsets[i] = offset;
        kcache_block_ptrs[i] = seq->blocks[block_idx].key_cache_ptr;
        vcache_block_ptrs[i] = seq->blocks[block_idx].value_cache_ptr;
    }



}
// split_qkv
void Attention::split_qkv(
    const Tensor& qkv, 
    Tensor& q, 
    Tensor& k, 
    Tensor& v,
    size_t batch_seq_len, 
    size_t num_heads, 
    size_t head_dim
) {
    size_t total = batch_seq_len * num_heads * head_dim;
    if (qkv.dtype == DataType::FLOAT32) {
        float* base = static_cast<float*>(qkv.data);
        q.data = base;
        k.data = base + total;
        v.data = base + 2 * total;
    } else if (qkv.dtype == DataType::FLOAT16) {
        uint16_t* base = static_cast<uint16_t*>(qkv.data);
        q.data = base;
        k.data = base + total;
        v.data = base + 2 * total;
    } else {
        q.data = nullptr;
        k.data = nullptr;
        v.data = nullptr;
    }
    q.size = total;
    k.size = total;
    v.size = total;
    q.shape = {batch_seq_len, num_heads, head_dim};
    k.shape = {batch_seq_len, num_heads, head_dim};
    v.shape = {batch_seq_len, num_heads, head_dim};
    q.dtype = qkv.dtype;
    k.dtype = qkv.dtype;
    v.dtype = qkv.dtype;
}
void Attention::qkv_projection(
    const Tensor& input, 
    const Tensor& weight, 
    Tensor& qkv, 
    size_t batch_seq_len, 
    size_t num_heads,
    size_t head_dim
) {

    qkv.size = batch_seq_len * num_heads * head_dim * 3;
    qkv.shape = {batch_seq_len, num_heads * head_dim * 3};
    qkv.dtype = input.dtype;

    dim3 grid(batch_seq_len, num_heads, 3);
    dim3 block(head_dim);
    projection_kernel<<<grid, block>>>(
        input.data, 
        weight.data, 
        nullptr, 
        batch_seq_len, 
        num_heads,
        head_dim
    );
    

}

void Attention::output_projection(const Tensor& input, const Tensor& weight, Tensor& output) {
    dim3 grid(batch_seq_len, num_heads);
    dim3 block(head_dim);
    output_projection_kernel<<<grid, block>>>(
        input.data, 
        weight.data, 
        output.data, 
        batch_seq_len, 
        num_heads, 
        head_dim
    );
}