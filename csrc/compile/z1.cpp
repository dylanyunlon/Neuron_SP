// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "z1.h"
#include "deepcompile.h"

namespace dc {

class Z1CustomOpExecutor : public CustomOpExecutor {
public:
    Z1CustomOpExecutor(c10::intrusive_ptr<c10d::ProcessGroup> process_group,
                       std::shared_ptr<DSParamRegistry> param_registry,
                       std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets,
                       std::vector<long> ds_ids,
                       ncclComm_t nccl_comm,
                       at::cuda::CUDAStream rs_stream,
                       at::cuda::CUDAStream copy_stream,
                       bool pre_div_reduce)
        : CustomOpExecutor(process_group,
                           param_registry,
                           reduce_buckets,
                           ds_ids,
                           nccl_comm,
                           rs_stream,
                           copy_stream,
                           pre_div_reduce)
    {
    }
    ~Z1CustomOpExecutor() {}

    at::Tensor reduceGrad(at::Tensor grad_tensor, long ds_id) override
    {
        if (!hasKey(grad_tensors_, ds_id)) {
            grad_tensors_[ds_id] = grad_tensor;
        } else {
            grad_tensors_[ds_id].add_(grad_tensor);
        }

        if (param_updated_) {
            CustomOpExecutor::reduceGrad(grad_tensors_[ds_id], ds_id);
            grad_tensors_.erase(ds_id);
        }

        return at::Tensor();
    }

    void flushReduceBucket(at::ScalarType scalar_type) override
    {
        if (!hasKey(reduce_tasks_, scalar_type)) { return; }

        flushSPReduceBucket(scalar_type);

        if (!shouldSyncDP()) {
            performCleanup(scalar_type);
            return;
        }

        blockCopyEvents(scalar_type);
        applyPreDivision(scalar_type);

        ncclGroupStart();
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            const DSParam& param = param_registry_->getParam(t.getDSId());
            if (sp_comm_ != nullptr && param.getSpGroupId() >= 0) continue;

            ncclResult_t result = ncclAllReduce(t.getSendBuf().data_ptr(),
                                                t.getSendBuf().data_ptr(),
                                                t.getSendBuf().numel(),
                                                get_nccl_data_type(scalar_type),
                                                getReductionOp(),
                                                nccl_comm_,
                                                rs_stream_);
            if (result != ncclSuccess) { throw std::runtime_error("NCCL AllReduce failed"); }
        }
        ncclGroupEnd();

        {
            at::cuda::CUDAStreamGuard guard(rs_stream_);
            for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
                auto param = param_registry_->getParam(t.getDSId());
                auto grad_buf = param.getGradBuffer().flatten();

                if (grad_buf.numel() == 0) { continue; }

                int64_t offset = param.getOffset();
                auto recv_buf = t.getSendBuf().flatten().index(
                    {torch::indexing::Slice(offset, offset + grad_buf.numel())});
                grad_buf.copy_(recv_buf);
            }
        }

        performCleanup(scalar_type);
    }

    void endBackward() override
    {
        step_++;
        CustomOpExecutor::endBackward();
    }

protected:
    std::unordered_map<long, at::Tensor> grad_tensors_;
};

static at::cuda::CUDAStream rs_stream = at::cuda::getStreamFromPool(true);
static at::cuda::CUDAStream copy_stream = at::cuda::getStreamFromPool(true);

void register_graph_z1(long graph_id, const std::vector<long>& ds_ids)
{
    executors[graph_id] = std::make_shared<Z1CustomOpExecutor>(process_group,
                                                               param_registry,
                                                               reduce_buckets,
                                                               ds_ids,
                                                               nccl_comm,
                                                               rs_stream,
                                                               copy_stream,
                                                               pre_div_reduce);
}

void register_graph_z1_sp(long graph_id,
                          const std::vector<long>& ds_ids,
                          int sp_size,
                          int kx,
                          int warmup_steps)
{
    auto executor = std::make_shared<Z1CustomOpExecutor>(process_group,
                                                         param_registry,
                                                         reduce_buckets,
                                                         ds_ids,
                                                         nccl_comm,
                                                         rs_stream,
                                                         copy_stream,
                                                         pre_div_reduce);
    executor->sp_size_ = sp_size;
    executor->kx_ = kx;
    executor->warmup_steps_ = warmup_steps;

    if (sp_size > 1) {
        ncclUniqueId sp_id;
        int rank = process_group->getRank();
        int sp_group_rank = rank % sp_size;

        if (sp_group_rank == 0) {
            ncclGetUniqueId(&sp_id);
        }

        auto vec = std::vector<uint8_t>(
            reinterpret_cast<uint8_t*>(&sp_id),
            reinterpret_cast<uint8_t*>(&sp_id) + NCCL_UNIQUE_ID_BYTES);
        auto tensor = torch::from_blob(vec.data(),
                                        {static_cast<long>(vec.size())},
                                        torch::kUInt8).to(torch::Device(torch::kCUDA));
        std::vector<at::Tensor> bcast_input = {tensor};
        process_group->broadcast(bcast_input, c10d::BroadcastOptions())->wait();
        std::memcpy(&sp_id, tensor.to(torch::Device(torch::kCPU)).data_ptr(),
                     NCCL_UNIQUE_ID_BYTES);

        ncclCommInitRank(&executor->sp_comm_, sp_size, sp_id, sp_group_rank);
    }

    executors[graph_id] = executor;
}

void register_param(long ds_id,
                    const std::vector<int64_t>& ds_shape,
                    at::Tensor ds_tensor,
                    at::Tensor grad_buffer,
                    int64_t offset)
{
    param_registry->registerParam(ds_id, ds_shape, ds_tensor, grad_buffer, false, offset, false);
}

void register_param_sp(long ds_id,
                       const std::vector<int64_t>& ds_shape,
                       at::Tensor ds_tensor,
                       at::Tensor grad_buffer,
                       int64_t offset,
                       int sp_group_id)
{
    param_registry->registerParam(ds_id, ds_shape, ds_tensor, grad_buffer, false, offset, false);
    if (sp_group_id >= 0) {
        auto& params = const_cast<std::unordered_map<long, DSParam>&>(param_registry->getParams());
        params.at(ds_id).setSpGroupId(sp_group_id);
    }
}

}  // namespace dc
