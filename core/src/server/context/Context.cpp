// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <utility>

#include "server/context/Context.h"

namespace milvus {
namespace server {

Context::Context(std::string req_id) : req_id_(std::move(req_id)) {
}

const tracing::TraceContextPtr&
Context::GetTraceContext() const {
    return trace_context_;
}

void
Context::SetTraceContext(const tracing::TraceContextPtr& trace_context) {
    trace_context_ = trace_context;
}
std::shared_ptr<Context>
Context::Child(const std::string& operation_name) const {
    auto new_context = std::make_shared<Context>(req_id_);
    new_context->SetTraceContext(trace_context_->Child(operation_name));
    return new_context;
}

std::shared_ptr<Context>
Context::Follower(const std::string& operation_name) const {
    auto new_context = std::make_shared<Context>(req_id_);
    new_context->SetTraceContext(trace_context_->Follower(operation_name));
    return new_context;
}

void
Context::SetConnectionContext(ConnectionContextPtr& context) {
    context_ = context;
}

bool
Context::IsConnectionBroken() const {
    if (context_ == nullptr) {
        return false;
    }

    return context_->IsConnectionBroken();
}

ReqType
Context::GetReqType() const {
    return req_type_;
}

void
Context::SetReqType(ReqType type) {
    req_type_ = type;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
ContextChild::ContextChild(const ContextPtr& context, const std::string& operation_name) {
    if (context) {
        context_ = context->Child(operation_name);
    }
}

ContextChild::~ContextChild() {
    Finish();
}

void
ContextChild::Finish() {
    if (context_) {
        context_->GetTraceContext()->GetSpan()->Finish();
        context_ = nullptr;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
ContextFollower::ContextFollower(const ContextPtr& context, const std::string& operation_name) {
    if (context) {
        context_ = context->Follower(operation_name);
    }
}

ContextFollower::~ContextFollower() {
    if (context_) {
        context_->GetTraceContext()->GetSpan()->Finish();
    }
}

void
ContextFollower::Finish() {
    if (context_) {
        context_->GetTraceContext()->GetSpan()->Finish();
        context_ = nullptr;
    }
}

}  // namespace server
}  // namespace milvus
