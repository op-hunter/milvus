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

#include "db/SnapshotVisitor.h"
#include <sstream>
#include "db/SnapshotHandlers.h"
#include "db/meta/MetaTypes.h"
#include "db/snapshot/Snapshots.h"

namespace milvus {
namespace engine {

SnapshotVisitor::SnapshotVisitor(snapshot::ScopedSnapshotT ss) : ss_(ss) {
}

SnapshotVisitor::SnapshotVisitor(const std::string& collection_name) {
    status_ = snapshot::Snapshots::GetInstance().GetSnapshot(ss_, collection_name);
}

SnapshotVisitor::SnapshotVisitor(snapshot::ID_TYPE collection_id) {
    status_ = snapshot::Snapshots::GetInstance().GetSnapshot(ss_, collection_id);
}

Status
SnapshotVisitor::SegmentsToSearch(meta::FilesHolder& files_holder) {
    STATUS_CHECK(status_);

    auto handler = std::make_shared<SegmentsToSearchCollector>(ss_, files_holder);
    handler->Iterate();

    return handler->GetStatus();
}

SegmentFieldElementVisitor::Ptr
SegmentFieldElementVisitor::Build(snapshot::ScopedSnapshotT ss, const snapshot::FieldElementPtr& field_element,
                                  const snapshot::SegmentPtr& segment, const snapshot::SegmentFilePtr& segment_file) {
    if (!ss || !segment || !field_element) {
        return nullptr;
    }

    if (segment_file) {
        if (segment_file->GetFieldElementId() != field_element->GetID()) {
            std::cout << "FieldElement " << segment_file->GetFieldElementId() << " is expected for SegmentFile ";
            std::cout << segment_file->GetID() << " while actual is " << field_element->GetID() << std::endl;
            return nullptr;
        }
        if (segment_file->GetSegmentId() != segment->GetID()) {
            std::cout << "Segment " << segment_file->GetSegmentId() << " is expected for SegmentFile ";
            std::cout << segment_file->GetID() << " while actual is " << segment->GetID() << std::endl;
            return nullptr;
        }
    }

    auto visitor = std::make_shared<SegmentFieldElementVisitor>();
    visitor->SetFieldElement(field_element);
    if (segment_file) {
        visitor->SetFile(segment_file);
    }

    return visitor;
}

SegmentFieldElementVisitor::Ptr
SegmentFieldElementVisitor::Build(snapshot::ScopedSnapshotT ss, snapshot::ID_TYPE segment_id,
                                  snapshot::ID_TYPE field_element_id) {
    if (!ss) {
        return nullptr;
    }

    auto element = ss->GetResource<snapshot::FieldElement>(field_element_id);
    if (!element) {
        return nullptr;
    }

    auto visitor = std::make_shared<SegmentFieldElementVisitor>();
    visitor->SetFieldElement(element);
    auto segment = ss->GetResource<snapshot::Segment>(segment_id);
    if (!segment) {
        return nullptr;
    }

    auto file = ss->GetSegmentFile(segment_id, field_element_id);
    if (file) {
        visitor->SetFile(file);
    }

    return visitor;
}

SegmentFieldVisitor::Ptr
SegmentFieldVisitor::Build(snapshot::ScopedSnapshotT ss, const snapshot::FieldPtr& field,
                           const snapshot::SegmentPtr& segment, const snapshot::SegmentFile::VecT& segment_files) {
    if (!ss || !segment || !field) {
        return nullptr;
    }
    if (ss->GetResource<snapshot::Field>(field->GetID()) != field) {
        return nullptr;
    }

    auto visitor = std::make_shared<SegmentFieldVisitor>();
    visitor->SetField(field);

    std::map<snapshot::ID_TYPE, snapshot::SegmentFilePtr> files;
    for (auto& f : segment_files) {
        files[f->GetFieldElementId()] = f;
    }

    auto executor = [&](const snapshot::FieldElement::Ptr& field_element,
                        snapshot::FieldElementIterator* itr) -> Status {
        if (field_element->GetFieldId() != field->GetID()) {
            return Status::OK();
        }
        snapshot::SegmentFilePtr file;
        auto it = files.find(field_element->GetID());
        if (it != files.end()) {
            file = it->second;
        }
        auto element_visitor = SegmentFieldElementVisitor::Build(ss, field_element, segment, file);
        if (!element_visitor) {
            return Status::OK();
        }
        visitor->InsertElement(element_visitor);
        return Status::OK();
    };

    auto iterator = std::make_shared<snapshot::FieldElementIterator>(ss, executor);
    iterator->Iterate();

    return visitor;
}

SegmentFieldVisitor::Ptr
SegmentFieldVisitor::Build(snapshot::ScopedSnapshotT ss, snapshot::ID_TYPE segment_id, snapshot::ID_TYPE field_id) {
    if (!ss) {
        return nullptr;
    }

    auto field = ss->GetResource<snapshot::Field>(field_id);
    if (!field) {
        return nullptr;
    }

    auto visitor = std::make_shared<SegmentFieldVisitor>();
    visitor->SetField(field);

    auto executor = [&](const snapshot::FieldElement::Ptr& field_element,
                        snapshot::FieldElementIterator* itr) -> Status {
        if (field_element->GetFieldId() != field_id) {
            return Status::OK();
        }
        auto element_visitor = SegmentFieldElementVisitor::Build(ss, segment_id, field_element->GetID());
        if (!element_visitor) {
            return Status::OK();
        }
        visitor->InsertElement(element_visitor);
        return Status::OK();
    };

    auto iterator = std::make_shared<snapshot::FieldElementIterator>(ss, executor);
    iterator->Iterate();

    return visitor;
}

SegmentVisitor::Ptr
SegmentVisitor::Build(snapshot::ScopedSnapshotT ss, const snapshot::SegmentPtr& segment,
                      const snapshot::SegmentFile::VecT& segment_files) {
    if (!ss || !segment) {
        return nullptr;
    }
    if (!ss->GetResource<snapshot::Partition>(segment->GetPartitionId())) {
        return nullptr;
    }

    auto visitor = std::make_shared<SegmentVisitor>();
    visitor->SetSegment(segment);

    auto executor = [&](const snapshot::Field::Ptr& field, snapshot::FieldIterator* itr) -> Status {
        auto field_visitor = SegmentFieldVisitor::Build(ss, field, segment, segment_files);
        if (!field_visitor) {
            return Status::OK();
        }
        visitor->InsertField(field_visitor);

        return Status::OK();
    };

    auto iterator = std::make_shared<snapshot::FieldIterator>(ss, executor);
    iterator->Iterate();

    return visitor;
}

SegmentVisitor::Ptr
SegmentVisitor::Build(snapshot::ScopedSnapshotT ss, snapshot::ID_TYPE segment_id) {
    if (!ss) {
        return nullptr;
    }
    auto segment = ss->GetResource<snapshot::Segment>(segment_id);
    if (!segment) {
        return nullptr;
    }

    auto visitor = std::make_shared<SegmentVisitor>();
    visitor->SetSegment(segment);

    auto executor = [&](const snapshot::Field::Ptr& field, snapshot::FieldIterator* itr) -> Status {
        auto field_visitor = SegmentFieldVisitor::Build(ss, segment_id, field->GetID());
        if (!field_visitor) {
            return Status::OK();
        }
        visitor->InsertField(field_visitor);

        return Status::OK();
    };

    auto iterator = std::make_shared<snapshot::FieldIterator>(ss, executor);
    iterator->Iterate();

    return visitor;
}

std::string
SegmentVisitor::ToString() const {
    std::stringstream ss;
    ss << "SegmentVisitor[" << GetSegment()->GetID() << "]: " << (GetSegment()->IsActive() ? "" : "*") << "\n";
    auto& field_visitors = GetFieldVisitors();
    for (auto& fkv : field_visitors) {
        ss << "  Field[" << fkv.first << "]\n";
        auto& fe_visitors = fkv.second->GetElementVistors();
        for (auto& fekv : fe_visitors) {
            ss << "    FieldElement[" << fekv.first << "] ";
            auto file = fekv.second->GetFile();
            if (file) {
                ss << "SegmentFile [" << file->GetID() << "]: " << (file->IsActive() ? "" : "*") << "\n";
            } else {
                ss << "No SegmentFile!\n";
            }
        }
    }

    return ss.str();
}

}  // namespace engine
}  // namespace milvus
