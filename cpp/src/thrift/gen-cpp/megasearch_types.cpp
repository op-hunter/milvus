/**
 * Autogenerated by Thrift Compiler (0.12.0)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
#include "megasearch_types.h"

#include <algorithm>
#include <ostream>

#include <thrift/TToString.h>

namespace megasearch { namespace thrift {

int _kErrorCodeValues[] = {
  ErrorCode::SUCCESS,
  ErrorCode::CONNECT_FAILED,
  ErrorCode::PERMISSION_DENIED,
  ErrorCode::TABLE_NOT_EXISTS,
  ErrorCode::ILLEGAL_ARGUMENT,
  ErrorCode::ILLEGAL_RANGE,
  ErrorCode::ILLEGAL_DIMENSION
};
const char* _kErrorCodeNames[] = {
  "SUCCESS",
  "CONNECT_FAILED",
  "PERMISSION_DENIED",
  "TABLE_NOT_EXISTS",
  "ILLEGAL_ARGUMENT",
  "ILLEGAL_RANGE",
  "ILLEGAL_DIMENSION"
};
const std::map<int, const char*> _ErrorCode_VALUES_TO_NAMES(::apache::thrift::TEnumIterator(7, _kErrorCodeValues, _kErrorCodeNames), ::apache::thrift::TEnumIterator(-1, NULL, NULL));

std::ostream& operator<<(std::ostream& out, const ErrorCode::type& val) {
  std::map<int, const char*>::const_iterator it = _ErrorCode_VALUES_TO_NAMES.find(val);
  if (it != _ErrorCode_VALUES_TO_NAMES.end()) {
    out << it->second;
  } else {
    out << static_cast<int>(val);
  }
  return out;
}


Exception::~Exception() throw() {
}


void Exception::__set_code(const ErrorCode::type val) {
  this->code = val;
}

void Exception::__set_reason(const std::string& val) {
  this->reason = val;
}
std::ostream& operator<<(std::ostream& out, const Exception& obj)
{
  obj.printTo(out);
  return out;
}


uint32_t Exception::read(::apache::thrift::protocol::TProtocol* iprot) {

  ::apache::thrift::protocol::TInputRecursionTracker tracker(*iprot);
  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_I32) {
          int32_t ecast0;
          xfer += iprot->readI32(ecast0);
          this->code = (ErrorCode::type)ecast0;
          this->__isset.code = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 2:
        if (ftype == ::apache::thrift::protocol::T_STRING) {
          xfer += iprot->readString(this->reason);
          this->__isset.reason = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t Exception::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  ::apache::thrift::protocol::TOutputRecursionTracker tracker(*oprot);
  xfer += oprot->writeStructBegin("Exception");

  xfer += oprot->writeFieldBegin("code", ::apache::thrift::protocol::T_I32, 1);
  xfer += oprot->writeI32((int32_t)this->code);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("reason", ::apache::thrift::protocol::T_STRING, 2);
  xfer += oprot->writeString(this->reason);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(Exception &a, Exception &b) {
  using ::std::swap;
  swap(a.code, b.code);
  swap(a.reason, b.reason);
  swap(a.__isset, b.__isset);
}

Exception::Exception(const Exception& other1) : TException() {
  code = other1.code;
  reason = other1.reason;
  __isset = other1.__isset;
}
Exception& Exception::operator=(const Exception& other2) {
  code = other2.code;
  reason = other2.reason;
  __isset = other2.__isset;
  return *this;
}
void Exception::printTo(std::ostream& out) const {
  using ::apache::thrift::to_string;
  out << "Exception(";
  out << "code=" << to_string(code);
  out << ", " << "reason=" << to_string(reason);
  out << ")";
}

const char* Exception::what() const throw() {
  try {
    std::stringstream ss;
    ss << "TException - service has thrown: " << *this;
    this->thriftTExceptionMessageHolder_ = ss.str();
    return this->thriftTExceptionMessageHolder_.c_str();
  } catch (const std::exception&) {
    return "TException - service has thrown: Exception";
  }
}


TableSchema::~TableSchema() throw() {
}


void TableSchema::__set_table_name(const std::string& val) {
  this->table_name = val;
}

void TableSchema::__set_index_type(const int32_t val) {
  this->index_type = val;
}

void TableSchema::__set_dimension(const int64_t val) {
  this->dimension = val;
}

void TableSchema::__set_store_raw_vector(const bool val) {
  this->store_raw_vector = val;
}
std::ostream& operator<<(std::ostream& out, const TableSchema& obj)
{
  obj.printTo(out);
  return out;
}


uint32_t TableSchema::read(::apache::thrift::protocol::TProtocol* iprot) {

  ::apache::thrift::protocol::TInputRecursionTracker tracker(*iprot);
  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;

  bool isset_table_name = false;

  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRING) {
          xfer += iprot->readString(this->table_name);
          isset_table_name = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 2:
        if (ftype == ::apache::thrift::protocol::T_I32) {
          xfer += iprot->readI32(this->index_type);
          this->__isset.index_type = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 3:
        if (ftype == ::apache::thrift::protocol::T_I64) {
          xfer += iprot->readI64(this->dimension);
          this->__isset.dimension = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 4:
        if (ftype == ::apache::thrift::protocol::T_BOOL) {
          xfer += iprot->readBool(this->store_raw_vector);
          this->__isset.store_raw_vector = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  if (!isset_table_name)
    throw TProtocolException(TProtocolException::INVALID_DATA);
  return xfer;
}

uint32_t TableSchema::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  ::apache::thrift::protocol::TOutputRecursionTracker tracker(*oprot);
  xfer += oprot->writeStructBegin("TableSchema");

  xfer += oprot->writeFieldBegin("table_name", ::apache::thrift::protocol::T_STRING, 1);
  xfer += oprot->writeString(this->table_name);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("index_type", ::apache::thrift::protocol::T_I32, 2);
  xfer += oprot->writeI32(this->index_type);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("dimension", ::apache::thrift::protocol::T_I64, 3);
  xfer += oprot->writeI64(this->dimension);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("store_raw_vector", ::apache::thrift::protocol::T_BOOL, 4);
  xfer += oprot->writeBool(this->store_raw_vector);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(TableSchema &a, TableSchema &b) {
  using ::std::swap;
  swap(a.table_name, b.table_name);
  swap(a.index_type, b.index_type);
  swap(a.dimension, b.dimension);
  swap(a.store_raw_vector, b.store_raw_vector);
  swap(a.__isset, b.__isset);
}

TableSchema::TableSchema(const TableSchema& other3) {
  table_name = other3.table_name;
  index_type = other3.index_type;
  dimension = other3.dimension;
  store_raw_vector = other3.store_raw_vector;
  __isset = other3.__isset;
}
TableSchema& TableSchema::operator=(const TableSchema& other4) {
  table_name = other4.table_name;
  index_type = other4.index_type;
  dimension = other4.dimension;
  store_raw_vector = other4.store_raw_vector;
  __isset = other4.__isset;
  return *this;
}
void TableSchema::printTo(std::ostream& out) const {
  using ::apache::thrift::to_string;
  out << "TableSchema(";
  out << "table_name=" << to_string(table_name);
  out << ", " << "index_type=" << to_string(index_type);
  out << ", " << "dimension=" << to_string(dimension);
  out << ", " << "store_raw_vector=" << to_string(store_raw_vector);
  out << ")";
}


Range::~Range() throw() {
}


void Range::__set_start_value(const std::string& val) {
  this->start_value = val;
}

void Range::__set_end_value(const std::string& val) {
  this->end_value = val;
}
std::ostream& operator<<(std::ostream& out, const Range& obj)
{
  obj.printTo(out);
  return out;
}


uint32_t Range::read(::apache::thrift::protocol::TProtocol* iprot) {

  ::apache::thrift::protocol::TInputRecursionTracker tracker(*iprot);
  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRING) {
          xfer += iprot->readString(this->start_value);
          this->__isset.start_value = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 2:
        if (ftype == ::apache::thrift::protocol::T_STRING) {
          xfer += iprot->readString(this->end_value);
          this->__isset.end_value = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t Range::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  ::apache::thrift::protocol::TOutputRecursionTracker tracker(*oprot);
  xfer += oprot->writeStructBegin("Range");

  xfer += oprot->writeFieldBegin("start_value", ::apache::thrift::protocol::T_STRING, 1);
  xfer += oprot->writeString(this->start_value);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("end_value", ::apache::thrift::protocol::T_STRING, 2);
  xfer += oprot->writeString(this->end_value);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(Range &a, Range &b) {
  using ::std::swap;
  swap(a.start_value, b.start_value);
  swap(a.end_value, b.end_value);
  swap(a.__isset, b.__isset);
}

Range::Range(const Range& other5) {
  start_value = other5.start_value;
  end_value = other5.end_value;
  __isset = other5.__isset;
}
Range& Range::operator=(const Range& other6) {
  start_value = other6.start_value;
  end_value = other6.end_value;
  __isset = other6.__isset;
  return *this;
}
void Range::printTo(std::ostream& out) const {
  using ::apache::thrift::to_string;
  out << "Range(";
  out << "start_value=" << to_string(start_value);
  out << ", " << "end_value=" << to_string(end_value);
  out << ")";
}


RowRecord::~RowRecord() throw() {
}


void RowRecord::__set_vector_data(const std::string& val) {
  this->vector_data = val;
}
std::ostream& operator<<(std::ostream& out, const RowRecord& obj)
{
  obj.printTo(out);
  return out;
}


uint32_t RowRecord::read(::apache::thrift::protocol::TProtocol* iprot) {

  ::apache::thrift::protocol::TInputRecursionTracker tracker(*iprot);
  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;

  bool isset_vector_data = false;

  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRING) {
          xfer += iprot->readBinary(this->vector_data);
          isset_vector_data = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  if (!isset_vector_data)
    throw TProtocolException(TProtocolException::INVALID_DATA);
  return xfer;
}

uint32_t RowRecord::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  ::apache::thrift::protocol::TOutputRecursionTracker tracker(*oprot);
  xfer += oprot->writeStructBegin("RowRecord");

  xfer += oprot->writeFieldBegin("vector_data", ::apache::thrift::protocol::T_STRING, 1);
  xfer += oprot->writeBinary(this->vector_data);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(RowRecord &a, RowRecord &b) {
  using ::std::swap;
  swap(a.vector_data, b.vector_data);
}

RowRecord::RowRecord(const RowRecord& other7) {
  vector_data = other7.vector_data;
}
RowRecord& RowRecord::operator=(const RowRecord& other8) {
  vector_data = other8.vector_data;
  return *this;
}
void RowRecord::printTo(std::ostream& out) const {
  using ::apache::thrift::to_string;
  out << "RowRecord(";
  out << "vector_data=" << to_string(vector_data);
  out << ")";
}


QueryResult::~QueryResult() throw() {
}


void QueryResult::__set_id(const int64_t val) {
  this->id = val;
}

void QueryResult::__set_score(const double val) {
  this->score = val;
}
std::ostream& operator<<(std::ostream& out, const QueryResult& obj)
{
  obj.printTo(out);
  return out;
}


uint32_t QueryResult::read(::apache::thrift::protocol::TProtocol* iprot) {

  ::apache::thrift::protocol::TInputRecursionTracker tracker(*iprot);
  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_I64) {
          xfer += iprot->readI64(this->id);
          this->__isset.id = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 2:
        if (ftype == ::apache::thrift::protocol::T_DOUBLE) {
          xfer += iprot->readDouble(this->score);
          this->__isset.score = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t QueryResult::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  ::apache::thrift::protocol::TOutputRecursionTracker tracker(*oprot);
  xfer += oprot->writeStructBegin("QueryResult");

  xfer += oprot->writeFieldBegin("id", ::apache::thrift::protocol::T_I64, 1);
  xfer += oprot->writeI64(this->id);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("score", ::apache::thrift::protocol::T_DOUBLE, 2);
  xfer += oprot->writeDouble(this->score);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(QueryResult &a, QueryResult &b) {
  using ::std::swap;
  swap(a.id, b.id);
  swap(a.score, b.score);
  swap(a.__isset, b.__isset);
}

QueryResult::QueryResult(const QueryResult& other9) {
  id = other9.id;
  score = other9.score;
  __isset = other9.__isset;
}
QueryResult& QueryResult::operator=(const QueryResult& other10) {
  id = other10.id;
  score = other10.score;
  __isset = other10.__isset;
  return *this;
}
void QueryResult::printTo(std::ostream& out) const {
  using ::apache::thrift::to_string;
  out << "QueryResult(";
  out << "id=" << to_string(id);
  out << ", " << "score=" << to_string(score);
  out << ")";
}


TopKQueryResult::~TopKQueryResult() throw() {
}


void TopKQueryResult::__set_query_result_arrays(const std::vector<QueryResult> & val) {
  this->query_result_arrays = val;
}
std::ostream& operator<<(std::ostream& out, const TopKQueryResult& obj)
{
  obj.printTo(out);
  return out;
}


uint32_t TopKQueryResult::read(::apache::thrift::protocol::TProtocol* iprot) {

  ::apache::thrift::protocol::TInputRecursionTracker tracker(*iprot);
  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->query_result_arrays.clear();
            uint32_t _size11;
            ::apache::thrift::protocol::TType _etype14;
            xfer += iprot->readListBegin(_etype14, _size11);
            this->query_result_arrays.resize(_size11);
            uint32_t _i15;
            for (_i15 = 0; _i15 < _size11; ++_i15)
            {
              xfer += this->query_result_arrays[_i15].read(iprot);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.query_result_arrays = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t TopKQueryResult::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  ::apache::thrift::protocol::TOutputRecursionTracker tracker(*oprot);
  xfer += oprot->writeStructBegin("TopKQueryResult");

  xfer += oprot->writeFieldBegin("query_result_arrays", ::apache::thrift::protocol::T_LIST, 1);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_STRUCT, static_cast<uint32_t>(this->query_result_arrays.size()));
    std::vector<QueryResult> ::const_iterator _iter16;
    for (_iter16 = this->query_result_arrays.begin(); _iter16 != this->query_result_arrays.end(); ++_iter16)
    {
      xfer += (*_iter16).write(oprot);
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(TopKQueryResult &a, TopKQueryResult &b) {
  using ::std::swap;
  swap(a.query_result_arrays, b.query_result_arrays);
  swap(a.__isset, b.__isset);
}

TopKQueryResult::TopKQueryResult(const TopKQueryResult& other17) {
  query_result_arrays = other17.query_result_arrays;
  __isset = other17.__isset;
}
TopKQueryResult& TopKQueryResult::operator=(const TopKQueryResult& other18) {
  query_result_arrays = other18.query_result_arrays;
  __isset = other18.__isset;
  return *this;
}
void TopKQueryResult::printTo(std::ostream& out) const {
  using ::apache::thrift::to_string;
  out << "TopKQueryResult(";
  out << "query_result_arrays=" << to_string(query_result_arrays);
  out << ")";
}

}} // namespace
