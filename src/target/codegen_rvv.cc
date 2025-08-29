/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file target/codegen.cc
 */

 #include "codegen_rvv.h"
 #include <tvm/arith/analyzer.h>
 #include <tvm/runtime/registry.h>
 #include <tvm/tir/index_map.h>
 #include <tvm/tir/op.h>
 
 #include <cmath>
 #include <string>
 #include <utility>
 #include <vector>
 
 #include "../op/builtin.h"
 #include "../op/bulk_copy.h"
 #include "../op/gemm.h"
 // #include "../../target/source/ptx.h"
 
 namespace tvm {
 namespace codegen {
 
CodeGenTileLangRVV::CodeGenTileLangRVV() {
  restrict_keyword_ = "void*";
}
 
void CodeGenTileLangRVV::PrintFuncPrefix(
    std::ostream &os) { /*os << "extern \"C\" __global__ "; */
}
 
 class LaunchConfigExtractor : public tir::StmtVisitor {
  private:
  void VisitStmt_(const AttrStmtNode *op) final {
     if (op->attr_key == tir::attr::thread_extent) {
       IterVar iv = Downcast<IterVar>(op->node);
      if (iv->var->name_hint == "threadIdx.x" ||
          iv->thread_tag == "threadIdx.x") {
         threadIdx_x_ext = op->value;
      } else if (iv->var->name_hint == "threadIdx.y" ||
                 iv->thread_tag == "threadIdx.y") {
         threadIdx_y_ext = op->value;
      } else if (iv->var->name_hint == "threadIdx.z" ||
                 iv->thread_tag == "threadIdx.z") {
         threadIdx_z_ext = op->value;
       }
     }
     StmtVisitor::VisitStmt_(op);
   }
 
  public:
   PrimExpr threadIdx_x_ext = Integer(1);
   PrimExpr threadIdx_y_ext = Integer(1);
   PrimExpr threadIdx_z_ext = Integer(1);
 };
 
void CodeGenTileLangRVV::PrintExtraAttrs(const PrimFunc &f, std::ostream &os) {}
 
 std::string CodeGenTileLangRVV::Finish() {
    decl_stream << "typedef struct {\n";
    decl_stream << "    void* addr;\n";
    decl_stream << "    size_t size;\n";
    decl_stream << "    size_t shape[4];\n";
    decl_stream << "    size_t stride[4];\n";
    decl_stream << "} Tensor;\n";
    return CodeGenC::Finish();
 }
 
 /* no need to change */
void CodeGenTileLangRVV::VisitStmt_(const tir::ForNode *op) {
 
   if (op->kind == tir::ForKind::kUnrolled) {
     PrintIndent();
     stream << "#pragma unroll\n";
   }
  std::string extent =
      PrintExpr(arith::Analyzer().Simplify(op->extent + op->min));
   PrintIndent();
   std::string vid = AllocVarID(op->loop_var.get());
   std::string start = PrintExpr(op->min);
   stream << "for (";
   PrintType(op->loop_var.dtype(), stream);
   stream << ' ' << vid << " = " << start << "; " << vid << " < " << extent
         << "; ++" << vid << ") {\n";
   int for_scope = BeginScope();
   PrintStmt(op->body);
   this->EndScope(for_scope);
   PrintIndent();
   stream << "}\n";
 }
 
void CodeGenTileLangRVV::BindThreadIndex(const IterVar &iv) {
   ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
 }
 
void CodeGenTileLangRVV::PrintType(DataType t, std::ostream &os) { // NOLINT(*)
   int lanes = t.lanes();
   if (t.is_handle()) {
     ICHECK(t.is_scalar()) << "do not yet support vector types";
     os << "void*";
     return;
   }
 
   if (t.is_void()) {
     os << "void";
     return;
   }
 
   if (t == tl::cuTensorMapType()) {
     os << "CUtensorMap";
     return;
   }
 
   bool fail = false;
   if (t.is_float()) {
     switch (t.bits()) {
       case 16:
         if (t.is_scalar()) {
           os << "half_t";
         } else if (lanes <= 8) {
           // Emit CUDA code to access fp16 vector elements.
           //
           // half4 is stored as uint2
           //
           // h4.x is emitted as *(half2*)(&(u2.x)).x
           // h4.y is emitted as *(half2*)(&(u2.x)).y
           // h4.z is emitted as *(half2*)(&(u2.y)).x
           // h4.w is emitted as *(half2*)(&(u2.y)).y
           //
           ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
           os << "uint" << lanes / 2;
         } else {
           fail = true;
         }
         break;
       case 32:
         if (lanes <= 4) {
           os << "float";
         } else if (lanes <= 8) {
           // Emit CUDA code to access fp32 vector elements for 4 < lanes <= 8.
           //
           // float8 is stored as ulonglong4
           //
           // f8.v1 is emitted as *(float2*)(&(ul4.x)).x
           // f8.v2 is emitted as *(float2*)(&(ul4.x)).y
           //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for float type with lanes > 4";
           os << "ulonglong" << lanes / 2;
         } else {
           fail = true;
         }
         break;
       case 64:
         os << "double";
         break;
       default:
         fail = true;
         break;
     }
    if (!fail && (t.is_scalar() || t.bits() == 16))
      return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32))
      return;
     if (!fail && (lanes >= 2 && lanes <= 4)) {
       os << lanes;
       return;
     }
   } else if (t.is_bfloat16()) {
     if (t.is_scalar()) {
       os << "bfloat16_t";
     } else if (lanes <= 8) {
       ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
       os << "uint" << lanes / 2;
     } else {
       fail = true;
     }
    if (!fail)
      return;
   } else if (t.is_float8()) {
     if (t.is_scalar()) {
      os << "unsigned char"; // __nv_fp8_storage_t is an alias of unsigned char
     } else if (lanes == 2) {
      os << "unsigned short int"; // __nv_fp8x2_storage_t is an alias of
                                  // unsigned short
     } else if (lanes == 4) {
      os << "unsigned int"; // __nv_fp8x4_storage_t is an alias of unsigned int
     } else {
       fail = true;
     }
    if (!fail)
      return;
   } else if (t == DataType::Bool()) {
     os << "bool";
     return;
   } else if (t.is_vector_bool()) {
     // CUDA does not support bool vectors.
     // Use ushort vectors to represent instead.
     int n = t.lanes();
     if (n <= 4) {
       os << "ushort" << n;
       return;
     }
   } else if (t.is_uint() || t.is_int()) {
     if (t.is_uint()) {
       os << "u";
     }
     switch (t.bits()) {
       case 1: {
         if (t.is_scalar()) {
           os << "int";
           return;
         } else if (t.lanes() == 8) {
           os << "int8_t";
           return;
         } else if (t.lanes() == 16) {
           os << "int16_t";
           return;
         } else if (t.lanes() == 32) {
           os << "int";
           return;
         } else {
           LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
         }
       }
       case 4: {
         if (t.is_scalar()) {
           os << "int";
           return;
         } else if (t.lanes() == 4) {
           os << "int16_t";
           return;
         } else if (t.lanes() == 8) {
           // directly 8 4-bit int in integer.
           os << "int";
           return;
         } else if (t.lanes() == 16) {
           os << "int2";
           return;
         } else if (t.lanes() == 32) {
           os << "int4";
           return;
         } else if (t.lanes() == 64) {
           os << "int8";
           return;
         } else {
           LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
         }
       }
       case 8: {
         if (t.lanes() == 4) {
           // directly 4 8 bit int in integer.
 
           // We use int for int8x4 instead of char4 because using char4 is
           // likely to produce extra instructions to pack four int8 elements
           // into 32-bit data.
           os << "int";
           return;
         } else if (t.lanes() == 8) {
           os << "int2";
           return;
         } else if (t.lanes() == 16) {
           os << "int4";
           return;
         } else if (!t.is_uint() && t.is_scalar()) {
           os << "signed char";
           break;
         } else {
           os << "char";
           break;
         }
       }
       case 16: {
         if (t.is_scalar()) {
           os << "short";
         } else if (t.lanes() <= 4) {
           os << "short" << lanes;
         } else if (t.lanes() <= 8) {
           // Emit CUDA code to access int16 vector elements.
           //
           // short4 is stored as int2
           //
           // s4.x is emitted as *(short2*)(&(i2.x)).x
           // s4.y is emitted as *(short2*)(&(i2.x)).y
           // s4.z is emitted as *(short2*)(&(i2.y)).x
           // s4.w is emitted as *(short2*)(&(i2.y)).y
           //
        ICHECK_EQ(t.lanes() % 2, 0)
            << "only support even lane for shorT type with lanes > 4";
           os << "int" << t.lanes() / 2;
         } else {
           fail = true;
         }
         if (!fail) {
           return;
         }
         break;
       }
       case 32: {
         if (t.is_scalar()) {
           os << "int";
         } else if (t.lanes() <= 4) {
           os << "int" << t.lanes();
         } else if (t.lanes() <= 8) {
           // Emit CUDA code to access int32 vector elements for 4 < lanes <= 8.
           //
           // int8 is stored as longlong4
           //
           // i8.v1 is emitted as *(int2*)(&(l4.x)).x
           // i8.v2 is emitted as *(int2*)(&(l4.x)).y
           //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for int32 type with lanes > 4";
           os << "longlong" << lanes / 2;
         } else {
           fail = true;
         }
         if (!fail) {
           return;
         }
         break;
       }
       case 64: {
         if (t.is_scalar()) {
           os << "int64_t";
         } else if (t.lanes() == 2) {
           os << "longlong2";
         } else if (t.lanes() == 3) {
           os << "longlong3";
         } else if (t.lanes() == 4) {
           os << "longlong4";
         }
         return;
       }
       default:
         fail = true;
         break;
     }
     if (!fail && lanes == 1) {
       return;
     }
     if (!fail && (lanes >= 2 && lanes <= 4)) {
       os << lanes;
       return;
     }
   }
   LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
 }
 
void CodeGenTileLangRVV::PrintVecBinaryOp(const std::string &op, DataType t,
                                          PrimExpr lhs, PrimExpr rhs,
                                          std::ostream &os) { // NOLINT(*)
   // Delcare the result.
   std::string sret = name_supply_->FreshName("_");
   this->PrintIndent();
   this->PrintType(t, stream);
   stream << ' ' << sret << ";\n";
   int ssa_scope = BeginScope();
   {
     // Unpack into individual ops.
     std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.dtype());
     std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.dtype());
 
     for (int i = 0, lanes = t.lanes(); i < lanes; ++i) {
       std::ostringstream value_temp;
       if (isalpha(op[0])) {
         value_temp << op << "(";
         PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
         value_temp << ", ";
         PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
         value_temp << ")";
       } else {
         value_temp << "(";
         PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
         value_temp << op;
         PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
         value_temp << ")";
       }
       PrintVecElemStore(sret, t, i, value_temp.str());
     }
   }
   EndScope(ssa_scope);
   os << sret;
 }
 
void CodeGenTileLangRVV::PrintVecElemLoad(const std::string &vec, DataType t,
                                          int i,
                                          std::ostream &os) { // NOLINT(*)
   if (t.is_scalar()) {
     os << vec;
     return;
   }
 
   static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8                        ? 16
                        : (t.bits() == 16 || t.bits() == 32) ? 8
                                                             : 4));
   if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
     std::string type_name = t.is_int() ? "char" : "unsigned char";
     if (t.lanes() == 2 || t.lanes() == 3) {
       os << vec << "." << access[i % t.lanes()];
     } else {
       std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
       os << "((" << type_name << ")(" << ac << " >> " << i % 4 * 8 << "))";
     }
   } else if (t.is_float16()) {
    os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
       << access[i % 2];
   } else if (t.is_bfloat16()) {
    os << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->"
       << access[i % 2];
   } else if (t.lanes() > 4 && t.lanes() <= 8) {
     std::string type_name;
     if (t.bits() == 16) {
       if (t.is_int()) {
         type_name = "short";
       } else if (t.is_uint()) {
         type_name = "ushort";
       }
     } else if (t.bits() == 32) {
       if (t.is_int()) {
         type_name = "int";
       } else if (t.is_uint()) {
         type_name = "uint";
       } else if (t.is_float()) {
         type_name = "float";
       }
     }
     ICHECK(!type_name.empty());
    os << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
       << ")))->" << access[i % 2];
   } else {
     os << vec << "." << access[i];
   }
 }
 
void CodeGenTileLangRVV::PrintVecElemStore(const std::string &vec, DataType t,
                                           int i, const std::string &value) {
   this->PrintIndent();
   static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8                        ? 16
                        : (t.bits() == 16 || t.bits() == 32) ? 8
                                                             : 4));
   if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
     if (t.lanes() == 2 || t.lanes() == 3) {
      stream << vec << '.' << access[i % t.lanes()] << "="
             << "(" << value << ");\n";
     } else {
       std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
       stream << ac << "=";
       // Do not read the first undef lane.
       if (i != 0) {
         stream << ac << " & ~(0x000000ff << " << i % 4 * 8 << ") |";
       }
       stream << "(" << value << " << " << i % 4 * 8 << ");\n";
     }
   } else if (t.is_float16()) {
    stream << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
           << access[i % 2] << " = " << value << ";\n";
   } else if (t.is_bfloat16()) {
    stream << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->"
           << access[i % 2] << " = " << value << ";\n";
   } else if (t.lanes() > 4 && t.lanes() <= 8) {
     std::string type_name;
     if (t.bits() == 16) {
       if (t.is_int()) {
         type_name = "short";
       } else if (t.is_uint()) {
         type_name = "ushort";
       }
     } else if (t.bits() == 32) {
       if (t.is_int()) {
         type_name = "int";
       } else if (t.is_uint()) {
         type_name = "uint";
       } else if (t.is_float()) {
         type_name = "float";
       }
     }
     ICHECK(!type_name.empty());
    stream << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
           << ")))->" << access[i % 2] << " = " << value << ";\n";
   } else {
     stream << vec << "." << access[i] << " = " << value << ";\n";
   }
 }
 
void CodeGenTileLangRVV::PrintStorageSync(const CallNode *op) {
  const std::string &sync = op->args[0].as<StringImmNode>()->value;
   if (sync == "warp") {
     // DO nothing.
   } else if (sync == "shared" || sync == "shared.dyn") {
     this->PrintIndent();
     this->stream << "__syncthreads();\n";
   }
 }
 
void CodeGenTileLangRVV::PrintStorageScope(const std::string &scope,
                                           std::ostream &os) { // NOLINT(*)
  ICHECK_NE(scope, "global")
      << "Cannot allocate global memory when targeting CUDA. You must pass "
                                 "all global arrays as input instead";
   if (scope == "shared") {
     os << "__shared__ ";
   } else if (scope == "shared.dyn") {
     os << "extern __shared__ __align__(1024) ";
   }
 }
 
std::string CodeGenTileLangRVV::CastFromTo(std::string value, DataType from,
                                           DataType target) {
  if (from == target)
    return value;
   std::ostringstream os;
   os << "((";
   this->PrintType(target, os);
   os << ")";
  if (from.is_float16() && (target.is_int() || target.is_uint()) &&
      target.bits() == 8) {
     os << "(";
     if (target.is_uint()) {
       os << "u";
     }
     os << "int)";
   }
   os << value << ")";
   return os.str();
 }
 
void CodeGenTileLangRVV::VisitExpr_(const CastNode *op, std::ostream &os) {
   DataType from_ty = op->value.dtype();
   DataType target_ty = op->dtype;
   ICHECK_EQ(target_ty.lanes(), from_ty.lanes());
 
   // Emit simple C-style type conversion.
  if (from_ty.is_scalar())
    return CodeGenC::VisitExpr_(op, os);
 
   // We could emit make_float4 like calls, but the emitted code looks
   // too compact to read. Emit this as vectorized unary ops.
   std::string sret = name_supply_->FreshName("_");
   this->PrintIndent();
   this->PrintType(target_ty, stream);
   stream << ' ' << sret << ";\n";
   {
     std::string src = SSAGetID(PrintExpr(op->value), from_ty);
     for (int i = 0, lanes = from_ty.lanes(); i < lanes; ++i) {
       std::ostringstream val;
       val << "(";
       PrintType(target_ty.element_of(), val);
       val << ")(";
       PrintVecElemLoad(src, from_ty, i, val);
       val << ")";
       PrintVecElemStore(sret, target_ty, i, val.str());
     }
   }
   os << sret;
 }
 
void CodeGenTileLangRVV::PrintCallExtern(Type ret_type, String global_symbol,
                                         const Array<PrimExpr> &args,
                                         bool skip_first_arg,
                                         std::ostream &os) { // NOLINT(*)
   DataType ret_dtype = GetRuntimeDataType(ret_type);
   if (ret_dtype.is_vector()) {
     //
     // Emit an unsupported vector call
     //
     // v = intrin_f((float4*)A[0], (float4*)B[0])
     //
     // as
     //
     // float4 __ret;
     // {
     //   float4 __arg0 = ((float4*)A)[0];
     //   float4 __arg1 = ((float4*)B)[0];
     //   __ret.x = intrin_f(__arg0.x, __arg1.x);
     //   __ret.y = intrin_f(__arg0.y, __arg1.y);
     //   __ret.z = intrin_f(__arg0.z, __arg1.z);
     //   __ret.w = intrin_f(__arg0.w, __arg1.w);
     // }
     // v = __ret;
     //
     // Declare the result vector.
     std::string sret = name_supply_->FreshName("_");
     this->PrintIndent();
     this->PrintType(ret_dtype, stream);
     stream << ' ' << sret << ";\n";
     {
       // Load arguments.
       std::vector<std::string> sargs;
       size_t arg_begin = static_cast<size_t>(skip_first_arg);
       for (size_t i = arg_begin; i < args.size(); ++i) {
         std::string val = SSAGetID(PrintExpr(args[i]), args[i].dtype());
         sargs.push_back(std::move(val));
       }
 
       // Emit a scalar call for each lane.
       for (int i = 0; i < ret_dtype.lanes(); ++i) {
         std::ostringstream scall;
         scall << global_symbol << "(";
         for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0)
            scall << ", ";
           PrintVecElemLoad(sargs[j], args[arg_begin + j].dtype(), i, scall);
         }
         scall << ")";
         PrintVecElemStore(sret, ret_dtype, i, scall.str());
       }
     }
     os << sret;
   } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg,
                              os);
   }
 }
 
 // Print a reference expression to a buffer.
std::string CodeGenTileLangRVV::GetBufferRef(DataType t,
                                             const BufferNode *buffer,
                                             PrimExpr index) {
  const VarNode *buffer_var = buffer->data.get();
   std::ostringstream os;
   std::string vid = GetVarID(buffer_var);
   std::string scope;
   if (alloc_storage_scope_.count(buffer_var)) {
     scope = alloc_storage_scope_.at(buffer_var);
   }
   // bool is_vol = IsVolatile(buffer_var);
   // always false for tl cutlass backend.
   bool is_vol = false;
 
   auto ptr_cast = [this, is_vol, scope](DataType pointed_to) {
     std::ostringstream ptr_os;
     ptr_os << "(";
     if (is_vol) {
       ptr_os << "volatile ";
     }
     if (!scope.empty() && IsScopePartOfType()) {
       PrintStorageScope(scope, ptr_os);
     }
     PrintType(pointed_to, ptr_os);
     ptr_os << "*)";
     return ptr_os.str();
   };
 
   DataType buffer_element_dtype = buffer->dtype;
 
   std::string buffer_str = vid;
   if (!HandleTypeMatch(buffer_var, buffer_element_dtype) || is_vol) {
     std::stringstream temp;
     temp << "(" << ptr_cast(buffer_element_dtype) << vid << ")";
     buffer_str = temp.str();
   }
 
   std::string index_str = PrintExpr(index);
   if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
     // This is a special case, because CodegenCUDA::PrintType()
     // returns "int" for bool and for 4-bit integers. In most cases,
     // we divide by the number of lanes to determine the index.
     // However, the backing type for scalar int4 and scalar bool is
     // int32.  Therefore, we need to divide by the ratio of their
     // sizes in that case.
     int div_factor = (t.lanes() == 1) ? (32 / t.bits()) : t.lanes();
 
     os << "*("
        << "(" << ptr_cast(t) << vid << ")"
        << " + " << index_str << " / " << div_factor << ")";
   } else if (t == buffer_element_dtype) {
     os << buffer_str << "[" << index_str << "]";
   } else {
     os << "*" << ptr_cast(t) << "(" << buffer_str << " + " << index_str << ")";
   }
 
   return os.str();
 }
 
inline std::string vector2string(const std::vector<int> &vec) {
   std::string ret = "{";
  for (auto &v : vec) {
     ret += std::to_string(v) + ", ";
   }
   ret[ret.size() - 2] = '}';
   return ret;
 }
 
void CodeGenTileLangRVV::VisitExpr_(const CallNode *op, std::ostream &os) {
  auto process_stride = [&,
                         this](const std::vector<int> &src0_shape,
                               const std::vector<int> &src1_shape,
                               const std::string &src0, const std::string &src1,
                               const std::string &dtype) -> std::stringstream {
    std::stringstream src1_stride;
    if (src1_shape[1] == 1 && src0_shape[1] != 1) {
      // void tpu_aligned_stride(dim4 *stride, int start_idx, const dim4 *shape,
      // data_type_t dtype) we must construct explicit stride
      // 1. initialize a dim4 struct
      this->PrintIndent();
      this->stream << "dim4 " << src1 << "_stride;\n";
      // 2. call tpu_aligned_stride
      this->PrintIndent();
      this->stream << "tpu_aligned_stride(&" << src1 << "_stride, 0, &" << src1
                   << ".shape, " << dtype << ");\n";
      this->PrintIndent();
      this->stream << src1 << "_stride.w = 0;\n";
      src1_stride << "&" << src1 << "_stride, ";
    } else if (src1_shape[1] == src0_shape[1]) {
      src1_stride << "(" << src1 << ".default_stride ? NULL : &" << src1
                  << ".stride), ";
    }
    return src1_stride;
  };

  auto handle_rvv_elementwise = [&, this](const std::string& rvv_op, bool is_binary) {
    auto dst = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
    auto src0 = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
    auto src1 = var_idmap_[op->args[3].as<CallNode>()->args[1].as<VarNode>()];
    auto shape = buffer_shape[dst];
    auto dtype = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;

    std::string rvv_type;
    uint rvv_eew;
    uint rvv_lmul;
    std::string rvv_vec_type;     // 新增：向量类型存储
    std::string type_suffix;       // 新增：intrinsic类型后缀

    if (dtype == DataType::Float(16)) {
      rvv_type = "_Float16";
      rvv_eew = 16;
      rvv_lmul = 1;
      rvv_vec_type = "vfloat16m1_t";  // 正确向量类型
      type_suffix = "f16m1";          // intrinsic后缀
    } else if (dtype == DataType::Float(32)) {
      rvv_type = "float";
      rvv_eew = 32;
      rvv_lmul = 1;
      rvv_vec_type = "vfloat32m1_t";
      type_suffix = "f32m1";
    } else if (dtype == DataType::Int(32)) {
      rvv_type = "int32_t";
      rvv_eew = 32;
      rvv_lmul = 1;
      rvv_vec_type = "vint32m1_t";
      type_suffix = "i32m1";
    } else if (dtype == DataType::Int(16)) {
      rvv_type = "int16_t";
      rvv_eew = 16;
      rvv_lmul = 1;
      rvv_vec_type = "vint16m1_t";
      type_suffix = "i16m1";
    } else if (dtype == DataType::UInt(8)) {
      rvv_type = "uint8_t";
      rvv_eew = 8;
      rvv_lmul = 1;
      rvv_vec_type = "vuint8m1_t";
      type_suffix = "u8m1";
    } else {
      throw std::runtime_error("Unsupported dtype for fill: ");
    }
    this->PrintIndent();
    this->stream << "{\n";
    this->PrintIndent();
    this->stream << "  " << rvv_type << "* dst_ptr = (" << rvv_type << "*)" << dst << ".addr;\n";
    this->PrintIndent();
    this->stream << "  " << rvv_type << "* src0_ptr = (" << rvv_type << "*)" << src0 << ".addr;\n";
    this->PrintIndent();
    this->stream << "  " << rvv_type << "* src1_ptr = (" << rvv_type << "*)" << src1 << ".addr;\n";
    this->PrintIndent();
    this->stream << "  size_t num_rows = " << dst << ".shape[1];\n";
    this->PrintIndent();
    this->stream << "  size_t row_size = " << dst << ".shape[3];\n";
    this->PrintIndent();
    this->stream << "  size_t vl;\n";

    this->PrintIndent();
    this->stream << "  for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {\n";
    this->PrintIndent();
    this->stream << "    float scale_val = src1_ptr[row_idx];\n";
    this->PrintIndent();
    this->stream << "    for (size_t col_offset = 0; col_offset < row_size; ) {\n";
    this->PrintIndent();
    this->stream << "      vl = __riscv_vsetvl_e" << rvv_eew << "m" << rvv_lmul << "(row_size - col_offset);\n";
    this->PrintIndent();
    if (dtype.is_float()) {
      this->PrintIndent();
      this->stream << "      " << rvv_vec_type << " v_src0 = __riscv_vle" << rvv_eew << "_v_" << "f" << rvv_eew << "m" << rvv_lmul << "(src0_ptr + row_idx * row_size + col_offset, vl);\n";
    } else {
      this->PrintIndent();
      this->stream << "      " << rvv_vec_type << " v_src0 = __riscv_vle" << rvv_eew << "_v_" << "e" << rvv_eew << "m" << rvv_lmul << "(row_idx * row_size + col_offset, vl);\n";
    }
    if (dtype.is_float()) {
      this->PrintIndent();
      this->stream << "      " << rvv_vec_type << " v_dst = __riscv_vf" << rvv_op << "_vf_" << "f" << rvv_eew << "m" << rvv_lmul << "(v_src0, scale_val, vl);\n";
    } else {
      this->PrintIndent();
      this->stream << "      " << rvv_vec_type << " v_dst = __riscv_vf" << rvv_op << "_vf_" << "e" << rvv_eew << "m" << rvv_lmul << "(v_src0, scale_val, vl);\n";
    }
    this->PrintIndent();
    const char* store_sign = dtype.is_float() ? "f" : (dtype.is_uint() ? "u" : "i");
    this->stream << "      __riscv_vse" << rvv_eew << "_v_"
                << store_sign << rvv_eew << "m" << rvv_lmul
                << "((" << rvv_type << "*)((uint8_t*)dst_ptr +  row_idx * row_size + col_offset), v_dst, vl);\n";
    this->PrintIndent();
    this->stream << "        col_offset += vl;\n";
    this->PrintIndent();
    this->stream << "    }\n";
    this->PrintIndent();
    this->stream << "  }\n";
    this->PrintIndent();
    this->stream << "  asm volatile (\"fence ow, ow\" ::: \"memory\");\n";
    this->PrintIndent();
    this->stream << "}\n";
  };
  auto handle_rvv_elementwise_const = [&, this](const std::string& rvv_op) {
    auto dst = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
    auto src0 = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
    auto shape = buffer_shape[dst];
    auto dtype = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
    float const_val = Downcast<FloatImm>(op->args[3])->value;

    std::string rvv_type;
    uint rvv_eew;
    uint rvv_lmul;
    std::string rvv_vec_type;     // 新增：向量类型存储
    std::string type_suffix;       // 新增：intrinsic类型后缀

    if (dtype == DataType::Float(16)) {
      rvv_type = "_Float16";
      rvv_eew = 16;
      rvv_lmul = 1;
      rvv_vec_type = "vfloat16m1_t";  // 正确向量类型
      type_suffix = "f16m1";          // intrinsic后缀
    } else if (dtype == DataType::Float(32)) {
      rvv_type = "float";
      rvv_eew = 32;
      rvv_lmul = 1;
      rvv_vec_type = "vfloat32m1_t";
      type_suffix = "f32m1";
    } else if (dtype == DataType::Int(32)) {
      rvv_type = "int32_t";
      rvv_eew = 32;
      rvv_lmul = 1;
      rvv_vec_type = "vint32m1_t";
      type_suffix = "i32m1";
    } else if (dtype == DataType::Int(16)) {
      rvv_type = "int16_t";
      rvv_eew = 16;
      rvv_lmul = 1;
      rvv_vec_type = "vint16m1_t";
      type_suffix = "i16m1";
    } else if (dtype == DataType::UInt(8)) {
      rvv_type = "uint8_t";
      rvv_eew = 8;
      rvv_lmul = 1;
      rvv_vec_type = "vuint8m1_t";
      type_suffix = "u8m1";
    } else {
      throw std::runtime_error("Unsupported dtype for fill: ");
    }
    this->PrintIndent();
    this->stream << "{\n";
    this->PrintIndent();
    this->stream << "  " << rvv_type << "* dst_ptr = (" << rvv_type << "*)" << dst << ".addr;\n";
    this->PrintIndent();
    this->stream << "  " << rvv_type << "* src0_ptr = (" << rvv_type << "*)" << src0 << ".addr;\n";

    this->PrintIndent();
    this->stream << "  size_t total_elements = " << dst << ".shape[1] * " << dst <<".shape[3];\n";
    this->PrintIndent();
    this->stream << "  size_t vl;\n";

    this->PrintIndent();
    this->stream << "  for (size_t offset = 0; offset < total_elements; offset += vl) {\n";

    this->PrintIndent();
    this->stream << "    vl = __riscv_vsetvl_e" << rvv_eew << "m" << rvv_lmul << "(total_elements - offset);\n";

    if (dtype.is_float()) {
      this->PrintIndent();
      this->stream << "    " << rvv_vec_type << " v_src0 = __riscv_vle" << rvv_eew << "_v_" << "f" << rvv_eew << "m" << rvv_lmul << "(src0_ptr + offset, vl);\n";
      this->PrintIndent();
      this->stream << "    " << rvv_type << " temp_const = " << const_val << ";\n";
      this->PrintIndent();
      this->stream << "    " << rvv_vec_type << " v_src1 = __riscv_vfmv_v_f_" << "f" << rvv_eew << "m" << rvv_lmul << "(temp_const, vl);\n";
    } else {
      this->PrintIndent();
      this->stream << "    " << rvv_vec_type << " v_src0 = __riscv_vle" << rvv_eew << "_v_" << "e" << rvv_eew << "m" << rvv_lmul << "(src0_ptr + offset, vl);\n";
      this->PrintIndent();
      this->stream << "    " << rvv_type << " temp_const = " << static_cast<int>(const_val) << ";\n";
      this->PrintIndent();
      this->stream << "    " << rvv_vec_type << " v_src1 = __riscv_vmv_v_x_" << "e" << rvv_eew << "m" << rvv_lmul << "(temp_const, vl);\n";
    }

    this->PrintIndent();
    if (dtype.is_float()) {
      this->stream << "    " << rvv_vec_type << " v_dst = __riscv_vf" << rvv_op << "_vv_" << "f" << rvv_eew << "m" << rvv_lmul << "(v_src0, v_src1, vl);\n";
    } else {
      this->stream << "    " << rvv_vec_type << " v_dst = __riscv_v" << rvv_op << "_vv_" << "e" << rvv_eew << "m" << rvv_lmul << "(v_src0, v_src1, vl);\n";
    }
    this->PrintIndent();
    const char* store_sign = dtype.is_float() ? "f" : (dtype.is_uint() ? "u" : "i");
    this->stream << "    __riscv_vse" << rvv_eew << "_v_"
                << store_sign << rvv_eew << "m" << rvv_lmul  // 动态生成正确后缀
                << "((" << rvv_type << "*)((uint8_t*)dst_ptr + offset * sizeof(" << rvv_type << ")), v_dst, vl);\n";
    this->PrintIndent();
    this->stream << "  }\n";
    this->PrintIndent();
    this->stream << "  asm volatile (\"fence ow, ow\" ::: \"memory\");\n";
    this->PrintIndent();
    this->stream << "}\n";
  };
   std::vector<std::string> inst;
   if (op->op.same_as(builtin::call_extern())) {
    std::string op_name = Downcast<StringImm>(op->args[0])->value;
    if (op_name == "rvv.copy") {
      struct CopyInfo {
        std::string min_expr;
        std::string scope;
        std::string dtype;
        uint rvv_eew;
        uint rvv_lmul;
        size_t total_elements;
        size_t elem_bytes;
        std::string tensor_id;
        std::string rvv_type0;
      };

      auto process_copy = [&, this](const tl::RegionOp& src) -> CopyInfo {
        auto src_buffer = src.GetBuffer();
        auto src_ranges = src.GetRanges();
        auto src_id = var_idmap_[src_buffer->data.get()];
        if (src_id.empty()){
          src_id = this->parameter_map[src_buffer->name];
        }
        
        std::string dtype;
        std::string rvv_type;
        std::string rvv_type0;
        uint rvv_eew;
        uint rvv_lmul;
        size_t elem_bytes = 0;
        if (src_buffer->dtype == DataType::Float(16)) {
          dtype = "_Float16";
          elem_bytes = 2;
          rvv_eew = 16;
          rvv_lmul = 1;
          rvv_type0 = "vfloat16m1_t";
          rvv_type = "f";
        } else if (src_buffer->dtype == DataType::Float(32)) {
          dtype = "float";
          elem_bytes = 4;
          rvv_eew = 32;
          rvv_lmul = 1;
          rvv_type0 = "vfloat32m1_t";
          rvv_type = "f";
        } else if (src_buffer->dtype == DataType::Int(8)) {
          dtype = "int8_t";
          elem_bytes = 1;
          rvv_eew = 8;
          rvv_lmul = 1;
          rvv_type0 = "vint8m1_t";
          rvv_type = "i";
        } else if (src_buffer->dtype == DataType::Int(16)) {
          dtype = "int16_t";
          elem_bytes = 2;
          rvv_eew = 16;
          rvv_lmul = 1;
          rvv_type0 = "vint16m1_t";
        } else if (src_buffer->dtype == DataType::Int(32)) {
          dtype = "int32_t";
          elem_bytes = 4;
          rvv_eew = 32;
          rvv_lmul = 1;
          rvv_type0 = "vint32m1_t";
        } else if (src_buffer->dtype == DataType::UInt(8)) {
          dtype = "uint8_t";
          elem_bytes = 1;
          rvv_eew = 8;
          rvv_lmul = 1;
          rvv_type0 = "vuint8m1_t";
        } else if (src_buffer->dtype == DataType::UInt(16)) {
          dtype = "uint16_t";
          elem_bytes = 2;
          rvv_eew = 16;
          rvv_lmul = 1;
          rvv_type0 = "vuint16m1_t";
        } else if (src_buffer->dtype == DataType::UInt(32)) {
          dtype = "uint32_t";
          elem_bytes = 4;
          rvv_eew = 32;
          rvv_lmul = 1;
          rvv_type0 = "vuint32m1_t";
        } else {
          LOG(FATAL) << "Unsupported data type: " << src_buffer->dtype;
        }
        size_t total_elements = 1;
        std::vector<size_t> shape_dims;
        for (auto& sr : src_ranges) {
          auto extent = sr->extent.as<IntImmNode>()->value;
          shape_dims.push_back(extent);
          total_elements *= extent;
        }
        
        std::string min_expr;
        if (src_buffer.scope() == "global") {
          std::string src_strides;
          auto strides = buffer_stride[src_buffer->name];
          src_strides = vector2string(strides);
          std::vector<int> stride_map;
          if (src_ranges.size() == 2) {
            stride_map = {1, 3};
          } else if (src_ranges.size() == 4) {
            stride_map = {0, 1, 2, 3};
          } else {
            LOG(FATAL) << "Unsupported number of ranges: " << src_ranges.size();
          }
          for (int i=0; i < src_ranges.size(); i++){
              auto sr = src_ranges[i];
              min_expr += "("+PrintExpr(sr->min) + ") * " + std::to_string(strides[stride_map[i]]) + "+";
          }
          min_expr[min_expr.size() - 1] = ' ';
          min_expr = "(" + min_expr + ")" + " * " + std::to_string(elem_bytes) + " + (uint8_t*)" + src_id + ".addr";
        } else if (src_buffer.scope() == "shared.dyn") {
          min_expr = "(uint8_t*)" + var_idmap_[src_buffer->data.get()] + ".addr";
        }

      return CopyInfo{
        min_expr,
        src_buffer.scope().operator std::string(),
        dtype,
        rvv_eew,
        rvv_lmul,
        total_elements,
        elem_bytes,
        src_id,
        rvv_type0
      };
    };
      
      // 创建一个空的BufferMap
      tl::BufferMap buffer_map;
      
      tl::RegionOp src = tl::RegionOp(op->args[1].as<CallNode>()->args, buffer_map);
      tl::RegionOp dst = tl::RegionOp(op->args[2].as<CallNode>()->args, buffer_map);
      auto src_info = process_copy(src);
      auto dst_info = process_copy(dst);
      
      size_t total_bytes = src_info.total_elements * src_info.elem_bytes;

      this->PrintIndent();
      this->stream << "{\n";
      this->PrintIndent();
      this->stream << "  size_t min_cols = " << src_info.tensor_id << ".shape[3] < " 
                   << dst_info.tensor_id << ".shape[3] ? " 
                   << src_info.tensor_id << ".shape[3] : " 
                   << dst_info.tensor_id << ".shape[3];\n";
      this->PrintIndent();
      this->stream << "  uint8_t* src_ptr = " << src_info.min_expr << ";\n";
      this->PrintIndent();
      this->stream << "  uint8_t* dst_ptr = " << dst_info.min_expr << ";\n";
      this->PrintIndent();
      this->stream << "  int t = (" << src_info.tensor_id << ".shape[1] > " << dst_info.tensor_id << ".shape[1]) ? " 
                   << dst_info.tensor_id << ".shape[1] : " << src_info.tensor_id << ".shape[1];\n";
      this->PrintIndent();
      this->stream << "  for (int i = 0; i < t; i++) {\n";
      this->PrintIndent();
      this->stream << "    size_t offset = 0;\n";
      this->PrintIndent();
      this->stream << "    size_t num_elements = min_cols;\n";
      this->PrintIndent();
      this->stream << "    while (offset < num_elements) {\n";
      this->PrintIndent();
      this->stream << "      size_t vl = __riscv_vsetvl_e" << src_info.rvv_eew << "m" << src_info.rvv_lmul << "(num_elements - offset);\n";
      this->PrintIndent();
      
      // 源数据的加载指令（所有数据类型）
      if (src_info.dtype == "_Float16") {
        this->stream << "      vfloat16m1_t data" << src_info.elem_bytes << " = __riscv_vle16_v_f16m1((" << src_info.dtype << "*)(src_ptr + i * " 
                     << src_info.tensor_id << ".shape[3] * sizeof(" << src_info.dtype << ") + offset * sizeof(" << src_info.dtype << ")), vl);\n";
      } else if (src_info.dtype == "float") {
        this->stream << "      vfloat32m1_t data" << src_info.elem_bytes << " = __riscv_vle32_v_f32m1((" << src_info.dtype << "*)(src_ptr + i * " 
                     << src_info.tensor_id << ".shape[3] * sizeof(" << src_info.dtype << ") + offset * sizeof(" << src_info.dtype << ")), vl);\n";
      } else if (src_info.dtype == "uint32_t") {
        this->stream << "      vuint32m1_t data" << src_info.elem_bytes << " = __riscv_vle32_v_u32m1((" << src_info.dtype << "*)(src_ptr + i * " 
                     << src_info.tensor_id << ".shape[3] * sizeof(" << src_info.dtype << ") + offset * sizeof(" << src_info.dtype << ")), vl);\n";
      } else if (src_info.dtype == "int32_t") {
        this->stream << "      vint32m1_t data" << src_info.elem_bytes << " = __riscv_vle32_v_i32m1((" << src_info.dtype << "*)(src_ptr + i * " 
                     << src_info.tensor_id << ".shape[3] * sizeof(" << src_info.dtype << ") + offset * sizeof(" << src_info.dtype << ")), vl);\n";
      } else if (src_info.dtype == "uint16_t") {
        this->stream << "      vuint16m1_t data" << src_info.elem_bytes << " = __riscv_vle16_v_u16m1((" << src_info.dtype << "*)(src_ptr + i * " 
                     << src_info.tensor_id << ".shape[3] * sizeof(" << src_info.dtype << ") + offset * sizeof(" << src_info.dtype << ")), vl);\n";
      } else if (src_info.dtype == "int16_t") {
        this->stream << "      vint16m1_t data" << src_info.elem_bytes << " = __riscv_vle16_v_i16m1((" << src_info.dtype << "*)(src_ptr + i * " 
                     << src_info.tensor_id << ".shape[3] * sizeof(" << src_info.dtype << ") + offset * sizeof(" << src_info.dtype << ")), vl);\n";
      } else if (src_info.dtype == "uint8_t") {
        this->stream << "      vuint8m1_t data" << src_info.elem_bytes << " = __riscv_vle8_v_u8m1((" << src_info.dtype << "*)(src_ptr + i * " 
                     << src_info.tensor_id << ".shape[3] * sizeof(" << src_info.dtype << ") + offset * sizeof(" << src_info.dtype << ")), vl);\n";
      } else if (src_info.dtype == "int8_t") {
        this->stream << "      vint8m1_t data" << src_info.elem_bytes << " = __riscv_vle8_v_i8m1((" << src_info.dtype << "*)(src_ptr + i * " 
                     << src_info.tensor_id << ".shape[3] * sizeof(" << src_info.dtype << ") + offset * sizeof(" << src_info.dtype << ")), vl);\n";
      } else {
        throw std::runtime_error("Unsupported source dtype for copy: " + src_info.dtype);
      }
      this->PrintIndent();
      if (src_info.dtype != dst_info.dtype){
        if (src_info.dtype == "float" && dst_info.dtype == "_Float16") {
          this->stream << "      float temp_f32[vl];\n";
          this->PrintIndent();
          this->stream << "      __riscv_vse32_v_f32m1(temp_f32, data" << src_info.elem_bytes << ", vl);\n";
          this->PrintIndent();
          this->stream << "      for (size_t idx = 0; idx < vl; idx++) {\n";
          this->PrintIndent();
          this->stream << "        " << dst_info.dtype << "* dst_elem = (" << dst_info.dtype << "*)(dst_ptr + i * " 
                      << dst_info.tensor_id << ".shape[3] * sizeof(" << dst_info.dtype << ") + (offset + idx) * sizeof(" << dst_info.dtype << "));\n";
          this->PrintIndent();
          this->stream << "        *dst_elem = (" << dst_info.dtype << ")temp_f32[idx];\n";
          this->PrintIndent();
          this->stream << "      }\n";
        } else {
          this->stream << "      " << dst_info.rvv_type0 << " data" << dst_info.elem_bytes << " = __riscv_vfncvt_f_f_w_f" << dst_info.rvv_eew << "m" << dst_info.rvv_lmul << "(data" << src_info.elem_bytes << ", vl);\n";
        }
      }
      if (!(src_info.dtype == "float" && dst_info.dtype == "_Float16")) {
        if (dst_info.dtype == "_Float16") {
          this->stream << "      __riscv_vse16_v_f16m1((" << dst_info.dtype << "*)(dst_ptr + i * " 
                      << dst_info.tensor_id << ".shape[3] * sizeof(" << dst_info.dtype << ") + offset * sizeof(" << dst_info.dtype << ")), data" << dst_info.elem_bytes << ", vl);\n";
        } else if (dst_info.dtype == "float") {
          this->stream << "      __riscv_vse32_v_f32m1((" << dst_info.dtype << "*)(dst_ptr + i * " 
                      << dst_info.tensor_id << ".shape[3] * sizeof(" << dst_info.dtype << ") + offset * sizeof(" << dst_info.dtype << ")), data" << dst_info.elem_bytes << ", vl);\n";
        } else if (dst_info.dtype == "uint32_t") {
          this->stream << "      __riscv_vse32_v_u32m1((" << dst_info.dtype << "*)(dst_ptr + i * " 
                      << dst_info.tensor_id << ".shape[3] * sizeof(" << dst_info.dtype << ") + offset * sizeof(" << dst_info.dtype << ")), data" << dst_info.elem_bytes << ", vl);\n";
        } else if (dst_info.dtype == "int32_t") {
          this->stream << "      __riscv_vse32_v_i32m1((" << dst_info.dtype << "*)(dst_ptr + i * " 
                      << dst_info.tensor_id << ".shape[3] * sizeof(" << dst_info.dtype << ") + offset * sizeof(" << dst_info.dtype << ")), data" << dst_info.elem_bytes << ", vl);\n";
        } else if (dst_info.dtype == "uint16_t") {
          this->stream << "      __riscv_vse16_v_u16m1((" << dst_info.dtype << "*)(dst_ptr + i * " 
                      << dst_info.tensor_id << ".shape[3] * sizeof(" << dst_info.dtype << ") + offset * sizeof(" << dst_info.dtype << ")), data" << dst_info.elem_bytes << ", vl);\n";
        } else if (dst_info.dtype == "int16_t") {
          this->stream << "      __riscv_vse16_v_i16m1((" << dst_info.dtype << "*)(dst_ptr + i * " 
                      << dst_info.tensor_id << ".shape[3] * sizeof(" << dst_info.dtype << ") + offset * sizeof(" << dst_info.dtype << ")), data" << dst_info.elem_bytes << ", vl);\n";
        } else if (dst_info.dtype == "uint8_t") {
          this->stream << "      __riscv_vse8_v_u8m1((" << dst_info.dtype << "*)(dst_ptr + i * " 
                      << dst_info.tensor_id << ".shape[3] * sizeof(" << dst_info.dtype << ") + offset * sizeof(" << dst_info.dtype << ")), data" << dst_info.elem_bytes << ", vl);\n";
        } else if (dst_info.dtype == "int8_t") {
          this->stream << "      __riscv_vse8_v_i8m1((" << dst_info.dtype << "*)(dst_ptr + i * " 
                      << dst_info.tensor_id << ".shape[3] * sizeof(" << dst_info.dtype << ") + offset * sizeof(" << dst_info.dtype << ")), data" << dst_info.elem_bytes << ", vl);\n";
        } else {
          throw std::runtime_error("Unsupported destination dtype for copy: " + dst_info.dtype);
        }
      }
      
      this->PrintIndent();
      this->stream << "      offset += vl;\n";
      this->PrintIndent();
      this->stream << "    }\n";
      this->PrintIndent();
      this->stream << "  }\n";
      
      // 内存屏障条件判断
      bool need_fence = (src_info.scope == "global" || dst_info.scope == "global");
      if (need_fence) {
          this->PrintIndent();
          this->stream << "  asm volatile (\"fence ow, ow\" ::: \"memory\");\n";
      }
      
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "rvv.fill") {
      auto var_ = op->args[1].as<CallNode>()->args[1].as<VarNode>();
      auto dst = var_idmap_[var_];
      auto dtype = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      std::string rvv_type;
      uint rvv_eew;
      uint rvv_lmul;
      std::string rvv_vec_type;     // 新增：向量类型存储
      std::string type_suffix;       // 新增：intrinsic类型后缀

      // 统一处理所有类型的定义
      if (dtype == DataType::Float(16)) {
        rvv_type = "_Float16";
        rvv_eew = 16;
        rvv_lmul = 1;
        rvv_vec_type = "vfloat16m1_t";  // 正确向量类型
        type_suffix = "f16m1";          // intrinsic后缀
      } else if (dtype == DataType::Float(32)) {
        rvv_type = "float";
        rvv_eew = 32;
        rvv_lmul = 1;
        rvv_vec_type = "vfloat32m1_t";
        type_suffix = "f32m1";
      } else if (dtype == DataType::Int(32)) {
        rvv_type = "int32_t";
        rvv_eew = 32;
        rvv_lmul = 1;
        rvv_vec_type = "vint32m1_t";
        type_suffix = "i32m1";
      } else if (dtype == DataType::Int(16)) {
        rvv_type = "int16_t";
        rvv_eew = 16;
        rvv_lmul = 1;
        rvv_vec_type = "vint16m1_t";
        type_suffix = "i16m1";
      } else if (dtype == DataType::UInt(8)) {
        rvv_type = "uint8_t";
        rvv_eew = 8;
        rvv_lmul = 1;
        rvv_vec_type = "vuint8m1_t";
        type_suffix = "u8m1";
      } else {
        throw std::runtime_error("Unsupported dtype for fill");
      }
      auto addr = dst + ".addr";
      double value;
      if (dtype.is_float()) {
        value = Downcast<FloatImm>(op->args[2])->value;
      } else {
        value = static_cast<double>(Downcast<IntImm>(op->args[2])->value);
      }
      this->PrintIndent();
      this->stream << "{\n";
      this->PrintIndent();
      this->stream << "  size_t vlen = " << dst << ".shape[1] * " << dst <<".shape[3];\n";
      this->PrintIndent();
      this->stream << "  size_t vl;\n";
      this->PrintIndent();
      if (dtype.is_float()) {
          if (std::isinf(value) || std::isnan(value)) {
              if (value < 0) {
                  this->stream << "  " << rvv_type << " broadcast_val = " << "(" << rvv_type << ")(-INFINITY);\n";
              } else {
                  this->stream << "  " << rvv_type << " broadcast_val = " << "(<" << rvv_type << ")(INFINITY);\n";
              }
          } else {
              this->stream << "  " << rvv_type << " broadcast_val = " << value << ";\n";
          }
      } else {
        // 使用实际类型转换避免精度丢失
        this->stream << "  " << rvv_type << " broadcast_val = ";
        if (dtype.bits() <= 32) {
          this->stream << static_cast<int32_t>(value);
        } else {
          this->stream << static_cast<int64_t>(value);
        }
        this->stream << ";\n";
      }
      this->PrintIndent();
      this->stream << "  for (size_t offset = 0; offset < vlen; offset += vl) {\n";
      this->PrintIndent();
      this->stream << "    vl = __riscv_vsetvl_e" << rvv_eew << "m" << rvv_lmul << "(vlen - offset);\n";
      this->PrintIndent();

      // 修正的向量定义和intrinsic调用
      if (dtype.is_float()) {
        this->stream << "    " << rvv_vec_type << " vec_val = "
                    << "__riscv_vfmv_v_f_" << type_suffix  // 使用统一后缀
                    << "(broadcast_val, vl);\n";
      } else {
        this->stream << "    " << rvv_vec_type << " vec_val = "
                    << "__riscv_vmv_v_x_" << type_suffix  // 使用统一后缀
                    << "(broadcast_val, vl);\n";
      }

      this->PrintIndent();
      // 使用正确的存储指令后缀
      const char* store_sign = dtype.is_float() ? "f" : (dtype.is_uint() ? "u" : "i");
      this->stream << "    __riscv_vse" << rvv_eew << "_v_"
                  << store_sign << rvv_eew << "m" << rvv_lmul  // 动态生成正确后缀
                  << "((" << rvv_type << "*)((uint8_t*)" << addr
                  << " + offset * sizeof(" << rvv_type << ")), vec_val, vl);\n";
      this->PrintIndent();
      this->stream << "  }\n";
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "rvv.gemm") {
      auto a_access_data = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto b_access_data = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto c_access_data = var_idmap_[op->args[3].as<CallNode>()->args[1].as<VarNode>()];
      auto M = Downcast<IntImm>(op->args[6])->value;
      auto N = Downcast<IntImm>(op->args[7])->value;
      auto K = Downcast<IntImm>(op->args[8])->value;
      auto trans_B = Downcast<Bool>(op->args[5])->value;
      auto dtype_ = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      std::string data_type = "fp16";
      if (dtype_ == DataType::Float(16)) {
          data_type = "fp16";
      } else if (dtype_ == DataType::Float(32)) {
          data_type = "fp32";
      } else if (dtype_ == DataType::Int(8)) {
          data_type = "int8";
      } else {
          throw std::runtime_error("Unsupported dtype for rvv.gemm: ");
      }
      this->PrintIndent();
      this->stream << "{\n";
      if (data_type == "fp16") {
          this->PrintIndent();
          this->stream << "  _Float16* A = (_Float16*)" << a_access_data << ".addr;\n";
          this->PrintIndent();
          this->stream << "  _Float16* B = (_Float16*)" << b_access_data << ".addr;\n";
          this->PrintIndent();
          this->stream << "  float* C = (float*)" << c_access_data << ".addr;\n";
      } else if (data_type == "fp32") {
          this->PrintIndent();
          this->stream << "  float* A = (float*)" << a_access_data << ".addr;\n";
          this->PrintIndent();
          this->stream << "  float* B = (float*)" << b_access_data << ".addr;\n";
          this->PrintIndent();
          this->stream << "  float* C = (float*)" << c_access_data << ".addr;\n";
      } else if (data_type == "int8") {
          this->PrintIndent();
          this->stream << "  int8_t* A = (int8_t*)" << a_access_data << ".addr;\n";
          this->PrintIndent();
          this->stream << "  int8_t* B = (int8_t*)" << b_access_data << ".addr;\n";
          this->PrintIndent();
          this->stream << "  int32_t* C = (int32_t*)" << c_access_data << ".addr;\n";
      }

      this->PrintIndent();
      this->stream << "  size_t avl, vl;\n";
      if (data_type == "fp16") {
          this->PrintIndent();
          this->stream << "  vfloat32m2_t acc, a_bcast, b_convert;\n";
      } else if (data_type == "fp32") {
          this->PrintIndent();
          this->stream << "  vfloat32m1_t acc, a_bcast, b_vec;\n";
      } else if (data_type == "int8") {
          this->PrintIndent();
          this->stream << "  vint32m1_t acc, a_bcast, b_ext;\n";
      }

      this->PrintIndent();
      this->PrintIndent();
      this->stream << "  for (size_t i = 0; i < " << M << "; i++) {\n";
      this->PrintIndent();
      this->stream << "    size_t j = 0;\n";
      this->PrintIndent();
      this->stream << "    avl = " << N << ";\n";

      this->PrintIndent();
      this->stream << "    while (avl > 0) {\n";
      this->PrintIndent();
      if (data_type == "fp16") {
          this->stream << "      vl = __riscv_vsetvl_e16m1(avl);\n";
          this->PrintIndent();
          this->stream << "      acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);\n";
          this->PrintIndent();
          this->stream << "      vfloat32m2_t current = __riscv_vle32_v_f32m2(C + i * " << N << " + j, vl);\n";
      } else if (data_type == "fp32") {
          this->stream << "      vl = __riscv_vsetvl_e32m1(avl);\n";
          this->PrintIndent();
          this->stream << "      acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);\n";
          this->PrintIndent();
          this->stream << "      vfloat32m1_t current = __riscv_vle32_v_f32m1(C + i * " << N << " + j, vl);\n";
      } else if (data_type == "int8") {
          this->stream << "      vl = __riscv_vsetvl_e8m1(avl);\n";
          this->PrintIndent();
          this->stream << "      acc = __riscv_vmv_v_x_i32m1(0, vl);\n";
          this->PrintIndent();
          this->stream << "      vint32m1_t current = __riscv_vle32_v_i32m1(C + i * " << NDEBUG << " + j, vl);\n";
      }
      this->PrintIndent();
      this->stream << "      for (size_t k = 0; k < " << K << "; k++) {\n";
      if (data_type == "fp16") {
        this->PrintIndent();
        this->stream << "        _Float16 a = A[i * " << K << " + k];\n";
        this->PrintIndent();
        this->stream << "        float a_val = (float)a;\n";
        this->PrintIndent();
        this->stream << "        a_bcast = __riscv_vfmv_v_f_f32m2(a_val, vl);\n";
      } else if (data_type == "fp32") {
        this->PrintIndent();
        this->stream << "        a_bcast = __riscv_vfmv_v_f_f32m1(A[i * " << K << " + k], vl);\n";
      } else if (data_type == "int8") {
        this->PrintIndent();
        this->stream << "        a_bcast = __riscv_vwadd_vx_i32m1(__riscv_vmv_v_x_i8m1(A[i * " << K << " + k], vl), 0, vl);\n";
      }
      // 修改：根据转置标志重新计算索引
      if (!trans_B) {
          if (data_type == "fp16") {
              this->PrintIndent();
              this->stream << "        vfloat16m1_t b_vec = __riscv_vle16_v_f16m1(B + k * " << N << " + j, vl);\n";
              this->PrintIndent();
              this->stream << "        b_convert = __riscv_vfwcvt_f_f_v_f32m2(b_vec, vl);\n";
              this->PrintIndent();
              this->stream << "        acc = __riscv_vfmacc_vv_f32m2(acc, a_bcast, b_convert, vl);\n";
          } else if (data_type == "fp32") {
              this->PrintIndent();
              this->stream << "        b_vec = __riscv_vle32_v_f32m1(&B[k * " << N << " + j], vl);\n";
              this->PrintIndent();
              this->stream << "        acc = __riscv_vfmacc_vv_f32m1(acc, a_bcast, b_vec, vl);\n";
          } else if (data_type == "int8") {
              this->PrintIndent();
              this->stream << "        b_ext = __riscv_vwadd_vx_i32m1(__riscv_vle8_v_i8m1(&B[k * " << N << " + j], vl), 0, vl);\n";
              this->PrintIndent();
              this->stream << "        acc = __riscv_vmacc_vv_i32m1(acc, a_bcast, b_ext, vl);\n";
          }
      } else {
          if (data_type == "fp16") {
              this->PrintIndent();
              this->stream << "        vfloat16m1_t b_vec = __riscv_vle16_v_f16m1(B + j * " << K << " + k, vl);\n";
              this->PrintIndent();
              this->stream << "        b_convert = __riscv_vfwcvt_f_f_v_f32m2(b_vec, vl);\n";
              this->PrintIndent();
              this->stream << "        acc = __riscv_vfmacc_vv_f32m2(acc, a_bcast, b_convert, vl);\n";
          } else if (data_type == "fp32") {
              this->PrintIndent();
              this->stream << "        b_vec = __riscv_vle32_v_f32m1(&B[j * " << K << " + k], vl);\n";
              this->PrintIndent();
              this->stream << "        acc = __riscv_vfmacc_vv_f32m1(acc, a_bcast, b_vec, vl);\n";
          } else if (data_type == "int8") {
              this->PrintIndent();
              this->stream << "        b_ext = __riscv_vwadd_vx_i32m1(__riscv_vle8_v_i8m1(&B[j * " << K << " + k], vl), 0, vl);\n";
              this->PrintIndent();
              this->stream << "        acc = __riscv_vmacc_vv_i32m1(acc, a_bcast, b_ext, vl);\n";
          }
      }

      this->PrintIndent();
      this->stream << "      }\n";
      this->PrintIndent();
      
      // 修改：根据转置标志调整结果存储
      if (data_type == "fp16") {
        this->PrintIndent();
        this->stream << "      current = __riscv_vfadd_vv_f32m2(current, acc, vl);\n";
        this->PrintIndent();
        this->stream << "      __riscv_vse32_v_f32m2(C + i * " << N << " + j, current, vl);\n";
      } else if (data_type == "fp32") {
        this->PrintIndent();
        this->stream << "      current = __riscv_vfadd_vv_f32m1(current, acc, vl);\n";
        this->PrintIndent();
        this->stream << "      __riscv_vse32_v_f32m1(C + i * " << N << " + j, current, vl);\n";
      } else if (data_type == "int8") {
        this->PrintIndent();
        this->stream << "      current = __riscv_vadd_vv_i32m1(current, acc, vl);\n";
        this->PrintIndent();
        this->stream << "      __riscv_vse32_v_i32m1(C + i * " << N << " + j, current, vl);\n";
      }

      this->PrintIndent();
      this->stream << "      j += vl;\n";
      this->PrintIndent();
      this->stream << "      avl -= vl;\n";

      this->PrintIndent();
      this->stream << "    }\n";

      this->PrintIndent();
      this->stream << "  }\n";
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "rvv.sub") {
      handle_rvv_elementwise("sub", true);
    } else if (op_name == "rvv.mul") {
      handle_rvv_elementwise("mul", true);
    } else if (op_name == "rvv.add") {
      handle_rvv_elementwise("add", true);
    } else if (op_name == "rvv.div") {
      handle_rvv_elementwise("div", true);
    } else if (op_name == "rvv.mul_C") {
      handle_rvv_elementwise_const("mul");
    } else if (op_name == "rvv.add_C") {
      handle_rvv_elementwise_const("add");
    } 
    /** The following op needs to be handled specially. */
    else if (op_name == "rvv.exp") {
      // 获取输入张量信息
      auto input_tensor = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      
      // 获取数据类型
      auto dtype_ = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      std::string rvv_type;
      std::string vec_type_prefix;
      std::string rvv_type_rvv;
      uint rvv_eew;
      uint rvv_lmul = 1;

      // 确定数据类型相关信息
      if (dtype_ == DataType::Float(16)) {
        rvv_type = "_Float16";
        vec_type_prefix = "float16";
        rvv_eew = 16;
        rvv_type_rvv = "f16";
        rvv_lmul = 1;
      } else if (dtype_ == DataType::Float(32)) {
        rvv_type = "float";
        vec_type_prefix = "float32";
        rvv_eew = 32;
        rvv_type_rvv = "f32";
        rvv_lmul = 1;
      } else {
        throw std::runtime_error("Unsupported dtype for exp2: only fp16/fp32 supported");
      }

      // 获取向量类型
      std::string vec_type = "v" + vec_type_prefix + "m" + std::to_string(rvv_lmul) + "_t";

      this->PrintIndent();
      this->stream << "{\n";
      this->PrintIndent();
      this->stream << "  " << rvv_type << "* input_ptr = (" << rvv_type << "*)" << input_tensor << ".addr;\n";
      this->PrintIndent();
      this->stream << "  size_t total_elements = " << input_tensor << ".shape[1] * " << input_tensor <<".shape[3];\n";
      this->PrintIndent();
      this->stream << "  size_t vl;\n";
      this->PrintIndent();
      this->stream << "  for (size_t offset = 0; offset < total_elements; offset += vl) {\n";
      this->PrintIndent();
      this->stream << "    vl = __riscv_vsetvl_e" << rvv_eew << "m" << rvv_lmul << "(total_elements - offset);\n";
      
      // 加载输入数据
      this->PrintIndent();
      if (dtype_ == DataType::Float(16)) {
        this->stream << "    vfloat16m1_t vec = __riscv_vle16_v_f16m1(input_ptr + offset, vl);\n";
        this->PrintIndent();
        this->stream << "    " << rvv_type << " temp[vl];\n";
        this->PrintIndent();
        this->stream << "    __riscv_vse16_v_f16m1(temp, vec, vl);\n";
      } else {
        this->stream << "    vfloat32m1_t vec = __riscv_vle" << rvv_eew << "_v_f" << rvv_eew << "m" << rvv_lmul << "(input_ptr + offset, vl);\n";
        this->PrintIndent();
        this->stream << "    " << rvv_type << " temp[vl];\n";
        this->PrintIndent();
        this->stream << "    __riscv_vse" << rvv_eew << "_v_f" << rvv_eew << "m" << rvv_lmul << "(temp, vec, vl);\n";
      }
        
      // 对向量中的每个元素计算exp2
      this->PrintIndent();
      this->stream << "    for (size_t j = 0; j < vl; j++) {\n";
      this->PrintIndent();
      if (dtype_ == DataType::Float(16)) {
        this->stream << "      temp[j] = (" << rvv_type << ")(expf(temp[j]));\n";
      } else {
        this->stream << "      temp[j] = expf(temp[j]);\n";
      }
      this->PrintIndent();
      this->stream << "    }\n";

      // 存储结果（就地操作）
      this->PrintIndent();
      if (dtype_ == DataType::Float(16)) {
        this->stream << "    vec = __riscv_vle16_v_f16m1(temp, vl);\n";
        this->PrintIndent();
        this->stream << "    __riscv_vse16_v_f16m1(input_ptr + offset, vec, vl);\n";
      } else {
        this->stream << "    vec = __riscv_vle32_v_f32m1(temp, vl);\n";
        this->PrintIndent();
        this->stream << "    __riscv_vse32_v_f32m1(input_ptr + offset, vec, vl);\n";
      }
      
      this->PrintIndent();
      this->stream << "  }\n";
      
      // 添加内存屏障确保写入完成
      this->PrintIndent();
      this->stream << "  asm volatile (\"fence ow, ow\" ::: \"memory\");\n";
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "rvv.reduce_max") {
      auto input_tensor = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto output_tensor = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto input_shape = buffer_shape[input_tensor];
      auto output_shape = buffer_shape[output_tensor];
      auto dtype_ = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      std::string rvv_type;
      std::string vec_type_prefix;
      std::string rvv_type_rvv;
      uint rvv_eew;
      uint rvv_lmul = 1;
      if (dtype_ == DataType::Float(16)) {
        rvv_type = "_Float16";
        vec_type_prefix = "float16";
        rvv_eew = 16;
        rvv_type_rvv = "f16";
        rvv_lmul = 1;
      } else if (dtype_ == DataType::Float(32)) {
        rvv_type = "float";
        vec_type_prefix = "float32";
        rvv_eew = 32;
        rvv_type_rvv = "f32";
        rvv_lmul = 1;
      } else if (dtype_ == DataType::Int(16)) {
        rvv_type = "int16_t";
        vec_type_prefix = "int16";
        rvv_eew = 16;
        rvv_type_rvv = "i16";
        rvv_lmul = 1;
      } else if (dtype_ == DataType::Int(32)) {
        rvv_type = "int32_t";
        vec_type_prefix = "int32";
        rvv_eew = 32;
        rvv_type_rvv = "i32";
        rvv_lmul = 1;
      } else {
        throw std::runtime_error("Unsupported dtype for reducce_max: ");
      }
      
      std::string vec_type = "v" + vec_type_prefix + "m" + std::to_string(rvv_lmul) + "_t";
      this->PrintIndent();
      this->stream << "{\n";
      this->PrintIndent();
      this->stream << "  " << rvv_type << "* input_ptr = (" << rvv_type << "*)" << input_tensor << ".addr;\n";
      this->PrintIndent();
      this->stream << "  " << rvv_type << "* output_ptr = (" << rvv_type << "*)" << output_tensor << ".addr;\n";
      this->PrintIndent();
      this->stream << "  " << rvv_type << " init_val = ";
      
      if (dtype_.is_float()) {
        this->stream << "(" << rvv_type << ")(-INFINITY)";
      } else if (dtype_.is_uint()) {
        this->stream << "0";
      } else {
        this->stream << "std::numeric_limits<" << rvv_type << ">::min()";
      }
      this->stream << ";\n";

      this->PrintIndent();
      this->stream << "  size_t max_vl = __riscv_vsetvlmax_e" << rvv_eew << "m" << rvv_lmul << "();\n";
      this->PrintIndent();
      if (dtype_.is_float()) {
        this->stream << "  " << vec_type << " vec_acc_init = __riscv_vfmv_v_f_"
                    << "f" << rvv_eew << "m" << rvv_lmul
                    << "(init_val, max_vl);\n";
      } else {
        this->stream << "  " << vec_type << " vec_acc_init = __riscv_vmv_v_x_"
                    << "i" << rvv_eew << "m" << rvv_lmul
                    << "(init_val, max_vl);\n";
      }
      this->PrintIndent();
      this->stream << "  int N = " << input_tensor << ".shape[1];\n";
      this->PrintIndent();
      this->stream << "  int M = " << input_tensor << ".shape[3];\n";
      
      this->PrintIndent();
      this->stream << "  for (size_t i = 0; i < N; i++) {\n";
      
      
      this->PrintIndent();
      this->stream << "    " << rvv_type << "* group_start = input_ptr + i * M;\n";
      
      this->PrintIndent();
      this->stream << "    " << vec_type << " vec_acc = vec_acc_init;\n";

      this->PrintIndent();
      this->stream << "    size_t j = 0;\n";
      this->PrintIndent();
      this->stream << "    while (j < M) {\n";

      this->PrintIndent();
      this->stream << "      size_t vl = __riscv_vsetvl_e" << rvv_eew << "m" << rvv_lmul << "(M - j);\n";

      this->PrintIndent();
      if (dtype_.is_float()) {
        this->stream << "      " << vec_type << " vec = __riscv_vle" << rvv_eew << "_v_"
                    << "f" << rvv_eew << "m" << rvv_lmul << "(group_start + j, vl);\n";
      } else {
        this->stream << "      " << vec_type << " vec = __riscv_vle" << rvv_eew << "_v_"
                    << "i" << rvv_eew << "m" << rvv_lmul << "(group_start + j, vl);\n";
      }

      this->PrintIndent();
      if (dtype_.is_float()) {
        this->stream << "      vec_acc = __riscv_vfredmax_vs_"
                    << "f" << rvv_eew << "m" << rvv_lmul << "_"
                    << "f" << rvv_eew << "m" << rvv_lmul << "(vec, vec_acc, vl);\n";
      } else if (dtype_.is_uint()) {
        this->stream << "      vec_acc = __riscv_vredmaxu_vs_"
                    << "u" << rvv_eew << "m" << rvv_lmul << "_"
                    << "u" << rvv_eew << "m" << rvv_lmul << "(vec, vec_acc, vl);\n";
      } else {
        this->stream << "      vec_acc = __riscv_vredmax_vs_"
                    << "i" << rvv_eew << "m" << rvv_lmul << "_"
                    << "i" << rvv_eew << "m" << rvv_lmul << "(vec, vec_acc, vl);\n";
      }

      this->PrintIndent();
      this->stream << "      j += vl;\n";
      this->PrintIndent();
      this->stream << "    }\n";
      this->PrintIndent();
      if (dtype_.is_float()) {
        this->stream << "    output_ptr[i] = __riscv_vfmv_f_s_f"
                    << rvv_eew << "m" << rvv_lmul << "_" << rvv_type_rvv << "(vec_acc);\n";
      } else if (dtype_.is_uint()) {
        this->stream << "    output_ptr[i] = __riscv_vmv_x_s_u"
                    << rvv_eew << "m" << rvv_lmul << "_" << rvv_type_rvv << "(vec_acc);\n";
      } else {
        this->stream << "    output_ptr[i] = __riscv_vmv_x_s_i"
                    << rvv_eew << "m" << rvv_lmul << "_" << rvv_type_rvv << "(vec_acc);\n";
      }
      this->PrintIndent();
      this->stream << "  }\n";
      this->PrintIndent();
      this->stream << "  asm volatile (\"fence ow, ow\" ::: \"memory\");\n";
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "rvv.reduce_sum") {
      auto input_tensor = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto output_tensor = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto input_shape = buffer_shape[input_tensor];
      auto output_shape = buffer_shape[output_tensor];

      auto dtype_ = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      std::string rvv_type;
      std::string vec_type_prefix;
      std::string rvv_type_rvv;
      uint rvv_eew;
      uint rvv_lmul = 1;
      if (dtype_ == DataType::Float(16)) {
        rvv_type = "_Float16";
        vec_type_prefix = "float16";
        rvv_eew = 16;
        rvv_type_rvv = "f16";
        rvv_lmul = 1;
      } else if (dtype_ == DataType::Float(32)) {
        rvv_type = "float";
        vec_type_prefix = "float32";
        rvv_eew = 32;
        rvv_type_rvv = "f32";
        rvv_lmul = 1;
      } else if (dtype_ == DataType::Int(16)) {
        rvv_type = "int16_t";
        vec_type_prefix = "int16";
        rvv_eew = 16;
        rvv_type_rvv = "i16";
        rvv_lmul = 1;
      } else if (dtype_ == DataType::Int(32)) {
        rvv_type = "int32_t";
        vec_type_prefix = "int32";
        rvv_eew = 32;
        rvv_type_rvv = "i32";
        rvv_lmul = 1;
      } else {
        throw std::runtime_error("Unsupported dtype for reducce_max: ");
      }

      std::string vec_type = "v" + vec_type_prefix + "m" + std::to_string(rvv_lmul) + "_t";

      this->PrintIndent();
      this->stream << "{\n";
      this->PrintIndent();
      this->stream << "  " << rvv_type << "* input_ptr = (" << rvv_type << "*)" << input_tensor << ".addr;\n";
      this->PrintIndent();
      this->stream << "  " << rvv_type << "* output_ptr = (" << rvv_type << "*)" << output_tensor << ".addr;\n";

      this->PrintIndent();
      if (dtype_.is_float()) {
        this->stream << "  " << rvv_type << " init_val = 0.0f;\n";
      } else {
        this->stream << "  " << rvv_type << " init_val = 0;\n";
      }

      this->PrintIndent();
      this->stream << "  size_t max_vl = __riscv_vsetvlmax_e" << rvv_eew << "m" << rvv_lmul << "();\n";

      this->PrintIndent();
      if (dtype_.is_float()) {
        this->stream << "  " << vec_type << " vec_acc_init = __riscv_vfmv_v_f_"
                    << "f" << rvv_eew << "m" << rvv_lmul
                    << "(init_val, max_vl);\n";
      } else {
        this->stream << "  " << vec_type << " vec_acc_init = __riscv_vmv_v_x_"
                    << "i" << rvv_eew << "m" << rvv_lmul
                    << "(init_val, max_vl);\n";
      }
      this->PrintIndent();
      this->stream << "  int N = " << input_tensor << ".shape[1];\n";
      this->PrintIndent();
      this->stream << "  int M = " << input_tensor << ".shape[3];\n";
      this->PrintIndent();
      this->stream << "  for (size_t i = 0; i < N; i++) {\n";
      this->PrintIndent();
      this->stream << "    " << rvv_type << "* group_start = input_ptr + i * M;\n";
      this->PrintIndent();
      this->stream << "    " << vec_type << " vec_acc = vec_acc_init;\n";

      this->PrintIndent();
      this->stream << "    size_t j = 0;\n";
      this->PrintIndent();
      this->stream << "    while (j < M) {\n";
      this->PrintIndent();
      this->stream << "      size_t vl = __riscv_vsetvl_e" << rvv_eew << "m" << rvv_lmul << "(M - j);\n";
      this->PrintIndent();
      if (dtype_.is_float()) {
        this->stream << "      " << vec_type << " vec = __riscv_vle" << rvv_eew << "_v_"
                    << "f" << rvv_eew << "m" << rvv_lmul << "(group_start + j, vl);\n";
      } else {
        this->stream << "      " << vec_type << " vec = __riscv_vle" << rvv_eew << "_v_"
                    << "i" << rvv_eew << "m" << rvv_lmul << "(group_start + j, vl);\n";
      }
      this->PrintIndent();
      if (dtype_.is_float()) {
        this->stream << "      vec_acc = __riscv_vfredusum_vs_"
                    << "f" << rvv_eew << "m" << rvv_lmul << "_"
                    << "f" << rvv_eew << "m" << rvv_lmul << "(vec, vec_acc, vl);\n";
      } else {
        this->stream << "      vec_acc = __riscv_vredsum_vs_"
                    << "i" << rvv_eew << "m" << rvv_lmul << "_"
                    << "i" << rvv_eew << "m" << rvv_lmul << "(vec, vec_acc, vl);\n";
      }

      this->PrintIndent();
      this->stream << "      j += vl;\n";
      this->PrintIndent();
      this->stream << "    }\n";
      this->PrintIndent();
      if (dtype_.is_float()) {
        this->stream << "    output_ptr[i] = __riscv_vfmv_f_s_f"
                    << rvv_eew << "m" << rvv_lmul << "_" << rvv_type_rvv << "(vec_acc);\n";
      } else if (dtype_.is_uint()) {
        this->stream << "    output_ptr[i] = __riscv_vmv_x_s_u"
                    << rvv_eew << "m" << rvv_lmul << "_" << rvv_type_rvv << "(vec_acc);\n";
      } else {
        this->stream << "    output_ptr[i] = __riscv_vmv_x_s_i"
                    << rvv_eew << "m" << rvv_lmul << "_" << rvv_type_rvv << "(vec_acc);\n";
      }
      this->PrintIndent();
      this->stream << "  }\n";
      this->PrintIndent();
      this->stream << "  asm volatile (\"fence ow, ow\" ::: \"memory\");\n";
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "rvv.embedding") {
      auto output_tensor = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto params_tensor = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto index_tensor = var_idmap_[op->args[3].as<CallNode>()->args[1].as<VarNode>()];

      auto outer_num = Downcast<IntImm>(op->args[6])->value;
      auto inner_num = Downcast<IntImm>(op->args[7])->value;
      auto select_num = Downcast<IntImm>(op->args[8])->value;
      auto index_num = Downcast<IntImm>(op->args[9])->value;

      auto dtype_ = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto index_dtype_ = op->args[3].as<CallNode>()->args[0].as<CallNode>()->dtype;

      std::string rvv_type;
      uint rvv_eew;
      uint rvv_lmul = 1;

      std::string index_rvv_type;
      uint index_rvv_eew;

      if (dtype_ == DataType::Float(16)) {
        rvv_type = "_Float16";
        rvv_eew = 16;
        rvv_lmul = 1;
      } else if (dtype_ == DataType::Float(32)) {
        rvv_type = "float";
        rvv_eew = 32;
        rvv_lmul = 1;
      } else {
        throw std::runtime_error("Unsupported dtype for embedding: ");
      }

      if (index_dtype_ == DataType::UInt(16) ||
          index_dtype_ == DataType::Int(16)) {
        index_rvv_type = "uint16_t";
        index_rvv_eew = 16;
      } else if (index_dtype_ == DataType::UInt(32) ||
              index_dtype_ == DataType::Int(32)) {
        index_rvv_type = "uint32_t";
        index_rvv_eew = 32;
      } else {
        throw std::runtime_error("Unsupported index dtype: ");
      }
      this->PrintIndent();
      this->stream << "{\n";
      this->PrintIndent();
      this->stream << "  " << rvv_type << "* params_ptr = (" << rvv_type << "*)" << params_tensor << ".addr;\n";
      this->PrintIndent();
      this->stream << "  " << rvv_type << "* output_ptr = (" << rvv_type << "*)" << output_tensor << ".addr;\n";
      this->PrintIndent();
      this->stream << "  " << index_rvv_type << "* index_ptr = (" << index_rvv_type << "*)" << index_tensor << ".addr;\n";
      this->PrintIndent();
      this->stream << "  for (int i = 0; i < " << index_num << "; i++) {\n";

      this->PrintIndent();
      this->stream << "    " << index_rvv_type << " idx = index_ptr[i];\n";
      this->PrintIndent();
      this->stream << "    int j = 0;\n";
      this->PrintIndent();
      this->stream << "    while (j < " << inner_num << ") {\n";
      this->PrintIndent();
      this->stream << "      size_t vl = __riscv_vsetvl_e" << rvv_eew << "m" << rvv_lmul << "(" << inner_num << " - j);\n";
      this->PrintIndent();
      this->stream << "      if (idx >= " << select_num << ") {\n";
      this->PrintIndent();
      if (dtype_ == DataType::Float(16)) {
        this->stream << "        vfloat16m1_t zero_vec = __riscv_vfmv_v_f_f16m1(0, vl);\n";
        this->PrintIndent();
        this->stream << "        __riscv_vse16_v_f16m1(output_ptr  + i * " << inner_num << " + j, zero_vec, vl);\n";
      } else {
        this->stream << "        vfloat32m1_t zero_vec = __riscv_vfmv_v_f_f32m1(0, vl);\n";
        this->PrintIndent();
        this->stream << "        __riscv_vse32_v_f32m1(output_ptr + i) * " << inner_num << " + j, zero_vec, vl);\n";
      }
      this->PrintIndent();
      this->stream << "      } else {\n";
      this->PrintIndent();
      if (dtype_ == DataType::Float(16)) {
        this->stream << "        vfloat16m1_t vec = __riscv_vle16_v_f16m1(params_ptr + idx * " << inner_num << " + j, vl);\n";
        this->PrintIndent();
        this->stream << "        __riscv_vse16_v_f16m1(output_ptr  + i * " << inner_num << " + j, vec, vl);\n";
      } else {
        this->stream << "        vfloat32m1_t vec = __riscv_vle32_v_f32m1(params_ptr + idx * " << inner_num << " + j, vl);\n";
        this->PrintIndent();
        this->stream << "        __riscv_vse32_v_f32m1(output_ptr + i) * " << inner_num << " + j, vec, vl);\n";
      }
      this->PrintIndent();
      this->stream << "      }\n";
      this->PrintIndent();
      this->stream << "      j += vl;\n";
      this->PrintIndent();
      this->stream << "    }\n";
      this->PrintIndent();
      this->stream << "  }\n";
      this->PrintIndent();
      this->stream << "  asm volatile (\"fence ow, ow\" ::: \"memory\");\n";
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "rvv.rsqrt") {
      auto dst = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto src0 = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto src0_shape = buffer_shape[src0];
      
      // 获取数据类型并设置向量配置
      auto dtype = op->args[2].as<CallNode>()->args[0].as<CallNode>()->dtype;
      std::string rvv_type;
      uint rvv_eew;
      uint rvv_lmul;
      std::string rvv_vec_type;
      std::string type_suffix;
      if (dtype == DataType::Float(16)) {
        rvv_type = "_Float16";
        rvv_eew = 16;
        rvv_lmul = 1;
        rvv_vec_type = "vfloat16m1_t";  // 正确向量类型
        type_suffix = "f16m1";          // intrinsic后缀
      } else if (dtype == DataType::Float(32)) {
        rvv_type = "float";
        rvv_eew = 32;
        rvv_lmul = 1;
        rvv_vec_type = "vfloat32m1_t";
        type_suffix = "f32m1";
      } else if (dtype == DataType::Int(32)) {
        rvv_type = "int32_t";
        rvv_eew = 32;
        rvv_lmul = 1;
        rvv_vec_type = "vint32m1_t";
        type_suffix = "i32m1";
      } else if (dtype == DataType::Int(16)) {
        rvv_type = "int16_t";
        rvv_eew = 16;
        rvv_lmul = 1;
        rvv_vec_type = "vint16m1_t";
        type_suffix = "i16m1";
      } else if (dtype == DataType::UInt(8)) {
        rvv_type = "uint8_t";
        rvv_eew = 8;
        rvv_lmul = 1;
        rvv_vec_type = "vuint8m1_t";
        type_suffix = "u8m1";
      } else {
        throw std::runtime_error("Unsupported dtype for rsqrt");
      }
      
      // 计算总元素数量
      int64_t total_elements = 1;
      for (auto dim : src0_shape) {
        total_elements *= dim;
      }
      
      this->PrintIndent();
      this->stream << "{\n";
      
      // 声明变量 - 与您的fill实现风格保持一致
      std::string dst_addr = dst + ".addr";
      std::string src0_addr = src0 + ".addr";
      
      this->PrintIndent();
      this->stream << "  size_t vl;\n";
      
      this->PrintIndent();
      this->stream << "  for (size_t i = 0; i < " << total_elements << "; i += vl) {\n";
      
      // 设置向量长度 - 保持与您的fill相同的表达式
      this->PrintIndent();
      this->stream << "    vl = __riscv_vsetvl_e" << rvv_eew << "m" << rvv_lmul 
                  << "(" << total_elements << " - i);\n";
      
      // 加载源数据 - 直接使用地址表达式
      this->PrintIndent();
      this->stream << "    " << rvv_vec_type << " vec_src = __riscv_vle" << rvv_eew 
                  << "_v_" << type_suffix 
                  << "((" << rvv_type << "*)((uint8_t*)" << src0_addr 
                  << " + i * sizeof(" << rvv_type << ")), vl);\n";
      
      // 计算平方根
      this->PrintIndent();
      this->stream << "    " << rvv_vec_type << " vec_sqrt = __riscv_vfsqrt_v_" 
                  << type_suffix << "(vec_src, vl);\n";
      
      // 初始倒数估计
      this->PrintIndent();
      this->stream << "    " << rvv_vec_type << " vec_rec = __riscv_vfrec7_v_" 
                  << type_suffix << "(vec_sqrt, vl);\n";
      
      // 牛顿迭代法优化精度
      this->PrintIndent();
      this->stream << "    " << "vec_rec = __riscv_vfmul_vv_" << type_suffix 
                  << "(vec_rec, __riscv_vfrsub_vf_" << type_suffix << "(";
      this->stream << "__riscv_vfmul_vv_" << type_suffix 
                  << "(vec_sqrt, vec_rec, vl), 2.0, vl), vl);\n";
      
      // 存储结果 - 直接使用地址表达式
      this->PrintIndent();
      this->stream << "    " << "__riscv_vse" << rvv_eew << "_v_" << type_suffix
                  << "((" << rvv_type << "*)((uint8_t*)" << dst_addr 
                  << " + i * sizeof(" << rvv_type << ")), vec_rec, vl);\n";
      this->PrintIndent();
      this->stream << "  }\n";
      this->PrintIndent();
      this->stream << "}\n";
    }

  } else if (op->op.same_as(builtin::if_then_else())) {
    // conditional that skips eval if cond evals to false
    std::string result = name_supply_->FreshName("condval");
    std::string cond = PrintExpr(op->args[0]);
    this->PrintIndent();

    // remove hard code "shared"
    if (auto var = op->args[1].as<VarNode>()) {
      auto search = buffer_addrs_.find(var);
      if (search != buffer_addrs_.end()) {
        this->stream << "Tensor ";
      }
    } else {
      PrintType(op->dtype, this->stream);
    }

      this->stream << " " << result << ";\n";
      this->PrintIndent();
      this->stream << "if (" << cond << ") {\n";
      {
        int then_scope = this->BeginScope();
        std::string true_val = PrintExpr(op->args[1]);
        this->PrintIndent();
        this->stream << result << " = " << true_val << ";\n";
        this->EndScope(then_scope);
        this->PrintIndent();
        this->stream << "} else {\n";
      }
      {
        int else_scope = this->BeginScope();
        std::string false_val = PrintExpr(op->args[2]);
        this->PrintIndent();
      this->stream << result << " = " << false_val << ";\n";
        this->EndScope(else_scope);
        this->PrintIndent();
        this->stream << "}\n";
      }
      os << result;
  } else {
      CodeGenC::VisitExpr_(op, os);
  }
 }

void CodeGenTileLangRVV::VisitStmt_(const LetStmtNode *op) {

  if (op->body.as<Evaluate>()) {
    Evaluate e = Downcast<Evaluate>(op->body);
  }

  std::string value = PrintExpr(op->value);
  if (print_ssa_form_) {
    ICHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    if (op->var.dtype() == DataType::Handle() &&
        handle_data_type_.count(op->var.get())) {
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "* " << AllocVarID(op->var.get()) << " = (";
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "*)" << value << ";\n";
    } else {

      // TODO: (xwh) Temporary fix
      auto var_node_ = op->var.get();
      std::string var_name = var_node_->name_hint;
      if (var_name.find("shared") != std::string::npos) {
        this->stream << "Tensor " << AllocVarID(op->var.get())
                     << " = " << value << ";\n";
      } else {
        PrintType(op->var.dtype(), this->stream);
        this->stream << ' ' << AllocVarID(op->var.get()) << " = " << value
                     << ";\n";
      }
    }
  }
  PrintStmt(op->body);
}
 
void CodeGenTileLangRVV::VisitStmt_(const AttrStmtNode *op) {
   this->PrintStmt(op->body);
}
 
std::string CodeGenTileLangRVV::AllocLocalVarID(const tir::VarNode *v) {
  // ICHECK(!local_buffer_name_map.count(v)) << "Need input to be in SSA form
  // dup " << v->name_hint;
   std::string key = v->name_hint;
   std::string vid = name_supply_->FreshName(key);
   std::replace(vid.begin(), vid.end(), ':', '_');
   std::replace(vid.begin(), vid.end(), '-', '_');
   std::replace(vid.begin(), vid.end(), '.', '_');
   // var_idmap_[v].push_back()
   local_buffer_name_map[v].push_back(vid);
   return vid;
 }
 
void CodeGenTileLangRVV::VisitStmt_(const AllocateNode *op) {
   ICHECK(!is_zero(op->condition));
   auto buffer_shape = op->extents;
   if (buffer_shape.size() == 2)
     buffer_shape.insert(buffer_shape.begin(), make_const(DataType::Int(32), 1));
   std::string bv_shape = "{ 1, ";
   std::vector<int> shapes;
   shapes.push_back(buffer_shape[1].as<IntImmNode>()->value);
   shapes.push_back(buffer_shape[2].as<IntImmNode>()->value);
   bv_shape += std::to_string(buffer_shape[1].as<IntImmNode>()->value);
   bv_shape += ", 1, ";
   bv_shape += std::to_string(buffer_shape[2].as<IntImmNode>()->value);
   bv_shape += "}";
   std::string op_dtype;
   int bytes_size = 0;
   if (op->dtype == DataType::Float(16)) {
     op_dtype = "_Float16";
     bytes_size = 2;
   } else if (op->dtype == DataType::Float(32)) {
     op_dtype = "float";
     bytes_size = 4;
   } else if (op->dtype == DataType::Int(32)) {
     op_dtype = "int32_t";
     bytes_size = 4;
   } else if (op->dtype == DataType::UInt(32)) {
     op_dtype = "uint32_t";
     bytes_size = 4;
   } else if (op->dtype == DataType::Int(16)) {
     op_dtype = "int16_t";
     bytes_size = 2;
   } else if (op->dtype == DataType::UInt(16)) {
     op_dtype = "uint16_t";
     bytes_size = 2;
   } else if (op->dtype == DataType::Int(8)) {
     op_dtype = "int8_t";
     bytes_size = 1;
   } else if (op->dtype == DataType::UInt(8)) {
     op_dtype = "uint8_t";
     bytes_size = 1;
   } else {
     LOG(FATAL) << "Unsupported dtype " << op->dtype;
   }
   auto buffer_num = buffer_shape[0].as<IntImmNode>()->value;
  for (size_t iter{0}; iter < buffer_num; iter++) {
    std::string vid = AllocVarID(op->buffer_var.get());
    this->PrintIndent();
    int vid_size = shapes[0] * shapes[1];
    auto addr = f_attrs.GetAttr(vid, PrimExpr(0)).as<IntImmNode>()->value;
      buffer_addrs_[op->buffer_var.get()] = addr;
    stream << "Tensor " << vid << " = (Tensor){"
           << ".addr = malloc(" << vid_size << " * sizeof(" << op_dtype << "))"
           << ", .size = " << vid_size << " * sizeof(" << op_dtype << ")"
           << ", .shape = " << bv_shape
           << ", .stride = {1, 1, 1, 1}"
           << "};\n";
    this->PrintIndent();
    stream << "memset(" << vid << ".addr, 0, " << vid << ".size);\n";
    this->PrintIndent();
    stream << "for (int i = 2; i >= 0; i--) {\n";
    this->PrintIndent();
    stream << "  " + vid + ".stride[i] = " + vid + ".shape[i+1] * " + vid + ".stride[i+1];\n";
    this->PrintIndent();
    stream << "}\n";
    this->buffer_shape[vid] = shapes;
      // store local tensor shape
  }

   this->PrintStmt(op->body);
 }
 
void CodeGenTileLangRVV::VisitExpr_(const RampNode *op, std::ostream &os) {
  //  int lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
  //  CHECK_LE(lanes, 4) << "ValueError: Ramp of more than 4 lanes is not
  //  allowed."; os << "(make_"; PrintType(op->dtype, os); os << "("; for (int i
  //  = 0; i < lanes; i++) {
  //    os << "(" << PrintExpr(op->base) << ")"
  //       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
  //    if (i != lanes - 1) os << ", ";
  //  }
  //  os << "))";
 }
 
inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangRVV *p) { // NOLINT(*)
   // Type code is kBFloat
   if (op->dtype.is_bfloat16()) {
     os << "bfloat16_t";
     os << '(' << std::scientific << op->value << 'f' << ')';
     return;
   }
   // Type code is kFloat
   switch (op->dtype.bits()) {
     case 64:
     case 32: {
       std::ostringstream temp;
       if (std::isinf(op->value)) {
         if (op->value < 0) {
           temp << "-";
         }
         temp << ((op->dtype.bits() == 32) ? "CUDART_INF_F" : "CUDART_INF");
       } else if (std::isnan(op->value)) {
         temp << ((op->dtype.bits() == 32) ? "CUDART_NAN_F" : "CUDART_NAN");
       } else {
         temp << std::scientific << op->value;
      if (op->dtype.bits() == 32)
        temp << 'f';
       }
       p->MarkConst(temp.str());
       os << temp.str();
       break;
     }
     case 16: {
       os << "half_t" << '(';
       FloatImm const_f32 = FloatImm(DataType::Float(32), op->value);
       PrintConst(const_f32.get(), os, p);
       os << ')';
       break;
     }
     default:
       LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
   }
 }
 
void CodeGenTileLangRVV::VisitExpr_(const FloatImmNode *op,
                                    std::ostream &os) { // NOLINT(*)
   PrintConst(op, os, this);
 }
 
 template <typename T>
inline void PrintBinaryExpr(const T *op, const char *opstr,
                            std::ostream &os, // NOLINT(*)
                            CodeGenC *p) {
  if (op->dtype.lanes() == 1) {
    if (isalpha(opstr[0])) {
      os << opstr << '(';
      p->PrintExpr(op->a, os);
      os << ", ";
      p->PrintExpr(op->b, os);
      os << ')';
    } else {
      os << '(';
      p->PrintExpr(op->a, os);
      os << ' ' << opstr << ' ';
      p->PrintExpr(op->b, os);
      os << ')';
    }
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->a, op->b, os);
  }
}

void CodeGenTileLangRVV::VisitExpr_(const FloorModNode *op,
                                    std::ostream &os) { // NOLINT(*)
    PrintBinaryExpr(op, "%", os, this);
 }
 
void CodeGenTileLangRVV::PrintWmmaScope(const std::string &scope, DataType t,
                                        const VarNode *variable,
                                        std::ostream &os) {}

int32_t CodeGenTileLangRVV::GetWmmaFragmentSize(const std::string &scope,
                                                const VarNode *variable,
          int32_t size) {
return 0;                                           
}

void CodeGenTileLangRVV::HandleVolatileLoads(const std::string &value,
                                             const BufferLoadNode *op,
                                             std::ostream &os) {}

void CodeGenTileLangRVV::PrintVecElemLoadExpr(DataType t, int i,
                                              const std::string &value,
                                              std::ostream &os) {
return;
}
 
void CodeGenTileLangRVV::AddFunction(const PrimFunc &f) {
   this->InitFuncState(f);
   ReserveKeywordsAsUnique();
   auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
   f_attrs = f->attrs;
   ICHECK(global_symbol.defined())
       << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
   bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
   auto buffer_map = f->buffer_map;
   this->PrintFuncPrefix(stream);
   CodeGenC::PrintType(f->ret_type, stream);
   this->PrintExtraAttrs(f, stream);
   std::string global_name = static_cast<std::string>(global_symbol.value());
 
   this->stream << " " << global_name << "(";
   std::vector<std::string> params_name;
   // auto bf_map = f->buffer_map;
  std::unordered_map<const tir::VarNode *, std::string> var_global_mem_map;
 
  auto default_stride = [this](const std::string &node) {
     auto buf_shape = buffer_shape[node];
    buffer_stride[node] = {1, 1, 1, 1};
    for (int i = 2; i >= 0; i--) {
      buffer_stride[node][i] = buf_shape[i + 1] * buffer_stride[node][i + 1];
    }
   };
 
   // don't use name hint, but can remove later.
  auto allocate_name = [&, this](const Var &v, int index, int length) {
     auto v_node = v.get();
     std::string vid = "v" + std::to_string(index + 1);
     std::string rid = "v" + std::to_string(index + 1 + length);

     auto buffer_node = buffer_map[v];
     auto shape = buffer_node->shape;
 
    std::string shape_s = "{";
    int tensor_size = 1;
    if (shape.size() == 2) {
      buffer_shape[buffer_node->name] = {1, shape[0].as<IntImmNode>()->value, 1,
                                         shape[1].as<IntImmNode>()->value};
     default_stride(buffer_node->name);
     shape_s += "1 ,";
     shape_s += std::to_string(shape[0].as<IntImmNode>()->value);
     tensor_size *= shape[0].as<IntImmNode>()->value;
     shape_s += ", 1, ";
     shape_s += std::to_string(shape[1].as<IntImmNode>()->value);
     tensor_size *= shape[1].as<IntImmNode>()->value;
    } else if (shape.size() == 4) {
      buffer_shape[buffer_node->name] = {};
      for (auto s : shape) {
        buffer_shape[buffer_node->name].push_back(s.as<IntImmNode>()->value);
        tensor_size *= s.as<IntImmNode>()->value;
      }
      default_stride(buffer_node->name);
      // 用下标循环来拼接带逗号的字符串
      for (size_t i = 0; i < shape.size(); i++) {
        int dim_i = shape[i].as<IntImmNode>()->value;
        shape_s += std::to_string(dim_i);
        if (i + 1 < shape.size()) {
          shape_s += ", ";
        }
      }
    }

    shape_s += "}";
     std::string dtype;
     int bytes_size = 0;
 
    if (buffer_node->dtype == DataType::Float(16)) {
       dtype = "_Float16";
       bytes_size = 2;
    } else if (buffer_node->dtype == DataType::Float(32)) {
       dtype = "float";
       bytes_size = 4;
    } else if (buffer_node->dtype == DataType::Int(32)) {
       dtype = "int32_t";
       bytes_size = 4;
    } else if (buffer_node->dtype == DataType::UInt(32)) {
       dtype = "uint32_t";
       bytes_size = 4;
    } else if (buffer_node->dtype == DataType::Int(16)) {
       dtype = "int16_t";
       bytes_size = 2;
    } else if (buffer_node->dtype == DataType::UInt(16)) {
       dtype = "uint16_t";
       bytes_size = 2;
    } else if (buffer_node->dtype == DataType::Int(8)) {
       dtype = "int8_t";
       bytes_size = 1;
    } else if (buffer_node->dtype == DataType::UInt(8)) {
       dtype = "uint8_t";
       bytes_size = 1;
    } else {
       LOG(FATAL) << "Unsupported dtype " << buffer_node->dtype;
    }
    std::string inst = 
        "Tensor " + rid + " = (Tensor){.addr = malloc(" + std::to_string(tensor_size) + " * sizeof(" + dtype + "))" +
        ", .size = " + std::to_string(tensor_size) + " * sizeof(" + dtype + ")" +
        ", .shape = " + shape_s +
        ", .stride = {1, 1, 1, 1}" +
        "};\n" + 
        "  memcpy(" + rid + ".addr, " + vid + ", " + rid + ".size);\n" + 
        "  for (int i = 2; i >= 0; i--) {\n" +
        "    " + rid + ".stride[i] = " + rid + ".shape[i+1] * " + rid + ".stride[i+1];\n" +
        "  }\n";
     var_global_mem_map[v_node] = inst;
     std::string name_hint = v_node->name_hint;
     this->var_idmap_[v_node] = rid;
 
     // remove "_handle"
    for (int i{0}; i < 7; i++) {
       name_hint.pop_back();
     }
     this->parameter_map[name_hint] = rid;
     return vid;
   };
   int param_len = f->params.size();
   for (size_t i = 0; i < param_len; ++i) {
     tir::Var v = f->params[i];
     std::string vid = allocate_name(v, i, param_len); 
     params_name.push_back(vid);
    if (i != 0)
      stream << ", ";
    stream << restrict_keyword_ << ' ' << vid;
   }
   stream << ") {\n";
 
   this->PreFunctionBody(f); // none
   int func_scope = this->BeginScope();
 
  for (auto [v, inst] : var_global_mem_map) {
     this->PrintIndent();
     this->stream << inst;
   }
   this->PrintStmt(f->body);
   this->EndScope(func_scope);
   for (size_t i = 0; i < param_len; ++i) {
    tir::Var v = f->params[i];
    std::string vid0 = allocate_name(v, i, param_len);
    std::string vid1 = allocate_name(v, i+param_len, param_len);
    this->PrintIndent();
    this->stream << "  memcpy(" << vid0 << ", " << vid1 << ".addr, " << vid1 << ".size);\n";
    this->PrintIndent();
    this->stream << "  free(" << vid1 << ".addr);\n";
   }
   this->PrintIndent();
   this->stream << "}\n\n";
   this->stream << "int main(){\n";
   this->PrintIndent();
   for (size_t i = 0; i < param_len; ++i) {
      this->stream << "  void* " << params_name[i] << "= malloc(16);\n";
   }
   this->stream << "  " << global_name << "(";
   for (size_t i = 0; i < param_len; ++i) {
      if (i != 0)
        this->stream << ", ";
      this->stream << params_name[i];
   }
   this->stream << ");\n";
   for (size_t i = 0; i < param_len; ++i) {
      this->stream << "  free(" << params_name[i] << ");\n";
   }
   this->PrintIndent();
   this->stream << "  return 0;\n";
   this->PrintIndent();
   this->stream << "}\n";   
 }
 
} // namespace codegen
} // namespace tvm
