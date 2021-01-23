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
 * \file operation_inline.cc
 */
#include "operation_inline.h"

#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <utility>

#include "../../tir/transforms/ir_util.h"

namespace tvm {
namespace te {

// inliner to inline a function
// the result may not be SSA,
// ConvertSSA need to be applied after this pass
class OperationInliner final : public StmtExprMutator {
 public:
  OperationInliner(Operation op, Array<Var> args, PrimExpr body)
      : operation_(op), args_(args), body_(body) {}

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<ProducerLoadNode>();
    auto tensor = Downcast<Tensor>(op->producer);

    if (tensor->op.same_as(operation_)) {
      CHECK_EQ(tensor->value_index, 0);
      expr = body_;
      CHECK_EQ(args_.size(), op->indices.size());

      bool has_side_effect = false;
      for (size_t i = 0; i < op->indices.size(); ++i) {
        if (SideEffect(op->indices[i]) > CallEffectKind::kReadState) has_side_effect = true;
      }
      if (has_side_effect) {
        for (size_t i = 0; i < args_.size(); ++i) {
          expr = Let(args_[i], op->indices[i], expr);
        }
      } else {
        Map<Var, PrimExpr> vmap;
        for (size_t i = 0; i < args_.size(); ++i) {
          // cast indices to the type of the original indexing variable
          vmap.Set(args_[i], cast(args_[i].dtype(), op->indices[i]));
        }
        expr = Substitute(Evaluate(expr), vmap).as<EvaluateNode>()->value;
      }
      return expr;
    } else {
      return expr;
    }
  }

 private:
  Operation operation_;     // ComputeOp
  Array<Var> args_;         // Lambda的那几个输入Axis
  PrimExpr body_;           // Lambda最核心的Compute过程
};

Stmt Inline(Stmt stmt, Operation f, Array<Var> args, PrimExpr body) {                 // 将(f, args, body)对应的一个ComputeOp Inline到一个stmt上
  CHECK_EQ(f->num_outputs(), 1) << "can only inline output single value operation";
  Stmt ret = OperationInliner(f, args, body)(std::move(stmt));
  if (ret.same_as(stmt)) return ret;
  return ConvertSSA(ret);
}
}  // namespace te
}  // namespace tvm
