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
 * \file relay/backend/graph_mem_alloca.cc
 * \brief Memory index assignment pass for executing
 *   the program in the graph runtime.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/tir/op.h>

#include "../../support/arena.h"

namespace tvm {
namespace relay {

using IntegerArray = Array<Integer>;

struct StorageToken {
  /*! \brief Reference counter */
  int ref_counter{0};
  /*! \brief number of bytes */
  size_t max_bytes{0};
  /*! \brief The corresponding tensor type node. */
  const TensorTypeNode* ttype{nullptr};
  /*! \brief virtual device index that corresponds to the device_type in
   * DLContext. */
  int device_type{0};
  /*! \brief The storage id */
  int64_t storage_id{-1};
};

class StorageAllocaBaseVisitor : public ExprVisitor {
 public:
  // run the visitor on a function.
  void Run(const Function& func) {
    for (Var param : func->params) {                  // 为data这个Var创建StorageToken
      // std::cout << "##############" << std::endl;
      // std::cout << param->name_hint() << std::endl;
      CreateToken(param.operator->(), false);
      // std::cout << "**************" << std::endl;
    }

    // must always keep output alive.
    // std::cout << "##############" << std::endl;
    // std::cout << func->body->GetTypeKey() << std::endl;
    for (StorageToken* tok : GetToken(func->body)) {  // 此函数会间接递归整个网络，为网络中所有节点分配StorageToken
      tok->ref_counter += 1;                          // 确保输出不能被release
    }
    // for (auto kv: token_map_) {
    //   std::cout << kv.first->GetTypeKey() << ", " << kv.second[0]->ref_counter << std::endl;
    // }
    // // std::cout << "**************" << std::endl;
  }

  void VisitExpr_(const ConstantNode* op) final { this->CreateToken(op, false); }

  void VisitExpr_(const VarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const FunctionNode* op) final {
    // do not recursive into sub function.
  }

  void VisitExpr_(const GlobalVarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const OpNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const TupleNode* op) final {
    std::vector<StorageToken*> fields;
    for (Expr field : op->fields) {
      auto tok = GetToken(field);
      CHECK_EQ(tok.size(), 1U);
      fields.push_back(tok[0]);
    }
    token_map_[op] = fields;
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    const auto& tok = GetToken(op->tuple);
    CHECK_LT(static_cast<size_t>(op->index), tok.size());
    token_map_[op] = {tok[op->index]};
  }

  void VisitExpr_(const IfNode* op) final { LOG(FATAL) << "if is not supported."; }

  void VisitExpr_(const LetNode* op) final {
    auto token = GetToken(op->value);
    token_map_[op->var.operator->()] = token;
    token_map_[op] = GetToken(op->body);
  }

 protected:
  /*! \brief internal token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> > token_map_;    // 真正的TokenMap

  /*!
   * \brief Get the necessary token.
   * \param expr The expression.
   * \return The corresponding token.
   */
  const std::vector<StorageToken*>& GetToken(const Expr& expr) {
    this->VisitExpr(expr);          // 递归整个网络，为网络中其他节点分配Token，这里this指针指向的是一个StorageAllocaInit类对象
    // std::cout << expr.operator->()->GetTypeKey() << std::endl;
    auto it = token_map_.find(expr.operator->());
    CHECK(it != token_map_.end());
    return it->second;
  }
  /*!
   * \brief Populate the token map to set op's tokens
   * \param op The node to be processed.
   * \param can_realloc Whether we can re-allocate the memory.
   */
  virtual void CreateToken(const ExprNode* op, bool can_realloc) = 0;
};

class StorageAllocaInit : protected StorageAllocaBaseVisitor {
 public:
  explicit StorageAllocaInit(support::Arena* arena) : arena_(arena) {}

  /*! \return The internal token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> > GetInitTokenMap(
      const Function& func) {
    node_device_map_ = CollectDeviceInfo(func);   // 收集设备信息，此例中node_device_map_为空
    // std::cout << "##################" << node_device_map_.size() << std::endl;
    this->Run(func);
    // std::cout << "!!!!!!!!!" << std::endl;
    return std::move(token_map_);
  }

 protected:
  using StorageAllocaBaseVisitor::VisitExpr_;

  void CreateToken(const ExprNode* op, bool can_realloc) final {    // 用于为一个Expr创建StorageToken
    CHECK(!token_map_.count(op));                                   // 如果这个Expr之前未创建StorageToken
    std::vector<StorageToken*> tokens;
    int device_type =                                               // 此例中均为0，即默认的CPU Device
        node_device_map_.count(GetRef<Expr>(op)) ? node_device_map_[GetRef<Expr>(op)]->value : 0;
    // std::cout << device_type << std::endl;
    if (const auto* tuple_type = op->checked_type().as<TupleTypeNode>()) {  // 是VarNode而非TupleTypeNode，不走此分支
      for (Type t : tuple_type->fields) {                                   // 猜测TupleNode是那种可能有多个输出的op/call
        const auto* ttype = t.as<TensorTypeNode>();                         // 这样要为每个输出构建一个StorageToken
        CHECK(ttype);
        StorageToken* token = arena_->make<StorageToken>();
        token->ttype = ttype;
        token->device_type = device_type;
        tokens.push_back(token);
      }
    } else {
      const auto* ttype = op->checked_type().as<TensorTypeNode>();    // VarNode的Type为TensorTypeNode，由前面的CheckType()函数Infer得到
      CHECK(ttype);
      StorageToken* token = arena_->make<StorageToken>();   // 用默认参数创建一个StorageToken
      token->ttype = ttype;                                 // 设置这个Token的参数
      token->device_type = device_type;
      tokens.push_back(token);
    }
    token_map_[op] = tokens;
  }

  void VisitExpr_(const CallNode* op) final {         
    // create token for the call node.
    CreateToken(op, true);                                  // 为这个CallNode创建StorageToken
    // for each input, visit argument token.
    for (Expr arg : op->args) {
      for (StorageToken* tok : GetToken(arg)) {             // 这个CallNode的每个输入的RefCount都+1
        tok->ref_counter += 1;
      }
    }
  }

 private:
  // allocator
  support::Arena* arena_;
  Map<Expr, Integer> node_device_map_;
};

class StorageAllocator : public StorageAllocaBaseVisitor {
 public:
  /*!
   * \return totoal number of bytes allocated
   */
  size_t TotalAllocBytes() const {
    size_t total = 0;
    for (const auto* p : data_) {
      total += p->max_bytes;
    }
    return total;
  }

  // Run storage allocation for a function.
  Map<Expr, Array<IntegerArray> > Plan(const Function& func) {
    prototype_ = StorageAllocaInit(&arena_).GetInitTokenMap(func);      // 遍历整个网络，获得一个初始的Expr->StorageToken的Map，存在Prototype_
    this->Run(func);                                                    // 遍历整个网络，更新了这个Map->StorageToken的Map，存在token_map_中
                                                                        // 其中相关VisitExpr_()被重载了，所以相同的Run()跑出了不同的结果
    // The value of smap contains two integer arrays where the first array
    // contains the planned storage ids and the second holds the device types.
    Map<Expr, Array<IntegerArray> > smap;                               
    int num_annotated_nodes = 0;
    int num_nodes = 0;

    for (const auto& kv : token_map_) {
      std::vector<Integer> storage_ids;
      std::vector<Integer> device_types;
      for (StorageToken* tok : kv.second) {
        if (tok->device_type) {
          // std::cout << "!!!!!!!!" << std::endl;
          num_annotated_nodes++;
        }
        num_nodes++;
        storage_ids.push_back(tok->storage_id);
        device_types.push_back(tok->device_type);
      }
      smap.Set(GetRef<Expr>(kv.first), Array<IntegerArray>({storage_ids, device_types}));   // 这就是所谓的smap，其相对于token_map_多了一个device_type的信息
    }
    // Either all or none of the nodes should be annotated.
    if (num_annotated_nodes != 0 && num_annotated_nodes != num_nodes) {
      LOG(FATAL) << num_annotated_nodes << " out of " << num_nodes
                 << "expressions are assigned with virtual device types. Either all "
                    "or none of the expressions are expected to be annotated.";
    }
    return smap;
  }

 protected:
  using StorageAllocaBaseVisitor::VisitExpr_;
  // override create token by getting token as prototype requirements.
  void CreateToken(const ExprNode* op, bool can_realloc) final {
    // std::cout << "Superise !!!" << std::endl;
    CHECK(!token_map_.count(op));
    auto it = prototype_.find(op);
    CHECK(it != prototype_.end());
    std::vector<StorageToken*> tokens;
    for (StorageToken* tok : it->second) {    // 从StorageAllocaInit的结果中找到这个初始的StorageToken  
      if (can_realloc) {
        tokens.push_back(Request(tok));       // 对于可以内存复用的Node的处理   -----   未看完 0112 - (8)
      } else {
        // Allocate a new token,              // 对这个StorageToken做进一步处理，别看这里有这么多指针，其实都是指向最开始这个InitStorageToken
        StorageToken* allocated_tok = Alloc(tok, GetMemorySize(tok));   // 并不是真正的分配memory，StorageToken只是维护这个Node占用Memory的大小，生存期等相关信息
        allocated_tok->device_type = tok->device_type;
        // ensure it never get de-allocated.
        allocated_tok->ref_counter += 1;
        tokens.push_back(allocated_tok);
      }
    }
    token_map_[op] = tokens;                // 将最终的这个Token存在StorageAllocator类的token_map_中
  }
  // The call map
  void VisitExpr_(const CallNode* op) final {
    std::vector<StorageToken*> args;
    // for each input, visit argument token.
    for (Expr arg : op->args) {
      for (StorageToken* tok : GetToken(arg)) {
        args.push_back(tok);
      }
    }
    // create token for the call node.
    CreateToken(op, true);              // CallNode的这个Can_release被设为true
    // check if there is orphaned output that can be released immediately.
    for (StorageToken* tok : token_map_.at(op)) {
      CheckForRelease(tok);             // 如果这个Call是个中间结果，则将之置为可以Release的，用free_来维护
    }
    for (StorageToken* tok : args) {    // 对Call的每个输入参数也这么做
      tok->ref_counter -= 1;
      CheckForRelease(tok);
    }
  }
  /*!
   * \brief ceil(size/word_size) to get number of words.
   * \param size The original size.
   * \param word_size The element size.
   */
  static size_t DivRoundUp(size_t size, size_t word_size) {
    return (size + word_size - 1) / word_size;
  }
  /*!
   * \brief Get the memory requirement.
   * \param prototype The prototype token.
   * \return The required memory size.
   */
  size_t GetMemorySize(StorageToken* prototype) {   // 获取一个Token对应的一个Memory Size，其实就是TensorType的shape相乘再对齐
    const TensorTypeNode* ttype = prototype->ttype;
    CHECK(ttype != nullptr);
    size_t size = 1;
    for (IndexExpr dim : ttype->shape) {
      const int64_t* pval = tir::as_const_int(dim);
      CHECK(pval != nullptr) << "Cannot allocate memory symbolic tensor shape " << ttype->shape;
      CHECK_GE(*pval, 0) << "Cannot allocate memory for tensor with negative shape" << *pval;
      size *= static_cast<size_t>(pval[0]);
    }
    size *= DivRoundUp(ttype->dtype.bits() * ttype->dtype.lanes(), 8);
    return size;
  }
  /*!
   * \brief Request a storage token for a given prototype.
   * \param prototype. The prototype storage token.
   * \return The result token.
   */
  StorageToken* Request(StorageToken* prototype) {
    // calculate the size;
    size_t size = GetMemorySize(prototype);
    // search memory block in [size / match_range_, size * match_range_)
    if (match_range_ == 0) {
      return this->Alloc(prototype, size);
    }
    auto begin = free_.lower_bound(size / match_range_);
    auto mid = free_.lower_bound(size);
    auto end = free_.upper_bound(size * match_range_);
    // search for memory blocks larger than requested
    for (auto it = mid; it != end; ++it) {
      StorageToken* tok = it->second;
      if (tok->device_type != prototype->device_type) continue;
      CHECK_EQ(tok->ref_counter, 0);
      // Use exect matching strategy
      tok->max_bytes = std::max(size, tok->max_bytes);
      tok->ref_counter = prototype->ref_counter;
      // find a exact match, erase from map and return
      free_.erase(it);
      return tok;
    }
    // then search for memory blocks smaller than requested space
    for (auto it = mid; it != begin;) {
      --it;
      StorageToken* tok = it->second;
      if (tok->device_type != prototype->device_type) continue;
      CHECK_EQ(tok->ref_counter, 0);
      // Use exect matching strategy
      tok->max_bytes = std::max(size, tok->max_bytes);
      tok->ref_counter = prototype->ref_counter;
      // erase from map and return
      free_.erase(it);
      return tok;
    }
    // cannot find anything return a new one.
    return this->Alloc(prototype, size);
  }
  /*!
   * \brief Allocate a storage token by consuming prototype
   * \param prototype The prototype token.
   * \param size The size of memory being requested.
   */
  StorageToken* Alloc(StorageToken* prototype, size_t size) {
    prototype->max_bytes = size;
    prototype->storage_id = static_cast<int64_t>(data_.size());
    data_.push_back(prototype);
    return prototype;
  }
  /*!
   * \brief Check if we can release token.
   * \tok The token to be released.
   */
  void CheckForRelease(StorageToken* tok) {
    CHECK_GE(tok->storage_id, 0);
    CHECK_GE(tok->ref_counter, 0);
    if (tok->ref_counter == 0) {
      free_.insert({tok->max_bytes, tok});
    }
  }

 private:
  // allocator
  support::Arena arena_;
  // scale used for rough match
  size_t match_range_{16};
  // free list of storage entry
  std::multimap<size_t, StorageToken*> free_;
  // all the storage resources available
  std::vector<StorageToken*> data_;                                         
  /*! \brief internal prototype token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> > prototype_;    // init的内存分配方案
};

Map<Expr, Array<IntegerArray> > GraphPlanMemory(const Function& func) {
  return StorageAllocator().Plan(func);
}

TVM_REGISTER_GLOBAL("relay.backend.GraphPlanMemory").set_body_typed(GraphPlanMemory);

}  // namespace relay
}  // namespace tvm
