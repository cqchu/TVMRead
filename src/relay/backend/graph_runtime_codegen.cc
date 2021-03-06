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
 * \file relay/backend/graph_codegen.cc
 * \brief Graph runtime codegen
 */

#include <dmlc/any.h>
#include <dmlc/json.h>
#include <tvm/ir/module.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/device_api.h>

#include <list>
#include <string>
#include <vector>

#include "compile_engine.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace backend {

class GraphNode;
class GraphInputNode;
class GraphOpNode;

using IntegerArray = Array<Integer>;
using ShapeVector = std::vector<std::vector<int64_t>>;
using GraphAttrs = std::unordered_map<std::string, dmlc::any>;
using GraphObjectPtr = std::shared_ptr<GraphNode>;
using GraphInputObjectPtr = std::shared_ptr<GraphInputNode>;
using GraphOpObjectPtr = std::shared_ptr<GraphOpNode>;
using TargetsMap = std::unordered_map<int, Target>;

/*! \brief Lowered outputs */
struct LoweredOutput {
  std::string graph_json;
  Map<String, IRModule> lowered_funcs;
  Array<tvm::runtime::Module> external_mods;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
};

/*! \brief Node types */
enum GraphNodeType {
  kGraphNop,
  kGraphInputNode,
  kGraphOpNode,
};

class GraphNodeRef {
 public:
  GraphNodeRef() {}
  GraphNodeRef(int ident, int index, int version = 0)
      : ident_(ident), index_(index), version_(version) {}

  inline void Save(dmlc::JSONWriter* writer) const {
    writer->BeginArray();
    writer->WriteArrayItem(ident_);
    writer->WriteArrayItem(index_);
    writer->WriteArrayItem(version_);
    writer->EndArray();
  }

  inline void Load(dmlc::JSONReader* reader) { LOG(FATAL) << "Not implemented."; }

 protected:
  int ident_;
  int index_{0};
  int version_{0};
};

/*! \brief Base Node class */
class GraphNode {
 public:
  GraphNode() {}
  virtual void Save(dmlc::JSONWriter* writer) const {}
  virtual void Load(dmlc::JSONReader* reader) {}
  virtual GraphNodeType Type() const { return kGraphNop; }
  virtual ~GraphNode() {}

 public:
  int num_outputs_{1};
  std::string name_;
  GraphAttrs attrs_;
};

/*! \brief Input Node */
class GraphInputNode : public GraphNode {
 public:
  GraphInputNode() {}
  GraphInputNode(const std::string& name, const GraphAttrs& attrs) {
    name_ = name;
    attrs_ = attrs;
  }

  GraphNodeType Type() const override { return kGraphInputNode; }

  void Save(dmlc::JSONWriter* writer) const override {
    const std::string op_name{"null"};
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_name);
    writer->WriteObjectKeyValue("name", this->name_);
    writer->WriteObjectKeyValue("inputs", std::list<int>());
    writer->EndObject();
  }
  static std::shared_ptr<GraphNode> make_node_ptr(const std::string& name,
                                                  const GraphAttrs& attrs) {
    auto ptr = std::make_shared<GraphInputNode>(name, attrs);
    return std::dynamic_pointer_cast<GraphNode>(ptr);
  }
};

/*! \brief Op Node */
class GraphOpNode : public GraphNode {
 public:
  GraphOpNode() {}
  GraphOpNode(const std::string& name, const GraphAttrs& nd_attrs, const std::string& op_name,
              const std::vector<GraphNodeRef>& inputs, const GraphAttrs& attrs,
              size_t num_outputs = 1) {
    name_ = name;
    attrs_ = nd_attrs;
    op_name_ = op_name;
    inputs_ = inputs;
    op_attrs_ = attrs_;
    num_outputs_ = num_outputs;
    op_attrs_["func_name"] = op_name_;
    op_attrs_["flatten_data"] = std::string("0");
    op_attrs_["num_inputs"] = std::to_string(inputs_.size());
    op_attrs_["num_outputs"] = std::to_string(num_outputs_);
  }

  GraphNodeType Type() const override { return kGraphOpNode; }

  void Save(dmlc::JSONWriter* writer) const override {
    GraphAttrs attrs = op_attrs_;
    attrs["func_name"] = this->op_name_;
    attrs["flatten_data"] = std::string("0");
    attrs["num_inputs"] = std::to_string(this->inputs_.size());
    attrs["num_outputs"] = std::to_string(this->num_outputs_);
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_type_name_);
    writer->WriteObjectKeyValue("name", name_);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->WriteObjectKeyValue("inputs", this->inputs_);
    writer->EndObject();
  }
  static std::shared_ptr<GraphNode> make_node_ptr(const std::string& name,
                                                  const GraphAttrs& nd_attrs,
                                                  const std::string& op_name,
                                                  const std::vector<GraphNodeRef>& inputs,
                                                  const GraphAttrs& attrs, size_t num_outputs = 1) {
    auto ptr = std::make_shared<GraphOpNode>(name, nd_attrs, op_name, inputs, attrs, num_outputs);
    return std::dynamic_pointer_cast<GraphNode>(ptr);
  }

 public:
  std::string op_name_;
  std::vector<GraphNodeRef> inputs_;
  GraphAttrs op_attrs_;

 private:
  const std::string op_type_name_{"tvm_op"};
};

/*! \brief Code generator for graph runtime */
class GraphRuntimeCodegen : public backend::MemoizedExprTranslator<std::vector<GraphNodeRef>> {
 public:
  GraphRuntimeCodegen(runtime::Module* mod, const TargetsMap& targets) : mod_(mod) {
    compile_engine_ = CompileEngine::Global();    // 这个ref类指向一个CompileEngineImpl类对象
    targets_ = targets;
  }

  LoweredOutput Codegen(relay::Function func) {
    auto pf = GetPackedFunc("relay.backend.GraphPlanMemory");     // src/relay/backend/graph_plan_memory.cc中398行
    storage_device_map_ = (*pf)(func);                            // 一个Map，其中存着若干关于 Expr -> <StorageID, DeviceType> 的映射
    // First we convert all the parameters into input nodes.
    std::cout << "***********************************************************************************" << std::endl;
    std::cout << "********************************* Code Generation *********************************" << std::endl;
    std::cout << "***********************************************************************************" << std::endl;
    for (auto param : func->params) {
      auto node_ptr = GraphInputNode::make_node_ptr(param->name_hint(), GraphAttrs());  // 用此时输入的那个Var构造一个GraphNode
      var_map_[param.get()] = AddNode(node_ptr, param);                                 // GraphRuntimeCodegen中维护了一个graph, 把这个GraphNode
    }                                                                                   // 插入Graph中，此时Graph中的Node都是这些输入的Node
    // std::cout << AsText(func, false) << std::endl;
    // std::cout << func->body->GetTypeKey() << std::endl;
    heads_ = VisitExpr(func->body);         // 真正的开始去low, func->body是Call类型
    // std::cout << AsText(func, false);
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);           // 构造一个默认JSONWriter，然后将之与这个os绑定
    GetJSON(&writer);
    LoweredOutput ret;
    ret.graph_json = os.str();
    ret.params = params_;

    for (auto& kv : lowered_funcs_) {
      if (ret.lowered_funcs.count(kv.first) == 0) {
        ret.lowered_funcs.Set(kv.first, IRModule());
      }
      auto& mod = ret.lowered_funcs[kv.first];
      mod->Update(kv.second);
      ret.lowered_funcs.Set(kv.first, mod);
    }
    ret.external_mods = compile_engine_->LowerExternalFunctions();
    return ret;
  }

 protected:
  /*!
   * \brief Extract shape from expr to vector<int64_t>
   *
   * \param shape
   * \return std::vector<int64_t>
   */
  std::vector<int64_t> _ShapeToJSON(tvm::Array<IndexExpr> shape) {
    std::vector<int64_t> ret;
    for (IndexExpr dim : shape) {
      const int64_t* pval = tir::as_const_int(dim);
      ret.push_back(*pval);
    }
    return ret;
  }

  /*!
   * \brief Add node to graph
   *
   * \param node
   * \param expr
   * \return std::vector<_NodeRef>
   */
  std::vector<GraphNodeRef> AddNode(GraphObjectPtr node, Expr expr) { // 传入一个Init的GraphNode以及Expr, 由Expr设置GraphNode的Attrs
    auto checked_type = expr->checked_type();                         // 存入GraphRuntimeCodegen::node_数组中，同时返回这个node的Ref
    size_t count = storage_device_map_.count(expr);
    CHECK_GT(count, 0) << "Expr is not existing in storage plan";
    auto storage_device_info = storage_device_map_[expr];
    CHECK_EQ(storage_device_info.size(), 2);                    // 获取这个Expr的 StorageID 和 DeviceID
    // storage
    std::vector<int64_t> storage_info;
    for (auto& v : storage_device_info[0]) {
      storage_info.push_back(v->value);
    }
    node->attrs_["storage_id"] = std::move(storage_info);       // 用Expr的StorageToken设置其对应node的信息
    // type
    std::vector<int64_t> device_types;
    for (auto& v : storage_device_info[1]) {
      device_types.push_back(v->value);
    }
    size_t num_unknown_devices = std::count(device_types.begin(), device_types.end(), 0);
    if (num_unknown_devices != 0 && num_unknown_devices != device_types.size()) {
      LOG(FATAL) << "The graph contains not annotated nodes for "
                 << "heterogeneous execution. All nodes must be "
                 << "annotated.";
    }
    if (num_unknown_devices == 0) {
      node->attrs_["device_index"] = device_types;              // 用Expr的StorageToken设置其对应node的信息
    }
    auto node_id = nodes_.size();
    nodes_.push_back(node);
    // Tuple return value, flatten as tuple
    if (const auto* tuple_type = checked_type.as<TupleTypeNode>()) {    // 此例中未执行
      std::vector<GraphNodeRef> ret;
      ShapeVector shape;
      std::vector<std::string> dtype;
      for (size_t i = 0; i < tuple_type->fields.size(); ++i) {
        if (const auto* typ = tuple_type->fields[i].as<TensorTypeNode>()) {
          ret.push_back(GraphNodeRef(node_id, i));
          shape.emplace_back(_ShapeToJSON(typ->shape));
          dtype.emplace_back(DType2String(typ->dtype));
        } else {
          LOG(FATAL) << "type " << checked_type->GetTypeKey() << " not supported";
        }
      }
      CHECK_EQ(node->Type(), kGraphOpNode);
      auto op_nd = std::dynamic_pointer_cast<GraphOpNode>(node);
      op_nd->attrs_["shape"] = shape;
      op_nd->attrs_["dtype"] = dtype;
      op_nd->num_outputs_ = tuple_type->fields.size();
      return ret;
    }
    // Normal tensor return type
    if (const auto* tensor_type = checked_type.as<TensorTypeNode>()) {
      ShapeVector shape;
      std::vector<std::string> dtype;
      shape.emplace_back(_ShapeToJSON(tensor_type->shape));
      dtype.emplace_back(DType2String(tensor_type->dtype));
      node->attrs_["shape"] = shape;                    // 用Expr的StorageToken设置其对应node的信息
      node->attrs_["dtype"] = dtype;                    // 用Expr的StorageToken设置其对应node的信息
    } else {
      LOG(FATAL) << "type " << checked_type->GetTypeKey() << " not supported";
    }
    return {GraphNodeRef(node_id, 0)};
  }

  std::vector<GraphNodeRef> VisitExpr_(const VarNode* op) override {
    Expr expr = GetRef<Expr>(op);
    return var_map_[expr.get()];
  }

  std::vector<GraphNodeRef> VisitExpr_(const ConstantNode* op) override {
    Expr expr = GetRef<Expr>(op);
    size_t index = params_.size();
    std::string name = "p" + std::to_string(index);
    params_[name] = op->data;
    auto node = GraphInputNode::make_node_ptr(name, GraphAttrs());
    return AddNode(node, expr);
  }

  std::vector<GraphNodeRef> VisitExpr_(const TupleNode* op) override {
    std::vector<GraphNodeRef> fields;
    for (auto field : op->fields) {
      auto ref_vec = VisitExpr(field);
      for (auto ref : ref_vec) {
        fields.push_back(ref);
      }
    }
    return fields;
  }

  std::vector<GraphNodeRef> GraphAddCallNode(const CallNode* op, const std::string& op_name,
                                             const std::string& func_name) {
    std::vector<GraphNodeRef> inputs;
    for (auto arg : op->args) {
      auto res = VisitExpr(arg);
      for (auto nr : res) {
        inputs.push_back(nr);
      }
    }
    auto node = GraphOpNode::make_node_ptr(op_name, GraphAttrs(), func_name, inputs, GraphAttrs());
    return AddNode(node, GetRef<Expr>(op));
  }

  std::vector<GraphNodeRef> VisitExpr_(const CallNode* op) override {     // 对Call类型Lower的过程
    Expr expr = GetRef<Expr>(op);
    Function func;
    if (op->op.as<OpNode>()) {
      LOG(FATAL) << "Operators should be transformed away; try applying"
                 << "the fuse_ops transformation to the expression.";
    } else if (op->op.as<GlobalVarNode>()) {
      LOG(FATAL) << "Not implemented";
    } else if (op->op.as<FunctionNode>()) {                 // 经过OP Fusion后，原来图中Call中的那些原始OP都被fuse转成了Function
      func = GetRef<Function>(op->op.as<FunctionNode>());   // 这也是这种数据结构为什么被命名为Call的原因
    } else {
      LOG(FATAL) << "TVM runtime does not support calls to " << op->op->GetTypeKey();
    }
    if (!func->HasNonzeroAttr(attr::kPrimitive)) {
      LOG(FATAL) << "TVM only support calls to primitive functions "
                 << "(i.e functions composed of fusable operator invocations)";
    }

    // 上面获取了这个Call对应的那个SubFunction (OpFusion后的子图)
    // std::cout << "###################" << std::endl;
    // std::cout << AsText(func, false) << std::endl;
    // std::cout << "*******************" << std::endl;

    auto pf0 = GetPackedFunc("relay.backend._make_CCacheKey");      // CCacheKey的构造函数，返回一个CCacheKey对象
    auto pf1 = GetPackedFunc("relay.backend._CompileEngineLower");  // CompileEngine::Lower()函数
    Target target;
    // Handle external function
    if (func->GetAttr<String>(attr::kCompiler).defined()) {                         // No external function，先不看
      target = tvm::target::ext_dev();
      CCacheKey key = (*pf0)(func, target);
      CachedFunc ext_func = (*pf1)(compile_engine_, key);
      CHECK(ext_func.defined()) << "External function is not defined.";

      // Step into the functions that are handled by external codegen to
      // collect metadata.
      const auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
      std::string symobl = std::string(name_node.value());
      ConstantUpdater const_visit(symobl, &params_);
      const_visit(func);

      return GraphAddCallNode(op, ext_func->func_name, ext_func->func_name);
    }

    CHECK_GE(storage_device_map_.count(expr), 0);
    auto& device_type = storage_device_map_[expr][1];
    auto call_dev_type = device_type[0]->value;
    // Normal Relay Function
    if (targets_.size() == 1) {             // 同构编译
      // homogeneous execution.
      const auto& it = targets_.begin();
      target = (*it).second;
    } else {
      // heterogeneous execution.
      std::string call_dev_name;
      if (call_dev_type == 0) {
        call_dev_name = "llvm";
      } else {
        call_dev_name = runtime::DeviceName(call_dev_type);
      }
      if (targets_.count(call_dev_type) == 0) {
        LOG(FATAL) << "No target is provided for device " << call_dev_name;
      }
      target = targets_[call_dev_type];
    }
    // 到此获取了这个Call中的那个函数，以及对应的编译的target
    CCacheKey key = (*pf0)(func, target);   // 构造CCacheKey
    CachedFunc lowered_func = (*pf1)(compile_engine_, key);   // CompileEngine::Lower()函数, Lower的结果是一个CachedFunc
    if (!lowered_funcs_.count(target->str())) {
      lowered_funcs_[target->str()] = IRModule();
    }
    lowered_funcs_[target->str()]->Update(lowered_func->funcs); // 将编译的结果加到GraphRuntimeCodegen相关成员中
    return GraphAddCallNode(op, _GetUniqueName(lowered_func->func_name), lowered_func->func_name);  // 在Graph中更新nodes_
  }

  std::vector<GraphNodeRef> VisitExpr_(const LetNode* op) override {
    CHECK_EQ(var_map_.count(op->var.get()), 0);
    var_map_[op->var.get()] = VisitExpr(op->value);
    return VisitExpr(op->body);
  }
  std::vector<GraphNodeRef> VisitExpr_(const TupleGetItemNode* op) override {
    auto vtuple = VisitExpr(op->tuple);
    return {vtuple[op->index]};
  }
  std::vector<GraphNodeRef> VisitExpr_(const OpNode* op) override {
    throw std::runtime_error("can not compile op in non-eta expanded form");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const GlobalVarNode* op) override {
    throw std::runtime_error("");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const IfNode* op) override {
    throw std::invalid_argument("if not supported");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const FunctionNode* op) override {
    CHECK(op->GetAttr<String>(attr::kCompiler).defined())
        << "Only functions supported by custom codegen";
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const RefCreateNode* op) override {
    throw std::invalid_argument("reference not supported");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const RefReadNode* op) override {
    throw std::invalid_argument("reference not supported");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const RefWriteNode* op) override {
    throw std::invalid_argument("reference not supported");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const ConstructorNode* op) override {
    throw std::invalid_argument("ADT constructor case not yet implemented");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const MatchNode* op) override {
    throw std::invalid_argument("match case not yet implemented");
    return {};
  }
  /*!
   * \brief Generate Graph JSON
   *
   * \param writer json writer
   */
  void GetJSON(dmlc::JSONWriter* writer) {
    std::vector<size_t> arg_nodes;
    for (size_t i = 0; i < nodes_.size(); ++i) {
      auto node = nodes_[i];
      if (node->Type() == kGraphInputNode) {
        arg_nodes.push_back(i);
      }
    }
    size_t num_entry = 0;
    ShapeVector shapes;
    std::vector<size_t> storage_ids;
    std::vector<size_t> device_types;
    std::vector<std::string> dltypes;
    std::vector<size_t> node_row_ptr{0};
    for (auto node : nodes_) {
      const auto& shape_vec = dmlc::get<ShapeVector>(node->attrs_["shape"]);
      const auto& storage_id = dmlc::get<std::vector<int64_t>>(node->attrs_["storage_id"]);
      const auto& dtype_vec = dmlc::get<std::vector<std::string>>(node->attrs_["dtype"]);

      CHECK_EQ(node->num_outputs_, shape_vec.size());
      num_entry += node->num_outputs_;

      shapes.insert(shapes.end(), shape_vec.begin(), shape_vec.end());
      dltypes.insert(dltypes.end(), dtype_vec.begin(), dtype_vec.end());
      storage_ids.insert(storage_ids.end(), storage_id.begin(), storage_id.end());
      if (node->attrs_.count("device_index")) {
        const auto& dev_types = dmlc::get<std::vector<int64_t>>(node->attrs_["device_index"]);
        device_types.insert(device_types.end(), dev_types.begin(), dev_types.end());
      }
      node_row_ptr.push_back(num_entry);
    }
    writer->BeginObject();
    writer->WriteObjectKeyValue("nodes", nodes_);
    writer->WriteObjectKeyValue("arg_nodes", arg_nodes);
    writer->WriteObjectKeyValue("heads", heads_);
    std::unordered_map<std::string, std::vector<dmlc::any>> attrs;
    attrs["shape"].emplace_back(std::string("list_shape"));
    attrs["shape"].emplace_back(shapes);
    attrs["storage_id"].emplace_back(std::string("list_int"));
    attrs["storage_id"].emplace_back(storage_ids);
    if (device_types.size()) {
      attrs["device_index"].emplace_back(std::string("list_int"));
      attrs["device_index"].emplace_back(device_types);
    }
    attrs["dltype"].emplace_back(std::string("list_str"));
    attrs["dltype"].emplace_back(dltypes);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->WriteObjectKeyValue("node_row_ptr", node_row_ptr);
    writer->EndObject();
  }

  /*!
   * \brief Get unique name for func
   *
   * \param name
   * \return std::string
   */
  std::string _GetUniqueName(const std::string& name) {
    if (!name_map_.count(name)) {
      name_map_[name] = 1;
      return name;
    }
    auto index = name_map_[name];
    name_map_[name] += 1;
    return _GetUniqueName(name + std::to_string(index));
  }

 protected:
  /*! \brief nodes */
  std::vector<GraphObjectPtr> nodes_;                                       // 维护这张Graph中的Node
  /*! \brief output of graph */
  std::vector<GraphNodeRef> heads_;
  /*! \brief mod */
  runtime::Module* mod_;
  /*! \brief variable map */
  std::unordered_map<const Object*, std::vector<GraphNodeRef>> var_map_;    // 记录网络中输入的那些节点对应的GraphNode
  /*! \brief target device */
  TargetsMap targets_;
  /*! \brief params */
  std::unordered_map<std::string, runtime::NDArray> params_;
  /*! \brief plan memory of device result */
  Map<Expr, Array<IntegerArray>> storage_device_map_;
  /*! \brief lowered funcs */
  std::unordered_map<std::string, IRModule> lowered_funcs_;                 // 一个Relay Func的Lower结果
  /*! \brief name map */
  std::unordered_map<std::string, size_t> name_map_;
  /*! \brief compile engine */
  CompileEngine compile_engine_;
};

class GraphRuntimeCodegenModule : public runtime::ModuleNode {
 public:
  GraphRuntimeCodegenModule() {}
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "init") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.num_args, 2) << "The expected of arguments are: "
                                   << "runtime::Module mod and Map<int, Target> targets";
        void* mod = args[0];                      // 此例中为nullptr
        Map<Integer, tvm::Target> tmp = args[1];  // TargetsMap, 比如 {1: llvm}
        TargetsMap targets;
        for (const auto& it : tmp) {              // 获取输入Targets中有效的那些，然后放在这个新的TargetsMap类型的targets中
          auto dev_type = it.first.as<tir::IntImmNode>();
          CHECK(dev_type);
          targets[dev_type->value] = it.second;
        }
        codegen_ =                                                                                    // 本函数最核心的工作，将本类的codegen_指针指向一个GraphRuntimeCodegen对象
            std::make_shared<GraphRuntimeCodegen>(reinterpret_cast<runtime::Module*>(mod), targets);  // 这个codegen_中有一个指向CompileEngine的指针
      });
    } else if (name == "codegen") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Function func = args[0];
        this->output_ = this->codegen_->Codegen(func);  // 真正的Codegen这个Func
      });
    } else if (name == "get_graph_json") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->output_.graph_json; });
    } else if (name == "list_params_name") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Array<runtime::String> ret;
        for (const auto& kv : this->output_.params) {
          ret.push_back(kv.first);
        }
        *rv = ret;
      });
    } else if (name == "get_param_by_name") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        String key = args[0];
        CHECK_GT(this->output_.params.count(key), 0);
        *rv = this->output_.params[key];
      });
    } else if (name == "get_irmodule") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->output_.lowered_funcs;
      });
    } else if (name == "get_external_modules") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->output_.external_mods;
      });
    } else {
      return PackedFunc([](TVMArgs args, TVMRetValue* rv) {});
    }
  }

  const char* type_key() const final { return "RelayGraphRuntimeCodegenModule"; }

 private:
  std::shared_ptr<GraphRuntimeCodegen> codegen_;
  LoweredOutput output_;
};

runtime::Module CreateGraphCodegenMod() {
  auto ptr = make_object<GraphRuntimeCodegenModule>();  // 创建了一个默认的GraphRuntimeCodegenModule类
  return runtime::Module(ptr);                          // 用之构造一个Ref类返回
}

TVM_REGISTER_GLOBAL("relay.build_module._GraphRuntimeCodegen")
    .set_body([](TVMArgs args, TVMRetValue* rv) { *rv = CreateGraphCodegenMod(); });

}  // namespace backend
}  // namespace relay
}  // namespace tvm

namespace dmlc {
namespace json {
// JSON utils
template <typename T>
inline bool SameType(const dmlc::any& data) {
  return std::type_index(data.type()) == std::type_index(typeid(T));
}

template <>
struct Handler<std::shared_ptr<tvm::relay::backend::GraphNode>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::shared_ptr<tvm::relay::backend::GraphNode>& data) {
    data->Save(writer);
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::shared_ptr<tvm::relay::backend::GraphNode>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};
template <>
struct Handler<std::unordered_map<std::string, dmlc::any>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::unordered_map<std::string, dmlc::any>& data) {
    writer->BeginObject();
    for (const auto& kv : data) {
      auto k = kv.first;
      const dmlc::any& v = kv.second;
      if (SameType<std::string>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::string>(v));
      } else if (SameType<int>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<int>(v));
      } else if (SameType<std::vector<size_t>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<size_t>>(v));
      } else if (SameType<std::vector<std::vector<int64_t>>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<std::vector<int64_t>>>(v));
      } else if (SameType<std::vector<std::string>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<std::string>>(v));
      } else if (SameType<std::vector<dmlc::any>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<dmlc::any>>(v));
      } else {
        LOG(FATAL) << "Not supported";
      }
    }
    writer->EndObject();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::unordered_map<std::string, dmlc::any>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};

template <>
struct Handler<std::vector<dmlc::any>> {
  inline static void Write(dmlc::JSONWriter* writer, const std::vector<dmlc::any>& data) {
    writer->BeginArray();
    for (const auto& v : data) {
      if (SameType<std::string>(v)) {
        writer->WriteArrayItem(dmlc::get<std::string>(v));
      } else if (SameType<int>(v)) {
        writer->WriteArrayItem(dmlc::get<int>(v));
      } else if (SameType<std::vector<size_t>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<size_t>>(v));
      } else if (SameType<std::vector<std::vector<int64_t>>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<std::vector<int64_t>>>(v));
      } else if (SameType<std::vector<std::string>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<std::string>>(v));
      } else {
        LOG(FATAL) << "Not supported";
      }
    }
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, std::vector<dmlc::any>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};
}  // namespace json
}  // namespace dmlc
