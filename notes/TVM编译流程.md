本文介绍TVM的整个编译流程，具体来说是构建了一个`conv-bn-relu`的block，主要是观察TVM如何对这样的block如何进行图优化以及底层的codegen，这个示例程序主要如下

```python
##### Net Structure #####
data = relay.var("data", shape=data_shape, dtype=dtype)
weight = relay.var("conv0_weight")
gamma = relay.var("bn0_gamma")
beta = relay.var("bn0_beta")
moving_mean = relay.var("bn0_moving_mean")
moving_var = relay.var("bn0_moving_var")

body = relay.nn.conv2d(data=data, weight=weight, channels=32, kernel_size=(3, 3), strides=(1, 1),   
                       padding=(1, 1), data_layout=data_layout, kernel_layout=kernel_layout)
body = relay.nn.batch_norm(data=body, gamma=gamma, beta=beta, moving_mean=moving_mean, 
                           moving_var=moving_var, axis=bn_axis)[0]
out = relay.nn.relu(data=body)
free_vars = relay.analysis.free_vars(out)
net = relay.Function(free_vars, out)
mod = tvm.IRModule.from_expr(net)

##### Compile #####
target = tvm.target.create("llvm")
with tvm.transform.PassContext(opt_level=opt_level):
    graph, lib, params = relay.build(mod, target, params=params)

```

其主要可以分为两个部分，第一个是网络结构的搭建，自己定义一个网络并将之转化为一个`tvm.IRModule`，然后再对这个`IRModule`进行真正的编译优化。

需要注意的是TVM在python的import阶段就已经做了很多事，包括`PackedFunc/OP`的注册等，这些此处就不再赘述了。

### 构建模型

构建模型主要分三件事，首先定义变量，然后用变量和算子构建网络，最后将网络转化为一个`IRModule`。

#### Var的创建

其中定义变量使用的是`var()`函数，该函数使用传进来的参数如`shape/dtype`等构造一个`type_annotation`，然后用之创建python的`Var`对象，看看怎么构造的

```python
class Var(ExprWithOp):            
    def __init__(self, name_hint, type_annotation=None):
        self.__init_handle_by_constructor__(_ffi_api.Var, name_hint, type_annotation)
        
    def __init_handle_by_constructor__(self, fconstructor, *args): # 继承自ObjectBase类
        self.handle = None
        handle = __init_by_constructor__(fconstructor, args)
        self.handle = handle
        
def __init_handle_by_constructor__(fconstructor, args):			# 根据传入的参数及构造函数类型在C++中创建对象并将之handle返回给python
    temp_args = []
    values, tcodes, num_args = _make_tvm_args(args, temp_args)  # 将python的参数打包成C++可以接收的参数 - TVMArgs
    ret_val = TVMValue()
    ret_tcode = ctypes.c_int()                                  # 设置返回值等等
    if _LIB.TVMFuncCall(                                        # 看来是用传进来的参数，构造函数构造一个对象了
            fconstructor.handle, values, tcodes, ctypes.c_int(num_args),    
            ctypes.byref(ret_val), ctypes.byref(ret_tcode)) != 0:
        raise get_last_ffi_error()
    handle = ret_val.v_handle                                   # 获取这个对象的handle并返回回去
    return handle
```

其中`_ffi_apt.api.Var`是C++中`Var`类构造函数的`PackedFunc`，显然其应该就在C++中构造了`Var`对象，并将之`handle`返回给python，python中的`Var`类就是维护这个C++中的对象的`handle`。

再看下C++，其中用`Var`和`VarNode`两个类维护`relay`中的`var`，瞅下代码

```C++
class Var : public Expr {
 public:
  TVM_DLL Var(String name_hint, Type type_annotation) : Var(Id(name_hint), type_annotation) {}
  TVM_DLL Var(Id vid, Type type_annotation);
  TVM_DEFINE_OBJECT_REF_METHODS(Var, RelayExpr, VarNode);
};

class VarNode : public ExprNode {
 public:
  Id vid;
  Type type_annotation;
  static constexpr const char* _type_key = "relay.Var";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarNode, ExprNode);
};
```

 其中`VarNode::Id`是对`string`的封装，`VarNode::Type`中维护的是这个`Var`的`shape`以及`dtype`。又看了下`Type`底层，其是有一个`Type`和`TypeNode`的二元组，然后其又是基于`Span`和`SpanNode`实现的，这些不重要先不管了。

到此`Var`的创建过程就已经很清楚了。

#### 模型的创建

下一步就是定义网络中的各个层，`relay.nn.conv2d/batch_norm/relu`底层调用的`_make.conv2d/_make.batch_norm/_make.relu`，这些其实也是C++中相关函数的`PackedFunc`，此处以卷积为例，看代码

```C++
template <typename T>
inline Expr MakeConv(/* some args */) {
  auto attrs = make_object<T>();                    // 以Conv2d为例，这里获取了一个默认的Conv2DAttrs
  attrs->strides = std::move(strides);              // TVM的Runtime中将这些对象使用Object去抽象，使用同一套内存管理
  // ... 一些其他attrs的设置如padding之类的，此处略去
  const Op& op = Op::Get(op_name);                  // 根据Name获取Op的引用
  return Call(op, {data, weight}, Attrs(attrs), {});// 将op, Expr类型的输入, 参数打包成一个Call对象，返回回去
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv2d")
    .set_body_typed([](/* some args */) {
      return MakeConv<Conv2DAttrs>(/* some args */);
    });
```

其实发现底层调用的`MakeConv<Conv2DAttrs>`函数，这个函数中创建了一个`Call`对象返回回去，而看起来这个`Call`对象就是对`Op`还有其输入，其`attr`的封装，看一下其代码

```C++
class Call : public Expr {
 public:
  TVM_DLL Call(Expr op, Array<Expr> args, Attrs attrs = Attrs(), Array<Type> type_args = Array<Type>());
  TVM_DEFINE_OBJECT_REF_METHODS(Call, RelayExpr, CallNode);
};

class CallNode : public ExprNode {
 public:
  Expr op;							  // op的类型
  tvm::Array<relay::Expr> args;		  // 输入列表, 一系列Expr
  Attrs attrs;						  // attrs
  tvm::Array<Type> type_args;         // 一般没啥用
};
```

可以看到这个类结构还是很清楚的，此处不多赘述。须知这里描述一个`Call`的输入是一个`Expr`的`Array`，而`Var`以及`Call`其实都继承自`Expr`，所以一个`Call`的输入也可以是其他的`Call`。

依次内推，我们就可以得到一个模型，由模型最后一个节点可以不难的得到前面的节点。

#### IRModule的生成

`IRModule`的生成也分为了三步，首先找到模型中的`free_vars`，然后构建`Function`，最后将这个`Function`转化为`IRModule`，先看第一个。

##### FreeVars

python中的`relay.analysis.free_vars()`函数其实底层调用的是C++中的`FreeVars()`函数，该函数代码如下：

```C++
tvm::Array<Var> FreeVars(const Expr& expr) { return VarVisitor().Free(expr); }
```

其根据一个`Expr`(在此处即网络中最后一个输出，其实是个`Call`对象)，返回一个`Free Var`的`Array`。所谓的`Free Var`其实就是不局限于一个`Let`表达式或者`Function`的`Var`，与之相对的就是`Bound Var`，其就是只在一个`Expr`内部有效的`Var`。具体实现时这里构造了一个匿名的`VarVisitor`对象，调用其`Free()`函数，再看下这个类和这个`Free()`函数

```C++
class VarVisitor : protected ExprVisitor, protected PatternVisitor {
 public:
  Array<Var> Free(const Expr& expr) {
    this->VisitExpr(expr);    // 这个函数继承自ExprVisitor, 其递归解析这个Expr中各个Var到自己的两个数据成员中了
    Array<Var> ret;
    for (const auto& v : vars_.data)
      if (bound_vars_.set.count(v) == 0)    // 如果不是bound_vars_，就将之作为返回值
        ret.push_back(v);
    return ret;
  }
 private:
  InsertionSet<Var> vars_;	      // 记录这个Expr所能递归访问到的所有Var
  InsertionSet<Var> bound_vars_;  // 记录这个Expr所能递归访问到的所有bound_vars_
};
```

其中`InsertionSet`是对`set`的封装，从逻辑上看，其调用了`this->VisitExpr(expr)`函数解析传入进来的`expr`将所有的`Var`存在`vars_`中，将`bound var`存入了`bound_vars_`中。之后遍历模型中`var`，判断其若不是`bound var`，就将之作为`free var`，最后将这些`free var`组织成一个`Array<Var>`返回回去。

所以现在问题就是`VisitExpr()`如何解析传入进来的参数呢，从代码中发现，这个函数继承自`ExprVisitor`类中，这个函数代码如下

```C++
class ExprVisitor: public ::tvm::relay::ExprFunctor<void(const Expr& n)> {
 public:
  void VisitExpr(const Expr& expr) {
    auto it = visit_counter_.find(expr.get());          // 获取一下这个expr的指针
    if (it != visit_counter_.end()) {                   // 若这个expr以前访问过，就++一下访问次数
      ++it->second;
    } else {                                            // 若没有访问过，则真正的进行访问
      using TParent = ExprFunctor<void(const Expr&)>;   // 调用ExprFunctor<void(const Expr&)>::VisitExpr()来对Expr进行处理
      TParent::VisitExpr(expr);
      visit_counter_.insert({expr.get(), 1});           // 将这个Expr存到这个unordered_map中
    }
  }
 protected:
  std::unordered_map<const Object*, size_t> visit_counter_; // 应该用于记录一个Expr被Visit了几次
};
```

这个函数逻辑很简单，其首先获取一下传入进来的这个`Expr`，若已经处理过了则只是将计数器++，只有没处理过才会进一步调用其父类`ExprFunctor<void(const Expr&)>`的`VisitExpr()`来处理，这个父类是个模板类，可以用调用类型来实例化这个类，来看下其父类这个函数的实现

```C++
template <typename R, typename... Args>
class ExprFunctor<R(const Expr& n, Args...)> {
  using TSelf = ExprFunctor<R(const Expr& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;
  virtual R VisitExpr(const Expr& n, Args... args) {
    static FType vtable = InitVTable();     // 获取一个NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>类型的对象vtable
    return vtable(n, this, std::forward<Args>(args)...);  // 用这个vtable的函数调用运算符来对传入进来的Expr进行处理
  } 
}
```

这个函数中调用`InitVTable()`函数获取了一个静态的`NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>`对象，然后这个对象应该重载了函数调用运算符，然后将传入的参数转发给了这个函数调用运算符，先看下`NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>`类

```C++
template <typename R, typename... Args>
class NodeFunctor<R(const ObjectRef& n, Args...)> {
 private:
  typedef R (*FPointer)(const ObjectRef& n, Args...);
  using TSelf = NodeFunctor<R(const ObjectRef& n, Args...)>;
  std::vector<FPointer> func_;                               // 这一个NodeFunctor中有函数指针的Vector

 public:                                                               
  R operator()(const ObjectRef& n, Args... args) const {
    return (*func_[n->type_index()])(n, std::forward<Args>(args)...);
  }
};
```

这个类中维护了一个函数指针的`vector`，应该对于每一种具体的`Expr`（的`type_index_`）其都有一个专门的处理函数。上面`VisitExpr()`函数应该首先通过`InitVTable()`函数得到了一个设置好的`NodeFunctor`对象，然后使用`NodeFunctor`中相关函数来处理相关的输入的`Expr`，所以看下`InitVTable()`函数

```C++
#define RELAY_EXPR_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitExpr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

static FType InitVTable() {
    FType vtable;                               // 在这个vtable中添加对不同Expr处理的指针
    RELAY_EXPR_FUNCTOR_DISPATCH(VarNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(CallNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(OpNode);
	// ... other Node
    return vtable;
}
```

`InitVTable()`函数中使用宏`RELAY_EXPR_FUNCTOR_DISPATCH`来完成各种`Node`处理函数的注册，这个处理函数的函数体其实就是调用`ExprFunctor`的重载的各种`virtual VisitExpr_()`进行处理，而这个函数又分别在`ExprVisitor`和`VarVisitor`中实现了，这里我们看下`ExprVisitor::VisitExpr_(const CallNode* op)`

```C++
void ExprVisitor::VisitExpr_(const CallNode* op) {
  this->VisitExpr(op->op);
  for (auto ty_arg : op->type_args)
    this->VisitType(ty_arg);
  for (auto arg : op->args)
    this->VisitExpr(arg);
}

void VarVisitor::VisitExpr_(const VarNode* var) { vars_.Insert(GetRef<Var>(var)); }

void ExprVisitor::VisitExpr_(const OpNode* op) { return; }
```

可以看到，其中先处理了这个`Call`中打包的那个`op`，然后处理输入，因为输入其实是`Call`或者`Var`，是`Var`则将之存入到`vars_`中，否则的话递归的解析`Call`。到此我们知道`vars_`是如何生成的，而在此例中没有任何`bound_vars_`，所以我们也最终得到了我们所需的`free_vars`。

**看起来就是一个字，绕。这里用简短的语言概括一下，`FreeVars()`函数逐层向上递归，递归到`ExprFunctor::VisitExpr()`，这个函数中构建了一个`NodeFunctor`类，这个类中维护了访问各种`Expr`的函数指针，其根据传入的`Expr`类型调用对应的函数执行。然而`NodeFunctor`中的这些函数指针都指向`ExprFunctor`类中的各种`VisitExpr_()`，这就很迷了。随后，这些`VisitExpr_()`函数在`ExprFunctor`的子类诸如`VarVisitor`，`ExprVisitor`中又被Override了。**

##### 构建Function以及IRModule

python中使用网络中最后一个`Call`对象，以及`free_vars`构建`Function`，与前面类似，这里的`Function`也是python中对C++的这个类的封装，直接看C++中的这个类，构造过程就是一个简单的赋值过程。赋值完成后，`FunctionNode::params`中存的是之前得到的`free_vars`，`FunctionNode::body`中存的是网络的输出，在此例中是一个`Call`对象。

```C++
class Function : public BaseFunc {
 public:
};

class FunctionNode : public BaseFuncNode {
 public:
  tvm::Array<Var> params;	// 之前得到的free_vars
  Expr body;				// 网络最后一个Call对象
  Type ret_type;
  tvm::Array<TypeVar> type_params;
  static constexpr const char* _type_key = "relay.Function";
};
```

然后来看如何用这个`Function`来构造`IRModule`，程序中使用`mod = tvm.IRModule.from_expr(func)`来构造，而`from_expr()`函数是`IRModule`类的一个`staticmethod`，其代码如下

```python
@staticmethod
def from_expr(expr, functions=None, type_defs=None):
    funcs = functions if functions is not None else {}
    defs = type_defs if type_defs is not None else {}
    return _ffi_api.Module_FromExpr(expr, funcs, defs)  # 在./src/ir/module.cc:449处注册: Module_FromExpr(expr, {}, {})
```

其中调用了`Module_FromExpr()`，这个函数是C++中的静态函数`FromExpr()`函数的封装，看这个C++函数

```C++
// 构造IRModule，并建立GlobalVar -> Function的映射，存在IRModule的一个Map中
IRModule IRModule::FromExpr(const RelayExpr& expr, const tvm::Map<GlobalVar, BaseFunc>& global_funcs,        
                            const tvm::Map<GlobalTypeVar, TypeData>& type_definitions) {
  auto mod = IRModule(global_funcs, type_definitions);    // 用后两个参数构造一个IRModule, run.py中这两个参数全为空
  BaseFunc func;
  std::string gv_name = "main";
  if (auto* func_node = expr.as<BaseFuncNode>()) {  // 如果Expr中的data_已经是一个BaseFuncNode了，则返回Expr的data_指针
    func = GetRef<BaseFunc>(func_node);             // 根据BaseFuncNode类型的指针创建BaseFunc对象
    if (auto opt = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
      gv_name = opt.value();	// 如果func中已经设置好了GlobalSymbal的话，那么就用设置好的name，否则就用默认的"main"
    }
  } else {
    func = relay::Function(relay::FreeVars(expr), expr, Type(), relay::FreeTypeVars(expr, mod), {});
  }
  auto main_gv = GlobalVar(gv_name); // 设置一个默认的GlobalVar
  mod->Add(main_gv, func);           // 将这个GlobalVar和BaseFunc构造一个映射，存储在IRModule的data_所指的IRModuleNode的functions中
  return mod;
}
```

这个函数中首先根据传入的参数`global_funcs`和`type_definitions`构造一个`IRModule`，然后根据传入的`Expr`（其实就是个`Function`）创建了一个`Basefunc`对象，然后将这个`Basefunc`对象以及名为`"main"`的`GlobalVar`对象构成一个二元组然后调用`Add()`函数将之加入到`IRModule`中，而`IRModule`中重载了`->`运算符，这里实际调用的是`IRModuleNode::Add()`，这里看下相关代码

```C++
class IRModule : public ObjectRef {
 public:
  using ContainerType = IRModuleNode;
  static constexpr bool _type_is_nullable = false;
};

class IRModuleNode : public Object {
 public:
  Map<GlobalVar, BaseFunc> functions;
  Map<GlobalTypeVar, TypeData> type_definitions;
    
  void Add(const GlobalVar& var, const BaseFunc& f, bool update) {
    BaseFunc checked_func = f;
    if (auto* ptr = f.as<relay::FunctionNode>()) {        // 对传入进来的Function做TypeCheck，这里先获取FunctionNode
      checked_func = RunTypeCheck(GetRef<IRModule>(this), var, GetRef<relay::Function>(ptr)); // InferType并检测
    }
    Type type = checked_func->checked_type();
    var->checked_type_ = type;          // 把这个GlobalVar的checked_type_设置为检测结果
    AddUnchecked(var, checked_func);    // 这个GlobalVar以及InferType之后的Function构造一个映射，存在IRModuleNode::functions中
  }										// 将GlobalVar::name_hint -> GlobalVar存在IRModuleNode::global_var_map_上

 private:
  Map<String, GlobalVar> global_var_map_;
  Map<String, GlobalTypeVar> global_type_var_map_;
  std::unordered_map<int32_t, Constructor> constructor_tag_map_;
  std::unordered_set<String> import_set_;
};
```

从`Add()`函数来看，其中对传入的`BaseFunc`调用`RunTypeCheck()`函数做类型检测，只有检测出来的函数不为`IncompleteTypeNode`才可以进行后续过程。之后将检测出来的`type`设给那个`GloablVar::checked_type_`，用这个设置好的`GlobalVar`来设置`IRModuleNode::global_var_map_`，然后将映射`GlobalVar->checked_func`送给`IRModuleNode::functions`。`RunTypeCheck()`中所得到的`checked_func`相对于原始的`func`是有一点变化，具体看代码

```C++
relay::Function RunTypeCheck(const IRModule& mod, const GlobalVar& var, relay::Function f) {
  // Deduplicate之后，Function类型的f变为Expr类型，再用Downcast将回Function类型
  // Deduplicate就是为重名的Var改名，使IR可以well-formed，嗯，大概就是这样，需要确认@TODO
  auto func = Downcast<relay::Function>(relay::DeDup(std::move(f)));  
  auto fv = relay::FreeVars(func);				// 一般fv和ftv都为空
  auto ftv = relay::FreeTypeVars(func, mod);

  func = relay::Function(concat(func->params, fv), func->body, func->ret_type,  // 构造一个新的Function，正常和原来的应该一样
                         concat(func->type_params, ftv), func->attrs);
  relay::Function checked_func = InferType(func, mod, var);                     // 做InferType
  return checked_func;                                                          // 把InferType后的Function返回回去
}

Function InferType(const Function& func, const IRModule& mod, const GlobalVar& var) {
  Function func_copy = Function(make_object<FunctionNode>(*func.operator->()));
  func_copy->checked_type_ = func_copy->func_type_annotation();
  mod->AddUnchecked(var, func_copy);		// 为了满足TypeInferencer的使用要求而添加
  Expr func_ret = TypeInferencer(mod, var).Infer(func_copy);	// 真正的TypeInferencer
  mod->Remove(var);							// 用完了就删掉
  return Downcast<Function>(func_ret);
}
```

`RunTypeCheck()`主要调用了`InferType()`，而这个函数中主要是利用`TypeInferencer`类来实现，具体的`TypeInferencer`这里就先不看了@TODO。

到此用于编译的`IRModule`构建完成。

### 编译

完成了图的创建之后下一个任务就是编译，编译时首先做了一些准备操作，包括`Target`和`PassContext`的设置，然后进行了真正的`Build`。

```C++
class TargetNode : public Object {
 public:
  TargetKind kind;
  String tag;
  Array<String> keys;
  Map<String, ObjectRef> attrs;
  static constexpr const char* _type_key = "Target";
    
 private:
  mutable std::string str_repr_;
  friend class Target;
};
```

#### 准备过程

##### Target的设置

python中的`tvm.target.create("llvm") `其实是调用了C++的`Target::Create()`函数，这个函数解析了传入的参数，然后调用了`Target::CreateTarget()`完成真正的创建，最后将创建好的`target`返回回去，看下这个函数的代码

```c++
Target Target::CreateTarget(const std::string& name, const std::vector<std::string>& options) {
    TargetKind kind = TargetKind::Get(name);					// 根据name(比如"llvm")来Get一个注册好的TargetKind
    ObjectPtr<TargetNode> target = make_object<TargetNode>();	// 创建一个空的target node
    target->kind = kind;
    target->tag = "";
    target->attrs = kind->ParseAttrsFromRaw(options);			// options一般为空
    std::vector<String> keys;
	// some code to set up keys
    target->keys = std::move(keys);
    return Target(target);
}
```

看完了之后发现平平无奇，就是根据输入设置了`TargetNode`的一些属性，然后用`TargetNode`构造了一个`Target`，最后将这个设置好的`Target`返回给python。需要注意的是，TVM中还有个`TargetKind`类，其中`Target`类表示此次编译的具体的Target，而`TargetKind`用于表示一个具体的硬件`Target`，其会被注册在系统中。

##### PassContext的设置

python中的那个`with`语句其实做了三件事，创建python中的`PassContext`，然后调用其`__enter__()`函数，在build之后又调用`__exit__()`中，这几个函数都是`PackedFunc`，其实换成C++是这样的

```C++
PassContext pctx = PassContext::Create(/* args */);	// 根据传入的参数创建一个PassContext对象
pctx.EnterWithScope();
// Build Code
pctx.ExitWithScope();
```

其中第一行是根据传入的参数创建一个`PassContext`对象，这个比较简单此处跳过。再看第二行，这个函数的内容如下：

```C++
struct PassContextThreadLocalEntry {
  PassContext default_context;
  std::stack<PassContext> context_stack;
};

typedef dmlc::ThreadLocalStore<PassContextThreadLocalEntry> RelayPassContextThreadLocalStore;

void PassContext::EnterWithScope() {    
  PassContextThreadLocalEntry* entry = RelayPassContextThreadLocalStore::Get();
  entry->context_stack.push(*this);     
}							// 自己这个PassContext只设置了opt_level
```

系统中维护一个`thread_local`的`static`的`PassContextThreadLocalEntry`指针，这个指针可以通过`RelayPassContextThreadLocalStore::Get()`函数获得，`EnterWithScope()`将上面创建的`PassContext`对象加入`PassContextThreadLocalEntry`的`context_stack`中，而另一个`ExitWithScope()`函数其实就是就是相反的出栈过程，此处就不多细说了。

总结起来一句话，系统中有一个`static`的`PassContextThreadLocalEntry`对象通过栈维护这个`PassContext`对象。

#### Build

`build`首先调用`_update_target`，然后设置autotvm相关的`tophub_context`，然后构建了一个`BuildModule()`，



```C++
class TVM_DLL ModuleNode : public Object {
 public:
 protected:
  std::vector<Module> imports_;
 private:
  std::unordered_map<std::string, std::shared_ptr<PackedFunc> > import_cache_;
};

class RelayBuildModule : public runtime::ModuleNode {
 public:
 protected:
  std::unique_ptr<GraphCodegen> graph_codegen_;
  TargetsMap targets_;										 	// using TargetsMap = Map<tvm::Integer, tvm::Target>;
  tvm::Target target_host_;
  std::unordered_map<std::string, runtime::NDArray> params_;    // 维护网络中的权值
  BuildOutput ret_;                                             // 
};
```



