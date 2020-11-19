### TVM中的注册器

TVM组件中很多东西都是以注册的方式添加到系统中的，这种设计理念为系统带来了非常良好的可拓展性，这里介绍一个TVM中的注册机制。

TVM中注册主要是基于一个叫`AttrRegistry`类实现的，其代码如下：

```C++
template <typename EntryType, typename KeyType>
class AttrRegistry {
 public:
 private:
  std::mutex mutex_;
  std::vector<std::unique_ptr<EntryType>> entries_;
  std::unordered_map<String, EntryType*> entry_map_;
  std::unordered_map<String, std::unique_ptr<AttrRegistryMapContainerMap<KeyType>>> attrs_;
};
```

然后TVM中会把正常的类封装成一个`RegEntry`类，比如`Op`类会对应一个`OpRegEntry`类，`TargetKind`类会对应`TargetKindRegEntry`类，然后以`RegEntry`类及其原始类实例化这个类模板

```C++
class OpRegEntry {
 private:
  std::string name;
  Op op_;
};

using OpRegistry = AttrRegistry<OpRegEntry, Op>;
```

来看`AttrRegistry`的几个数据成员，其中`mutex_`是用于多线程的不管，`entries_`其实就是个`vector`，而`entry_map_`中存的是`name`到`EntryType`的映射。注册的时候程序会更新`entries_`和`entry_map_`。最后一个`attrs_`为每个注册对象维护其相关属性，具体使用`AttrRegistryMapContainerMap<KeyType>`类，这个类代码先列在这里：

```C++
template <typename KeyType>
class AttrRegistryMapContainerMap {
 public:
 private:
  String attr_name_;
  std::vector<std::pair<runtime::TVMRetValue, int>> data_;
};
```

在看下`AttrRegistry`中的相关函数，这里只列出核心代码，具体函数的作用看注释

```C++
// 根据name查找对应EntryType的指针
const EntryType* Get(const String& name) const {
    auto it = entry_map_.find(name);
    if (it != entry_map_.end()) return it->second;
    return nullptr;
}

// 输入name，若该name已经注册就返回对应EntryType的指针，否则就注册在系统中再返回该指针
EntryType& RegisterOrGet(const String& name) {
    auto it = entry_map_.find(name);
    if (it != entry_map_.end()) return *it->second;		// 若有则直接注册
    uint32_t registry_index = static_cast<uint32_t>(entries_.size());
    auto entry = std::unique_ptr<EntryType>(new EntryType(registry_index));	// 构造一个新的EntryType
    auto* eptr = entry.get();							// 获取这个EntryType的指针
    eptr->name = name;
    entry_map_[name] = eptr;							// 将Pair<String, EntryType*>存入entry_map_
    entries_.emplace_back(std::move(entry));			// 将EntryType*存入entries_
    return *eptr;										// 返回这个EntryType
}

Array<String> ListAllNames() const {
    Array<String> names;
    for (const auto& kv : entry_map_) {
        names.push_back(kv.first);
    }
    return names;
}

void UpdateAttr(const String& attr_name, const KeyType& key, runtime::TVMRetValue value, int plevel) {
    auto& op_map = attrs_[attr_name];
    if (op_map == nullptr) {
        op_map.reset(new AttrRegistryMapContainerMap<KeyType>());
        op_map->attr_name_ = attr_name;
    }

    uint32_t index = key->AttrRegistryIndex();
    if (op_map->data_.size() <= index) {
        op_map->data_.resize(index + 1, std::make_pair(TVMRetValue(), 0));
    }
    std::pair<TVMRetValue, int>& p = op_map->data_[index];
    if (p.second < plevel && value.type_code() != kTVMNullptr) {
        op_map->data_[index] = std::make_pair(value, plevel);
    }
}

void ResetAttr(const String& attr_name, const KeyType& key) {
    auto& op_map = attrs_[attr_name];
    uint32_t index = key->AttrRegistryIndex();
    if (op_map->data_.size() > index) {
        op_map->data_[index] = std::make_pair(TVMRetValue(), 0);
    }
}

const AttrRegistryMapContainerMap<KeyType>& GetAttrMap(const String& attr_name) {
    auto it = attrs_.find(attr_name);
    return *it->second.get();
}

bool HasAttrMap(const String& attr_name) {
    return attrs_.count(attr_name);
}

static TSelf* Global() {
    static TSelf* inst = new TSelf();
    return inst;
}
```

### Op的注册过程

以`conv2d`为例说明一个算子注册的过程，TVM中该算子的注册在`src/relay/op/nn/convolution.cc`中，其主要代码如下：

```C++
// First thing
TVM_REGISTER_NODE_TYPE(Conv2DAttrs);

// Second thing
TVM_REGISTER_GLOBAL("relay.op.nn._make.conv2d")	// 那个python函数调用到最后其实就是这个
    .set_body_typed([](Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                       Array<IndexExpr> dilation, int groups, IndexExpr channels,
                       Array<IndexExpr> kernel_size, String data_layout, String kernel_layout,
                       String out_layout, DataType out_dtype) {
      return MakeConv<Conv2DAttrs>(data, weight, strides, padding, dilation, groups, channels,
                                   kernel_size, data_layout, kernel_layout, out_layout, out_dtype,
                                   "nn.conv2d");
    });

// Third thing
RELAY_REGISTER_OP("nn.conv2d")
```

Op的注册过程主要用如上代码来实现，可以看出这里用三个宏做了三件事，看起来大概是首先注册了一个`Conv2DAttrs`的类，这个类维护`conv2d`的一些参数信息。之后注册了一个`PackedFunc`叫做`relay.op.nn._make.conv2d`，这个`PackedFunc`应该是将创建算子实例的接口暴露给python，由于这里是注册一个函数而没有真正的去执行它，所以其先于第三点这个算子的真正注册也是OK的。第三件事应该就是算子实际的注册。

#### 第一件事

先看下第一件事，`Conv2DAttrs`的注册，这个宏层层展开后真正的代码是

```C++
static __attribute__((unused)) uint32_t __make_Object_tid##__COUNTER__ = Conv2DAttrs::_GetOrAllocRuntimeTypeIndex();
static __attribute__((unused)) ::tvm::ReflectionVTable::Registry __make_reflectiion##__COUNTER__ = 
    ::tvm::ReflectionVTable::Global()->Register<Conv2DAttrs, ::tvm::detail::ReflectionTrait<Conv2DAttrs>>()
        .set_creator([](const std::string&) -> ObjectPtr<Object> {
            return ::tvm::runtime::make_object<Conv2DAttrs>();
        });
```

大致看起来不外乎就是一个注册的过程，不过这里似乎是实现了一个反射的机制来维护注册的结果，这里先不细看了，后续需要时再补充 **---{TODO}---**。

再看一下`Conv2dAttrs`的数据结构

```C++
struct Conv2DAttrs : public tvm::AttrsNode<Conv2DAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  // TVM_ATTR_FIELD这个宏就需要上面的这个flection机制支持
  TVM_DECLARE_ATTRS(Conv2DAttrs, "relay.attrs.Conv2DAttrs") {
    TVM_ATTR_FIELD(strides)					
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    // ...
  }
};
```

这个数据结构继承自`AttrsNode<Conv2DAttrs>`，粗略看了下其中使用了奇异模板递归模式，其中的用来设置Field的`TVM_DECLARE_ATTRS`以及`TVM_ATTR_FIELD`宏就需要上面的反射机制支持，这些不重要也先不管了，之后有时间再细看吧**---{TODO}---**。所以到此完成了`OpAttrs`的注册。

#### 第三件事

然后第二件事是个`PackedFunc`的注册，此处先看第三个宏OP的注册，`RELAY_REGISTER_OP("nn.conv2d")`展开后如下

```C++
static __attribute__((unused)) ::tvm::OpRegEntry& __make_##Op##__COUNTER__ = 
    ::tvm::OpRegEntry::RegisterOrGet("nn.conv2d").set_name()
```

显然这里用`OpRegEntry::RegisterOrGet()`函数注册算子，`OpRegEntry`其实是将`Op`封装成一个可注册的一个类

```C++
class OpRegEntry {
 public:
  TVM_DLL static OpRegEntry& RegisterOrGet(const String& name) {
  	return OpRegistry::Global()->RegisterOrGet(name);
  }
 private:
  template <typename, typename> friend class AttrRegistry;
  std::string name;
  Op op_;
};
```

显而易见，`OpRegEntry`其实就是一个`name`和`Op`的二元组，其被用来维护一个op。系统中还有个`OpRegistry`，其是`AttrRegistry<OpRegEntry, Op>`的类型别名，然后调用`AttrRegistry<OpRegEntry, Op>::RegisterOrGet()`完成算子的注册。

现在再看一下`Op`的数据结构，其中并没有定义任何数据成员，但查看其继承树发现其继承自`ObjectRef`类，显然这里就是上面所说的TVM中对对象的描述，这里不再赘述，只需要知道其维护的`data_`实质上是一个`OpNode`的类即可，而`OpNode`中是真正维护`op`信息的地方，这里列出这三个类的代码

```C++
class Op: public RelayExpr {
  // some functions
};

class ObjectRef {
 protected:
  ObjectPtr<Object> data_;
};

class OpNode : public RelayExprNode {
 public:
  String name;
  mutable FuncType op_type;
  String description;
  Array<AttrFieldInfo> arguments;	// 维护add_argument("data", "Tensor", "The input tensor.")中的信息
  String attrs_type_key;			// Conv2dAttrs::_type_key
  uint32_t attrs_type_index{0};		// Conv2dAttrs::RuntimeTypeIndex()
  int32_t num_inputs = -1;
  int32_t support_level = 10;					
  static constexpr const char* _type_key = "Op";
 private:
  uint32_t index_{0};
  mutable int is_primitive_{-1};
};
```

这个时候发现这个注册过程中竟然没有设置`OpNode`这个类，且往下看，`RegisterOrGet()`注册后会返回这个`OpRegEntry`的引用，然后依次调用了`OpRegEntry`的一系列函数，完成了`OpNode`最终的设置。

```C++
RELAY_REGISTER_OP("nn.conv2d")
	.describe(R"code(Some Descriptions)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv2DAttrs>()					// 根据Conv2dAttrs的type进行设置
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(2)
    .add_type_rel("Conv2D", Conv2DRel<Conv2DAttrs>)	// 用来设置OpNode::op_type
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<Conv2DAttrs>);
```

需要注意的是，这里并没有设置Op的compute，schedule等等，这些后续再看吧，另外之后重点关注`set_attr`这个函数**---{TODO}---**。

#### 第二件事

这里就是个平平无奇的注册`PackedFunc`的过程，其函数体就是将传入进来的参数打包

```C++
return MakeConv<Conv2DAttrs>(data, weight, strides, padding, dilation, groups, channels, kernel_size,
                             data_layout, kernel_layout, out_layout, out_dtype, "nn.conv2d");
```

再看看这个`MakeConv`的函数

```C++
template <typename T>
inline Expr MakeConv(Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                     Array<IndexExpr> dilation, int groups, IndexExpr channels,
                     Array<IndexExpr> kernel_size, std::string data_layout,
                     std::string kernel_layout, std::string out_layout, DataType out_dtype,
                     std::string op_name) {
  auto attrs = make_object<T>();                    // 以Conv2d为例，这里获取了一个默认的Conv2DAttrs
  attrs->strides = std::move(strides);              // TVM的Runtime中将这些对象使用Object去抽象，使用同一套内存管理
  attrs->padding = std::move(padding);              // 跳进去看一下
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get(op_name);                  // 根据Name获取Op的引用
  return Call(op, {data, weight}, Attrs(attrs), {});// 将op, 输入, 参数打包成一个Call对象，返回回去
}
```

可以看到其核心就是将`Op`还有传入进来的参数打包成一个`Call`对象，并返回回去，所以我们python中调用的什么`relay.nn.conv2d()`返回的其实就是这玩意，再瞅一下其的代码

```C++
class Call : public Expr {
 public:
  TVM_DLL Call(Expr op, Array<Expr> args, Attrs attrs = Attrs(),
               Array<Type> type_args = Array<Type>());
  TVM_DEFINE_OBJECT_REF_METHODS(Call, RelayExpr, CallNode);
};
```

也没什么特殊的，到此就完成了Op的注册啦。

### Target的注册

`Target`注册是基于`TVM_REGISTER_TARGET_KIND()`来实现的，以`llvm`为例这个宏展开后代码如下：

```C++
TVM_REGISTER_TARGET_KIND("llvm") ==>
static __attribute__((unused)) ::tvm::TargetKindRegEntry& __make_##TargetKind##__COUNTER__ = \
    ::tvm::TargetKindRegEntry::RegisterOrGet("llvm").set_name()
```

系统中用`TargetKind`类描述我们自定义的各种`target`，其是`TargetKindNode`类的`Ref`类，而`TargetKindRegEntry`类则是对`TargetKind`类的封装而没有什么有意义的数据成员。与`Op`的注册类似，`TargetKindRegEntry::RegisterOrGet()`中也是调用了`AttrRegistry::RegisterOrGet()`完成注册，并返回了`TargetKindRegEntry`的引用。这部分没什么好说的了，这里我们细看一下`TargetKindNode`的代码

```C++
class TargetKindNode : public Object {
 public:
  String name;					// "llvm", "cuda", "opencl", ...
  int device_type;				// kDLCPU, kDLGPU, kDLOpenCL, ...
  Array<String> default_keys;   // {"cpu"}, {"cuda", "gpu"}, {"rocm", "gpu"}
 private:
  std::unordered_map<String, ValueTypeInfo> key2vtype_;	// add_attr_option()函数维护
  std::unordered_map<String, ObjectRef> key2default_;
  uint32_t index_;
};
```

需要注意的是这个`TargetKind`和`TargetKindNode`中维护的是系统中注册的`target`，此外系统中还有个`Target`和`TargetNode`，这个类维护用户在编译时具体选择了什么`Target`

```C++
class TargetNode : public Object {
 public:
  TargetKind kind;
  String tag;						// 一般均为空字符串
  Array<String> keys;
  Map<String, ObjectRef> attrs;
  static constexpr const char* _type_key = "Target";

 private:
  mutable std::string str_repr_;
};

```



