### Object, ObjectPtr, ObjectRef

TVM在`include/tvm/runtime/object.h`中定义了三个类，`Object`，`ObjectPtr`，`ObjectRef`，其runtime中各种主要的类都是继承自这三个类，所以此处关注一下这三个类，先看下这三个类是啥，然后再看其他类都是怎么继承的，首先看Object

```C++
class Object {
 public:
  static constexpr const char* _type_key = "runtime.Object";

  static constexpr bool _type_final = false;
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_child_slots_can_overflow = true;

  static constexpr bool _type_has_method_visit_attrs = true;
  static constexpr bool _type_has_method_sequal_reduce = false;
  static constexpr bool _type_has_method_shash_reduce = false;
  static constexpr uint32_t _type_index = TypeIndex::kDynamic;
 protected:
  uint32_t type_index_{0};
  RefCounterType ref_counter_{0};	// using RefCounterType = std::atomic<int32_t>;
  FDeleter deleter_ = nullptr;		// typedef void (*FDeleter)(Object* self);
 private:
};
```

其是一些比较抽象的类，其中给的都是一些关于`Object`属性的描述及其默认值，并没有什么信息，而`ObjectPtr`看了之后其实就是一个自己实现的`shared_ptr`，这个也不多细说了，然后再看下`ObjectRef`

```C++
class ObjectRef {
 public:
  using ContainerType = Object;
  static constexpr bool _type_is_nullable = true;
 protected:
  ObjectPtr<Object> data_;
};
```

可以看到其是对`Object`的封装，这样系统中传指针的开销会小很多。

### 实际的应用

现在以`IRModule`类和`IRModuleNode`类来说明这三者的应用，先看一下这两个类的代码

```C++
class IRModule : public ObjectRef {
 public:
  using ContainerType = IRModuleNode;
  static constexpr bool _type_is_nullable = false;

  TVM_DEFINE_OBJECT_REF_COW_METHOD(IRModuleNode);
};

class IRModuleNode : public Object {
 public:
  Map<GlobalVar, BaseFunc> functions;
  Map<GlobalTypeVar, TypeData> type_definitions;
  static constexpr const char* _type_key = "IRModule";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(IRModuleNode, Object);
 private:
  Map<String, GlobalVar> global_var_map_;
  Map<String, GlobalTypeVar> global_type_var_map_;
  std::unordered_map<int32_t, Constructor> constructor_tag_map_;
  std::unordered_set<String> import_set_;
};
```

可以看到，`IRModule`中并没有定义什么其他数据成员了，其数据成员还是继承自基类的`ObjectRef::ObjectPtr<Object> data_`，而`IRModuleNode`继承自`Object`，其中定义了一些真正有用的数据成员，同时也自己设置了一些与`Object`中同名的`static constexpr`成员覆盖了基类的定义的默认值，`IRModule`则是对其的封装。

### 一些常用宏/函数

可以看到上面两个`IRModule/IRModuleNode`中都有一些宏，其实在`include/tvm/runtime/object.h`中还定义了很多辅助函数和宏，现在列举一些主要的，下面继承自`ObjectRef`的类均称之为`Ref`类，继承自`Object`的类均称之为`Node`类

#### TVM_DEFINE_OBJECT_REF_METHODS

用于`Ref`类中，其帮之定义了默认构造函数，一个只有一个参数（这个参数其实就是其对应`node`类的指针）的构造函数，然后是拷贝构造函数及赋值运算符重载，之后是定义了获取自己维护的那个指针的两个函数，最后设置了`ContainerType`为自己对应的`node`类

```C++
TVM_DEFINE_OBJECT_REF_METHODS(Var, RelayExpr, VarNode);

#define TVM_DEFINE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName)                        \
  // 无参以及有参构造函数
  TypeName() = default; 	                                                                   \
  explicit TypeName(::tvm::runtime::ObjectPtr<::tvm::runtime::Object> n) : ParentType(n) {}    \
  // 定义了拷贝构造及赋值运算符(左/右引用版本)
  TVM_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TypeName);                                           \
  // 定义了两个函数用于获取自己维护的那个node的指针
  const ObjectName* operator->() const { return static_cast<const ObjectName*>(data_.get()); } \
  const ObjectName* get() const { return operator->(); }                                       \
  // 重新设置了自己的那个ContainerType
  using ContainerType = ObjectName;
```

#### TVM_DECLARE_BASE_OBJECT_INFO

用于`Node`类中，不要被名字搞混了，其实就是为类定义了两个函数`RuntimeTypeIndex()`和`_GetOrAllocRuntimeTypeIndex()`，这两个函数之后用到时候再说吧

```C++
TVM_DECLARE_BASE_OBJECT_INFO(VarNode, ExprNode);
#define TVM_DECLARE_BASE_OBJECT_INFO(TypeName, ParentType)                                     \
  static_assert(!ParentType::_type_final, "ParentObj maked as final");                         \
  static uint32_t RuntimeTypeIndex() {                                                         \
    static_assert(TypeName::_type_child_slots == 0 || ParentType::_type_child_slots == 0 ||    \
                      TypeName::_type_child_slots < ParentType::_type_child_slots,             \
                  "Need to set _type_child_slots when parent specifies it.");                  \
    if (TypeName::_type_index != ::tvm::runtime::TypeIndex::kDynamic) {                        \
      return TypeName::_type_index;                                                            \
    }                                                                                          \
    return _GetOrAllocRuntimeTypeIndex();                                                      \
  }                                                                                            \
  static uint32_t _GetOrAllocRuntimeTypeIndex() {                                              \
    static uint32_t tidx = Object::GetOrAllocRuntimeTypeIndex(                                 \
        TypeName::_type_key, TypeName::_type_index, ParentType::_GetOrAllocRuntimeTypeIndex(), \
        TypeName::_type_child_slots, TypeName::_type_child_slots_can_overflow);                \
    return tidx;                                                                               \
  }
```

#### TVM_DECLARE_FINAL_OBJECT_INFO

用于在一个需要被设为`final`的`Node`类中，通过设置相关标志位及其子类数量来达到设为`final`的效果，然后也为之设置了两个函数`RuntimeTypeIndex()`和`_GetOrAllocRuntimeTypeIndex()`

```C++
TVM_DECLARE_FINAL_OBJECT_INFO(VarNode, ExprNode);
#define TVM_DECLARE_FINAL_OBJECT_INFO(TypeName, ParentType) \
  static const constexpr bool _type_final = true;           \
  static const constexpr int _type_child_slots = 0;         \
  TVM_DECLARE_BASE_OBJECT_INFO(TypeName, ParentType)
```

#### TVM_DEFINE_OBJECT_REF_COW_METHOD

用在`Ref`类中，为这个类设置了`CopyOnWrite()`函数，具体这个函数干啥的之后用到再细看吧

```C++
TVM_DEFINE_OBJECT_REF_COW_METHOD(FunctionNode);
#define TVM_DEFINE_OBJECT_REF_COW_METHOD(ObjectName)     \
  ObjectName* CopyOnWrite() {                            \
    if (!data_.unique()) {                               \
      auto n = make_object<ObjectName>(*(operator->())); \
      ObjectPtr<Object>(std::move(n)).swap(data_);       \
    }                                                    \
    return static_cast<ObjectName*>(data_.get());        \
  }
```

#### IsInstance()

用于`Node`类，如果这个`Node`是`TargetType`，则返回真否则返回假。

```C++
template <typename TargetType>
inline bool Object::IsInstance() const {
  const Object* self = this;
  if (self != nullptr) {
    if (std::is_same<TargetType, Object>::value) 
      return true;
      
    if (TargetType::_type_final) {
      return self->type_index_ == TargetType::RuntimeTypeIndex();
    } else {
      uint32_t begin = TargetType::RuntimeTypeIndex();
      if (TargetType::_type_child_slots != 0) {
        uint32_t end = begin + TargetType::_type_child_slots;
        if (self->type_index_ >= begin && self->type_index_ < end) return true;
      } else {
        if (self->type_index_ == begin) return true;
      }
      if (!TargetType::_type_child_slots_can_overflow) return false;
      if (self->type_index_ < TargetType::RuntimeTypeIndex()) return false;
      return self->DerivedFrom(TargetType::RuntimeTypeIndex());
    }
  } else {
    return false;
  }
}
```

#### as()

用于`Ref`类，判断这个`Ref`类中的`data_`成员是否是`ObjectType`类，若是则返回`ObjectType`指针，否则返回`nullptr`。

```C++
template <typename ObjectType>
inline const ObjectType* ObjectRef::as() const {    // 若是目标类型，则返回data_，也即那个Node成员
  if (data_ != nullptr && data_->IsInstance<ObjectType>()) {
    return static_cast<ObjectType*>(data_.get());
  } else {
    return nullptr;
  }
}
```

#### GetRef()

用于`Node`类，构造一个`Ref`类封装这个`Node`类，然后返回这个`Ref`类

```C++
template <typename RefType, typename ObjType>
inline RefType GetRef(const ObjType* ptr) {     // 根据一个*Node，返回其Ref封装
  return RefType(ObjectPtr<Object>(const_cast<Object*>(static_cast<const Object*>(ptr))));
}
```

#### DownCast()

用于`Ref`类，用一个父类`Ref`对象，通过其`Node`构造一个子类`Ref`类对象并返回

```C++
template <typename SubRef, typename BaseRef>
inline SubRef Downcast(BaseRef ref) {       // 用一个父类Ref对象，通过其*Node构造一个子类Ref类对象
  return SubRef(std::move(ref.data_));
}

```

#### CopyOnWrite()

用在`Ref`类中，若有多个`Ref`对象指向同一个`Node`资源，当这个`Ref`需要修改的时候，这个`Ref`中自动复制一份新的`Node`，然后基于这个新的`Node`修改

```C++
#define TVM_DEFINE_OBJECT_REF_COW_METHOD(ObjectName)     \
  ObjectName* CopyOnWrite() {                            \
    CHECK(data_ != nullptr);                             \
    if (!data_.unique()) {                               \
      auto n = make_object<ObjectName>(*(operator->())); \
      ObjectPtr<Object>(std::move(n)).swap(data_);       \
    }                                                    \
    return static_cast<ObjectName*>(data_.get());        \
  }

MyCOWObjectRef ref, ref2; 
ref2 = ref;								// ref和ref2指向同一个资源
ref.CopyOnWrite()->value = new_value;	// ref需要修改，此时会copy一个新对象，ref指向新对象
assert(ref2->value == old_value);		// ref2仍然指向原来的老对象，然后再对新对象中内容做修改
assert(ref->value == new_value);
```

