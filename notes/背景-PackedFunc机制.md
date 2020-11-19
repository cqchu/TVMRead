### 基础数据结构

`PackedFunc`是TVM中将C++函数接口暴露给Python的一种机制，这里先看其中C++的部分。在C++代码中，有一个`PackedFunc`类，其是TVM中对函数的一个封装，关注一下它的代码，此处仅列出其核心成员：

```C++
class PackedFunc {
 public:
  using FType = std::function<void(TVMArgs args, TVMRetValue* rv)>; // 一个C++11的调用类型，参数为TVMArgs和TVMRetValue*类型的对象
  template <typename... Args>
  inline TVMRetValue operator()(Args&&... args) const;          	// 这里应该是声明了两个调用PackedFunc的接口
  inline void CallPacked(TVMArgs args, TVMRetValue* rv) const;
 private:
  FType body_;    // PackedFunc其实就是上面定义的这个调用类型FType的实例
};
```

可以看出，其最核心的就是一个`std::function`的调用类型对象，这个调用类型对象可以指向一个函数指针，Lambda，函数名等等。而其返回值为`void`，参数类型为`TVMArgs`和`TVMRetValue*`，然后提供了两个接口函数用于调用。从名字上就可以猜出，这个`FType body_`中应该对实际函数做了封装，`TVMArgs`中封装了实际函数的参数，`TVMRetValue*`中封装了实际参数的返回值。为了能更清楚这个过程，现在再看一下`TVMArgs`和`TVMRetValue`的实现：

```C++
class TVMArgs {
 public:
  const TVMValue* values;   // TVMArgs中可以包含若干个TVMValue
  const int* type_codes;    // 从python中代码来看，这里应该代表传进来TVMValue的种类，比如int, float, NDArray, context, packedfunc等
  int num_args;             // TVMArgs中包含的参数个数
};

class TVMRetValue : public TVMPODValue_ {     // 这个类中定义了超多的赋值运算符
  // some functions
};

class TVMPODValue_ {
 private:
  TVMValue value_;			// 只有两个数据成员，TVMValue和type_code，这样看来其实和TVMArgs差不多了，只不过这里只有一个TVMValue而已
  int type_code_;          	// 看一下PackedFunc(TVMArgs, TVMRetVal*)，返回值本身就是个指针了
};                         	// 所以这里确实也没必要用个TVMValue *

```

可以看到这两个类最核心的部分都是`TVMValue`类型的对象/指针，其中`TVMArgs`可以包含多个`TVMValue`而`TVMRetValue`中而只有一个`TVMValue`，不过考虑到`PackedFunc(TVMArgs, TVMRetVal*)`中后者是一个指针，所以知道`PackedFunc`也是可以返回多个返回值的。

而更进一步的再看TVM系统中对`PackedFunc`的封装。在代码中一个`PackedFunc`和一个`name`会封装成一个`Registry`，其核心成员如下：

```C++
class Registry {
  TVM_DLL Registry& set_body(PackedFunc f);
  template <typename FLambda>
  Registry& set_body_typed(FLambda f) {
    using FType = typename detail::function_signature<FLambda>::FType;
    return set_body(TypedPackedFunc<FType>(std::move(f)).packed());
  }
  TVM_DLL static Registry& Register(const std::string& name, bool override = false);  
  TVM_DLL static bool Remove(const std::string& name);
  TVM_DLL static std::vector<std::string> ListNames();
 protected:             
  std::string name_;
  PackedFunc func_;     // 再看下这个PackedFunc的定义，其中就是用了C++11的function
  friend struct Manager;
};
```

可以看到其中有个叫`Manager`的友元类，从名字来看就知道其应该是管理`Registry`的，`Registry`中的几个静态函数都是基于这个类完成功能，其数据结构如下：

```C++
struct Registry::Manager {
  std::unordered_map<std::string, Registry*> fmap;
  std::mutex mutex;
  static Manager* Global() {
    static Manager* inst = new Manager();
    return inst;
  }
};
```

其中有两个数据成员，最主要的就是那个`name -> Registry`的`unordered_map`，然后真正再用的时候，系统中通过一个静态函数`Global()`构建了一个静态全局的`Manager`指针`inst`，然后用这个`inst`来管理这些注册的`Registry`。

### 函数注册过程

已知TVM中有个`Var`类，用于描述系统中的variable，现在看一下其构造函数`Var::Var(String name_hint, Type type_annotation)`如何注册成一个`PackedFunc/Registry`，程序中其注册代码如下：

```C++
TVM_REGISTER_GLOBAL("relay.ir.Var").set_body_typed(
    [](String str, Type type_annotation) {return Var(str, type_annotation);}
);
```

其中`TVM_REGISTER_GLOBAL`宏展开后变为：

```C++
static __attribute__((unused)) ::tvm::runtime::Registry& __mk_TVM__COUNTER__
    = ::tvm::runtime::Registry::Register(OpName)
```

可以看到，这里将`OpName`也即此处的`relay.ir.Var`送给`Registry`的静态函数`Register()`，这个函数其实就是在`Manager`类中先检查一下这个`name`是否已注册，若没注册就`new`一个新的`Registry`，并在`Manager::fmap`中构建`name`到这个`Registry`的映射，最后返回了这个`Registry`的对象，之后调用这个对象的`set_body_typed`函数，完成函数体的设置。

### 将PackedFunc暴露给Python

现在的问题就是python中如何调用这些实现在C++中的函数，可以看到在TVM的python中有很多个`_ffi_api.py`，而代码中经常会出现`_ffi_api.C++Funcname`之类的代码，所以这个暴露过程肯定和`_ffi_api`有关，现在以`_ffi_api.Var()`函数和`python/tvm/relay/_ffi_api.py`文件为描述这个过程，先看下这个`.py`文件的内容：

```python
"""FFI APIs for Relay program IR."""
import tvm._ffi
tvm._ffi._init_api("relay.ir", __name__) 
```

可以看到其中并没有显式的`Var()`函数，那么他必然通过这个`_init_api()`函数获取了这个函数，再追进去看`_init_api("relay.ir", "tvm.relay._ffi_api")`这个函数发现其内部又调用了`_init_api_prefix()`函数，看看其内容：

```python
def _init_api_prefix(module_name, prefix): 			# _init_api_prefix("tvm.relay._ffi_api", "relay.ir")
    module = sys.modules[module_name]               # 先获取tvm/relay/_ffi_api.py的handle
    for name in list_global_func_names():           # 获取了C++中注册的所有PackedFunc函数，看一下怎么获得的
        											# 调用TVMFuncListGlobalNames(int* out_size, const char*** out_array)实现的
        if not name.startswith(prefix):
            continue
        fname = name[len(prefix)+1:]                # 获取这个Registry的去除prefix后的名字，relay.ir.Var -> Var
        target_module = module                      # tvm/relay/_ffi_api.py的handle

        if fname.find(".") != -1:
            continue
        f = get_global_func(name)                   # 根据"Var"这个名字获取对应函数handle，然后封装成python PackedFunc对象的指针
        ff = _get_api(f)                            # 把这个PackedFunc设置为global的
        ff.__name__ = fname
        ff.__doc__ = ("TVM PackedFunc %s. " % fname)
        setattr(target_module, ff.__name__, ff)     # tvm/relay/_ffi_api中设置tvm.runtime.packed_func.PackedFunc对象的实例
```

其获取了C++中的`PackedFunc`函数，并将之`setattr`成`_ffi_api`的函数，其中比较关键的是`get_global_func()`函数，其中又封装了`_get_global_func()`，看其内容：

```python
def _get_global_func(name, allow_missing=False):     # _get_global_func("Var")
    handle = PackedFuncHandle()                      # 这里定义了一个默认的PackedFuncHandle, 这个handle其实就是一个void *
    check_call(_LIB.TVMFuncGetGlobal(c_str(name),    # 根据传入的"Var"找到其对应函数的handle，并赋值给上面定义的这个handle
                                     ctypes.byref(handle)))  # 再跳进去看这个C++函数 - /src/runtime/registry.cc:127  
                                                                            
    if handle.value:								 # 用这个c++的PackedFunc类型指针构建最终的函数handle，跳进去看一下
        return _make_packed_func(handle, False)      # 看完之后发现，其就是用c++的handle构造了一个python中PackedFunc的对象
                                                    
    if allow_missing:
        return None
    raise ValueError("Cannot find global function %s" % name)
```

可以看到其中调用`C++`函数`TVMFuncGetGlobal()`根据`name`获取对应的那个`PackedFunc`对象的指针，然后将这个指针作为`handle`返回给python，而python中调用了`_make_packed_func()`将之封装，看其代码：

```python
def _make_packed_func(handle, is_global):  # 这里新建了一个_CLASS_PACKED_FUNC的对象，然后把C++得到的handle赋值给了这个对象的handle成员
    """Make a packed function class"""     # 看一下这个_CLASS_PACKED_FUNC到底是个什么类
    obj = _CLASS_PACKED_FUNC.__new__(_CLASS_PACKED_FUNC) # 追根溯源之后，我们发现这个类其实是在tvm/runtime/packed_func.py中最后一行
    obj.is_global = is_global                            # 被设为了PackedFunc类，过去看看
    obj.handle = handle
    return obj
```

这里就是将C++的`PackedFunc`封装成python中的`PackedFunc`，python中这个类如下：

```python
class PackedFuncBase(object):       	# python中对packedfunc对象的抽象
    """Function base."""
    __slots__ = ["handle", "is_global"]
    def __init__(self, handle, is_global):
        self.handle = handle
        self.is_global = is_global

    def __del__(self):
        if not self.is_global and _LIB is not None:
            if _LIB.TVMFuncFree(self.handle) != 0:
                raise get_last_ffi_error()

    def __call__(self, *args):			# 一个可调用对象
        temp_args = []
        values, tcodes, num_args = _make_tvm_args(args, temp_args)
        ret_val = TVMValue()
        ret_tcode = ctypes.c_int()
        if _LIB.TVMFuncCall(
                self.handle, values, tcodes, ctypes.c_int(num_args),
                ctypes.byref(ret_val), ctypes.byref(ret_tcode)) != 0:
            raise get_last_ffi_error()
        _ = temp_args
        _ = args
        return RETURN_SWITCH[ret_tcode.value](ret_val)

class PackedFunc(PackedFuncBase):
    # nothing here
```

`PackedFunc`继承自`PackedFuncBase`，其中定义了一个`__call__`将这个类的对象变为可调用对象(类似C++重载了函数调用运算符)，其中调用了来自C++的`TVMFuncCall()`函数，这样可以完成一个`PackedFunc`的真正执行。



