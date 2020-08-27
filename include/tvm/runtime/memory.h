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
 * \file tvm/runtime/memory.h
 * \brief Runtime memory management.
 */
#ifndef TVM_RUNTIME_MEMORY_H_
#define TVM_RUNTIME_MEMORY_H_

#include <tvm/runtime/object.h>

#include <cstdlib>
#include <type_traits>
#include <utility>

namespace tvm {
namespace runtime {
/*!
 * \brief Allocate an object using default allocator.
 * \param args arguments to the constructor.
 * \tparam T the node type.
 * \return The ObjectPtr to the allocated object.
 */
template <typename T, typename... Args>
inline ObjectPtr<T> make_object(Args&&... args);

// Detail implementations after this
//
// The current design allows swapping the
// allocator pattern when necessary.
//
// Possible future allocator optimizations:
// - Arena allocator that gives ownership of memory to arena (deleter_= nullptr)
// - Thread-local object pools: one pool per size and alignment requirement.
// - Can specialize by type of object to give the specific allocator to each object.

/*!
 * \brief Base class of object allocators that implements make.
 *  Use curiously recurring template pattern.
 *
 * \tparam Derived The derived class.
 */
template <typename Derived>               // 其子类会实例化这个模板，即奇异递归模板模式
class ObjAllocatorBase {                  // 这个类中也没有数据成员
 public:
  /*!
   * \brief Make a new object using the allocator.
   * \tparam T The type to be allocated.
   * \tparam Args The constructor signature.
   * \param args The arguments.
   */
  template <typename T, typename... Args>         // 看看Allocate的具体方式
  inline ObjectPtr<T> make_object(Args&&... args) {
    using Handler = typename Derived::template Handler<T>;  // 这个在C++ Primer第593页有讲, 
                                                            // 使用这种类型成员需要显式的使用typename
                                                            // 而Handler本身就是个模板类，所以这里用Derived::template
                                                            // 即这里获取了T的那个Handler类
    static_assert(std::is_base_of<Object, T>::value, "make can only be used to create Object");
    T* ptr = Handler::New(static_cast<Derived*>(this), std::forward<Args>(args)...);  // 调用Handle的New函数，Handler::New<Conv2dAttrs, Args...>
    ptr->type_index_ = T::RuntimeTypeIndex();
    ptr->deleter_ = Handler::Deleter();
    return ObjectPtr<T>(ptr);
  }

  /*!
   * \tparam ArrayType The type to be allocated.
   * \tparam ElemType The type of array element.
   * \tparam Args The constructor signature.
   * \param num_elems The number of array elements.
   * \param args The arguments.
   */
  template <typename ArrayType, typename ElemType, typename... Args>
  inline ObjectPtr<ArrayType> make_inplace_array(size_t num_elems, Args&&... args) {
    using Handler = typename Derived::template ArrayHandler<ArrayType, ElemType>;
    static_assert(std::is_base_of<Object, ArrayType>::value,
                  "make_inplace_array can only be used to create Object");
    ArrayType* ptr =
        Handler::New(static_cast<Derived*>(this), num_elems, std::forward<Args>(args)...);
    ptr->type_index_ = ArrayType::RuntimeTypeIndex();
    ptr->deleter_ = Handler::Deleter();
    return ObjectPtr<ArrayType>(ptr);
  }
};

// Simple allocator that uses new/delete.
class SimpleObjAllocator : public ObjAllocatorBase<SimpleObjAllocator> {    // 有是TVM的骚操作，继承ObjAllocatorBase模板，但这个模板实例化的其实就是自己这个类本身
 public:                                                                    // 也即所谓的奇异递归模板模式，这样基类中可以使用子类的成员
  template <typename T>   // 里面定义了两个Handler类，但是并没有任何数据成员    // 可以理解为一种静态多态，在编译实例化时完成了多态
  class Handler {                                                           // 相比于动态多态需要虚表虚函数的支持，这样子性能会好很多
   public:
    using StorageType = typename std::aligned_storage<sizeof(T), alignof(T)>::type; // 使用C++的aligned_storage

    template <typename... Args>
    static T* New(SimpleObjAllocator*, Args&&... args) {                    //Handler::New<Conv2dAttrs, Args...>
      // NOTE: the first argument is not needed for SimpleObjAllocator
      // It is reserved for special allocators that needs to recycle
      // the object to itself (e.g. in the case of object pool).
      //
      // In the case of an object pool, an allocator needs to create
      // a special chunk memory that hides reference to the allocator
      // and call allocator's release function in the deleter.

      // NOTE2: Use inplace new to allocate
      // This is used to get rid of warning when deleting a virtual
      // class with non-virtual destructor.
      // We are fine here as we captured the right deleter during construction.
      // This is also the right way to get storage type for an object pool.
      StorageType* data = new StorageType();      // 使用C++的aligned_storage创建Conv2dAttrs的存储空间
      new (data) T(std::forward<Args>(args)...);  // 用传进来的参数构造这个存储空间，使用的就是new
      return reinterpret_cast<T*>(data);          // 然后返回这个Conv2dAttrs类型指针
    }

    static Object::FDeleter Deleter() { return Deleter_; }

   private:
    static void Deleter_(Object* objptr) {
      // NOTE: this is important to cast back to T*
      // because objptr and tptr may not be the same
      // depending on how sub-class allocates the space.
      T* tptr = static_cast<T*>(objptr);
      // It is important to do tptr->T::~T(),
      // so that we explicitly call the specific destructor
      // instead of tptr->~T(), which could mean the intention
      // call a virtual destructor(which may not be available and is not required).
      tptr->T::~T();                              // 显式调用析构函数来delete
      delete reinterpret_cast<StorageType*>(tptr);
    }
  };

  // Array handler that uses new/delete.
  template <typename ArrayType, typename ElemType>
  class ArrayHandler {
   public:
    using StorageType = typename std::aligned_storage<sizeof(ArrayType), alignof(ArrayType)>::type;
    // for now only support elements that aligns with array header.
    static_assert(alignof(ArrayType) % alignof(ElemType) == 0 &&
                      sizeof(ArrayType) % alignof(ElemType) == 0,
                  "element alignment constraint");

    template <typename... Args>
    static ArrayType* New(SimpleObjAllocator*, size_t num_elems, Args&&... args) {
      // NOTE: the first argument is not needed for ArrayObjAllocator
      // It is reserved for special allocators that needs to recycle
      // the object to itself (e.g. in the case of object pool).
      //
      // In the case of an object pool, an allocator needs to create
      // a special chunk memory that hides reference to the allocator
      // and call allocator's release function in the deleter.
      // NOTE2: Use inplace new to allocate
      // This is used to get rid of warning when deleting a virtual
      // class with non-virtual destructor.
      // We are fine here as we captured the right deleter during construction.
      // This is also the right way to get storage type for an object pool.
      size_t unit = sizeof(StorageType);
      size_t requested_size = num_elems * sizeof(ElemType) + sizeof(ArrayType);
      size_t num_storage_slots = (requested_size + unit - 1) / unit;
      StorageType* data = new StorageType[num_storage_slots];
      new (data) ArrayType(std::forward<Args>(args)...);
      return reinterpret_cast<ArrayType*>(data);
    }

    static Object::FDeleter Deleter() { return Deleter_; }

   private:
    static void Deleter_(Object* objptr) {
      // NOTE: this is important to cast back to ArrayType*
      // because objptr and tptr may not be the same
      // depending on how sub-class allocates the space.
      ArrayType* tptr = static_cast<ArrayType*>(objptr);
      // It is important to do tptr->ArrayType::~ArrayType(),
      // so that we explicitly call the specific destructor
      // instead of tptr->~ArrayType(), which could mean the intention
      // call a virtual destructor(which may not be available and is not required).
      tptr->ArrayType::~ArrayType();
      StorageType* p = reinterpret_cast<StorageType*>(tptr);
      delete[] p;
    }
  };
};

template <typename T, typename... Args>
inline ObjectPtr<T> make_object(Args&&... args) {                           // SimpleObjAllocator继承自ObjAllocatorBase, 是个Allocator对象
  return SimpleObjAllocator().make_object<T>(std::forward<Args>(args)...);  // 
}

template <typename ArrayType, typename ElemType, typename... Args>
inline ObjectPtr<ArrayType> make_inplace_array_object(size_t num_elems, Args&&... args) {
  return SimpleObjAllocator().make_inplace_array<ArrayType, ElemType>(num_elems,
                                                                      std::forward<Args>(args)...);
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MEMORY_H_
