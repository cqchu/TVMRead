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
 * \file registry.cc
 * \brief The global registry of packed function.
 */
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "runtime_base.h"

namespace tvm {
namespace runtime {

struct Registry::Manager {
  // map storing the functions.
  // We delibrately used raw pointer
  // This is because PackedFunc can contain callbacks into the host languge(python)
  // and the resource can become invalid because of indeterminstic order of destruction and forking.
  // The resources will only be recycled during program exit.
  std::unordered_map<std::string, Registry*> fmap;
  // mutex
  std::mutex mutex;

  Manager() {}

  static Manager* Global() {
    // We deliberately leak the Manager instance, to avoid leak sanitizers
    // complaining about the entries in Manager::fmap being leaked at program
    // exit.
    static Manager* inst = new Manager();
    return inst;
  }
};

Registry& Registry::set_body(PackedFunc f) {  // NOLINT(*)
  func_ = f;
  return *this;
}

Registry& Registry::Register(const std::string& name, bool can_override) {  // NOLINT(*)
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  if (m->fmap.count(name)) {
    CHECK(can_override) << "Global PackedFunc " << name << " is already registered";
  }

  Registry* r = new Registry();
  r->name_ = name;
  m->fmap[name] = r;
  return *r;
}

bool Registry::Remove(const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) return false;
  m->fmap.erase(it);
  return true;
}

const PackedFunc* Registry::Get(const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);                 // 根据name，找到Manager中fmap中所对应的那个Registry
  if (it == m->fmap.end()) return nullptr;
  return &(it->second->func_);                  // 获取Registry中的PackedFunc对象
}

std::vector<std::string> Registry::ListNames() {
  Manager* m = Manager::Global();               // 获取一个static的Manager对象的指针
                                                // Manager中维护了一个mutex，一个unordered_map<std::string, Registry*> fmap
                                                // 这里再看一下Registry的数据结构
                                                // 其中只有两个数据成员: std::string name_; PackedFunc func_;
                                                // 所以也很清楚了，这个Registry其实就是对一个Function的封装
  std::lock_guard<std::mutex> lock(m->mutex);
  std::vector<std::string> keys;                // 遍历fmap，返回了其所有元素的first
  keys.reserve(m->fmap.size());
  for (const auto& kv : m->fmap) {
    keys.push_back(kv.first);
  }
  return keys;
}

}  // namespace runtime
}  // namespace tvm

/*! \brief entry to to easily hold returning information */
struct TVMFuncThreadLocalEntry {
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char*> ret_vec_charp;
};

/*! \brief Thread local store that can be used to hold return values. */
typedef dmlc::ThreadLocalStore<TVMFuncThreadLocalEntry> TVMFuncThreadLocalStore;

int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) {
  API_BEGIN();
  tvm::runtime::Registry::Register(name, override != 0)
      .set_body(*static_cast<tvm::runtime::PackedFunc*>(f));
  API_END();
}

int TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out) {        // 根据传入进来的name，获取name对应的那个PackedFunc，并返回回去
  API_BEGIN();
  const tvm::runtime::PackedFunc* fp = tvm::runtime::Registry::Get(name);
  if (fp != nullptr) {
    *out = new tvm::runtime::PackedFunc(*fp);  // NOLINT(*)
  } else {
    *out = nullptr;
  }
  API_END();
}

int TVMFuncListGlobalNames(int* out_size, const char*** out_array) {
  API_BEGIN();
  TVMFuncThreadLocalEntry* ret = TVMFuncThreadLocalStore::Get();  // 获取一个static的TVMFuncThreadLocalEntry对象的指针
                                                                  // 这个对象中维护的是一个vector<string>和一个vector<const char *>
                                                                  // 从下面的代码看这两者实际存的东西是一样的
  ret->ret_vec_str = tvm::runtime::Registry::ListNames();         // 获取了在Manager中注册的所有Registry所对应的name，
                                                                  // 这里的Registry是TVM对PackedFunc的封装，可以跳进去看
  ret->ret_vec_charp.clear();                                     // 到此剩下的一个问题就是这些Registry什么时候注册的，不过现在先不急
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {          
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_array = dmlc::BeginPtr(ret->ret_vec_charp);                // 将这些const char*类型的name作为返回值返回回去
  *out_size = static_cast<int>(ret->ret_vec_str.size());
  API_END();
}
