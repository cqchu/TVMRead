import numpy as np
from tvm import relay
from tvm.relay import testing
from tvm.relay.testing.init import Xavier
import tvm
from tvm import te
from tvm.contrib import graph_runtime

print("##################### MY CODE START #####################")

data_shape = (1, 3, 64, 64)
dtype = "float32"
data_layout = "NCHW"
kernel_layout = "OIHW"
bn_axis = data_layout.index('C')

data = relay.var("data", shape=data_shape, dtype=dtype) # 终于看完了Var了
weight = relay.var("conv0_weight")
gamma = relay.var("bn0_gamma")
beta = relay.var("bn0_beta")
moving_mean = relay.var("bn0_moving_mean")
moving_var = relay.var("bn0_moving_var")

##### Net Structure #####
body = relay.nn.conv2d(data=data, weight=weight, channels=32, kernel_size=(3, 3), strides=(1, 1),   # 继续看看op吧，这个relay.nn.conv2d是对_make.conv2d的封装
                       padding=(1, 1), data_layout=data_layout, kernel_layout=kernel_layout)        # 获取了一个Call对象，其继承自Relay::Expr
body = relay.nn.batch_norm(data=body, gamma=gamma, beta=beta, moving_mean=moving_mean, moving_var=moving_var, axis=bn_axis)[0]
out = relay.nn.relu(data=body)                                                                      # 完成了网络的构建
free_vars = relay.analysis.free_vars(out)   # 其实就是自己定义的这几个变量 --------- 这一行需要跳进去有遍历图的函数 ExprVistor, 后续需要细看
net = relay.Function(free_vars, out)        # 调用C++构造函数，构建一个Function
                                            # include/tvm/relay/function.h:104
mod = tvm.IRModule.from_expr(net)           # 用这个Function构建这个Module
shape_dict = {v.name_hint : v.checked_type for v in mod["main"].params}
initializer = Xavier()
params = {}
for k, v in shape_dict.items():
    if k == "data":
        continue
    init_value = np.zeros(v.concrete_shape).astype(v.dtype)
    initializer(k, init_value)
    params[k] = tvm.nd.array(init_value, ctx=tvm.cpu(0))

# print(mod.astext(show_meta_data=False))

opt_level = 3
target = tvm.target.create("llvm")                      # 创建Target
with tvm.transform.PassContext(opt_level=opt_level):    # 构建了一个PassContext对象
    graph, lib, params = relay.build(mod, target, params=params)
