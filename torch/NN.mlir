#loc = loc(unknown)
module attributes {torch.debug_module_name = "NN"} {
  func.func @forward(%arg0: !torch.vtensor<[1,3,32,64],f32> loc(unknown)) -> !torch.vtensor<[1,3,64,128],f32> {
    %true = torch.constant.bool true loc(#loc17)
    %0 = torch.vtensor.literal(dense<[-0.00152999116, 0.0651566982, -0.0527642556]> : tensor<3xf32>) : !torch.vtensor<[3],f32> loc(#loc)
    %1 = torch.vtensor.literal(dense_resource<__elided__> : tensor<3x3x2x2xf32>) : !torch.vtensor<[3,3,2,2],f32> loc(#loc)
    %int1 = torch.constant.int 1 loc(#loc18)
    %int0 = torch.constant.int 0 loc(#loc19)
    %int2 = torch.constant.int 2 loc(#loc20)
    %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int> loc(#loc24)
    %3 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int> loc(#loc22)
    %4 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int> loc(#loc22)
    %5 = torch.aten.convolution %arg0, %1, %0, %3, %2, %4, %true, %2, %int1 : !torch.vtensor<[1,3,32,64],f32>, !torch.vtensor<[3,3,2,2],f32>, !torch.vtensor<[3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,3,64,128],f32> loc(#loc17)
    %6 = torch.aten.relu %5 : !torch.vtensor<[1,3,64,128],f32> -> !torch.vtensor<[1,3,64,128],f32> loc(#loc25)
    return %6 : !torch.vtensor<[1,3,64,128],f32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py":956:15)
#loc2 = loc("/usr/local/lib/python3.8/dist-packages/torch/nn/modules/container.py":217:20)
#loc3 = loc("convTranspose2d_relu.py":12:6)
#loc4 = loc("/usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py":954:30)
#loc5 = loc("/usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py":953:45)
#loc6 = loc("/usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py":951:27)
#loc7 = loc("/usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py":952:25)
#loc8 = loc("/usr/local/lib/python3.8/dist-packages/torch/nn/functional.py":1475:17)
#loc9 = loc("/usr/local/lib/python3.8/dist-packages/torch/nn/modules/activation.py":103:15)
#loc10 = loc(callsite(#loc1 at #loc2))
#loc11 = loc(callsite(#loc4 at #loc2))
#loc12 = loc(callsite(#loc5 at #loc2))
#loc13 = loc(callsite(#loc6 at #loc2))
#loc14 = loc(callsite(#loc at #loc7))
#loc15 = loc(callsite(#loc at #loc2))
#loc16 = loc(callsite(#loc8 at #loc9))
#loc17 = loc(callsite(#loc10 at #loc3))
#loc18 = loc(callsite(#loc11 at #loc3))
#loc19 = loc(callsite(#loc12 at #loc3))
#loc20 = loc(callsite(#loc13 at #loc3))
#loc21 = loc(callsite(#loc14 at #loc2))
#loc22 = loc(callsite(#loc15 at #loc3))
#loc23 = loc(callsite(#loc16 at #loc2))
#loc24 = loc(callsite(#loc21 at #loc3))
#loc25 = loc(callsite(#loc23 at #loc3))
