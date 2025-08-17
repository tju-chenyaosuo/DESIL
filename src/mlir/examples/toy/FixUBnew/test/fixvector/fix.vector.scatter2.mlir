module {
  func.func @func1() {
    %idx44 = index.constant 44
    %alloca = memref.alloca(%idx44) : memref<2x?xf32>
    %idx30 = index.constant 30
    %0 = vector.create_mask %idx30 : vector<3xi1>
    %cst = arith.constant 5.318000e+01 : f32
    %1 = vector.broadcast %cst : f32 to vector<3xf32>
    %c83_i32 = arith.constant 83 : i32
    %2 = vector.broadcast %c83_i32 : i32 to vector<3xi32>
    %idx1 = index.constant 1
    %idx48 = index.constant 48
    vector.scatter %alloca[%idx1, %idx48] [%2], %0, %1 : memref<2x?xf32>, vector<3xi32>, vector<3xi1>, vector<3xf32>
    return
  }
}
