module {
  func.func @func1() {
    %idx63 = index.constant 63
    %idx88 = index.constant 88
    %alloca = memref.alloca(%idx63, %idx88) : memref<2x?x?xf32>
    %idx25 = index.constant 25
    %0 = vector.create_mask %idx25 : vector<8xi1>
    %cst = arith.constant 9.152000e+01 : f32
    %1 = vector.broadcast %cst : f32 to vector<8xf32>
    %c60_i32 = arith.constant 60 : i32
    %2 = vector.broadcast %c60_i32 : i32 to vector<8xi32>
    %idx19 = index.constant 19
    %idx90 = index.constant 90
    %idx56 = index.constant 56
    vector.scatter %alloca[%idx19, %idx90, %idx56] [%2], %0, %1 : memref<2x?x?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>
    return
  }
}
