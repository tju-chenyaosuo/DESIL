module {
  func.func @func1() {
    %idx10 = index.constant 10
    %alloca = memref.alloca(%idx10, %idx10) : memref<?x?xf32>
    %idx2 = index.constant 2
    %0 = vector.create_mask %idx2 : vector<16xi1>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = vector.broadcast %cst : f32 to vector<16xf32>
    %c100_i32 = arith.constant 100 : i32
    %2 = vector.broadcast %c100_i32 : i32 to vector<16xi32>
    %idx0 = index.constant 0
    vector.scatter %alloca[%idx0, %idx0] [%2], %0, %1 : memref<?x?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
    return
  }
}