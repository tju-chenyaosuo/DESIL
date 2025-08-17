module {
  func.func @func1() {
    %idx11 = index.constant 11
    %idx4 = index.constant 4
    %alloca = memref.alloca(%idx11, %idx4) : memref<?x?xf32>
    %idx26 = index.constant 26
    %0 = vector.create_mask %idx26 : vector<7xi1>
    %cst = arith.constant 4.024000e+01 : f32
    %1 = vector.broadcast %cst : f32 to vector<7xf32>
    %idx82 = index.constant 82
    %idx71 = index.constant 71
    vector.compressstore %alloca[%idx82, %idx71], %0, %1 : memref<?x?xf32>, vector<7xi1>, vector<7xf32>
    return
  }
}
