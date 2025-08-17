func.func @func1() {
  %i = arith.constant 1 : i32
  %index = index.constant 1
  %memref0 = memref.alloc() : memref<64xi32>
  %0 = memref.realloc %memref0 : memref<64xi32> to memref<64xi32>
  memref.store %i, %memref0[%index] : memref<64xi32>
  return
}