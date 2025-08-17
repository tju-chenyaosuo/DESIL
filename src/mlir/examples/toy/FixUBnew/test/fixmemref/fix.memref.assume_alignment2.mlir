func.func @func1() {
  %0 = memref.alloca() : memref<8x64xi64>
  memref.assume_alignment %0, 2 : memref<8x64xi64>
  return
}