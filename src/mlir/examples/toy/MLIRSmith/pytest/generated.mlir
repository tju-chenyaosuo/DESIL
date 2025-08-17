module {
  func.func private @func1(%arg0: f32, %arg1: index, %arg2: tensor<?x?xi64>) -> tensor<?x?x?xf16> {
    %true = arith.constant true
    %true_0 = arith.constant true
    %cst = arith.constant 6.153600e+04 : f16
    %false = arith.constant false
    %cst_1 = arith.constant 2.270400e+04 : f16
    %c685031725_i64 = arith.constant 685031725 : i64
    %false_2 = arith.constant false
    %cst_3 = arith.constant 6.224000e+03 : f16
    %c2088370536_i64 = arith.constant 2088370536 : i64
    %c1862809330_i64 = arith.constant 1862809330 : i64
    %c-32610_i16 = arith.constant -32610 : i16
    %cst_4 = arith.constant 6.550400e+04 : f16
    %c406327914_i64 = arith.constant 406327914 : i64
    %c25943_i16 = arith.constant 25943 : i16
    %true_5 = arith.constant true
    %false_6 = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index
    %c17 = arith.constant 17 : index
    %c18 = arith.constant 18 : index
    %c19 = arith.constant 19 : index
    %c20 = arith.constant 20 : index
    %c21 = arith.constant 21 : index
    %c22 = arith.constant 22 : index
    %c23 = arith.constant 23 : index
    %c24 = arith.constant 24 : index
    %c25 = arith.constant 25 : index
    %c26 = arith.constant 26 : index
    %c27 = arith.constant 27 : index
    %c28 = arith.constant 28 : index
    %c29 = arith.constant 29 : index
    %c30 = arith.constant 30 : index
    %c31 = arith.constant 31 : index
    %0 = tensor.empty(%c7) : tensor<?x6x1xi64>
    %1 = tensor.empty(%c7, %c27, %c20) : tensor<?x?x?xi64>
    %2 = tensor.empty() : tensor<1xf16>
    %3 = tensor.empty(%c5, %c17) : tensor<?x?x1xi32>
    %4 = tensor.empty() : tensor<10x1x10xi64>
    %5 = tensor.empty(%c17) : tensor<?xf16>
    %6 = tensor.empty(%c10) : tensor<?x1x10xi16>
    %7 = tensor.empty() : tensor<10x6xf32>
    %8 = tensor.empty(%c10) : tensor<?x6x1xi64>
    %9 = tensor.empty(%c14, %c19) : tensor<?x?xi1>
    %10 = tensor.empty(%c5) : tensor<?xi16>
    %11 = tensor.empty(%c2) : tensor<?xf16>
    %12 = tensor.empty(%c17, %c14, %c4) : tensor<?x?x?xi1>
    %13 = tensor.empty(%c30) : tensor<?xi16>
    %14 = tensor.empty(%c29, %c15, %c31) : tensor<?x?x?xi16>
    %15 = tensor.empty(%c31) : tensor<?x1x10xi32>
    %alloc = memref.alloc() : memref<1xi32>
    %alloc_7 = memref.alloc(%c5, %c9, %c11) : memref<?x?x?xi32>
    %alloc_8 = memref.alloc(%c31) : memref<?x6xi64>
    %alloc_9 = memref.alloc(%c6) : memref<?x6xi64>
    %alloc_10 = memref.alloc(%c7, %c11, %c16) : memref<?x?x?xf32>
    %alloc_11 = memref.alloc() : memref<10x6xf16>
    %alloc_12 = memref.alloc(%c8) : memref<?x6x1xi16>
    %alloc_13 = memref.alloc(%c5, %c0, %c15) : memref<?x?x?xi1>
    %alloc_14 = memref.alloc(%c8, %c14) : memref<?x?xf32>
    %alloc_15 = memref.alloc() : memref<10x6xf16>
    %alloc_16 = memref.alloc() : memref<10x1x10xf16>
    %alloc_17 = memref.alloc(%c29, %c3) : memref<?x?x1xi32>
    %alloc_18 = memref.alloc() : memref<1xf16>
    %alloc_19 = memref.alloc(%c27, %c28, %c30) : memref<?x?x?xi64>
    %alloc_20 = memref.alloc() : memref<10x6x1xf32>
    %alloc_21 = memref.alloc(%c15, %c30, %c22) : memref<?x?x?xf32>
    %16 = "tosa.reduce_min"(%11) {axis = 0 : i64} : (tensor<?xf16>) -> tensor<1xf16>
    %17 = math.atan2 %cst, %cst_4 : f16
    %18 = tensor.empty(%c9, %c8) : tensor<?x?x10xf16>
    %19 = "tosa.reduce_min"(%9) {axis = 0 : i64} : (tensor<?x?xi1>) -> tensor<1x?xi1>
    %20 = arith.mulf %cst, %cst : f16
    %21 = math.atan2 %cst_1, %cst_4 : f16
    %22 = "tosa.reduce_prod"(%10) {axis = 0 : i64} : (tensor<?xi16>) -> tensor<1xi16>
    %23 = arith.shli %c-32610_i16, %c25943_i16 : i16
    %24 = arith.maxui %false_2, %false_6 : i1
    %25 = index.maxu %c28, %c12
    %26 = index.or %c11, %c12
    %27 = index.divu %c13, %c26
    memref.assume_alignment %alloc_15, 2 : memref<10x6xf16>
    %28 = arith.cmpf ugt, %cst_3, %cst_4 : f16
    %29 = math.powf %2, %2 : tensor<1xf16>
    %30 = tensor.empty() : tensor<60xf32>
    %unpack = tensor.unpack %7 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [%c6] into %30 : tensor<10x6xf32> -> tensor<60xf32>
    %expanded = tensor.expand_shape %15 [[0], [1], [2, 3]] : tensor<?x1x10xi32> into tensor<?x1x10x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %31 = math.fpowi %cst_1, %c0_i32 : f16, i32
    %32 = index.divu %c2, %c14
    %33 = index.or %c7, %c14
    %34 = tensor.empty() : tensor<1xi1>
    %35 = vector.broadcast %true : i1 to vector<1xi1>
    %36 = vector.broadcast %c0_i32 : i32 to vector<1xi32>
    %37 = vector.gather %34[%c17] [%36], %35, %35 : tensor<1xi1>, vector<1xi32>, vector<1xi1>, vector<1xi1> into vector<1xi1>
    %generated = tensor.generate %c26 {
    ^bb0(%arg3: index, %arg4: index, %arg5: index):
      %141 = arith.cmpf one, %cst, %cst_4 : f16
      %142 = affine.for %arg6 = 0 to 7 iter_args(%arg7 = %6) -> (tensor<?x1x10xi16>) {
        %145 = tensor.empty(%33) : tensor<?x1x10xi16>
        affine.yield %145 : tensor<?x1x10xi16>
      }
      %143 = arith.maxui %c0_i32, %c0_i32 : i32
      %144 = "tosa.max_pool2d"(%expanded) {kernel = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x1x10x1xi32>) -> tensor<?x1x10x1xi32>
      %cst_36 = arith.constant 1.000000e+00 : f32
      tensor.yield %cst_36 : f32
    } : tensor<?x6x1xf32>
    %38 = "tosa.reduce_max"(%3) {axis = 0 : i64} : (tensor<?x?x1xi32>) -> tensor<1x?x1xi32>
    %39 = "tosa.reduce_min"(%16) {axis = 0 : i64} : (tensor<1xf16>) -> tensor<1xf16>
    %40 = math.absi %0 : tensor<?x6x1xi64>
    %41 = math.expm1 %11 : tensor<?xf16>
    %42 = tensor.empty() : tensor<1x1xf16>
    %pack = tensor.pack %2 padding_value(%cst : f16) outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [%c1] into %42 : tensor<1xf16> -> tensor<1x1xf16>
    %43 = "tosa.bitwise_and"(%39, %39) : (tensor<1xf16>, tensor<1xf16>) -> tensor<1xf16>
    %44 = "tosa.reciprocal"(%10) : (tensor<?xi16>) -> tensor<?xi16>
    %true_22 = index.bool.constant true
    %45 = "tosa.bitwise_xor"(%14, %14) : (tensor<?x?x?xi16>, tensor<?x?x?xi16>) -> tensor<?x?x?xi16>
    %46 = vector.flat_transpose %35 {columns = 1 : i32, rows = 1 : i32} : vector<1xi1> -> vector<1xi1>
    %47 = arith.mulf %cst_1, %cst_3 : f16
    %cst_23 = arith.constant 1.000000e+00 : f32
    %48 = vector.broadcast %cst_23 : f32 to vector<10x1x10xf32>
    %49 = vector.fma %48, %48, %48 : vector<10x1x10xf32>
    %50 = tensor.empty() : tensor<1xf16>
    %51 = "tosa.sub"(%2, %50) : (tensor<1xf16>, tensor<1xf16>) -> tensor<1xf16>
    %52 = bufferization.to_memref %50 : memref<1xf16>
    %53 = "tosa.reciprocal"(%6) : (tensor<?x1x10xi16>) -> tensor<?x1x10xi16>
    %54 = arith.divui %false_2, %false_2 : i1
    %55 = vector.broadcast %cst_1 : f16 to vector<10xf16>
    %56 = vector.broadcast %true_22 : i1 to vector<10xi1>
    vector.compressstore %alloc_11[%c7, %c3], %56, %55 : memref<10x6xf16>, vector<10xi1>, vector<10xf16>
    %unpack_24 = tensor.unpack %42 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [%c1] into %51 : tensor<1x1xf16> -> tensor<1xf16>
    %57 = math.log1p %7 : tensor<10x6xf32>
    %58 = memref.realloc %alloc : memref<1xi32> to memref<10xi32>
    %59 = math.cos %30 : tensor<60xf32>
    %expanded_25 = tensor.expand_shape %42 [[0], [1, 2]] : tensor<1x1xf16> into tensor<1x1x1xf16>
    %cast = tensor.cast %44 : tensor<?xi16> to tensor<1xi16>
    %splat = tensor.splat %c0_i32 : tensor<10x1x10xi32>
    %60 = vector.flat_transpose %46 {columns = 1 : i32, rows = 1 : i32} : vector<1xi1> -> vector<1xi1>
    %61 = vector.broadcast %c406327914_i64 : i64 to vector<i64>
    vector.transfer_write %61, %alloc_9[%c28, %c26] : vector<i64>, memref<?x6xi64>
    %extracted = tensor.extract %1[%c0, %c0, %c0] : tensor<?x?x?xi64>
    %62 = math.powf %16, %2 : tensor<1xf16>
    %63 = tensor.empty(%32, %27) : tensor<?x?x1xi32>
    %64 = tensor.empty() : tensor<1x1xi32>
    %65 = "tosa.scatter"(%3, %64, %63) : (tensor<?x?x1xi32>, tensor<1x1xi32>, tensor<?x?x1xi32>) -> tensor<?x?x1xi32>
    %66 = vector.load %alloc[%c0] : memref<1xi32>, vector<10x1x10xi32>
    %dim = tensor.dim %10, %c0 : tensor<?xi16>
    %67 = "tosa.transpose"(%2, %3) : (tensor<1xf16>, tensor<?x?x1xi32>) -> tensor<1xf16>
    %68 = arith.remf %cst, %cst_4 : f16
    %alloca = memref.alloca() : memref<1xf32>
    %extracted_26 = tensor.extract %4[%c1, %c0, %c0] : tensor<10x1x10xi64>
    %69 = math.log1p %16 : tensor<1xf16>
    %70 = arith.floordivsi %false_6, %true_0 : i1
    %71 = arith.floordivsi %c-32610_i16, %c-32610_i16 : i16
    %72 = bufferization.clone %alloc_11 : memref<10x6xf16> to memref<10x6xf16>
    %73 = arith.floordivsi %c1862809330_i64, %c406327914_i64 : i64
    %74 = index.floordivs %c6, %c21
    %75 = memref.atomic_rmw maxs %extracted, %alloc_8[%c0, %c2] : (i64, memref<?x6xi64>) -> i64
    %cst_27 = arith.constant 1.000000e+00 : f32
    %76 = vector.transfer_read %7[%c14, %c27], %cst_27 : tensor<10x6xf32>, vector<1xf32>
    %alloca_28 = memref.alloca() : memref<10x6xi64>
    %dim_29 = tensor.dim %6, %c0 : tensor<?x1x10xi16>
    %77 = arith.divui %c406327914_i64, %c2088370536_i64 : i64
    %78 = arith.andi %c-32610_i16, %c25943_i16 : i16
    %alloca_30 = memref.alloca(%c30) : memref<?x6xi64>
    %79 = vector.broadcast %cst_23 : f32 to vector<10x1x10xf32>
    %80 = vector.fma %79, %79, %49 : vector<10x1x10xf32>
    %81 = vector.matrix_multiply %60, %46 {lhs_columns = 1 : i32, lhs_rows = 1 : i32, rhs_columns = 1 : i32} : (vector<1xi1>, vector<1xi1>) -> vector<1xi1>
    %82 = index.shl %c17, %c3
    %83 = index.ceildivu %c27, %c14
    %cast_31 = tensor.cast %11 : tensor<?xf16> to tensor<6xf16>
    %84 = math.log10 %generated : tensor<?x6x1xf32>
    %85 = math.log1p %2 : tensor<1xf16>
    %86 = "tosa.reduce_sum"(%67) {axis = 0 : i64} : (tensor<1xf16>) -> tensor<1xf16>
    %expanded_32 = tensor.expand_shape %3 [[0], [1], [2, 3]] : tensor<?x?x1xi32> into tensor<?x?x1x1xi32>
    %87 = arith.negf %cst_4 : f16
    %88 = "tosa.max_pool2d"(%expanded) {kernel = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x1x10x1xi32>) -> tensor<?x1x10x1xi32>
    %89 = arith.minsi %true_22, %false_2 : i1
    %90 = arith.cmpi sgt, %c-32610_i16, %c25943_i16 : i16
    %91 = tensor.empty(%dim_29, %c24) : tensor<?x?xi1>
    %92 = tensor.empty(%c6, %c2) : tensor<?x?xi1>
    %93 = "tosa.select"(%9, %91, %92) : (tensor<?x?xi1>, tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>
    %94 = math.floor %86 : tensor<1xf16>
    %95 = index.shrs %32, %c0
    %96 = vector.broadcast %cst : f16 to vector<1xf16>
    %97 = arith.cmpi ugt, %extracted, %extracted : i64
    %98 = "tosa.bitwise_not"(%splat) : (tensor<10x1x10xi32>) -> tensor<10x1x10xi32>
    %dim_33 = tensor.dim %4, %c2 : tensor<10x1x10xi64>
    %99 = index.maxs %c20, %c2
    %100 = math.log10 %50 : tensor<1xf16>
    %101 = tensor.empty(%c6, %c22, %c6) : tensor<?x?x?xi16>
    %102 = "tosa.add"(%14, %101) : (tensor<?x?x?xi16>, tensor<?x?x?xi16>) -> tensor<?x?x?xi16>
    %103 = vector.load %alloc_8[%c0, %c5] : memref<?x6xi64>, vector<10x1x10xi64>
    %104 = index.casts %c2088370536_i64 : i64 to index
    %105 = tensor.empty(%c6, %c4, %c7) : tensor<?x?x?xi16>
    %106 = "tosa.pow"(%45, %105) : (tensor<?x?x?xi16>, tensor<?x?x?xi16>) -> tensor<?x?x?xi16>
    %107 = arith.cmpi slt, %c2088370536_i64, %extracted_26 : i64
    %108 = tensor.empty() : tensor<60xf32>
    %109 = "tosa.add"(%30, %108) : (tensor<60xf32>, tensor<60xf32>) -> tensor<60xf32>
    %110 = vector.load %alloc_13[%c0, %c0, %c0] : memref<?x?x?xi1>, vector<1xi1>
    %111 = arith.divsi %false_6, %true : i1
    %112 = arith.floordivsi %c-32610_i16, %c-32610_i16 : i16
    %113 = arith.addi %c1862809330_i64, %extracted : i64
    %114 = bufferization.to_tensor %alloc_16 : memref<10x1x10xf16>
    %115 = index.or %c7, %c25
    %116 = arith.andi %extracted, %extracted_26 : i64
    %117 = "tosa.tile"(%50) {multiples = array<i64>} : (tensor<1xf16>) -> tensor<1xf16>
    %118 = "tosa.tile"(%4) {multiples = array<i64>} : (tensor<10x1x10xi64>) -> tensor<10x1x10xi64>
    %119 = index.shl %99, %c5
    %120 = vector.load %alloc_19[%c0, %c0, %c0] : memref<?x?x?xi64>, vector<10x1x10xi64>
    %121 = vector.broadcast %false_6 : i1 to vector<10x6xi1>
    %expanded_34 = tensor.expand_shape %67 [[0, 1]] : tensor<1xf16> into tensor<1x1xf16>
    %122 = index.shru %c6, %c20
    %cast_35 = tensor.cast %9 : tensor<?x?xi1> to tensor<10x1xi1>
    %123 = index.and %104, %27
    %124 = arith.addi %false_2, %true_22 : i1
    %125 = math.ceil %117 : tensor<1xf16>
    %126 = bufferization.to_memref %splat : memref<10x1x10xi32>
    %127 = index.xor %dim_33, %c28
    %128 = tensor.empty() : tensor<10x1xi1>
    %129 = tensor.empty() : tensor<10x1xi1>
    %130 = "tosa.select"(%cast_35, %128, %129) : (tensor<10x1xi1>, tensor<10x1xi1>, tensor<10x1xi1>) -> tensor<10x1xi1>
    %131 = math.rsqrt %expanded_34 : tensor<1x1xf16>
    %132 = arith.minui %true_5, %false_6 : i1
    %133 = vector.shuffle %81, %35 [0, 1] : vector<1xi1>, vector<1xi1>
    %134 = "tosa.reduce_all"(%65) {axis = 0 : i64} : (tensor<?x?x1xi32>) -> tensor<1x?x1xi32>
    %135 = index.add %c7, %c21
    %136 = arith.minui %c25943_i16, %c-32610_i16 : i16
    %137 = vector.extract_strided_slice %36 {offsets = [0], sizes = [1], strides = [1]} : vector<1xi32> to vector<1xi32>
    %138 = affine.vector_load %alloc_18[%26] : memref<1xf16>, vector<10xf16>
    %139 = arith.muli %true_5, %true_5 : i1
    vector.print %35 : vector<1xi1>
    vector.print %36 : vector<1xi32>
    vector.print %37 : vector<1xi1>
    vector.print %46 : vector<1xi1>
    vector.print %48 : vector<10x1x10xf32>
    vector.print %49 : vector<10x1x10xf32>
    vector.print %60 : vector<1xi1>
    vector.print %61 : vector<i64>
    vector.print %66 : vector<10x1x10xi32>
    vector.print %79 : vector<10x1x10xf32>
    vector.print %80 : vector<10x1x10xf32>
    vector.print %81 : vector<1xi1>
    vector.print %96 : vector<1xf16>
    vector.print %103 : vector<10x1x10xi64>
    vector.print %110 : vector<1xi1>
    vector.print %120 : vector<10x1x10xi64>
    vector.print %121 : vector<10x6xi1>
    vector.print %137 : vector<1xi32>
    vector.print %138 : vector<10xf16>
    vector.print %true : i1
    vector.print %true_0 : i1
    vector.print %cst : f16
    vector.print %false : i1
    vector.print %cst_1 : f16
    vector.print %c685031725_i64 : i64
    vector.print %false_2 : i1
    vector.print %cst_3 : f16
    vector.print %c2088370536_i64 : i64
    vector.print %c1862809330_i64 : i64
    vector.print %c-32610_i16 : i16
    vector.print %cst_4 : f16
    vector.print %c406327914_i64 : i64
    vector.print %c25943_i16 : i16
    vector.print %true_5 : i1
    vector.print %false_6 : i1
    vector.print %c0_i32 : i32
    vector.print %true_22 : i1
    vector.print %cst_23 : f32
    vector.print %extracted : i64
    vector.print %extracted_26 : i64
    vector.print %cst_27 : f32
    %140 = tensor.empty(%c15, %83, %dim) : tensor<?x?x?xf16>
    return %140 : tensor<?x?x?xf16>
  }
}
