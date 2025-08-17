module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @printF32(f32)
  llvm.func @printF16(i16)
  llvm.func @printNewline()
  llvm.func @printI64(i64)
  llvm.func @func1(%arg0: f32, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(840 : index) : i64
    %1 = llvm.mlir.constant(30 : index) : i64
    %2 = llvm.mlir.constant(27 : index) : i64
    %3 = llvm.mlir.constant(31 : index) : i64
    %4 = llvm.mlir.constant(15 : index) : i64
    %5 = llvm.mlir.constant(dense<6.153600e+04> : vector<1xf16>) : vector<1xf16>
    %6 = llvm.mlir.constant(5 : index) : i64
    %7 = llvm.mlir.constant(dense<406327914> : vector<i64>) : vector<1xi64>
    %8 = llvm.mlir.constant(dense<2.270400e+04> : vector<10xf16>) : vector<10xf16>
    %9 = llvm.mlir.constant(dense<1.000000e+00> : vector<10x1x10xf32>) : !llvm.array<10 x array<1 x vector<10xf32>>>
    %10 = llvm.mlir.constant(dense<0> : vector<1xi32>) : vector<1xi32>
    %11 = llvm.mlir.constant(dense<true> : vector<1xi1>) : vector<1xi1>
    %12 = llvm.mlir.constant(dense<false> : vector<10x6xi1>) : !llvm.array<10 x vector<6xi1>>
    %13 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %14 = llvm.mlir.constant(6.153600e+04 : f16) : f16
    %15 = llvm.mlir.constant(2.270400e+04 : f16) : f16
    %16 = llvm.mlir.constant(685031725 : i64) : i64
    %17 = llvm.mlir.constant(6.224000e+03 : f16) : f16
    %18 = llvm.mlir.constant(2088370536 : i64) : i64
    %19 = llvm.mlir.constant(1862809330 : i64) : i64
    %20 = llvm.mlir.constant(6.550400e+04 : f16) : f16
    %21 = llvm.mlir.constant(406327914 : i64) : i64
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.mlir.constant(2 : index) : i64
    %24 = llvm.mlir.constant(3 : index) : i64
    %25 = llvm.mlir.constant(7 : index) : i64
    %26 = llvm.mlir.constant(17 : index) : i64
    %27 = llvm.mlir.constant(26 : index) : i64
    %28 = llvm.mlir.constant(28 : index) : i64
    %29 = llvm.mlir.constant(dense<0.000000e+00> : vector<10x1x10xf32>) : !llvm.array<10 x array<1 x vector<10xf32>>>
    %30 = llvm.mlir.constant(1 : i64) : i64
    %31 = llvm.mlir.constant(0 : i64) : i64
    %32 = llvm.mlir.constant(-32610 : i64) : i64
    %33 = llvm.mlir.constant(25943 : i64) : i64
    %34 = llvm.mlir.constant(0 : i32) : i32
    %35 = llvm.mlir.constant(0 : index) : i64
    %36 = llvm.mlir.constant(6 : index) : i64
    %37 = llvm.mlir.constant(dense<0.000000e+00> : vector<1x10xf32>) : !llvm.array<1 x vector<10xf32>>
    %38 = builtin.unrealized_conversion_cast %26 : i64 to index
    %39 = builtin.unrealized_conversion_cast %22 : i64 to index
    %40 = builtin.unrealized_conversion_cast %35 : i64 to index
    %41 = builtin.unrealized_conversion_cast %12 : !llvm.array<10 x vector<6xi1>> to vector<10x6xi1>
    %42 = builtin.unrealized_conversion_cast %9 : !llvm.array<10 x array<1 x vector<10xf32>>> to vector<10x1x10xf32>
    %43 = builtin.unrealized_conversion_cast %7 : vector<1xi64> to vector<i64>
    %44 = builtin.unrealized_conversion_cast %6 : i64 to index
    %45 = bufferization.alloc_tensor() : tensor<7x27x20xi64>
    %46 = bufferization.alloc_tensor() : tensor<10x1x10xi64>
    %47 = llvm.mlir.null : !llvm.ptr
    %48 = llvm.getelementptr %47[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.call @malloc(%49) : (i64) -> !llvm.ptr
    %51 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.insertvalue %50, %51[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %53 = llvm.insertvalue %50, %52[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.insertvalue %35, %53[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %22, %54[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.insertvalue %22, %55[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = builtin.unrealized_conversion_cast %56 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<1xi32>
    %58 = llvm.mlir.null : !llvm.ptr
    %59 = llvm.getelementptr %58[186] : (!llvm.ptr) -> !llvm.ptr, i64
    %60 = llvm.ptrtoint %59 : !llvm.ptr to i64
    %61 = llvm.call @malloc(%60) : (i64) -> !llvm.ptr
    %62 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.insertvalue %61, %62[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.insertvalue %61, %63[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %35, %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.insertvalue %3, %65[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.insertvalue %36, %66[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %36, %67[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %22, %68[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = builtin.unrealized_conversion_cast %69 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<31x6xi64>
    %71 = llvm.mlir.null : !llvm.ptr
    %72 = llvm.getelementptr %71[36] : (!llvm.ptr) -> !llvm.ptr, i64
    %73 = llvm.ptrtoint %72 : !llvm.ptr to i64
    %74 = llvm.call @malloc(%73) : (i64) -> !llvm.ptr
    %75 = llvm.mlir.null : !llvm.ptr
    %76 = llvm.getelementptr %75[60] : (!llvm.ptr) -> !llvm.ptr, f16
    %77 = llvm.ptrtoint %76 : !llvm.ptr to i64
    %78 = llvm.call @malloc(%77) : (i64) -> !llvm.ptr
    %79 = llvm.mlir.null : !llvm.ptr
    %80 = llvm.getelementptr %79[75] : (!llvm.ptr) -> !llvm.ptr, i1
    %81 = llvm.ptrtoint %80 : !llvm.ptr to i64
    %82 = llvm.call @malloc(%81) : (i64) -> !llvm.ptr
    %83 = llvm.mlir.null : !llvm.ptr
    %84 = llvm.getelementptr %83[60] : (!llvm.ptr) -> !llvm.ptr, f16
    %85 = llvm.ptrtoint %84 : !llvm.ptr to i64
    %86 = llvm.call @malloc(%85) : (i64) -> !llvm.ptr
    %87 = llvm.mlir.null : !llvm.ptr
    %88 = llvm.getelementptr %87[1] : (!llvm.ptr) -> !llvm.ptr, f16
    %89 = llvm.ptrtoint %88 : !llvm.ptr to i64
    %90 = llvm.call @malloc(%89) : (i64) -> !llvm.ptr
    %91 = llvm.mlir.null : !llvm.ptr
    %92 = llvm.getelementptr %91[22680] : (!llvm.ptr) -> !llvm.ptr, i64
    %93 = llvm.ptrtoint %92 : !llvm.ptr to i64
    %94 = llvm.call @malloc(%93) : (i64) -> !llvm.ptr
    %95 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %96 = llvm.insertvalue %94, %95[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %97 = llvm.insertvalue %94, %96[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %98 = llvm.insertvalue %35, %97[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %99 = llvm.insertvalue %2, %98[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %100 = llvm.insertvalue %28, %99[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %101 = llvm.insertvalue %1, %100[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %102 = llvm.insertvalue %0, %101[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %103 = llvm.insertvalue %1, %102[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %104 = llvm.insertvalue %22, %103[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %105 = builtin.unrealized_conversion_cast %104 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<27x28x30xi64>
    %106 = llvm.ptrtoint %86 : !llvm.ptr to i64
    %107 = llvm.and %106, %22  : i64
    %108 = llvm.icmp "eq" %107, %35 : i64
    "llvm.intr.assume"(%108) : (i1) -> ()
    %109 = bufferization.alloc_tensor() : tensor<1xi1>
    %110 = vector.gather %109[%38] [%10], %11, %11 : tensor<1xi1>, vector<1xi32>, vector<1xi1>, vector<1xi1> into vector<1xi1>
    %111 = llvm.intr.matrix.transpose %11 {columns = 1 : i32, rows = 1 : i32} : vector<1xi1> into vector<1xi1>
    %112 = llvm.extractvalue %9[0, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %113 = llvm.extractvalue %9[0, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %114 = llvm.extractvalue %9[0, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %115 = llvm.intr.fmuladd(%112, %113, %114)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %116 = llvm.insertvalue %115, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %117 = llvm.insertvalue %116, %29[0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %118 = llvm.extractvalue %9[1, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %119 = llvm.extractvalue %9[1, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %120 = llvm.extractvalue %9[1, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %121 = llvm.intr.fmuladd(%118, %119, %120)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %122 = llvm.insertvalue %121, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %123 = llvm.insertvalue %122, %117[1] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %124 = llvm.extractvalue %9[2, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %125 = llvm.extractvalue %9[2, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %126 = llvm.extractvalue %9[2, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %127 = llvm.intr.fmuladd(%124, %125, %126)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %128 = llvm.insertvalue %127, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %129 = llvm.insertvalue %128, %123[2] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %130 = llvm.extractvalue %9[3, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %131 = llvm.extractvalue %9[3, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %132 = llvm.extractvalue %9[3, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %133 = llvm.intr.fmuladd(%130, %131, %132)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %134 = llvm.insertvalue %133, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %135 = llvm.insertvalue %134, %129[3] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %136 = llvm.extractvalue %9[4, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %137 = llvm.extractvalue %9[4, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %138 = llvm.extractvalue %9[4, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %139 = llvm.intr.fmuladd(%136, %137, %138)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %140 = llvm.insertvalue %139, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %141 = llvm.insertvalue %140, %135[4] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %142 = llvm.extractvalue %9[5, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %143 = llvm.extractvalue %9[5, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %144 = llvm.extractvalue %9[5, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %145 = llvm.intr.fmuladd(%142, %143, %144)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %146 = llvm.insertvalue %145, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %147 = llvm.insertvalue %146, %141[5] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %148 = llvm.extractvalue %9[6, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %149 = llvm.extractvalue %9[6, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %150 = llvm.extractvalue %9[6, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %151 = llvm.intr.fmuladd(%148, %149, %150)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %152 = llvm.insertvalue %151, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %153 = llvm.insertvalue %152, %147[6] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %154 = llvm.extractvalue %9[7, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %155 = llvm.extractvalue %9[7, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %156 = llvm.extractvalue %9[7, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %157 = llvm.intr.fmuladd(%154, %155, %156)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %158 = llvm.insertvalue %157, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %159 = llvm.insertvalue %158, %153[7] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %160 = llvm.extractvalue %9[8, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %161 = llvm.extractvalue %9[8, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %162 = llvm.extractvalue %9[8, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %163 = llvm.intr.fmuladd(%160, %161, %162)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %164 = llvm.insertvalue %163, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %165 = llvm.insertvalue %164, %159[8] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %166 = llvm.extractvalue %9[9, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %167 = llvm.extractvalue %9[9, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %168 = llvm.extractvalue %9[9, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %169 = llvm.intr.fmuladd(%166, %167, %168)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %170 = llvm.insertvalue %169, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %171 = llvm.insertvalue %170, %165[9] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %172 = builtin.unrealized_conversion_cast %171 : !llvm.array<10 x array<1 x vector<10xf32>>> to vector<10x1x10xf32>
    %173 = llvm.mul %25, %36  : i64
    %174 = llvm.add %173, %24  : i64
    %175 = llvm.getelementptr %78[%174] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    llvm.store %8, %175 {alignment = 2 : i64} : vector<10xf16>, !llvm.ptr
    %176 = llvm.intr.matrix.transpose %111 {columns = 1 : i32, rows = 1 : i32} : vector<1xi1> into vector<1xi1>
    %177 = llvm.extractelement %7[%35 : i64] : vector<1xi64>
    %178 = llvm.mul %28, %36  : i64
    %179 = llvm.add %178, %27  : i64
    %180 = llvm.getelementptr %74[%179] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %177, %180 : i64, !llvm.ptr
    %extracted = tensor.extract %45[%40, %40, %40] : tensor<7x27x20xi64>
    %181 = vector.load %57[%40] : memref<1xi32>, vector<10x1x10xi32>
    %extracted_0 = tensor.extract %46[%39, %40, %40] : tensor<10x1x10xi64>
    %182 = llvm.mul %35, %36  : i64
    %183 = llvm.add %182, %23  : i64
    %184 = llvm.getelementptr %61[%183] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %185 = llvm.atomicrmw max %184, %extracted acq_rel : !llvm.ptr, i64
    %186 = llvm.extractvalue %9[0, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %187 = llvm.extractvalue %9[0, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %188 = llvm.extractvalue %117[0, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %189 = llvm.intr.fmuladd(%186, %187, %188)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %190 = llvm.insertvalue %189, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %191 = llvm.insertvalue %190, %29[0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %192 = llvm.extractvalue %9[1, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %193 = llvm.extractvalue %9[1, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %194 = llvm.extractvalue %123[1, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %195 = llvm.intr.fmuladd(%192, %193, %194)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %196 = llvm.insertvalue %195, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %197 = llvm.insertvalue %196, %191[1] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %198 = llvm.extractvalue %9[2, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %199 = llvm.extractvalue %9[2, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %200 = llvm.extractvalue %129[2, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %201 = llvm.intr.fmuladd(%198, %199, %200)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %202 = llvm.insertvalue %201, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %203 = llvm.insertvalue %202, %197[2] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %204 = llvm.extractvalue %9[3, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %205 = llvm.extractvalue %9[3, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %206 = llvm.extractvalue %135[3, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %207 = llvm.intr.fmuladd(%204, %205, %206)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %208 = llvm.insertvalue %207, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %209 = llvm.insertvalue %208, %203[3] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %210 = llvm.extractvalue %9[4, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %211 = llvm.extractvalue %9[4, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %212 = llvm.extractvalue %141[4, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %213 = llvm.intr.fmuladd(%210, %211, %212)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %214 = llvm.insertvalue %213, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %215 = llvm.insertvalue %214, %209[4] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %216 = llvm.extractvalue %9[5, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %217 = llvm.extractvalue %9[5, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %218 = llvm.extractvalue %147[5, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %219 = llvm.intr.fmuladd(%216, %217, %218)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %220 = llvm.insertvalue %219, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %221 = llvm.insertvalue %220, %215[5] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %222 = llvm.extractvalue %9[6, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %223 = llvm.extractvalue %9[6, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %224 = llvm.extractvalue %153[6, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %225 = llvm.intr.fmuladd(%222, %223, %224)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %226 = llvm.insertvalue %225, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %227 = llvm.insertvalue %226, %221[6] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %228 = llvm.extractvalue %9[7, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %229 = llvm.extractvalue %9[7, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %230 = llvm.extractvalue %159[7, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %231 = llvm.intr.fmuladd(%228, %229, %230)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %232 = llvm.insertvalue %231, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %233 = llvm.insertvalue %232, %227[7] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %234 = llvm.extractvalue %9[8, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %235 = llvm.extractvalue %9[8, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %236 = llvm.extractvalue %165[8, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %237 = llvm.intr.fmuladd(%234, %235, %236)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %238 = llvm.insertvalue %237, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %239 = llvm.insertvalue %238, %233[8] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %240 = llvm.extractvalue %9[9, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %241 = llvm.extractvalue %9[9, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %242 = llvm.extractvalue %171[9, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %243 = llvm.intr.fmuladd(%240, %241, %242)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %244 = llvm.insertvalue %243, %37[0] : !llvm.array<1 x vector<10xf32>> 
    %245 = llvm.insertvalue %244, %239[9] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %246 = builtin.unrealized_conversion_cast %245 : !llvm.array<10 x array<1 x vector<10xf32>>> to vector<10x1x10xf32>
    %247 = llvm.intr.matrix.multiply %176, %111 {lhs_columns = 1 : i32, lhs_rows = 1 : i32, rhs_columns = 1 : i32} : (vector<1xi1>, vector<1xi1>) -> vector<1xi1>
    %248 = vector.load %70[%40, %44] : memref<31x6xi64>, vector<10x1x10xi64>
    %249 = llvm.mul %35, %4  : i64
    %250 = llvm.mul %35, %4  : i64
    %251 = llvm.add %249, %250  : i64
    %252 = llvm.add %251, %35  : i64
    %253 = llvm.getelementptr %82[%252] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    %254 = llvm.load %253 : !llvm.ptr -> i1
    %255 = llvm.mlir.undef : vector<1xi1>
    %256 = llvm.insertelement %254, %255[%34 : i32] : vector<1xi1>
    %257 = llvm.shufflevector %256, %255 [0] : vector<1xi1> 
    %258 = vector.load %105[%40, %40, %40] : memref<27x28x30xi64>, vector<10x1x10xi64>
    %259 = llvm.getelementptr %90[15] : (!llvm.ptr) -> !llvm.ptr, f16
    %260 = llvm.load %259 {alignment = 2 : i64} : !llvm.ptr -> vector<10xf16>
    vector.print %11 : vector<1xi1>
    vector.print %10 : vector<1xi32>
    vector.print %110 : vector<1xi1>
    vector.print %111 : vector<1xi1>
    vector.print %42 : vector<10x1x10xf32>
    vector.print %172 : vector<10x1x10xf32>
    vector.print %176 : vector<1xi1>
    vector.print %43 : vector<i64>
    vector.print %181 : vector<10x1x10xi32>
    vector.print %42 : vector<10x1x10xf32>
    vector.print %246 : vector<10x1x10xf32>
    vector.print %247 : vector<1xi1>
    vector.print %5 : vector<1xf16>
    vector.print %248 : vector<10x1x10xi64>
    vector.print %257 : vector<1xi1>
    vector.print %258 : vector<10x1x10xi64>
    vector.print %41 : vector<10x6xi1>
    vector.print %10 : vector<1xi32>
    vector.print %260 : vector<10xf16>
    llvm.call @printI64(%30) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%30) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    %261 = llvm.bitcast %14 : f16 to i16
    llvm.call @printF16(%261) : (i16) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%31) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    %262 = llvm.bitcast %15 : f16 to i16
    llvm.call @printF16(%262) : (i16) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%16) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%31) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    %263 = llvm.bitcast %17 : f16 to i16
    llvm.call @printF16(%263) : (i16) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%18) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%19) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%32) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    %264 = llvm.bitcast %20 : f16 to i16
    llvm.call @printF16(%264) : (i16) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%21) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%33) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%30) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%31) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%31) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%30) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printF32(%13) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%extracted) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%extracted_0) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printF32(%13) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    %265 = bufferization.alloc_tensor() : tensor<15x2x5xf16>
    %266 = bufferization.to_memref %265 : memref<15x2x5xf16>
    %cast = memref.cast %266 : memref<15x2x5xf16> to memref<?x?x?xf16>
    %267 = builtin.unrealized_conversion_cast %cast : memref<?x?x?xf16> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.return %267 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}

