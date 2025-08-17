import os
import re

# *****这仨是要改的*****

# llvm项目文件夹下mlir_opt
mlir_opt = '/data/llvm-project/build/bin/mlir-opt'
# mlirsmith的项目文件夹
mlirsmith_base = '/data/mlirsmith-dev'
# cmake路径
cmake = '/data/cmake-3.27.0-rc2-linux-x86_64/bin/cmake'
# cmake = 'cmake'


# 这个文件夹的位置
# '/data/mlirsmith-dev/mlir/examples/toy/Ch5_dynamicLowering/pytest'
pytest_base = mlirsmith_base + '/mlir/examples/toy/Ch5_dynamicLowering/pytest'
# mlirsmith的build文件夹
# /data/mlirsmith-dev/build
mlirsmith_build = mlirsmith_base + '/build'
# smith的toyc-ch5
# /data/mlirsmith-dev/build/bin/toyc-ch5
toyc_ch5 = mlirsmith_build + '/bin/toyc-ch5'
# template.mlir的位置
# /data/mlirsmith-dev/mlir/examples/toy/Ch5_dynamicLowering/pytest/template.mlir
template_mlir = pytest_base + '/template.mlir'
# template直接生成的mlir
# /data/files/generated.mlir
generated_mlir = pytest_base + '/generated.mlir'

# 因为smith生成时必须需要i64的integerAttr，但是llvm的opt需要i32的，应该是smith版本落后的缘故，这里为了方便直接replace全部i64
generated_changei64_to_i32_mlir = pytest_base + '/changei64.mlir'

afteropt1 = pytest_base + '/afteropt1.mlir'
afteropt2 = pytest_base + '/afteropt2.mlir'

opts1 = [
    '--canonicalize',
    '--lower-affine',
    '--convert-vector-to-llvm',
    '--tosa-to-tensor',
    '--tosa-to-arith',
    '--tosa-validate',
    '--convert-tensor-to-linalg',
    '--empty-tensor-to-alloc-tensor',
    '--finalize-memref-to-llvm',
    '--func-bufferize',
    '--convert-func-to-llvm',
    '--convert-index-to-llvm',
    '--normalize-memrefs',
    '--convert-linalg-to-affine-loops',
    '--lower-affine',
    '--convert-vector-to-llvm',
    '--finalize-memref-to-llvm',
    '--canonicalize',
]
opts2 = [
    '--one-shot-bufferize',
    '--convert-bufferization-to-memref',
    '--finalize-memref-to-llvm',
    '--canonicalize',
]

# /data/mlirsmith-dev/build/bin/mlir-tblgen -gen-op-defs /data/mlirsmith-dev/mlir/include/mlir/Dialect/Tosa/IR/TosaOps.td -I /data/mlirsmith-dev/mlir/include  > /data/files/tblgen.txt

def build():
    os.system(f'{cmake} --build {mlirsmith_build} --target check-mlir')

def generate():
    os.system(f'{toyc_ch5} --emit=mlir-affine {template_mlir} 2> {generated_mlir}')

def changei64():
    with open(generated_mlir, 'r+') as f:
        with open(generated_changei64_to_i32_mlir, 'w+') as f2:
            fileContext = f.read()
            # 算一下有多少个i64
            i64cnt = re.findall(r'i64}', fileContext).__len__()
            # 挨个替换了
            for i in range(i64cnt):
                fileContext = fileContext.replace("i64}", "i32}")
            # 然后写到另一个文件里
            f2.write(fileContext)
            
def opt1():
    cmd = mlir_opt
    for i,x in enumerate(opts1):
        cmd += ' ' + x
    cmd += ' ' + generated_changei64_to_i32_mlir
    with os.popen(cmd) as p:
        # '/data/files/output/{}.mlir'.format(i)
        f = open(afteropt1, 'w+')
        f.write(p.read())
        # print('-------------------------------------------------')
        f.close()

def opt2():
    cmd = mlir_opt
    for i,x in enumerate(opts2):
        cmd += ' ' + x
    cmd += ' ' + afteropt1
    with os.popen(cmd) as p:
        # '/data/files/output/{}.mlir'.format(i)
        f = open(afteropt2, 'w+')
        f.write(p.read())
        # print('-------------------------------------------------')
        f.close()


if __name__ == '__main__':
    # 配置好路径这些函数应该是都能动的(??)
    build()
    generate()
    changei64()
    # 两次opt，50%概率low到llvm
    opt1()
    opt2()