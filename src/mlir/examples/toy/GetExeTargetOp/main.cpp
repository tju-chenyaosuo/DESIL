#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "Debug.h"
#include "MutationUtil.h"
#include "Traverse.h"

using namespace mlir;

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
static cl::opt<int> inconsistentIndex(cl::Positional,
                                      cl::desc("<op count to be remove>"),
                                      cl::init(1), cl::value_desc("integer"));
// static cl::opt<std::string> outputFilename(cl::Positional,
//                                            cl::desc("<output mlir file>"),
//                                            cl::init("-"),
//                                            cl::value_desc("filename"));

unsigned lineNumber() {
  std::ifstream file(inputFilename);
  if (!file.is_open()) {
    std::cerr << "Unable to open file: " << inputFilename << std::endl;
    return 1;
  }
  std::string line;
  unsigned lineNumber = 0;
  while (std::getline(file, line)) {
    lineNumber++;
    // Process the line as needed
    std::cout << "Line " << lineNumber << ": " << line << std::endl;
  }
  file.close();
  return lineNumber;
}

// Usage: /data/src/mlirsmith-dev/build/bin/mutator --operation-prob=0.1
// --operand-prob=0.5 test.mlir 1.mlir
int main(int argc, char **argv) {
  // regist the options
  cl::ParseCommandLineOptions(argc, argv, "mlir mutator\n");
  llvm::InitLLVM(argc, argv);

  // Load the mlir file
  MLIRContext context;
  registerAllDialects(context);
  context.allowUnregisteredDialects();
  OwningOpRef<ModuleOp> module;
  llvm::SourceMgr sourceMgr;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "[main] Error can't load file " << inputFilename << "\n";
    return 3;
  }
#ifdef DEBUG
  llvm::errs() << "[main] Load file " << inputFilename << " success!\n";
#endif

  // auto firstOp = module.getOps();
  // llvm::errs() << "[main] " << firstOp.getName() << "\n";

  // unsigned fileLineNumber = lineNumber();
  // unsigned mutPos = rollIdx(fileLineNumber);

  srand(time(0));
  Operation *op = module.get();
  // MutationParser mp = MutationParser(mutPos);
  // MutationBlock *mb = new MutationBlock();
  // mp.printOperation(op, mb);

  // traverse(op);

  op->walk([](func::FuncOp funcOp) {
    unsigned numArgs = funcOp.getNumArguments();
    if (numArgs == 0) {
      llvm::SmallVector<Operation *> ops = llvm::SmallVector<Operation *>();
      for (Region &region : funcOp.getOperation()->getRegions()) {
        for (Block &block : region.getBlocks()) {
          for (Operation &op : block.getOperations()) {
            ops.push_back(&op);
          }
        }
      }

      mlir::Operation* targetOp = ops[ops.size()-4];
      while (true) {
        std::string opName = getOperationName(targetOp);
        if (opName == "func.call" || opName == "tensor.collapse_shape" ||
            opName == "memref.collapse_shape") 
              targetOp = targetOp->getOperand(0).getDefiningOp();
        else
          break;
      }

      if (targetOp->getNumResults() != 0) 
        llvm::outs() << getOperationName(targetOp) << "#" << getValueTypeStr(targetOp->getResult(0));
      else
        llvm::outs() << getOperationName(targetOp) << "#None";
    }
  });

  // std::error_code error;
  // llvm::raw_fd_ostream output(outputFilename, error);
  // if (error) {
  //   llvm::errs() << "Error opening file for writing: " << error.message()
  //                << "\n";
  //   return 1;
  // }
  // module->print(output);

  return 0;
}
