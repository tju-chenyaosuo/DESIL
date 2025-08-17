/**
 * This file aims to reduce the operation of the execute crash MLIR file.
 * Specifically, the file conducts the followering things:
 * (1) Remove the last n top-level operations.
 * (2) Add new vector.print to the end of the function, to ensure the output of
 * the program.
 */

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
static cl::opt<int> removeCount(cl::Positional,
                                cl::desc("<op count to be remove>"),
                                cl::init(1), cl::value_desc("integer"));
static cl::opt<std::string> outputFilename(cl::Positional,
                                           cl::desc("<output mlir file>"),
                                           cl::init("-"),
                                           cl::value_desc("filename"));

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

      // 1. remove all of the vector.print in the file
      llvm::SmallVector<Operation *> ops2remove =
          llvm::SmallVector<Operation *>();
      for (int64_t i = 0; i < removeCount && ops.size() - 2 - i >= 0; ++i)
        ops[ops.size() - 2 - i]->erase();
      // 2. add new vector.print
      mlir::Operation *returnOp = ops[ops.size() - 1];
      mlir::OpBuilder builder(returnOp->getBlock(),
                              returnOp->getBlock()->begin());
      builder.setInsertionPoint(returnOp);
      mlir::Location loc = returnOp->getLoc();
      mlir::Value constantVal =
          builder
              .create<mlir::arith::ConstantOp>(
                  loc, builder.getIntegerAttr(builder.getI1Type(), 1))
              .getOperation()
              ->getResult(0);
      mlir::Operation *printOp =
          builder.create<mlir::vector::PrintOp>(loc, constantVal);
    }
  });

  std::error_code error;
  llvm::raw_fd_ostream output(outputFilename, error);
  if (error) {
    llvm::errs() << "Error opening file for writing: " << error.message()
                 << "\n";
    return 1;
  }
  module->print(output);

  return 0;
}
