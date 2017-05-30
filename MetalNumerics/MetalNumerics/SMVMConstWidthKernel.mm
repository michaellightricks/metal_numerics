// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "SMVMConstWidthKernel.h"

#import "MNConstantWidthSparseMatrix.h"

NS_ASSUME_NONNULL_BEGIN

@implementation SMVMConstWidthKernel

- (instancetype)initWithContext:(MNContext *)context {
  return [super initWithContext:context functionName:@"SMVM_CONST_WIDTH_WARP"];
}

- (void)configureCommandEncoder:(id<MTLComputeCommandEncoder>)encoder {
  [encoder setBuffer:self.matrix.buffer offset:0 atIndex:0];
  [encoder setBuffer:self.matrix.columnIndices offset:0 atIndex:1];
  [encoder setBuffer:self.vector offset:0 atIndex:2];
  [encoder setBuffer:self.result offset:0 atIndex:3];
}

- (MTLSize)threadGroupSize {
  return MTLSizeMake(32, 1, 1);
}

- (MTLSize)threadGroupsCount:(MTLSize)threadGroupSize {
  return MTLSizeMake(self.matrix.rowsNumber / 32, 1, 1);
}

@end

NS_ASSUME_NONNULL_END
