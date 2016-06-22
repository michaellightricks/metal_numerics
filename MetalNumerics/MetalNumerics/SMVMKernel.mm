//
//  SMVMKernel.m
//  MetalNumerics
//
//  Created by Michael Kupchick on 22/06/2016.
//  Copyright © 2016 Michael Kupchick. All rights reserved.
//

#import "SMVMKernel.h"
#import <Metal/Metal.h>

#import "MNCRSSparseMatrix.h"

@implementation SMVMKernel

- (instancetype)initWithContext:(MNContext *)context {
  if (self = [super initWithContext:context functionName:@"SMVM"]) {

  }

  return self;
}

- (void)configureCommandEncoder:(id<MTLComputeCommandEncoder>)encoder {
  [encoder setBuffer:self.matrix.buffer offset:0 atIndex:0];
  [encoder setBuffer:self.matrix.nonZeroElementsNumberInRow offset:0 atIndex:1];
  [encoder setBuffer:self.matrix.columnIndices offset:0 atIndex:2];
  [encoder setBuffer:self.vector offset:0 atIndex:3];
  [encoder setBuffer:self.result offset:0 atIndex:4];
}

@end
