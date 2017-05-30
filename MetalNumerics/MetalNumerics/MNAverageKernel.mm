// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNAverageKernel.h"

NS_ASSUME_NONNULL_BEGIN

@implementation MNAverageKernel

- (instancetype)initWithContext:(MNContext *)context {
  return [super initWithContext:context functionName:@"SMVM_CONST_WIDTH_WARP"];
}

- (void)configureCommandEncoder:(id<MTLComputeCommandEncoder>)encoder {
  [encoder setBuffer:self.buffer offset:0 atIndex:0];
  [encoder setBuffer:self.result offset:0 atIndex:1];
}

- (MTLSize)threadGroupSize {
  return MTLSizeMake(512, 1, 1);
}

- (MTLSize)threadGroupsCount:(MTLSize)threadGroupSize {
  return MTLSizeMake(self.buffer.length / sizeof(float) , 1, 1);
}

@end

NS_ASSUME_NONNULL_END
