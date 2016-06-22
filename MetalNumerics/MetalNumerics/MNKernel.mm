//
//  MNKernel.m
//  MetalNumerics
//
//  Created by Michael Kupchick on 22/06/2016.
//  Copyright © 2016 Michael Kupchick. All rights reserved.
//

#import "MNKernel.h"

#import <Metal/Metal.h>

#import "MNContext.h"

@interface MNKernel()



@end

@implementation MNKernel

- (instancetype)initWithContext:(MNContext *)context functionName:(NSString *)name {
  if (self = [super init]) {
    _context = context;

    _kernelFunction = [_context.library newFunctionWithName:name];
    NSError *error = nil;
    _pipeline = [_context.device newComputePipelineStateWithFunction:_kernelFunction
                                                               error:&error];
    if (!_pipeline)
    {
      NSLog(@"Error occurred when building compute pipeline for function %@", name);
      NSLog(@"%@", error);
      return nil;
    }
  }

  return self;
}

- (void)run {
  id<MTLCommandBuffer> commandBuffer = [self.context.queue commandBuffer];

  id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

  [commandEncoder setComputePipelineState:self.pipeline];

  [self configureCommandEncoder:commandEncoder];

  MTLSize threadgroupSize = [self threadGroupSize];
  MTLSize threadgroups = [self threadGroupsCount:threadgroupSize];

  [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
  [commandEncoder endEncoding];

  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}

- (void)configureCommandEncoder:(id<MTLComputeCommandEncoder>)encoder {

}

- (MTLSize)threadGroupSize {
  return MTLSizeMake(0, 0, 0);
}

- (MTLSize)threadGroupsCount:(MTLSize)threadGroupSize {
  return MTLSizeMake(0, 0, 0);
}

@end
