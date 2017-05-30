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
  id<MTLCommandBuffer> commandBuffer = [self enque];
  [commandBuffer waitUntilCompleted];
}

- (id<MTLCommandBuffer>)enque {
  id<MTLCommandBuffer> commandBuffer = [self.context.queue commandBuffer];
  [self enqueTo:commandBuffer];
  [commandBuffer commit];
  return commandBuffer;
}

- (void)enqueTo:(id<MTLCommandBuffer>)commandBuffer {
  id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

  [commandEncoder setComputePipelineState:self.pipeline];

  [self configureCommandEncoder:commandEncoder];

  MTLSize threadgroupSize = [self threadGroupSize];
  MTLSize threadgroups = [self threadGroupsCountWithGroupSize:threadgroupSize];

  [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
  [commandEncoder endEncoding];
}

- (void)configureCommandEncoder:(id<MTLComputeCommandEncoder>)encoder {

}

- (MTLSize)threadGroupSize {
  return MTLSizeMake(0, 0, 0);
}

- (MTLSize)threadGroupsCountWithGroupSize:(MTLSize)threadGroupSize {
  return MTLSizeMake(0, 0, 0);
}

@end
