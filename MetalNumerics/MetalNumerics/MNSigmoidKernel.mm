// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNSigmoidKernel.h"

#import <MetalKit/MetalKit.h>
#import <UIKit/UIKit.h>

#import <algorithm>

#import "MNContext.h"

NS_ASSUME_NONNULL_BEGIN

@interface MNSigmoidKernel ()

@property (strong, nonatomic) id<MTLBuffer> inputBuffer;

@property (readwrite, nonatomic) uint elementsNumber;

@end

@implementation MNSigmoidKernel

- (instancetype)initWithContext:(MNContext *)context {
  return [super initWithContext:context functionName:@"sigmoid"];
}

- (void)configureCommandEncoder:(id<MTLComputeCommandEncoder>)encoder {
  [encoder setBuffer:self.inputBuffer offset:0 atIndex:0];
  [encoder setBuffer:self.result offset:0 atIndex:1];
  [encoder setBytes:&_elementsNumber length:sizeof(uint) atIndex:2];
  //[encoder setThreadgroupMemoryLength:self.threadGroupSize.width * sizeof(float) atIndex:0];
}

- (void)setInputBuffer:(id<MTLBuffer>)inputBuffer withElementsCount:(uint)count {
  _inputBuffer = inputBuffer;
  self.elementsNumber = (uint)count;
}

- (MTLSize)threadGroupSize {
  return MTLSizeMake(512, /*std::max((uint)32, std::min(self.elementsNumber, (uint)512)),*/ 1, 1);
}

- (MTLSize)threadGroupsCountWithGroupSize:(MTLSize)threadGroupSize {
  /// Dividing by 64 is the most important optimization. The devisor may be dependent on the input size
  /// but from experimentation 64 gives good result on all the sizes.
  return MTLSizeMake(std::max((NSUInteger)1, self.elementsNumber / (threadGroupSize.width * 16)), 1, 1);
  //return MTLSizeMake(1, 1, 1);
}

+ (void)testWithContext:(MNContext *)context {
  uint bufferElements = 8192 * 8192;
  MNSigmoidKernel *kernel = [[MNSigmoidKernel alloc] initWithContext:context];
  id<MTLBuffer> buffer1 = [context.device newBufferWithLength:bufferElements * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
  float *bufferPtr = (float *)[buffer1 contents];
  for (size_t i = 0; i < bufferElements; ++i) {
    bufferPtr[i] = 1;
  }
  [kernel setInputBuffer:buffer1 withElementsCount:bufferElements];

  id<MTLBuffer> result = [context.device newBufferWithLength:bufferElements * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
  kernel.result = result;

  id<MTLCommandBuffer> commandBuffer = [context.queue commandBuffer];

  //NSDate *methodStart = [NSDate date];
  int iterationsNumber = 50;

  for (int i = 0; i < iterationsNumber; ++i) {
    [kernel enqueTo:commandBuffer];
  }

  NSDate *methodStart = [NSDate date];

  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  NSDate *methodFinish = [NSDate date];
  NSTimeInterval time = [methodFinish timeIntervalSinceDate:methodStart];
  NSLog(@"MNSigmoidKernel Took %g sec", time / iterationsNumber);
  if ([commandBuffer error]) {
    NSLog(@"error");
  }
}

@end

NS_ASSUME_NONNULL_END
