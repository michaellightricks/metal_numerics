// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNAverageKernel.h"

#import "MNContext.h"
#import <algorithm>
#import <Accelerate/Accelerate.h>

NS_ASSUME_NONNULL_BEGIN

@interface MNAverageKernel ()

@property (strong, nonatomic) id<MTLBuffer> inputBuffer;

@property (nonatomic) uint elementsNumber;

@end

@implementation MNAverageKernel

- (instancetype)initWithContext:(MNContext *)context {
  return [super initWithContext:context functionName:@"averageGroupReduce2"];
}

- (void)configureCommandEncoder:(id<MTLComputeCommandEncoder>)encoder {
  [encoder setBuffer:self.inputBuffer offset:0 atIndex:0];
  //[encoder setBuffer:self.result offset:0 atIndex:1];
  [encoder setBytes:&_elementsNumber length:sizeof(uint) atIndex:1];
  [encoder setThreadgroupMemoryLength:self.threadGroupSize.width * sizeof(float) atIndex:0];
}

- (void)setInputBuffer:(id<MTLBuffer>)inputBuffer withElementsCount:(uint)count {
  _inputBuffer = inputBuffer;
  self.elementsNumber = (uint)count;
}

- (MTLSize)threadGroupSize {
  return MTLSizeMake(512, /*std::max((uint)32, std::min(self.elementsNumber, (uint)512)),*/ 1, 1);
}

- (MTLSize)threadGroupsCountWithGroupSize:(MTLSize)threadGroupSize {
  /// Dividing by 32 is the most important optimization. The devisor may be dependent on the input size
  /// but from experimentation 64 gives good result on all the sizes.
  return MTLSizeMake(std::max((NSUInteger)1, self.elementsNumber / (threadGroupSize.width * 64)), 1, 1);
}

+ (void)testWithContext:(MNContext *)context {
  uint bufferElements = 8192 * 8192;
  MNAverageKernel *kernel = [[MNAverageKernel alloc] initWithContext:context];
  id<MTLBuffer> buffer1 = [context.device newBufferWithLength:bufferElements * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
  float *bufferPtr = (float *)[buffer1 contents];
  for (size_t i = 0; i < bufferElements; ++i) {
    bufferPtr[i] = 1;
  }

  [self testVDSPWithBuffer:bufferPtr elementsNumber:bufferElements];

  id<MTLCommandBuffer> commandBuffer = [context.queue commandBuffer];

  NSTimeInterval time = 0;
  NSDate *methodStart = [NSDate date];
  int iterationsNumber = 50;
  uint count = 0;
  for (int i = 0; i < iterationsNumber; ++i) {
    count = bufferElements;
    while (count > 1) {
      [kernel setInputBuffer:buffer1 withElementsCount:count];
      kernel.result = buffer1;
      count = (uint)[kernel threadGroupsCountWithGroupSize:kernel.threadGroupSize].width;
      [kernel enqueTo:commandBuffer];
    }
  }

  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  NSDate *methodFinish = [NSDate date];
  NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
  time += executionTime;
  NSLog(@"MNAverageKernel Took %g sec", time / iterationsNumber);
  if ([commandBuffer error]) {
    NSLog(@"error");
  } else {
    float *resultPtr = (float *)((uint *)kernel.result.contents);
    for (size_t i = 0; i < count; ++i) {
        NSLog(@"aaa %f", resultPtr[i]);
    }
    NSLog(@"aaa %f", resultPtr[count]);
  }
}

+ (void)testVDSPWithBuffer:(float *)bufferPtr elementsNumber:(uint)elementsNumber {
  size_t iterationsNumber = 10;
  float rrrr = 0.0;

  NSTimeInterval time = 0;
  NSDate *methodStart = [NSDate date];
  for (size_t i = 0; i < iterationsNumber; ++i) {
    vDSP_sve(bufferPtr, 1, &rrrr, elementsNumber);
  }
  NSDate *methodFinish = [NSDate date];
  NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
  time += executionTime;
  NSLog(@"vDSP_sve Took %g sec", time / iterationsNumber);
  NSLog(@"vDSP sum %f", rrrr);

}

@end

NS_ASSUME_NONNULL_END
