// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNAverageKernel.h"

#import "MNContext.h"
#import <algorithm>

NS_ASSUME_NONNULL_BEGIN

@interface MNAverageKernel ()

@property (strong, nonatomic) id<MTLBuffer> inputBuffer;

@property (nonatomic) uint elementsNumber;

@end

@implementation MNAverageKernel

- (instancetype)initWithContext:(MNContext *)context {
  return [super initWithContext:context functionName:@"averageGroupReduce"];
}

- (void)configureCommandEncoder:(id<MTLComputeCommandEncoder>)encoder {
  [encoder setBuffer:self.inputBuffer offset:0 atIndex:0];
  [encoder setBuffer:self.result offset:0 atIndex:1];
  [encoder setBytes:&_elementsNumber length:sizeof(uint) atIndex:2];
  [encoder setThreadgroupMemoryLength:self.threadGroupSize.width * sizeof(float) atIndex:0];
}

- (void)setInputBuffer:(id<MTLBuffer>)inputBuffer withElementsCount:(uint)count {
  _inputBuffer = inputBuffer;
  self.elementsNumber = (uint)count;
}

- (MTLSize)threadGroupSize {
  return MTLSizeMake(512, 1, 1);
}

- (MTLSize)threadGroupsCountWithGroupSize:(MTLSize)threadGroupSize {
  /// Dividing by 32 is the most important optimization.
  return MTLSizeMake(std::max((NSUInteger)1, self.elementsNumber / threadGroupSize.width / 32), 1, 1);
}

+ (void)testWithContext:(MNContext *)context {
  uint bufferElements = 1024 * 1024;
  MNAverageKernel *kernel = [[MNAverageKernel alloc] initWithContext:context];
  id<MTLBuffer> buffer1 = [context.device newBufferWithLength:bufferElements * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
  id<MTLBuffer> buffer2 = [context.device newBufferWithLength:bufferElements * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
  float *bufferPtr = (float *)[buffer1 contents];
  for (size_t i = 0; i < bufferElements; ++i) {
    bufferPtr[i] = 1;
  }

  id<MTLCommandBuffer> commandBuffer = [context.queue commandBuffer];

  NSTimeInterval time = 0;
  NSDate *methodStart = [NSDate date];
  int iterationsNumber = 10;
  uint count;
  for (int i = 0; i < iterationsNumber; ++i) {
    count = bufferElements;
    while (count > 1) {
      [kernel setInputBuffer:buffer1 withElementsCount:count];
      kernel.result = buffer2;
      buffer2 = buffer1;
      buffer1 = kernel.result;

      count = (uint)[kernel threadGroupsCountWithGroupSize:kernel.threadGroupSize].width;
      [kernel enqueTo:commandBuffer];
    }
  }

  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  NSDate *methodFinish = [NSDate date];
  NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
  time += executionTime;
  NSLog(@"Took %g sec", time / iterationsNumber);
  if ([commandBuffer error]) {
    NSLog(@"error");
  } else {
    float *inputPtr = (float *)((uint *)kernel.inputBuffer.contents);
    float *resultPtr = (float *)((uint *)kernel.result.contents);
    for (size_t i = 0; i < count; ++i) {
//      if (resultPtr[i] != 512) {
        NSLog(@"aaa %f", resultPtr[i]);
        NSLog(@"bbb %f", inputPtr[i]);
//      }
    }
    NSLog(@"aaa %f", resultPtr[count]);
  }
}

@end

NS_ASSUME_NONNULL_END
