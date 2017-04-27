// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNTest.h"

#import <Metal/Metal.h>

#import "MNContext.h"
#import "MNConstantWidthSparseMatrix.h"
#import "SMVMConstWidthKernel.h"

NS_ASSUME_NONNULL_BEGIN

static const ushort kWidth = 32;

const ushort kRowsNumber = 64 * 10;

@interface MNTest ()

@property (readonly, nonatomic) MNContext *context;

@property (readonly, nonatomic) MNConstantWidthSparseMatrix *matrix;

@property (readonly, nonatomic) id<MTLBuffer> vectorBuffer;

@property (readonly, nonatomic) id<MTLBuffer> resultBuffer;

@end

@implementation MNTest

- (instancetype)initWithContext:(MNContext *)context {
  if (self = [super init]) {
    _context = context;
    _matrix = [[MNConstantWidthSparseMatrix alloc] initWithContext:self.context
                                                      numberOfRows:kRowsNumber
                                                             width:kWidth];
    for (int row = 0; row < kRowsNumber; ++row) {
      for (int i = 0; i < kWidth; ++i) {
        //float element = rand() / (float)RAND_MAX;
        ushort column = rand() / (float)RAND_MAX * kRowsNumber;
        [self.matrix insertElement:2.0 row:row index:i column:column];
      }
    }

    _vectorBuffer = [self.context.device newBufferWithLength:sizeof(float16_t) * kRowsNumber
                                                    options:MTLResourceStorageModeShared];
    float16_t *vectorPtr = (float16_t *)self.vectorBuffer.contents;
    for (int i = 0; i < kRowsNumber; ++i) {
      vectorPtr[i] = 1;
    }

    _resultBuffer = [self.context.device newBufferWithLength:sizeof(float16_t) * kRowsNumber
                                                     options:MTLResourceStorageModeShared];
  }

  return self;
}


- (void)run {
  id<MTLCommandBuffer> commandBuffer = [self.context.queue commandBuffer];
  SMVMConstWidthKernel *kernel = [[SMVMConstWidthKernel alloc] initWithContext:self.context];
  kernel.matrix = self.matrix;
  kernel.vector = self.vectorBuffer;
  kernel.result = self.resultBuffer;

  NSTimeInterval time = 0;
  NSDate *methodStart = [NSDate date];
  for (int i = 0; i < 1; ++i) {
    [kernel enqueTo:commandBuffer];
  }

  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  if ([commandBuffer error]) {
    NSLog(@"error");
  }
  NSDate *methodFinish = [NSDate date];
  NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
  time += executionTime;

  NSLog(@"Took %g", time / 10);

  float *resultPtr = (float *)kernel.result.contents;
  for (int i = 0; i < kRowsNumber; i += 10) {
    NSLog(@"%g", resultPtr[i]);
  }
}

@end

NS_ASSUME_NONNULL_END
