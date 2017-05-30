// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNTest.h"

#import <Metal/Metal.h>

#import "MNContext.h"
#import "MNConstantWidthSparseMatrix.h"
#import "SMVMConstWidthKernel.h"

NS_ASSUME_NONNULL_BEGIN

static const ushort kWidth = 32;

const ushort kRowsNumber = 64 * 500;

@interface MNTest ()

@property (readonly, nonatomic) MNContext *context;

@property (readonly, nonatomic) MNConstantWidthSparseMatrix *matrix;

@property (readonly, nonatomic) id<MTLBuffer> vectorBuffer;

@property (readonly, nonatomic) id<MTLBuffer> resultBuffer;

@property (readonly, nonatomic) SMVMConstWidthKernel *kernel;

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
    float16_t *resultPtr = (float16_t *)self.resultBuffer.contents;
    for (int i = 0; i < kRowsNumber; ++i) {
      resultPtr[i] = 0;
    }

    _kernel = [[SMVMConstWidthKernel alloc] initWithContext:self.context];
    self.kernel.matrix = self.matrix;
    self.kernel.vector = self.vectorBuffer;
    self.kernel.result = self.resultBuffer;
  }

  return self;
}

- (void)run {
  id<MTLCommandBuffer> commandBuffer = [self.context.queue commandBuffer];

  NSTimeInterval time = 0;
  NSDate *methodStart = [NSDate date];
  int count = 100;
  for (int i = 0; i < count; ++i) {
    [self.kernel enqueTo:commandBuffer];
  }

  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  NSDate *methodFinish = [NSDate date];
  NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
  time += executionTime;
  NSLog(@"Took %g", time / count);
  if ([commandBuffer error]) {
    NSLog(@"error");
  }

//  float16_t *resultPtr = (float16_t *)self.kernel.result.contents;
//  for (int i = 0; i < kRowsNumber; i += 10) {
//    NSLog(@"%g", (double)resultPtr[i]);
//  }
}

@end

NS_ASSUME_NONNULL_END
