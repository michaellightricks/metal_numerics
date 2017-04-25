// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNTest.h"

#import "MNContext.h"
#import "MNConstantWidthSparseMatrix.h"
#import "SMVMConstWidthKernel.h"

NS_ASSUME_NONNULL_BEGIN

@interface MNTest ()

@property (readonly, nonatomic) MNContext *context;

@end

@implementation MNTest

- (instancetype)initWithContext:(MNContext *)context {
  if (self = [super init]) {
    _context = context;
  }

  return self;
}

const ushort kRowsNumber = 64 * 100;

- (void)run {
  MNConstantWidthSparseMatrix *matrix =
      [[MNConstantWidthSparseMatrix alloc] initWithContext:self.context numberOfRows:kRowsNumber
                                                     width:25];

  for (int row = 0; row < kRowsNumber; ++row) {
    for (int i = 0; i < 25; ++i) {
      float element = rand() / (float)RAND_MAX;
      ushort column = rand() / (float)RAND_MAX * kRowsNumber;
      [matrix insertElement:2.0 row:row index:i column:column];
    }
  }

  SMVMConstWidthKernel *kernel = [[SMVMConstWidthKernel alloc] initWithContext:self.context];
  kernel.matrix = matrix;
  kernel.vector = [self.context.device newBufferWithLength:sizeof(float) * kRowsNumber
                                                   options:MTLResourceStorageModeShared];
  float *vectorPtr = (float *)kernel.vector.contents;
  for (int i = 0; i < kRowsNumber; ++i) {
    vectorPtr[i] = 1;
  }
  kernel.result = [self.context.device newBufferWithLength:sizeof(float) * kRowsNumber
                                                   options:MTLResourceStorageModeShared];
  NSTimeInterval time = 0;
  for (int i = 0; i < 10; ++i) {
    NSDate *methodStart = [NSDate date];
    [kernel run];
    NSDate *methodFinish = [NSDate date];
    NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
    time += executionTime;
  }

  NSLog(@"Took %g", time / 10);

  float *resultPtr = (float *)kernel.result.contents;
//  for (int i = 0; i < kRowsNumber; ++i) {
//    NSLog(@"%g", resultPtr[i]);
//  }
}

@end

NS_ASSUME_NONNULL_END
