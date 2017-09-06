// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNCovarianceKernel.h"

#import <MetalKit/MetalKit.h>
#import <UIKit/UIKit.h>


#import "MNContext.h"

NS_ASSUME_NONNULL_BEGIN

@implementation MNCovarianceKernel

- (instancetype)initWithContext:(MNContext *)context {
  return [super initWithContext:context functionName:@"covarianceKernel"];
}

- (void)configureCommandEncoder:(id<MTLComputeCommandEncoder>)encoder {
  [encoder setTexture:self.inputTexture atIndex:0];
  [encoder setBuffer:self.outputBuffer offset:0 atIndex:0];
}

- (MTLSize)threadGroupSize {
  return MTLSizeMake(1, 1, 0);
}

- (MTLSize)threadGroupsCountWithGroupSize:(MTLSize)threadGroupSize {
  return MTLSizeMake(1, 1, 0);
//  return MTLSizeMake(self.inputTexture.width / threadGroupSize.width,
//                     self.inputTexture.height / threadGroupSize.height, 0);
}

+ (void)testWithContext:(MNContext *)context {
  MNCovarianceKernel *kernel = [[MNCovarianceKernel alloc] initWithContext:context];
  UIImage *input = [UIImage imageNamed:@"lena.png"];
  kernel.inputTexture = [MNCovarianceKernel textureFromImage:input device:context.device];
  NSUInteger outputBytes = 6 * input.size.width * input.size.height * sizeof(float);
  kernel.outputBuffer = [context.device newBufferWithLength:outputBytes
                                                    options:MTLResourceStorageModeShared];

  id<MTLCommandBuffer> buffer = [context.queue commandBuffer];
  int count = 1;
  for (int i = 0; i < count; ++i) {
    [kernel enqueTo:buffer];
  }

  NSDate *methodStart = [NSDate date];
  [buffer commit];
  [buffer waitUntilCompleted];

  NSDate *methodEnd = [NSDate date];
  NSTimeInterval executionTime = [methodEnd timeIntervalSinceDate:methodStart];

  NSLog(@"covariance kernel took %g ms", executionTime * 1000);

}

+ (id<MTLTexture>)textureFromImage:(UIImage *)image device:(id<MTLDevice>)device {
  NSError *error;
  id<MTLTexture> texture = [[[MTKTextureLoader alloc] initWithDevice:device]
                            newTextureWithCGImage:image.CGImage options:nil error:&error];

  if (error) {
    NSLog(@"errror loading input texture");
  }
  return texture;
}

@end

NS_ASSUME_NONNULL_END
