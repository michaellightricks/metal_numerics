// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNCovarianceKernel.h"

#import <MetalKit/MetalKit.h>
#import <UIKit/UIKit.h>

#import <simd/simd.h>
#import <opencv2/opencv.hpp>

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
  return MTLSizeMake(32, 16, 1);
}

- (MTLSize)threadGroupsCountWithGroupSize:(MTLSize)threadGroupSize {
  //return MTLSizeMake(1, 1, 1);
  return MTLSizeMake(self.inputTexture.width / threadGroupSize.width,
                     self.inputTexture.height / threadGroupSize.height, 1);
}

+ (void)testWithContext:(MNContext *)context {
  MNCovarianceKernel *kernel = [[MNCovarianceKernel alloc] initWithContext:context];
  UIImage *input = [UIImage imageNamed:@"lena.png"];
  kernel.inputTexture = [MNCovarianceKernel textureFromImage:input device:context.device];
  NSUInteger outputBytes = 6 * input.size.width * input.size.height * sizeof(float);
  kernel.outputBuffer = [context.device newBufferWithLength:outputBytes
                                                    options:MTLResourceStorageModeShared];

  id<MTLCommandBuffer> buffer = [context.queue commandBuffer];
  int count = 50;
  for (int i = 0; i < count; ++i) {
    [kernel enqueTo:buffer];
  }

  NSDate *methodStart = [NSDate date];
  [buffer commit];
  [buffer waitUntilCompleted];

  NSDate *methodEnd = [NSDate date];
  NSTimeInterval executionTime = [methodEnd timeIntervalSinceDate:methodStart];

  NSLog(@"covariance kernel took %g ms", executionTime * 1000 / count);

  float *outputPtr = (float *)kernel.outputBuffer.contents;
  cv::Mat mat = [self cvMatFromUIImage:input];
  uint8_t window[9] = {0};
  for (int row = 1; row < input.size.height - 1; ++row) {
    for (int col = 1; col < input.size.width - 1; ++col) {
      cv::Mat window = mat(cv::Rect(row - 1, col - 1, 3, 3));
      cv::Mat covariance = [self covarianceWithMat:window];
    }
  }
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

+ (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
  CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
  CGFloat cols = image.size.width;
  CGFloat rows = image.size.height;

  cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)

  CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                  cols,                       // Width of bitmap
                                                  rows,                       // Height of bitmap
                                                  8,                          // Bits per component
                                                  cvMat.step[0],              // Bytes per row
                                                  colorSpace,                 // Colorspace
                                                  kCGImageAlphaNoneSkipLast |
                                                  kCGBitmapByteOrderDefault); // Bitmap info flags

  CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
  CGContextRelease(contextRef);

  return cvMat;
}

+ (cv::Mat)covarianceWithMat:(const cv::Mat &)mat {
  cv::Mat4f floatMat;
  mat.convertTo(floatMat, CV_32FC4, 1.0 / 255.0, 0.0);
  cv::Scalar mean = cv::mean(floatMat);
  cv::Mat4f column = floatMat.reshape(0, mat.cols * mat.rows);
  column.sub


  cv::calcCovarMatrix(<#InputArray samples#>, <#OutputArray covar#>, <#InputOutputArray mean#>, <#int flags#>)
}

@end

NS_ASSUME_NONNULL_END
