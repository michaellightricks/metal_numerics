// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNCovarianceKernel.h"

#import <MetalKit/MetalKit.h>
#import <UIKit/UIKit.h>

#import <simd/simd.h>
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>

#import "MNContext.h"

NS_ASSUME_NONNULL_BEGIN

@interface MNCovarianceKernel ()

@property (nonatomic) CGSize inputSize;

@property (nonatomic) id<MTLBuffer> inputBuffer;

@property (nonatomic) id<MTLBuffer> sizeBuffer;

@end

@implementation MNCovarianceKernel

- (instancetype)initWithContext:(MNContext *)context {
  return [super initWithContext:context functionName:@"covarianceKernel"];
}

- (void)setInputBuffer:(id<MTLBuffer>)buffer size:(CGSize)size {
  self.inputSize = size;
  self.inputBuffer = buffer;
}

- (void)configureCommandEncoder:(id<MTLComputeCommandEncoder>)encoder {
  //[encoder setTexture:self.inputTexture atIndex:0];
  [encoder setBuffer:self.inputBuffer offset:0 atIndex:0];
  [encoder setBuffer:self.outputBuffer offset:0 atIndex:1];

//  simd_uint2 size;
//  size.x = (uint)self.inputSize.width;
//  size.y = (uint)self.inputSize.height;
//  [encoder setBytes:&size length:sizeof(simd_uint2) atIndex:2];
}

- (MTLSize)threadGroupSize {
  return MTLSizeMake(16, 16, 1);
}

- (MTLSize)threadGroupsCountWithGroupSize:(MTLSize)threadGroupSize {
  //return MTLSizeMake(1, 1, 1);
  return MTLSizeMake(cv::max(1, (int)(self.inputSize.width / threadGroupSize.width)),
                     cv::max(1, (int)(self.inputSize.height / threadGroupSize.height)), 1);
}

+ (void)testWithContext:(MNContext *)context {
  MNCovarianceKernel *kernel = [[MNCovarianceKernel alloc] initWithContext:context];
  UIImage *input = [UIImage imageNamed:@"lena"];
  cv::Mat4b mat;
  UIImageToMat(input, mat);

  id<MTLBuffer> inputBuffer = [context.device newBufferWithLength:mat.cols * mat.rows * sizeof(simd_uchar4)
                                                         options:MTLResourceStorageModeShared];
  simd_uchar4 *inputData = (simd_uchar4 *)[inputBuffer contents];
  for (int row = 0; row < mat.rows; ++row) {
    for (int col = 0; col < mat.cols; ++col) {
      cv::Vec4b v = mat(row, col);
      inputData[row * mat.cols + col] = simd_make_uchar4(v(0), v(1), v(2), v(3));
    }
  }

  [kernel setInputBuffer:inputBuffer size:input.size];
//  kernel.inputTexture = [MNCovarianceKernel textureFromImage:input device:context.device];
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
  NSLog(@"%@", buffer.error.description);

  NSDate *methodEnd = [NSDate date];
  NSTimeInterval executionTime = [methodEnd timeIntervalSinceDate:methodStart];

  NSLog(@"covariance kernel took %g ms", executionTime * 1000 / count);


//  for (int row = 1; row < input.size.height - 1; ++row) {
//    for (int col = 1; col < input.size.width - 1; ++col) {
//      cv::Mat window = mat(cv::Rect(col - 1, row - 1, 3, 3));
//      std::cout << "windiw byte: " << std::endl << window << std::endl;
//      cv::Mat4f floatWindow;
//      window.convertTo(floatWindow, CV_32FC4, 1.0 / 255.0, 0.0);
//
//      std::cout << "window:" << std::endl << floatWindow << std::endl;
//      cv::Mat covarianceFromInput = [self invCovarianceWithMat:floatWindow];
//      cv::Mat covarianceFromMetal = [self invCovarianceWithBuffer:kernel.outputBuffer
//                                                         position:cv::Point(col, row)
//                                                             size:cv::Size(mat.cols, mat.rows)];
//      std::cout << covarianceFromInput << std::endl;
//      std::cout << covarianceFromMetal << std::endl;
////      cv::Mat difference = cv::abs(covarianceFromInput - covarianceFromMetal);
////      std::cout << difference << std::endl;
//    }
//  }
}

+ (id<MTLTexture>)textureFromImage:(UIImage *)image device:(id<MTLDevice>)device {
  NSError *error;
  id<MTLTexture> texture = [[[MTKTextureLoader alloc] initWithDevice:device]
                            newTextureWithCGImage:image.CGImage
                            options:@{MTKTextureLoaderOriginTopLeft: @YES} error:&error];

  if (error) {
    NSLog(@"errror loading input texture");
  }
  return texture;
}

cv::Mat3f FTColumnMatrix(cv::Mat4f mat, cv::Vec3f mean) {
  int elemNum = mat.cols * mat.rows;
  cv::Mat1f result(elemNum, 3, 0.0);
  for (int i = 0; i < elemNum; ++i) {
    cv::Vec3f elem(mat(i)(0), mat(i)(1), mat(i)(2));
    //cv::Vec3f centered = elem - mean;
    result(i, 0) = elem(0);
    result(i, 1) = elem(1);
    result(i, 2) = elem(2);
  }
  return result;
}

+ (cv::Mat)invCovarianceWithMat:(const cv::Mat4f &)floatMat {
  cv::Scalar mean = cv::mean(floatMat);
  cv::Mat1f meanMat(1, 3, mean(0));
  meanMat(1) = mean(1);
  meanMat(2) = mean(2);
  std::cout << "meanMat: " << meanMat  << std::endl;
  cv::Mat1f column = FTColumnMatrix(floatMat, cv::Vec3f(mean(0), mean(1), mean(2)));
  cv::Mat1f covariance(3, 3, 0.0);

  cv::calcCovarMatrix(column, covariance, meanMat, cv::CovarFlags::COVAR_ROWS |
                      cv::CovarFlags::COVAR_NORMAL | cv::CovarFlags::COVAR_USE_AVG |
                      cv::CovarFlags::COVAR_SCALE,
                      CV_32F);
  return covariance.inv();
}

+ (cv::Mat)invCovarianceWithBuffer:(id<MTLBuffer>)buffer position:(cv::Point)point
                              size:(cv::Size)size {
  float *ptr = (float *)buffer.contents;
  cv::Mat1f covariance(3, 3, 0.0);
  float *covariancePtr = ptr + (point.y * size.width + point.x) * 6;

  std::cout << "metal row1:3" << std::endl;
  for (int i = 0; i < 6; ++i) {
    std::cout << *(covariancePtr + i) << " ";
  }
  std::cout << std::endl;

  covariance(0,0) = covariancePtr[0];
  covariance(1,1) = covariancePtr[1];
  covariance(2, 2) = covariancePtr[2];
  covariance(0, 1) = covariancePtr[3];
  covariance(1, 0) = covariancePtr[3];
  covariance(0, 2) = covariancePtr[4];
  covariance(2, 0) = covariancePtr[4];
  covariance(1, 2) = covariancePtr[5];
  covariance(2, 1) = covariancePtr[5];

  return covariance;
}

@end

NS_ASSUME_NONNULL_END
