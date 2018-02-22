//
//  CovarianceKernel.metal
//  MetalNumerics
//
//  Created by Michael Kupchick on 7/9/17.
//  Copyright © 2017 Michael Kupchick. All rights reserved.
//

#include <metal_stdlib>

using namespace metal;

constexpr sampler pixelSampler(coord::pixel,
                               address::clamp_to_edge,
                               filter::nearest);
constant float ninth = 0.11111111111;

float3 readPixel(device uchar4 *inputBuffer, uint2 rowCenter, uint2 inputBufferSize) {
  float3 result = float3(0);

  uchar4 pixel = inputBuffer[rowCenter.y * inputBufferSize.x + rowCenter.x];
  result.x = (float)pixel.x / 255.0f;
  result.y = (float)pixel.y / 255.0f;
  result.z = (float)pixel.z / 255.0f;

  return result;
}

float3x3 readRow(device uchar4 *inputBuffer, uint2 rowCenter, uint2 inputBufferSize) {
  float3x3 row;
  row[0] = readPixel(inputBuffer, rowCenter + uint2(-1, 0), inputBufferSize);
  row[1] = readPixel(inputBuffer, rowCenter + uint2(0, 0), inputBufferSize);
  row[2] = readPixel(inputBuffer, rowCenter + uint2(1, 0), inputBufferSize);

  return row;
}

float3x3 covarianceAndMean(float3x3 row1, float3x3 row2, float3x3 row3) {
  float3x3 result(0.0);
  for (int i = 0; i < 3; ++i) {
    // mean
    result[2] += row1[i];
    result[2] += row2[i];
    result[2] += row3[i];
  }

  result[2] *= ninth;

  for (int i = 0; i < 3; ++i) {
    row1[i] -= result[2];
    row2[i] -= result[2];
    row3[i] -= result[2];
  }

  for (int i = 0; i < 3; ++i) {
    // XX
    result[0][0] += row1[i][0] * row1[i][0];
    result[0][0] += row2[i][0] * row2[i][0];
    result[0][0] += row3[i][0] * row3[i][0];
    // YY
    result[0][1] += row1[i][1] * row1[i][1];
    result[0][1] += row2[i][1] * row2[i][1];
    result[0][1] += row3[i][1] * row3[i][1];
    // ZZ
    result[0][2] += row1[i][2] * row1[i][2];
    result[0][2] += row2[i][2] * row2[i][2];
    result[0][2] += row3[i][2] * row3[i][2];
    // XY
    result[1][0] += row1[i][0] * row1[i][1];
    result[1][0] += row2[i][0] * row2[i][1];
    result[1][0] += row3[i][0] * row3[i][1];
    // XZ
    result[1][1] += row1[i][0] * row1[i][2];
    result[1][1] += row2[i][0] * row2[i][2];
    result[1][1] += row3[i][0] * row3[i][2];
    // YZ
    result[1][2] += row1[i][1] * row1[i][2];
    result[1][2] += row2[i][1] * row2[i][2];
    result[1][2] += row3[i][1] * row3[i][2];
  }

  // XX
  result[0][0] = ninth * result[0][0];// - result[2][0] * result[2][0];
  // YY
  result[0][1] = ninth * result[0][1];// - result[2][1] * result[2][1];
  // ZZ
  result[0][2] = ninth * result[0][2];// - result[2][2] * result[2][2];
  // XY
  result[1][0] = ninth * result[1][0];// - result[2][0] * result[2][1];
  // XZ
  result[1][1] = ninth * result[1][1];// - result[2][0] * result[2][2];
  // YZ
  result[1][2] = ninth * result[1][2];// - result[2][1] * result[2][2];

  //result[0] += float3(ninth * 1e-4);

  return result;
}

float3x3 inverseCovariance(float3x3 covariance) {
  // fullCovariance = [ a b c
  //                    d e f
  //                    g h i ]
  // covariance is symmetric so input is two columns:
  // covariance = [a e i; b(d) c(g) f(h)]
  //ei − ff
  float M11 = covariance[0][1] * covariance[0][2] - covariance[1][2] * covariance[1][2];
  //bi − fc
  float M12 = covariance[1][0] * covariance[0][2] - covariance[1][2] * covariance[1][1];
  //bf − ec
  float M13 = covariance[1][0] * covariance[1][2] - covariance[0][1] * covariance[1][1];

  // a * M11 - b * M12 + c * M13
  float det = covariance[0][0] * M11 - covariance[1][0] * M12 + covariance[1][1] * M13;

  // ai - gc
  float M22 = covariance[0][0] * covariance[0][2] - covariance[1][1] * covariance[1][1];
  // ae - bd
  float M33 = covariance[0][0] * covariance[0][1] - covariance[1][0] * covariance[1][0];
  // ah - bg
  float M23 = covariance[0][0] * covariance[1][2] - covariance[1][0] * covariance[1][1];
  // since inverse of symmetric matrix is also symmetric the result value will be in same format as
  // the input

  //highp vec3 diagonal = det < 0.000001 ? vec3(10000, 10000, 10000) : vec3(M11, M22, M33) / det;
  float3 diagonal = float3(M11, M22, M33) / det;

  //highp vec3 offDiagonal = det < 0.000001 ? vec3(0, 0, 0) : vec3(-M12, M13, -M23) / det;
  float3 offDiagonal = float3(-M12, M13, -M23) / det;

  return float3x3(diagonal, offDiagonal, float3(det));
}

kernel void covarianceKernel(//texture2d<float, access::read> inputTexture [[texture(0)]],
                             device uchar4 *inputBuffer [[buffer(0)]],
                             device float *outputBuffer [[buffer(1)]],
                             //constant uint2 *inputBufferSize [[buffer(2)]],
                             uint2 tid [[ thread_position_in_grid]]) {
  uint2 inputBufferSize(512, 512);
  if (!(tid.x < (inputBufferSize.x - 1) && tid.x > 0) ||
      !(tid.y < (inputBufferSize.y - 1) && tid.y > 0)){
    return;
  }
  
  float3x3 row1 = readRow(inputBuffer, tid + uint2(0, -1), inputBufferSize);
  float3x3 row2 = readRow(inputBuffer, tid, inputBufferSize);
  float3x3 row3 = readRow(inputBuffer, tid + uint2(0, 1), inputBufferSize);

  float3x3 covariance = covarianceAndMean(row1, row2, row3);
  float3x3 inverse = inverseCovariance(covariance);
//  if (tid.x == 0 && tid.y == 0) {
//    outputBuffer[0] = inverse[0][0];
//  }
  uint offset = 6 * (tid.y * inputBufferSize.x + tid.x);
  outputBuffer[offset] = inverse[0][0];
  outputBuffer[offset + 1] = inverse[0][1];
  outputBuffer[offset + 2] = inverse[0][2];
  outputBuffer[offset + 3] = inverse[1][0];
  outputBuffer[offset + 4] = inverse[1][1];
  outputBuffer[offset + 5] = inverse[1][2];
//  outputBuffer[offset] = row1[0][0];
//  outputBuffer[offset + 1] = row1[0][1];
//  outputBuffer[offset + 2] = row1[0][2];
//  outputBuffer[offset + 3] = row3[2][0];
//  outputBuffer[offset + 4] = row3[2][1];
//  outputBuffer[offset + 5] = row3[2][2];
}
