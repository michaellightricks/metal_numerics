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
                               filter::linear);
constant float ninth = 0.11111111111;

float3x3 readRow(texture2d<float, access::sample> inputTexture, uint2 rowCenter);
float3x3 covarianceAndMean(float3x3 row1, float3x3 row2, float3x3 row3);
float3x3 inverseCovariance(float3x3 covariance);

float3x3 readRow(texture2d<float, access::sample> inputTexture, uint2 rowCenter) {
  float3x3 row;
  //row[0] = inputTexture.sample(pixelSampler, float2(rowCenter + uint2(-1, 0))).rgb;
  row[1] = inputTexture.sample(pixelSampler, float2(rowCenter + uint2(0, 0))).rgb;
  //row[2] = inputTexture.sample(pixelSampler, float2(rowCenter + uint2(1, 0))).rgb;

  return row;
}

float3x3 covarianceAndMean(float3x3 row1, float3x3 row2, float3x3 row3) {
  float3x3 result(0.0);
  for (int i = 0; i < 3; ++i) {
    // mean
    result[2] += row1[i];
    result[2] += row2[i];
    result[2] += row3[i];
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

  result[2] *= ninth;

  // XX
  result[0][0] = ninth * result[0][0] - result[2][0] * result[2][0];
  // YY
  result[0][1] = ninth * result[0][1] - result[2][1] * result[2][1];
  // ZZ
  result[0][2] = ninth * result[0][2] - result[2][2] * result[2][2];
  // XY
  result[1][0] = ninth * result[1][0] - result[2][0] * result[2][1];
  // XZ
  result[1][1] = ninth * result[1][1] - result[2][0] * result[2][2];
  // YZ
  result[1][2] = ninth * result[1][2] - result[2][1] * result[2][2];

  result[0] += float3(ninth * 1e-4);

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

kernel void covarianceKernel(texture2d<float, access::sample> inputTexture [[texture(0)]],
                             device float *outputBuffer [[buffer(0)]],
                             uint2 tid [[ thread_position_in_grid]]) {
  //float3x3 row1 = readRow(inputTexture, tid + uint2(0, -1));
  //float3x3 row2 = readRow(inputTexture, tid);
  //float3x3 row3 = readRow(inputTexture, tid + uint2(0, 1));

//  float3x3 inverse = covarianceAndMean(row1, row2, row3);
  //float3x3 inverse = inverseCovariance(covariance);
  if (tid.x == 0 && tid.y == 0) {
    outputBuffer[0] = 1;
  }
  //outputBuffer[tid.y * 6 * inputTexture.get_width() + tid.x * 6] = 1;//row2[1][0];
//  outputBuffer[tid.y * 6 * inputTexture.get_width() + tid.x * 6 + 1] = inverse[0][1];
//  outputBuffer[tid.y * 6 * inputTexture.get_width() + tid.x * 6 + 2] = inverse[0][2];
//  outputBuffer[tid.y * 6 * inputTexture.get_width() + tid.x * 6 + 3] = inverse[1][0];
//  outputBuffer[tid.y * 6 * inputTexture.get_width() + tid.x * 6 + 4] = inverse[1][1];
//  outputBuffer[tid.y * 6 * inputTexture.get_width() + tid.x * 6 + 5] = inverse[1][2];
}
