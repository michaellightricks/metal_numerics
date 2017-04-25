//
//  SMVM.metal
//  MetalNumerics
//
//  Created by Michael Kupchick on 22/06/2016.
//  Copyright © 2016 Michael Kupchick. All rights reserved.
//

#include <metal_stdlib>

using namespace metal;

typedef device float4* pdfloat4;
typedef device float* pdfloat;

/// arithmetics is scalar but load/store are vectorized
/// TODO: try to use blocks of 2x2. or first float4 without border checking assuming that every
/// length is dividable by 4.

kernel void SMVM_CSR(device float *csrBuffer [[buffer(0)]], device int *nzRow [[buffer(1)]],
                     device int* columnIdxs [[buffer(2)]],
                     device float *vector [[buffer(3)]], device float *outVector [[buffer(4)]],
                     uint tid [[ thread_position_in_grid]]) {
  int startIdx = nzRow[tid-1];
  int elemInRow = nzRow[tid];
  int elemInRow4 = (elemInRow / 4) * 4;
  float accumulator = 0;
  int i = startIdx;
  for (; i < elemInRow4; i += 4) {
    /// casting is "free" in metal
    pdfloat4 val = (pdfloat4)(csrBuffer + i);
    for (int j = i; j < 4; ++j) {
      if (j < elemInRow) {
         accumulator += *((pdfloat)(val + j)) * vector[columnIdxs[i + j]];
      }
    }
  }

  if (i < elemInRow) {
    pdfloat4 val = (pdfloat4)(csrBuffer + i);
    for (int j = i; j < 4; ++j) {
      if (j < elemInRow) {
        accumulator += *(pdfloat)(val + j) * vector[columnIdxs[i + j]];
      }
    }
  }

  outVector[tid] = accumulator;
}

constant ushort kWidth = 25;
constant ushort k4Stride = 7;

kernel void SMVM_CONST_WIDTH(device float4 *matrixByRow [[buffer(0)]],
                             device ushort4 * columnIdxs [[buffer(1)]],
                             constant float *vector [[buffer(2)]],
                             device float *outVector [[buffer(3)]],
                             ushort tid [[ thread_position_in_grid]]) {
  ushort row = tid;
  float result = 0;
  for (int i = 0; i < k4Stride; ++i) {
    float4 elements = matrixByRow[row * k4Stride + i];
    ushort4 columns = columnIdxs[row * k4Stride + i];
    float4 vectorElements;
    vectorElements.x = vector[columns.x];
    vectorElements.y = vector[columns.y];
    vectorElements.z = vector[columns.z];
    vectorElements.w = vector[columns.w];

    result += dot(elements, vectorElements);
  }

  outVector[row] = result;
}


