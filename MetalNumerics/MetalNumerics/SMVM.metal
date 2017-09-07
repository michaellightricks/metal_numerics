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

constant ushort k4Stride = 7;

kernel void SMVM_CONST_WIDTH(device half4 *matrixByRow [[buffer(0)]],
                             device ushort4 * columnIdxs [[buffer(1)]],
                             device half *vector [[buffer(2)]],
                             device half *outVector [[buffer(3)]],
                             ushort tid [[ thread_position_in_grid]]) {
  ushort row = tid;
  half result = 0;
  for (int i = 0; i < k4Stride; ++i) {
    half4 elements = matrixByRow[row * k4Stride + i];
    ushort4 columns = columnIdxs[row * k4Stride + i];
    half4 vectorElements;
    vectorElements.x = vector[columns.x];
    vectorElements.y = vector[columns.y];
    vectorElements.z = vector[columns.z];
    vectorElements.w = vector[columns.w];

    result += dot(elements, vectorElements);
  }

  outVector[row] = result;
}

kernel void SMVM_CONST_WIDTH_WARP(device half *matrixByRow [[buffer(0)]],
                                  device ushort * columnIdxs [[buffer(1)]],
                                  device half *vector [[buffer(2)]],
                                  device half *outVector [[buffer(3)]],
                                  uint tid [[ thread_position_in_grid]],
                                  uint threadInGroupIdx [[ thread_index_in_threadgroup ]]) {
  uint warpIdx = tid / 32;
  uint indexInWarp = tid - warpIdx * 32;

  threadgroup half sharedMemory[32];

  for (int i = 0; i < 32; ++i) {
    uint row = warpIdx * 32 + i;
    //uint row = warpIdx;
    half coeff = matrixByRow[32 * row + indexInWarp];
    half rhs = vector[columnIdxs[32 * row + indexInWarp]];

    half product = coeff * rhs;
    sharedMemory[threadInGroupIdx] = product;

    if (indexInWarp < 16) {
      sharedMemory[threadInGroupIdx] =
          sharedMemory[threadInGroupIdx] + sharedMemory[threadInGroupIdx + 16];
    }

    if (indexInWarp < 8) {
      sharedMemory[threadInGroupIdx] =
          sharedMemory[threadInGroupIdx] + sharedMemory[threadInGroupIdx + 8];
    }

    if (indexInWarp < 4) {
      sharedMemory[threadInGroupIdx] =
          sharedMemory[threadInGroupIdx] + sharedMemory[threadInGroupIdx + 4];
    }

    if (indexInWarp < 2) {
      sharedMemory[threadInGroupIdx] =
          sharedMemory[threadInGroupIdx] + sharedMemory[threadInGroupIdx + 2];
    }

    if (indexInWarp == 0) {
      outVector[row] = sharedMemory[threadInGroupIdx] + sharedMemory[threadInGroupIdx + 1];
    }
  }
}

void warpReduce(threadgroup float *sharedMemory, uint warpIdx, uint indexInWarp);
void groupReduce(threadgroup float *sharedMemory, uint threadInGroupIdx, uint groupDim);
void groupReduceAfterWarpReduce(threadgroup float *sharedMemory, uint threadInGroupIdx, uint dims);

void groupReduce(threadgroup float *sharedMemory, uint threadInGroupIdx, uint groupDim) {
  for (uint s = groupDim; s > 0; s = s / 2) {
    if  (threadInGroupIdx < s) {
      sharedMemory[threadInGroupIdx] = sharedMemory[threadInGroupIdx] +
          sharedMemory[threadInGroupIdx + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void averageGroupReduce(device float *buffer [[buffer(0)]],
                    device float *result [[buffer(1)]],
                    constant uint *bufferSize [[buffer(2)]],
                    threadgroup float *sharedMemory [[ threadgroup(0) ]],
                    uint threadInGroupIdx [[ thread_index_in_threadgroup ]],
                    uint groupIdx [[threadgroup_position_in_grid]],
                    uint gridDim [[threadgroups_per_grid]],
                    uint groupDim [[threads_per_threadgroup]]) {
  //uint warpIdx = threadInGroupIdx / 32;
  //uint indexInWarp = threadInGroupIdx - warpIdx * 32;

  float sum = 0;
  uint N = bufferSize[0];
  for (uint i = groupIdx * groupDim + threadInGroupIdx; i < N;
          i += gridDim * groupDim) {
    sum += buffer[i];
  }

  sharedMemory[threadInGroupIdx] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  groupReduce(sharedMemory, threadInGroupIdx, groupDim / 2);
  if (threadInGroupIdx == 0) {
    result[groupIdx] = sharedMemory[0];
  }
}

void warpReduce(threadgroup float *sharedMemory,
                uint warpIdx, uint indexInWarp) {
  uint baseIndex = warpIdx * 32 + indexInWarp;
  for (uint s = 16; s > 0; s = s / 2) {
    if  (indexInWarp < s) {
      sharedMemory[baseIndex] = sharedMemory[baseIndex] + sharedMemory[baseIndex + s];
    }
  }
}

void groupReduceAfterWarpReduce(threadgroup float *sharedMemory, uint threadInGroupIdx, uint dims) {
  for (uint s = dims; s > 0; s = s / 2) {
    if (threadInGroupIdx < s) {
      sharedMemory[threadInGroupIdx * 32] = sharedMemory[threadInGroupIdx * 32] + 
          sharedMemory[(threadInGroupIdx + s) * 32];
    }
  }
}

kernel void averageGroupReduce2(device float *buffer [[buffer(0)]],
                                //device float *result [[buffer(1)]],
                               constant uint *bufferSize [[buffer(1)]],
                               threadgroup float *sharedMemory [[ threadgroup(0) ]],
                               uint threadInGroupIdx [[ thread_index_in_threadgroup ]],
                               uint groupIdx [[threadgroup_position_in_grid]],
                               uint gridDim [[threadgroups_per_grid]],
                               uint groupDim [[threads_per_threadgroup]]) {
  uint warpIdx = threadInGroupIdx / 32;
  uint indexInWarp = threadInGroupIdx - warpIdx * 32;

  float sum = 0;
  uint N = bufferSize[0];
  for (uint i = groupIdx * groupDim + threadInGroupIdx; i < N;
       i += gridDim * groupDim) {
    sum += buffer[i];
  }

  sharedMemory[threadInGroupIdx] = sum;
  warpReduce(sharedMemory, warpIdx, indexInWarp);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  /// this is performed by one warp since max threadgroup size is 512.
  groupReduceAfterWarpReduce(sharedMemory, threadInGroupIdx, groupDim / (32 * 2));
  if (threadInGroupIdx == 0) {
    buffer[groupIdx] = sharedMemory[0];
  }
}

// TODO: try utilize texture cahces using samplers (different types of data (float4)).
