// Zein interface file - import this file directly into your project to begin using Zein.

// import core SizesAndStrides version
const SizesAndStridesVersion = @import("./sizes_and_strides.zig");
pub const SizeAndStride = SizesAndStridesVersion.SizeAndStride;
pub const SizesAndStrides = SizesAndStridesVersion.SizesAndStrides;

// import core tensor version... this can be swapped for different tensor implementations.
const TensorVersion = @import("./tensor.zig");
pub const Tensor = TensorVersion.Tensor;
pub const TensorError = TensorVersion.TensorError;
pub const Rowwise = TensorVersion.Rowwise;
pub const Colwise = TensorVersion.Colwise;

// import core TensorFactory version
const TensorFactoryVersion = @import("./tensor_factory.zig");
pub const TensorFactory = TensorFactoryVersion.TensorFactory;
pub const AllocatorError = TensorFactoryVersion.AllocatorError;

// import core TensorOps version
const TensorOpsVersion = @import("./tensor_ops.zig");

pub const sum = TensorOpsVersion.sum;
pub const sumUnchecked = TensorOpsVersion.sumUnchecked;

pub const product = TensorOpsVersion.product;
pub const productUnchecked = TensorOpsVersion.productUnchecked;

pub const min = TensorOpsVersion.min;
pub const minUnchecked = TensorOpsVersion.minUnchecked;

pub const max = TensorOpsVersion.max;
pub const maxUnchecked = TensorOpsVersion.maxUnchecked;

pub const contraction = TensorOpsVersion.contraction;
pub const contractionUnchecked = TensorOpsVersion.contractionUnchecked;

pub const scale = TensorOpsVersion.scale;
pub const scaleUnchecked = TensorOpsVersion.scaleUnchecked;

pub const bias = TensorOpsVersion.bias;
pub const biasUnchecked = TensorOpsVersion.biasUnchecked;

pub const add = TensorOpsVersion.add;
pub const addUnchecked = TensorOpsVersion.addUnchecked;

pub const mul = TensorOpsVersion.mul;
pub const mulUnchecked = TensorOpsVersion.mulUnchecked;

pub const sub = TensorOpsVersion.subUnchecked;
pub const subUnchecked = TensorOpsVersion.subUnchecked;

pub const absmax = TensorOpsVersion.absmax;
pub const absmaxUnchecked = TensorOpsVersion.absmaxUnchecked;

pub const absmin = TensorOpsVersion.absmin;
pub const absminUnchecked = TensorOpsVersion.absminUnchecked;

pub const quantize = TensorOpsVersion.quantize;
pub const unquantize = TensorOpsVersion.unquantize;

pub const fill = TensorOpsVersion.fill;
