
// Zein interface file - import this file directly into your project to begin using Zein.

// import core SizesAndStrides version
const SizesAndStridesVersion = @import("Core/V1/SizesAndStrides.zig");
pub const SizeAndStride = SizesAndStridesVersion.SizeAndStride;
pub const SizesAndStrides = SizesAndStridesVersion.SizesAndStrides;

// import core tensor version... this can be swapped for different tensor implementations.
const TensorVersion = @import("Core/V1/Tensor.zig");
pub const Tensor = TensorVersion.Tensor;
pub const TensorError = TensorVersion.TensorError;
pub const OrderType = TensorVersion.OrderType;
pub const Rowwise = TensorVersion.Rowwise;
pub const Colwise = TensorVersion.Colwise;

// import core TensorFactory version
const TensorFactoryVersion = @import("Core/V1/TensorFactory.zig");
pub const TensorFactory = TensorFactoryVersion.TensorFactory;
pub const AllocatorError = TensorFactoryVersion.AllocatorError;

// import core TensorOps version
const TensorOpsVersion = @import("Core/V1/TensorOps.zig");
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
