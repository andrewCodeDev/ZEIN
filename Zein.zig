
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

// import core TensorAllocator version
const TensorAllocatorVersion = @import("Core/V1/TensorAllocator.zig");
pub const TensorAllocator = TensorAllocatorVersion.TensorAllocator;
pub const AllocatorError = TensorAllocatorVersion.AllocatorError;
