
// Zein interface file - import this file directly into your project to begin using Zein.

// import core tensor version... this can be swapped for different tensor implementations.
const Version = @import("Core/V1/Tensor.zig");

pub const Tensor = Version.Tensor;

pub const OrderType = Version.OrderType;
pub const Rowwise = Version.Rowwise;
pub const Colwise = Version.Colwise;

