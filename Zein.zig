
// Zein interface file - import this file directly into your project to begin using Zein.

// import core tensor version... this can be swapped for different tensor implementations.
const Version = @import("Core/V1/Tensor.zig");
const Tensor = Version.Tensor;
const Rowwise = Version.Rowwise;
const Colwise = Version.Colwise;
