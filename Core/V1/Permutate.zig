
// make this an enum at some point
pub const SizeAndStride = @import("SizesAndStrides.zig").SizeAndStride;
pub const SizesAndStrides = @import("SizesAndStrides.zig").SizesAndStrides;
pub const SizesType = SizeAndStride.ValueType;
const OrderType = @import("SizesAndStrides.zig").OrderType;

pub fn permutateInput(    
    comptime rank : usize,
    comptime order : OrderType,
    x_s: *SizesAndStrides(rank, order),
    permutation: *const [rank]u32
) void {

    var tmp: SizesAndStrides(rank, order) = undefined;

    var i :usize = 0;
    for(permutation) |p| {
        tmp.setSizeAndStride(i, x_s.*.getSizeAndStride(p));
        i += 1;
    }
    x_s.* = tmp;
}

pub fn checkBitwisePermutation(
    comptime rank: usize, 
    permutation: *const [rank]SizesType) bool {
    // O(N) operation to check for valid permutations.

    // All indices of the SizesAndStrides must be
    // checked before we can permutate. Otherwise,
    // this could mean that a transpose operation
    // could leave a tensor in an invalid state.

    // bitwise limit to check if an index is out of bounds
    const limit: usize = ((rank + 1) << 1);

    // storage for bitwise OR operations checks
    var checked: usize = 0;

    // bit shifting zero by one is a no-op
    // this is a work-around for indexing
    var is_zero: usize = 0;
    
    for(permutation.*) |i| { 
        checked |= (i << 1); 
        is_zero |= @boolToInt((i == 0));
    }
    checked += is_zero;
    
    return (checked < limit) and @popCount(checked) == rank;
}

test "Bitwise-Permutation" {
    const expect = @import("std").testing.expect;

    // valid permutation checks...
    try expect(checkBitwisePermutation(3, &.{ 0, 1, 2 }));
    try expect(checkBitwisePermutation(3, &.{ 0, 2, 1 }));
    try expect(checkBitwisePermutation(3, &.{ 1, 0, 2 }));
    try expect(checkBitwisePermutation(3, &.{ 1, 2, 0 }));
    try expect(checkBitwisePermutation(3, &.{ 2, 0, 1 }));
    try expect(checkBitwisePermutation(3, &.{ 2, 1, 0 }));
    
    // invalid permutation checks...
    try expect(!checkBitwisePermutation(3, &.{ 0, 1, 0 }));
    try expect(!checkBitwisePermutation(3, &.{ 0, 2, 6 }));
    try expect(!checkBitwisePermutation(3, &.{ 0, 0, 0 }));
    try expect(!checkBitwisePermutation(3, &.{ 1, 1, 1 }));
    try expect(!checkBitwisePermutation(3, &.{ 6, 7, 8 }));
    try expect(!checkBitwisePermutation(3, &.{ 1, 2, 2 }));
    try expect(!checkBitwisePermutation(3, &.{ 1, 2, 3 }));
}
