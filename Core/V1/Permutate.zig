
// make this an enum at some point
pub const SizeAndStride = @import("SizesAndStrides.zig").SizeAndStride;
pub const SizesAndStrides = @import("SizesAndStrides.zig").SizesAndStrides;
pub const SizesType = SizeAndStride.ValueType;
const OrderType = @import("SizesAndStrides.zig").OrderType;

const permutateParse = @import("ExpressionParsing.zig").permutateParse;

pub fn permutate(    
    comptime rank : usize,
    comptime order : OrderType,
    comptime str: [] const u8,
    ss: *SizesAndStrides(rank, order)
) void {
    const permutation = comptime permutateParse(rank, str);

    var tmp: SizesAndStrides(rank, order) = undefined;

    var i : usize = 0;
    for(permutation) |p| {
        tmp.setSizeAndStride(i, ss.getSizeAndStride(p));
        tmp.permutation[i] = p;
        i += 1;
    }
    ss.* = tmp;
}

test "Permutation" {
    const expectEqual = @import("std").testing.expectEqual;

    var ss = SizesAndStrides(3, OrderType.rowwise).init(.{10, 20, 30});
    
    try expectEqual(ss.permutation[0], 0);
    try expectEqual(ss.permutation[1], 1);
    try expectEqual(ss.permutation[2], 2);
    try expectEqual(ss.sizes[0], 10);
    try expectEqual(ss.sizes[1], 20);
    try expectEqual(ss.sizes[2], 30);

    permutate(3, OrderType.rowwise, "ijk->kji", &ss);

    try expectEqual(ss.permutation[0], 2);
    try expectEqual(ss.permutation[1], 1);
    try expectEqual(ss.permutation[2], 0);
    try expectEqual(ss.sizes[0], 30);
    try expectEqual(ss.sizes[1], 20);
    try expectEqual(ss.sizes[2], 10);

}
