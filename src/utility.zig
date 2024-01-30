/// src/utility.zig
const std = @import("std");
const tensor = @import("./tensor.zig");
const builtin = @import("builtin");

// use this to compile out safety checks
// and enforce invariants for debug builds.
const debug: bool = (builtin.mode == .Debug);

pub fn arrayProduct(comptime rank: usize, comptime T: type, values: *const [rank]T) T {
    const s: @Vector(rank, T) = values.*;
    return @reduce(std.builtin.ReduceOp.Mul, s);
}

pub fn arraySum(comptime rank: usize, comptime T: type, values: *const [rank]T) T {
    const s: @Vector(rank, T) = values.*;
    return @reduce(std.builtin.ReduceOp.Sum, s);
}

pub fn sliceProduct(comptime T: type, values: []const T) T {
    if (values.len == 0) {
        return 0;
    }
    var total: T = 1;
    for (values) |n| {
        total *= n;
    }
    return total;
}

pub fn sliceSum(comptime T: type, values: []const T) T {
    var total: T = 0;
    for (values) |n| {
        total += n;
    }
    return total;
}

test "basic tensor access" {
    var data = [9]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var X = tensor.Tensor(i32, 2, tensor.Rowwise).init(&data, .{ 3, 3 });
    const x = X.getValue(.{ 0, 2 });
    try std.testing.expect(x == 3);
}
