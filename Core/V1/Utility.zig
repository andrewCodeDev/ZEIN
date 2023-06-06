
const ReduceOp = @import("std").builtin.ReduceOp;

pub fn arrayProduct(comptime rank: usize, comptime T: type, ints: *const [rank]T) T {
    const s : @Vector(rank, T) = ints.*;
    return @reduce(ReduceOp.Mul, s);
}

pub fn arraySum(comptime rank: usize, comptime T: type, ints: *const [rank]T) T {
    const s : @Vector(rank, T) = ints.*;
    return @reduce(ReduceOp.Sum, s);
}

pub fn sliceProduct(comptime T: type, ints: [] const T) T {
    var total: T = @boolToInt(0 < ints.len);
    for(ints) |n| { total *= n; }
    return total;
}

pub fn sliceSum(comptime T: type, ints: [] const T) T {
    var total: T = 0;
    for(ints) |n| { total += n; }
    return total;
}

