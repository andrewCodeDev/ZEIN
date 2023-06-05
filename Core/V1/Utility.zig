
const ReduceOp = @import("std").builtin.ReduceOp;

pub fn arrayProduct(comptime rank: usize, comptime value_type: type, ints: *const [rank]value_type) value_type {
    const s : @Vector(rank, value_type) = ints.*;
    return @reduce(ReduceOp.Mul, s);
}

pub fn arraySum(comptime rank: usize, comptime value_type: type, ints: *const [rank]value_type) value_type {
    const s : @Vector(rank, value_type) = ints.*;
    return @reduce(ReduceOp.Sum, s);
}

pub fn sliceProduct(comptime value_type: type, ints: [] const value_type) value_type {
    var total: value_type = @boolToInt(0 < ints.len);
    for(ints) |n| { total *= n; }
    return total;
}

pub fn sliceSum(comptime value_type: type, ints: [] const value_type) value_type {
    var total: value_type = 0;
    for(ints) |n| { total += n; }
    return total;
}
