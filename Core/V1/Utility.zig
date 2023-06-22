
const ReduceOp = @import("std").builtin.ReduceOp;

pub fn arrayProduct(comptime rank: usize, comptime T: type, values: *const [rank]T) T {
    const s : @Vector(rank, T) = values.*;
    return @reduce(ReduceOp.Mul, s);
}

pub fn arraySum(comptime rank: usize, comptime T: type, values: *const [rank]T) T {
    const s : @Vector(rank, T) = values.*;
    return @reduce(ReduceOp.Sum, s);
}

pub fn sliceProduct(comptime T: type, values: [] const T) T {
    if(values.len == 0){ return 0; }
    var total: T = 1;
    for(values) |n| { total *= n; }
    return total;
}

pub fn sliceSum(comptime T: type, values: [] const T) T {
    var total: T = 0;
    for(values) |n| { total += n; }
    return total;
}

