
// make this an enum at some point
pub const SizeAndStride = @import("SizesAndStrides.zig").SizeAndStride;
pub const SizesAndStrides = @import("SizesAndStrides.zig").SizesAndStrides;
const OrderType = @import("SizesAndStrides.zig").OrderType;

pub fn permutateInput(    
    comptime rank : usize,
    comptime order_type : OrderType,
    x_s: *SizesAndStrides(rank, order_type),
    permutation: *const [rank]u32
) void {
    var tmp = SizesAndStrides(rank, order_type).init(null);

    var i :usize = 0;
    for(permutation) |p| {
        tmp.setSizeAndStride(i, x_s.*.getSizeAndStride(p));
        i += 1;
    }
    x_s.* = tmp;  
}

pub fn permutateOutput(
    comptime rank_x : usize,
    comptime value_type_x: type,
    comptime order_type_x: OrderType,
    x_d: *[] value_type_x,
    x_s: *SizesAndStrides(order_type_x, rank_x),
    // tensor y components
    comptime rank_y : usize,
    comptime value_type_y: type,
    comptime order_type_y: OrderType,
    y_d: *[] value_type_y,
    y_s: *SizesAndStrides(order_type_y, rank_y),
    // output permutation of axis
    permutation: [] const SizesAndStrides.Valuetype
) void {
    // todo - consider enabling casting for the case of
    // producing a new tensor? Could be a good idea,
    // could be a terrible idea. Skipping for now.
    if(value_type_x != value_type_y) {
        @compileError("Cannot transpose to tensors of different types - Core:V1");
    }
    if(rank_x != rank_y) {
        @compileError("Cannot transpose to tensors of different ranks - Core:V1");
    }
    if(order_type_x != order_type_y) {
        @compileError("Cannot transpose to tensors of different ranks - Core:V1");
    }

    // share data if input is different than output
    if(x_d.ptr != y_d.ptr) {
        y_d.* = x_d.*;
    }

    // modify SizesAndStrides of tensor y
    if(x_s.ptr != y_s.ptr){
        var i :usize = 0;
        for(permutation) |*p| {
            y_s.*.setSizeAndStride(i, x_s.*.getSizeAndStride(p));
            i += 1;
        }
    }
    else {
        permutateInput(rank_x, x_d, permutation);
    }
}