
// Another implementation of this similar to Pytorch:C10 is to make a union with a dynamic
// memory member variable that allows for extending the tensor modes beyond the static 
// storage size. Unfortunately, that incurs the cost of checking which member is in use.

// A potential work around is to return a slice (or some reference object) and use that.
// That is cumbersome though, especially for internal implementation details.


pub const OrderType = enum {
    rowwise,
    colwise,
};

pub const Rowwise = OrderType.rowwise;
pub const Colwise = OrderType.colwise;

pub const SizeAndStride = struct {
    pub const ValueType = u32;
    size : ValueType = 0,
    stride : ValueType = 0
};

/////////////////////////////////////////////////////////
// Split SizeAndStrides into a contiguous segmented array

inline fn unpackOptionalSizes(
    comptime rank: usize, 
    sizes: ?[rank]SizeAndStride.ValueType) [rank]SizeAndStride.ValueType {

    if(sizes) |data| {
        return data;
    }
    else {
        var data : [rank]SizeAndStride.ValueType = undefined;
        @memset(&data, 0);
        return data;
    }
}


fn inferStridesFromSizes(
        comptime rank: usize, 
        comptime order: OrderType,
        sizes: ?[rank]SizeAndStride.ValueType
    ) [rank]SizeAndStride.ValueType {
    
    var strides : [rank]SizeAndStride.ValueType = undefined;

    if(sizes) |data| { 
        if (order == OrderType.rowwise) {
                
            // the farthest right element needs to have a stride of one
            strides[rank - 1] = 1;

            // all of the other elements step stride over the next size up
            var i: usize = rank - 1;

            while(0 < i) : (i -= 1) {
                strides[i - 1] = data[i];
            }
        }

        else {
            // the farthest left element needs to have a stride of one
            strides[0] = 1;

            // all of the other elements step stride over the next size up
            var i: usize = 1;

            while(i < rank) : (i += 1) {
                strides[i] = data[i - 1];
            }
        }
    }

    else {
        @memset(&strides, 0); // zero seems like a sensible default...
    }    
    return strides;
 }

/////////////////////////////////////////
// SizesAndStrides Struct Implementation 

 pub fn SizesAndStrides(comptime rank: usize, comptime order: OrderType) type {

    return struct {

        const Rank = rank;

        const Order = order;

        const Self = @This();

        const SelfPtr = *Self;

        const ConstSelfPtr = *const Self;

        pub const ValueType = SizeAndStride.ValueType;
    
        sizes:   [Rank]ValueType = undefined,
        strides: [Rank]ValueType = undefined,
    
        pub fn init(sizes: ?[Rank]ValueType) Self {
            return Self {
                .sizes = unpackOptionalSizes(Rank, sizes),
                .strides = inferStridesFromSizes(Rank, Order, sizes),
            };
        }
        
        //// pairwise setters/getter
        pub fn getSizeAndStride(self: ConstSelfPtr, i: usize) SizeAndStride {
            return .{ .size = self.*.sizes[i], .stride = self.*.strides[i] };
        }
        pub fn setSizeAndStride(self: SelfPtr, i: usize, pair: SizeAndStride) void {
            self.*.sizes[i] = pair.size;
            self.*.strides[i] = pair.stride;
        }
    };
}
    
/////////////////////////////////
//////////// TESTING ////////////

test "Initialization" {
    const std = @import("std");

    var s1 = SizesAndStrides(5, Rowwise).init(
            .{ 100, 101, 102, 103, 104, }
        );

    var s2 = SizesAndStrides(5, Rowwise).init(null);

    s2.setSizeAndStride(0, .{ .size = 100, .stride = 101 });
    s2.setSizeAndStride(1, .{ .size = 101, .stride = 102 });
    s2.setSizeAndStride(2, .{ .size = 102, .stride = 103 });
    s2.setSizeAndStride(3, .{ .size = 103, .stride = 104 });
    s2.setSizeAndStride(4, .{ .size = 104, .stride =   1 });
        
    try std.testing.expect(s1.sizes[0] == s2.sizes[0]);
    try std.testing.expect(s1.sizes[1] == s2.sizes[1]);
    try std.testing.expect(s1.sizes[2] == s2.sizes[2]);
    try std.testing.expect(s1.sizes[3] == s2.sizes[3]);
    try std.testing.expect(s1.sizes[4] == s2.sizes[4]);

    try std.testing.expect(s1.strides[0] == s2.strides[0]);
    try std.testing.expect(s1.strides[1] == s2.strides[1]);
    try std.testing.expect(s1.strides[2] == s2.strides[2]);
    try std.testing.expect(s1.strides[3] == s2.strides[3]);
    try std.testing.expect(s1.strides[4] == s2.strides[4]);
}

test "Rowwise/Colwise Ordering" {

    const std = @import("std");

    {
        var s1 = SizesAndStrides(3, Rowwise).init(.{ 100, 101, 102, });
        try std.testing.expect(s1.sizes[0] == 100);
        try std.testing.expect(s1.sizes[1] == 101);
        try std.testing.expect(s1.sizes[2] == 102);
        try std.testing.expect(s1.strides[0] == 101);
        try std.testing.expect(s1.strides[1] == 102);
        try std.testing.expect(s1.strides[2] ==   1);
    } 
    {
        var s1 = SizesAndStrides(3, Colwise).init(.{ 100, 101, 102, });
        try std.testing.expect(s1.sizes[0] == 100);
        try std.testing.expect(s1.sizes[1] == 101);
        try std.testing.expect(s1.sizes[2] == 102);
        try std.testing.expect(s1.strides[0] ==   1);
        try std.testing.expect(s1.strides[1] == 100);
        try std.testing.expect(s1.strides[2] == 101);
     } 
}