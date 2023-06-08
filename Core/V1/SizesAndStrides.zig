
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

const SizesType = u32;

pub const SizeAndStride = struct {
    pub const ValueType = SizesType;
    size : ValueType = 0,
    stride : ValueType = 0
};

/////////////////////////////////////////////////////////
// Split SizeAndStrides into a contiguous segmented array

inline fn unpackOptionalSizes(comptime rank: usize,  sizes: ?[rank]SizesType) [rank]SizesType {

    if(sizes) |data| {
        return data;
    }
    else {
        var data : [rank]SizeAndStride.ValueType = undefined;
        @memset(&data, 0);
        return data;
    }
}


fn inferStridesFromSizes(comptime rank: usize, comptime order: OrderType, sizes: ?[rank]SizesType) [rank]SizesType {
    
    var strides : [rank]SizesType = undefined;

    if(rank == 1) {
        strides[0] = 1;
        return strides;
    }

    if(sizes) |data| { 
        strides = data;

        if (order == OrderType.rowwise) {
                
            var i: usize = (rank - 1);
            var n: SizesType = 1;

            while(i > 0) : (i -= 1) {
                strides[i] = n;
                n *= data[i];
            }
            strides[0] = n;
        }

        else {

            var i: usize = 0;
            var n: SizesType = 1;

            while(i < (rank - 1)) : (i += 1) {
                strides[i] = n;
                n *= data[i];
            }
            strides[rank - 1] = n;
        }
    }

    else {
        @memset(&strides, 0); // zero seems like a sensible default...
    }    
    return strides;
 }

pub fn defaultPermutation(comptime rank: usize) [rank]SizesType {
    var tmp: [rank]SizesType = undefined;
    var i: SizesType = 0;
    while(i < rank) : (i += 1) { tmp[i] = i; }
    return tmp;
}

/////////////////////////////////////////
// SizesAndStrides Struct Implementation 

 pub fn SizesAndStrides(comptime rank: usize, comptime order: OrderType) type {

    return struct {

        const Rank = rank;

        const Self = @This();

        const SelfPtr = *Self;

        const Order = order;

        const ConstSelfPtr = *const Self;

        pub const ValueType = SizesType;
    
        sizes: [Rank]ValueType = undefined,
        strides: [Rank]ValueType = undefined,
        permutation: [Rank]ValueType = undefined,
    
        pub fn init(sizes: ?[Rank]ValueType) Self {
            return Self {
                .sizes = unpackOptionalSizes(Rank, sizes),
                .strides = inferStridesFromSizes(Rank, Order, sizes),
                .permutation = defaultPermutation(Rank)
            };
        }
        
        //// pairwise setters/getter
        pub fn getSizeAndStride(self: ConstSelfPtr, i: usize) SizeAndStride {
            return .{ .size = self.sizes[i], .stride = self.strides[i] };
        }
        pub fn setSizeAndStride(self: SelfPtr, i: usize, pair: SizeAndStride) void {
            self.sizes[i] = pair.size;
            self.strides[i] = pair.stride;
        }
    };
}
    
/////////////////////////////////
//////////// TESTING ////////////

test "Rowwise/Colwise Ordering" {

    const std = @import("std");

    { ////////////////////////////////////////////
        var s1 = SizesAndStrides(3, Rowwise).init(.{ 3, 2, 2 });
        try std.testing.expect(s1.sizes[0] == 3);
        try std.testing.expect(s1.sizes[1] == 2);
        try std.testing.expect(s1.sizes[2] == 2);
        try std.testing.expect(s1.strides[0] == 4);
        try std.testing.expect(s1.strides[1] == 2);
        try std.testing.expect(s1.strides[2] == 1);
    } 
    { ////////////////////////////////////////////
        var s1 = SizesAndStrides(3, Colwise).init(.{ 3, 2, 2 });
        try std.testing.expect(s1.sizes[0] == 3);
        try std.testing.expect(s1.sizes[1] == 2);
        try std.testing.expect(s1.sizes[2] == 2);
        try std.testing.expect(s1.strides[0] == 1);
        try std.testing.expect(s1.strides[1] == 3);
        try std.testing.expect(s1.strides[2] == 6);
     } 
}