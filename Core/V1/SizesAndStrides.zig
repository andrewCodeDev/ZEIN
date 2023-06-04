
// Since the sizes and the strides type will be the same int type,
// we can use the same array, split half-way through for both.

// so... if segment size is 3...
//            
//      v - start of first segement
//      v
//    [ 0, 1, 2, 3, 4, 5 ]
//               ^
//               ^ - Start of the second segment

// This class makes an assumption that each index has an stride and size pair.
// Therefore, it doesn't make sense to allow only placing a single element.

// V1 makes the assumption that we probably need 5 strides (at most) to be practical

// Another implementation of this similar to Pytorch:C10 is to make a union with a dynamic
// memory member variable that allows for extending the tensor modes beyond the static 
// storage size. Unfortunately, that incurs the cost of checking which member is in use.

// A potential work around is to return a slice (or some reference object) and use that.
// That is cumbersome though, especially for internal implementation details.

////////////////////////////////////////////////////
// This struct is debatable. It may be better to use
// a generic pair class instead of this, because I
// forsee some tedious converions in the future.

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

///////////////////////////////////////////////////////
// Split SizeAndStrides into a contiguous segmented array

fn initSizesAndStrides(comptime n: usize, pairs: ?[n]SizeAndStride) [n * 2]u32 {
   // this is a hack because init will cause an error
   // if we don't return the struct type directly    
    var memory: [n * 2]u32 = undefined;

    if (pairs) |data| 
    {
        var i: usize = 0;
        while(i < n) : (i += 1) {                        
            memory[i]     = data[i].size;            
            memory[i + n] = data[i].stride;
        }
    }
    else {
        @memset(&memory, 0); // zero seems like a sensible default...
    }
    return memory;
}

 pub fn inferStridesFromSizes(
        comptime rank: usize, 
        comptime order: OrderType,
        sizes: ?[rank]SizeAndStride.ValueType
    ) [rank * 2]SizeAndStride.ValueType {

    const full = rank * 2;
    
    var memory : [full]SizeAndStride.ValueType = undefined;

    if(sizes) |data| { 
        if (order == OrderType.rowwise) {
                
            // the farthest right element needs to have a stride of one
            memory[rank - 1] = data[rank - 1];
            memory[full - 1] = 1;

            // all of the other elements step stride over the next size up
            var i: usize = rank - 1;
            var j: usize = full - 1;

            while(0 < i) {
                i -= 1;
                j -= 1;
                memory[i] = data[i];
                memory[j] = data[i + 1];
            }
        }

        else {
            // the farthest left element needs to have a stride of one
            memory[0] = data[0];
            memory[rank] = 1;

            // all of the other elements step stride over the next size up
            var i: usize = 0;
            var j: usize = rank;

            while(i < rank) {
                i += 1;
                j += 1;
                memory[i] = data[i];
                memory[j] = data[i - 1];
            }
        }
    }

    else {
        @memset(&memory, 0); // zero seems like a sensible default...
    }
    
    return memory;
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
    
        memory: [Rank * 2]u32 = undefined,
    
        pub fn init(sizes: ?[Rank]ValueType) Self {
            return Self {
                .memory = inferStridesFromSizes(Rank, Order, sizes),
            };
        }
        
        //// segmented index getters
        pub fn getSize(self: Self, i: usize) u32 {
            return self.memory[i];
        }
        pub fn getStride(self: Self, i: usize) u32 {
            return self.memory[Rank + i];
        }
    
        //// segemented index setters
        pub fn setSize(self: SelfPtr, i: usize, value: u32) void {
            self.*.memory[i] = value;
        }
        pub fn setStride(self: SelfPtr, i: usize, value: u32) void {
            self.*.memory[Rank + i] = value;
        }
    
        //// pairwise setters/getter
        pub fn getSizeAndStride(self: Self, i: usize) SizeAndStride {
            return .{ .size= self.getSize(i), .stride = self.getStride(i) };
        }
        pub fn setSizeAndStride(self: SelfPtr, i: usize, pair: SizeAndStride) void {
            self.*.setSize(i, pair.size);
            self.*.setStride(i, pair.stride);
        }

        //// segmented slicing functions
        pub fn sliceSizes(self: ConstSelfPtr) [] const u32 {
            return self.*.memory[0..Rank];
        }
        pub fn sliceStrides(self: ConstSelfPtr) [] const u32 {
            return self.*.memory[Rank..Rank * 2];
        }    

        // type sizes... member functions instead?
        pub fn segmentSize() usize {
            return Rank;
        }
        pub fn capacity() usize {
            return Rank * 2;
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

    try std.testing.expect(SizesAndStrides(5, Rowwise).capacity() == 10);
    try std.testing.expect(SizesAndStrides(5, Rowwise).capacity() == 10);
    try std.testing.expect(SizesAndStrides(5, Rowwise).segmentSize() == 5);
    try std.testing.expect(SizesAndStrides(5, Rowwise).segmentSize() == 5);

    try std.testing.expect(s1.getSize(0) == s2.getSize(0));
    try std.testing.expect(s1.getSize(1) == s2.getSize(1));
    try std.testing.expect(s1.getSize(2) == s2.getSize(2));
    try std.testing.expect(s1.getSize(3) == s2.getSize(3));
    try std.testing.expect(s1.getSize(4) == s2.getSize(4));

    try std.testing.expect(s1.getStride(0) == s2.getStride(0));
    try std.testing.expect(s1.getStride(1) == s2.getStride(1));
    try std.testing.expect(s1.getStride(2) == s2.getStride(2));
    try std.testing.expect(s1.getStride(3) == s2.getStride(3));
    try std.testing.expect(s1.getStride(4) == s2.getStride(4));
}

test "Slicing" {

    const std = @import("std");

    var s1 = SizesAndStrides(3, Rowwise).init(.{ 100, 101, 102, });

    { // test size slice
        const slice = s1.sliceSizes();
        try std.testing.expect(slice.len == 3);
        try std.testing.expect(slice[0] == 100);
        try std.testing.expect(slice[1] == 101);
        try std.testing.expect(slice[2] == 102);
    }
    { // test stride slice
        const slice = s1.sliceStrides();
        try std.testing.expect(slice.len == 3);
        try std.testing.expect(slice[0] == 101);
        try std.testing.expect(slice[1] == 102);
        try std.testing.expect(slice[2] ==   1);
    }    
}