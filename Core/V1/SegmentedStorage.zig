
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

pub const SizeAndStride = struct {
    size : u32 = 0,
    stride : u32 = 0
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

/////////////////////////////////////////
// SizesAndStrides Struct Implementation 

 pub fn SizesAndStrides(comptime pair_count: usize) type {

    return struct {

        const Self = @This();
    
        memory: [pair_count * 2]u32 = undefined,
    
        // init SizesAndStrides from an optional pair array
        pub fn init(pairs: ?[pair_count]SizeAndStride) Self {
            return Self {
                .memory = initSizesAndStrides(pair_count, pairs),
            };
        }
        
        //// segmented index getters
        pub fn getSize(self: Self, i: usize) u32 {
            return self.memory[i];
        }
        pub fn getStride(self: Self, i: usize) u32 {
            return self.memory[pair_count + i];
        }
    
        //// segemented index setters
        pub fn setSize(self: *Self, i: usize, value: u32) void {
            self.*.memory[i] = value;
        }
        pub fn setStride(self: *Self, i: usize, value: u32) void {
            self.*.memory[pair_count + i] = value;
        }
    
        //// pairwise setters/getter
        pub fn getSizeAndStride(self: Self, i: usize) SizeAndStride {
            return .{ .size= self.getSize(i), .stride = self.getStride(i) };
        }
        pub fn setSizeAndStride(self: *Self, i: usize, pair: SizeAndStride) void {
            self.*.setSize(i, pair.size);
            self.*.setStride(i, pair.stride);
        }

        //// segmented slicing functions
        pub fn sliceSizes(self: *const Self) [] const u32 {
            return self.*.memory[0..pair_count];
        }
        pub fn sliceStrides(self: *const Self) [] const u32 {
            return self.*.memory[pair_count..pair_count * 2];
        }
    
        pub fn copyFrom(self: *Self, other: * const Self) void {
            @memcpy(&self.*.memory, &other.*.memory);        
        }
    
        // type sizes... member functions instead?
        pub fn segmentSize() usize {
            return pair_count;
        }
        pub fn capacity() usize {
            return pair_count * 2;
        }
    };
}
    
/////////////////////////////////
//////////// TESTING ////////////

test "Initialization" {
    const std = @import("std");

    var s1 = SizesAndStrides(5).init([_]SizeAndStride{
            .{ .size = 100, .stride = 100 },
            .{ .size = 101, .stride = 101 },
            .{ .size = 102, .stride = 102 },
            .{ .size = 103, .stride = 103 },
            .{ .size = 104, .stride = 104 },
        });

    var s2 = SizesAndStrides(5).init(null);

    s2.setSizeAndStride(0, .{ .size = 100, .stride = 100 });
    s2.setSizeAndStride(1, .{ .size = 101, .stride = 101 });
    s2.setSizeAndStride(2, .{ .size = 102, .stride = 102 });
    s2.setSizeAndStride(3, .{ .size = 103, .stride = 103 });
    s2.setSizeAndStride(4, .{ .size = 104, .stride = 104 });

    try std.testing.expect(SizesAndStrides(5).capacity() == 10);
    try std.testing.expect(SizesAndStrides(5).capacity() == 10);
    try std.testing.expect(SizesAndStrides(5).segmentSize() == 5);
    try std.testing.expect(SizesAndStrides(5).segmentSize() == 5);

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

    var s1 = SizesAndStrides(3).init([_]SizeAndStride{
            .{ .size = 100, .stride = 200 },
            .{ .size = 101, .stride = 201 },
            .{ .size = 102, .stride = 202 },
        });

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
        try std.testing.expect(slice[0] == 200);
        try std.testing.expect(slice[1] == 201);
        try std.testing.expect(slice[2] == 202);
    }    
}

test "Copying" {

    const std = @import("std");

    var s1 = SizesAndStrides(5).init([_]SizeAndStride{
            .{ .size = 100, .stride = 100 },
            .{ .size = 101, .stride = 101 },
            .{ .size = 102, .stride = 102 },
            .{ .size = 103, .stride = 103 },
            .{ .size = 104, .stride = 104 },
        });

    var s2 = SizesAndStrides(5).init(null);

    s2.copyFrom(&s1);

    for(s1.memory, s2.memory) |i, j| {
        try std.testing.expect(i == j);
    }
}