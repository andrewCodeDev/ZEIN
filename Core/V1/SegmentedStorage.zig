
// Since the sizes and the strides type will be the same int type,
// we can use the same array, split half-way through for both.

// so... if segment size is 3...
//            
//      v - start of first segement
//      v
//    [ 0, 1, 2, 3, 4, 5 ]
//               ^
//               ^ - Start of the second segment

// This class makes an assumption that each index has an upper and lower pair.
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

pub const SegmentPair = struct {
    lower : u32 = 0,
    upper : u32 = 0
};

///////////////////////////////////////////////////////
// Split SegmentPairs into a contiguous segmented array

fn initializeSegmentMemory(comptime n: usize, pairs: ?[n]SegmentPair) [n * 2]u32 {
   // this is a hack because init will cause an error
   // if we don't return the struct type directly    
    var memory: [n * 2]u32 = undefined;

    if (pairs) |data| 
    {
        var i: usize = 0;
        while(i < n) : (i += 1) {                        
            memory[i]     = data[i].lower;            
            memory[i + n] = data[i].upper;
        }
    }
    else {
        @memset(&memory, 0); // zero seems like a sensible default...
    }
    return memory;
}

/////////////////////////////////////////
// SegmentedStorage Struct Implementation 

 fn SegmentedStorage(comptime segment_size: usize) type {

    return struct {

        const Self = @This();
    
        memory: [segment_size * 2]u32 = undefined,
    
        // init SegmentedMemory from an optional pair array
        pub fn init(pairs: ?[segment_size]SegmentPair) Self {
            return Self {
                .memory = initializeSegmentMemory(segment_size, pairs),
            };
        }
        
        //// segmented index getters
        pub fn getLower(self: Self, i: usize) u32 {
            return self.memory[i];
        }
        pub fn getUpper(self: Self, i: usize) u32 {
            return self.memory[segment_size + i];
        }
    
        //// segemented index setters
        pub fn setLower(self: *Self, i: usize, value: u32) void {
            self.*.memory[i] = value;
        }
        pub fn setUpper(self: *Self, i: usize, value: u32) void {
            self.*.memory[segment_size + i] = value;
        }
    
        //// pairwise setters/getter
        pub fn getPair(self: Self, i: usize) SegmentPair {
            return .{ .lower= self.getLower(i), .upper = self.getUpper(i) };
        }
        pub fn setPair(self: *Self, i: usize, pair: SegmentPair) void {
            self.*.setLower(i, pair.lower);
            self.*.setUpper(i, pair.upper);
        }
        pub fn setPairs(self: *Self, pairs: ?[segment_size]SegmentPair) void {
            self.*.memory = initializeSegmentMemory(segment_size, pairs);
        }

        //// segmented slicing functions
        pub fn sliceLower(self: *const Self) [] const u32 {
            return self.*.memory[0..segment_size];
        }
        pub fn sliceUpper(self: *const Self) [] const u32 {
            return self.*.memory[segment_size..segment_size * 2];
        }
    
        pub fn copyFrom(self: *Self, other: * const Self) void {
            @memcpy(&self.*.memory, &other.*.memory);        
        }
    
        // type sizes... member functions instead?
        pub fn segmentSize() usize {
            return segment_size;
        }
        pub fn capacity() usize {
            return segment_size * 2;
        }
    };
}
    
/////////////////////////////////
//////////// TESTING ////////////

test "Initialization" {
    const std = @import("std");

    var s1 = SegmentedStorage(5).init([_]SegmentPair{
            .{ .lower = 100, .upper = 100 },
            .{ .lower = 101, .upper = 101 },
            .{ .lower = 102, .upper = 102 },
            .{ .lower = 103, .upper = 103 },
            .{ .lower = 104, .upper = 104 },
        });

    var s2 = SegmentedStorage(5).init(null);

    s2.setPair(0, .{ .lower = 100, .upper = 100 });
    s2.setPair(1, .{ .lower = 101, .upper = 101 });
    s2.setPair(2, .{ .lower = 102, .upper = 102 });
    s2.setPair(3, .{ .lower = 103, .upper = 103 });
    s2.setPair(4, .{ .lower = 104, .upper = 104 });

    try std.testing.expect(SegmentedStorage(5).capacity() == 10);
    try std.testing.expect(SegmentedStorage(5).capacity() == 10);
    try std.testing.expect(SegmentedStorage(5).segmentSize() == 5);
    try std.testing.expect(SegmentedStorage(5).segmentSize() == 5);

    try std.testing.expect(s1.getLower(0) == s2.getLower(0));
    try std.testing.expect(s1.getLower(1) == s2.getLower(1));
    try std.testing.expect(s1.getLower(2) == s2.getLower(2));
    try std.testing.expect(s1.getLower(3) == s2.getLower(3));
    try std.testing.expect(s1.getLower(4) == s2.getLower(4));

    try std.testing.expect(s1.getUpper(0) == s2.getUpper(0));
    try std.testing.expect(s1.getUpper(1) == s2.getUpper(1));
    try std.testing.expect(s1.getUpper(2) == s2.getUpper(2));
    try std.testing.expect(s1.getUpper(3) == s2.getUpper(3));
    try std.testing.expect(s1.getUpper(4) == s2.getUpper(4));
}

test "Slicing" {

    const std = @import("std");

    var s1 = SegmentedStorage(3).init([_]SegmentPair{
            .{ .lower = 100, .upper = 200 },
            .{ .lower = 101, .upper = 201 },
            .{ .lower = 102, .upper = 202 },
        });

    { // test lower slice
        const slice = s1.sliceLower();
        try std.testing.expect(slice.len == 3);
        try std.testing.expect(slice[0] == 100);
        try std.testing.expect(slice[1] == 101);
        try std.testing.expect(slice[2] == 102);
    }
    { // test upper slice
        const slice = s1.sliceUpper();
        try std.testing.expect(slice.len == 3);
        try std.testing.expect(slice[0] == 200);
        try std.testing.expect(slice[1] == 201);
        try std.testing.expect(slice[2] == 202);
    }    
}

test "Copying" {

    const std = @import("std");

    var s1 = SegmentedStorage(5).init([_]SegmentPair{
            .{ .lower = 100, .upper = 100 },
            .{ .lower = 101, .upper = 101 },
            .{ .lower = 102, .upper = 102 },
            .{ .lower = 103, .upper = 103 },
            .{ .lower = 104, .upper = 104 },
        });

    var s2 = SegmentedStorage(5).init(null);

    s2.copyFrom(&s1);

    for(s1.memory, s2.memory) |i, j| {
        try std.testing.expect(i == j);
    }
}