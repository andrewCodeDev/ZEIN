
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

const std = @import("std");

pub const HALF_INLINE_SIZE : usize = 5;
pub const FULL_INLINE_SIZE : usize = HALF_INLINE_SIZE * 2;

pub const SegmentPair = struct {
    lower : u32 = 0,
    upper : u32 = 0
};

pub const SegmentedStorage = struct {

    memory: [FULL_INLINE_SIZE]u32,
    used: usize = 0,

    fn init(pairs: ?[] const SegmentPair) SegmentedStorage {
        var storage: SegmentedStorage = undefined;

        storage.clear(); // zero out all memory

        if(pairs) |slice|{            
            const n: usize = @min(slice.len, HALF_INLINE_SIZE);

            var i: usize = 0;

            while(i < n) : (i += 1) {
                storage.addPair(slice[i]);            
            }
        }
        return storage;
    }

    // segmented index getters
    fn getLower(self: * const SegmentedStorage, i: usize) u32 {
        return self.*.memory[i];
    }
    fn getUpper(self: * const SegmentedStorage, i: usize) u32 {
        return self.*.memory[HALF_INLINE_SIZE + i];
    }

    // segemented index setters
    fn setLower(self: *SegmentedStorage, i: usize, value: u32) void {
        self.*.memory[i] = value;
    }
    fn setUpper(self: *SegmentedStorage, i: usize, value: u32) void {
        self.*.memory[HALF_INLINE_SIZE + i] = value;
    }

    // pairwise setter/getter
    fn getPair(self: * const SegmentedStorage, i: usize) SegmentPair {
        return .{ .lower= self.*.getLower(i), .upper = self.*.getUpper(i) };
    }
    fn setPair(self: *SegmentedStorage, i: usize, pair: SegmentPair) void {
        self.*.setLower(i, pair.lower);
        self.*.setUpper(i, pair.upper);
    }
    fn addPair(self: *SegmentedStorage, pair: SegmentPair) void {
        self.*.setPair(self.*.used, pair);
        self.*.used += 1;
    }

    // segmented slicing functions
    fn sliceLower(self: * const SegmentedStorage) [] const u32 {
        return self.*.memory[0..self.*.used];
    }
    fn sliceUpper(self: * const SegmentedStorage) [] const u32 {
        return self.*.memory[HALF_INLINE_SIZE..HALF_INLINE_SIZE + self.*.used];
    }

    fn copyFrom(self: *SegmentedStorage, other: * const SegmentedStorage) void {
        @memcpy(&self.*.memory, &other.*.memory);        
        self.*.used = other.*.used;
    }

    fn clear(self: *SegmentedStorage) void {
        @memset(&self.*.memory, 0);
        self.*.used = 0;
    }

    fn empty(self: * const SegmentedStorage) bool {
        return self.*.used == 0;
    }

    fn size(self: * const SegmentedStorage) usize {
        return self.*.used;
    }
};

/////////////////////////////////
//////////// TESTING ////////////

test "Initialization" {

    var s1 = SegmentedStorage.init(&[_]SegmentPair{
            .{ .lower = 100, .upper = 100 },
            .{ .lower = 101, .upper = 101 },
            .{ .lower = 102, .upper = 102 },
            .{ .lower = 103, .upper = 103 },
            .{ .lower = 104, .upper = 104 },
        });

    var s2 = SegmentedStorage.init(null);
    s2.addPair(.{ .lower = 100, .upper = 100 });
    s2.addPair(.{ .lower = 101, .upper = 101 });
    s2.addPair(.{ .lower = 102, .upper = 102 });
    s2.addPair(.{ .lower = 103, .upper = 103 });
    s2.addPair(.{ .lower = 104, .upper = 104 });

    try std.testing.expect(s1.size() == 5);
    try std.testing.expect(s1.size() == s2.size());

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

    var s1 = SegmentedStorage.init(&[_]SegmentPair{
            .{ .lower = 100, .upper = 100 },
            .{ .lower = 101, .upper = 101 },
            .{ .lower = 102, .upper = 102 },
        });

    var s2 = SegmentedStorage.init(null);
    s2.addPair(.{ .lower = 100, .upper = 100 });
    s2.addPair(.{ .lower = 101, .upper = 101 });
    s2.addPair(.{ .lower = 102, .upper = 102 });
    
    try std.testing.expect(s1.size() == 3);
    try std.testing.expect(s1.size() == s2.size());

    var count: usize = 0;

    for(s1.sliceLower(), s2.sliceLower()) |i, j| {
        try std.testing.expect(i == j);
        count += 1;
    }
    try std.testing.expect(count == 3);
    
    count = 0;
    
    for(s1.sliceUpper(), s2.sliceUpper()) |i, j| {
        try std.testing.expect(i == j);
        count += 1;
    }

    try std.testing.expect(count == 3);
}

test "Copying" {

    var s1 = SegmentedStorage.init(&[_]SegmentPair{
            .{ .lower = 100, .upper = 100 },
            .{ .lower = 101, .upper = 101 },
            .{ .lower = 102, .upper = 102 },
            .{ .lower = 103, .upper = 103 },
            .{ .lower = 104, .upper = 104 },
        });

    var s2 = SegmentedStorage.init(null);

    s2.copyFrom(&s1);

    for(s1.memory, s2.memory) |i, j| {
        try std.testing.expect(i == j);
    }
}