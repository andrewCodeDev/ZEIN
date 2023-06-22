////////////////////////////////////////////////////////////////
// Expression Parsing for Einsum style string expressions.

// Currently, the expression parser does not tolerate
// whitespace in expressions. This will be reviewed
// at a later date, but currently is not required to
// create well-formed strings.

// parser utility functions. These functions are intended
// to be executed at comptime.

const SizesType = @import("SizesAndStrides.zig").SizeAndStride.ValueType;

fn between(comptime value: u8, comptime lower: u8, comptime upper: u8) bool {
    return lower <= value and value <= upper;
}

fn isAlpha(comptime value: u8) bool {
    return between(value, 65, 90) or between(value, 97, 122); // [91, 96] are: [\]^_`
}

fn allAlpha(comptime str: [] const u8) bool {
    comptime var i: usize = 0;
    inline while(i < str.len) : (i += 1) {
        if(!isAlpha(str[i])) { return false; }
    }
    return true;
}

// check that a permutation is both full and accounted for
fn isPermutation(comptime source: [] const u8, comptime target: [] const u8) bool {

    if(source.len != target.len) {
        return false;
    }
    if(source.len == 0) { // the empty set is a permutation of itself
        return true;
    }
    // create mask for proper permutation
    const full: usize = (1 << source.len) - 1;
    comptime var i_mask: usize = 0;
    comptime var j_mask: usize = 0;

    comptime var i: usize = 0;
    comptime var j: usize = 0;
    inline while(i < source.len) : ({ i += 1; j = 0; }) {
        inline while(j < target.len) : (j += 1) {
            if(source[i] == target[j]) { 
                i_mask |= (1 << i);
                j_mask |= (1 << j); 
            }
        }
    }
    return i_mask == j_mask and i_mask == full;
}

pub fn IndicesPair(comptime lRank: usize, comptime rRank: usize) type {
    return struct {
        lhs : [lRank]SizesType = undefined,
        rhs : [rRank]SizesType = undefined,
    };
}

const ArrowOp = struct {
    tail: usize = 0,
    head: usize = 0,  
};

fn findArrowOp(str: [] const u8) ArrowOp { 
    // reference for array operator
    const arrow: [] const u8 = "->";

    comptime var head: usize = 0;
    comptime var tail: usize = 0;    
    comptime var index: usize = 0;
    inline while(index < str.len) : (index += 1) {
        if(str[index] == arrow[0]) { tail = index; }
        if(str[index] == arrow[1]) { head = index; }
    }

    ///////////////////////////////////////
    // check for valid infix arrow operator

    if((tail + 1) != head) {
        @compileError("Malformed arrow operator: " ++ str);
    }
    if(tail == 0 or head > (str.len - 2)) {
        @compileError("Arrow must be used as infix operator: " ++ str);
    }

    return ArrowOp{ .tail = tail, .head = head };
}

// Contraction parsing is expects strings of the form:
//
//     example: ijk->jk
//
// The expressions must be larger on the left-operand than
// the right operand (denoting contracted indices).
//
// The left and right operands must be alpha-characters.

pub fn permutateParse(
    comptime Rank: usize,
    comptime str: [] const u8
) [Rank]SizesType {

    const arrow = comptime findArrowOp(str);
    const lhs = str[0..arrow.tail];
    const rhs = str[arrow.head + 1..];

    if(lhs.len != Rank) {
        @compileError("Left operand is not equal to the rank: " ++ lhs);
    }
    if(rhs.len != Rank) {
        @compileError("Right operand is not equal to the rank: " ++ rhs);
    }
    if(!comptime allAlpha(lhs)) {
        @compileError("Non-alphabetical character found in: " ++ lhs);
    }
    if(!comptime allAlpha(rhs)) {
        @compileError("Non-alphabetical character found in: " ++ rhs);
    }
    if(!comptime isPermutation(lhs, rhs)) {
        @compileError("Permutate requires left and right operands to be permutations of eachother." ++ str);
    }

    ////////////////////////////////////////
    // build permutation contraction indices

    comptime var i: usize = 0;
    comptime var j: usize = 0;
    comptime var indices: [Rank]SizesType = undefined;

    inline while(i < Rank) : ({ i += 1; j = 0; }) {
        inline while(j < Rank) : (j += 1) {
            if (rhs[i] == lhs[j]) {
                indices[i] = j;
                break;
            }
        }
    }
    return indices;
}

pub fn contractionParse(
    comptime lRank: usize,
    comptime rRank: usize,
    comptime str: [] const u8
) IndicesPair(lRank, rRank) {
    comptime var index: usize = 0;

    // reference for array operator
    const arrow: [] const u8 = "->";

    comptime var a: usize = 0;
    comptime var b: usize = 0;

    // mark one before the arrow and one after the arrow
    inline while(index < str.len) : (index += 1) {
        if(str[index] == arrow[0]) { a = index; }
        if(str[index] == arrow[1]) { b = index; }
    }

    ///////////////////////////////////////
    // check for valid infix arrow operator

    if((a + 1) != b) {
        @compileError("Malformed arrow operator: " ++ str);
    }
    if(a == 0 or b > (str.len - 2)) {
        @compileError("Arrow must be used as infix operator: " ++ str);
    }

    const lhs = str[0..a];
    const rhs = str[b+1..];

    if (lhs.len == 0) {
        @compileError("Empty left-side operand: " ++ str);
    }
    if (rhs.len == 0) {
        @compileError("Empty right-side operand: " ++ str);
    }
    if(lhs.len != lRank) {
        @compileError("Provided indices do not match left-side operand rank: " ++ lhs);
    }
    if(rhs.len != rRank) {
        @compileError("Provided indices do not match right-side operand rank: " ++ rhs);
    }
    if(!comptime allAlpha(lhs)) {
        @compileError("Non-alphabetical character found in: " ++ lhs);
    }
    if(!comptime allAlpha(rhs)) {
        @compileError("Non-alphabetical character found in: " ++ rhs);
    }

    ////////////////////////////////////////
    // build permutation contraction indices

    comptime var x_indices: [lhs.len]u32 = undefined;
    comptime var y_indices: [rhs.len]u32 = undefined;
    comptime var remainder: [lhs.len + rhs.len]u32 = undefined;
    comptime var char: u8 = undefined;
    comptime var match: u32 = 0;
    comptime var rhs_i: u32 = 0;
    comptime var rem_i: u32 = 0;
    comptime var found: bool = false;

    index = 0;
    inline while(index < lhs.len) : (index += 1) {

        // matched + unmatched = total
        if(match == rhs.len and rem_i == remainder.len) {
             break; 
        }

        char = lhs[index];

        found = false;

        // try to match the current char
        // in both rhs and lhs operands
        
        rhs_i = 0;
        inline while(rhs_i < rhs.len) : (rhs_i += 1) {
            if (rhs[rhs_i] == char) {
                x_indices[match] = index;
                y_indices[match] = rhs_i;
                found = true;
                match += 1;
                break;
            }
        }

        // if no match, add to remainder
        
        if(!found) {
            remainder[rem_i] = index;
            rem_i += 1;
        }
    }

    if(match != rhs.len) {
        @compileError("Unmatched dimensions between operands:" ++ str);
    }

    rem_i = 0;
    index = rhs.len;
    inline while(index < lhs.len) : ({ index += 1; rem_i += 1; }){
        x_indices[index] = remainder[rem_i];
    }
        
    return IndicesPair(lRank, rRank){ .lhs = x_indices, .rhs = y_indices };
}