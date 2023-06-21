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

pub fn IndicesPair(comptime lRank: usize, comptime rRank: usize) type {
    return struct {
        lhs : [lRank]SizesType = undefined,
        rhs : [rRank]SizesType = undefined,
    };
}

// Contraction parsing is expects strings of the form:
//
//     example: ijk->jk
//
// The expressions must be larger on the left-operand than
// the right operand (denoting contracted indices).
//
// The left and right operands must be alpha-characters.

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