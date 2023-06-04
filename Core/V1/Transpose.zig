
//OPERATOR_SCHEMA(Transpose)
//    .NumInputs(1)
//    .NumOutputs(1)
//    .TensorInferenceFunction([](const OperatorDef& def,
//                                const vector<TensorShape>& in) {
//      ArgumentHelper helper(def);
//      vector<int> axes = helper.GetRepeatedArgument<int>("axes");
//      vector<TensorShape> out(1);
//      out[0].set_data_type(in[0].data_type());
//
//      if (axes.empty()) {
//        for (auto axis = in [0].dims().rbegin(); axis != in[0].dims().rend();
//             ++axis) {
//          out[0].add_dims(*axis);
//        }
//      } else {
//        auto tensor_size = in[0].dims().size();
//        auto valid_axes =
//            std::all_of(axes.begin(), axes.end(), [&tensor_size](int& axis) {
//              return axis >= 0 && axis < tensor_size;
//            });
//
//        CAFFE_ENFORCE(valid_axes, "Axes argument passed in had invalid values");
//        CAFFE_ENFORCE(
//            // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
//            axes.size() == tensor_size,
//            "Axes argument passed in had the incorrect size");
//
//        // NOLINTNEXTLINE(modernize-loop-convert)
//        for (auto axis = axes.begin(); axis != axes.end(); ++axis) {
//          out[0].add_dims(in[0].dims().Get(*axis));
//        }
//      }
//
//      return out;
//    })

// make this an enum at some point
const SizesAndStrides = @import("SizesAndStrides.zig").SizeAndStride;

const SizeAndStride = @import("SizesAndStrides.zig").SizeAndStride;

pub fn transposeInput(    
    comptime rank_x : usize,
    x_s: *SizesAndStrides(rank_x),
    permutation: *const [rank_x]u32
) void {
    var tmp = SizesAndStrides(rank_x).init(null);

    var i :usize = 0;
    for(permutation) |p| {
        tmp.setSizeAndStride(i, x_s.*.getSizeAndStride(p));
        i += 1;
    }
    x_s.* = tmp;  
}

pub fn transposeToOutput(
    comptime rank_x : usize,
    comptime data_type_x: type,
    x_d: *[] data_type_x,
    x_s: *SizesAndStrides(rank_x),
    // tensor y components
    comptime rank_y : usize,
    comptime data_type_y: type,
    y_d: *[] data_type_y,
    y_s: *SizesAndStrides(rank_y),
    // output permutation of axis
    permutation: * const [rank_x]u32
) void {
    // todo - consider enabling casting for the case of
    // producing a new tensor? Could be a good idea,
    // could be a terrible idea. Skipping for now.
    if(data_type_x != data_type_y) {
        @compileError("Cannot transpose to tensors of different types - Core:V1");
    }
    if(rank_x != rank_y) {
        @compileError("Cannot transpose to tensors of different ranks - Core:V1");
    }
    if(x_d.*.len == 0){
        return; // should come up with a better check here
    }

    // share data if input is different than output
    if(x_d.ptr != y_d.ptr) {
        y_d.* = x_d.*;
    }

    // other-permutation branch... modify tensor y
    if(x_s.ptr != y_s.ptr){
        var i :usize = 0;
        for(permutation) |*p| {
            y_s.*.setSizeAndStride(i, x_s.*.getSizeAndStride(p));
            i += 1;
        }
    }
    else {
        transposeInput(rank_x, x_d, permutation);
    }
}