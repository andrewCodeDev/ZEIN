# ZEIN
Zig-based implementation of tensors


# Project Structure
The main ZEIN file provides an interface for the library implementation.

The implementations are in the Core folder. They will be labled as "VX" where X is the verion number.

This enables swapping out the implentation for both testing purposes and for providing variable behavior based on the CoreTensor version that is being used.
Additionally, this helps with backwards compatibility as this library may be aggressively changed. Likewise, some decisions may not be supported on all 
architectures (AVX or CUDA, for instance).
