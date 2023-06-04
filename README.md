# ZEIN
Zig-based implementation of tensors

# Project Structure
The main ZEIN file provides an interface for the library implementation.

The implementations are in the Core folder. They will be labled as "VX" where X is the verion number.

This enables swapping out the implentation for both testing purposes and for providing variable behavior based on the Core version that is being used.
Additionally, this helps with backwards compatibility as this library may be aggressively changed. Likewise, some decisions may not be supported on all 
architectures (AVX or CUDA, for instance).

# Tensor Operations
It is important to make a distinction between tensor operations that create a new tensor 
and ones that modify and existing tensor. Here are the general rules:

  Free functions will have the ability to create new tensors. The exact mechanism of
  how this will take place is currently undecided.

  Member functions will operate on a tensor inplace (that is to say, it will use the
  memory of that same tensor to perform the operation).

# Additonal information coming soon.
This library is still in the beginning phases. If you want to contribute, please
contact me! This is a big job and I'll take the help!
