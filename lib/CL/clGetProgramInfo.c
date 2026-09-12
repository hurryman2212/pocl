/* OpenCL runtime library: clGetProgramInfo()

   Copyright (c) 2011 Erik Schnetter

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include <string.h>
#include "pocl_llvm.h"
#include "pocl_util.h"
#include "pocl_cache.h"
#include "pocl_binary.h"
#include "pocl_shared.h"

/******************************************************************************/

static cl_int
prepare_binary (cl_program program, unsigned index)
{
  if (program->device_states && program->device_states[index].status != CL_BUILD_SUCCESS)
    return CL_SUCCESS;
  cl_device_id device = program->devices[index];
  if (device->ops->build_poclbinary)
    {
      ++program->kernel_creations;
      POCL_UNLOCK_OBJ (program);
      cl_int status = device->ops->build_poclbinary (program, index);
      POCL_LOCK_OBJ (program);
      --program->kernel_creations;
      if (status != CL_SUCCESS)
        return status;
    }
  if (!program->pocl_binaries[index] && program->binaries[index] &&
      !device->ops->external_build_options)
    {
      if (!pocl_binary_sizeof_binary (program, index))
        return CL_OUT_OF_HOST_MEMORY;
    }
  return CL_SUCCESS;
}

static cl_int
get_binary_sizes (cl_program program, size_t *sizes)
{
  for (unsigned index = 0; index < program->associated_num_devices; ++index)
    {
      sizes[index] = 0;
      if (program->device_states && program->device_states[index].status == CL_BUILD_ERROR)
        continue;
      cl_int status = prepare_binary (program, index);
      if (status != CL_SUCCESS)
        return status;
      sizes[index] = program->pocl_binaries[index] ? program->pocl_binary_sizes[index] : program->binary_sizes[index];
    }
  return CL_SUCCESS;
}

static cl_int
get_binaries (cl_program program, unsigned char **binaries)
{
  for (unsigned index = 0; index < program->associated_num_devices; ++index)
    {
      if (!binaries[index] || (program->device_states && program->device_states[index].status == CL_BUILD_ERROR))
        continue;
      cl_int status = prepare_binary (program, index);
      if (status != CL_SUCCESS)
        return status;
      if (program->pocl_binaries[index])
        memcpy (binaries[index], program->pocl_binaries[index], program->pocl_binary_sizes[index]);
      else if (program->binaries[index])
        memcpy (binaries[index], program->binaries[index], program->binary_sizes[index]);
    }
  return CL_SUCCESS;
}

/******************************************************************************/

static cl_int
clGetProgramInfo_locked (cl_program program, cl_program_info param_name,
                         size_t param_value_size, void *param_value,
                         size_t *param_value_size_ret)
{
  unsigned i;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (program)), CL_INVALID_PROGRAM);

  if (program->build_in_progress
      && (param_name == CL_PROGRAM_BINARY_SIZES
          || param_name == CL_PROGRAM_BINARIES
          || param_name == CL_PROGRAM_NUM_KERNELS
          || param_name == CL_PROGRAM_KERNEL_NAMES))
    return CL_INVALID_OPERATION;

  switch (param_name)
    {
    case CL_PROGRAM_REFERENCE_COUNT:
      POCL_RETURN_GETINFO (cl_uint, (cl_uint)program->pocl_refcount);
    case CL_PROGRAM_CONTEXT:
      POCL_RETURN_GETINFO (cl_context, program->context);

    case CL_PROGRAM_SOURCE:
      {
        const char *source = program->source;
        if (source == NULL)
          source = "";

        size_t source_size
            = program->source_size ? program->source_size : strlen (source);
        POCL_RETURN_GETINFO_SIZE (source_size + 1, source);
      }

    case CL_PROGRAM_BINARY_SIZES:
    case CL_PROGRAM_BINARIES:
      {
        size_t bytes = program->associated_num_devices *
            (param_name == CL_PROGRAM_BINARY_SIZES ? sizeof (size_t) : sizeof (unsigned char *));
        if (param_value)
          {
            POCL_RETURN_ERROR_COND (param_value_size < bytes, CL_INVALID_VALUE);
            cl_int status = param_name == CL_PROGRAM_BINARY_SIZES
                ? get_binary_sizes (program, param_value) : get_binaries (program, param_value);
            if (status != CL_SUCCESS)
              return status;
          }
        if (param_value_size_ret)
          *param_value_size_ret = bytes;
        return CL_SUCCESS;
      }

    case CL_PROGRAM_IL:
      {
        POCL_RETURN_GETINFO_INNER (program->program_il_size,
                                   memcpy (param_value, program->program_il,
                                           program->program_il_size));
      }

    case CL_PROGRAM_NUM_DEVICES:
      POCL_RETURN_GETINFO (cl_uint, program->associated_num_devices);

    case CL_PROGRAM_DEVICES:
      {
        size_t const value_size
            = sizeof (cl_device_id) * program->associated_num_devices;
        POCL_RETURN_GETINFO_SIZE (value_size, program->associated_devices);
      }

    case CL_PROGRAM_NUM_KERNELS:
      {
        POCL_RETURN_ERROR_ON (
            (!pocl_program_has_executable (program)),
            CL_INVALID_PROGRAM_EXECUTABLE,
            "This information is only available after a "
            "successful program executable has been built\n");
        POCL_RETURN_GETINFO (size_t, program->num_kernels);
      }

    case CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT:
      {
        POCL_RETURN_GETINFO (cl_bool, CL_FALSE);
      }

    case CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT:
      {
        POCL_RETURN_GETINFO (cl_bool, CL_FALSE);
      }

    case CL_PROGRAM_KERNEL_NAMES:
      {
        POCL_RETURN_ERROR_ON (
            (!pocl_program_has_executable (program)),
            CL_INVALID_PROGRAM_EXECUTABLE,
            "This information is only available after a "
            "successful program executable has been built\n");

        /* Note: In the specification (2.0) of the other XXXInfo
          functions, param_value_size_ret is described as follows:

          > param_value_size_ret returns the actual size in bytes of data
          > being *queried* by param_value.

          while in GetProgramInfo and APIs defined later in the
          documentation, it is:

          > param_value_size_ret returns the actual size in bytes of data
          > *copied* to param_value.

          it reads as if the spec allows the implementation to stop copying
          the string at an arbitrary point where the limit
          (param_value_size) is reached, but that's not the case. When it
          happens, it should instead raise an error CL_INVALID_VALUE.

          Also note the specification of the param_value_size_ret to param_name
          CL_PROGRAM_SOURCE.  It says "The actual number of characters that
          represents[sic] the program source code including the null terminator
          is returned in param_value_size_ret." By an analogy, it is sane to
          return the size of entire concatenated string, not the size of
          bytes copied (partially).

          Also note the specification of GetPlatformInfo +
          CL_PLATFORM_EXTENSIONS.  it refers to "param_value_size_ret" as
          the actual size in bytes of data being *queried*, and its
          description of param_value_size is the same.

          --- guicho271828
       */
        size_t num_kernels = program->num_kernels;
        size_t size = 0;

        /* optimized for clarity */
        for (i = 0; i < num_kernels; ++i)
          {
            size += strlen (program->kernel_meta[i].name);
            if (i != num_kernels - 1)
              size += 1; /* a semicolon */
          }
        size += 1; /* a NULL */
        if (param_value_size_ret)
          *param_value_size_ret = size;
        if (param_value)
          {
            /* only when param_value is non-NULL */
            if (size > param_value_size)
              return CL_INVALID_VALUE;
            for (i = 0; i < num_kernels; ++i)
              {
                if (i == 0)
                  strcpy (
                      (char *)param_value,
                      program->kernel_meta[i].name); /* copy including NULL */
                else
                  strcat ((char *)param_value, program->kernel_meta[i].name);
                if (i != num_kernels - 1)
                  strcat ((char *)param_value, ";");
              }
          }
        return CL_SUCCESS;
      }
    default:
      POCL_RETURN_ERROR (CL_INVALID_VALUE, "Parameter %i not implemented\n",
                         param_name);
    }
}
CL_API_ENTRY cl_int CL_API_CALL
POname (clGetProgramInfo) (cl_program program, cl_program_info param_name,
                           size_t param_value_size, void *param_value,
                           size_t *param_value_size_ret)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (program)), CL_INVALID_PROGRAM);
  POCL_LOCK_OBJ (program);
  POCL_RETAIN_OBJECT_UNLOCKED (program);
  cl_int status
      = clGetProgramInfo_locked (program, param_name, param_value_size,
                                 param_value, param_value_size_ret);
  POCL_UNLOCK_OBJ (program);
  pocl_release_owned (POCL_RELEASE_PROGRAM, program);
  return status;
}
POsym (clGetProgramInfo)
