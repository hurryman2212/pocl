/* OpenCL runtime library: clLinkProgram()

   Copyright (c) 2017 Michal Babej / Tampere University of Technology

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

#include "pocl_cl.h"
#include "pocl_shared.h"
#include "pocl_util.h"

/** Caller holds the program lock or a program mutation guard. */
static int
has_link_input (cl_program program, unsigned index)
{
  cl_program_binary_type type
      = pocl_program_device_binary_type (program, index);
  cl_build_status status = program->device_states
                               ? program->device_states[index].status
                               : program->build_status;
  if (status != CL_BUILD_SUCCESS && status != CL_BUILD_NONE)
    return 0;
  if (type == CL_PROGRAM_BINARY_TYPE_LIBRARY
      || type == CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT)
    return 1;
  /** A validator may defer classification until allocation-free validation
   * has been followed by the driver's native LOAD step.
   */
  return status == CL_BUILD_NONE && type == CL_PROGRAM_BINARY_TYPE_NONE
         && ((program->binaries[index] && program->binary_sizes[index])
             || (program->pocl_binaries[index]
                 && program->pocl_binary_sizes[index]));
}

CL_API_ENTRY cl_program CL_API_CALL
POname (clLinkProgram) (
    cl_context context, cl_uint num_devices, const cl_device_id *device_list,
    const char *options, cl_uint num_input_programs,
    const cl_program *input_programs,
    void (CL_CALLBACK *pfn_notify) (cl_program program, void *user_data),
    void *user_data, cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
  int errcode = CL_SUCCESS;
  cl_program program = NULL;
  cl_device_id *unique_devlist = NULL;
  cl_device_id *link_devices = NULL;
  cl_program *inputs = NULL;
  unsigned guarded = 0;
  unsigned link_count = 0;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);
  POCL_GOTO_ERROR_COND ((!num_input_programs || !input_programs),
                        CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND ((num_devices && !device_list)
                            || (!num_devices && device_list),
                        CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND ((!pfn_notify && user_data), CL_INVALID_VALUE);

  inputs = calloc (num_input_programs, sizeof (*inputs));
  POCL_GOTO_ERROR_COND ((!inputs), CL_OUT_OF_HOST_MEMORY);
  memcpy (inputs, input_programs, num_input_programs * sizeof (*inputs));
  for (unsigned i = 0; i < num_input_programs; ++i)
    {
      cl_program input = inputs[i];
      POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (input)), CL_INVALID_PROGRAM);
      POCL_LOCK_OBJ (input);
      if (input->context != context)
        errcode = CL_INVALID_CONTEXT;
      else if (input->build_in_progress)
        errcode = CL_INVALID_OPERATION;
      else if (input->kernel_creations == CL_UINT_MAX)
        errcode = CL_OUT_OF_RESOURCES;
      else
        {
          int available = 0;
          for (unsigned d = 0; d < input->num_devices; ++d)
            available |= has_link_input (input, d);
          if (!available)
            errcode = CL_INVALID_OPERATION;
        }
      if (errcode == CL_SUCCESS)
        {
          ++input->kernel_creations;
          POCL_RETAIN_OBJECT_UNLOCKED (input);
          ++guarded;
        }
      POCL_UNLOCK_OBJ (input);
      if (errcode != CL_SUCCESS)
        goto ERROR;
    }

  if (!num_devices)
    {
      num_devices = context->num_devices;
      device_list = context->devices;
    }
  for (unsigned i = 0; i < num_devices; ++i)
    {
      POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (device_list[i])),
                            CL_INVALID_DEVICE);
      int found = 0;
      for (unsigned d = 0; d < context->num_devices; ++d)
        found |= context->devices[d] == device_list[i];
      POCL_GOTO_ERROR_COND ((!found), CL_INVALID_DEVICE);
    }
  unsigned unique_count = 0;
  unique_devlist
      = pocl_unique_device_list (device_list, num_devices, &unique_count);
  POCL_GOTO_ERROR_COND ((!unique_devlist), CL_OUT_OF_HOST_MEMORY);
  num_devices = unique_count;
  device_list = unique_devlist;
  link_devices = calloc (num_devices, sizeof (*link_devices));
  POCL_GOTO_ERROR_COND ((!link_devices), CL_OUT_OF_HOST_MEMORY);

  for (unsigned d = 0; d < num_devices; ++d)
    {
      unsigned available = 0;
      for (unsigned i = 0; i < num_input_programs; ++i)
        {
          cl_program input = inputs[i];
          unsigned index = pocl_program_find_device (input, device_list[d]);
          if (index == CL_UINT_MAX)
            continue;
          POCL_LOCK_OBJ (input);
          available += has_link_input (input, index);
          POCL_UNLOCK_OBJ (input);
        }
      POCL_GOTO_ERROR_COND ((available && available != num_input_programs),
                            CL_INVALID_OPERATION);
      if (available)
        {
          POCL_GOTO_ERROR_COND ((!device_list[d]->linker_available),
                                CL_LINKER_NOT_AVAILABLE);
          link_devices[link_count++] = device_list[d];
        }
    }

  /** Association includes requested devices with no linkable inputs. */
  program = create_program_skeleton (context, num_devices, device_list, NULL,
                                     NULL, NULL, &errcode, 1);
  if (errcode != CL_SUCCESS)
    goto ERROR;
  if (link_count)
    errcode = compile_and_link_program (
        0, 1, program, link_count, link_devices, options, 0, NULL, NULL,
        num_input_programs, inputs, NULL, NULL);
  /** No callback is issued for requests rejected before linking can begin. */
  if (pfn_notify
      && (errcode == CL_SUCCESS || errcode == CL_LINK_PROGRAM_FAILURE))
    pfn_notify (program, user_data);

ERROR:
  for (unsigned i = 0; i < guarded; ++i)
    {
      POCL_LOCK_OBJ (inputs[i]);
      --inputs[i]->kernel_creations;
      POCL_UNLOCK_OBJ (inputs[i]);
      pocl_release_owned (POCL_RELEASE_PROGRAM, inputs[i]);
    }
  POCL_MEM_FREE (inputs);
  POCL_MEM_FREE (link_devices);
  POCL_MEM_FREE (unique_devlist);
  if (errcode_ret)
    *errcode_ret = errcode;
  if (errcode == CL_SUCCESS || errcode == CL_LINK_PROGRAM_FAILURE)
    return program;
  pocl_release_owned (POCL_RELEASE_PROGRAM, program);
  return NULL;
}

POsym (clLinkProgram)
