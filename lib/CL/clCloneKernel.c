/* OpenCL runtime library: clCloneKernel()

   Copyright (c) 2022 Michal Babej / Tampere University

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

#include "pocl_binary.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_file_util.h"
#include "pocl_util.h"
#include "utlist.h"

CL_API_ENTRY cl_kernel CL_API_CALL
POname (clCloneKernel) (cl_kernel source_kernel,
                        cl_int *errcode_ret) CL_API_SUFFIX__VERSION_2_1
{
  cl_int errcode = CL_SUCCESS;
  cl_kernel kernel = NULL;
  int retained = 0;
  POCL_GOTO_ERROR_COND (!IS_CL_OBJECT_VALID (source_kernel),
                        CL_INVALID_KERNEL);
  POCL_RETAIN_OBJECT (source_kernel);
  retained = 1;
  /* Reuse normal construction, including native resources and mutation guards.
   */
  kernel = POname (clCreateKernel) (source_kernel->program,
                                    source_kernel->name, &errcode);
  if (!kernel)
    goto ERROR;
  for (unsigned i = 0; i < kernel->meta->num_args; ++i)
    {
      const pocl_argument *source = &source_kernel->dyn_arguments[i];
      pocl_argument *destination = &kernel->dyn_arguments[i];
      *destination = *source;
      destination->value = NULL;
      if (!source->value)
        continue;
      if (kernel->meta->total_argument_storage_size)
        destination->value = kernel->dyn_argument_offsets[i];
      else
        {
          size_t alignment = pocl_size_ceil2 (source->size);
          if (alignment > MAX_EXTENDED_ALIGNMENT)
            alignment = MAX_EXTENDED_ALIGNMENT;
          size_t allocation_size
              = source->size < alignment ? alignment : source->size;
          destination->value
              = pocl_aligned_malloc (alignment, allocation_size);
          POCL_GOTO_ERROR_COND (!destination->value, CL_OUT_OF_HOST_MEMORY);
        }
      memcpy (destination->value, source->value, source->size);
    }
  kernel->can_access_all_raw_buffers_indirectly
      = source_kernel->can_access_all_raw_buffers_indirectly;
  pocl_ptr_list *source;
  DL_FOREACH (source_kernel->indirect_raw_ptrs, source)
  {
    pocl_ptr_list *entry = calloc (1, sizeof (*entry));
    POCL_GOTO_ERROR_COND (!entry, CL_OUT_OF_HOST_MEMORY);
    entry->ptr = source->ptr;
    entry->param_name = source->param_name;
    DL_APPEND (kernel->indirect_raw_ptrs, entry);
  }
  goto FINISH;
ERROR:
  if (kernel)
    {
      pocl_release_owned (POCL_RELEASE_KERNEL, kernel);
      kernel = NULL;
    }
FINISH:
  if (retained)
    pocl_release_owned (POCL_RELEASE_KERNEL, source_kernel);
  if (errcode_ret)
    *errcode_ret = errcode;
  return kernel;
}
POsym (clCloneKernel)
