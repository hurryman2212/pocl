/* OpenCL runtime library: clSVMFree()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "pocl_util.h"
#include "pocl_debug.h"
#include "utlist.h"

/* POCL_EXPORT is for loadable drivers. */
POCL_EXPORT CL_API_ENTRY void CL_API_CALL
POname(clSVMFree)(cl_context context,
                  void *svm_pointer) CL_API_SUFFIX__VERSION_2_0
{
  if (!IS_CL_OBJECT_VALID (context))
    {
      POCL_MSG_ERR ("Invalid cl_context\n");
      return;
    }

  if (context->svm_allocdev == NULL)
    {
      POCL_MSG_ERR ("None of the devices in this context is SVM-capable\n");
      return;
    }

  if (svm_pointer == NULL)
    {
      POCL_MSG_WARN ("NULL pointer passed\n");
      return;
    }

  POCL_LOCK_OBJ (context);
  pocl_raw_ptr *item
      = pocl_raw_ptr_set_lookup_with_vm_ptr (context->raw_ptrs, svm_pointer);
  if (item && item->vm_ptr == svm_pointer && item->kind == POCL_RAW_PTR_SVM)
    pocl_raw_ptr_set_remove (context->raw_ptrs, item);
  else
    item = NULL;
  POCL_UNLOCK_OBJ (context);

  if (item == NULL)
    {
      POCL_MSG_ERR ("can't find pointer in list of allocated SVM pointers");
      return;
    }

  cl_device_id owner = context->svm_allocdev;
  if (item->shadow_cl_mem)
    pocl_release_owned (POCL_RELEASE_MEM, item->shadow_cl_mem);
  POCL_MEM_FREE (item);
  if (owner->ops->free_pointer)
    owner->ops->free_pointer (owner, context, svm_pointer, CL_FALSE, CL_TRUE);
  else
    owner->ops->svm_free (owner, svm_pointer);
  POCL_ATOMIC_DEC (svm_buffer_c);
  /* The allocation may hold the final context reference. */
  pocl_release_owned (POCL_RELEASE_CONTEXT, context);
}

POsym (clSVMFree)
