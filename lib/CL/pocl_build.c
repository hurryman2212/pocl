/* OpenCL runtime library: compile_and_link_program()

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos,
                 2011-2023 Pekka Jääskeläinen

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal
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
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <assert.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stddef.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef _WIN32
#include <unistd.h>
#else
#include "vccompat.hpp"
#endif

#include "pocl_cl.h"
#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif
#include "pocl_util.h"
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include "config.h"
#include "pocl_runtime_config.h"
#include "pocl_binary.h"
#include "pocl_shared.h"

#define REQUIRES_CR_SQRT_DIV_ERR                                              \
  "-cl-fp32-correctly-rounded-divide-sqrt build option "                      \
  "was specified, but CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT "                   \
  "is not set for device"

/* supported compiler parameters which should pass to the frontend directly
   by using -Xclang */
static const char cl_parameters[] = "-cl-single-precision-constant "
                                    "-cl-fp32-correctly-rounded-divide-sqrt "
                                    "-cl-opt-disable "
                                    "-cl-mad-enable "
                                    "-cl-unsafe-math-optimizations "
                                    "-cl-finite-math-only "
                                    "-cl-fast-relaxed-math "
                                    "-cl-std=CL1.2 "
                                    "-cl-std=CL1.1 "
                                    "-cl-std=CL2.0 "
                                    "-cl-std=CL2.1 "
                                    "-cl-std=CL2.2 "
                                    "-cl-std=CL3.0 "
                                    "-cl-std=CL3.1 "
                                    "-cl-kernel-arg-info "
                                    "-cl-strict-aliasing "
                                    "-cl-denorms-are-zero "
                                    "-cl-no-signed-zeros "
                                    "-w "
                                    "-g "
                                    "-Werror ";

/*
static const char cl_library_link_options[] =
  "-create-library "
  "-enable-link-options ";
*/

static const char cl_program_link_options[] = "-cl-denorms-are-zero "
                                              "-cl-no-signed-zeros "
                                              "-cl-unsafe-math-optimizations "
                                              "-cl-finite-math-only "
                                              "-cl-fast-relaxed-math ";

/* TODO: In case of a PoCL-R, we should pass on unhandled
   target-specific/extension-specific options to the
   native driver's clBuildProgram() call. Might be difficult
   to filter the kept ones as the set of target devices can
   be diverse. */
static const char cl_parameters_not_yet_supported_by_clang[]
    = "-cl-uniform-work-group-size "
      "-cl-no-subgroup-ifp "
      "-cl-intel-no-prera-scheduling";

#define MEM_ASSERT(x, err_jmp)                                                \
  do                                                                          \
    {                                                                         \
      if (x)                                                                  \
        {                                                                     \
          errcode = CL_OUT_OF_HOST_MEMORY;                                    \
          goto err_jmp;                                                       \
        }                                                                     \
    }                                                                         \
  while (0)

// append token, growing modded_options, if necessary, by max(strlen(token)+1,
// 256)
#define APPEND_TOKEN()                                                        \
  do                                                                          \
    {                                                                         \
      needed = strlen (token) + 1;                                            \
      assert (size > (i + needed));                                           \
      i += needed;                                                            \
      strcat (modded_options, token);                                         \
      strcat (modded_options, " ");                                           \
    }                                                                         \
  while (0)

#define APPEND_TO_OPTION_BUILD_LOG(...)                                       \
  do                                                                          \
    {                                                                         \
      POCL_MSG_ERR (__VA_ARGS__);                                             \
      size_t l = strlen (program->main_build_log);                            \
      if (l < 640)                                                            \
        snprintf (program->main_build_log + l, (640 - l), __VA_ARGS__);       \
    }                                                                         \
  while (0)

static void
append_to_build_log (cl_program program, unsigned device_i, const char *format,
                     ...)
{
  char temp[4096];
  va_list args;
  va_start (args, format);
  int written = vsnprintf (temp, 4096, format, args);
  va_end (args);
  size_t l = 0;
  if (written > 0)
    {
      if (written > 4096)
        written = 4096;
      if (program->build_log[device_i])
        l = strlen (program->build_log[device_i]);
      size_t newl = l + (size_t)written;
      char *newp = (char *)realloc (program->build_log[device_i], newl + 1);
      assert (newp);
      memcpy (newp + l, temp, (size_t)written);
      newp[newl] = 0;
      program->build_log[device_i] = newp;
    }
}

#define APPEND_TO_BUILD_LOG_GOTO(err, ...)                                    \
  do                                                                          \
    {                                                                         \
      append_to_build_log (program, device_i, __VA_ARGS__);                   \
      if (err == CL_COMPILE_PROGRAM_FAILURE)                                  \
        POCL_MSG_ERR2 ("CL_COMPILE_PROGRAM_FAILURE", __VA_ARGS__);            \
      if (err == CL_BUILD_PROGRAM_FAILURE)                                    \
        POCL_MSG_ERR2 ("CL_BUILD_PROGRAM_FAILURE", __VA_ARGS__);              \
      else                                                                    \
        POCL_MSG_ERR2 (#err, __VA_ARGS__);                                    \
      errcode = err;                                                          \
      goto ERROR;                                                             \
    }                                                                         \
  while (0)

/* options must be non-NULL.
 * modded_options[size] + link_options are preallocated outputs
 */
static cl_int
process_options (const char *options, char *modded_options, char *link_options,
                 cl_program program, int compiling, int linking,
                 int *create_library, unsigned *flush_denorms,
                 int *requires_correctly_rounded_sqrt_div, int *spir_build,
                 cl_version *cl_c_version, size_t size)
{
  cl_int error;
  char *token = NULL;
  char *saveptr = NULL;

  *create_library = 0;
  *flush_denorms = 0;
  *cl_c_version = 0;
  *requires_correctly_rounded_sqrt_div = 0;
  *spir_build = 0;
  int enable_link_options = 0;
  link_options[0] = 0;
  modded_options[0] = 0;
  int ret_error = (linking ? (compiling ? CL_INVALID_BUILD_OPTIONS
                                        : CL_INVALID_LINKER_OPTIONS)
                           : CL_INVALID_COMPILER_OPTIONS);

  assert (options);
  assert (modded_options);
  assert (compiling || linking);

  char replace_me = 0;

  size_t i = 1; /* terminating char */
  size_t needed = 0;
  char *temp_options = (char *)malloc (strlen (options) + 1);

  memset (temp_options, 0, strlen (options) + 1);
  strncpy (temp_options, options, strlen (options));

  if (pocl_escape_quoted_whitespace (temp_options, &replace_me) == -1)
    {
      error = CL_INVALID_BUILD_OPTIONS;
      goto ERROR;
    }

  token = strtok_r (temp_options, " ", &saveptr);
  while (token != NULL)
    {
      /* check if parameter is supported compiler parameter */
      if (strncmp (token, "-cl", 3) == 0 || strncmp (token, "-w", 2) == 0
          || strncmp (token, "-Werror", 7) == 0)
        {
          if (strstr (cl_program_link_options, token))
            {
              /* when linking, only a subset of -cl* options are valid,
               * and only with -enable-link-options */
              if (linking && (!compiling))
                {
                  if (!enable_link_options)
                    {
                      APPEND_TO_OPTION_BUILD_LOG (
                          "Not compiling but link options were not enabled, "
                          "therefore %s is an invalid option\n",
                          token);
                      error = ret_error;
                      goto ERROR;
                    }
                  strcat (link_options, token);
                }
              if (strstr (token, "-cl-denorms-are-zero"))
                {
                  *flush_denorms = 1;
                }
              if (strstr (token, "-cl-fp32-correctly-rounded-divide-sqrt"))
                {
                  *requires_correctly_rounded_sqrt_div = 1;
                }
            }

          if (strstr (cl_parameters, token))
            {
              /* the LLVM API call pushes the parameters directly to the
                 frontend without using -Xclang */

              // LLVM 11 has removed "-cl-denorms-are-zero" option
              // https://reviews.llvm.org/D69878
              if (strncmp (token, "-cl-denorms-are-zero", 20) == 0)
                {
                  token = "-fdenormal-fp-math=positive-zero";
                }

              if (strncmp (token, "-cl-std=CL", 10) == 0)
                {
                  unsigned major = token[10] - '0';
                  unsigned minor = token[12] - '0';
                  *cl_c_version = CL_MAKE_VERSION (major, minor, 0);
                }
            }
          else if (strstr (cl_parameters_not_yet_supported_by_clang, token))
            {
              APPEND_TO_OPTION_BUILD_LOG (
                  "This build option is not yet supported by clang: %s\n",
                  token);
              token = strtok_r (NULL, " ", &saveptr);
              continue;
            }
          else
            {
              APPEND_TO_OPTION_BUILD_LOG ("Invalid build option: %s\n", token);
              error = ret_error;
              goto ERROR;
            }
        }
      else if (strncmp (token, "-g", 2) == 0)
        {
          APPEND_TOKEN ();
        }
      else if (strncmp (token, "-D", 2) == 0 || strncmp (token, "-I", 2) == 0)
        {
          APPEND_TOKEN ();
          /* if there is a space in between, then next token is part
             of the option */
          if (strlen (token) == 2)
            token = strtok_r (NULL, " ", &saveptr);
          else
            {
              token = strtok_r (NULL, " ", &saveptr);
              continue;
            }
        }
      else if (strncmp (token, "-x", 2) == 0 && strlen (token) == 2)
        {
          /* only "-x spir" is valid for the "-x" option */
          token = strtok_r (NULL, " ", &saveptr);
          if (!token || strncmp (token, "spir", 4) != 0)
            {
              APPEND_TO_OPTION_BUILD_LOG (
                  "Invalid parameter to -x build option\n");
              error = ret_error;
              goto ERROR;
            }
          /* "-x spir" is not valid if we are building from source */
          else if (program->source)
            {
              APPEND_TO_OPTION_BUILD_LOG (
                  "\"-x spir\" is not valid when building from source\n");
              error = ret_error;
              goto ERROR;
            }
          else
            *spir_build = 1;
          token = strtok_r (NULL, " ", &saveptr);
          continue;
        }
      else if (strncmp (token, "-spir-std=1.2", 13) == 0)
        {
          /* "-spir-std=" flags are not valid when building from source */
          if (program->source)
            {
              APPEND_TO_OPTION_BUILD_LOG ("\"-spir-std=\" flag is not valid "
                                          "when building from source\n");
              error = ret_error;
              goto ERROR;
            }
          else
            *spir_build = 1;
          token = strtok_r (NULL, " ", &saveptr);
          continue;
        }
      else if (strncmp (token, "-create-library", 15) == 0)
        {
          if (!linking)
            {
              APPEND_TO_OPTION_BUILD_LOG (
                  "\"-create-library\" flag is only valid when linking\n");
              error = ret_error;
              goto ERROR;
            }
          *create_library = 1;
          token = strtok_r (NULL, " ", &saveptr);
          continue;
        }
      else if (strncmp (token, "-enable-link-options", 20) == 0)
        {
          if (!linking)
            {
              APPEND_TO_OPTION_BUILD_LOG ("\"-enable-link-options\" flag is "
                                          "only valid when linking\n");
              error = ret_error;
              goto ERROR;
            }
          if (!(*create_library))
            {
              APPEND_TO_OPTION_BUILD_LOG ("\"-enable-link-options\" flag is "
                                          "only valid when -create-library "
                                          "option was given\n");
              error = ret_error;
              goto ERROR;
            }
          enable_link_options = 1;
          token = strtok_r (NULL, " ", &saveptr);
          continue;
        }
      else
        {
          APPEND_TO_OPTION_BUILD_LOG ("Invalid build option: %s\n", token);
          error = ret_error;
          goto ERROR;
        }
      APPEND_TOKEN ();
      token = strtok_r (NULL, " ", &saveptr);
    }

  error = CL_SUCCESS;

  /* remove trailing whitespace */
  i = strlen (modded_options);
  if ((i > 0) && (modded_options[i - 1] == ' '))
    modded_options[i - 1] = 0;

  /* put back replaced whitespaces if needed */
  if (replace_me != 0)
    {
      for (size_t x = 0; x < i; x++)
        {
          if (modded_options[x] == replace_me)
            modded_options[x] = ' ';
        }
    }

ERROR:
  POCL_MEM_FREE (temp_options);
  return error;
}

/* Unique hash for a device + program build + kernel name combination.
   NOTE: this does NOT take into account the local WG sizes or other
   specialization properties. */
static void
pocl_calculate_kernel_hash (cl_program program, unsigned kernel_i,
                            unsigned device_i)
{
  SHA1_CTX hash_ctx;
  pocl_SHA1_Init (&hash_ctx);

  char *n = program->kernel_meta[kernel_i].name;
  assert (n != NULL && program->build_hash[device_i] != NULL);
  pocl_SHA1_Update (&hash_ctx, (uint8_t *)program->build_hash[device_i],
                    sizeof (SHA1_digest_t));
  pocl_SHA1_Update (&hash_ctx, (uint8_t *)n, strlen (n));

  uint8_t digest[SHA1_DIGEST_SIZE];
  pocl_SHA1_Final (&hash_ctx, digest);

  memcpy (program->kernel_meta[kernel_i].build_hash[device_i], digest,
          sizeof (pocl_kernel_hash_t));
}

/* Canonical metadata is a read-only array of views into device-owned metadata. */
static void
free_meta (cl_program program)
{
  POCL_MEM_FREE (program->kernel_meta);
  program->num_kernels = 0;
}

static cl_int
clear_program_device (cl_program program, unsigned index)
{
  pocl_program_device_state *state = &program->device_states[index];
  cl_device_id device = program->devices[index];
  pocl_kernel_metadata_t *saved = program->kernel_meta;
  size_t saved_count = program->num_kernels;
  program->kernel_meta = state->kernel_meta;
  program->num_kernels = state->num_kernels;
  cl_int status = CL_SUCCESS;
  if (device->ops->free_program)
    {
      POCL_UNLOCK_OBJ (program);
      status = device->ops->free_program (device, program, index);
      POCL_LOCK_OBJ (program);
    }
  program->kernel_meta = saved;
  program->num_kernels = saved_count;
  if (status != CL_SUCCESS)
    return status;
  program->data[index] = NULL;
  program->gvar_storage[index] = NULL;
  program->llvm_irs[index] = NULL;
  pocl_free_program_device_metadata (program, index);
  if (program->source || program->program_il)
    {
      POCL_MEM_FREE (program->binaries[index]);
      program->binary_sizes[index] = 0;
      POCL_MEM_FREE (program->pocl_binaries[index]);
      program->pocl_binary_sizes[index] = 0;
    }
  program->global_var_total_size[index] = 0;
  memset (program->build_hash[index], 0, sizeof (SHA1_digest_t));
  return CL_SUCCESS;
}

static int
compatible_kernel_arguments (const pocl_kernel_metadata_t *a,
                              const pocl_kernel_metadata_t *b)
{
  if (a->num_args != b->num_args)
    return 0;
  for (unsigned i = 0; i < a->num_args; ++i)
    {
      const struct pocl_argument_info *x = &a->arg_info[i], *y = &b->arg_info[i];
      if (x->type != y->type || x->address_qualifier != y->address_qualifier ||
          x->access_qualifier != y->access_qualifier ||
          x->type_qualifier != y->type_qualifier || x->type_size != y->type_size)
        return 0;
      if (x->type_name && y->type_name && strcmp (x->type_name, y->type_name) != 0)
        return 0;
    }
  return 1;
}

/* Preserve existing successful slots before comparing newly built definitions. */
static cl_int
rebuild_public_metadata (cl_program program, const unsigned char *selected,
                         unsigned *conflict)
{
  pocl_kernel_metadata_t *view = NULL;
  size_t count = 0;
  for (unsigned pass = 0; pass < 2; ++pass)
    for (unsigned index = 0; index < program->num_devices; ++index)
      {
        if (!!selected[index] != pass || !pocl_program_device_executable (program, index))
          continue;
        const pocl_program_device_state *state = &program->device_states[index];
        for (size_t k = 0; k < state->num_kernels; ++k)
          {
            const pocl_kernel_metadata_t *candidate = &state->kernel_meta[k];
            size_t found = 0;
            while (found < count && strcmp (view[found].name, candidate->name) != 0)
              ++found;
            if (found < count)
              {
                if (!compatible_kernel_arguments (&view[found], candidate))
                  {
                    free (view);
                    *conflict = index;
                    return CL_INVALID_KERNEL_DEFINITION;
                  }
                continue;
              }
            if (count == SIZE_MAX / sizeof (*view))
              {
                free (view);
                return CL_OUT_OF_HOST_MEMORY;
              }
            pocl_kernel_metadata_t *grown = realloc (view, (count + 1) * sizeof (*view));
            if (!grown)
              {
                free (view);
                return CL_OUT_OF_HOST_MEMORY;
              }
            view = grown;
            view[count++] = *candidate;
          }
      }
  free_meta (program);
  program->kernel_meta = view;
  program->num_kernels = count;
  return CL_SUCCESS;
}

static int
setup_kernel_metadata (cl_program program, unsigned device_i)
{
  size_t i, j;
  assert (program->kernel_meta == NULL && program->num_kernels == 0);
  cl_device_id device = program->devices[device_i];
  int setup_successful = 0;
  if (program->pocl_binaries[device_i])
    {
      program->num_kernels = pocl_binary_get_kernel_count (program, device_i);
      if (program->num_kernels)
        {
          program->kernel_meta = calloc (program->num_kernels, sizeof (*program->kernel_meta));
          if (!program->kernel_meta)
            return CL_OUT_OF_HOST_MEMORY;
          pocl_binary_get_kernels_metadata (program, device_i);
        }
      setup_successful = 1;
    }
  else if (device->ops->setup_metadata)
    {
      POCL_UNLOCK_OBJ (program);
      setup_successful = device->ops->setup_metadata (device, program, device_i);
      POCL_LOCK_OBJ (program);
    }
  POCL_RETURN_ERROR_ON (
      (setup_successful == 0), CL_INVALID_BINARY,
      "Could not find kernel metadata in the built program\n");

  /* DBKs are named by the application. Overwrite the kernel names
     initialized by the metadata setup code with the requested ones.  */
  if (program->builtin_kernel_attributes)
    {
      for (size_t i = 0; i < program->num_kernels; ++i)
        {
          if (program->kernel_meta[i].name)
            free (program->kernel_meta[i].name);
          program->kernel_meta[i].name
              = strdup (program->builtin_kernel_names[i]);
        }
    }

  /* calculate argument storage size */
  for (i = 0; i < program->num_kernels; ++i)
    {
      pocl_kernel_metadata_t *kmeta = &program->kernel_meta[i];
      kmeta->total_argument_storage_size = 0;
      if (kmeta->num_args > 0)
        {
          size_t total = 0;
          for (j = 0; j < kmeta->num_args; ++j)
            {
              /* if one of the arguments have size 0,
                 the driver couldn't figure it out. In that case,
                 leave total_argument_storage_size == zero, and use
                 the old way of setting arguments. */
              if (kmeta->arg_info[j].type_size == 0)
                break;
              unsigned type_size = kmeta->arg_info[j].type_size;
              size_t alignment = pocl_size_ceil2 (type_size);
              if (total & (alignment - 1))
                total = (total | (alignment - 1)) + 1;
              total += kmeta->arg_info[j].type_size;
            }
          if (j >= kmeta->num_args)
            kmeta->total_argument_storage_size = total;
        }
    }

  return CL_SUCCESS;
}

static void
setup_device_kernel_hashes (cl_program program, unsigned device_i)
{
  cl_uint i;

  if ((program->num_kernels == 0) || (program->num_devices == 0))
    return;

  assert (program->kernel_meta);
  for (i = 0; i < program->num_kernels; ++i)
    {
      assert (program->kernel_meta[i].build_hash == NULL);
      program->kernel_meta[i].build_hash = (pocl_kernel_hash_t *)calloc (
          program->num_devices, sizeof (pocl_kernel_hash_t));
    }

  for (i = 0; i < program->num_kernels; ++i)
    pocl_calculate_kernel_hash (program, i, device_i);
}

static int
check_device_supports (cl_device_id device, cl_version cl_c_version)
{
  if (device->num_opencl_c_with_version > 0)
    {
      for (size_t i = 0; i < device->num_opencl_c_with_version; ++i)
        {
          if (device->opencl_c_with_version[i].version == cl_c_version)
            return 0;
        }
      return -1;
    }
  else
    {
      return cl_c_version > device->opencl_c_version_as_cl;
    }
}

struct pocl_device_build_parameters {
  int compile_program, link_program, create_library, requires_cr_sqrt_div, spir_build;
  cl_version cl_c_version;
  cl_uint num_input_headers, num_input_programs;
  const cl_program *input_headers, *input_programs;
  const char **header_include_names;
};

static cl_int
build_program_device (cl_program program, unsigned device_i,
                       const struct pocl_device_build_parameters *parameters)
{
  int compile_program = parameters->compile_program;
  int link_program = parameters->link_program;
  int create_library = parameters->create_library;
  int requires_cr_sqrt_div = parameters->requires_cr_sqrt_div;
  int spir_build = parameters->spir_build;
  cl_version cl_c_version = parameters->cl_c_version;
  cl_uint num_input_headers = parameters->num_input_headers;
  cl_uint num_input_programs = parameters->num_input_programs;
  const cl_program *input_headers = parameters->input_headers;
  const cl_program *input_programs = parameters->input_programs;
  const char **header_include_names = parameters->header_include_names;
  int errcode = CL_SUCCESS, error = CL_SUCCESS;
  int build_error_code = link_program ? CL_BUILD_PROGRAM_FAILURE : CL_COMPILE_PROGRAM_FAILURE;
      cl_device_id device = program->devices[device_i];
      if ((program->source || program->program_il) && !device->compiler_available)
        return CL_COMPILER_NOT_AVAILABLE;
      if (num_input_programs && !device->linker_available)
        return CL_LINKER_NOT_AVAILABLE;

      if (!pocl_get_bool_option ("POCL_IGNORE_CL_STD", 0) && cl_c_version
          && check_device_supports (device, cl_c_version))
        {
          APPEND_TO_BUILD_LOG_GOTO (
              build_error_code,
              "Build option -cl-std specified OpenCL C version %u.%u,"
              "but device %s doesn't support that OpenCL C version.\n",
              CL_VERSION_MAJOR (cl_c_version), CL_VERSION_MINOR (cl_c_version),
              device->short_name);
        }

      if (requires_cr_sqrt_div
          && !(device->single_fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT))
        APPEND_TO_BUILD_LOG_GOTO (build_error_code,
                                  REQUIRES_CR_SQRT_DIV_ERR " %s\n",
                                  device->short_name);

      /* clCreateProgramWithDefinedBuiltinKernels */
      if (program->builtin_kernel_attributes)
        {
          if (device->ops->build_defined_builtin == NULL)
            APPEND_TO_BUILD_LOG_GOTO (
                build_error_code,
                "%s device's driver does not support building "
                "programs with defined builtin kernels (DBKs)\n",
                device->long_name);

          POCL_UNLOCK_OBJ (program);
          error = device->ops->build_defined_builtin (program, device_i);
          POCL_LOCK_OBJ (program);
          if (error != CL_SUCCESS)
            APPEND_TO_BUILD_LOG_GOTO (CL_BUILD_PROGRAM_FAILURE,
                                      "Device %s failed to build the "
                                      "program with DBKs\n",
                                      device->long_name);
        }
      /* clCreateProgramWithBuiltinKernels */
      else if (program->builtin_kernel_names)
        {
          if (device->ops->build_builtin == NULL)
            APPEND_TO_BUILD_LOG_GOTO (
                build_error_code,
                "%s device's driver does not "
                "support building programs with builtin kernels\n",
                device->long_name);

          POCL_UNLOCK_OBJ (program);
          error = device->ops->build_builtin (program, device_i);
          POCL_LOCK_OBJ (program);
          if (error != CL_SUCCESS)
            APPEND_TO_BUILD_LOG_GOTO (CL_BUILD_PROGRAM_FAILURE,
                                      "Device %s failed to build the "
                                      "program with builtin kernels\n",
                                      device->long_name);
        }
      /* only link the program/library */
      else if (!compile_program && link_program)
        {
          assert (num_input_programs > 0);

          if (device->ops->link_program == NULL)
            APPEND_TO_BUILD_LOG_GOTO (CL_LINK_PROGRAM_FAILURE,
                                      "%s device's driver does "
                                      "not support linking programs\n",
                                      device->long_name);

          POCL_UNLOCK_OBJ (program);
          error = device->ops->link_program (program, device_i,
                                             num_input_programs,
                                             input_programs, create_library);
          POCL_LOCK_OBJ (program);
          if (error != CL_SUCCESS)
            {
              if (device->ops->external_build_options)
                {
                  errcode = error;
                  goto ERROR;
                }
              APPEND_TO_BUILD_LOG_GOTO (CL_LINK_PROGRAM_FAILURE,
                                        "Device %s failed to link the program\n",
                                        device->long_name);
            }
        }
      /* compile and/or link from source */
      else if (program->source)
        {
          if (device->ops->build_source == NULL)
            APPEND_TO_BUILD_LOG_GOTO (
                build_error_code,
                "%s device's driver does not "
                "support building programs from source\n",
                device->long_name);

          POCL_UNLOCK_OBJ (program);
          error = device->ops->build_source (
              program, device_i, num_input_headers, input_headers,
              header_include_names, (create_library ? 0 : link_program));
          POCL_LOCK_OBJ (program);

          if (error != CL_SUCCESS)
            {
              if (device->ops->external_build_options)
                {
                  errcode = error;
                  goto ERROR;
                }
              if (program->build_log[device_i])
                POCL_MSG_ERR ("Build log for device %s:\n%s\n",
                              device->long_name, program->build_log[device_i]);
              APPEND_TO_BUILD_LOG_GOTO (build_error_code,
                                        "Device %s failed to build"
                                        " the program\n",
                                        device->long_name);
            }
        }
      /* compile and/or link from binary */
      else
        {
          if (device->ops->build_binary == NULL)
            APPEND_TO_BUILD_LOG_GOTO (build_error_code,
                                      "%s device's driver does not support "
                                      "building programs from binaries\n",
                                      device->long_name);

          if ((program->binary_sizes[device_i] == 0)
              && (program->pocl_binary_sizes[device_i] == 0)
              && (program->program_il_size == 0))
            APPEND_TO_BUILD_LOG_GOTO (CL_INVALID_BINARY,
                                      "No poclbinaries nor binaries "
                                      "for device %s - can't build "
                                      "the program\n",
                                      device->short_name);

          POCL_UNLOCK_OBJ (program);
          error = device->ops->build_binary (
              program, device_i, (create_library ? 0 : link_program),
              spir_build);
          POCL_LOCK_OBJ (program);

          if (error != CL_SUCCESS)
            {
              if (device->ops->external_build_options)
                {
                  errcode = error;
                  goto ERROR;
                }
              if (program->build_log[device_i])
                POCL_MSG_ERR ("Build log for device %s:\n%s\n",
                              device->long_name, program->build_log[device_i]);
              APPEND_TO_BUILD_LOG_GOTO (build_error_code,
                                        "Device %s failed to build"
                                        " the program\n",
                                        device->long_name);
            }
        }


  if (!program->builtin_kernel_names)
    pocl_cache_update_program_last_access (program, device_i);
  if (program->binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE || program->num_builtin_kernels)
    {
      errcode = setup_kernel_metadata (program, device_i);
      if (errcode != CL_SUCCESS)
        goto ERROR;
      setup_device_kernel_hashes (program, device_i);
    }
  if (link_program && device->ops->post_build_program)
    {
      POCL_UNLOCK_OBJ (program);
      errcode = device->ops->post_build_program (program, device_i);
      POCL_LOCK_OBJ (program);
    }
ERROR:
  return errcode;
}

static cl_int
compile_and_link_program_body (
    int compile_program, int link_program, cl_program program,
    cl_uint num_devices, const cl_device_id *device_list, const char *options,
    cl_uint num_input_headers, const cl_program *input_headers,
    const char **header_include_names, cl_uint num_input_programs,
    const cl_program *input_programs,
    void (CL_CALLBACK *pfn_notify) (cl_program program, void *user_data),
    void *user_data)
{
  cl_int errcode = CL_SUCCESS;
  unsigned char *selected = NULL;
  char *combined = NULL, *normalized = NULL, *original = NULL;
  int owns_build = 0, external_options = 1;
  struct pocl_device_build_parameters parameters = {
      compile_program, link_program, 0, 0, 0, 0,
      num_input_headers, num_input_programs, input_headers, input_programs,
      header_include_names};
  unsigned flush_denorms = 0;
  char link_options[512] = { 0 };
  POCL_GOTO_LABEL_COND (NOTIFY, !IS_CL_OBJECT_VALID (program), CL_INVALID_PROGRAM);
  POCL_GOTO_LABEL_COND (NOTIFY, (num_devices && !device_list) || (!num_devices && device_list), CL_INVALID_VALUE);
  POCL_GOTO_LABEL_COND (NOTIFY, !pfn_notify && user_data, CL_INVALID_VALUE);
  POCL_LOCK_OBJ (program);
  POCL_GOTO_LABEL_COND (FINISH, program->build_in_progress || program->kernel_creations || program->kernels, CL_INVALID_OPERATION);
  POCL_GOTO_LABEL_COND (FINISH, !program->source && !program->binaries && !program->builtin_kernel_names, CL_INVALID_PROGRAM);
  POCL_GOTO_LABEL_COND (FINISH, !program->source && !program->program_il && !link_program, CL_INVALID_OPERATION);
  selected = calloc (program->num_devices, 1);
  POCL_GOTO_LABEL_COND (FINISH, !selected, CL_OUT_OF_HOST_MEMORY);
  if (!num_devices)
    memset (selected, 1, program->num_devices);
  else
    for (unsigned requested = 0; requested < num_devices; ++requested)
      {
        POCL_GOTO_LABEL_COND (FINISH, !IS_CL_OBJECT_VALID (device_list[requested]), CL_INVALID_DEVICE);
        cl_device_id device = pocl_real_dev (device_list[requested]);
        unsigned index = 0;
        while (index < program->num_devices && program->devices[index] != device)
          ++index;
        POCL_GOTO_LABEL_COND (FINISH, index == program->num_devices, CL_INVALID_DEVICE);
        selected[index] = 1;
      }
  for (unsigned index = 0; index < program->num_devices; ++index)
    if (selected[index])
      {
        POCL_GOTO_LABEL_COND (FINISH, POCL_ATOMIC_LOAD_PTR (program->devices[index]->available) == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
        if (!program->devices[index]->ops->external_build_options)
          external_options = 0;
      }
  program->build_in_progress = CL_TRUE;
  owns_build = 1;
  POCL_RETAIN_OBJECT_UNLOCKED (program);
  program->main_build_log[0] = 0;
  const char *extra = external_options ? NULL : pocl_get_string_option ("POCL_EXTRA_BUILD_FLAGS", NULL);
  size_t original_size = options ? strlen (options) : 0;
  size_t extra_size = extra ? strlen (extra) : 0;
  POCL_GOTO_LABEL_COND (FINISH, original_size > SIZE_MAX - extra_size - 514, CL_OUT_OF_HOST_MEMORY);
  combined = malloc (original_size + extra_size + 2);
  normalized = calloc (original_size + extra_size + 514, 1);
  original = options ? strdup (options) : NULL;
  POCL_GOTO_LABEL_COND (FINISH, !combined || !normalized || (options && !original), CL_OUT_OF_HOST_MEMORY);
  combined[0] = 0;
  if (options)
    memcpy (combined, options, original_size + 1);
  if (extra)
    {
      if (original_size)
        strcat (combined, " ");
      strcat (combined, extra);
    }
  if (external_options)
    strcpy (normalized, combined);
  else
    {
      errcode = process_options (combined, normalized, link_options, program,
                                 compile_program, link_program, &parameters.create_library,
                                 &flush_denorms, &parameters.requires_cr_sqrt_div,
                                 &parameters.spir_build, &parameters.cl_c_version,
                                 original_size + extra_size + 514);
      if (errcode != CL_SUCCESS)
        goto FINISH;
    }
  parameters.spir_build |= program->program_il != NULL;
  POCL_MEM_FREE (program->original_options);
  POCL_MEM_FREE (program->compiler_options);
  program->original_options = original;
  program->compiler_options = normalized;
  original = normalized = NULL;
  cl_int first_error = CL_SUCCESS;
  for (unsigned index = 0; index < program->num_devices; ++index)
    {
      if (!selected[index])
        continue;
      pocl_program_device_state *state = &program->device_states[index];
      char *slot_options = program->original_options ? strdup (program->original_options) : NULL;
      char *slot_compiler_options = strdup (program->compiler_options ? program->compiler_options : "");
      if ((program->original_options && !slot_options) || !slot_compiler_options)
        {
          free (slot_options);
          free (slot_compiler_options);
          if (first_error == CL_SUCCESS)
            first_error = CL_OUT_OF_HOST_MEMORY;
          continue;
        }
      program->active_build_device = index;
      program->binary_type = state->binary_type;
      program->flush_denorms = state->flush_denorms;
      cl_int status = clear_program_device (program, index);
      POCL_MEM_FREE (state->options);
      POCL_MEM_FREE (state->compiler_options);
      state->options = slot_options;
      state->compiler_options = slot_compiler_options;
      POCL_MEM_FREE (program->build_log[index]);
      if (status != CL_SUCCESS)
        {
          state->status = CL_BUILD_ERROR;
          state->binary_type = CL_PROGRAM_BINARY_TYPE_NONE;
          append_to_build_log (program, index, "Previous device artifact cleanup failed.\n");
          if (first_error == CL_SUCCESS)
            first_error = status;
          continue;
        }
      state->status = CL_BUILD_IN_PROGRESS;
      program->binary_type = parameters.create_library ? CL_PROGRAM_BINARY_TYPE_LIBRARY :
          compile_program && !link_program ? CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT :
          program->num_builtin_kernels ? CL_PROGRAM_BINARY_TYPE_NONE : CL_PROGRAM_BINARY_TYPE_EXECUTABLE;
      program->flush_denorms = flush_denorms;
      pocl_kernel_metadata_t *canonical = program->kernel_meta;
      size_t canonical_count = program->num_kernels;
      program->kernel_meta = NULL;
      program->num_kernels = 0;
      status = build_program_device (program, index, &parameters);
      state->kernel_meta = program->kernel_meta;
      state->num_kernels = program->num_kernels;
      program->kernel_meta = canonical;
      program->num_kernels = canonical_count;
      state->binary_type = status == CL_SUCCESS ? program->binary_type : CL_PROGRAM_BINARY_TYPE_NONE;
      state->flush_denorms = program->flush_denorms;
      state->status = status == CL_SUCCESS ? CL_BUILD_SUCCESS : CL_BUILD_ERROR;
      if (status != CL_SUCCESS)
        {
          if (first_error == CL_SUCCESS)
            first_error = status;
          /* Failed native cleanup remains owned by this slot for release/retry. */
          clear_program_device (program, index);
        }
    }
  program->active_build_device = CL_UINT_MAX;
  for (;;)
    {
      unsigned conflict = CL_UINT_MAX;
      cl_int status = rebuild_public_metadata (program, selected, &conflict);
      if (status == CL_INVALID_KERNEL_DEFINITION && conflict != CL_UINT_MAX)
        {
          program->device_states[conflict].status = CL_BUILD_ERROR;
          program->device_states[conflict].binary_type = CL_PROGRAM_BINARY_TYPE_NONE;
          append_to_build_log (program, conflict, "Incompatible kernel definition for this device.\n");
          if (first_error == CL_SUCCESS)
            first_error = status;
          continue;
        }
      if (status != CL_SUCCESS && first_error == CL_SUCCESS)
        first_error = status;
      break;
    }
  program->build_status = CL_BUILD_NONE;
  program->binary_type = CL_PROGRAM_BINARY_TYPE_NONE;
  for (unsigned index = 0; index < program->num_devices; ++index)
    {
      const pocl_program_device_state *state = &program->device_states[index];
      if (state->status == CL_BUILD_SUCCESS)
        {
          program->build_status = CL_BUILD_SUCCESS;
          if (state->binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE || program->binary_type == CL_PROGRAM_BINARY_TYPE_NONE)
            program->binary_type = state->binary_type;
        }
      else if (state->status == CL_BUILD_ERROR && program->build_status == CL_BUILD_NONE)
        program->build_status = CL_BUILD_ERROR;
    }
  errcode = first_error;
FINISH:
  free (selected);
  free (combined);
  free (normalized);
  free (original);
  if (owns_build)
    {
      program->active_build_device = CL_UINT_MAX;
      program->build_in_progress = CL_FALSE;
    }
  POCL_UNLOCK_OBJ (program);
NOTIFY:
  if (pfn_notify)
    pfn_notify (program, user_data);
  if (owns_build)
    pocl_release_owned (POCL_RELEASE_PROGRAM, program);
  return errcode;
}

/* Keep the frontend compiler implementation shared; selected drivers may
 * admit the entire mutation before old executable state is touched. */
struct pocl_build_call {
  int compile_program;
  int link_program;
  cl_program program;
  cl_uint num_devices;
  const cl_device_id *device_list;
  const char *options;
  cl_uint num_input_headers;
  const cl_program *input_headers;
  const char **header_include_names;
  cl_uint num_input_programs;
  const cl_program *input_programs;
  void (CL_CALLBACK *pfn_notify) (cl_program, void *);
  void *user_data;
  cl_uint guard_index;
};

static cl_int
run_program_build (void *data)
{
  struct pocl_build_call *call = data;
  const cl_device_id *selected = call->num_devices ? call->device_list : call->program->associated_devices;
  cl_uint count = call->num_devices ? call->num_devices : call->program->associated_num_devices;
  while (call->guard_index < count)
    {
      cl_uint index = call->guard_index++;
      cl_device_id device = pocl_real_dev (selected[index]);
      int duplicate = 0;
      for (cl_uint prior = 0; prior < index; ++prior)
        if (pocl_real_dev (selected[prior]) == device)
          duplicate = 1;
      if (!duplicate && device->ops->guard_program_build)
        return device->ops->guard_program_build (device, call->program,
                                                 run_program_build, call);
    }
  return compile_and_link_program_body (
      call->compile_program, call->link_program, call->program,
      call->num_devices, call->device_list, call->options,
      call->num_input_headers, call->input_headers, call->header_include_names,
      call->num_input_programs, call->input_programs,
      call->pfn_notify, call->user_data);
}

cl_int
compile_and_link_program (
    int compile_program, int link_program, cl_program program,
    cl_uint num_devices, const cl_device_id *device_list, const char *options,
    cl_uint num_input_headers, const cl_program *input_headers,
    const char **header_include_names, cl_uint num_input_programs,
    const cl_program *input_programs,
    void (CL_CALLBACK *pfn_notify) (cl_program program, void *user_data),
    void *user_data)
{
  /* Let normal argument validation report malformed calls without admission. */
  if (!IS_CL_OBJECT_VALID (program) || (num_devices && !device_list) ||
      (!num_devices && device_list))
    return compile_and_link_program_body (
        compile_program, link_program, program, num_devices, device_list, options,
        num_input_headers, input_headers, header_include_names,
        num_input_programs, input_programs, pfn_notify, user_data);
  const cl_device_id *selected = num_devices ? device_list : program->associated_devices;
  cl_uint count = num_devices ? num_devices : program->associated_num_devices;
  for (cl_uint index = 0; index < count; ++index)
    if (!IS_CL_OBJECT_VALID (selected[index]))
      return CL_INVALID_DEVICE;
  POCL_RETAIN_OBJECT (program);
  struct pocl_build_call call = {
      compile_program, link_program, program, num_devices, device_list, options,
      num_input_headers, input_headers, header_include_names,
      num_input_programs, input_programs, pfn_notify, user_data, 0};
  cl_int status = run_program_build (&call);
  pocl_release_owned (POCL_RELEASE_PROGRAM, program);
  return status;
}
