/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <string.h>
#include <unistd.h>

#include <cstdio>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

using namespace tflite;

void print_tensor(TfLiteTensor* tensor) {
  switch(tensor->type) {
    case kTfLiteFloat32:
    {
      float* data = tensor->data.f;
      printf("%p\n", data);
      if (data) {
        for(int ii=0; ii<tensor->bytes/sizeof(float); ++ii) {
          printf("[%3d]\t 0x%08x \t %f\n", ii,  data[ii], data[ii]);
        }
      }
    }
    break;
    case kTfLiteInt32:
    {
      int32_t* data = tensor->data.i32;
      printf("%p\n", data);
      if (data) {
        for(int ii=0; ii<tensor->bytes/sizeof(int32_t); ++ii) {
          printf("[%3d]\t 0x%08x \t %d\n", ii,  data[ii], data[ii]);
        }
      }
    }
    break;

    case kTfLiteUInt8:
    {
      uint8_t* data = tensor->data.uint8;
      printf("%p\n", data);
      if (data) {
        for(int ii=0; ii<tensor->bytes/sizeof(uint8_t); ++ii) {
          printf("[%3d]\t 0x%08x \t %d\n", ii,  data[ii], data[ii]);
        }
      }
    }
    break;
    case kTfLiteInt16:
    {
      int16_t* data = tensor->data.i16;
      printf("%p\n", data);
      if (data) {
        for(int ii=0; ii<tensor->bytes/sizeof(int16_t); ++ii) {
          printf("[%3d]\t 0x%08x \t %d\n", ii,  data[ii], data[ii]);
        }
      }
    }
    break;

    case kTfLiteString:
    case kTfLiteInt64:
    case kTfLiteBool:
    case kTfLiteComplex64:
    case kTfLiteNoType:
    default:
      printf("Not supported type %d\n", tensor->type);
    break;
  }
}

char* dot_tensor_name(TfLiteTensor* tensor) {
  char* name = NULL;
  if (NULL != tensor->name) {
    name = strdup(tensor->name);
  } else {
    name = strdup("def");
  }

  for (char* pch=name; *pch; pch++) {
    if (*pch == '/')
      *pch = '_';
  }

  return name;
}
#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

void usage(const char* arg0) {
  fprintf(stderr, "%s: -[vh] -[i binary_input] -[o binary_output] <graph.tflite>\n", basename(arg0));

  exit(1);
}

int main(int argc, char* argv[]) {
  const char* filename = NULL;
  const char* fname_bout = NULL;
  const char* fname_bin = NULL;
  size_t verbosity = 0;

  int ch = -1;
  while (-1 != (ch = getopt(argc, argv, "hvo:i:"))) {
    switch(ch) {
      case 'v':
        verbosity++;
        break;
      case 'h':
        usage(argv[0]);
        break;
      case 'o':
        fname_bout = optarg;
        break;
      case 'i':
        fname_bin = optarg;
        break;
      default:
        fprintf(stderr, "Unknown option %c\n", ch);
        usage(argv[0]);
        break;
    }
  }

  if (argc < optind) {
    usage(argv[0]);
  }
  filename = argv[optind];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model.get(), resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors
  if (fname_bin) {
    FILE* fbin = fopen(fname_bin, "rb");
    if(NULL == fbin) {
      fprintf(stderr, "Unable to open binary input file '%s'\n", fname_bin);
      return EXIT_FAILURE;
    }

    for (auto ii : interpreter->inputs()) {
      auto tensor = interpreter->tensor(ii);
      void* tmp = tensor->data.raw;
      //uint8_t tmp[1500];

      size_t bw = fread(tmp, 1, tensor->bytes, fbin);
      if (bw != tensor->bytes) {
        fprintf(stderr, "Failed to %d bytes from %s for %s\n", tensor->bytes, fname_bin, tensor->name);
      }
    }
    fclose(fbin);
  }

  for (auto ii : interpreter->inputs()) {
    auto tensor = interpreter->tensor(ii);
    print_tensor(tensor);
  }

  char* fname_dot = "graph.dot";
  if (fname_dot) {
    FILE* fdot = fopen(fname_dot, "w");

    fprintf(fdot,"digraph tflite {\n");

    //inputs
    auto inputs = interpreter->inputs();
    auto outputs = interpreter->outputs();

    for (int ii=0; ii<interpreter->tensors_size(); ++ii) {
      char* attr="";
      if (std::count(inputs.begin(), inputs.end(), ii) > 0) {
        attr = "[color=blue]";
      }
      if (std::count(outputs.begin(), outputs.end(), ii) > 0) {
        attr = "[color=red]";
      }
      auto tensor = interpreter->tensor(ii);
      if (tensor) {
        char* name = dot_tensor_name(tensor);
        fprintf(fdot, "%s%d %s;\n", name, ii, attr);
        free(name);
      } else {
        fprintf(stderr, "No tensor %d\n", ii);
      }
    }

    //ops
    for (int ii=0; ii< interpreter->nodes_size(); ++ii) {
      auto node_and_reg = interpreter->node_and_registration(ii);
      if (node_and_reg) {
        auto node = node_and_reg->first;
        fprintf(fdot, "op%d [shape=box];\n", ii);

        for (int jj=0; jj < node.inputs->size; ++jj) {
          int ti = node.inputs->data[jj];
          auto tensor = interpreter->tensor(ti);
          char* name = dot_tensor_name(tensor);
          fprintf(fdot, "%s%d -> op%d;\n", name, ti, ii);
          free(name);
        }
        for (int jj=0; jj < node.outputs->size; ++jj) {
          int ti = node.outputs->data[jj];
          auto tensor = interpreter->tensor(ti);
          char* name = dot_tensor_name(tensor);
          fprintf(fdot, "op%d -> %s%d;\n", ii, name, ti);
          free(name);
        }
      } else {
        fprintf(stderr, "No node %d\n", ii);
      }
    }
    fprintf(fdot, "}\n");
    fclose(fdot);
  }
  printf("\n\n=== Invoking Interpreter State ===\n");

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  for(auto ii : interpreter->outputs()) {
    auto tensor = interpreter->tensor(ii);
    print_tensor(tensor);
  }

  if (fname_bout) {
    size_t noutputs = interpreter->outputs().size();
    if (noutputs > 1) {
      fprintf(stderr, "more than 1 output tensor, not outputing binary\n");
      exit(2);
    } else if (noutputs == 0) {
      fprintf(stderr, "no output tensors! Not outputing binary\n");
      exit(3);
    }

    TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
    FILE* fbout = fopen(fname_bout, "wb");
    if(NULL == fbout) {
      fprintf(stderr, "Unable to open binary output file '%s'\n", fname_bout);
      return EXIT_FAILURE;
    }
    size_t bw = fwrite(tensor->data.raw, 1, tensor->bytes, fbout);
    if (bw != tensor->bytes) {
      fprintf(stderr, "Failed to write %d bytes to %s\n", tensor->bytes, fname_bout);
    }
    fclose(fbout);
  }

  return 0;
}
