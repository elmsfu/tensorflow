TFLITE_ROOT = $(CURDIR)/../..
TF_ROOT = $(CURDIR)/../../../..
FLATC = flatc-build/flatc
SCHEMA = $(TFLITE_ROOT)/schema/schema.fbs
LIBTFLITE = $(TFLITE_ROOT)/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a

# MAKEFILE_DIR=$(TFLITE_ROOT)/experimental/micro/tools/make
# include $(MAKEFILE_DIR)/third_party_downloads.inc
# THIRD_PARTY_DOWNLOADS :=
# $(eval $(call add_third_party_download,$(GEMMLOWP_URL),$(GEMMLOWP_MD5),gemmlowp,))
# $(eval $(call add_third_party_download,$(FLATBUFFERS_URL),$(FLATBUFFERS_MD5),flatbuffers,))


TFLITE_DOWNLOADS = $(TFLITE_ROOT)/tools/make/downloads
TFLITE_DEP_DOWNLOAD = $(TFLITE_ROOT)/tools/make/download_dependencies.sh

EXEC_TEST_GRAPH = exec_test_graph

CXXFLAGS = -g --std=c++11 -fPIC -pthread
CPPFLAGS = -I $(TF_ROOT) -I$(TFLITE_ROOT)/tools/make/downloads/flatbuffers/include

TEST_JSONS = $(wildcard tests/test*.json)

TEST_TFS   = $(patsubst %.json,%.tflite,$(TEST_JSONS))
TEST_ASM   = $(patsubst %.json,%.asm,$(TEST_JSONS))
TEST_BIN   = $(patsubst %.json,%.bin,$(TEST_JSONS))
TEST_SIM_OUT  = $(patsubst %.json,%_out.bin,$(TEST_JSONS))
TEST_TF_OUT   = $(patsubst %.json,%_tflite.bin,$(TEST_JSONS))

TEST_OUTPUTS = $(TEST_TFS) $(TEST_ASM) $(TEST_BIN) $(TEST_SIM_OUT) $(TEST_TF_OUT)

.PHONY: all clean
.PHONY: clean_tflite
.PHONY: tests clean_tests

all: tests

clean: clean_tflite clean_tests

# THIRD_PARTY_TARGETS := $(foreach DOWNLOAD,$(THIRD_PARTY_DOWNLOADS),$(eval $(call create_download_rule,$(DOWNLOAD))))
# third_party_downloads: $(THIRD_PARTY_TARGETS)

$(FLATC):
	mkdir -p $(dir $@)
	cd $(dir $@) && cmake $(TFLITE_DOWNLOADS)/flatbuffers && $(MAKE) flatc

tests: $(TEST_OUTPUTS)
	python utils/data_check.py -o 0x10000 -c 1 -d 1,2,4,2
	python utils/data_check.py -o 0x10000 -c 2 -d 1,1,2,1

	python utils/data_check.py -o 0x10000 -c 10 -d 1,4,9,8
	python utils/data_check.py -o 0x10000 -c 11 -d 1,4,9,8
	python utils/data_check.py -o 0x10000 -c 12 -d 1,3,9,10


	python utils/data_check.py -o 0x10000 -c 20 -d 1,4
	python utils/data_check.py -o 0x10000 -c 30 -d 1,3,9,5
	python utils/data_check.py -o 0x10000 -c 40 -d 1,7,11,2

	python utils/data_check.py -o 0x10000 -c 101 -d 1,4

clean_tests:
	$(RM) $(TEST_OUTPUTS)

tflite: $(FLATC)
	./$(FLATC) -p $(SCHEMA)

clean_tflite:
	$(RM) -r tflite
	$(MAKE) -C $(TF_ROOT) -f tensorflow/contrib/lite/tools/make/Makefile clean

$(TFLITE_DOWNLOADS): $(TFLITE_DEP_DOWNLOAD)
	$(TFLITE_DEP_DOWNLOAD)

src/exec_test_graph.o: $(TFLITE_DOWNLOADS)

$(EXEC_TEST_GRAPH): src/exec_test_graph.o $(LIBTFLITE)
	g++ --std=c++11 -o $@ $^ -lm -pthread -ldl

$(LIBTFLITE): $(TFLITE_DOWNLOADS)
	$(MAKE) -C $(TF_ROOT) -f tensorflow/lite/tools/make/Makefile

.PRECIOUS: %.tflite %.bin %.asm

%.tflite: $(SCHEMA) %.json $(FLATC)
	./$(FLATC) -b -o $(dir $@) $(filter-out $(FLATC),$^)

%_tflite.bin: %.tflite $(EXEC_TEST_GRAPH)
	./$(EXEC_TEST_GRAPH) $< -o $@


MLASM_DIR = third_party/mlaccel/asm
MLASM = $(MLASM_DIR)/mlasm
$(MLASM): $(foreach ii,mlasm.h mlasm.cc main.cc,$(MLASM_DIR)/$(ii))
	$(MAKE) -C $(MLASM_DIR) mlasm

MLSIM_DIR = third_party/mlaccel/sim
MLSIM = $(MLSIM_DIR)/mlsim
$(MLSIM): $(foreach ii,mlsim.h mlsim.cc main.cc,$(MLSIM_DIR)/$(ii))
	$(MAKE) -C $(MLSIM_DIR) mlsim

TFLITE2MLASM = tflite2mlasm.py

%.asm: %.tflite $(TFLITE2MLASM)
	./$(TFLITE2MLASM) -i $(filter-out $(TFLITE2MLASM),$^) -o $@

%.bin: %.asm $(MLASM)
	$(MLASM) -b $@ $(filter-out $(MLASM),$^)

%_out.bin: %.bin $(MLSIM)
	$(MLSIM) -b $@ $(filter-out $(MLSIM),$^)
