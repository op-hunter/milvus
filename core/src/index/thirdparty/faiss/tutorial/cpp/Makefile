# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

-include ../../makefile.inc

CPU_TARGETS = ivfflat-profile

default: cpu

all: cpu 

cpu: $(CPU_TARGETS)

%: %.cpp ../../libfaiss.a
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I../../ $(LIBS)

clean:
	rm -f $(CPU_TARGETS) $(GPU_TARGETS)

.PHONY: all cpu default clean
