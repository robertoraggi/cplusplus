// RUN: %cxx -verify -fcheck %s

#define tkgt 1
#define tkappend 2
#define tkpipe 3

static char outmodes[] = {tkgt, tkappend, tkpipe, 0};
