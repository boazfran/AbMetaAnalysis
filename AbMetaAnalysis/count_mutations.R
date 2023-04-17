#!/usr/bin/env Rscript
library(shazam)
library(parallel)

args = commandArgs(trailingOnly=TRUE)

if (length(args)<=1) {
  stop("At least two arguments must be supplied (clone_col input file [output_file]).n", call.=FALSE)
}
clone_col = args[1]
if (length(args)==2) {
  # default output file
  input_file = args[2]
  output_file = args[2]
} else if (length(args) == 3) {
  input_file = args[2]
  output_file = args[3]
} else {
  stop("At no more than three arguments can be supplied (clone_col input file [output_file]).n", call.=FALSE)
}

dfv1 = read.table(input_file, sep='\t', header=TRUE)
dfv1[!is.na(dfv1[clone_col]), 'vseq'] = apply(dfv1[!is.na(dfv1[clone_col]),], 1, function(x) substr(x['sequence_alignment'], x['v_germline_start'], x['v_germline_end']))
dfv1[!is.na(dfv1[clone_col]), 'vgerseq'] = apply(dfv1[!is.na(dfv1[clone_col]),], 1, function(x) substr(x['germline_alignment'], x['v_germline_start'], x['v_germline_end']))
dfv1m <- observedMutations(dfv1[!is.na(dfv1[clone_col]),], sequenceColumn="vseq",germlineColumn="vgerseq", nproc=round(detectCores()*0.25), frequency=TRUE, combine=TRUE)
dfv1[!is.na(dfv1[clone_col]), 'mu_freq'] = dfv1m$mu_freq
dfv1 = subset(dfv1, select=-c(vseq, vgerseq))
if (endsWith(output_file, '.gz')) {
  gz_file = gzfile(output_file, "w")
  write.table(dfv1, gz_file, sep='\t')
  close(gz_file)
} else {
  write.table(dfv1, output_file, sep='\t')
}
