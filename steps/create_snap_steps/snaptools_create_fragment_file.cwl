#!/usr/bin/env cwl-runner

class: CommandLineTool
label: snaptools create fragment file
cwlVersion: v1.1

baseCommand: sinto

inputs:
  input_bam:
    type: File
    inputBinding:
      position: 1
      prefix: -b
    doc: The bam file to use

  output_bed:
    type: int
    inputBinding:
      position: 2
      prefix: -f
    default: "fragment_file.bed"
    doc: The name to use for the output fragment file

  num_proc:
    type: string?
    inputBinding:
      position: 3
      prefix: -p
    default: 1
    doc: The number of processors to use

outputs:
  fragment_file:
    type: File
    outputBinding:
      glob: $inputs.output_bed
