#!/usr/bin/env cwl-runner

class: CommandLineTool
id: snaptools_align_paired_end
label: snaptools align paired end reads
cwlVersion: v1.1

s:author:
  - class: s:Person
    s:identifier: https://orcid.org/0000-0001-5173-4627
    s:email: jshands@ucsc.edu
    s:name: Walter Shands

s:codeRepository: https://github.com/wshands/SnapTools/tree/feature/docker_cwl
s:dateCreated: "2019-11-15"
s:license: https://spdx.org/licenses/Apache-2.0

s:keywords: edam:topic_0091 , edam:topic_0622
s:programmingLanguage: Python

$namespaces:
  s: https://schema.org/
  edam: http://edamontology.org/
  dct: http://purl.org/dc/terms/
  foaf: http://xmlns.com/foaf/0.1/

$schemas:
  - https://schema.org/docs/schema_org_rdfa.html
  - http://edamontology.org/EDAM_1.18.owl


dct:creator:
  '@id':  https://orcid.org/0000-0001-5173-4627
  foaf:name: Walter Shands
  foaf:mbox: jshands@ucsc.edu

requirements:
  DockerRequirement:
    dockerPull: "hubmap/sc-atac-seq-grch38"
  ResourceRequirement:
    coresMin: 1
    ramMin: 1024
    outdirMin: 100000

arguments:
  - position: 1
    valueFrom: "--"

inputs:
  alignment_index:
    type: Directory?
    inputBinding:
      position: 2
      prefix: --alignment-index
    doc: The alignment index to use.

  input_fastq1:
    type: File
    inputBinding:
      position: 3
      prefix: --input-fastq1
    doc: The first paired end fastq file to be aligned.

  input_fastq2:
    type: File
    inputBinding:
      position: 4
      prefix: --input-fastq2
    doc: The second paired end fastq file to be aligned.

  output_bam:
    type: string
    inputBinding:
      position: 5
      prefix: --output-bam
    default: "snaptools_alignment.bam"
    doc: The name to use for the output bam file containing unfiltered alignments.

  aligner:
    type: string?
    inputBinding:
      position: 6
      prefix: --aligner
    default: "bwa"
    doc: The name of the aligner, e.g. 'bwa'.

  path_to_aligner:
    type: string?
    inputBinding:
      position: 7
      prefix: --path-to-aligner
    default: "/usr/local/bin"
    doc: The file system path to the aligner.

  aligner_options:
    type: string[]?
    inputBinding:
      prefix: --aligner-options
      position: 8
    doc: List of strings indicating options you would like passed to aligner.

  read_fastq_command:
    type: string?
    inputBinding:
      position: 9
      prefix: --read-fastq-command
    doc: Command line to execute for each of the input files.

  min_cov:
    type: string?
    inputBinding:
      position: 10
      prefix: --min-cov
    doc: Minimum number of fragments per barcode.

  num_threads:
    type: int?
    inputBinding:
      position: 11
      prefix: --num-threads
    default: 8
    doc: The number of threads to use.

  if_sort:
    type: string?
    inputBinding:
      position: 12
      prefix: --if-sort
    doc: Whether to sort the bam file based on the read name.

  tmp_folder:
    type: string?
    inputBinding:
      position: 13
      prefix: --tmp-folder
    default: "/tmp"
    doc: Directory to store temporary files.

  overwrite:
    type: string?
    inputBinding:
      position: 14
      prefix: --overwrite
    doc: Whether to overwrite the output file if it already exists.

  verbose:
    type: string?
    inputBinding:
      position: 15
      prefix: --verbose
    doc: A boolean tag; if true output the progress.

outputs:
  paired_end_bam:
    type: File
    outputBinding:
      glob: $(inputs.output_bam)

baseCommand: [/opt/snaptools_wrapper.py, align-paired-end]
