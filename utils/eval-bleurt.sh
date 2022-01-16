
python -m bleurt.score \
  -candidate_file=$2 \
  -reference_file=$1 \
  -bleurt_checkpoint=bleurt-tiny-512
