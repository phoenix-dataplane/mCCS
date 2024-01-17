build:
  ./_private/remote_build.sh

sync:
  ./_private/sync.sh

back host LEVEL='info':
  RUST_LOG={{LEVEL}} ./target/debug/mccs --host {{host}}

root_addr := '192.168.211.130'

base rank num_ranks size='128' comm='42' cuda_dev='0':
  ./target/debug/allgather_bench --root-addr {{root_addr}} --rank {{rank}} \
  --num-ranks {{num_ranks}} --cuda-device-idx {{cuda_dev}} --size {{size}} --communicator {{comm}}

double0 size='128' comm='42':
  just base 0 2 {{size}} {{comm}}

shm-double1 size='128' comm='42':
  just base 1 2 {{size}} {{comm}} 1

double1 size='128' comm='42':
  just base 1 2 {{size}} {{comm}}

triple0 size='128' comm='42':
  just base 0 3 {{size}} {{comm}}

triple1 size='128' comm='42':
  just base 1 3 {{size}} {{comm}} 1

triple2 size='128' comm='42':
  just base 2 3 {{size}} {{comm}}

auto-triple size='128' comm='42':
  just base $RK 3 {{size}} {{comm}} $DEV

auto-back LEVEL='info':
  just back $MACHINE {{LEVEL}}