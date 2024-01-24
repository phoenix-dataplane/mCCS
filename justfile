build:
  ./_private/remote_build.sh

sync:
  ./_private/sync.sh

back host LEVEL='info':
  RUST_LOG={{LEVEL}} ./target/debug/mccs --host {{host}}

root_addr := '192.168.211.66'

bench rank num_ranks round='10' size='128' comm='42' cuda_dev='0':
  ./target/debug/allgather_bench --root-addr {{root_addr}} --rank {{rank}} \
  --num-ranks {{num_ranks}} --cuda-device-idx {{cuda_dev}} --size {{size}} --communicator {{comm}} --round {{round}}

allreduce_bench rank num_ranks round='10' size='128' comm='42' cuda_dev='0':
  ./target/debug/allreduce_bench --root-addr {{root_addr}} --rank {{rank}} \
  --num-ranks {{num_ranks}} --cuda-device-idx {{cuda_dev}} --size {{size}} --communicator {{comm}} --round {{round}}


allreduce_base rank num_ranks size='128' comm='42' cuda_dev='0':
  ./target/debug/allreduce_proto --root-addr {{root_addr}} --rank {{rank}} \
  --num-ranks {{num_ranks}} --cuda-device-idx {{cuda_dev}} --size {{size}} --communicator {{comm}}

auto-3reduce size='128' round='10' comm='42':
  just allreduce_bench $RK 3 {{round}} {{size}} {{comm}} $DEV

auto-triple size='128' round='10' comm='42':
  just bench $RK 3 {{round}} {{size}} {{comm}} $DEV

auto-back LEVEL='info':
  just back $MACHINE {{LEVEL}}

kill host:
  ssh danyang-0{{host}} -t "pkill mccs"

killall:
  just kill 3
  just kill 5