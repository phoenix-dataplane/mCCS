back03 LEVEL='info':
  RUST_LOG={{LEVEL}} ./target/debug/mccs --host 0
back06 LEVEL='info':
  RUST_LOG={{LEVEL}} ./target/debug/mccs --host 1

front03 SIZE='128' COMM='42':
  ./target/debug/allgather_proto --root-addr 192.168.211.66 --rank 0 \
  --num-ranks 2 --cuda-device-idx 0 --size {{SIZE}} --communicator {{COMM}}

front06 SIZE='128' COMM='42':
  ./target/debug/allgather_proto --root-addr 192.168.211.66 --rank 1 \
  --num-ranks 2 --cuda-device-idx 0 --size {{SIZE}} --communicator {{COMM}}

alt-front03 SIZE='128' COMM='42':
  ./target/debug/allgather_proto --root-addr 192.168.211.66 --rank 1 \
  --num-ranks 2 --cuda-device-idx 1 --size {{SIZE}} --communicator {{COMM}}

triple03 SIZE='128' COMM='42':
  ./target/debug/allgather_proto --root-addr 192.168.211.66 --rank 0 \
  --num-ranks 3 --cuda-device-idx 0 --size {{SIZE}} --communicator {{COMM}}

alt-triple03 SIZE='128' COMM='42':
  ./target/debug/allgather_proto --root-addr 192.168.211.66 --rank 1 \
  --num-ranks 3 --cuda-device-idx 1 --size {{SIZE}} --communicator {{COMM}}

triple06 SIZE='128' COMM='42':
  ./target/debug/allgather_proto --root-addr 192.168.211.66 --rank 2 \
  --num-ranks 3 --cuda-device-idx 0 --size {{SIZE}} --communicator {{COMM}}