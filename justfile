backend-03 LEVEL='info':
  RUST_LOG={{LEVEL}} ./target/debug/mccs --host 0
backend-06 LEVEL='info':
  RUST_LOG={{LEVEL}} ./target/debug/mccs --host 1

frontend-03 SIZE='128' COMM='42':
  ./target/debug/allgather_proto --root-addr 192.168.211.66 --rank 0 \
  --num-ranks 2 --cuda-device-idx 0 --size {{SIZE}} --communicator {{COMM}}

frontend-06 SIZE='128' COMM='42':
  ./target/debug/allgather_proto --root-addr 192.168.211.66 --rank 1 \
  --num-ranks 2 --cuda-device-idx 0 --size {{SIZE}} --communicator {{COMM}}
