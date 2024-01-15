backend-03:
  ./target/debug/mccs --host 0
backend-06:
  ./target/debug/mccs --host 1

frontend-03:
  ./target/debug/allgather_proto --root-addr 192.168.211.66 --rank 0 \
  --num-ranks 2 --cuda-device-idx 0

frontend-06:
  ./target/debug/allgather_proto --root-addr 192.168.211.66 --rank 1 \
  --num-ranks 2 --cuda-device-idx 0
