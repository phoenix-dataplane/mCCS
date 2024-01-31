build:
  ./_private/remote_build.sh

sync:
  ./_private/sync.sh

[private]
back host LEVEL='info':
  RUST_LOG={{LEVEL}} ./target/debug/mccs --host {{host}}

root_addr := '192.168.211.34'

[private]
bench rank num_ranks round='10' size='128' comm='42' cuda_dev='0':
  ./target/debug/allgather_bench --root-addr {{root_addr}} --rank {{rank}} \
  --num-ranks {{num_ranks}} --cuda-device-idx {{cuda_dev}} --size {{size}} --communicator {{comm}} --round {{round}}

[private]
allreduce_bench rank num_ranks round='10' size='128' comm='42' cuda_dev='0':
  ./target/debug/allreduce_bench --root-addr {{root_addr}} --rank {{rank}} \
  --num-ranks {{num_ranks}} --cuda-device-idx {{cuda_dev}} --size {{size}} --communicator {{comm}} --round {{round}}


[private]
allreduce_base rank num_ranks size='128' comm='42' cuda_dev='0':
  ./target/debug/allreduce_proto --root-addr {{root_addr}} --rank {{rank}} \
  --num-ranks {{num_ranks}} --cuda-device-idx {{cuda_dev}} --size {{size}} --communicator {{comm}}

[private]
auto-3reduce size='128' round='10' comm='42':
  just allreduce_bench $RK 3 {{round}} {{size}} {{comm}} $DEV

[private]
auto-triple size='128' round='10' comm='42':
  just bench $RK 3 {{round}} {{size}} {{comm}} $DEV

[private]
auto-double size='128' round='10' comm='42':
  just bench $RK 2 {{round}} {{size}} {{comm}} $DEV

auto-back LEVEL='info':
  just back $MACHINE {{LEVEL}}

kill host:
  ssh danyang-0{{host}} -t "pkill mccs"

killall:
  just kill 1; just kill 2; just kill 3; just kill 5

[private]
launch group folder:
  cargo run --bin launcher -- --configfile launcher/config.toml --benchmark eval/single-app/output/ --group {{group}} --silent --output-dir /tmp/single_v3/{{folder}}

[private]
launch-multi group folder:
  cargo run --bin launcher -- --configfile launcher/config.toml --benchmark eval/multi-app/output/ --group {{group}} --silent --output-dir /tmp/single_v3/{{folder}}

[private]
one_4gpu_ecmp cnt="0":
  just launch 4GPU_ECMP single-app-ecmp{{cnt}}

[private]
one_8gpu_ecmp cnt="0":
  just launch 8GPU_ECMP single-app-ecmp{{cnt}}

[private]
one_4gpu_flow cnt:
  just launch 4GPU_FLOW single-app-flow{{cnt}}

[private]
one_8gpu_flow cnt:
  just launch 8GPU_FLOW single-app-flow{{cnt}}

four_gpu_ecmp:
  #!/usr/bin/env bash
  ./eval/set_ecmp_hashing_algo.sh everything
  for i in {0..19}; do
    just one_4gpu_ecmp $i
  done

eight_gpu_ecmp:
  #!/usr/bin/env bash
  ./eval/set_ecmp_hashing_algo.sh everything
  for i in {0..19}; do
    just one_8gpu_ecmp $i
  done

four_gpu_flow:
  #!/usr/bin/env bash
  ./eval/set_ecmp_hashing_algo.sh source-port
  for i in {0..19}; do
    just one_4gpu_flow $i
  done

eight_gpu_flow:
  #!/usr/bin/env bash
  ./eval/set_ecmp_hashing_algo.sh source-port
  for i in {0..19}; do
    just one_8gpu_flow $i
  done


single-app-all:
  #!/usr/bin/env bash
  ./eval/set_ecmp_hashing_algo.sh everything
  for i in {0..9}; do
    just one_4gpu_ecmp $i
  done
  for i in {0..9}; do
    just one_8gpu_ecmp $i
  done
  # ./eval/set_ecmp_hashing_algo.sh source-port
  # for i in {0..9}; do
  #   just one_4gpu_flow $i
  # done
  # for i in {0..9}; do
  #   just one_8gpu_flow $i
  done
  ./eval/set_ecmp_hashing_algo.sh everything
  for i in {10..19}; do
    just one_4gpu_ecmp $i
  done
  for i in {10..19}; do
    just one_8gpu_ecmp $i
  done
  # ./eval/set_ecmp_hashing_algo.sh source-port
  # for i in {10..19}; do
  #   just one_4gpu_flow $i
  # done
  # for i in {10..19}; do
  #   just one_8gpu_flow $i
  # done
  

allreduce-multi type setup cnt:
  cargo run --bin launcher -- --configfile launcher/config.toml --benchmark eval/multi-app/output/multi-allreduce-{{type}}-setup{{setup}}.toml --silent --output-dir /tmp/multi-allreduce-{{type}}-{{cnt}}

batched-allreduce-multi:
  #!/usr/bin/env bash
  ./eval/set_ecmp_hashing_algo.sh everything
  for i in {1..3}; do
    for j in {0..9}; do
      just allreduce-multi ecmp $i $j
    done
  done
  ./eval/set_ecmp_hashing_algo.sh source-port
  for i in {1..3}; do
    just allreduce-multi flow $i 0
  done

batched-allreduce-multi2:
  #!/usr/bin/env bash
  ./eval/set_ecmp_hashing_algo.sh everything
  for j in {0..9}; do
    just allreduce-multi ecmp 3 $j
  done
  ./eval/set_ecmp_hashing_algo.sh source-port
  for i in {1..3}; do
    for j in {0..9}; do
      just allreduce-multi flow $i $j
    done
  done

allreduce-setup cnt:
  #!/usr/bin/env bash
  ./eval/set_ecmp_hashing_algo.sh everything
  for j in {0..9}; do
    just allreduce-multi ecmp {{cnt}} $j
  done
   ./eval/set_ecmp_hashing_algo.sh source-port
  for j in {0..9}; do
    just allreduce-multi flow {{cnt}} $j
  done


collect-cdf: 
  ./eval/set_ecmp_hashing_algo.sh source-port
  cargo run --bin launcher -- --configfile launcher/config.toml --benchmark eval/multi-app/output/setup4-real-fair.toml --silent --output-dir /tmp/setup4-cdf --timeout 600
  cargo run --bin launcher -- --configfile launcher/config.toml --benchmark eval/multi-app/output/setup4-real-qosv1.toml --silent --output-dir /tmp/setup4-cdf --timeout 600
  cargo run --bin launcher -- --configfile launcher/config.toml --benchmark eval/multi-app/output/setup4-real-qosv2.toml --silent --output-dir /tmp/setup4-cdf --timeout 600

one-setup4-normal type cnt:
  cargo run --bin launcher -- --configfile launcher/config.toml --benchmark eval/multi-app/output/setup4-real-{{type}}.toml --silent --output-dir /tmp/setup4-real-normal-{{cnt}} --timeout 600

one-setup4-ecmp type cnt:
  cargo run --bin launcher -- --configfile launcher/config.toml --benchmark eval/multi-app/output/setup4-real-ecmp-{{type}}.toml --silent --output-dir /tmp/setup4-real-ecmp-{{cnt}} --timeout 600

collect-setup4-ecmp:
  #!/usr/bin/env bash
  ./eval/set_ecmp_hashing_algo.sh everything
  for i in {0..9}; do
    for t in {'fair','qosv1','qosv2'}; do
      just one-setup4-ecmp $t $i
    done
  done

collect-setup4-normal:
  #!/usr/bin/env bash
  ./eval/set_ecmp_hashing_algo.sh source-port
  for i in {0..9}; do
    for t in {'fair','qosv1','qosv2'}; do
      just one-setup4-normal $t $i
    done
  done


setup2-vgg:
  ./eval/set_ecmp_hashing_algo.sh source-port
  cargo run --bin launcher -- --configfile launcher/config.toml --benchmark eval/multi-app/output/setup2-vgg-qos.toml --silent --output-dir /tmp/setup2-vgg-qos --timeout 180

setup4-vgg:
  ./eval/set_ecmp_hashing_algo.sh source-port
  cargo run --bin launcher -- --configfile launcher/config.toml --benchmark eval/multi-app/output/setup4-vgg-qos.toml --silent --output-dir /tmp/setup4-vgg-qos --timeout 180

setup1 what:
  ./eval/set_ecmp_hashing_algo.sh source-port
  cargo run --bin launcher -- --configfile launcher/config.toml --benchmark eval/multi-app/output/setup1-trace-{{what}}.toml --silent --output-dir /tmp/setup1-trace-{{what}}