#!/usr/bin/env bash

usage() {
        echo "Usage: $0: [mccs|base]"
}

if [ $# -ne 1 ]; then
        usage
        exit 1
fi

arg=$1

case $arg in
        base)
                sudo -u cjr \
                ssh danyang-01 \
                "ssh -oKexAlgorithms=+diffie-hellman-group14-sha1 danyang@danyang-mellanox-switch.cs.duke.edu \
                        cli -h '\"enable\" \"config terminal\" \"spanning-tree\" \"no protocol openflow\" \"show running-config\"'"
                ;;
        mccs)
read -r -d '' cmds_example << EOF
\"enable\" \
\"config terminal\" \
\"spanning-tree\" \
\"no protocol openflow\" \
\"# This line is a comment\" \
\"show running-config\"
EOF
read -r -d '' pysrc << EOF
print('\n'.join(f'\\"{l.strip()}\\" \\' for l in open('switch_enable_mccs_commands.txt', 'r')))
EOF
cmds=`python3 -c $"print(' '.join(f'\x5c\"{l.strip()}\x5c\"' for l in open('switch_enable_mccs_commands.txt', 'r')))"`
echo $cmds
                sudo -u cjr \
                ssh danyang-01 \
                "ssh -oKexAlgorithms=+diffie-hellman-group14-sha1 danyang@danyang-mellanox-switch.cs.duke.edu \
                        cli -h ${cmds}"
                ;;
        *)
                echo "Error: argument should be either 'mccs' or 'base', got $arg"
                usage
                exit 1
                ;;
esac

        # cli -h '\"enable\" \"show lldp remote\"'"
