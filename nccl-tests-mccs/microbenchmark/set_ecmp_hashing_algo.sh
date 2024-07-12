#!/usr/bin/env bash

usage() {
	echo "Usage: $0: [everything|source-port]"
}

if [ $# -ne 1 ]; then
	usage
	exit 1
fi

algo=$1

case $algo in
	everything)
		algo_args="source-destination-mac source-destination-ip source-destination-port l3-protocol l2-protocol flow-label"
		;;
	source-port)
		algo_args="source-port"
		;;
	*)
		echo "Error: algo should be either 'everything' or 'source-port', got $algo"
		usage
		exit 1
		;;
esac

sudo -u cjr \
ssh danyang-01 \
"ssh -oKexAlgorithms=+diffie-hellman-group14-sha1 danyang@danyang-mellanox-switch.cs.duke.edu \
	cli -h '\"enable\" \"config terminal\" \"port-channel load-balance ethernet $algo_args\" \"show interfaces port-channel load-balance\"'"

	# cli -h '\"enable\" \"show lldp remote\"'"
