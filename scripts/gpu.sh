#!/bin/bash
#
# simply displays the current GPU usage and memory consumption every
# second on the console
#
# works only on Nvidia cards
#

watch -n 1 nvidia-smi
