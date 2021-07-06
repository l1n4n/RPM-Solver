#!/bin/bash

basicD=$(cat results.txt | grep -A1 'Basic D-' | grep -o Pass | wc -l)
basicE=$(cat results.txt | grep -A1 'Basic E-' | grep -o Pass | wc -l)
testD=$(cat results.txt | grep -A1 'Test D-' | grep -o Pass | wc -l)
testE=$(cat results.txt | grep -A1 'Test E-' | grep -o Pass | wc -l)
challengeD=$(cat results.txt | grep -A1 'Challenge D-' | grep -o Pass | wc -l)
challengeE=$(cat results.txt | grep -A1 'Challenge E-' | grep -o Pass | wc -l)
ravensD=$(cat results.txt | grep -A1 "Raven's D-" | grep -o Pass | wc -l)
ravensE=$(cat results.txt | grep -A1 "Raven's E-" | grep -o Pass | wc -l)

printf "Basic (D, E) Score: ($basicD, $basicE) \n"
printf "Test (D, E) Score: ($testD, $testE) \n"
printf "Challenge (D, E) Score: ($challengeD, $challengeE) \n"
printf "Raven's (D, E) Score: ($ravensD, $ravensE) \n"
