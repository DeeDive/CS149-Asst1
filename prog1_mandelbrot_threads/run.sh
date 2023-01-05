#!/bin/bash

while getopts v: flag
do
    case "${flag}" in
        v) view=${OPTARG};;
    esac
done
for i in {1..12}
do
   ./mandelbrot -t $i -v $view
done
./mandelbrot -t 24 -v $view
