#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=28G
#SBATCH -J nimbus_fiducial
#SBATCH --output=log.out
#SBATCH --partition roma 
#SBATCH --account kipac:kipac

date
#python3 cache_stars.py
python3 galaxy_properties.py
date
