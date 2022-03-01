#!/bin/bash -x

if [ "x$SLURM_NPROCS" = "x" ]
then
  if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
  then
    SLURM_NTASKS_PER_NODE=1
  fi
  SLURM_NPROCS=`expr $SLURM_JOB_NUM_NODES \* $SLURM_NTASKS_PER_NODE`
else
  if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
  then
    SLURM_NTASKS_PER_NODE=`expr $SLURM_NPROCS / $SLURM_JOB_NUM_NODES`
  fi
fi

srun hostname -s | sort -u > /tmp/hosts.$SLURM_JOB_ID
grep -q 'release 7\.' /etc/redhat-release
if [ $? -eq 0 ]; then
  net_suffix=-ib
fi
awk "{ print \$0 \"$net_suffix slots=$SLURM_NTASKS_PER_NODE\"; }" /tmp/hosts.$SLURM_JOB_ID >/tmp/tmp.$SLURM_JOB_ID
mv /tmp/tmp.$SLURM_JOB_ID /tmp/hosts.$SLURM_JOB_ID

module load spectrum-mpi
mpirun --bind-to core --report-bindings -hostfile /tmp/hosts.$SLURM_JOB_ID -np $SLURM_NPROCS /PATH/TO/YOUR/MPI-Executable

#kokkos users should add: --kokkos-num-devices=N
#where N is the number of gpus being used on each node

rm /tmp/hosts.$SLURM_JOB_ID
