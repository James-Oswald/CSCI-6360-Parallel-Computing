
# Notes

## Strange IO segfault issue
I have to run locally with `--mca io romio321` when using mpirun on my local machine or the program segfaults. I have gotten this advice from the OpenBSD mailing list at (https://www.mail-archive.com/misc@openbsd.org/msg177101.html)[https://www.mail-archive.com/misc@openbsd.org/msg177101.html]
```sh
mpirun -n 4 --mca io romio321 benchmark.out
```

## WSL debugging
I am able to use this hack for WSL debugging with GDB in multiple terms using:
```sh
mpirun -np 2 cmd.exe /c start cmd.exe /c "wsl.exe gdb benchmark.out"
```
I came up with this after seeing the xterm gdb hack seen at (https://www.open-mpi.org/faq/?category=debugging#serial-debuggers)[https://www.open-mpi.org/faq/?category=debugging#serial-debuggers] 