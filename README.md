## KSimplex Multi-Precision Integer Simplex Management ##
The KSimplex project is a proof-of-concept for efficient multi-precision integer arithmetic using the CUDA GPGPU framework. Preliminary results show comparable performance to the highly-optimized GMP library on host.

The KiloMP subproject contains the multi-precision integer implmentation, while KSimplex uses KiloMP for an integer simplex data structure modeled on David Avis' LRS (also vendored in this repository), along with a number of comparable implementations.