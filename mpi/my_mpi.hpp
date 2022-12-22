#pragma once

#include <vector>

#include <mpi.h>


/* int */
/* mpi_recv(void* buf, int count); */

/* int */
/* mpi_send(const void* buf, int count, int dest); */

/* int */
/* mpi_gatherv( */
/*     const void* sendbuf, */
/*     int sendcount, */
/*     void* recvbuf, */
/*     const int* recvcounts, */
/*     const int* displs */
/* ); */


int
mpi_recv(void* buf, int count);

int
mpi_send(const void* buf, int count, int dest);

int
mpi_gatherv(
    const std::vector<int>& sendbuf,
    int sendcount,
    std::vector<int>& recvbuf,
    const std::vector<int>& recvcounts,
    const std::vector<int>& displs
);

