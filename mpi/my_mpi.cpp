#include "my_mpi.hpp"

int
mpi_recv(void* buf, int count)
{
    return MPI_Recv(
        buf, count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
};

int
mpi_send(const void* buf, int count, int dest)
{
    return MPI_Send(buf, count, MPI_INT, dest, 0, MPI_COMM_WORLD);
};

int
mpi_gatherv(const std::vector<int>& sendbuf,
            int sendcount,
            std::vector<int>& recvbuf,
            const std::vector<int>& recvcounts,
            const std::vector<int>& displs)
{
    return MPI_Gatherv(sendbuf.data(),
                       sendcount,
                       MPI_INT,
                       recvbuf.data(),
                       recvcounts.data(),
                       displs.data(),
                       MPI_INT,
                       0,
                       MPI_COMM_WORLD);
}
