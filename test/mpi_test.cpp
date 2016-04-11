#include "mpi.h"
#include <iostream>

#define MASTER 0

int main(int argc, char *argv[])
{

    int numtasks, taskid, len, partner, message;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;

    MPI_Status stats[2];
    MPI_Request reqs[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    std::cout << "Blocking stuff..." << std::endl;

    if (numtasks % 2 != 0) {
        if (taskid == MASTER)
            std::cout << "Quitting. Need an even number of tasks. Current number of tasks: " << numtasks << std::endl;
    } else {
        if (taskid == MASTER)
            std::cout << "MASTER: the number of MPI tasks is " << numtasks << std::endl;

        MPI_Get_processor_name(hostname, &len);
        std::cout << "Hello from task " << taskid << " on " << hostname << std::endl;

        if (taskid < numtasks / 2) {
            partner = numtasks / 2 + taskid;
            MPI_Send(&taskid, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
            MPI_Recv(&message, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &status);
        } else if (taskid >= numtasks /2) {
            partner = taskid - numtasks / 2;
            MPI_Recv(&message, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &status);
            MPI_Send(&taskid, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
        }

        std::cout << "Task " << taskid << " is partner with " << message << std::endl;
    }

    std::cout << "Non-blocking stuff..." << std::endl;

    if (numtasks % 2 != 0) {
        if (taskid == MASTER)
            std::cout << "Quitting. Need an even number of tasks. Current number of tasks: " << numtasks << std::endl;
    } else {
        if (taskid == MASTER)
            std::cout << "MASTER: the number of MPI tasks is " << numtasks << std::endl;

        MPI_Get_processor_name(hostname, &len);
        std::cout << "Hello from task " << taskid << " on " << hostname << std::endl;

        if (taskid < numtasks / 2) {
            partner = numtasks / 2 + taskid;
        } else if (taskid >= numtasks /2) {
            partner = taskid - numtasks / 2;
        }

        MPI_Irecv(&message, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(&taskid, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &reqs[1]);

        MPI_Waitall(2, reqs, stats);

        std::cout << "Task " << taskid << " is partner with " << message << std::endl;
    }

    MPI_Finalize();

    return 0;
}
