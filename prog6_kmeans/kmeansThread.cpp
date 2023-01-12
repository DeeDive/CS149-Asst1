#include <algorithm>
#include <cmath>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "CycleTimer.h"

using namespace std;

typedef struct
{
  // Control work assignments
  int start, end;

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int M, N, K;
  double *data_point_dist_to_centroids; // new arg member for parallelization
  int threadId;
  int numThreads;
} WorkerArgs;

/**
 * Checks if the algorithm has converged.
 *
 * @param prevCost Pointer to the K dimensional array containing cluster costs
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 *
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K)
{
  for (int k = 0; k < K; k++)
  {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimension nDim.
 *
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim)
{
  double accum = 0.0;
  for (int i = 0; i < nDim; i++)
  {
    accum += pow((x[i] - y[i]), 2);
  }
  return sqrt(accum);
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 */
void computeAssignments(WorkerArgs *const args)
{
  double *minDist = new double[args->M];

  // Initialize arrays
  for (int m = 0; m < args->M; m++)
  {
    minDist[m] = 1e30;
    args->clusterAssignments[m] = -1;
  }

  // Assign datapoints to closest centroids
  for (int k = args->start; k < args->end; k++)
  {
    for (int m = 0; m < args->M; m++)
    {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);

      if (d < minDist[m])
      {
        minDist[m] = d;
        args->clusterAssignments[m] = k;
      }
    }
  }

  free(minDist);
}
void workerThreadStart(WorkerArgs *const args)
{
  int K = args->end - args->start;
  for (int i = args->threadId; i < args->M; i += args->numThreads)
  {
    for (int k = args->start; k < args->end; k++)
    {
      args->data_point_dist_to_centroids[i * K + k] = dist(&args->data[i * args->N],
                                                           &args->clusterCentroids[k * args->N], args->N);
    }
  }
}
void computeAssignments_threaded(WorkerArgs *const args)
{
  double *minDist = new double[args->M];

  // Initialize arrays
  for (int m = 0; m < args->M; m++)
  {
    minDist[m] = 1e30;
    args->clusterAssignments[m] = -1;
  }

  static constexpr int MAX_THREADS = 32;

  if (args->numThreads > MAX_THREADS)
  {
    fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
    exit(1);
  }

  // Creates thread objects that do not yet represent a thread.
  std::thread workers[MAX_THREADS];
  WorkerArgs worker_args[MAX_THREADS];
  // Assign datapoints to closest centroids
  for (int i = 0; i < args->numThreads; i++)
  {

    worker_args[i].start = args->start;
    worker_args[i].end = args->end;
    worker_args[i].data = args->data;
    worker_args[i].clusterCentroids = args->clusterCentroids;
    worker_args[i].clusterAssignments = args->clusterAssignments;
    worker_args[i].currCost = args->currCost;
    worker_args[i].M = args->M;
    worker_args[i].N = args->N;
    worker_args[i].K = args->K;
    worker_args[i].data_point_dist_to_centroids = args->data_point_dist_to_centroids;
    worker_args[i].numThreads = args->numThreads;

    worker_args[i].threadId = i;
  }

  // Spawn the worker threads.  Note that only numThreads-1 std::threads
  // are created and the main application thread is used as a worker
  // as well.
  for (int i = 1; i < args->numThreads; i++)
  {
    workers[i] = std::thread(workerThreadStart, &worker_args[i]);
  }

  workerThreadStart(&worker_args[0]);

  // join worker threads
  for (int i = 1; i < args->numThreads; i++)
  {
    workers[i].join();
  }

  // synchronize
  int K = args->end - args->start;
  for (int i = 0; i < args->M; ++i)
  {
    double min_d = args->data_point_dist_to_centroids[i * K + args->start];
    args->clusterAssignments[i] = args->start;
    for (int k = args->start + 1; k < args->end; k++)
    {
      if (args->data_point_dist_to_centroids[i * K + k] < min_d)
      {
        min_d = args->data_point_dist_to_centroids[i * K + k];
        args->clusterAssignments[i] = k;
      }
    }
  }

  // free(args->data_point_dist_to_centroids);
}
/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids(WorkerArgs *const args)
{
  int *counts = new int[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++)
  {
    counts[k] = 0;
    for (int n = 0; n < args->N; n++)
    {
      args->clusterCentroids[k * args->N + n] = 0.0;
    }
  }

  // Sum up contributions from assigned examples
  for (int m = 0; m < args->M; m++)
  {
    int k = args->clusterAssignments[m];
    for (int n = 0; n < args->N; n++)
    {
      args->clusterCentroids[k * args->N + n] +=
          args->data[m * args->N + n];
    }
    counts[k]++;
  }

  // Compute means
  for (int k = 0; k < args->K; k++)
  {
    counts[k] = max(counts[k], 1); // prevent divide by 0
    for (int n = 0; n < args->N; n++)
    {
      args->clusterCentroids[k * args->N + n] /= counts[k];
    }
  }

  free(counts);
}

/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args)
{
  double *accum = new double[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++)
  {
    accum[k] = 0.0;
  }

  // Sum cost for all data points assigned to centroid
  for (int m = 0; m < args->M; m++)
  {
    int k = args->clusterAssignments[m];
    accum[k] += dist(&args->data[m * args->N],
                     &args->clusterCentroids[k * args->N], args->N);
  }

  // Update costs
  for (int k = args->start; k < args->end; k++)
  {
    args->currCost[k] = accum[k];
  }

  free(accum);
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in
 *     the array. The N values of the i'th datapoint are the N values in the
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
                  int M, int N, int K, double epsilon, int numThreads = 1)
{

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];

  double *data_point_dist_to_centroids = new double[M * K];

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  WorkerArgs args;
  args.data = data;
  args.clusterCentroids = clusterCentroids;
  args.clusterAssignments = clusterAssignments;
  args.currCost = currCost;
  args.data_point_dist_to_centroids = data_point_dist_to_centroids;
  args.numThreads = numThreads; // TODO: get_opt
  args.M = M;
  args.N = N;
  args.K = K;

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++)
  {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K))
  {
    // Update cost arrays (for checking convergence criteria)
    for (int k = 0; k < K; k++)
    {
      prevCost[k] = currCost[k];
    }

    // Setup args struct
    args.start = 0;
    args.end = K;

    double startTime1 = CycleTimer::currentSeconds();
    computeAssignments_threaded(&args);
    double endTime1 = CycleTimer::currentSeconds();
    // printf("[computeAssignments Time]: %.3f ms\n", (endTime1 - startTime1) * 1000);

    double startTime2 = CycleTimer::currentSeconds();
    computeCentroids(&args);
    double endTime2 = CycleTimer::currentSeconds();
    // printf("[computeCentroids Time]: %.3f ms\n", (endTime2 - startTime2) * 1000);

    double startTime3 = CycleTimer::currentSeconds();
    computeCost(&args);
    double endTime3 = CycleTimer::currentSeconds();
    // printf("[computeCost Time]: %.3f ms\n", (endTime3 - startTime3) * 1000);

    iter++;
  }

  free(currCost);
  free(prevCost);
}
