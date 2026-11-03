% Drona MATLAB wrapper: set the threads, optionally start a local pool, run the main script.
% Settings come from the job script (DRONA_MATLAB_*), so this file is the same for every job.
t = str2double(getenv('DRONA_MATLAB_THREADS'));  if isnan(t) || t < 1, t = 1; end
w = str2double(getenv('DRONA_MATLAB_WORKERS'));  if isnan(w) || w < 0, w = 0; end
maxNumCompThreads(t);
if w > 0
    c = parcluster('local');
    % per-job storage: concurrent jobs sharing ~/.matlab/local_cluster_jobs can corrupt each other
    c.JobStorageLocation = fullfile(pwd, '.drona_matlab', getenv('SLURM_JOB_ID'));
    if ~exist(c.JobStorageLocation, 'dir'), mkdir(c.JobStorageLocation); end
    c.NumWorkers = w;
    c.NumThreads = t;
    parpool(c, w);
end
run(getenv('DRONA_MATLAB_MAIN'));
delete(gcp('nocreate'));
