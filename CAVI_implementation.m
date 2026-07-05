%% CAVI IMPLEMENTATION FOR MIXTURE OF GAUSSIANS

disp('CAVI implementation for mixture of gaussians');

% Input validation loop
while true
    user_input = input('Enter the true cluster means separated by spaces (e.g., -5 0 5): ', 's');
    % str2num takes space-separated strings and converts them to an array
    TRUE_MEANS = str2num(user_input); % if it fails it returns an empty array
    % and I check that below so I get the error
    
    if isempty(TRUE_MEANS)
        fprintf('Error: You either entered no number, or you entered an invalid expression.\n\n');
        continue;
    end

    if length(TRUE_MEANS)==1
        fprintf('Error: you must enter at least two numbers. \n \n')
        continue;
    end
    break; % Exit loop if successful
end

K = length(TRUE_MEANS);
sigma = 10.0; % High variance for prior to avoid anchoring to the prior

% Generate data
n_points = 1000;
fprintf('\nGenerating %d data points for %d clusters...\n', n_points, K);
[data, cluster_assignments] = generate_data(n_points, TRUE_MEANS, 1); % we do it here so we always use the same data for each attempt
fprintf('Starting CAVI (until ELBO converges) with prior variance on the means sigma = %.1f.\n\n', sigma);

% Run CAVI
[guess_means, guess_variances, guess_clusters, ELBO_values] = CAVI_Mixture_of_gaussians(data, K, sigma, 100, 1e-7, 20);

fprintf('\n---------------------------------------------\n');
fprintf('True cluster centers  : %s\n', sprintf('%5.2f, ', TRUE_MEANS));
fprintf('---------------------------------------------\n');

% Sort results by means for easier reading
[sorted_means, sorted_indices] = sort(guess_means);
sorted_variances = guess_variances(sorted_indices);

for k = 1:K
    fprintf('Cluster %d Estimation:\n', k);
    fprintf('Mean (m)     : %5.4f\n', sorted_means(k));
    fprintf('Variance (s²): %5.4f\n\n', sorted_variances(k));
end



% guess_clusters is an n x K matrix with for each point i probability to
% belong to each cluster k. Since we cannot display this, we check the
% mixture weights we found correspond to the ones of the data

% Account for label switching. The estimates are already
% ordered through sorted_indices = sort(guess_means); we apply the same rule
% to the true clusters so that true and estimated labels correspond.
sorted_phi = guess_clusters(:, sorted_indices);        % n x K, columns reordered

% Estimated mixture weights
mixture_weights = mean(sorted_phi, 1);                 % 1 x K

% we assgn each point i to the cluster with the highest phi_ik
[max_prob, cluster] = max(sorted_phi, [], 2);      % n x 1 each
cluster_sizes = accumarray(cluster, 1, [K 1]);     % K x 1, accumarray takes the sum of the number of points belonging to the cluster

% mixture_weights from the data
true_probs = accumarray(cluster_assignments(:), 1, [K 1]) / n_points; % K x 1
[~, true_order] = sort(TRUE_MEANS);
sorted_true_probs = true_probs(true_order);               % K x 1, aligned with the approximated probs

fprintf('\n----- Variational cluster assignment (phi) -----\n');
fprintf('Cluster |  approximate mixture weights  | true mixture weights | cluster size\n');
for k = 1:K
    fprintf('   %d    |  %10f  |     %9f     |      %4d\n', ...
            k, mixture_weights(k), sorted_true_probs(k), cluster_sizes(k));
end


% Plotting ELBO
figure('Position', [100, 100, 800, 500]);
plot(ELBO_values, '-ob', 'MarkerSize', 2, 'LineWidth', 1.5);
title('ELBO convergence', 'FontSize', 14);
xlabel('Iterations', 'FontSize', 12);
ylabel('ELBO', 'FontSize', 12);
grid on;

