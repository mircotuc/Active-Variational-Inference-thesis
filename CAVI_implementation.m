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
data = generate_data(n_points, TRUE_MEANS, 1); % we do it here so we always use the same data for each attempt
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

% Plotting ELBO
figure('Position', [100, 100, 800, 500]);
plot(ELBO_values, '-ob', 'MarkerSize', 2, 'LineWidth', 1.5);
title('ELBO convergence', 'FontSize', 14);
xlabel('Iterations', 'FontSize', 12);
ylabel('ELBO', 'FontSize', 12);
grid on;

