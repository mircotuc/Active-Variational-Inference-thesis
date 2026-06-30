function data = generate_data(n, true_means, variance) % it returns the data

    K = length(true_means);
    cluster_assignments = randi([1, K], n, 1); %it randomly chooses the cluster, 
    %returning an array nx1 where each element can go from 1 to K
    data = zeros(n, 1); % Initialize column vector
    
    for k = 1:K
        index = (cluster_assignments == k); %it returns an array of trues and falses.
        num_points_in_cluster = sum(index);
        % Generate normal distribution points for the specific cluster,
        % modifying all the points from that cluster in data at the same time
        data(index) = true_means(k) + sqrt(variance) * randn(num_points_in_cluster, 1);
        % randn draws from the standard normal distribution, so I need to
        % multiply by the variance (1) and add the means
    end
end