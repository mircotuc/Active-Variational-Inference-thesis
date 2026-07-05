function [best_m, best_s, best_phi, best_ELBO_values] = CAVI_Mixture_of_gaussians(data, K, sigma, iter_max, abs_tol, starts)
    %inputs: fake data, number of gaussians K, prior variance for the means, number of restarts of the algorithm
    n = length(data);

    best_ELBO = -inf;
    best_m = zeros(K, 1);
    best_s = zeros(K, 1);
    best_phi = zeros(n, K);
    best_ELBO_values = zeros(iter_max, 1);

    for attempt = 1:starts % we are starting the algorithm many times and then choosing the best output
        % Initialize m by drawing random unique samples from data
        % here we want the clusters to have different mean so they won't go in the same direction.
        % randperm permutates integers from 1 to n (so none is repeated)
        % and then takes the first K values
        % we choose them from the data points to guarantee they are in the right range
        fprintf('Attempt %d \n', attempt);
        idx = randperm(n, K); % K x 1 vector with integers

        m = data(idx); % Kx1 vector        
        % now we are mapping these integers to points in data
        
        % this is the variance for the guess of the distribution of the means
        % starting with a higher variance allows for more exploration of the data
        s = 5 * ones(K, 1);   % K x 1 vector
        phi = zeros(n, K);    % n x K matrix, this is updated straight away so it's ok to leave it at zero

        ELBO = -inf;
        ELBO_values = zeros(iter_max,1); % to plot

        for iter = 1:iter_max
            % Update phi
            expected_mu = m'; % 1 x K
            expect_mu_squared = (s + m.^2)'; % 1 x K
            
            % n x K
            exponent = data .* expected_mu - 0.5 * expect_mu_squared;
            % here we are subtracting a 1xK vector from a nxK matrix.
            % MATLAB implicity expands the vector to a matrix nxK with rows
            % all equal and then does the subtraction
            
            % Normalize for numerical stability
            exponent = exponent - max(exponent, [], 2); % taking the max of the columns
            phi = exp(exponent);
            phi = phi ./ sum(phi, 2); % Normalize to make it a probability
            % Update m and s
            for k = 1:K
                sum_phi = sum(phi(:, k)); %1x1 because k fixed
                sum_phi_x = sum(phi(:, k) .* data); % sum(nx1 .* nx1) = sum(nx1) = 1x1

                if sum_phi > 1e-10 % I need this because otherwise m would update to 0 and go back to my prior beliefs
                    % but sum_phi=0 doesn't mean that. It means that no points belong to that cluster, so we must treat it as 'dead' and not update it
                    s(k) = 1.0 / (1.0/sigma + sum_phi);
                    m(k) = sum_phi_x / (1.0/sigma + sum_phi);
                end
            end

            % Compute current ELBO
            % E[log p(mu)]
            E_log_p_mu = sum(-0.5*log(2*pi*sigma) - 0.5*(m.^2 + s)/sigma);
            
            % E[log p(c)]
            E_log_p_c = -n * log(K);
            
            % E[log p(x | c, mu)]
            % (data - m') expands to an n x K matrix
            E_log_p_x = sum(sum(phi .* (-0.5*log(2*pi) - 0.5*((data - m').^2 + s'))));
            
            % E[log q(mu)]
            E_log_q_mu = sum(0.5*log(2*pi .* s .* exp(1))); % here I incoroporated the 1/2 as 0.5*log(e). There is not the minus in front bec from calculations we deduced it was positive
            
            % E[log q(c)]
            E_log_q_c = -sum(sum(phi .* log(phi + 1e-10))); % I added a tolerance to avoid log(0)

            current_ELBO = E_log_p_mu + E_log_p_c + E_log_p_x + E_log_q_mu + E_log_q_c;
            ELBO_values(iter) = current_ELBO; 
            
            %fprintf('Iter %2d | ELBO: %12.4f\n', iter, current_ELBO);

            % Check convergence
            if iter > 1 && abs(current_ELBO - ELBO) < abs_tol
                fprintf('\n>> The algorithm converges at iter %d!\n', iter);
                break;
            end
            ELBO = current_ELBO;
        end

        ELBO_values = ELBO_values(1:iter); % so I don't print the zeros of the iters after the final iter
        
        for k = 1:K
            fprintf('Cluster %d Estimation:\n', k);
            fprintf('Mean (m)     : %5.4f\n', m(k));
            fprintf('Variance (s²): %5.4f\n\n', s(k));
        end
    
        if ELBO > best_ELBO
            best_ELBO = ELBO;
            best_m = m;
            best_phi = phi;
            best_s = s;
            best_ELBO_values = ELBO_values;
        end
    end
end