%% IMPLEMENTATION OF A POMDP

% Slot machine problem from A step-by-step guide to active variational
% inference by Smith et al.

% INITIALIZING MATRICES

% we won't consider habits so we won't inser matrix E
% and we consider deep policies V, not shallow ones U

% OUTCOME MODALITIES (OBSERVATIONS) o: 
    % 1) HINT: no hint, hint-left, hint-right
    % 2) WIN: start, lose, win
    % 3) OBSERVED ACTION: start, hint, choose-left, choose-right
% POSSIBLE STATE FACTORS s:
    % 1) CONTEXT: left-better, right-better
    % 2) CHOICE: start, hint, choose-left, choose-right
% POSSIBLE ACTIONS: (go to) start, hint, left, right

%THE PROBLEM IS THAT WHEN EVALUATINNG G, THE AGENT DOES NOT KNOW THAT IF HE
%CHOOSES THE HINT HE WILL CHANGE ITS DECISION AND WIN ALMOST SURELY. THAT
%IS DONE WITH RECURSIVE COMPLICATED ALGORITHM. HERE WE ARTIFICIALLY
%SIMULATE IT

[A, B, C, D, V] = building_matrices(0, 1, 8);  % CHANGE this parameters to simulate risk-aversion.
%If you increase c_loss, the probability of choosing the hint will increase. 
%If you increase c_reward, the probability of skipping the hint will rise


num_obs = [3, 3, 4]; % [HINT, WIN, OBSERVED ACTION]
num_states = [2, 4]; % [CONTEXT, CHOICE]
num_actions = 4;
num_policies = 4;
T = 3; % time periods
%gamma= [0.1, 4, 4]; % trust parameter in EFE. at the start it is lower because there is more uncertainty

% initialize states and beliefs

true_context = 1; % let's assume left-better
true_choice = 1; % the agent always begins at start

% BELIEFS TENSORS qs{factor}(state, tau, policy)
qs = cell(1, 2); % one tensor per state factor
qs{1} = ones(num_states(1), T, num_policies) / num_states(1); % tensor 2x3x5
qs{2} = ones(num_states(2), T, num_policies) / num_states(2); % tensor 4x3x5
% I'm dividing by num_states to get a probability distribution (a uniform
% one in particular) otherwise I would have had all ones and that wouldn't
% have been a prob distribution. since Qs is Q(s|pi) if i ask myself at
% every time t how much is the probability of s|pi1 I get [0.5,0.5]
% Initialize observations for every time t
O1 = zeros(1, T); 
O2 = zeros(1, T); 
O3 = zeros(1, T);

% I'm already calculating the backward transition matrices, because I will
% need them multiple times
% Create B_dagger (Transpose and column-normalize B)
% I have to normalize because I want a probability
% distribution in every column, like I had in B.
% Transposing ruins that.
B1_dag = B{1}(:, :, 1)'; % I'm choosing action 1 because B{1} is the same for every action anyway
% now every column will have the transition prob from a
% certain state tau+1 to the poissible states tau
B1_dag = B1_dag ./ sum(B1_dag, 1); % I divide each element by the sum of the column to normalize
B1_dag(isnan(B1_dag)) = 0; % Prevent division by zero if i get a column with all zero (0/0 would return NaN)


B2_dag_array = cell(1, 4); % here instead the backward transition matrix depends on the action, so I create one for every action and put them in an array
for a = 1:4
    temp_B = B{2}(:, :, a)';
    temp_B = temp_B ./ sum(temp_B, 1);
    temp_B(isnan(temp_B)) = 0;
    B2_dag_array{a} = temp_B;
end

% for plotting
true_context_history = ones(1, T) * true_context; 
true_choice_history  = zeros(1, T);
belief_context_history   = zeros(num_states(1), T);
belief_choice_history    = zeros(num_states(2), T);
pi_history           = zeros(num_policies, T);
% for plotting

for t = 1:T
    fprintf('\n--- REAL TIME STEP t = %d ---\n', t);
    % here we update Qs
    % t is the time when we infer
    % tau is the time we infer about

    true_choice_history(t) = true_choice; % for plotting
    
    % Generate true observations
    % Extract the true probability distributions from A
    P_hint   = A{1}(:, true_context, true_choice);
    P_win    = A{2}(:, true_context, true_choice);
    P_observed_action = A{3}(:, true_context, true_choice);
    
    % Sample the observations and store them in the history arrays
    % cumsum returns a vector of the cumulative sums
    % rand generates a number between 0 and 1
    % < returns an array of true or false for every element of cumsum
    % find takes the first true and returns the index, so for ex if O1 is
    % 1, this means that there is no hint.
    % at iter=1 the observations are deterministically No hint, start,
    % start
    O1(t) = find(rand < cumsum(P_hint), 1);
    O2(t) = find(rand < cumsum(P_win), 1);
    O3(t) = find(rand < cumsum(P_observed_action), 1);
    
    fprintf('Observed: Hint=%d, Win=%d, Action=%d\n', O1(t), O2(t), O3(t));

    % I can calculate the not-marginalized likelyhood here because they 
    % do not depend on policies. they only depend on tau
    L_total_array = cell(1,t); % I need an L_total for each time step
    for tau = 1:t
        % here we are doing A*Oi which just gives us the Oi
        % slice of A. Squeeze just deletes the dimension of o,
        % leaving a 2x4
        L1 = squeeze(A{1}(O1(tau), :, :)); 
        L2 = squeeze(A{2}(O2(tau), :, :)); 
        L3 = squeeze(A{3}(O3(tau), :, :)); 
        L_total_array{tau} = L1 .* L2 .* L3; % since O1, O2,O3 are independent, we can multiply them to get A*o
    end

    % Marginal message passing algorithm for updating Qs
    num_iter = 16; % we're using equations that derive from a gradient descent on F so we need convergence
    

    % the agent beliefs depend on the policy he is considering so we must update for each policy
    for pi = 1:num_policies
        for iter = 1:num_iter
            % we're using marginal message passing, so to update Qs we
            % consider likelyhood, info from the future (where must I be today in order to be there tomorrow?),
            % info from the past (if I was there in the past how likely do
            % I end up here?)
            for tau = 1:T
                
                % LIKELIHOOD MESSAGE 
                if tau <= t
                    % If tau is in the past or present, we have real data
                    L_total = L_total_array{tau}; 
                    
                    % Marginalize based on the OTHER state factor
                    % otherwise they would influence each other
                    likelyhood_context = log(L_total + eps) * qs{2}(:, tau, pi); % add eps to avoid log(0)
                    likelyhood_choice  = log(L_total + eps)' * qs{1}(:, tau, pi);
                else
                    % If tau is in the future, we have no empirical data yet
                    likelyhood_context = 0; 
                    likelyhood_choice  = 0;
                end
                
                % FORWARD MESSAGE (From the Past)
                if tau == 1
                    % At tau=1, the past is the prior D
                    forward_context = log(D{1} + eps);
                    forward_choice  = log(D{2} + eps);
                else
                    % At tau>1, project the previous state forward via B
                    action_taken = V{2}(tau-1, pi); % Action leading into tau, taken at tau-1
                    % here I apply the transition of the markov chain 
                    forward_context = log(B{1}(:, :, 1) * qs{1}(:, tau-1, pi) + eps);
                    forward_choice  = log(B{2}(:, :, action_taken) * qs{2}(:, tau-1, pi) + eps);
                end
                
                % BACKWARD MESSAGE (From the Future)
                if tau < T

                    action_future = V{2}(tau, pi); % Action leading out of tau (taken now, I want to study the consequences)
                    %B1_dag is the same I calculated at the start because
                    %it doesn't depedn on the action chosen
                    B2_dag = B2_dag_array{action_future}; % here instead I take the prob associated with the
                    % state that the action I took at time tau will lead
                    % into
                    
                    % Pull the future state backward, transition from my
                    % belief in state tau+1 to state tau. I'm asking myself
                    % where I wanna be today in order to be there tomorrow.
                    % at iter=1 qs(tau+1) is simply a uniform distribution
                    % soi gives me no info, but as iterations go on, it
                    % becomes more accurate
                    backward_context = log(B1_dag * qs{1}(:, tau+1, pi) + eps);
                    backward_choice  = log(B2_dag * qs{2}(:, tau+1, pi) + eps);
                else
                    % At tau=T, there is no future to send messages backward
                    backward_context = 0; 
                    backward_choice  = 0;
                end
                
                % UPDATE EQUATIONS
                if tau < T  % at tau = 1 we already set the forward with D and not B
                    total_messages_context = 0.5 * (forward_context + backward_context) + likelyhood_context;
                    total_messages_choice  = 0.5 * (forward_choice  + backward_choice)  + likelyhood_choice;
                else % it's true we already set the backward to 0 for tau=T but without the else the forward would be multiplied by only 0.5
                    % At the end of the timeline, only the forward message exists
                    total_messages_context = forward_context + likelyhood_context;
                    total_messages_choice  = forward_choice  + likelyhood_choice;
                end
                
                % Softmax for Context
                total_messages_context = total_messages_context - max(total_messages_context); 
                % since I will have to exponentiate this quantity, I don't
                % want it to be too large or it will become inf. So I
                % divide every element of my vector by the max element.
                % this doesn't change the result because of exponential
                % properties.
                qs{1}(:, tau, pi) = exp(total_messages_context) / sum(exp(total_messages_context));
                
                % Softmax for Choice
                total_messages_choice = total_messages_choice - max(total_messages_choice);
                qs{2}(:, tau, pi) = exp(total_messages_choice) / sum(exp(total_messages_choice));
                
            end % End tau loop
        end % End iter loop
    end % End policy loop
    
    % Print the agent's current belief about reality (average across policies for tau=t)
    avg_belief_context = mean(qs{1}(1, t, :));
    avg_belief_choice  = mean(qs{2}(1, t, :));
    fprintf('Agent Belief -> P(Context is Left-Better): %.2f%%\n', avg_belief_context * 100);

    belief_context_history(:, t) = mean(qs{1}(:, t, :), 3); % for plotting
    belief_choice_history(:, t)  = mean(qs{2}(:, t, :), 3); % for plotting

    % UPDATE F AND G
    if t < T % We only plan and act if there is still a future left to experience
        
        F = zeros(1, num_policies); % VFE (Past/Present), for each policy
        G = zeros(1, num_policies); % EFE (Future), for each policy
        
        % calculate Ambiguity matrices for G (Shannon Entropy of A)
        % This corresponds to -diag(A^T * ln A). It doesn't depend on pi
        H_A = cell(1, 3);
        for m = 1:3
            % We sum across the first dimension (the observation probabilities)
            H_A{m} = squeeze(sum(-A{m} .* log(A{m} + eps), 1));
        end
        
        for pi = 1:num_policies
            
            % calculate F
            for tau = 1:t % only for past and present (it stops at t)
                state_context = qs{1}(:, tau, pi);
                state_choice  = qs{2}(:, tau, pi);
                
                % LIKELYHOOD
                L_total = L_total_array{tau};
                % to explain observations I need context and choice
                % together (P(obs|context,choice)). I cannot divide tem
                % simply in P(obs|context) and P(obs|choice)
                % we need to do a double sum over context and choice states
                % (to marginalize them out). it is the same thing as
                % multiplying the vectors. the result is a scalar
                accuracy = state_context' * log(L_total + eps) * state_choice;
                
                % FORWARD (as above but q has changed)
                if tau == 1
                    forward_context = log(D{1} + eps);
                    forward_choice  = log(D{2} + eps);
                else
                    action_taken = V{2}(tau-1, pi);
                    forward_context = log(B{1}(:, :, 1) * qs{1}(:, tau-1, pi) + eps);
                    forward_choice  = log(B{2}(:, :, action_taken) * qs{2}(:, tau-1, pi) + eps);
                end
                
                % BACKWARD
                if tau < T
                    action_future = V{2}(tau, pi);
                    B2_dag  = B2_dag_array{action_future};
                    backward_context = log(B1_dag * qs{1}(:, tau+1, pi) + eps);
                    backward_choice  = log(B2_dag * qs{2}(:, tau+1, pi) + eps);
                else
                    backward_context = 0; 
                    backward_choice  = 0;
                end
                
                
                if tau < T
                    prior_context = 0.5 * (forward_context + backward_context);
                    prior_choice  = 0.5 * (forward_choice  + backward_choice);
                else
                    prior_context = forward_context;
                    prior_choice  = forward_choice;
                end
                
                % complexity, unlike accuracy, is additive, because in the
                % transitions from a state to another, context and choice
                % are independent
                complexity_context = state_context' * (log(state_context + eps) - prior_context);
                complexity_choice = state_choice' * (log(state_choice + eps) - prior_choice);

                % UPDATE VFE: s * (ln s - prior - likelihood)  
                % total VFE = sum over tau of VFE(tau)
                F(pi) = F(pi) + complexity_choice + complexity_context - accuracy;
            end
            
            % calculate G
            for tau = t+1:T % G instead is calculated only in the future
                state_context = qs{1}(:, tau, pi);
                state_choice  = qs{2}(:, tau, pi);
                
                G_tau = 0; % EFE for this specific future time step, at the end I will add it to the total EFE for that policy
                
                for m = 1:3 % Loop through the 3 observation modalities
                    
                    % Expected Observation (Qo) (A*s) given current beliefs
                    Qo = zeros(num_obs(m), 1);
                    for obs_mod = 1:num_obs(m)
                        % Marginalize over both states as above
                        Qo(obs_mod) = state_context' * squeeze(A{m}(obs_mod, :, :)) * state_choice;
                    end

                    % RISK (or pragmatic value or reward)
                    % KL divergence between expected obs and preferences C
                    % (As * (ln(As) - lnC))
                    % we write C in place of lnC because we already
                    % definedd C as a log-probability matrix

                    % ARTIFICIALLY ENSURING THAT IF WE GET THE HINT WE DO
                    % NOT HAVE C_LOSS
                    if t==1 && V{2}(1,pi)==2 && m==2 % if t=1, because after that I know i Got the hint, if i choose hint and I'm looking at outcome modality win
                        temp_C = C{m}(:, tau);
                        temp_C(temp_C < 0) = 0; % we delete the c_loss penalty
                        risk = Qo' * (log(Qo + eps) - temp_C);
                    else
                        risk = Qo' * (log(Qo + eps) - C{m}(:, tau)); % we extract the preference of that observation at this time (tau)
                    end
                    % AMBIGUITY (or epistemic value or information gain)
                    % (-diag(Atraspose*lnA)*s)
                    ambiguity = state_context' * H_A{m} * state_choice;
                    
                    G_tau = G_tau + ambiguity + risk;
                end
                
                G(pi) = G(pi) + G_tau;
            end
        end
        
        % POLICY SELECTION

        % update pi prob distribution
        % (Assuming precision gamma = 1 and prior over policies E is uniform)
        % pi_nosoftmax= -F -gamma(t)*G;
        pi_nosoftmax = - F - G; 
        pi_nosoftmax = pi_nosoftmax - max(pi_nosoftmax); % as above
        pi_updated  = exp(pi_nosoftmax) / sum(exp(pi_nosoftmax));

        pi_history(:, t) = pi_updated; % for plotting
        
        fprintf('Policy Probabilities [Hint+L, Hint+R, L+Wait, R+Wait]:\n');
        disp(round(pi_updated, 4)); % Rounding just for cleaner console output
        
        % Sample a policy using the cumsum as above
        selected_policy = find(rand < cumsum(pi_updated), 1);
        
        % select the action to take for the time step we're in
        action_to_take = V{2}(t, selected_policy);
        
        fprintf('Selected Policy: %d -> Executing Action: %d\n', selected_policy, action_to_take);
        
        % state transition
        % The true context (Left/Right better) never changes.
        % The true choice updates based on the B matrix and the action taken.
        true_choice = find(rand < cumsum(B{2}(:, true_choice, action_to_take)), 1);
    end
end

% PLOTTING LIKE THE ARTICLE
figure('Name', 'Active Inference: T-Maze Trial', 'Color', 'white', 'Position', [100, 100, 900, 800]);

% --- 1. Hidden States (Context) ---
subplot(3, 2, 1);
% 1 - belief makes high probability black, low probability white
imagesc(1 - belief_context_history); colormap(gray); hold on;
% Overlay true states as cyan dots
scatter(1:T, true_context_history, 120, 'c', 'filled', 'MarkerEdgeColor', 'k');
yticks(1:2); yticklabels({'left-better', 'right-better'});
xticks(1:T); title('Hidden states - contexts', 'FontWeight', 'bold');

% --- 2. Action - Choice States ---
subplot(3, 2, 2);
imagesc(1 - belief_choice_history); colormap(gray); hold on;
scatter(1:T, true_choice_history, 120, 'c', 'filled', 'MarkerEdgeColor', 'k');
yticks(1:4); yticklabels({'start', 'hint', 'left', 'right'});
xticks(1:T); title('Action - choice states', 'FontWeight', 'bold');

% --- 3. Posterior Probability (Policies) ---
subplot(3, 2, 3);
% We only plot 1 to T-1 because no planning happens at the final time step
imagesc(1 - pi_history(:, 1:T-1)); colormap(gray);
yticks(1:num_policies); ylabel('Policy Number');
xticks(1:T-1); title('Posterior probability (Policies)', 'FontWeight', 'bold');

% --- 4. Outcomes and Preferences (Win/Lose) ---
subplot(3, 2, 4);
% We map the preferences. This figure will always be the same
imagesc(-C{2}); colormap(gray); hold on;
scatter(1:T, O2, 120, 'c', 'filled', 'MarkerEdgeColor', 'k');
yticks(1:3); yticklabels({'start', 'lose', 'win'});
xticks(1:T); title('Outcomes & preferences - win/lose', 'FontWeight', 'bold');

% --- 5. Outcomes (Hint) ---
% it is going to always be gray, just the dots change
subplot(3, 2, 5);
imagesc(-C{1}); colormap(gray); hold on;
scatter(1:T, O1, 120, 'c', 'filled', 'MarkerEdgeColor', 'k');
yticks(1:3); yticklabels({'null', 'left hint', 'right hint'});
xticks(1:T); title('Outcomes - hint', 'FontWeight', 'bold');

% --- 6. Outcomes (Observed Action) ---
% it is going to always be gray, just the dots change
subplot(3, 2, 6);
imagesc(-C{3}); colormap(gray); hold on;
scatter(1:T, O3, 120, 'c', 'filled', 'MarkerEdgeColor', 'k');
yticks(1:4); yticklabels({'start', 'hint', 'left', 'right'});
xticks(1:T); title('Outcomes - observed action', 'FontWeight', 'bold');