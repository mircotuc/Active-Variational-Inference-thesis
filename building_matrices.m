%% BUILDING THE MATRICES OF THE MODEL
function [A, B, C, D, V] = building_matrices(b, c_loss, c_reward)
% b is the noise parameter. thanks to this, sometimes I win choosing right
% even if the better slot-machine is the left one.
% c_loss tells me how much I'm risk-averse
% c_reward tells me how much I'm reward-seeking



% OUTCOME MODALITIES (OBSERVATIONS) o: 
    % 1) HINT: no hint, hint-left, hint-right
    % 2) WIN: start, lose, win
    % 3) OBSERVED ACTION: start, hint, choose-left, choose-right
% POSSIBLE STATE FACTORS s:
    % 1) CONTEXT: left-better, right-better
    % 2) CHOICE: start, hint, choose-left, choose-right
% POSSIBLE ACTIONS: (go to) start, hint, left, right

num_obs = [3, 3, 4]; % [HINT, WIN, OBSERVED ACTION]
num_states = [2, 4]; % [CONTEXT, CHOICE]
num_actions = 4;
time_periods = 3;
% LIKELYHOOD MATRIX A
A= cell(1,3); % cell creates an array 1x3 where you can fit things of different dimensions
% here the 3 stands for 3 different outcome modalities

% P(HINT|CONTEXT,CHOICE) is a tensor 3x2x4
A{1} = zeros(num_obs(1), num_states(1), num_states(2)); % initialize
for context = 1:num_states(1)
    for choice = 1:num_states(2)
        if choice == 2 % when I choose HINT
            if context == 1 % Left-better context
                A{1}(2, context, choice) = 1; % if left is better and i choose HINT than i observe hint-left
            else            % Right-better context
                A{1}(3, context, choice) = 1; % same
            end
        else % Anywhere else
            A{1}(1, context, choice) = 1; % if I don't choose hint, i observe no hint
        end
    end
end

% P(WIN|CONTEXT,CHOICE) is a tensor 3x2x4
A{2} = zeros(num_obs(2), num_states(1), num_states(2)); % initialize

for context = 1:num_states(1)
    % if I don't choose left or right (so start or hint) I will observe
    % start
    A{2}(1, context, 1) = 1; 
    A{2}(1, context, 2) = 1; 
end
% Choice 3 (Choose-left)
A{2}(2, 1, 3) = b; % Context 1 (Left-better), lose
A{2}(3, 1, 3) = 1-b; % Context 1 (Left-better), win
A{2}(2, 2, 3) = 1-b; % Context 2 (Right-better), lose
A{2}(3, 2, 3) = b; % Context 2 (Right-better), win
% Choice 4 (Choose-right)
A{2}(2, 1, 4) = 1-b; % Context 1 (Left-better), lose
A{2}(3, 1, 4) = b; % Context 1 (Left-better), win
A{2}(2, 2, 4) = b; % Context 2 (Right-better), lose
A{2}(3, 2, 4) = 1-b; % Context 2 (Right-better), win

% P(OBSERVED ACTION|CONTEXT,CHOICE) is a tensor 4x2x4
A{3} = zeros(num_obs(3), num_states(1), num_states(2)); % initialize
for choice = 1:num_states(2)
    A{3}(choice, :, choice) = [1 1]; 
end



% TRANSITION MATRIX B
B = cell(1,2); % one tensor per state factor

% P(CONTEXT t|CONTEXT t-1, ACTION) is a tensor 2x2x4
B{1} = zeros(num_states(1), num_states(1), num_actions); % initialize
for action = 1:num_actions
    B{1}(:, :, action) = eye(2); % context independent from my actions
end

% P(CHOICE t|CHOICE t-1, ACTION) is a tensor 4x4x4
B{2} = zeros(num_states(2), num_states(2), num_actions);
for action = 1:num_actions
    % Action 'u' moves the agent to state 'u' no matter what state in t-1
    B{2}(action, :, action) = 1; 
end

% PREFERENCES MATRIX C
C = cell(1,3); % one matrix per outcome modality
% the matrix has observations on rows and time on columns (pref change in
% different time periods)

% HINT preferences (indifferent)
C{1} = zeros(num_obs(1), time_periods);

% WIN preferences (not indifferent)
C{2} = zeros(num_obs(2), time_periods);
C{2}(:,:) =    [-1  -1   -1   ;  % Null
                0 -c_loss -c_loss  ;  % Loss (independent of time)
                0  c_reward  c_reward/2]; % win; % If I get the hint, reward gets smaller

% OBSERVED ACTIONS preferences (indifferent)
C{3} = zeros(num_obs(3), time_periods);

% PRIOR VECTORS D OVER STATES s
D = cell(1, 2); % one vector per state

% CONTEXT prior (column vector)
D{1} = [0.5; 0.5]; % no bias on which slot is better

% CHOICE prior (column vector)
D{2} = [1; 0; 0; 0]; % I always begin in the state start

% POLICIES MATRIX V
V = cell(1, 2); % one matrix per state factor, each matrix has different actions on columns per each time on rows

% CONTEXT matrix 
V{1} = [ 1 1 1 1;  1 1 1 1]; % I cannot act on the context

% CHOICE matrix
V{2} = [ 2 2 3 4;  3 4 1 1]; %  policy 1 takes the hint then chooses left
% policy 3 chooses left and then does nothin

