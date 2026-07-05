function G_pi = compute_policy_G(pol, t, T, qs, A, C, V, H_A, num_obs, num_states)
% COMPUTE_POLICY_G  Expected free energy of one policy over its future.
%
%   G_pi = COMPUTE_POLICY_G(pol, t, T, qs, A, C, V, H_A, num_obs, num_states)
%   returns the expected free energy of policy `pol`, summed over the future
%   steps t+1..T.
%
%   It replaces the artificial removal of the c_loss penalty by a
%   belief-correction. Under simple inference the future context belief
%   qs{1}(:,tau,pol) stays uniform, because future observations are not fed
%   back into it; a policy that requests the hint is therefore scored as if
%   it played the machine blindly. Here, for the steps after the hint, we
%   form the context belief the agent WOULD hold under each possible hint
%   outcome and evaluate the play choosing the action the agent would then
%   take -- a depth-one sophisticated backup. The preference matrix C is
%   left unchanged.

    % --- model-specific indices (this example) ---
    HINT_MODALITY = 1;    % observation modality x^1 reveals the context
    HINT_CHOICE   = 2;    % choice state "hint"
    CTX           = 1;    % context is state factor 1
    PLAY_ACTIONS  = [3 4];% choose-left, choose-right

    G_pi = 0;

    % Choice state along the policy is deterministic:
    % choice(1) = start, choice(tau) = action taken at tau-1.
    choice_state = [1, V{2}(1:T-1, pol).'];
    hint_step    = find(choice_state == HINT_CHOICE, 1);   % [] if never hints

    for tau = t+1:T
        if isempty(hint_step) || hint_step <= t || tau <= hint_step
            % No hint observed yet by this step: use the propagated
            % (uninformed) belief and the policy's own action.
            q_ctx_updated = qs{CTX}(:, tau, pol);
            sh = qs{2}(:, tau, pol);
            %g_step = step_efe(sc, sh, tau, A, C, H_A, num_obs);
            %fprintf('EVAL: t=%d, pol=%d, tau=%d | sc=[%.2f, %.2f] | sh=[%.2f, %.2f, %.2f, %.2f] | G=%.2f\n', ...
                    %t, pol, tau, sc(1), sc(2), sh(1), sh(2), sh(3), sh(4), g_step);
            G_pi = G_pi + step_efe(q_ctx_updated, sh, tau, A, C, H_A, num_obs);
        else
            %fprintf('LOOKAHEAD used: t=%d pol=%d hint_step=%d tau=%d\n', t, pol, hint_step, tau);
            % Hint already observed: average over hint outcomes. Each gives a
            % posterior context belief; under it the play action is chosen to
            % minimise the expected free energy (the agent's replanning).
            q_ctx = qs{CTX}(:, hint_step, pol);            % belief before hint
            for o = 1:num_obs(HINT_MODALITY)
                av  = A{HINT_MODALITY}(o, :, HINT_CHOICE); %extract probability for seeing hint_left, hint_right or no hint
                av  = av(:);                               % force it column vector
                P_o = av.' * q_ctx;     %dot product                   % probability of seeing the hint (4.15), it is a scalar 
                if P_o <= eps, continue; end  %skips no hint
                q_ctx_updated = (av .* q_ctx);  %element-wise product
                q_ctx_updated = q_ctx_updated / sum(q_ctx_updated);                         % posterior context (vector) with bayes
                best = inf;
                for a = PLAY_ACTIONS
                    sh = zeros(num_states(2), 1);  sh(a) = 1;
                    best = min(best, step_efe(q_ctx_updated, sh, tau, A, C, H_A, num_obs)); % calculate the minimum G given the possible actions after observation o
                end
                G_pi = G_pi + P_o * best; % for each o add its contribute to G
            end
        end
    end
end


function g = step_efe(sc, sh, tau, A, C, H_A, num_obs)
% STEP_EFE  Expected free energy of a single future step (risk + ambiguity),
% given a context belief sc and a choice belief sh, with C unchanged.
    g = 0;
    for m = 1:numel(num_obs)
        Qo = zeros(num_obs(m), 1);
        for o = 1:num_obs(m)
            Qo(o) = sc.' * squeeze(A{m}(o, :, :)) * sh;
        end
        risk      = Qo.' * (log(Qo + eps) - C{m}(:, tau));   % KL to preferences
        ambiguity = sc.'  * H_A{m} * sh;                     % expected entropy
        g = g + risk + ambiguity;
    end
end
