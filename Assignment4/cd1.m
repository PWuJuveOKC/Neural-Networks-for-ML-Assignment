function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    %error('not yet implemented');
    %visible_data0 = sample_bernoulli(visible_data);
    
    visible_data = sample_bernoulli(visible_data)
    p1 = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    hid = sample_bernoulli(p1);
    p2 = hidden_state_to_visible_probabilities(rbm_w, hid);
    gd1 = configuration_goodness_gradient(visible_data, hid);
    vis = sample_bernoulli(p2);
    
    p3 = visible_state_to_hidden_probabilities(rbm_w, vis);
    %hid2 = sample_bernoulli(p3); 
    
    %gd2 = configuration_goodness_gradient(vis, hid2); 
    gd2 = configuration_goodness_gradient(vis, p3);
    
    ret = gd1 - gd2;
end
